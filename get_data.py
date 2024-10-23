from api_functions import ce_api
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd 
import numpy as np
import sqlalchemy
import workalendar.asia

 
def get_ce_data(field_name, start_date, end_date):
    ce_series_map = {
            'ce_tr_gas_dem_ldc' : 2796,
            'ce_tr_gas_dem_pwr' : 2797,
            'ce_tr_gas_dem_ind' : 2798,
            'ce_tr_gas_sup_ru' : 2717,
            'ce_tr_gas_sup_ir' : 2719,
            'ce_tr_gas_sup_az' : 2718,
        }
    series_id = ce_series_map[field_name]
    df = ce_api(series_id, start_date, end_date)
    return df


def get_data(sql_text, server, database):
    engine = sqlalchemy.create_engine('mssql+pyodbc://' + str(server) + '/' + str(database) + '?driver=ODBC+Driver+13+for+SQL+Server')
    returned_data = pd.read_sql_query(sql_text, engine)
    return returned_data


def get_weather_actuals(country):
    sql = f"SELECT * FROM [Meteorology].[dbo].[WeatherStationTimeSeriesLatest] WHERE ModelSourceName = 'ecmwf-era5' AND ParameterName = 't_2m:C' and CountryName IN ('{country}')"
    temps_actuals_m = get_data(sql, 'PRD-DB-SQL-211', 'Meteorology').set_index('ValueDate')
    temps_actuals_m = pd.pivot_table(temps_actuals_m, columns=['CountryName'], values='Value', index=temps_actuals_m.index).resample('d').mean()
    temps_actuals_m = temps_actuals_m.rename({temps_actuals_m.columns[0] : 'Temps'}, axis=1)
    return temps_actuals_m


def get_weather_forecasts(asof, country, days):
    parameter_name = 't_2m:C'
    if asof is None:
        asof = datetime.now()
    forecast_date_str = asof.strftime('%Y-%m-%d')
    min_value_date = (asof + timedelta(days = days*-1)).date().strftime("%Y-%m-%d")
    max_value_date = (datetime.now() + relativedelta(years=3)).date() # '2025-01-01'
    sql14 = f'''
        SET NOCOUNT ON;
        exec [Meteorology].[dbo].[GetAsOfViewWeatherStations] @model_source = 'ecmwf-ens', @parameter_name = '{parameter_name}', @country_name = '{country}', @forecast_date = '{forecast_date_str}', @min_value_date = '{min_value_date}'
    '''
    fcst14 = get_data(sql14, 'PRD-DB-SQL-211', 'Meteorology').groupby(['ValueDate'])['value_adj'].sum()
    sql46 = f'''
        SET NOCOUNT ON;
        exec [Meteorology].[dbo].[GetAsOfViewWeatherStations] @model_source = 'ecmwf-vareps', @parameter_name = '{parameter_name}', @country_name = '{country}', @forecast_date = '{forecast_date_str}', @min_value_date = '{min_value_date}'
    '''
    fcst46 = get_data(sql46, 'PRD-DB-SQL-211', 'Meteorology').groupby(['ValueDate'])['value_adj'].sum()
    hist_norm_sql = f'''
        select t.[index], t.[{country}] as [Country] --.[China]
        from LNG.ana.TempNormal t
        where t.[index] >= '{min_value_date}'
            and t.[index] < '{max_value_date}'
    '''
    hist_norm_df = get_data(hist_norm_sql, 'PRD-DB-SQL-211', 'Meteorology').rename({'index' : 'ValueDate', 'Country' : 'value_adj'}, axis=1).set_index('ValueDate')
    hist_norm_df.index = hist_norm_df.index.astype('datetime64[ns]')
    hist_norm_df = hist_norm_df.resample('H').mean().ffill()
    wfcst2 = pd.DataFrame(np.nan, index = pd.date_range(start = fcst14.index[0], end = hist_norm_df.index[-1]), columns = [fcst14.name])
    wfcst2 = wfcst2.resample('H').mean()
    wfcst2.update(hist_norm_df)
    wfcst2.update(fcst46)
    wfcst2.update(fcst14)
    wfcst2 = wfcst2.fillna(method = 'bfill').rename({fcst14.name : 'Temps'}, axis=1).resample('d').mean()
    return wfcst2


def weather_data(country, asof):
    temps_actuals_m = get_weather_actuals(country)
    temps_SN_30year_m = get_weather_forecasts(asof, country, 30)    
    temps_m = pd.concat([temps_actuals_m, temps_SN_30year_m])
    temps_m = temps_m[~temps_m.index.duplicated(keep='first')]
    return temps_m


def get_solar_data(country, start_date, end_date):
    if country == 'turkey':
        series_id = 54773
        solar_hist = ce_api(series_id, start_date, end_date).rename({series_id : 'Solar'}, axis=1)[['Solar']].asfreq('h').ffill()
        solar_hist = solar_hist.copy()
    else:
        sql = f"SELECT * FROM [Meteorology].[dbo].[WeatherStationTimeSeriesLatest] WHERE CountryName = '{country}' AND ParameterName in ('global_rad:W') and ModelSourceName = 'ecmwf-era5' ORDER BY valuedate"
        solar_hist = get_data(sql, 'PRD-DB-SQL-211','Meteorology').set_index('ValueDate')
        solar_hist['WeightedValue'] = solar_hist['Value'] * solar_hist['Weighting']
        solar_hist = solar_hist[['WeightedValue']].reset_index()
        solar_hist = solar_hist.groupby(['ValueDate']).sum().reset_index().set_index('ValueDate')
        solar_hist = solar_hist.rename({'WeightedValue' : 'Solar'}, axis=1)[['Solar']]
        solar_sn = solar_hist.copy()
        solar_sn['month'] = solar_sn.index.month
        solar_sn['day'] = solar_sn.index.day
        solar_sn['hour'] = solar_sn.index.hour
        solar_sn.index = solar_sn.groupby(['month', 'day', 'hour']).mean().reset_index()
        solar_full = pd.DataFrame(np.nan, index = pd.date_range(start = solar_hist.index[0], end = '2025-12-31 23:00:00', freq='h'), columns = ['Solar'])
        solar_full.update(solar_hist)
        solar_full['month'] = solar_full.index.month
        solar_full['day'] = solar_full.index.day
        solar_full['hour'] = solar_full.index.hour
        solar_full = solar_full.reset_index().merge(how='left', on=['month', 'day', 'hour'], right=solar_sn).set_index('index')
        solar_full['Solar'] = solar_full['Solar_x'].fillna(solar_full['Solar_y'])
        solar_full = solar_full[['Solar']].resample('MS').mean()
    return solar_full
  

def get_nuke_data(country):
    if country == 'turkey':
        return pd.DataFrame()
    country_abbrev = 'JP' if country == 'japan' else 'KR' # temp
    SQL_Nuke = f'''select *
        from [LNG].[ana].[{country_abbrev}NukPlanned] j
        where j.[timestamp] = (select max([timestamp]) from [LNG].[ana].[{country_abbrev}NukPlanned])
    '''
    Nuclear_planned_df = get_data(SQL_Nuke,'PRD-DB-SQL-211','LNG').fillna(0)
    Nuclear_planned_df['index']= pd.to_datetime(Nuclear_planned_df['index'], infer_datetime_format=True)
    Nuclear_planned_df = Nuclear_planned_df.sort_index()
    Nuclear_planned_df = Nuclear_planned_df.set_index('index')
    Nuclear_planned_df = Nuclear_planned_df.drop(['Sum of OP and Outage','Planned 2023', 'Planned 2024', 'timestamp'], axis=1)
    Nuclear_planned_df = Nuclear_planned_df.sum(axis=1)
    SQL = f'''select al.ValueDate, sum(al.[Value]) as [Value]
        from AnalyticsModel.ts.AnalyticsLatest al
        where al.CurveId in (
            select c.CurveId
            from AnalyticsModel.aux.CurveDetail c
            where c.Country = '{country}' and c.Commodity = 'Power' and c.[Type] = 'Generation' and c.SubType = 'Availability' and c.Fuel = 'Nuclear'
        )
        group by al.ValueDate
        order by al.ValueDate
    '''
    Nuclear_history_df=get_data(SQL,'PRD-DB-SQL-209','AnalyticsModel').rename({'ValueDate':'index','Value':0}, axis=1)
    Nuclear_history_df['index']= pd.to_datetime(Nuclear_history_df['index'], infer_datetime_format=True)
    Nuclear_history_df = Nuclear_history_df.set_index('index')
    Nuclear_df = pd.concat([Nuclear_history_df,Nuclear_planned_df]).reset_index().drop_duplicates(subset='index', keep='first').set_index('index').resample('MS').mean()
    Nuclear_df = Nuclear_df.rename({0:'Nuclear'}, axis=1)
    nuke_avg = Nuclear_df[Nuclear_df.index >= datetime.today() - timedelta(days=3*365)]
    nuke_avg = nuke_avg.groupby(nuke_avg.index.month).mean()
    nuke_remaining_fcst = pd.DataFrame(pd.date_range(Nuclear_df.index.max() + relativedelta(months=1), '2025-12-01', freq='MS')).set_index(0)
    nuke_remaining_fcst = nuke_remaining_fcst.merge(how='left', left_on=nuke_remaining_fcst.index.month, right_index=True, right=nuke_avg)
    Nuclear_df = pd.concat([Nuclear_df, nuke_remaining_fcst])
    if country == 'japan':
        Nuclear_backfill_df = pd.read_excel(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Salman\Power_Regression_v1.xlsx', sheet_name = 'Regression_2.1', header = 2)[['Date', 'Nuclear']]
    elif country == 'south korea':
        Nuclear_backfill_df = pd.read_excel(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Korea_Nukes_Refinitiv_chart_100134191.xlsx', sheet_name = 'Data').iloc[:-1,[0,-1]]
        Nuclear_backfill_df = Nuclear_backfill_df.rename({Nuclear_backfill_df.columns[1] : 'Nuclear'}, axis=1)
    Nuclear_backfill_df = Nuclear_backfill_df.set_index('Date').resample('MS').mean()
    Nuclear_backfill_df = Nuclear_backfill_df[Nuclear_backfill_df.index <= '2019-10-01']
    Nuclear_df = pd.concat([Nuclear_backfill_df,Nuclear_df])
    Nuclear_df = Nuclear_df.sort_index()
    return Nuclear_df


def get_platts_jkm():
    Platts_SQL='''Select z2.price_month, avg(z2.[Close]) as [avg_price]
    from (
    SELECT
           case when z.[day] < 16 then dateadd(month, datediff(month, 0, dateadd(month, 1, [Timestamp])), 0) else dateadd(month, datediff(month, 0, dateadd(month, 2, [Timestamp])), 0) end as [price_month], z.*
    from (
             select
             [Timestamp]
             ,datepart(year, [Timestamp]) as [year]
             ,datepart(month, [Timestamp]) as [month]
             ,datepart(day, [Timestamp]) as [day]
          ,[Close]
       FROM [LNG].[dbo].[ReutersActualPrices] r
      where r.RIC = 'AAOVQ00'
    ) z
    ) z2
    group by z2.price_month
    order by z2.price_month'''
    JKM_Platts_df=get_data(Platts_SQL,'PRD-DB-SQL-211','LNG')
    JKM_Platts_df['index']= pd.to_datetime(JKM_Platts_df['price_month'], infer_datetime_format=True)
    JKM_Platts_df = JKM_Platts_df.set_index('index').drop('price_month', axis=1)
    return JKM_Platts_df


def get_brent_curve(prices, prices_all_pivot):
    adder = 0.12
    brent_curve_rolling, brent_curve_lag, brent_curve_freq = 3, 0, 1
    brent_curve = prices_all_pivot['ICE Brent Fwd'].shift(brent_curve_lag).rolling(window=brent_curve_rolling).mean().reset_index()
    for i in range(1,brent_curve_freq):
        brent_curve.iloc[np.arange(brent_curve_rolling+i-1+brent_curve_lag,len(brent_curve),brent_curve_freq), 1] = np.nan
    brent_curve = brent_curve.set_index('CURVE_START_DT').ffill()
    prices['ICE Brent Fwd (' + str(brent_curve_rolling) + '-' + str(brent_curve_lag) + '-' + str(brent_curve_freq) + ')'] = brent_curve * adder    
    return prices


def get_gdp(country):
    if country == 'japan':
        GDP_df = pd.read_excel(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Salman\Power_Regression_v1.xlsx', sheet_name = 'Regression_2.1', index_col='Date', header = 2)[['GDP']]    
    elif country == 'south korea':
        GDP_df = pd.read_excel(r'U:\\Trading - Gas\\LNG\\quant\\Scripts\\Korea_forecast\japan_mapping_data.xlsx', sheet_name = 'gdp_kr', index_col=0, usecols="A:B" ).resample('MS').mean().ffill()
    return GDP_df


def calc_hdds_cdds(temps_m, regression_hdd_base, regression_cdd_base):
    HCDDs = temps_m.copy()
    HCDDs['HDDs'] = (regression_hdd_base - temps_m).clip(lower=0)
    HCDDs['CDDs'] = (temps_m - regression_cdd_base).clip(lower=0)
    HCDDs['HDD2s'] = HCDDs['HDDs']**2
    HCDDs['CDD2s'] = HCDDs['CDDs']**2
    HCDDs['dummy_sum'] = np.where(HCDDs['Temps'] >= regression_hdd_base, 1, 0)
    HCDDs['dummy_win'] = np.where(HCDDs['Temps'] < regression_cdd_base, 1, 0)    
    HCDDs = HCDDs.drop('Temps', axis=1).resample('MS').sum()
    return HCDDs


def add_date_related_features(country, start_date, end_date):
    if country == 'japan':
        cal = workalendar.asia.Japan()
    elif country == 'south korea':
        cal = workalendar.asia.SouthKorea()
    x_date_features = pd.DataFrame(data=pd.date_range(start=start_date, end=end_date - timedelta(days=1), freq='d'), columns=['Date']).set_index('Date')    
    x_date_features['ny'] = np.where(((x_date_features.index.month == 12) & (x_date_features.index.day >= 28)) | ((x_date_features.index.month == 1) & (x_date_features.index.day <= 4)), 1, 0)
    x_date_features['hol'] = np.where(x_date_features.index.isin([i[0] for x in [cal.holidays(x) for x in range(start_date.year, end_date.year)] for i in x]), 1, 0)
    x_date_features['we'] = np.where(x_date_features.index.weekday > 4, 1, 0)
    x_date_features['wd'] = np.where(x_date_features[['ny', 'hol', 'we']].sum(axis=1) == 0, 1, 0)    
    x_date_features = x_date_features.resample('MS').sum()
    return x_date_features


def merge_inputs(x_date_features, prices_all_pivot, JKM_Platts_df, GDP_df, Nuclear_df, solar_full, HCDDs, ihs_actuals):
    inputs_df = x_date_features.merge(how='outer', left_index=True, right_index=True, right=prices_all_pivot)
    inputs_df = inputs_df.merge(how='outer', left_index=True, right_index=True, right=JKM_Platts_df)
    inputs_df['JKM'] = np.where(inputs_df.index > '2023-11-01', inputs_df['JKM Fwd'], inputs_df['avg_price'])
    inputs_df = inputs_df.drop(['JKM Fwd','avg_price'], axis=1)
    inputs_df = get_brent_curve(inputs_df)
    inputs_df = inputs_df.merge(how='outer', left_index=True, right_index=True, right=GDP_df)
    inputs_df = inputs_df.merge(how='outer', left_index=True, right_index=True, right=Nuclear_df)
    inputs_df = inputs_df.merge(how='outer', left_index=True, right_index=True, right=solar_full)
    inputs_df = inputs_df.merge(how='outer', left_index=True, right_index=True, right=HCDDs)    
    inputs_df = inputs_df.merge(how='outer', left_index=True, right_index=True, right=ihs_actuals)
    inputs_df = inputs_df.drop('ICE Brent Fwd', axis=1)
    return inputs_df


def get_y_data(y_parameter): # temp
    if y_parameter == 'db_kr_pwr_dem_kpx':
        sql = 'SELECT [Date], [Id], [Total] FROM [LNG].[dbo].[KoreaEpsisTradingVolume] order by [Date], [Id]'
        y_parameter_data = get_data(sql, 'PRD-DB-SQL-211', 'LNG').drop_duplicates(subset='Date', keep='last')[['Date', 'Total']].rename({'Total' : 'db_kr_pwr_dem_kpx'}, axis=1).set_index('Date')
        y_parameter_data.index = y_parameter_data.index.astype('datetime64[ns]')
        
    return y_parameter_data


def get_reg_unique_dates(start_date, end_date):
    dates_delta = relativedelta(end_date, start_date)
    num_months = dates_delta.months + (dates_delta.years * 12) + 1
    dates_list = []
    for i in range(num_months):
        d1 = start_date + relativedelta(months=i)
        for k in range(i, num_months):
            d2 = start_date + relativedelta(months=k)            
            dates_list.append([d1, d2])
    return dates_list
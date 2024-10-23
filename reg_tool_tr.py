
# Packages
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from os import listdir
import json
import requests
import numpy as np
import pandas as pd
import sqlalchemy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import workalendar.asia

def ihs_api(ihs_parameter_name, index):
    ihs_url = 'https://api.connect.ihsmarkit.com/cs/v1/lngmarkets/retrieve/'
    username = 'ee80410e-6698-4173-a2bb-7495b89ad613'
    password = 'sy85nbpj2Tfbwqgw'
    response = requests.get(ihs_url + ihs_parameter_name, auth=(username, password)).json()
    df = pd.DataFrame(response)
    df[index] = pd.to_datetime(df[index], infer_datetime_format=True)
    df = df.set_index(index)
    df = df.drop(['market'], axis=1)
    return df

#%%

# def get_ihs_actuals(ihs_parameter_name):
#     ihs_actuals                     = ihs_api(ihs_parameter_name, 'month') #.round(1)
#     #ihs_actuals                     = (ihs_actuals.T / ihs_actuals.index.daysinmonth ).T
#     return ihs_actuals

#%%
# z = get_ihs_actuals(ihs_parameter_name)
# z = (z.T * z.index.daysinmonth ).T
# z1 = z.sum(axis=1)

#
# z = get_ihs_actuals('kr_power_generation_by_fuel_m')
# z = (z.T * z.index.daysinmonth ).T
# z1 = z.sum(axis=1)

# qwert = get_ihs_actuals('kr_power_generation_by_fuel_m')
# qwert = get_ihs_actuals('jp_power_generation_by_fuel_m')

# qwert_n = (qwert[['nuclear_gwh']].T * qwert.index.days_in_month).T
# qwert_m = qwert_n.merge(how='inner', left_index=True, right_index=True, right=Nuclear_df)
# qwert_m['diff'] = qwert_m['Nuclear'] - qwert_m['nuclear_gwh']
# qwert_m['%'] = qwert_m['nuclear_gwh'] / qwert_m['Nuclear']

#%%

def ce_api(series_id, start_date, end_date):
    ce_url = 'https://commodityessentials.com/api/'
    url = ce_url + f'eugasseries?id={series_id}&nd=5475&unit=mcm&dateFrom={start_date}&dateTo={end_date}'
    username = 'swasti@freepoint.com'
    password = 'kfGa5uwL1+'
    headers = {'Accept':'application/json'}
    response = requests.get(url, auth=(username, password), headers=headers).json()
    unit = response['headers'][0]['Unit']
    data_list = [(x['Date'], x[str(series_id)], unit) for x in response['values']]
    df = pd.DataFrame(data_list, columns=['date', series_id, 'unit']).set_index('date')
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    return df
    
def get_ce_data(field_name, start_date, end_date):
    ce_series_map = { # temp
            'ce_tr_gas_dem_ldc' : 2796,
            'ce_tr_gas_dem_pwr' : 2797,
            'ce_tr_gas_dem_ind' : 2798,
            'ce_tr_gas_sup_ru' : 2717,
            'ce_tr_gas_sup_ir' : 2719,
            'ce_tr_gas_sup_az' : 2718,
            # '' : ,
            # '' : ,
            # '' : ,
            # '' : ,
            # '' : ,
            # '' : ,
            # '' : ,
        }
    
    series_id = ce_series_map[field_name]
    df = ce_api(series_id, start_date, end_date)
    #df = df.fillna(method='ffill')
    return df

#%%

def get_data(sql_text, server, database):
    # runs the SQL and returns the data as a pandas dataframe
    engine = sqlalchemy.create_engine('mssql+pyodbc://' + str(server) + '/' + str(database) + '?driver=ODBC+Driver+13+for+SQL+Server')
    returned_data = pd.read_sql_query(sql_text, engine)
    return returned_data

#%%

def get_weather_actuals(country):
    sql = f"SELECT * FROM [Meteorology].[dbo].[WeatherStationTimeSeriesLatest] WHERE ModelSourceName = 'ecmwf-era5' AND ParameterName = 't_2m:C' and CountryName IN ('{country}')"
    temps_actuals_m = get_data(sql, 'PRD-DB-SQL-211', 'Meteorology').set_index('ValueDate')
    temps_actuals_m = pd.pivot_table(temps_actuals_m, columns=['CountryName'], values='Value', index=temps_actuals_m.index).resample('d').mean()
    temps_actuals_m = temps_actuals_m.rename({temps_actuals_m.columns[0] : 'Temps'}, axis=1)
    
    return temps_actuals_m

#%%
#
#days=30

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
    #fcst14 = pd.read_sql(sql14, sql_conn_lng)
    fcst14 = get_data(sql14, 'PRD-DB-SQL-211', 'Meteorology').groupby(['ValueDate'])['value_adj'].sum()
    sql46 = f'''
        SET NOCOUNT ON;
        exec [Meteorology].[dbo].[GetAsOfViewWeatherStations] @model_source = 'ecmwf-vareps', @parameter_name = '{parameter_name}', @country_name = '{country}', @forecast_date = '{forecast_date_str}', @min_value_date = '{min_value_date}'
    '''
    fcst46 = get_data(sql46, 'PRD-DB-SQL-211', 'Meteorology').groupby(['ValueDate'])['value_adj'].sum()
    # sqlseas = f'''
    #     SET NOCOUNT ON;
    #     exec [Meteorology].[dbo].[GetAsOfViewWeatherStations] @model_source = 'ecmwf-mmsf', @parameter_name = '{parameter_name}', @country = '{country}', @forecast_date = '{forecast_date_str}', @min_value_date = '{min_value_date}'
    # '''
    # fcstseas = pd.read_sql(sqlseas, sql_conn_lng).groupby(['ValueDate'])['value_adj'].sum()
    hist_norm_sql = f'''
        select t.[index], t.[{country}] as [Country] --.[China]
        from LNG.ana.TempNormal t
        where t.[index] >= '{min_value_date}'
            and t.[index] < '{max_value_date}'
    '''
    hist_norm_df = get_data(hist_norm_sql, 'PRD-DB-SQL-211', 'Meteorology').rename({'index' : 'ValueDate', 'Country' : 'value_adj'}, axis=1).set_index('ValueDate')
    hist_norm_df.index = hist_norm_df.index.astype('datetime64[ns]')
    hist_norm_df = hist_norm_df.resample('H').mean().ffill()
    #wfcst2 = pd.DataFrame(np.nan, index = pd.date_range(start = fcst14.index[0], end = fcstseas.index[-1]), columns = [fcst14.name])
    wfcst2 = pd.DataFrame(np.nan, index = pd.date_range(start = fcst14.index[0], end = hist_norm_df.index[-1]), columns = [fcst14.name])
    wfcst2 = wfcst2.resample('H').mean()
    #wfcst2.update(fcstseas)
    wfcst2.update(hist_norm_df)
    wfcst2.update(fcst46)
    wfcst2.update(fcst14)
    wfcst2 = wfcst2.fillna(method = 'bfill').rename({fcst14.name : 'Temps'}, axis=1).resample('d').mean()
    return wfcst2

#%%

def weather_data(country, asof):
    temps_actuals_m = get_weather_actuals(country)
    temps_SN_30year_m = get_weather_forecasts(asof, country, 30)    

    temps_m = pd.concat([temps_actuals_m, temps_SN_30year_m])
    temps_m = temps_m[~temps_m.index.duplicated(keep='first')]
    
    return temps_m

#%%

def get_solar_data(country):
    if country == 'turkey':
        series_id = 54773
        solar_hist = ce_api(series_id, start_date, end_date).rename({series_id : 'Solar'}, axis=1)[['Solar']].asfreq('h').ffill()
        solar_hist = solar_hist2.copy()
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

#%%
#country='japan'
#country='south korea'
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

#%%

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



def get_prices_cxl(asof, start_date, end_date):
    curves = [
        'ICE Brent Fwd',
        'NEWC Fwd',
        # 'Henry Hub',
        'JKM FWD',
        # 'TTF FWD',
        # 'TTF USD Fwd',
        # 'NBP FWD',
        # 'PEG FWD',
        # 'PSV FWD',
        # 'PVB FWD',
        # 'ZEE HUB FWD'
    ]
    
    time_period_id = 2
    
    prices_all = pd.DataFrame()
    curve = 'JKM FWD'
    for curve in curves:
        url = f'http://prd-rsk-app-06:28080/price-manager/service/curveJson?name={curve}&cob={asof}&time_period_id={time_period_id}&start={start_date}&end={end_date}'
        response = requests.request('GET', url)
        data_json = json.loads(response.text)
        prices = pd.DataFrame.from_dict(data_json)
        prices_all = prices_all.append(prices, sort=False)
    
    prices_all_pivot = prices_all.pivot_table(values=['PRICE'], index=['CURVE_START_DT'], columns=['CURVE_NAME'])
    prices_all_pivot.columns = prices_all_pivot.columns.get_level_values(1)
    prices_all_pivot.index = prices_all_pivot.index.astype('datetime64[ns]')
    return prices_all_pivot

#%%

def get_brent_curve(prices):
    adder = 0.12
    brent_curve_rolling, brent_curve_lag, brent_curve_freq = 3, 0, 1
    brent_curve = prices_all_pivot['ICE Brent Fwd'].shift(brent_curve_lag).rolling(window=brent_curve_rolling).mean().reset_index()
    for i in range(1,brent_curve_freq):
        brent_curve.iloc[np.arange(brent_curve_rolling+i-1+brent_curve_lag,len(brent_curve),brent_curve_freq), 1] = np.nan
    brent_curve = brent_curve.set_index('CURVE_START_DT').ffill()
    prices['ICE Brent Fwd (' + str(brent_curve_rolling) + '-' + str(brent_curve_lag) + '-' + str(brent_curve_freq) + ')'] = brent_curve * adder    
    return prices

#%%

def get_gdp(country):
    if country == 'japan':
        GDP_df = pd.read_excel(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Salman\Power_Regression_v1.xlsx', sheet_name = 'Regression_2.1', index_col='Date', header = 2)[['GDP']]    
    elif country == 'south korea':
        GDP_df = pd.read_excel(r'U:\\Trading - Gas\\LNG\\quant\\Scripts\\Korea_forecast\japan_mapping_data.xlsx', sheet_name = 'gdp_kr', index_col=0, usecols="A:B" ).resample('MS').mean().ffill()
        #GDP_df.index = GDP_df.apply(lambda x: datetime.datetime(x.name.year, x.name.month, 1), axis = 1)
    
    # GDP_History_df = get_data('select * from [LNG].[ana].[JKCGDP]','PRD-DB-SQL-211','LNG')
    # GDP_History_df['index']= pd.to_datetime(GDP_History_df['index'], infer_datetime_format=True)
    # GDP_History_df = GDP_History_df.set_index('index')
    # GDP_History_df = GDP_History_df[['JGDOQOQ Index']].resample('MS').mean().rename({'JGDOQOQ Index' : 'GDP'}, axis=1) #.drop(['CNGDGDP Index','JGDOSGDP Index', 'KOGCGDP Index'], axis=1)
    # d = pd.concat([GDP_df, GDP_History_df]).sort_index()
    
    return GDP_df

#%%

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

#%%

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

#%%

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

#%%

def get_y_data(y_parameter): # temp
    if y_parameter == 'db_kr_pwr_dem_kpx':
        sql = 'SELECT [Date], [Id], [Total] FROM [LNG].[dbo].[KoreaEpsisTradingVolume] order by [Date], [Id]'
        y_parameter_data = get_data(sql, 'PRD-DB-SQL-211', 'LNG').drop_duplicates(subset='Date', keep='last')[['Date', 'Total']].rename({'Total' : 'db_kr_pwr_dem_kpx'}, axis=1).set_index('Date')
        y_parameter_data.index = y_parameter_data.index.astype('datetime64[ns]')
        
    return y_parameter_data

#%%

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

#%%
#df = inputs_df.copy()
#y = 'residential_mmcm'

def plot_xs_vs_ys(country, inputs_df, xs_ys):    
    for i in xs_ys:
        y, x = i
        chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\xs_vs_ys\{country}_{y}_vs_{x}.html'
        
        df = inputs_df[[x] + [y]].dropna(how='any')
        
        fig = go.Figure(data=[
             go.Scatter(x=df[x], y=df[y], mode='markers+text', text=df.index.month, textposition='top center')
         ])
        
        fig.write_html(chart_file, include_plotlyjs='cdn')  

#%%
# regression_start_date = '2018-01-01'
# regression_end_date = '2023-02-01'
# regression_degrees = 1

#poly_reg_model.intercept_
#poly_reg_model.coef_

def run_regressions(inputs_df, regression_start_date, regression_end_date, x_parameters, y_parameter, regression_degrees, all_iterations):
    regression_df = inputs_df[(inputs_df.index >= regression_start_date) & (inputs_df.index <= regression_end_date)].copy()
    
    y = regression_df[y_parameter].values.reshape(-1,1)
    X = regression_df[x_parameters]
    
    poly_reg = PolynomialFeatures(degree=regression_degrees, include_bias=False)
    poly_features = poly_reg.fit_transform(X)
    poly_reg_model = LinearRegression()
    
    poly_reg_model.fit(poly_features, y)
    #X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=1, random_state=42)
    #poly_reg_model.fit(X_train, y_train)    
    
    y_predicted = poly_reg_model.predict(poly_features)
    in_dps = len(regression_df)
    in_rmse = np.nan
    in_r2 = np.nan
    if in_dps >= 2:
        in_rmse = np.sqrt(mean_squared_error(y_predicted, y))
        in_r2 = r2_score(y_predicted, y)
        if not all_iterations:
            print('in r sq: ', in_r2)
            print(len(y_predicted),len(y))
    
    forecast_df = inputs_df[x_parameters + [y_parameter]].dropna(subset=x_parameters, how='any')
    X = forecast_df[x_parameters]
    poly_features = poly_reg.fit_transform(X)
    y_predicted = poly_reg_model.predict(poly_features)
    
    forecast_df[y_parameter + '_predicted'] = y_predicted
    forecast_df[y_parameter + '_delta'] = forecast_df[y_parameter + '_predicted'] - forecast_df[y_parameter]
    
    out_sample = forecast_df[(forecast_df.index < regression_start_date) | (forecast_df.index > regression_end_date)].dropna(how='any')
    out_dps = len(out_sample)
    out_rmse = np.nan
    out_r2 = np.nan
    
    if out_dps >= 2:
        out_y_predicted = out_sample[y_parameter + '_predicted'].values.reshape(-1,1)
        out_y = out_sample[y_parameter].values.reshape(-1,1)        
        out_rmse = np.sqrt(mean_squared_error(out_y_predicted, out_y))
        out_r2 = r2_score(out_y_predicted, out_y)
        if not all_iterations:
            print('out r sq: ', out_r2)
            print(len(out_y_predicted),len(out_y))

    
    forecast_df = forecast_df.drop(x_parameters, axis=1)
    
    return forecast_df, in_dps, out_dps, in_rmse, out_rmse, in_r2, out_r2

#%%

def chart(country, title, df, latest_actual_date):
    chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\scaled_inputs_{country}_{title}.html'
    
    fig = go.Figure(data=[
             go.Scatter(name=x, x=df.index, y=df[x]) for x in df.columns             
         ])

    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")
        
    fig.update_layout(
        xaxis=dict(
                rangeselector=dict(
                        buttons=[
                                dict(count=7,
                                     label='1w',
                                     step='day',
                                     stepmode='backward'),
                                dict(count=1,
                                     label='1m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=3,
                                     label='3m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=6,
                                     label='6m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=1,
                                     label='YTD',
                                     step='year',
                                     stepmode='todate'),
                                dict(count=1,
                                     label='1y',
                                     step='year',
                                     stepmode='backward'),
                                dict(step='all', label='All')
                            ]
                    ),
                rangeslider=dict(visible=True),
                type='date',
                dtick='M1'
                #range=[today - relativedelta(days=365), today + relativedelta(days=1)],
                # rangebreaks=[
                #          dict(bounds=["sat", "mon"]), #hide weekends        
                #     ],
            ),
        hovermode='x unified',
        # height=750,
        title={
                'text' : title,
                'x' : 0.5, #'center',
                'yanchor' : 'top',
                'font_size' : 16
            })

    fig.write_html(chart_file, include_plotlyjs='cdn')   

#%%

def stacked_bar_chart(country, title, df, df2, latest_actual_date):
    chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\scaled_inputs_stacked_bar_{country}_{title}.html'
    
    fig = go.Figure(data=[
             go.Bar(name=x, x=df.index, y=df[x], text=x) for x in df.columns             
         ])

    fig.add_scatter(name=df2.columns[0], x=df2.index, y=df2.iloc[:,0], line=dict(color='black', width=3))
    fig.add_scatter(name=df2.columns[1], x=df2.index, y=df2.iloc[:,1], line=dict(color='red', width=3))

    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")
        
    fig.update_layout(
        barmode='stack',
        # xaxis=dict(
        #         rangeselector=dict(
        #                 buttons=[
        #                         dict(count=7,
        #                              label='1w',
        #                              step='day',
        #                              stepmode='backward'),
        #                         dict(count=1,
        #                              label='1m',
        #                              step='month',
        #                              stepmode='backward'),
        #                         dict(count=3,
        #                              label='3m',
        #                              step='month',
        #                              stepmode='backward'),
        #                         dict(count=6,
        #                              label='6m',
        #                              step='month',
        #                              stepmode='backward'),
        #                         dict(count=1,
        #                              label='YTD',
        #                              step='year',
        #                              stepmode='todate'),
        #                         dict(count=1,
        #                              label='1y',
        #                              step='year',
        #                              stepmode='backward'),
        #                         dict(step='all', label='All')
        #                     ]
        #             ),
        #         rangeslider=dict(visible=True),
        #         type='date',
        #         dtick='Y1'
        #         #range=[today - relativedelta(days=365), today + relativedelta(days=1)],
        #         # rangebreaks=[
        #         #          dict(bounds=["sat", "mon"]), #hide weekends        
        #         #     ],
        #     ),
        hovermode='x unified',
        # height=750,
        title={
                'text' : title,
                'x' : 0.5, #'center',
                'yanchor' : 'top',
                'font_size' : 16
            })

    fig.write_html(chart_file, include_plotlyjs='cdn')

#%%

def create_inputs_chart(country, forecast_df, x_parameters, y_parameter, latest_actual_date):
    standardized_inputs = pd.DataFrame(MinMaxScaler().fit_transform(forecast_df), columns=forecast_df.columns, index=forecast_df.index)
    chart(country, rf'standardized_inputs_{y_parameter}', standardized_inputs, latest_actual_date)
    
    #normalized_inputs = pd.DataFrame(StandardScaler().fit_transform(forecast_df), columns=inputs_df[x_parameters].columns, index=inputs_df.index) # rem x params
    
    standardized_std_inputs = standardized_inputs[x_parameters].sum(axis=1)
    standardized_std_inputs = standardized_inputs[x_parameters].apply(lambda x: x / standardized_std_inputs) #pd.DataFrame(MinMaxScaler().fit_transform(standardized_inputs.T), columns=standardized_inputs.T.columns, index=standardized_inputs.T.index).T # rem x params
    #standardized_std_inputs[y_parameter] = standardized_inputs[y_parameter]
    #standardized_std_inputs[y_parameter + '_predicted'] = standardized_inputs[y_parameter + '_predicted']
    
    for i in range(1, 13):
        i=1
        s = standardized_std_inputs[standardized_std_inputs.index.month == i]
        f = standardized_inputs[standardized_inputs.index.month == i][[y_parameter, y_parameter + '_predicted']]
        stacked_bar_chart(country, rf'standardized_inputs_{y_parameter}_{i}', s, f, latest_actual_date)
    
    #chart(country, rf'normalized_inputs_{y_parameter}', normalized_inputs, latest_actual_date)

#%%

def bar_line_chart(country, df, df1, df2, title, primary_y_axis_title, latest_actual_date):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    data = list((df2).round(1))

    fig.add_bar(x=df2.index, y=df2, name=df2.name+ ' (left y axis)', text=list(data), secondary_y=False, textposition='inside', textfont=dict(color='black'))

    fig.add_trace(go.Scatter(x=df.index, y=df, name=df.name + ' (right y axis)', line=dict(color='black', width=3)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df1.index, y=df1, name=df1.name + ' (right y axis)', line=dict(color='red', width=3)), secondary_y=True)

    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")

    fig.update_layout(title=title + ' last updated: ' + str(datetime.now()), xaxis_title='Date', yaxis_title=primary_y_axis_title, hovermode='x unified')
    fig.update_yaxes(title_text='mcm/d', secondary_y=True)
    fig.layout.yaxis.showgrid = False
    
    fig.write_html(rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\{country}_{title}.html', include_plotlyjs='cdn')   
    
    #fig.show()

#%%

asof = datetime.now()
start_date = datetime(2012,1,1)
end_date = datetime(2026,1,1)

#%%

prices_all_pivot = get_prices_cxl(asof, start_date, end_date)
JKM_Platts_df = get_platts_jkm()

#%%

models_results = []

#%%
#reg_conf_df_o = reg_conf_df.copy()
#ihs_actuals_orig = ihs_actuals.copy()

#z = ihs_actuals_orig[['residential_mmcm', 'commercial_mmcm', 'industrial_mmcm', 'other_mmcm', 'power_mmcm']].sum(axis=1)

#%%

models_config = {}
with open(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\fund_reg_conf.json', 'r') as f:
    models_config = json.loads(f.read())
    
#%%

reg_conf = []
final_df = pd.DataFrame()

#%%
for country in models_config:
    #%%
    temps_m = weather_data(country, asof)
    solar_full = get_solar_data(country)
    Nuclear_df = get_nuke_data(country)
    GDP_df = get_gdp(country)
    x_date_features = add_date_related_features(country, start_date, end_date)
    #%%
    for commodity in models_config[country]:
        #%%
        for rtype in models_config[country][commodity]:
        #%%
            ihs_parameter_name = models_config[country][commodity][rtype]['ihs_parameter_name']
            ihs_actuals = pd.DataFrame()
            if ihs_parameter_name:            # temp to get y
                ihs_actuals = ihs_api(ihs_parameter_name, 'month') #get_ihs_actuals(ihs_parameter_name)      
                ihs_actuals = (ihs_actuals.T / ihs_actuals.index.daysinmonth ).T
            #%%    
            for model in models_config[country][commodity][rtype]['models']:
                ##%%
                #model='power'
                #model='overall'
                ihs_field_names = models_config[country][commodity][rtype]['models'][model]['ihs_field_names']
                y_parameter = '+'.join(ihs_field_names)        
                if y_parameter[:3] == 'db_':
                    ihs_actuals[y_parameter] = get_y_data(y_parameter)
                elif y_parameter[:3] == 'ce_':
                    ihs_actuals[y_parameter] = get_ce_data(field_name, start_date, end_date)
                else:            
                    ihs_actuals[y_parameter] = ihs_actuals[ihs_field_names].sum(axis=1)
                
                for month_split in models_config[country][commodity][rtype]['models'][model]['month_splits']:
                    regression_conf = models_config[country][commodity][rtype]['models'][model]['month_splits'][month_split]
                    regression_hdd_base = regression_conf['parameters']['hdd_base']
                    regression_cdd_base = regression_conf['parameters']['cdd_base']
                    
                    HCDDs = calc_hdds_cdds(temps_m, regression_hdd_base, regression_cdd_base)
                    inputs_df = merge_inputs(x_date_features, prices_all_pivot, JKM_Platts_df, GDP_df, Nuclear_df, solar_full, HCDDs, ihs_actuals)
                    
                    regression_start_date = regression_conf['reg_start_date']
                    regression_end_date = regression_conf['reg_end_date']
                    #regression_df = inputs_df[(inputs_df.index >= regression_start_date)].copy()
                    
                    #y_parameter = y_parameter_orig #+ '_' + month_split
                    #regression_df = regression_df.rename({y_parameter_orig : y_parameter}, axis=1)
                    
                    regression_months = np.nan            
                    if 'months' in regression_conf:
                        regression_months = regression_conf['months']
                        inputs_df = inputs_df[inputs_df.index.month.isin(regression_months)]
                    
                    regression_degrees = regression_conf['parameters']['degrees']
                    
                    regression_others = np.nan
                    x_parameters = [] # ['HDDs', 'CDDs']                                    
                    if 'others' in regression_conf['parameters']:
                        regression_others = regression_conf['parameters']['others']
                        x_parameters = x_parameters + regression_others
                    
                    #inputs_df.loc[inputs_df.index >= regression_start_date, x_parameters].to_csv(rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\demand_model_inputs\jp-{model}-{month_split}-2023-10-10.csv')
                    #inputs_df.loc[inputs_df.index >= regression_start_date, x_parameters].to_csv(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\korea\demand_model_inputs\sk-{model}-{month_split}-2023-10-10.csv')
                    
                    print('Running: ', ihs_parameter_name, ihs_field_names, regression_months, regression_start_date, regression_end_date, regression_hdd_base, regression_cdd_base, regression_degrees)
                    forecast_df, in_dps, out_dps, in_rmse, out_rmse, in_r2, out_r2 = run_regressions(inputs_df, regression_start_date, regression_end_date, x_parameters, y_parameter, regression_degrees, False)
                    r_shrinkage = in_r2 - out_r2
                    if commodity == 'gas': # temp, needs more logic (differentiate between other countries etc)
                        final_df = pd.concat([final_df, forecast_df], axis=1)
                    reg_conf.append([country, commodity, rtype, ihs_parameter_name, ihs_field_names, regression_months, regression_start_date, regression_end_date, regression_hdd_base, regression_cdd_base, regression_others, regression_degrees, in_dps, out_dps, in_rmse, out_rmse, in_r2, out_r2, r_shrinkage])
                    
                    latest_actual_date = inputs_df[y_parameter].dropna().index.max()
                    #create_inputs_chart(inputs_df, forecast_df, y_parameter, latest_actual_date)
                    #bar_line_chart(forecast_df[y_parameter], forecast_df[y_parameter + '_predicted'], forecast_df[y_parameter + '_delta'], y_parameter + ' act vs fcst for ' + month_split, y_parameter + '_delta', latest_actual_date)
                    bar_line_chart2(country, forecast_df[y_parameter], forecast_df[y_parameter + '_predicted'], forecast_df[y_parameter + '_delta'], y_parameter + ' act vs fcst for ' + month_split, y_parameter + '_delta', regression_start_date, regression_end_date, -1, y_parameter)
                    #bar_line_chart2(final_df2[reg_id][y_parameter], final_df2[reg_id][y_parameter + '_predicted'], final_df2[reg_id][y_parameter + '_delta'], y_parameter + ' act vs fcst for ' + month_split, y_parameter + '_delta', earliest_actual_date, latest_actual_date, -1, y_parameter)
                   
#%%      

reg_conf_df = pd.DataFrame(reg_conf, columns=['country', 'commodity', 'rtype', 'ihs_parameter_name', 'ihs_field_names', 'regression_months', 'regression_start_date', 'regression_end_date', 'regression_hdd_base', 'regression_cdd_base', 'regression_others', 'regression_degrees', 'in_dps', 'out_dps', 'in_rmse', 'out_rmse', 'in_r2', 'out_r2', 'r_shrinkage'])
reg_conf_df['ihs_field_names'] = reg_conf_df['ihs_field_names'].astype(str)
reg_conf_df['regression_months'] = reg_conf_df['regression_months'].astype(str)
reg_conf_df['regression_others'] = reg_conf_df['regression_others'].astype(str)
reg_conf_df = reg_conf_df.drop_duplicates()
reg_conf_df = reg_conf_df.sort_values(['country', 'commodity', 'rtype', 'ihs_parameter_name', 'ihs_field_names', 'regression_months', 'regression_start_date', 'regression_end_date', 'regression_hdd_base', 'regression_cdd_base', 'regression_others', 'regression_degrees'])

#%%

#reg_conf_df = reg_conf_df[['country', 'ihs_parameter_name', 'ihs_field_names', 'regression_months', 'regression_start_date', 'regression_end_date', 'regression_hdd_base', 'regression_cdd_base', 'regression_others', 'regression_degrees', 'r_sq']]

#%%

final_df = final_df.groupby(level=0, axis=1).sum().replace(0, np.nan).dropna(subset=[x for x in final_df.columns if x.endswith('_predicted')], how='any').sort_index()
final_df['actual'] = final_df[[x for x in final_df.columns if x.endswith('_mmcm')]].sum(axis=1).replace(0, np.nan) #careful will need to fix later suffix
final_df['predicted'] = final_df[[x for x in final_df.columns if x.endswith('_predicted')]].sum(axis=1).replace(0, np.nan)
final_df['delta'] = final_df[[x for x in final_df.columns if x.endswith('_delta')]].sum(axis=1).replace(0, np.nan)

#%%

latest_actual_date = final_df['actual'].dropna().index.max()
bar_line_chart(country, final_df['actual'], final_df['predicted'], final_df['delta'], country + ' act vs fcst', 'delta', latest_actual_date)

#%%

final_df.to_csv(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\demand_model_output\jp-2023-10-24.csv')

final_df.index

#%%

prev_final_df = pd.read_csv(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\demand_model_output\jp-2023-10-03.csv').set_index('CURVE_START_DT')
prev_final_df.index = prev_final_df.index.astype('datetime64[ns]')
wow_changes = final_df - prev_final_df


#%%

country = 'south korea'
country = 'japan'
country = 'turkey'
commodity = 'gas'
commodity = 'power'
rtype = 'demand'
#rtype = 'supply'
#model = 'residential'
model = 'commercial'
(model = 'power')

month_split = 'all'
#month_split = 'q4'
#month_split = 'roy'

#%%





#%%



#reg_test_final_df = pd.concat([reg_test_q4_df,reg_test_roy_df]).sort_index()



#%%


#%%

import itertools

#%%

# 'HDDs', 'CDDs',

xs = ['NEWC Fwd', 'JKM', 'ICE Brent Fwd (3-0-1)', 'GDP', 'Solar', 'ny', 'hol', 'we', 'wd', 'HDDs', 'HDD2s', 'CDDs', 'CDD2s', 'dummy_sum', 'dummy_win'] #'Nuclear',
#xs = ['NEWC Fwd', 'JKM', 'ICE Brent Fwd (3-0-1)', 'GDP', 'Nuclear', 'Solar']
num_xs = len(xs)
xs_list = []

for i in range(len(xs) + 1):
    for subset in itertools.combinations(xs, i):
        print(subset)
        #if len(subset) > 0:
        xs_list.append(subset)

#%%

#ys = ['residential_mmcm', 'power_mmcm', 'commercial_mmcm', 'industrial_mmcm', 'other_mmcm']
ys = ['power_mmcm']#, 'industry_mmcm'] #, 'residential_mmcm', 'commercial_mmcm', 'transportation_mmcm', 'own_use_and_losses_mmcm'] #

#%%

len(ys) * len(xs_list) * len(dates_list) * len(range(1, 4)) * len(models_config[country][commodity][rtype]['models'][model]['month_splits'])

#%%

#xs_ys = { y : xs for y in ys }
xs_ys = [(y, x) for y in ys for x in xs + ['HDDs', 'CDDs']]

#%%

plot_xs_vs_ys(country, inputs_df, xs_ys)

#%%

reg_conf_2 = []
final_df2 = {} #pd.DataFrame()
reg_id = 0
#model='commercial'
#model='other'
#model='residential'
#month_split = 'q4'

errs = []

#%%

models_config = {}
with open(r'\\UK1-W-Z8-16\Users\swasti\Documents\Share\fund_reg_conf.json', 'r') as f:
    models_config = json.loads(f.read())

HCDD = calc_hdds_cdds(temps_m, regression_hdd_base, regression_cdd_base)
inputs_df_orig = merge_inputs(x_date_features, prices_all_pivot, JKM_Platts_df, GDP_df, Nuclear_df, solar_full, HCDDs, ihs_actuals)

st = datetime.now()

for model in models_config[country][commodity][rtype]['models']:
    ihs_field_names = models_config[country][commodity][rtype]['models'][model]['ihs_field_names']
    y_parameter = '+'.join(ihs_field_names)        
    ihs_actuals[y_parameter] = ihs_actuals[ihs_field_names].sum(axis=1)
    
    for month_split in models_config[country][commodity][rtype]['models'][model]['month_splits']:
        regression_conf = models_config[country][commodity][rtype]['models'][model]['month_splits'][month_split]
        regression_hdd_base = regression_conf['parameters']['hdd_base']
        regression_cdd_base = regression_conf['parameters']['cdd_base']    
        
        inputs_df = inputs_df_orig.copy()
    
        regression_months = np.nan            
        if 'months' in regression_conf:
            regression_months = regression_conf['months']
            inputs_df = inputs_df[inputs_df.index.month.isin(regression_months)]
        
        regression_others = np.nan
        x_parameters = ['HDDs', 'CDDs']
        if 'others' in regression_conf['parameters']:
            regression_others = regression_conf['parameters']['others']
            x_parameters = x_parameters + regression_others
        
        #regression_start_date = datetime(2018,1,1)
        regression_start_date = inputs_df[x_parameters + [y_parameter]].dropna(how='any').index.min() # [~inputs_df[y_parameter].isna()].index.min() #datetime(2023,4,1)
        regression_end_date = inputs_df[x_parameters + [y_parameter]].dropna(how='any').index.max() #datetime(2023,4,1)
        
        dates_list = get_reg_unique_dates(max(datetime(2018,6,1), regression_start_date), regression_end_date) #'regression_start_date'
        dates_list = [x for x in dates_list if (x[1] - x[0]).days > 365*3.5 and x[1] <= datetime(2022, 9, 1)]
        
        for regression_others in xs_list:
            x_parameters = [] #'HDDs', 'CDDs'
            x_parameters = x_parameters + list(regression_others)
            for regression_start_date, regression_end_date in dates_list:
                if len(inputs_df[(inputs_df.index >= regression_start_date) & (inputs_df.index <= regression_end_date)]) < 2:
                    continue
                for regression_degrees in range(1, 4):
                    print(reg_id)
                    #reg_id += 1
                    #print('Running: ', ihs_parameter_name, ihs_field_names, regression_months, regression_start_date, regression_end_date, regression_hdd_base, regression_cdd_base, regression_degrees)
                    try:
                        forecast_df, in_dps, out_dps, in_rmse, out_rmse, in_r2, out_r2 = run_regressions(inputs_df, regression_start_date, regression_end_date, x_parameters, y_parameter, regression_degrees, True)
                        r_shrinkage = in_r2 - out_r2                    
                        #if r_shrinkage <= 0.15 and in_r2 >= 0.6 and out_r2 >= 0.6 and out_dps >= 4:
                        reg_conf_2.append([reg_id, country, ihs_parameter_name, ihs_field_names, regression_months, regression_start_date, regression_end_date, regression_hdd_base, regression_cdd_base, regression_others, regression_degrees, in_dps, out_dps, in_rmse, out_rmse, in_r2, out_r2, r_shrinkage])
                        #final_df2[reg_id] = forecast_df
                        #final_df2 = pd.concat([final_df2, forecast_df], axis=1)
                        reg_id += 1
                    except Exception as e:
                        errs.append([regression_start_date, regression_end_date, x_parameters, y_parameter, regression_degrees, str(e)])

et = datetime.now()

print('Time taken (mins): ', (et - st).total_seconds() / 60)

print(reg_id)

#%%


reg_conf_df2 = pd.DataFrame(reg_conf_2, columns=['reg_id', 'country', 'ihs_parameter_name', 'ihs_field_names', 'regression_months', 'regression_start_date', 'regression_end_date', 'regression_hdd_base', 'regression_cdd_base', 'regression_others', 'regression_degrees', 'in_dps', 'out_dps', 'in_rmse', 'out_rmse', 'in_r2', 'out_r2', 'r_shrinkage'])
reg_conf_df2['ihs_field_names'] = reg_conf_df2['ihs_field_names'].astype(str)
reg_conf_df2['regression_months'] = reg_conf_df2['regression_months'].astype(str)
reg_conf_df2['regression_others'] = reg_conf_df2['regression_others'].astype(str)
reg_conf_df2 = reg_conf_df2.drop_duplicates()
reg_conf_df2 = reg_conf_df2.sort_values(['country', 'ihs_parameter_name', 'ihs_field_names', 'regression_months', 'regression_start_date', 'regression_end_date', 'regression_hdd_base', 'regression_cdd_base', 'regression_others', 'regression_degrees', 'in_rmse', 'out_rmse', 'in_r2', 'out_r2'])

#%%

r = reg_conf_df2[reg_conf_df2['r_shrinkage'] <= 0.1].sort_values(['country', 'ihs_parameter_name', 'ihs_field_names', 'regression_months', 'out_r2', 'in_r2', 'out_rmse', 'in_rmse'], ascending=False)
rc = r[(r['ihs_field_names'] == '[\'commercial_mmcm\']')][:1000]
ri = r[(r['ihs_field_names'] == '[\'industry_mmcm\']')][:1000]
ro = r[(r['ihs_field_names'] == '[\'own_use_and_losses_mmcm\']')][:1000]
rp = r[(r['ihs_field_names'] == '[\'power_mmcm\']')][:1000]
rr = r[(r['ihs_field_names'] == '[\'residential_mmcm\']')][:1000]
rt = r[(r['ihs_field_names'] == '[\'transportation_mmcm\']')][:1000]

#reg_conf_df2_ind = reg_conf_df2.copy()
#reg_conf_df2_oth = reg_conf_df2.copy()
#reg_conf_df2 = reg_conf_df2[(reg_conf_df2['in_rmse'] != 0) & (reg_conf_df2['out_rmse'] != 0)]
#reg_conf_df2['score'] = (((reg_conf_df2['in_dps'] * reg_conf_df2['in_r2']) / reg_conf_df2['in_rmse']) * 0.33) + (((reg_conf_df2['out_dps'] * reg_conf_df2['out_r2']) / reg_conf_df2['out_rmse']) * 0.66)

#reg_conf_df2['r_shrinkage'] = reg_conf_df2['in_r2'] - reg_conf_df2['out_r2']
#r = reg_conf_df2[reg_conf_df2['regression_degrees'] == 1].sort_values('out_r2', ascending=False)
#r = reg_conf_df2[(reg_conf_df2['out_r2'] >= 0.65) & (reg_conf_df2['out_dps'] >= 10) & (reg_conf_df2['out_rmse'] <= 12)].sort_values('out_r2', ascending=False)
reg_conf_df2['ihs_parameter_name'].drop_duplicates()

reg_conf_df2['ihs_field_names'].drop_duplicates()

r = {x[0] : x[1] for x in reg_conf_df2.groupby('ihs_field_names')}

#%%

r = reg_conf_df2[(reg_conf_df2['r_shrinkage'] <= 0.1) & (reg_conf_df2['regression_end_date'] <= '2022-09-01') & (reg_conf_df2['out_r2'] >= 0.8)].sort_values('out_r2', ascending=False) #  & (reg_conf_df2['out_dps'] >= 10) & (reg_conf_df2['in_r2'] >= 0.8) & (reg_conf_df2['out_rmse'] <= 12) & ]
r = {x[0] : x[1][:200] for x in r.groupby('ihs_field_names')}
r = reg_conf_df2[reg_conf_df2['ihs_field_names'] == '[\'residential_mmcm\']'].sort_values('out_r2', ascending=False)[:1000]

#%%

rids = r['reg_id'].drop_duplicates().to_list()
final_df3 = {x : final_df2[x] for x in final_df2.keys() if x in rids}

#%%

rr = r[:1000] #[r['ihs_field_names'] != '[\'residential_mmcm\']']

#%%

rr = {x[0] : x[1] for x in r.groupby('ihs_field_names')}

#%%

for reg_id in rids:
    earliest_actual_date, latest_actual_date = reg_conf_df2.loc[reg_conf_df2['reg_id'] == reg_id, ['regression_start_date', 'regression_end_date']].iloc[0]
    bar_line_chart2(country, final_df2[reg_id][y_parameter], final_df2[reg_id][y_parameter + '_predicted'], final_df2[reg_id][y_parameter + '_delta'], y_parameter + ' act vs fcst for ' + month_split, y_parameter + '_delta', earliest_actual_date, latest_actual_date, reg_id, y_parameter)

#%%

chart2(country, y_parameter, final_df3)

#%%

def bar_line_chart2(country, df, df1, df2, title, primary_y_axis_title, earliest_actual_date, latest_actual_date, reg_id, y_parameter):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    data = list((df2).round(1))

    fig.add_bar(x=df2.index, y=df2, name=df2.name+ ' (left y axis)', text=list(data), secondary_y=False, textposition='inside', textfont=dict(color='black'))

    fig.add_trace(go.Scatter(x=df.index, y=df, name=df.name + ' (right y axis)', line=dict(color='black', width=3)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df1.index, y=df1, name=df1.name + ' (right y axis)', line=dict(color='red', width=3)), secondary_y=True)

    fig.add_vline(x=earliest_actual_date, line_width=3, line_dash="dash", line_color="yellow")

    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")

    fig.update_layout(title=title + ' last updated: ' + str(datetime.now()), xaxis_title='Date', yaxis_title=primary_y_axis_title, hovermode='x unified')
    fig.update_yaxes(title_text='mcm/d', secondary_y=True)
    fig.layout.yaxis.showgrid = False
    
    if reg_id == -1:
        fig.write_html(rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\{country}_{title}.html', include_plotlyjs='cdn')
    else:
        fig.write_html(rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\iterations\{country}\{y_parameter}\{reg_id}_{title}.html', include_plotlyjs='cdn')
    
    #fig.show()

#%%
    
def chart2(country, y_parameter, final_df3):
    chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\all_sig_runs_{country}_{y_parameter}.html'
    
    fig = go.Figure(data=[
             go.Scatter(name=x, x=final_df3[x].index, y=final_df3[x][y_parameter + '_predicted']) for x in final_df3.keys()
         ])
    
    df2 = pd.DataFrame()
    #ewq = 0
    for x in final_df3.keys():    
        df2 = pd.concat([df2, final_df3[x]], axis=1)
    #    ewq += 1
        
    df2['min'] = df2[[y_parameter + '_predicted']].min(axis=1)
    df2['max'] = df2[[y_parameter + '_predicted']].max(axis=1)
    df2['actual'] = df2[[y_parameter]].max(axis=1)
    
    fig.add_scatter(name='min', x=df2.index, y=df2['min'], line=dict(color='red', width=3))
    fig.add_scatter(name='max', x=df2.index, y=df2['max'], line=dict(color='green', width=3))
    fig.add_scatter(name='actual', x=df2.index, y=df2['actual'], line=dict(color='black', width=3))
    
    latest_actual_date = datetime(2023, 4, 1)
    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")
        
    fig.update_layout(
        xaxis=dict(
                rangeselector=dict(
                        buttons=[
                                dict(count=7,
                                     label='1w',
                                     step='day',
                                     stepmode='backward'),
                                dict(count=1,
                                     label='1m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=3,
                                     label='3m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=6,
                                     label='6m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=1,
                                     label='YTD',
                                     step='year',
                                     stepmode='todate'),
                                dict(count=1,
                                     label='1y',
                                     step='year',
                                     stepmode='backward'),
                                dict(step='all', label='All')
                            ]
                    ),
                rangeslider=dict(visible=True),
                type='date',
                dtick='M1'
                #range=[today - relativedelta(days=365), today + relativedelta(days=1)],
                # rangebreaks=[
                #          dict(bounds=["sat", "mon"]), #hide weekends        
                #     ],
            ),
        #hovermode='x unified',
        # height=750,
        title={
                'text' : y_parameter,
                'x' : 0.5, #'center',
                'yanchor' : 'top',
                'font_size' : 16
            })

    fig.write_html(chart_file, include_plotlyjs='cdn')  



inputs_df_orig.drop('city_gas_sum_mmcm', axis=1).to_csv(r'C:\Users\svaiyani\Documents\MyFiles\epics\Data Scrapes\Salman\Japan\inputs_df.csv')

import pandas as pd
import requests


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
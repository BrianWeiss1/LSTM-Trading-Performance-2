import requests
import pandas as pd

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

from special_functions import get_conventional_indicators_data, log_returns, normalize_cci, normalize_macd

def get_data(timeFrame="5"):
    api_key = 'your_api_key'
    url = 'https://www.alphavantage.co/query'
    full_data = pd.DataFrame()

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 5, 1)

    current_date = start_date
    while current_date < end_date:
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': 'SPY',
            'interval': f'{timeFrame}min',
            'outputsize': 'full', 
            'datatype': 'json',
            'apikey': api_key,
            'month': datetime.strptime(str(current_date), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m')
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()  
            print(data)
            if f'Time Series ({timeFrame}min)' in data:
                time_series = data[f'Time Series ({timeFrame}min)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index = pd.to_datetime(df.index)
                full_data = pd.concat([full_data, df])
                current_date += timedelta(days=30)  
            
            else:
                print("Time series data not found in response.")
                input("Did you change your VPN?")
        else:
            print(f"Error: {response.status_code}")

        print(datetime.strptime(str(current_date), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m'))

    # Sort the data by date
    full_data = full_data.sort_index()
    full_data.to_csv(f'{timeFrame}min_data_SPY_2019_to_2024.csv')

    print(f"Data saved to {timeFrame}min_data_SPY_2019_to_2024.csv")
    print(full_data)

def fix_data(csv_file: str):
    df = pd.read_csv(f"Data/SPY/{csv_file}", index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.between_time('04:00', '20:00')
    return df

    

def normalize_data(csv_file: str):
    df = fix_data(csv_file)
    scaler = MinMaxScaler()
    normalized_volume = scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
    df['Close_L'] = log_returns(df, 'Close')
    df['High_L'] = log_returns(df, 'High')
    df['Low_L'] = log_returns(df, 'Low')
    df['Volume_N'] = normalized_volume

    df = get_conventional_indicators_data(df)
    df['SMA_20'] = log_returns(df, 'SMA_20')
    df['EMA_9'] = log_returns(df, 'EMA_9')
    df['EMA_21'] = log_returns(df, 'EMA_21')
    df['UBBand'] = log_returns(df, 'UBBand')
    df['LBBand'] = log_returns(df, 'LBBand')
    df['AD'] = scaler.fit_transform(df['AD'].values.reshape(-1, 1))

    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    df = df.dropna()  

    df = df[(df.index >= '2018-01-01')]
    df.to_csv(f"Data/SPY/Normalized/normalized_{csv_file}.csv")
    print("saved")
    return df


if __name__ == '__main__':
    df = normalize_data("5min_data_SPY_2019_to_2024.csv")
    

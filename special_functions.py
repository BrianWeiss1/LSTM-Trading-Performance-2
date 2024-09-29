import numpy as np
import pandas as pd
import talib

def normalize_cci(value):
    if value > 200:
        return 0.99
    elif value < -200:
        return -0.99
    else:
        return (value / 200) * 0.99
    
def log_returns(df: pd.DataFrame, column: str):
    return np.log(df[column] / df[column].shift(1))

def inverse_log_returns(df: pd.DataFrame, column: str):
    # Initialize a new column for original prices
    original_prices = np.zeros(len(df))  # Create an array to hold original prices
    original_prices[0] = df[column].iloc[0]  # Assume the first price is the initial price
    
    # Calculate original prices using log returns
    for i in range(1, len(df)):
        original_prices[i] = original_prices[i - 1] * np.exp(df[column].iloc[i])
    
    return pd.Series(original_prices, index=df.index)  # Return a Series with the same index as df
def get_conventional_indicators_data(df):
    # ------- Normalized After ------- #
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['EMA_9'] = talib.EMA(df['Close'], timeperiod=9)
    df['EMA_21']  = talib.EMA(df['Close'], timeperiod=21)
    
    upperband, middleband, lowerband = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['UBBand'] = upperband
    df['LBBand'] = lowerband
    
    df["AD"] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    # ------- Normalized After ------- #
    
    
    # ------- Custom Normalization ------- #
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = normalize_macd(macdhist)
    
    k, d = talib.STOCH(df['High'], df['Low'], df['Close'], 
            fastk_period=14, slowk_period=3, slowd_period=3)    
    df['STOCH'] = (k - d)/100 

    df['RSI']  = talib.RSI(df['Close'], timeperiod=14)/100 
    
    
    df['CCI'] = (talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)).apply(normalize_cci)
    
    df["ATR"] = (talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)) / df['Close']    
    # ------- Custom Normalization ------- #


    # ------- Already Normalized ------- #
    df['CMF'] = (( ( (df['Close'] - df['Low']) - (df['High'] - df['Close']) ) / (df['High'] - df['Low']) ) * df['Volume']).rolling(window=21).sum() / df['Volume'].rolling(window=21).sum()
    # ------- Already Normalized ------- #
    
    return df


def normalize_macd(macd_series):
    normalized_macd = macd_series.copy()
    normalized_macd[normalized_macd > 1] = 1
    normalized_macd[normalized_macd < -1] = -1
    in_range_mask = (normalized_macd > -1) & (normalized_macd < 1)
    normalized_macd[in_range_mask] = normalized_macd[in_range_mask] * 0.99
    return normalized_macd
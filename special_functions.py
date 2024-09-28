import numpy as np
import pandas as pd

def normalize_cci(value):
    if value > 200:
        return 0.99
    elif value < -200:
        return -0.99
    else:
        # Scale CCI from [-200, 200] to [-0.99, 0.99]
        return (value / 200) * 0.99
    
def log_returns(df: pd.DataFrame, column: str):
    return np.log(df[column] / df[column].shift(1))

def normalize_macd(macd_series):
    # Apply normalization based on the specified criteria
    normalized_macd = macd_series.copy()
    
    # Set values greater than 1 to 1
    normalized_macd[normalized_macd > 1] = 1
    
    # Set values less than -1 to -1
    normalized_macd[normalized_macd < -1] = -1
    
    # Adjust values in between to be within 0.99 and -0.99
    # Here, we rescale the values between -1 and 1 to fit between -0.99 and 0.99
    in_range_mask = (normalized_macd > -1) & (normalized_macd < 1)
    normalized_macd[in_range_mask] = normalized_macd[in_range_mask] * 0.99
    
    return normalized_macd
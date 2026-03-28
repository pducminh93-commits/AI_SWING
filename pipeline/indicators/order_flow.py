import pandas as pd
import talib as ta

def calculate_obv(df):
    """
    Calculates On Balance Volume (OBV).
    """
    df["obv"] = ta.obv(df["close"], df["volume"])
    return df

def calculate_cmf(df, period=20):
    """
    Calculates Chaikin Money Flow (CMF).
    """
    cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=period)
    df = pd.concat([df, cmf], axis=1)
    return df

def calculate_vwap(df):
    """
    Calculates Volume Weighted Average Price (VWAP).
    """
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    return df

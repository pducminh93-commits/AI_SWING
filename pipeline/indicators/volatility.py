import pandas as pd
import talib as ta
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR).
    """
    atr = ta.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    df["atr"] = atr
    return df

import pandas as pd
import talib as ta
import numpy as np

def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    df["rsi"] = ta.RSI(df["close"], timeperiod=period)
    return df

def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Calculates the Stochastic Oscillator.
    """
    slowk, slowd = ta.STOCH(df["high"], df["low"], df["close"], 
                            fastk_period=k_period, slowk_period=3, slowk_matype=0, 
                            slowd_period=d_period, slowd_matype=0)
    df["slowk"] = slowk
    df["slowd"] = slowd
    return df

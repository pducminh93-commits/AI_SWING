import pandas as pd
import talib as ta
import numpy as np

def calculate_ema(df, periods=[9, 21, 55, 200]):
    """
    Calculates Exponential Moving Averages (EMA).
    """
    for period in periods:
        df[f"ema_{period}"] = ta.EMA(df["close"], timeperiod=period)
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculates Moving Average Convergence Divergence (MACD).
    """
    macd, macdsignal, macdhist = ta.MACD(df["close"], fastperiod=fast, slowperiod=slow, signalperiod=signal)
    df["macd"] = macd
    df["macdsignal"] = macdsignal
    df["macdhist"] = macdhist
    return df

def calculate_adx(df, period=14):
    """
    Calculates the Average Directional Index (ADX).
    """
    adx = ta.ADX(df["high"], df["low"], df["close"], timeperiod=period)
    df["adx"] = adx
    return df

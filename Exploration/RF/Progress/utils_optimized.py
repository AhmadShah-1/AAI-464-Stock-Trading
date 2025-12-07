import numpy as np
import pandas as pd

def create_features(df):
    df = df.copy()

    # Volatility & range 
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()

    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    df['realized_vol_15'] = df['Close'].pct_change().rolling(15).std()
    df['realized_vol_20'] = df['Close'].pct_change().rolling(20).std()
    df['realized_vol_30'] = df['Close'].pct_change().rolling(30).std()

    df['vol_ratio_15_30'] = df['realized_vol_15'] / df['realized_vol_30']

    # Trend 
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['sma_5'] = df['Close'].rolling(5).mean().shift()
    df['sma_10'] = df['Close'].rolling(10).mean().shift()
    df['sma_30'] = df['Close'].rolling(30).mean().shift()
    df['sma_50'] = df['Close'].rolling(50).mean().shift()

    def slope(series):
        y = series.values
        x = np.arange(len(y))
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]

    df['sma_slope_20'] = df['Close'].rolling(20).apply(slope)
    df['sma_slope_30'] = df['Close'].rolling(30).apply(slope)

    # Volume 
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Momentum 
    df['momentum_30'] = df['Close'].pct_change(30)

    # MACD 
    EMA_12 = df['Close'].ewm(span=12, min_periods=12).mean()
    EMA_26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD_12_26'] = EMA_12 - EMA_26
    df['MACD_signal_9'] = df['MACD_12_26'].ewm(span=9, min_periods=9).mean()

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

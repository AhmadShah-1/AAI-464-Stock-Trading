"""Feature engineering utilities for stock data."""

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    if df.empty or len(df) < 50:
        return df
    
    df = df.copy()
    
    try:
        # Add all technical indicators
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume",
            fillna=True
        )
    except Exception as e:
        # If automatic addition fails, add key indicators manually
        print(f"Warning: Could not add all indicators automatically: {e}")
        
        # Moving Averages
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based features.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added price features
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Price changes
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_change_pct'] = df['price_change'] / df['close'].shift(1) * 100
    
    # High-Low range
    df['hl_range'] = df['high'] - df['low']
    df['hl_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Add lagged features for time series prediction.
    
    Args:
        df: DataFrame with price data
        lags: List of lag periods
    
    Returns:
        DataFrame with lagged features
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag) if 'returns' in df.columns else np.nan
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    return df


def create_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        df: DataFrame with price data
        horizon: Prediction horizon in days
    
    Returns:
        DataFrame with target variable
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Future return
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    
    # Classification targets
    df['target_direction'] = np.where(df['future_return'] > 0, 1, 0)  # 1: up, 0: down
    
    # Multi-class target (buy, hold, sell)
    df['target_action'] = pd.cut(
        df['future_return'],
        bins=[-np.inf, -0.02, 0.02, np.inf],
        labels=[0, 1, 2]  # 0: sell, 1: hold, 2: buy
    )
    
    return df


def prepare_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: DataFrame with raw OHLCV data
        horizon: Prediction horizon
    
    Returns:
        DataFrame with all features
    """
    if df.empty:
        return df
    
    df = add_price_features(df)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = create_target_variable(df, horizon)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

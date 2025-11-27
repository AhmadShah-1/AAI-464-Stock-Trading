import pandas as pd
import numpy as np



# Create technical features from OHLCV data
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """        
    Args:
        df: DataFrame with columns: open, high, low, close, volume, timestamp
        
    Returns: DataFrame with additional feature columns
    """
    df = df.copy()
    
    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility (rolling standard deviation of returns)
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # Simple Moving Averages (SMA)
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Price relative to moving averages
    df['price_to_sma_5'] = df['close'] / df['sma_5']
    df['price_to_sma_10'] = df['close'] / df['sma_10']
    df['price_to_sma_20'] = df['close'] / df['sma_20']
    
    # Moving average crossovers (signals)
    df['sma_5_10_cross'] = df['sma_5'] - df['sma_10']
    df['sma_10_20_cross'] = df['sma_10'] - df['sma_20']
    
    # Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    
    # Price range features
    df['high_low_range'] = df['high'] - df['low']
    df['close_to_high'] = (df['high'] - df['close']) / df['high_low_range']
    df['close_to_low'] = (df['close'] - df['low']) / df['high_low_range']
    
    return df

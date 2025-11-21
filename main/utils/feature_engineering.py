"""
Feature engineering for stock market data.
Creates technical features and target variables for ML model training.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Creates features from raw OHLCV stock data."""
    
    def __init__(self, forward_days: int = 5, threshold: float = 0.02):
        """
        Initialize feature engineer.
        
        Args:
            forward_days: Number of days to look forward for target calculation
            threshold: Percentage threshold for buy/sell signals (e.g., 0.02 = 2%)
        """
        self.forward_days = forward_days
        self.threshold = threshold
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, timestamp
            
        Returns:
            DataFrame with additional feature columns
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
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for classification.
        
        Target classes:
        - 0 (sell): Future return < -threshold
        - 1 (hold): Future return between -threshold and +threshold
        - 2 (buy): Future return > +threshold
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        
        # Calculate forward returns
        df['forward_returns'] = df['close'].shift(-self.forward_days) / df['close'] - 1
        
        # Create target based on forward returns
        df['target'] = 1  # Default to hold
        df.loc[df['forward_returns'] > self.threshold, 'target'] = 2  # Buy
        df.loc[df['forward_returns'] < -self.threshold, 'target'] = 0  # Sell
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw OHLCV DataFrame
            include_target: Whether to create target variable (False for prediction)
            
        Returns:
            DataFrame with all features and optionally target
        """
        # Create features
        df = self.create_features(df)
        
        # Create target if requested
        if include_target:
            df = self.create_target(df)
        
        # Drop rows with NaN values (from rolling calculations)
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        Get list of feature column names (excluding target and original OHLCV).
        
        Returns:
            List of feature column names
        """
        return [
            'returns', 'log_returns',
            'volatility_5', 'volatility_10',
            'volume_change', 'volume_ratio',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20',
            'sma_5_10_cross', 'sma_10_20_cross',
            'momentum_5', 'momentum_10',
            'close_to_high', 'close_to_low'
        ]


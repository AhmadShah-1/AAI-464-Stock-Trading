"""
Feature engineering for stock market data.
Creates technical features and target variables for ML model training.
"""

import pandas as pd
import numpy as np
from .create_features import create_features
from .create_target import create_target
from .prepare_data import prepare_data

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
        return create_features(df)

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_target(df, forward_days=self.forward_days, threshold=self.threshold)

    def prepare_data(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        return prepare_data(df, include_target=include_target, forward_days=self.forward_days, threshold=self.threshold)
    
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


import pandas as pd
import numpy as np
from .create_features import create_features
from .create_target import create_target

# Complete feature engineering pipeline
def prepare_data(df: pd.DataFrame, include_target: bool = True, forward_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """        
    Args:
        df: Raw OHLCV DataFrame
        include_target: Whether to create target variable (False for prediction)
        forward_days: Number of days to look forward for target calculation
        threshold: Percentage threshold for buy/sell signals (e.g., 0.02 = 2%)
        
    Returns:
        DataFrame with all features and optionally target
    """
    # Create features
    df = create_features(df)
    
    # Create target if requested
    if include_target:
        df = create_target(df, forward_days=forward_days, threshold=threshold)
    
    # Drop rows with NaN values (from rolling calculations)
    df = df.dropna()
    
    return df
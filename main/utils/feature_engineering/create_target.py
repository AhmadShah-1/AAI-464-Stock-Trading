import pandas as pd
import numpy as np



# Create target variable for classification
def create_target(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """        
    Target classes:
    - 0 (sell): Future return < -threshold
    - 1 (hold): Future return between -threshold and +threshold
    - 2 (buy): Future return > +threshold
    
    Args:
        df: DataFrame with price data
        forward_days: Number of days to look forward for target calculation
        threshold: Percentage threshold for buy/sell signals (e.g., 0.02 = 2%)
        
    Returns:
        DataFrame with 'target' column added
    """
    df = df.copy()
    
    # Calculate forward returns
    df['forward_returns'] = df['close'].shift(-forward_days) / df['close'] - 1
    
    # Create target based on forward returns
    df['target'] = 1                                                # Default to HOLD
    df.loc[df['forward_returns'] > threshold, 'target'] = 2    # BUY
    df.loc[df['forward_returns'] < -threshold, 'target'] = 0   # Sell
    
    return df
    
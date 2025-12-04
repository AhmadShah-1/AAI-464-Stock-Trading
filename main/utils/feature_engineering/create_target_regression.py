import pandas as pd
import numpy as np


# Create target variable for regression
def create_target_regression(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Create continuous forward returns as target for regression.

    Unlike classification, this returns the raw forward returns (continuous values)
    instead of categorical labels (0/1/2).

    Args:
        df: DataFrame with price data
        forward_days: Number of days to look forward for target calculation (default: 5)

    Returns:
        DataFrame with 'target' column added (continuous forward returns)
    """
    df = df.copy()

    # Calculate forward returns (continuous values)
    df['forward_returns'] = df['close'].shift(-forward_days) / df['close'] - 1

    # For regression, target is the continuous forward returns
    df['target'] = df['forward_returns']

    return df

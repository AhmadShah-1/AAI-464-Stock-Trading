import numpy as np
import pandas as pd
from features_karina import compute_returns


def backtest_static_weights(price_df: pd.DataFrame, weights: np.ndarray):
    """
    Backtests a fixed-weight portfolio over historical prices.
    """
    returns = compute_returns(price_df)
    port_rets = returns @ weights
    return port_rets

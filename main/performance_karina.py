import numpy as np
import pandas as pd
from config_karina import Config


def portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Computes daily portfolio returns from asset returns and weights.
    """
    port_rets = returns_df @ weights
    return port_rets


def cumulative_returns(port_returns: pd.Series) -> pd.Series:
    """
    Converts daily returns into cumulative growth curve.
    """
    return (1 + port_returns).cumprod()


def annualized_return(port_returns: pd.Series) -> float:
    """
    Annualized geometric return.
    """
    total_growth = (1 + port_returns).prod()
    n_years = len(port_returns) / Config.TRADING_DAYS_PER_YEAR
    return total_growth ** (1 / n_years) - 1


def annualized_volatility(port_returns: pd.Series) -> float:
    """
    Annualized standard deviation.
    """
    return port_returns.std() * np.sqrt(Config.TRADING_DAYS_PER_YEAR)


def sharpe_ratio(port_returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    Risk-adjusted return.
    """
    ann_ret = annualized_return(port_returns)
    ann_vol = annualized_volatility(port_returns)

    if ann_vol == 0:
        return np.nan

    return (ann_ret - risk_free) / ann_vol


def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Worst peak-to-trough loss.
    """
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

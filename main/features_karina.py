import numpy as np
import pandas as pd
from config_karina import Config


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns from price data.
    """
    returns = np.log(price_df / price_df.shift(1))
    returns = returns.dropna()
    return returns


def estimate_mu_sigma(returns_df: pd.DataFrame):
    """
    Estimates annualized expected returns vector (mu)
    and annualized covariance matrix (Sigma).
    """

    mu = returns_df.mean() * Config.TRADING_DAYS_PER_YEAR
    sigma = returns_df.cov() * Config.TRADING_DAYS_PER_YEAR

    return mu.values, sigma.values

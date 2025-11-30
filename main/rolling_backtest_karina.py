import numpy as np
import pandas as pd
from features_karina import compute_returns, estimate_mu_sigma
from optimizer_karina import max_sharpe_portfolio


def rolling_backtest(price_df: pd.DataFrame,
                      train_window: int = 252,
                      test_window: int = 21):
    """
    Rolling out-of-sample backtest.
    Re-optimizes every test_window days using past train_window data.
    """

    returns = compute_returns(price_df)
    dates = returns.index

    portfolio_returns = []
    rebalance_dates = []

    i = train_window

    while i + test_window <= len(returns):
        train_returns = returns.iloc[i - train_window:i]
        test_returns = returns.iloc[i:i + test_window]

        mu, sigma = estimate_mu_sigma(train_returns)
        weights = max_sharpe_portfolio(mu, sigma)
        weights = weights / weights.sum()

        realized_port_rets = test_returns @ weights

        portfolio_returns.append(realized_port_rets)
        rebalance_dates.append(test_returns.index[0])

        i += test_window

    portfolio_returns = pd.concat(portfolio_returns)

    return portfolio_returns, rebalance_dates

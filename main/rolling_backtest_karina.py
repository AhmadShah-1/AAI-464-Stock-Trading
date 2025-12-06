import numpy as np
import pandas as pd
from main.utils.feature_engineering.features_karina import compute_returns, estimate_mu_sigma
from main.models.optimizer_karina import max_sharpe_portfolio

import numpy as np
import pandas as pd

from main.utils.feature_engineering.features_karina import (
    compute_returns,
    estimate_mu_sigma
)
from main.models.optimizer_karina import min_variance_portfolio
from main.utils.feature_engineering.features_karina import shrink_covariance


#############################
# DIAGNOSTIC HELPERS
#############################

def compute_turnover(weights_history):
    turnovers = []
    for t in range(1, len(weights_history)):
        turnover = np.sum(np.abs(weights_history[t] - weights_history[t - 1]))
        turnovers.append(turnover)
    return np.array(turnovers)


def weight_stability(weights_history):
    deltas = []
    for t in range(1, len(weights_history)):
        deltas.append(np.linalg.norm(weights_history[t] - weights_history[t - 1]))
    return np.array(deltas)


def marginal_risk_contribution(weights, sigma):
    portfolio_vol = np.sqrt(weights.T @ sigma @ weights)
    mrc = (sigma @ weights) / portfolio_vol
    trc = weights * mrc
    return trc, mrc


def asset_return_contributions(weights, test_returns):
    contrib = test_returns.mul(weights, axis=1)
    return contrib.sum()


def return_distribution_stats(portfolio_returns):
    stats = {
        "skew": portfolio_returns.skew(),
        "kurtosis": portfolio_returns.kurtosis(),
        "min_daily_return": portfolio_returns.min(),
        "max_daily_return": portfolio_returns.max()
    }
    return stats


#############################
# ROLLING BACKTEST
#############################

def rolling_backtest(
    price_df: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 21
):
    """
    Rolling out-of-sample backtest.
    Re-optimizes every test_window days using past train_window data.
    """

    returns = compute_returns(price_df)

    portfolio_returns = []
    rebalance_dates = []
    weights_history = []
    sigma_history = []

    i = train_window

    while i + test_window <= len(returns):

        train_returns = returns.iloc[i - train_window:i]
        test_returns = returns.iloc[i:i + test_window]

        _, sigma = estimate_mu_sigma(train_returns)

        sigma = shrink_covariance(sigma, shrinkage=0.2)

        weights = min_variance_portfolio(sigma)
        weights = weights / weights.sum()

        realized_port_rets = test_returns @ weights

        portfolio_returns.append(realized_port_rets)
        rebalance_dates.append(test_returns.index[0])
        weights_history.append(weights)
        sigma_history.append(sigma)

        i += test_window

    portfolio_returns = pd.concat(portfolio_returns)

    #############################
    # RUN DIAGNOSTICS
    #############################

    turnover = compute_turnover(weights_history)
    stability = weight_stability(weights_history)

    final_weights = weights_history[-1]
    final_sigma = sigma_history[-1]
    final_test_index = portfolio_returns.index

    trc, mrc = marginal_risk_contribution(final_weights, final_sigma)

    asset_contrib = asset_return_contributions(
        final_weights,
        returns.loc[final_test_index]
    )

    diagnostics = {
        "avg_turnover": float(np.mean(turnover)),
        "max_turnover": float(np.max(turnover)),
        "avg_weight_instability": float(np.mean(stability)),
        "risk_concentration_top_5": float(np.sort(trc)[-5:].sum()),
        "weight_concentration_top_5": float(np.sort(final_weights)[-5:].sum()),
        "asset_return_contribution": asset_contrib.sort_values(ascending=False),
        "distribution_stats": return_distribution_stats(portfolio_returns)
    }

    return portfolio_returns, rebalance_dates, weights_history, diagnostics

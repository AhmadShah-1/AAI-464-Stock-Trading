import cvxpy as cp
import numpy as np
from config_karina import Config


def min_variance_portfolio(Sigma: np.ndarray) -> np.ndarray:
    """
    Computes the minimum variance portfolio under:
    - fully invested
    - long-only
    - max weight per stock
    """

    n = Sigma.shape[0]
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, Sigma))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= Config.MAX_WEIGHT_PER_STOCK
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    weights = w.value
    weights = weights / weights.sum()  # safety normalization

    return weights


def max_sharpe_portfolio(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Practical max-Sharpe approximation using
    risk-penalized return:
        maximize(mu^T w - w^T Sigma w)
    """

    n = len(mu)
    w = cp.Variable(n)

    expected_return = mu @ w
    risk = cp.quad_form(w, Sigma)

    objective = cp.Maximize(expected_return - risk)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= Config.MAX_WEIGHT_PER_STOCK
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    weights = w.value
    weights = weights / weights.sum()  # safety normalization

    return weights

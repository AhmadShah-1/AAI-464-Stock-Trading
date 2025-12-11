import cvxpy as cp
import numpy as np
from main.config_karina import Config




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

def min_variance_portfolio(sigma):
    """
    Long-only minimum variance portfolio with a 10% cap per stock.

    Solves the optimization problem:

        minimize      wᵀ Σ w
        subject to    ∑ wᵢ = 1
                      wᵢ ≥ 0
                      wᵢ ≤ 0.10
    """
    import cvxpy as cp
    n = sigma.shape[0]

    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, sigma))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.10
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return w.value

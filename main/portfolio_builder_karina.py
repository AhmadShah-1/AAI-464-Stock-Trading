import numpy as np
import pandas as pd
from config_karina import Config


def build_portfolio(weights: np.ndarray,
                    prices: pd.Series,
                    capital: float = None):
    """
    Converts portfolio weights into dollar allocations and share counts.
    """

    if capital is None:
        capital = Config.TOTAL_CAPITAL

    prices = prices.astype(float)

    alloc_dollars = capital * weights
    shares = np.floor(alloc_dollars / prices.values)

    invested = shares * prices.values
    leftover_cash = capital - invested.sum()

    portfolio = pd.DataFrame({
        "price": prices.values,
        "weight": weights,
        "target_dollars": alloc_dollars,
        "shares": shares,
        "invested_dollars": invested
    }, index=prices.index)

    return portfolio, leftover_cash

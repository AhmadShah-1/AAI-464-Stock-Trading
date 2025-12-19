from main.data_loader import get_top_50_us_stocks, get_price_history
from main.utils.feature_engineering.features_karina import (
    compute_returns, estimate_mu_sigma
)
from main.models.optimizer_karina import (
    min_variance_portfolio, 
    max_sharpe_portfolio
)
from main.config_karina import Config
import numpy as np


symbols = get_top_50_us_stocks()
prices = get_price_history(symbols)

returns = compute_returns(prices)
mu, sigma = estimate_mu_sigma(returns)

w_min = min_variance_portfolio(sigma)
w_sharpe = max_sharpe_portfolio(mu, sigma)

print("Min-Variance Weights:")
print("Sum:", np.sum(w_min))
print("Min:", np.min(w_min))
print("Max:", np.max(w_min))

print("\nMax-Sharpe Weights:")
print("Sum:", np.sum(w_sharpe))
print("Min:", np.min(w_sharpe))
print("Max:", np.max(w_sharpe))

print("\nFirst 10 weights (Sharpe):")
print(w_sharpe[:10])

import matplotlib.pyplot as plt

INITIAL_CAPITAL = 1_000_000

#Convert returns to portfolio value
portfolio_value = INITIAL_CAPITAL * (1 + port_rets).cumprod()

#Plot
plt.figure()
portfolio_value.plot()
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()

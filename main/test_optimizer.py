from data_loader import get_top_50_us_stocks, get_price_history
from features_karina import compute_returns, estimate_mu_sigma
from optimizer_karina import min_variance_portfolio, max_sharpe_portfolio
from config_karina import Config
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

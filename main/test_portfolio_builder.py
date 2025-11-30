from data_loader import get_top_50_us_stocks, get_price_history
from features_karina import compute_returns, estimate_mu_sigma
from optimizer_karina import max_sharpe_portfolio
from portfolio_builder_karina import build_portfolio
from config_karina import Config
import numpy as np


symbols = get_top_50_us_stocks()
prices = get_price_history(symbols)

latest_prices = prices.iloc[-1]

returns = compute_returns(prices)
mu, sigma = estimate_mu_sigma(returns)

weights = max_sharpe_portfolio(mu, sigma)
weights = weights / weights.sum()

portfolio, leftover = build_portfolio(weights, latest_prices, Config.TOTAL_CAPITAL)

print(portfolio.sort_values("target_dollars", ascending=False).head(10))
print("\nTotal Invested:", portfolio["invested_dollars"].sum())
print("Leftover Cash:", leftover)

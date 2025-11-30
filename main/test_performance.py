from data_loader import get_top_50_us_stocks, get_price_history
from features_karina import compute_returns, estimate_mu_sigma
from optimizer_karina import max_sharpe_portfolio
from performance_karina import (
    cumulative_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown
)
from backtest_karina import backtest_static_weights


symbols = get_top_50_us_stocks()
prices = get_price_history(symbols)

returns = compute_returns(prices)
mu, sigma = estimate_mu_sigma(returns)

weights = max_sharpe_portfolio(mu, sigma)
weights = weights / weights.sum()

port_rets = backtest_static_weights(prices, weights)
cum = cumulative_returns(port_rets)

print("Annualized Return:", annualized_return(port_rets))
print("Annualized Volatility:", annualized_volatility(port_rets))
print("Sharpe Ratio:", sharpe_ratio(port_rets))
print("Max Drawdown:", max_drawdown(cum))

print("\nFirst 5 cumulative values:")
print(cum.head())

print("\nLast 5 cumulative values:")
print(cum.tail())

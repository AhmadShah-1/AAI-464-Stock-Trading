from main.data_loader import get_top_50_us_stocks, get_price_history
from main.rolling_backtest_karina import rolling_backtest
from main.performance_karina import (
    cumulative_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown
)


symbols = get_top_50_us_stocks()
prices = get_price_history(symbols)

port_rets, rebalance_dates, weights_history, diagnostics = rolling_backtest(
    prices,
    train_window=252,
    test_window=21
)

cum = cumulative_returns(port_rets)

print("OUT-OF-SAMPLE RESULTS")
print("Annualized Return:", annualized_return(port_rets))
print("Annualized Volatility:", annualized_volatility(port_rets))
print("Sharpe Ratio:", sharpe_ratio(port_rets))
print("Max Drawdown:", max_drawdown(cum))

print("\nFirst 5 cumulative values:")
print(cum.head())

print("\nLast 5 cumulative values:")
print(cum.tail())

print("\nNumber of Rebalances:", len(rebalance_dates))

print("\n==== DIAGNOSTICS ====\n")
print("Avg Turnover:", diagnostics["avg_turnover"])
print("Max Turnover:", diagnostics["max_turnover"])
print("Avg Weight Instability:", diagnostics["avg_weight_instability"])
print("Top-5 Risk Concentration:", diagnostics["risk_concentration_top_5"])
print("Top-5 Weight Concentration:", diagnostics["weight_concentration_top_5"])

print("\nTop Asset Return Contributors:")
print(diagnostics["asset_return_contribution"].head(10))

print("\nReturn Distribution Stats:")
for k, v in diagnostics["distribution_stats"].items():
    print(f"{k}: {v}")

import matplotlib.pyplot as plt

INITIAL_CAPITAL = 1_000_000

# Convert returns to portfolio value
portfolio_value = INITIAL_CAPITAL * (1 + port_rets).cumprod()

# Plot
plt.figure()
portfolio_value.plot()
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()

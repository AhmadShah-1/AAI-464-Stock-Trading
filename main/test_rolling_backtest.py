from data_loader import get_top_50_us_stocks, get_price_history
from rolling_backtest_karina import rolling_backtest
from performance_karina import (
    cumulative_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown
)


symbols = get_top_50_us_stocks()
prices = get_price_history(symbols)

port_rets, rebalance_dates = rolling_backtest(
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

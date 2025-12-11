import pandas as pd
import alpaca_trade_api as tradeapi
from main.config_karina import Config


# ================================
# Alpaca Connection
# ================================

api = tradeapi.REST(
    Config.ALPACA_API_KEY,
    Config.ALPACA_SECRET_KEY,
    Config.ALPACA_BASE_URL,
    api_version="v2"
)


# ================================
# FIXED STOCK UNIVERSE (ROBUST)
# ================================

STATIC_UNIVERSE_40 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC",
    "XOM", "CVX",
    "JNJ", "PFE", "UNH",
    "PG", "KO", "PEP",
    "DIS", "NFLX",
    "CSCO", "INTC", "AMD",
    "V", "MA",
    "WMT", "MCD", "COST",
    "ABBV", "MRK",
    "T", "VZ",
    "ADBE", "CRM",
    "CAT", "BA",
    "IBM", "GE",
    "GS", "MS"
]


def get_top_50_us_stocks():
    """
    Returns a fixed, survivorship-bias-safe universe of large-cap stocks.
    """
    return STATIC_UNIVERSE_40.copy()


# ================================
# PRICE HISTORY LOADER
# ================================

def get_price_history(symbols):
    """
    Pulls daily adjusted close prices using a rolling window defined
    by Config.LOOKBACK_DAYS.
    """

    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=Config.LOOKBACK_DAYS + 10)

    bars = api.get_bars(
        symbols,
        tradeapi.TimeFrame.Day,
        start=start.isoformat(),
        end=end.isoformat(),
        adjustment="all",
        feed="iex"
    ).df

    if bars.empty:
        raise RuntimeError("No price data returned from Alpaca.")

    # Pivot into price matrix
    prices = bars.pivot_table(
        values="close",
        index="timestamp",
        columns="symbol"
    )

    prices = prices.sort_index()

    # Only drop rows where ALL prices are missing
    prices = prices.dropna(how="all")

    # Keep rolling window
    prices = prices.tail(Config.LOOKBACK_DAYS)

    return prices

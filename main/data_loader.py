import pandas as pd
import alpaca_trade_api as tradeapi
from main.config_karina import Config

# Alpaca Connection


api = tradeapi.REST(
    Config.ALPACA_API_KEY,
    Config.ALPACA_SECRET_KEY,
    Config.ALPACA_BASE_URL,
    api_version="v2"
)


#fixed stock universe


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
    Returns a fixed universe of large-cap stocks

    """
    return STATIC_UNIVERSE_40.copy()


# loading price history
def get_price_history(symbols):
    """
    Pulls daily adjusted close prices for the given symbols using a fixed 
    historical window large enough to support:
      - 2-year training period
      - 1-year+ test period
    """

    # Hard-coded start date for guaranteed coverage
    start = pd.Timestamp("2022-10-01")
    end = pd.Timestamp.utcnow()

    bars = api.get_bars(
        symbols,
        tradeapi.TimeFrame.Day,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        adjustment="all",
        feed="iex"   
    ).df

    if bars.empty:
        raise RuntimeError("No price data returned from Alpaca.")

    # Pivot into wide format
    prices = bars.pivot_table(
        values="close",
        index="timestamp",
        columns="symbol"
    )

    prices = prices.sort_index()
    prices = prices.dropna(axis=1)

    return prices

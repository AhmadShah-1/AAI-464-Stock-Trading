import pandas as pd
import alpaca_trade_api as tradeapi
from config_karina import Config



# Initialize Alpaca connection
api = tradeapi.REST(
    Config.ALPACA_API_KEY,
    Config.ALPACA_SECRET_KEY,
    Config.ALPACA_BASE_URL,
    api_version="v2"
)



def get_top_50_us_stocks():
    """
    Pulls all active tradable US equities and returns 50 symbols.
    This is a placeholder universe definition for the MVP.
    """

    assets = api.list_assets(status="active", asset_class="us_equity")

    symbols = [
        a.symbol for a in assets
        if a.tradable and a.exchange in ["NYSE", "NASDAQ"]
    ]

    symbols = sorted(symbols)
    return symbols[:50]


def get_price_history(symbols):
    """
    Pulls daily adjusted close prices for the given symbols using a
    rolling lookback window defined in config.
    Returns a DataFrame with:
        index = date
        columns = symbols
        values = adjusted close prices
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

    # pivot logic
    if "symbol" in bars.columns:
        prices = bars.pivot_table(
            values="close",
            index="timestamp",
            columns="symbol"
        )
    else:
        # Fallback for very old SDK versions
        prices = bars["close"].unstack(level=0)

    prices = prices.sort_index()
    prices = prices.dropna(axis=1)
    prices = prices.tail(Config.LOOKBACK_DAYS)

    return prices

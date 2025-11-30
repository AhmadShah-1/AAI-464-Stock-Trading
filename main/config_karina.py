import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    # API
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Execution safety
    TRADING_ENABLED = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'

    # Portfolio sizing
    TOTAL_CAPITAL = float(os.getenv('TOTAL_CAPITAL', '1000000'))
    MAX_WEIGHT_PER_STOCK = float(os.getenv('MAX_WEIGHT_PER_STOCK', '0.10'))

    # Data window
    LOOKBACK_DAYS = int(os.getenv('LOOKBACK_DAYS', '504'))  # ~2 years
    TRADING_DAYS_PER_YEAR = 252

    # Rebalancing controls
    REBALANCE_FREQUENCY = os.getenv('REBALANCE_FREQUENCY', 'daily')
    MIN_DOLLAR_TRADE = float(os.getenv('MIN_DOLLAR_TRADE', '500'))
    MIN_WEIGHT_CHANGE = float(os.getenv('MIN_WEIGHT_CHANGE', '0.005'))
    MAX_TURNOVER = float(os.getenv('MAX_TURNOVER', '0.10'))

    @classmethod
    def display(cls):
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Alpaca Base URL: {cls.ALPACA_BASE_URL}")
        print(f"API Key: {cls.ALPACA_API_KEY[:8]}..." if cls.ALPACA_API_KEY else "API Key: Not set")
        print(f"Trading Enabled: {cls.TRADING_ENABLED}")
        print(f"Total Capital: {cls.TOTAL_CAPITAL}")
        print(f"Max Weight Per Stock: {cls.MAX_WEIGHT_PER_STOCK}")
        print(f"Lookback Days: {cls.LOOKBACK_DAYS}")
        print(f"Min Dollar Trade: {cls.MIN_DOLLAR_TRADE}")
        print(f"Max Daily Turnover: {cls.MAX_TURNOVER}")
        print("=" * 60)

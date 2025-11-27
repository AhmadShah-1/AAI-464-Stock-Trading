"""
Configuration management for the stock trading system.
Loads settings from .env file for secure credential management.

# REMEMBER TO DELETE ENV FILE THAT HAS MY CREDENTIALS
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    # Initialize the config
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Trading settings (EDIT THESE FROM CONFIG.PY)
    TRADING_ENABLED = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '1000'))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))

    # Stock settings
    DEFAULT_SYMBOL = os.getenv('DEFAULT_SYMBOL', 'AAPL')
    HISTORICAL_DAYS = int(os.getenv('HISTORICAL_DAYS', '365'))

    @classmethod
    def display(cls):
        """Display current configuration (hiding sensitive data)."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Alpaca Base URL: {cls.ALPACA_BASE_URL}")
        print(f"API Key: {cls.ALPACA_API_KEY[:8]}..." if cls.ALPACA_API_KEY else "API Key: Not set")
        print(f"Trading Enabled: {cls.TRADING_ENABLED}")
        print(f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"Default Symbol: {cls.DEFAULT_SYMBOL}")
        print(f"Historical Days: {cls.HISTORICAL_DAYS}")
        print("=" * 60)
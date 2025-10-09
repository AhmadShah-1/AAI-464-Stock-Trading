"""Configuration management for the stock trading system."""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for managing application settings."""
    
    # Alpaca API Configuration
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Trading Configuration
    TRADING_ENABLED = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '1000'))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
    
    # Model Configuration
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'random_forest')
    PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '5'))
    
    # Tech/AI Stock Watchlist
    TECH_STOCKS_STR = os.getenv('TECH_STOCKS', 'AAPL,MSFT,GOOGL,NVDA,META,AMZN,TSLA,AMD,INTC,ORCL,CRM,ADBE,PLTR,AI')
    TECH_STOCKS: List[str] = [s.strip() for s in TECH_STOCKS_STR.split(',') if s.strip()]
    
    # Reporting
    REPORT_OUTPUT_DIR = os.getenv('REPORT_OUTPUT_DIR', './reports')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.ALPACA_API_KEY or not cls.ALPACA_SECRET_KEY:
            return False
        return True

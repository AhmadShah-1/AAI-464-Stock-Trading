"""
Alpaca API client for fetching stock market data.
Handles connection to Alpaca's paper trading API and retrieves historical OHLCV data.
"""

import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import Config


class AlpacaClient:
    """Client for interacting with Alpaca's stock market data API."""
    
    def __init__(self):
        """Initialize the Alpaca client with API credentials from config."""
        Config.validate()
        self.client = StockHistoricalDataClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY
        )
    
    def fetch_historical_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            days: Number of days of historical data to fetch (default from config)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
            
        Raises:
            Exception: If API request fails
        """
        if days is None:
            days = Config.HISTORICAL_DAYS
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching {days} days of historical data for {symbol}...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        try:
            # Create request for daily bars
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            # Fetch data
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            
            # Reset index to make symbol and timestamp regular columns
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index()
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Successfully fetched {len(df)} trading days of data")
            print(f"Data columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch data from Alpaca API: {str(e)}")
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Latest closing price
        """
        df = self.fetch_historical_data(symbol, days=5)
        return df.iloc[-1]['close']


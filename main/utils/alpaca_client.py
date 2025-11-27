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

    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY,
        )
        
    # Fetch historical data from Alpaca
    def fetch_historical_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """        
        Args:
            symbol: Stock symbol (e.g. 'AAPL')
            days: Number of days of historical data to fetch (default: None for max available)
            
        Returns:
            DataFrame with OHLCV data
        """

        # If days is not provided, use the default number of days from config
        if days is None:
            days = Config.HISTORICAL_DAYS

        # Calculate date range to fetch data for
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"Date range: {start_date.date()} to {end_date.date()} for {symbol}")


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
            raise Exception(f"Error with Alpaca API (Check API Keys and Base URL): {str(e)}")


    # Get the most recent closing price for a symbol
    def get_latest_price(self, symbol: str) -> float:
        """        
        Args:
            symbol: Stock ticker symbol
            
        Returns: Latest closing price
        """
        df = self.fetch_historical_data(symbol, days=5)
        return df.iloc[-1]['close']

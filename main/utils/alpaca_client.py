"""
Alpaca API client for fetching stock market data.
Handles connection to Alpaca's paper trading API and retrieves historical OHLCV data.
"""

import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from config import Config

class AlpacaClient:

    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY,
        )
        self.news_client = NewsClient(
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
            
            # Fetch bars
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Ensure timestamp is datetime and normalized to midnight UTC
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize().dt.tz_convert('UTC')
            
            # Filter for the specific symbol (in case multiple returned)
            df = df[df['symbol'] == symbol]
            
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

    def _calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using a simple dictionary-based approach
        since external NLP libraries are not available.
        """
        if not text:
            return 0.0
            
        text = text.lower()
        
        # Financial sentiment dictionary
        positive_words = {
            'surge', 'jump', 'gain', 'rise', 'rose', 'up', 'beat', 'profit', 'growth', 
            'record', 'strong', 'buy', 'bullish', 'outperform', 'higher', 'positive', 
            'deal', 'agreement', 'launch', 'approval', 'upgrade', 'success', 'rally',
            'soar', 'spike', 'climb', 'recover', 'exceed', 'top', 'win'
        }
        
        negative_words = {
            'fall', 'fell', 'drop', 'loss', 'down', 'miss', 'weak', 'decline', 'sell', 
            'bearish', 'lower', 'negative', 'crash', 'slump', 'lawsuit', 'investigation', 
            'penalty', 'warning', 'cut', 'downgrade', 'fail', 'risk', 'concern', 'fear',
            'uncertainty', 'plunge', 'tumble', 'sink', 'retreat', 'disappoint'
        }
        
        score = 0
        words = text.split()
        
        for word in words:
            # Simple cleaning
            word = word.strip('.,!?()[]"\'')
            if word in positive_words:
                score += 1
            elif word in negative_words:
                score -= 1
                
        # Normalize score between -1 and 1
        if len(words) > 0:
            # A score of 5 is considered very strong sentiment
            normalized_score = max(min(score / 5.0, 1.0), -1.0)
            return normalized_score
        return 0.0

    def fetch_news_sentiment(self, symbol: str, days: int = None) -> pd.DataFrame:
        """
        Fetch news and calculate daily sentiment scores.
        
        Returns:
            DataFrame with columns: ['timestamp', 'news_sentiment', 'news_volume']
            Indexed by timestamp (daily)
        """
        if days is None:
            days = Config.HISTORICAL_DAYS
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            request_params = NewsRequest(
                symbols=symbol,
                start=start_date,
                end=end_date,
                limit=10000  # Max limit to get as much as possible
            )
            
            news_set = self.news_client.get_news(request_params)
            
            # Access the actual list of news items
            # Based on debug, news_set.data['news'] contains the list of dictionaries
            if hasattr(news_set, 'data') and 'news' in news_set.data:
                news_items = news_set.data['news']
            else:
                # Fallback if structure is different (e.g. empty)
                news_items = []
            
            if not news_items:
                return pd.DataFrame(columns=['timestamp', 'news_sentiment', 'news_volume'])
                
            # Process news
            news_data = []
            for article in news_items:
                # Calculate sentiment from headline and summary
                # News object attributes are accessed via dot notation
                headline = getattr(article, 'headline', '')
                summary = getattr(article, 'summary', '')
                full_text = f"{headline} {summary}"
                
                sentiment_score = self._calculate_sentiment(full_text)
                
                # Use created_at for timestamp
                timestamp = getattr(article, 'created_at', None)
                if timestamp:
                    timestamp = pd.to_datetime(timestamp).normalize().tz_convert('UTC')
                
                    news_data.append({
                        'timestamp': timestamp,
                        'sentiment': sentiment_score
                    })
                
            news_df = pd.DataFrame(news_data)
            
            if news_df.empty:
                 return pd.DataFrame(columns=['timestamp', 'news_sentiment', 'news_volume'])

            # Aggregate by day
            daily_sentiment = news_df.groupby('timestamp').agg(
                news_sentiment=('sentiment', 'mean'),
                news_volume=('sentiment', 'count')
            ).reset_index()
            
            return daily_sentiment
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            # Return empty DataFrame on error to avoid breaking the pipeline
            return pd.DataFrame(columns=['timestamp', 'news_sentiment', 'news_volume'])

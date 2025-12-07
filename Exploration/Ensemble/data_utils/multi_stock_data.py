"""
Multi-stock data fetching and aggregation utilities.
Handles fetching data from multiple stocks and combining them for training.
"""

import pandas as pd
import numpy as np
from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../main')))
from utils.alpaca_client import AlpacaClient


def fetch_multi_stock_data(symbols: List[str], days: int = None) -> pd.DataFrame:
    """
    Fetch historical data for multiple stocks and combine into a single DataFrame.

    Args:
        symbols: List of stock symbols (e.g., ['BAC', 'JPM', 'WFC'])
        days: Number of days of historical data per stock (default: from Config)

    Returns:
        Combined DataFrame with data from all stocks
    """
    client = AlpacaClient()
    market_symbols = ['SPY', 'VXX', 'XLF']
    market_dfs = {}

    print(f"\n{'='*70}")
    print("FETCHING MULTI-STOCK DATA (WITH MARKET CONTEXT)")
    print(f"{'='*70}")
    print(f"Target Symbols: {', '.join(symbols)}")
    print(f"Market Indicators: {', '.join(market_symbols)}")
    print(f"Days per stock: {days if days else 'default'}\n")

    print("Fetching Market Indicators...")
    for msym in market_symbols:
        try:
            print(f"  Fetching {msym}...")
            df = client.fetch_historical_data(msym, days=days)
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
                
            market_dfs[msym] = df[['timestamp', 'close', 'volume', 'high', 'low']]
        except Exception as e:
            print(f"  ✗ Failed to fetch {msym}: {e}")

    all_data = []
    print("\nFetching Target Stocks...")

    for symbol in symbols:
        try:
            df = client.fetch_historical_data(symbol, days=days)

            df['symbol'] = symbol
            
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            
            print(f"    Fetching news for {symbol}...\n") # Newline for cleaner output
            news_df = client.fetch_news_sentiment(symbol, days=days)
            
            if not news_df.empty:
                if news_df['timestamp'].dt.tz is None:
                    news_df['timestamp'] = news_df['timestamp'].dt.tz_localize('UTC')
                
                df = pd.merge(df, news_df, on='timestamp', how='left')
                
                df['news_sentiment'] = df['news_sentiment'].fillna(0)
                df['news_volume'] = df['news_volume'].fillna(0)
                
                print(f"    ✓ Added news data: {len(news_df)} days with news")
            else:
                df['news_sentiment'] = 0
                df['news_volume'] = 0
                print(f"    - No news data found")

            for msym, mdf in market_dfs.items():
                suffix = f"_{msym}"
                mdf_renamed = mdf.rename(columns={
                    'close': f'close{suffix}',
                    'volume': f'volume{suffix}',
                    'high': f'high{suffix}',
                    'low': f'low{suffix}'
                })
                
                df = pd.merge(df, mdf_renamed, on='timestamp', how='left')
                
                cols_to_fill = [c for c in mdf_renamed.columns if c != 'timestamp']
                df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill')

            all_data.append(df)

            print(f"  ✓ {symbol}: {len(df)} trading days + Market Context")

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {str(e)}")
            continue

    if not all_data:
        raise ValueError("Failed to fetch data for any stocks")

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"  Stocks: {combined_df['symbol'].nunique()}")
    print(f"{'='*70}\n")

    return combined_df


def split_by_stock(df: pd.DataFrame, train_symbols: List[str], test_symbols: List[str]) -> tuple:
    """
    Split combined dataframe into train and test sets by stock symbol.

    Args:
        df: Combined DataFrame with 'symbol' column
        train_symbols: List of symbols for training (e.g., ['BAC', 'JPM', 'WFC'])
        test_symbols: List of symbols for testing (e.g., ['C'])

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df[df['symbol'].isin(train_symbols)].reset_index(drop=True)
    test_df = df[df['symbol'].isin(test_symbols)].reset_index(drop=True)

    print(f"\n{'='*70}")
    print("TRAIN/TEST SPLIT BY STOCK")
    print(f"{'='*70}")
    print(f"Training stocks: {', '.join(train_symbols)}")
    print(f"  Samples: {len(train_df)}")
    print(f"  Symbols distribution:")
    for symbol in train_symbols:
        count = len(train_df[train_df['symbol'] == symbol])
        print(f"    {symbol}: {count} samples")

    print(f"\nTesting stocks: {', '.join(test_symbols)}")
    print(f"  Samples: {len(test_df)}")
    print(f"  Symbols distribution:")
    for symbol in test_symbols:
        count = len(test_df[test_df['symbol'] == symbol])
        print(f"    {symbol}: {count} samples")

    print(f"{'='*70}\n")

    return train_df, test_df

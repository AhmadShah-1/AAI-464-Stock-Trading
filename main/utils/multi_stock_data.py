"""
Multi-stock data fetching and aggregation utilities.
Handles fetching data from multiple stocks and combining them for training.
"""

import pandas as pd
import numpy as np
from typing import List
from .alpaca_client import AlpacaClient


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
    all_data = []

    print(f"\n{'='*70}")
    print("FETCHING MULTI-STOCK DATA")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Days per stock: {days if days else 'default'}")
    print()

    for symbol in symbols:
        try:
            # Fetch data for this symbol
            df = client.fetch_historical_data(symbol, days=days)

            # Add symbol identifier (already exists but ensure it's set)
            df['symbol'] = symbol

            all_data.append(df)

            print(f"  ✓ {symbol}: {len(df)} trading days")

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {str(e)}")
            continue

    if not all_data:
        raise ValueError("Failed to fetch data for any stocks")

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp to maintain temporal order
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
    # Filter by symbols
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

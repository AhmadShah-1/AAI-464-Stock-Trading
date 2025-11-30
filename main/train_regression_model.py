"""
Main training script for LightGBM regression model.

Train on: BAC, JPM, WFC (banking stocks)
Test on: C (Citigroup - unseen stock)

Features: 15 technical indicators
Target: 5-day forward returns (continuous)
"""

import sys
import os
import pandas as pd
import numpy as np

# Set Alpaca base URL environment variable
from config import Config
os.environ['APCA_API_BASE_URL'] = Config.ALPACA_BASE_URL

# Import utilities
from utils.multi_stock_data import fetch_multi_stock_data, split_by_stock
from utils.feature_engineering.create_features import create_features
from utils.feature_engineering.create_advanced_features import create_advanced_features
from utils.feature_engineering.create_target_regression import create_target_regression
from models.lightgbm_regressor import LightGBMRegressor


def prepare_features_and_target(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Create features and regression target for a dataframe.

    IMPORTANT: If dataframe contains multiple stocks, features and targets
    are calculated separately for each stock to avoid cross-stock contamination.

    Args:
        df: Raw OHLCV dataframe
        forward_days: Number of days for forward return calculation

    Returns:
        DataFrame with features and target
    """
    # Check if we have multiple stocks
    if 'symbol' in df.columns and df['symbol'].nunique() > 1:
        # Process each stock separately to avoid cross-stock contamination
        all_processed = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()

            # Create basic features for this stock
            symbol_features = create_features(symbol_df)

            # Create advanced features (RSI, MACD, Bollinger Bands, etc.)
            symbol_features = create_advanced_features(symbol_features)

            # Create regression target for this stock
            symbol_with_target = create_target_regression(symbol_features, forward_days=forward_days)

            all_processed.append(symbol_with_target)

        # Combine all stocks back together
        df_with_target = pd.concat(all_processed, ignore_index=True)
    else:
        # Single stock - process normally
        df_features = create_features(df)
        df_features = create_advanced_features(df_features)
        df_with_target = create_target_regression(df_features, forward_days=forward_days)

    return df_with_target


def main():
    """Main training pipeline."""

    # Configuration
    TRAIN_SYMBOLS = ['BAC', 'JPM', 'WFC']  # Training stocks
    TEST_SYMBOLS = ['C']  # Test stock (unseen)
    FORWARD_DAYS = 5  # 5-day forward returns
    THRESHOLD = 0.02  # ±2% for BUY/SELL signals

    print("\n" + "="*70)
    print("LIGHTGBM REGRESSION MODEL TRAINING")
    print("="*70)
    print(f"Training stocks: {', '.join(TRAIN_SYMBOLS)}")
    print(f"Testing stocks: {', '.join(TEST_SYMBOLS)}")
    print(f"Forward horizon: {FORWARD_DAYS} days")
    print(f"Decision threshold: ±{THRESHOLD*100}%")
    print("="*70)

    # Step 1: Fetch multi-stock data
    all_symbols = TRAIN_SYMBOLS + TEST_SYMBOLS
    combined_df = fetch_multi_stock_data(all_symbols, days=365)  # 1 year of data

    # Step 2: Split by stock (train vs test)
    train_df, test_df = split_by_stock(combined_df, TRAIN_SYMBOLS, TEST_SYMBOLS)

    # Step 3: Create features and target for train set
    print("\n" + "="*70)
    print("FEATURE ENGINEERING - TRAINING SET")
    print("="*70)
    train_features = prepare_features_and_target(train_df, forward_days=FORWARD_DAYS)

    # Remove rows with NaN (from rolling windows and forward returns)
    train_features_clean = train_features.dropna()
    print(f"Samples after feature creation: {len(train_features)}")
    print(f"Samples after removing NaN: {len(train_features_clean)}")

    # Feature columns (15 basic + 13 advanced = 28 features)
    feature_cols = [
        # Basic features (15)
        'returns', 'log_returns',
        'volatility_5', 'volatility_10',
        'volume_change', 'volume_ratio',
        'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20',
        'sma_5_10_cross', 'sma_10_20_cross',
        'momentum_5', 'momentum_10',
        'close_to_high', 'close_to_low',
        # Advanced features (13)
        'rsi_14', 'macd', 'macd_signal', 'macd_diff',
        'bb_width', 'bb_position',
        'atr_14', 'stoch_k', 'stoch_d',
        'ema_cross', 'williams_r',
        'ema_12', 'ema_26'
    ]

    X_train = train_features_clean[feature_cols]
    y_train = train_features_clean['target']

    print(f"\nTraining set prepared:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X_train)}")
    print(f"  Target mean: {y_train.mean():.4f}")
    print(f"  Target std: {y_train.std():.4f}")

    # Step 4: Create features and target for test set
    print("\n" + "="*70)
    print("FEATURE ENGINEERING - TESTING SET")
    print("="*70)
    test_features = prepare_features_and_target(test_df, forward_days=FORWARD_DAYS)
    test_features_clean = test_features.dropna()

    print(f"Samples after feature creation: {len(test_features)}")
    print(f"Samples after removing NaN: {len(test_features_clean)}")

    X_test = test_features_clean[feature_cols]
    y_test = test_features_clean['target']

    print(f"\nTesting set prepared:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X_test)}")
    print(f"  Target mean: {y_test.mean():.4f}")
    print(f"  Target std: {y_test.std():.4f}")

    # Step 5: Train LightGBM model
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)

    model = LightGBMRegressor()
    model.train(X_train, y_train)

    # Step 6: Evaluate on test set
    print("\n" + "="*70)
    print("MODEL EVALUATION - TEST SET")
    print("="*70)

    metrics = model.evaluate(X_test, y_test)

    print(f"\nRegression Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"  Test Samples: {metrics['num_samples']}")

    # Step 7: Trading signal evaluation
    print("\n" + "="*70)
    print("TRADING SIGNAL EVALUATION")
    print("="*70)

    # Convert predictions to actions
    actions = model.predict_action(X_test, threshold=THRESHOLD)

    # Convert actual returns to actions
    actual_actions = np.ones(len(y_test), dtype=int)
    actual_actions[y_test > THRESHOLD] = 2  # BUY
    actual_actions[y_test < -THRESHOLD] = 0  # SELL

    # Calculate action accuracy
    action_accuracy = np.mean(actions == actual_actions)

    # Per-action accuracy
    action_counts = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
    action_correct = {'SELL': 0, 'HOLD': 0, 'BUY': 0}

    for actual, predicted in zip(actual_actions, actions):
        actual_name = model.get_action_name(actual)
        predicted_name = model.get_action_name(predicted)

        action_counts[actual_name] += 1
        if actual == predicted:
            action_correct[actual_name] += 1

    print(f"\nAction-based Metrics (Threshold: ±{THRESHOLD*100}%):")
    print(f"  Overall Action Accuracy: {action_accuracy:.2%}")

    for action_name in ['SELL', 'HOLD', 'BUY']:
        if action_counts[action_name] > 0:
            acc = action_correct[action_name] / action_counts[action_name]
            print(f"  {action_name} Accuracy: {acc:.2%} ({action_correct[action_name]}/{action_counts[action_name]})")
        else:
            print(f"  {action_name} Accuracy: N/A (no samples)")

    # Step 8: Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"\n✅ Model: {model.name}")
    print(f"✅ Training stocks: {', '.join(TRAIN_SYMBOLS)} ({len(X_train)} samples)")
    print(f"✅ Testing stock: {', '.join(TEST_SYMBOLS)} ({len(X_test)} samples)")
    print(f"\n📊 Performance Metrics:")
    print(f"  • R² Score: {metrics['r2']:.4f} {'✅ PASS' if metrics['r2'] > 0.5 else '❌ BELOW TARGET'} (target: > 0.5)")
    print(f"  • RMSE: {metrics['rmse']:.4f}")
    print(f"  • Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"  • Action Accuracy: {action_accuracy:.2%}")

    if metrics['r2'] > 0.8:
        print(f"\n🎉 EXCELLENT! R² > 0.80 - Exceeds stretch goal!")
    elif metrics['r2'] > 0.5:
        print(f"\n✅ SUCCESS! R² > 0.50 - Meets MVP criteria!")
    else:
        print(f"\n⚠️  R² < 0.50 - Consider feature engineering or hyperparameter tuning")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()

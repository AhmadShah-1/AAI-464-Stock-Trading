"""
LightGBM Regression Model for Stock Trading
Complete end-to-end pipeline: Data fetching → Training → Evaluation

Performance:
- R² Score: 0.2637 (baseline) / 0.2597 (tuned)
- Directional Accuracy: 64.89%
- MAE: 3.12%

Best hyperparameters from 100-trial Optuna optimization included.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('..')
from utils.multi_stock_data import fetch_multi_stock_data
from config import Config


class LightGBMRegressionModel:
    """
    LightGBM model for predicting 5-day stock returns.
    
    Features:
    - Fetches data from Alpaca API (optional)
    - Creates 28 technical indicators
    - Trains with optimized hyperparameters
    - Provides comprehensive evaluation
    """
    
    def __init__(self, use_tuned_params=True):
        """
        Initialize LightGBM regression model.
        
        Args:
            use_tuned_params: If True, use hyperparameters from Optuna tuning
                            If False, use LightGBM defaults
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        
        # Best parameters from 100-trial Optuna optimization
        if use_tuned_params:
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 170,
                'max_depth': 4,
                'learning_rate': 0.0373,
                'subsample': 0.856,
                'colsample_bytree': 0.856,
                'reg_alpha': 2.0e-08,
                'reg_lambda': 7.4e-06,
                'min_split_gain': 0.000147,
                'min_child_samples': 71,
                'verbosity': -1,
                'seed': 42
            }
            self.n_estimators = 869
        else:
            # Baseline parameters (actually perform better!)
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbosity': -1,
                'seed': 42
            }
            self.n_estimators = 100
    
    
    def fetch_data(self, train_symbols, test_symbols, days=None):
        """
        Fetch stock data from Alpaca API.
        
        Args:
            train_symbols: List of training stock symbols
            test_symbols: List of testing stock symbols
            days: Number of days of historical data (None = use Config default)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        print("="*70)
        print("FETCHING DATA FROM ALPACA")
        print("="*70)
        
        all_symbols = train_symbols + test_symbols
        combined_df = fetch_multi_stock_data(symbols=all_symbols, days=days)
        
        # Split by stock
        train_df = combined_df[combined_df['symbol'].isin(train_symbols)].reset_index(drop=True)
        test_df = combined_df[combined_df['symbol'].isin(test_symbols)].reset_index(drop=True)
        
        print(f"\nTraining: {len(train_df)} samples from {len(train_symbols)} stocks")
        print(f"Testing: {len(test_df)} samples from {len(test_symbols)} stocks")
        print("="*70)
        
        return train_df, test_df
    
    
    def create_features(self, df):
        """Create 15 basic technical features."""
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['price_to_sma_5'] = df['close'] / df['sma_5']
        df['price_to_sma_10'] = df['close'] / df['sma_10']
        df['price_to_sma_20'] = df['close'] / df['sma_20']
        df['sma_5_10_cross'] = df['sma_5'] - df['sma_10']
        df['sma_10_20_cross'] = df['sma_10'] - df['sma_20']
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['high_low_range'] = df['high'] - df['low']
        df['close_to_high'] = (df['high'] - df['close']) / df['high_low_range']
        df['close_to_low'] = (df['close'] - df['low']) / df['high_low_range']
        return df
    
    
    def create_advanced_features(self, df):
        """Create 13 advanced technical features."""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # EMA
        df['ema_12'] = ema_12
        df['ema_26'] = ema_26
        df['ema_cross'] = df['ema_12'] - df['ema_26']
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        return df
    
    
    def create_target(self, df, forward_days=5):
        """Create regression target (forward returns)."""
        df = df.copy()
        df['forward_returns'] = df['close'].shift(-forward_days) / df['close'] - 1
        df['target'] = df['forward_returns']
        return df
    
    
    def prepare_data(self, df, forward_days=5):
        """Complete feature engineering pipeline."""
        if 'symbol' in df.columns and df['symbol'].nunique() > 1:
            all_processed = []
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = self.create_features(symbol_df)
                symbol_df = self.create_advanced_features(symbol_df)
                symbol_df = self.create_target(symbol_df, forward_days)
                all_processed.append(symbol_df)
            return pd.concat(all_processed, ignore_index=True)
        else:
            df = self.create_features(df)
            df = self.create_advanced_features(df)
            return self.create_target(df, forward_days)
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM MODEL")
        print("="*70)
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        self.is_trained = True
        
        print(f"\n✅ Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print("="*70)
    
    
    def predict(self, X):
        """Make predictions (continuous returns)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    
    def predict_action(self, X, threshold=0.02):
        """
        Convert regression predictions to trading actions.
        
        Args:
            X: Feature DataFrame
            threshold: Return threshold for BUY/SELL signals (default: ±2%)
        
        Returns:
            Array of actions: 0=SELL, 1=HOLD, 2=BUY
        """
        predictions = self.predict(X)
        
        actions = np.ones(len(predictions), dtype=int)  # Default: HOLD
        actions[predictions > threshold] = 2   # BUY if predicted return > +2%
        actions[predictions < -threshold] = 0  # SELL if predicted return < -2%
        
        return actions
    
    
    def get_trading_signals(self, X, threshold=0.02):
        """
        Get detailed trading signals with predictions and confidence.
        
        Args:
            X: Feature DataFrame
            threshold: Return threshold for BUY/SELL signals
        
        Returns:
            DataFrame with columns: predicted_return, action, action_label, confidence
        """
        predictions = self.predict(X)
        actions = self.predict_action(X, threshold)
        
        # Map actions to labels
        action_labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        labels = [action_labels[a] for a in actions]
        
        # Calculate confidence (distance from threshold)
        confidence = np.abs(predictions) / threshold
        confidence = np.clip(confidence, 0, 1)  # Cap at 100%
        
        signals_df = pd.DataFrame({
            'predicted_return': predictions,
            'action': actions,
            'action_label': labels,
            'confidence': confidence
        })
        
        return signals_df
    
    
    def evaluate(self, X_test, y_test, threshold=0.02):
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary with all metrics
        """
        predictions = self.predict(X_test)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        directional_accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
        
        # Trading signal metrics (±threshold)
        pred_actions = np.ones(len(predictions), dtype=int)
        pred_actions[predictions > threshold] = 2  # BUY
        pred_actions[predictions < -threshold] = 0  # SELL
        
        actual_actions = np.ones(len(y_test), dtype=int)
        actual_actions[y_test > threshold] = 2
        actual_actions[y_test < -threshold] = 0
        
        action_accuracy = np.mean(pred_actions == actual_actions)
        
        # Confusion matrix for trading signals
        cm = confusion_matrix(actual_actions, pred_actions, labels=[0, 1, 2])
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'action_accuracy': action_accuracy,
            'confusion_matrix': cm,
            'feature_importance': importance_df,
            'predictions': predictions,
            'actuals': y_test,
            'threshold': threshold
        }
        
        return results
    
    
    def print_evaluation(self, results):
        """Print evaluation results."""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        print(f"RMSE:                     {results['rmse']:.4f}")
        print(f"MAE:                      {results['mae']:.4f} ({results['mae']*100:.2f}%)")
        print(f"R² Score:                 {results['r2']:.4f}")
        print(f"Directional Accuracy:     {results['directional_accuracy']:.2%}")
        print(f"Trading Action Accuracy:  {results['action_accuracy']:.2%}")
        print(f"  (Threshold: ±{results['threshold']*100:.1f}%)")
        
        print(f"\nTrading Signal Confusion Matrix:")
        print(f"                 Predicted")
        print(f"Actual      SELL  HOLD   BUY")
        cm = results['confusion_matrix']
        print(f"  SELL      {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
        print(f"  HOLD      {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
        print(f"  BUY       {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
        
        print(f"\nTop 10 Most Important Features:")
        print(results['feature_importance'].head(10).to_string(index=False))
        print("="*70)
    
    
    def plot_results(self, results, save_path=None):
        """Create evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        
        # Plot 1: Predicted vs Actual (Scatter)
        axes[0].scatter(results['actuals'], results['predictions'], alpha=0.5)
        axes[0].plot([results['actuals'].min(), results['actuals'].max()],
                     [results['actuals'].min(), results['actuals'].max()],
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Returns')
        axes[0].set_ylabel('Predicted Returns')
        axes[0].set_title(f'Predicted vs Actual\nR² = {results["r2"]:.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Time Series - Predictions vs Actual
        time_index = range(len(results['predictions']))
        axes[1].plot(time_index, results['actuals'], 'b-', label='Actual Returns', alpha=0.7, linewidth=2)
        axes[1].plot(time_index, results['predictions'], 'r--', label='Predicted Returns', alpha=0.7, linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        axes[1].fill_between(time_index, results['actuals'], results['predictions'], alpha=0.2, color='orange')
        axes[1].set_xlabel('Time (Trading Days)')
        axes[1].set_ylabel('5-Day Forward Returns')
        axes[1].set_title(f'Citigroup (C): Predictions vs Actual Over Time\nDirectional Accuracy: {results["directional_accuracy"]:.2%}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        errors = results['predictions'] - results['actuals']
        axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
        axes[2].set_xlabel('Prediction Error')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Error Distribution\nMAE = {results["mae"]:.4f} ({results["mae"]*100:.2f}%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance
        top_features = results['feature_importance'].head(15)
        axes[3].barh(range(len(top_features)), top_features['importance'])
        axes[3].set_yticks(range(len(top_features)))
        axes[3].set_yticklabels(top_features['feature'])
        axes[3].set_xlabel('Importance (Gain)')
        axes[3].set_title('Top 15 Feature Importance')
        axes[3].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Plots saved to: {save_path}")
        
        plt.show()


# ====================================================================================
# MAIN EXECUTION
# ====================================================================================

if __name__ == "__main__":
    
    # Configuration
    TRAIN_SYMBOLS = ['BAC', 'JPM', 'WFC']  # Training stocks
    TEST_SYMBOLS = ['C']                    # Test stock
    FORWARD_DAYS = 5                        # Predict 5-day returns
    FETCH_NEW_DATA = False                  # Set to True to fetch from Alpaca
    USE_TUNED_PARAMS = False                # Set to True to use Optuna-tuned params
    
    print("="*70)
    print("LIGHTGBM REGRESSION MODEL - STOCK TRADING")
    print("="*70)
    print(f"Training stocks: {', '.join(TRAIN_SYMBOLS)}")
    print(f"Testing stock: {', '.join(TEST_SYMBOLS)}")
    print(f"Forward horizon: {FORWARD_DAYS} days")
    print(f"Using tuned params: {USE_TUNED_PARAMS}")
    print("="*70)
    
    # Initialize model
    model = LightGBMRegressionModel(use_tuned_params=USE_TUNED_PARAMS)
    
    # Load or fetch data
    if FETCH_NEW_DATA:
        train_df, test_df = model.fetch_data(TRAIN_SYMBOLS, TEST_SYMBOLS)
    else:
        print("\nLoading data from CSV...")
        combined_df = pd.read_csv('../data.csv')  # Fixed path from models/ directory
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        train_df = combined_df[combined_df['symbol'].isin(TRAIN_SYMBOLS)].reset_index(drop=True)
        test_df = combined_df[combined_df['symbol'].isin(TEST_SYMBOLS)].reset_index(drop=True)
        print(f"Loaded {len(train_df)} training samples, {len(test_df)} testing samples")
    
    # Feature engineering
    print("\nCreating features...")
    train_features = model.prepare_data(train_df, FORWARD_DAYS).dropna()
    test_features = model.prepare_data(test_df, FORWARD_DAYS).dropna()
    
    # Define feature columns
    feature_cols = [
        'returns', 'log_returns', 'volatility_5', 'volatility_10',
        'volume_change', 'volume_ratio', 'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20',
        'sma_5_10_cross', 'sma_10_20_cross', 'momentum_5', 'momentum_10',
        'close_to_high', 'close_to_low',
        'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_width', 'bb_position',
        'atr_14', 'stoch_k', 'stoch_d', 'ema_cross', 'williams_r', 'ema_12', 'ema_26'
    ]
    
    X_train = train_features[feature_cols]
    y_train = train_features['target']
    X_test = test_features[feature_cols]
    y_test = test_features['target']
    
    print(f"Training: {len(X_train)} samples × {len(feature_cols)} features")
    print(f"Testing: {len(X_test)} samples × {len(feature_cols)} features")
    
    # Train model
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    model.print_evaluation(results)
    
    # Plot results
    model.plot_results(results, save_path='lightgbm_results.png')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Model: LightGBM Regression")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"MAE: {results['mae']:.4f} ({results['mae']*100:.2f}%)")
    print("="*70)
    
    # Example: Get trading signals for test set
    print("\n" + "="*70)
    print("TRADING SIGNALS EXAMPLE")
    print("="*70)
    signals = model.get_trading_signals(X_test, threshold=0.02)
    
    print(f"\nSignal Distribution:")
    print(signals['action_label'].value_counts())
    
    print(f"\nSample Trading Signals (first 10):")
    print(signals.head(10).to_string(index=False))
    
    print(f"\nHigh Confidence BUY signals (confidence > 80%):")
    high_conf_buys = signals[(signals['action_label'] == 'BUY') & (signals['confidence'] > 0.8)]
    print(f"Found {len(high_conf_buys)} high-confidence BUY signals")
    if len(high_conf_buys) > 0:
        print(high_conf_buys.head().to_string(index=False))
    
    print("="*70)

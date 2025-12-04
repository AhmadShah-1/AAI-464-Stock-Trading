"""
CatBoost Regression Model for Stock Trading
Complete end-to-end pipeline: Data fetching → Training → Evaluation

Performance Goal: Compare against LightGBM (R² ~0.64, Acc ~77%)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
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


class CatBoostRegressionModel:
    """
    CatBoost model for predicting 5-day stock returns.
    
    Features:
    - Fetches data from Alpaca API
    - Creates 77+ technical & sentiment features
    - Uses CatBoostRegressor for robust prediction
    """
    
    def __init__(self):
        """Initialize CatBoost regression model."""
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        
        # CatBoost Parameters
        self.params = {
            'loss_function': 'RMSE',
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 50
        }
    
    
    def fetch_data(self, train_symbols, test_symbols, days=None):
        """
        Fetch stock data from Alpaca API.
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
    
    
    def create_comprehensive_features(self, df):
        """Create 39 additional comprehensive features for improved prediction."""
        df = df.copy()
        
        # === 1. PRICE ACTION FEATURES (8) ===
        df['overnight_gap'] = (df['open'] - df['close'].shift()) / df['close'].shift()
        df['gap_filled'] = ((df['high'] >= df['close'].shift()) & (df['low'] <= df['close'].shift())).astype(int)
        df['intraday_range'] = (df['high'] - df['low']) / df['open']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        df['daily_return_volatility'] = df['returns'].rolling(window=10).std()
        
        # === 2. VOLUME FEATURES (6) ===
        price_change = df['close'] - df['close'].shift()
        df['volume_price_trend'] = (price_change / df['close'].shift() * df['volume']).cumsum()
        df['on_balance_volume'] = (np.sign(price_change) * df['volume']).cumsum()
        df['volume_weighted_return'] = df['returns'] * (df['volume'] / df['volume'].rolling(20).mean())
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_momentum'] = df['volume'].pct_change(5)
        df['high_volume_days'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).rolling(10).sum()
        
        # === 3. MOMENTUM & TREND FEATURES (7) ===
        df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = df['atr_14'] * 14  # Approximate true range sum
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx_14'] = dx.rolling(14).mean()
        
        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['cci_20'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
        
        # TRIX (Triple Exponential Moving Average)
        ema1 = df['close'].ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        df['trix'] = ema3.pct_change() * 100
        
        # Ultimate Oscillator (multi-timeframe)
        bp = df['close'] - df[['low', 'close']].shift().min(axis=1)
        tr_uo = df[['high', 'close']].shift().max(axis=1) - df[['low', 'close']].shift().min(axis=1)
        avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum()
        df['ultimate_oscillator'] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
        
        # KST (Know Sure Thing)
        roc1 = df['close'].pct_change(10)
        roc2 = df['close'].pct_change(15)
        roc3 = df['close'].pct_change(20)
        roc4 = df['close'].pct_change(30)
        df['kst'] = (roc1.rolling(10).mean() * 1 + roc2.rolling(10).mean() * 2 + 
                     roc3.rolling(10).mean() * 3 + roc4.rolling(15).mean() * 4)
        
        # === 4. VOLATILITY FEATURES (5) ===
        df['historical_volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Parkinson volatility (high-low range estimator)
        df['parkinson_volatility'] = np.sqrt(1/(4*np.log(2)) * ((np.log(df['high']/df['low']))**2).rolling(20).mean()) * np.sqrt(252)
        
        # Garman-Klass volatility (OHLC estimator)
        log_hl = (np.log(df['high']) - np.log(df['low']))**2
        log_co = (np.log(df['close']) - np.log(df['open']))**2
        df['garman_klass_volatility'] = np.sqrt((0.5 * log_hl - (2*np.log(2)-1) * log_co).rolling(20).mean()) * np.sqrt(252)
        
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_10']
        df['volatility_breakout'] = (df['volatility_10'] > df['volatility_10'].rolling(50).mean() + 2*df['volatility_10'].rolling(50).std()).astype(int)
        
        # === 5. MARKET MICROSTRUCTURE FEATURES (4) ===
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        df['price_efficiency'] = df['close'] / (df['high'] - df['low']).rolling(20).sum()
        df['trade_intensity'] = df['trade_count'] / df['volume']
        df['avg_trade_size'] = df['volume'] / df['trade_count']
        
        # === 6. TIME-BASED FEATURES (5) ===
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['week_of_month'] = (df['timestamp'].dt.day - 1) // 7 + 1
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            # Last 3 trading days of month
            df['is_month_end'] = (df['timestamp'].dt.is_month_end | 
                                  (df['timestamp'] + pd.Timedelta(days=1)).dt.is_month_end |
                                  (df['timestamp'] + pd.Timedelta(days=2)).dt.is_month_end).astype(int)
                                  
        # === 7. NEWS SENTIMENT FEATURES (New) ===
        if 'news_sentiment' in df.columns:
            # Sentiment momentum (change in sentiment)
            df['sentiment_momentum'] = df['news_sentiment'].diff()
            
            # Sentiment moving average
            df['sentiment_ma_5'] = df['news_sentiment'].rolling(5).mean()
            
            # High news volume day
            df['high_news_volume'] = (df['news_volume'] > df['news_volume'].rolling(20).mean() * 1.5).astype(int)
            
            # Sentiment impact (sentiment * volume)
            df['sentiment_impact'] = df['news_sentiment'] * df['news_volume']
        
        return df
    
    
    def create_target(self, df, forward_days=5):
        """Create regression target (forward returns)."""
        df = df.copy()
        df['forward_returns'] = df['close'].shift(-forward_days) / df['close'] - 1
        df['target'] = df['forward_returns']
        return df
    
    
    def prepare_data(self, df, forward_days=5):
        """Complete feature engineering pipeline with comprehensive features."""
        if 'symbol' in df.columns and df['symbol'].nunique() > 1:
            all_processed = []
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = self.create_features(symbol_df)
                symbol_df = self.create_advanced_features(symbol_df)
                symbol_df = self.create_comprehensive_features(symbol_df)
                symbol_df = self.create_target(symbol_df, forward_days)
                all_processed.append(symbol_df)
            return pd.concat(all_processed, ignore_index=True)
        else:
            df = self.create_features(df)
            df = self.create_advanced_features(df)
            df = self.create_comprehensive_features(df)
            return self.create_target(df, forward_days)
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train CatBoost model.
        """
        print("\n" + "="*70)
        print("TRAINING CATBOOST MODEL")
        print("="*70)
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Initialize model
        self.model = CatBoostRegressor(**self.params)
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val) if X_val is not None else None,
            use_best_model=True if X_val is not None else False,
            verbose=100
        )
        
        self.is_trained = True
        
        print(f"\n✅ Training complete!")
        print(f"   Best iteration: {self.model.get_best_iteration()}")
        print("="*70)
    
    
    def predict(self, X):
        """Make predictions (continuous returns)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    
    def predict_action(self, X, threshold=0.02):
        """Convert regression predictions to trading actions."""
        predictions = self.predict(X)
        
        actions = np.ones(len(predictions), dtype=int)  # Default: HOLD
        actions[predictions > threshold] = 2   # BUY if predicted return > +2%
        actions[predictions < -threshold] = 0  # SELL if predicted return < -2%
        
        return actions
    
    
    def get_trading_signals(self, X, threshold=0.02):
        """Get detailed trading signals with predictions and confidence."""
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
        """Comprehensive model evaluation."""
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
            'importance': self.model.get_feature_importance()
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
        axes[3].set_xlabel('Importance')
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
    TRAIN_SYMBOLS = [
        'BAC', 'JPM', 'WFC',       # Original Banks
        'GS', 'MS',                # Investment Banks
        'USB', 'PNC',              # Regional/Diversified Banks
        'AXP', 'COF',              # Credit Services
        'SCHW', 'BLK',             # Asset Management
        'BK', 'STT', 'TFC'         # Custody & Regional Banks
    ]
    TEST_SYMBOLS = ['C']           # Citigroup (remains the test case)
    FORWARD_DAYS = 5                        # Predict 5-day returns
    FETCH_NEW_DATA = True                  # Set to True to fetch from Alpaca
    
    print("="*70)
    print("CATBOOST REGRESSION MODEL - STOCK TRADING")
    print("="*70)
    print(f"Training stocks: {', '.join(TRAIN_SYMBOLS)}")
    print(f"Testing stock: {', '.join(TEST_SYMBOLS)}")
    print(f"Forward horizon: {FORWARD_DAYS} days")
    print("="*70)
    
    # Initialize model
    model = CatBoostRegressionModel()
    
    # Load or fetch data
    if FETCH_NEW_DATA:
        train_df, test_df = model.fetch_data(TRAIN_SYMBOLS, TEST_SYMBOLS)
    else:
        print("\nLoading data from CSV...")
        all_data = pd.read_csv('../data.csv')
        all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
        
        # Separate Market Data
        market_symbols = ['SPY', 'VXX', 'XLF']
        market_df = all_data[all_data['symbol'].isin(market_symbols)].copy()
        stock_df = all_data[~all_data['symbol'].isin(market_symbols)].copy()
        
        # Calculate Stock Returns (needed for relative strength)
        stock_df['returns'] = stock_df.groupby('symbol')['close'].pct_change()
        
        # Calculate Market Returns
        market_df['market_return'] = market_df.groupby('symbol')['close'].pct_change()
        
        # Pivot market data to have columns like 'SPY_return', 'XLF_return'
        market_pivoted = market_df.pivot(index='timestamp', columns='symbol', values='market_return')
        market_pivoted.columns = [f"{col}_return" for col in market_pivoted.columns]
        
        # Merge Market Data with Stocks
        print("Merging market context features...")
        combined_df = stock_df.merge(market_pivoted, on='timestamp', how='left')
        
        # Create Relative Strength Features
        for m_sym in market_symbols:
            if f"{m_sym}_return" in combined_df.columns:
                combined_df[f'rs_{m_sym}'] = combined_df['returns'] - combined_df[f"{m_sym}_return"]
        
        # Split Train/Test
        train_df = combined_df[combined_df['symbol'].isin(TRAIN_SYMBOLS)].reset_index(drop=True)
        test_df = combined_df[combined_df['symbol'].isin(TEST_SYMBOLS)].reset_index(drop=True)
        print(f"Loaded {len(train_df)} training samples, {len(test_df)} testing samples")
    
    # Feature engineering
    print("\nCreating features...")
    train_features = model.prepare_data(train_df, FORWARD_DAYS).dropna()
    test_features = model.prepare_data(test_df, FORWARD_DAYS).dropna()
    
    # Automatically detect all feature columns (exclude target and metadata)
    exclude_cols = ['target', 'forward_returns', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    
    all_feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    print(f"\n✅ Total features created: {len(all_feature_cols)}")
    
    # === CORRELATION-BASED FEATURE SELECTION ===
    print(f"\n🔍 Performing correlation analysis with target...")
    
    # Calculate correlation with target
    correlations = train_features[all_feature_cols + ['target']].corr()['target'].drop('target')
    correlations_abs = correlations.abs().sort_values(ascending=False)
    
    # Select top N features by correlation
    TOP_N_FEATURES = 35  # Keep top 35 most correlated features
    top_features = correlations_abs.head(TOP_N_FEATURES).index.tolist()
    
    # Force include news features if they exist
    news_features = ['news_sentiment', 'news_volume', 'sentiment_momentum', 'sentiment_ma_5', 'high_news_volume', 'sentiment_impact']
    forced_news_features = [f for f in news_features if f in all_feature_cols]
    
    if forced_news_features:
        print(f"\n📰 News Feature Correlations:")
        for f in forced_news_features:
            corr = correlations.get(f, 0)
            print(f"   {f:20s} → {corr:.4f}")
            if f not in top_features:
                top_features.append(f)
                print(f"   (Forced inclusion of {f})")
    
    print(f"\n📊 Top 10 features by correlation with target:")
    for i, (feat, corr) in enumerate(correlations_abs.head(10).items(), 1):
        print(f"   {i:2d}. {feat:30s} → {corr:.4f}")
    
    print(f"\n✅ Selected {len(top_features)} features (including forced news features)")
    print(f"   Correlation range: {correlations_abs.iloc[TOP_N_FEATURES-1]:.4f} to {correlations_abs.iloc[0]:.4f}")
    
    # Use selected features
    feature_cols = top_features
    
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
    model.plot_results(results, save_path='catboost_results.png')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Model: CatBoost Regression")
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

"""
Ensemble Regression Model for Stock Trading
Combines LightGBM and CatBoost predictions for superior performance.

Strategy: Weighted Average Ensemble
Prediction = (w1 * LightGBM) + (w2 * CatBoost)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('..')
from models.lightgbm_regression_model import LightGBMRegressionModel
from models.catboost_regression_model import CatBoostRegressionModel
from config import Config


class EnsembleModel:
    """
    Ensemble model combining LightGBM and CatBoost.
    """
    
    def __init__(self, lgb_weight=0.5, cat_weight=0.5):
        """
        Initialize Ensemble model.
        
        Args:
            lgb_weight: Weight for LightGBM predictions (0.0 to 1.0)
            cat_weight: Weight for CatBoost predictions (0.0 to 1.0)
        """
        self.lgb_weight = lgb_weight
        self.cat_weight = cat_weight
        
        # Initialize sub-models
        self.lgb_model = LightGBMRegressionModel(use_tuned_params=False) # Baseline often generalizes better
        self.cat_model = CatBoostRegressionModel()
        
        self.is_trained = False
        self.feature_columns = []
        
    
    def fetch_data(self, train_symbols, test_symbols, days=None):
        """Fetch data using one of the sub-models (they share the same logic)."""
        return self.lgb_model.fetch_data(train_symbols, test_symbols, days)
    
    
    def prepare_data(self, df, forward_days=5):
        """Prepare data using one of the sub-models."""
        # Both models use the same feature engineering pipeline now
        return self.cat_model.prepare_data(df, forward_days)
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train both sub-models."""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE MODEL")
        print("="*70)
        
        # Train LightGBM
        print("\n>>> Training LightGBM Component...")
        self.lgb_model.train(X_train, y_train, X_val, y_val)
        
        # Train CatBoost
        print("\n>>> Training CatBoost Component...")
        self.cat_model.train(X_train, y_train, X_val, y_val)
        
        self.is_trained = True
        self.feature_columns = X_train.columns.tolist()
        print("\n✅ Ensemble Training Complete!")
        print("="*70)
        
    
    def predict(self, X):
        """Make predictions using weighted average."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get individual predictions
        pred_lgb = self.lgb_model.predict(X)
        pred_cat = self.cat_model.predict(X)
        
        # Weighted average
        final_pred = (pred_lgb * self.lgb_weight) + (pred_cat * self.cat_weight)
        
        return final_pred
    
    
    def predict_action(self, X, threshold=0.02):
        """Convert ensemble predictions to trading actions."""
        predictions = self.predict(X)
        
        actions = np.ones(len(predictions), dtype=int)  # Default: HOLD
        actions[predictions > threshold] = 2   # BUY
        actions[predictions < -threshold] = 0  # SELL
        
        return actions
    
    
    def evaluate(self, X_test, y_test, threshold=0.02):
        """Comprehensive ensemble evaluation."""
        predictions = self.predict(X_test)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        directional_accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
        
        # Trading signal metrics
        pred_actions = self.predict_action(X_test, threshold)
        
        actual_actions = np.ones(len(y_test), dtype=int)
        actual_actions[y_test > threshold] = 2
        actual_actions[y_test < -threshold] = 0
        
        action_accuracy = np.mean(pred_actions == actual_actions)
        
        # Confusion matrix
        cm = confusion_matrix(actual_actions, pred_actions, labels=[0, 1, 2])
        
        # Feature importance (Proxy using LightGBM component)
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.lgb_model.model.feature_importance(importance_type='gain')
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
        print("ENSEMBLE MODEL EVALUATION")
        print("="*70)
        print(f"Weights: LightGBM={self.lgb_weight}, CatBoost={self.cat_weight}")
        print("-" * 30)
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
        print("="*70)
        
    
    def plot_results(self, results, save_path=None):
        """Create evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        
        # Plot 1: Predicted vs Actual (Scatter)
        axes[0].scatter(results['actuals'], results['predictions'], alpha=0.5, color='purple')
        axes[0].plot([results['actuals'].min(), results['actuals'].max()],
                     [results['actuals'].min(), results['actuals'].max()],
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Returns')
        axes[0].set_ylabel('Predicted Returns')
        axes[0].set_title(f'Ensemble: Predicted vs Actual\nR² = {results["r2"]:.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Time Series
        time_index = range(len(results['predictions']))
        axes[1].plot(time_index, results['actuals'], 'b-', label='Actual', alpha=0.6)
        axes[1].plot(time_index, results['predictions'], 'purple', linestyle='--', label='Ensemble Prediction', alpha=0.8)
        axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        axes[1].set_title(f'Predictions vs Actual Over Time\nDirectional Accuracy: {results["directional_accuracy"]:.2%}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        errors = results['predictions'] - results['actuals']
        axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[2].axvline(0, color='red', linestyle='--', lw=2)
        axes[2].set_title(f'Error Distribution\nMAE = {results["mae"]:.4f}')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Empty for now (or could plot model contributions)
        axes[3].text(0.5, 0.5, 'Ensemble Model\n(LightGBM + CatBoost)', 
                     horizontalalignment='center', verticalalignment='center', fontsize=15)
        axes[3].axis('off')
        
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
        'BAC', 'JPM', 'WFC', 'GS', 'MS', 'USB', 'PNC', 'AXP', 'COF', 'SCHW', 'BLK', 'BK', 'STT', 'TFC'
    ]
    TEST_SYMBOLS = ['C']
    FORWARD_DAYS = 5
    FETCH_NEW_DATA = True
    
    # Initialize Ensemble
    # We give slightly more weight to CatBoost because it had better R2 and Action Accuracy
    model = EnsembleModel(lgb_weight=0.4, cat_weight=0.6)
    
    # Load/Fetch Data
    if FETCH_NEW_DATA:
        train_df, test_df = model.fetch_data(TRAIN_SYMBOLS, TEST_SYMBOLS)
    else:
        # Fallback to CSV loading logic if needed (omitted for brevity, same as others)
        pass
        
    # Feature Engineering
    print("\nCreating features...")
    # We can use either model's prepare_data since they are identical now
    train_features = model.prepare_data(train_df, FORWARD_DAYS).dropna()
    test_features = model.prepare_data(test_df, FORWARD_DAYS).dropna()
    
    # Feature Selection (Same logic as before)
    exclude_cols = ['target', 'forward_returns', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    all_feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    # Correlation Analysis
    correlations = train_features[all_feature_cols + ['target']].corr()['target'].drop('target')
    correlations_abs = correlations.abs().sort_values(ascending=False)
    TOP_N_FEATURES = 35
    top_features = correlations_abs.head(TOP_N_FEATURES).index.tolist()
    
    # Force News Features
    news_features = ['news_sentiment', 'news_volume', 'sentiment_momentum', 'sentiment_ma_5', 'high_news_volume', 'sentiment_impact']
    forced_news_features = [f for f in news_features if f in all_feature_cols]
    for f in forced_news_features:
        if f not in top_features:
            top_features.append(f)
            
    feature_cols = top_features
    print(f"\n✅ Selected {len(feature_cols)} features for training")
    
    # Prepare X and y
    X_train = train_features[feature_cols]
    y_train = train_features['target']
    X_test = test_features[feature_cols]
    y_test = test_features['target']
    
    # Train
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    model.print_evaluation(results)
    
    # Plot
    model.plot_results(results, save_path='ensemble_results.png')

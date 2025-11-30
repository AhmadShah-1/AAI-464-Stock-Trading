"""
LightGBM Regression Model for stock price prediction.
Predicts continuous forward returns and converts to trading signals.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from .base_regression_model import BaseRegressionModel


class LightGBMRegressor(BaseRegressionModel):
    """
    LightGBM Gradient Boosting Regressor for stock return prediction.

    Predicts continuous 5-day forward returns, which are then converted to
    BUY/SELL/HOLD signals using threshold-based rules.
    """

    def __init__(self, params: dict = None):
        """
        Initialize LightGBM regressor.

        Args:
            params: LightGBM parameters (optional). If None, uses defaults.
        """
        super().__init__(name="LightGBM Regressor")

        # Default parameters optimized for financial data
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,  # Increased for more complex patterns
            'learning_rate': 0.01,  # Lower learning rate for better generalization
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 10,  # Prevent overfitting
            'lambda_l1': 0.1,  # L1 regularization
            'lambda_l2': 0.1,  # L2 regularization
            'max_depth': 8,  # Limit tree depth
            'verbose': -1,
            'seed': 42
        }

        self.model = None
        self.feature_importance_df = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the LightGBM model.

        Args:
            X: Feature DataFrame
            y: Target Series (continuous forward returns)
        """
        print(f"\nTraining {self.name}...")
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(X.columns)}")

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Remove NaN values from target
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]

        print(f"Samples after removing NaN targets: {len(X_clean)}")

        # Create LightGBM dataset
        train_data = lgb.Dataset(X_clean, label=y_clean)

        # Train model with early stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,  # Increased from 100
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # Calculate feature importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 Most Important Features:")
        print(self.feature_importance_df.head(5).to_string(index=False))

        self.is_trained = True
        print(f"\n{self.name} training complete!")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict forward returns.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predicted forward returns (continuous values)
        """
        self.check_trained()

        # Ensure features are in the same order as training
        X_ordered = X[self.feature_columns]

        # Make predictions
        predictions = self.model.predict(X_ordered, num_iteration=self.model.best_iteration)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance dataframe.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_df is None:
            raise RuntimeError("Model must be trained before accessing feature importance")

        return self.feature_importance_df

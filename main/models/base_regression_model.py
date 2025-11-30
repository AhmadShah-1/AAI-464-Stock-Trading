"""
Base class for regression models.
All regression models should inherit from this class.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseRegressionModel(ABC):
    """
    Abstract base class for trading prediction models using regression.
    All models must implement train(), predict(), and optionally override evaluate().
    """

    def __init__(self, name: str):
        """
        Args:
            name: Name of the model
        """
        self.name = name
        self.is_trained = False
        self.feature_columns = []

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on historical data.

        Args:
            X: Feature DataFrame
            y: Target Series (continuous forward returns)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (continuous forward returns)
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data with regression metrics.

        Args:
            X: Feature DataFrame
            y: True target values (continuous forward returns)

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)

        # Remove NaN values for evaluation
        mask = ~(np.isnan(predictions) | np.isnan(y))
        predictions_clean = predictions[mask]
        y_clean = y[mask]

        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_clean, predictions_clean))
        mae = mean_absolute_error(y_clean, predictions_clean)
        r2 = r2_score(y_clean, predictions_clean)

        # Directional accuracy: % of times we predicted the correct direction
        directional_accuracy = np.mean(np.sign(predictions_clean) == np.sign(y_clean))

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'num_samples': len(y_clean)
        }

    def predict_action(self, X: pd.DataFrame, threshold: float = 0.02) -> np.ndarray:
        """
        Convert regression predictions to trading actions (BUY/HOLD/SELL).

        Args:
            X: Feature DataFrame
            threshold: Return threshold for BUY/SELL signals (default: 0.02 = 2%)

        Returns:
            Array of actions (0=SELL, 1=HOLD, 2=BUY)
        """
        predictions = self.predict(X)

        # Convert continuous predictions to actions
        actions = np.ones(len(predictions), dtype=int)  # Default to HOLD (1)
        actions[predictions > threshold] = 2  # BUY
        actions[predictions < -threshold] = 0  # SELL

        return actions

    def get_action_name(self, action: int) -> str:
        """
        Convert numeric action to action name.

        Args:
            action: Numeric action (0, 1, or 2)

        Returns:
            Action name ('SELL', 'HOLD', or 'BUY')
        """
        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return action_map.get(action, 'UNKNOWN')

    def check_trained(self):
        """Check if model has been trained."""
        if not self.is_trained:
            raise RuntimeError(f"{self.name} must be trained before making predictions")

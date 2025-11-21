"""
Abstract base class for all trading models.
Defines the standard interface that all models must implement.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict


class BaseModel(ABC):
    """
    Abstract base class for trading prediction models.
    All models must implement train(), predict(), get_confidence(), and evaluate().
    """
    
    def __init__(self, name: str):
        """
        Initialize the base model.
        
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
            y: Target Series (0=sell, 1=hold, 2=buy)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (0=sell, 1=hold, 2=buy)
        """
        pass
    
    @abstractmethod
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get confidence scores for predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of confidence scores (0 to 1)
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature DataFrame
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        # Calculate per-class accuracy
        classes = [0, 1, 2]
        class_names = ['sell', 'hold', 'buy']
        class_accuracies = {}
        
        for cls, name in zip(classes, class_names):
            mask = y == cls
            if mask.sum() > 0:
                class_acc = np.mean(predictions[mask] == y[mask])
                class_accuracies[f'{name}_accuracy'] = class_acc
        
        return {
            'overall_accuracy': accuracy,
            'num_samples': len(y),
            **class_accuracies
        }
    
    def get_action_name(self, prediction: int) -> str:
        """
        Convert numeric prediction to action name.
        
        Args:
            prediction: Numeric prediction (0, 1, or 2)
            
        Returns:
            Action name ('SELL', 'HOLD', or 'BUY')
        """
        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return action_map.get(prediction, 'UNKNOWN')
    
    def check_trained(self):
        """Check if model has been trained. Raises error if not."""
        if not self.is_trained:
            raise RuntimeError(f"{self.name} must be trained before making predictions")


'''
This file is used to define the base model class
All models should inherit from this class
'''

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class BaseModel(ABC):
    """
    Abstract base class for trading prediction models.
    All models must implement train(), predict(), get_confidence(), and evaluate().
    """

    # Initialize the base model
    def __init__(self, name: str):
        """
        Args:
            name: Name of the model
        """
        self.name = name
        self.is_trained = False
        self.feature_columns = []

    # Train the model on historical data
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """        
        Args:
            X: Feature DataFrame
            y: Target Series (0=sell, 1=hold, 2=buy)
        """
        pass
    
    # Make predictions on new data
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (0=sell, 1=hold, 2=buy)
        """
        pass
    
    # Get confidence scores for predictions
    @abstractmethod
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of confidence scores (0 to 1)
        """
        pass
    
    # Evaluate the model on test data
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """        
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
    
    
    # Convert numeric prediction to action name
    def get_action_name(self, prediction: int) -> str:
        """        
        Args:
            prediction: Numeric prediction (0, 1, or 2)
            
        Returns: Action name ('SELL', 'HOLD', or 'BUY')
        """
        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return action_map.get(prediction, 'UNKNOWN')    # Return UNKNOWN if prediction is not 0, 1, or 2
    

    # Check if model has been trained
    def check_trained(self):
        if not self.is_trained:
            raise RuntimeError(f"{self.name} must be trained before making predictions")


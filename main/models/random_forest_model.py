"""
Random Forest classification model for stock trading predictions.
Uses ensemble learning with decision trees.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest model for predicting buy/hold/sell actions.
    Uses scikit-learn's RandomForestClassifier with probability estimates.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        super().__init__(name="Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Initialize model and scaler
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Random Forest model on historical data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (0=sell, 1=hold, 2=buy)
        """
        print(f"\nTraining {self.name} model...")
        print(f"Training samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Mark as trained
        self.is_trained = True
        
        # Display feature importances
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head().to_string(index=False))
        print(f"\n{self.name} training complete!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (0=sell, 1=hold, 2=buy)
        """
        self.check_trained()
        
        # Ensure correct feature columns
        X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get confidence scores for predictions.
        Returns the maximum probability across all classes.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of confidence scores (0 to 1)
        """
        self.check_trained()
        
        # Ensure correct feature columns
        X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)
        
        # Return max probability for each sample
        confidence = np.max(probabilities, axis=1)
        
        return confidence
    
    def get_class_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability distribution across all classes.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of shape (n_samples, 3) with probabilities for [sell, hold, buy]
        """
        self.check_trained()
        
        # Ensure correct feature columns
        X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities


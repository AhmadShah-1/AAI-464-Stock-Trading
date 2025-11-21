"""
Technical Indicator-based model for stock trading predictions.
Uses rule-based strategies with moving averages and momentum indicators.
"""

import pandas as pd
import numpy as np
from models.base_model import BaseModel


class TechnicalIndicatorModel(BaseModel):
    """
    Rule-based model using technical indicators for trading decisions.
    
    Strategy:
    - BUY: Fast MA > Slow MA AND positive momentum AND price above SMA_20
    - SELL: Fast MA < Slow MA AND negative momentum AND price below SMA_20
    - HOLD: Otherwise
    """
    
    def __init__(self):
        """Initialize Technical Indicator model."""
        super().__init__(name="Technical Indicator")
        self.rules = {}
        self.statistics = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        'Train' the model by analyzing historical patterns.
        For rule-based models, this calculates statistics for confidence estimation.
        
        Args:
            X: Feature DataFrame
            y: Target Series (0=sell, 1=hold, 2=buy)
        """
        print(f"\nTraining {self.name} model...")
        print(f"Analyzing {len(X)} historical samples...")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Make predictions on training data
        predictions = self._apply_rules(X)
        
        # Calculate accuracy for each predicted class (for confidence estimation)
        for pred_class in [0, 1, 2]:
            mask = predictions == pred_class
            if mask.sum() > 0:
                actual_class = y[mask]
                accuracy = np.mean(actual_class == pred_class)
                self.statistics[pred_class] = accuracy
            else:
                self.statistics[pred_class] = 0.33  # Default to random guess
        
        # Calculate overall accuracy
        overall_accuracy = np.mean(predictions == y)
        self.statistics['overall'] = overall_accuracy
        
        print(f"\nHistorical Pattern Analysis:")
        print(f"  SELL signals accuracy: {self.statistics.get(0, 0):.2%}")
        print(f"  HOLD signals accuracy: {self.statistics.get(1, 0):.2%}")
        print(f"  BUY signals accuracy: {self.statistics.get(2, 0):.2%}")
        print(f"  Overall accuracy: {overall_accuracy:.2%}")
        
        # Mark as trained
        self.is_trained = True
        print(f"\n{self.name} analysis complete!")
    
    def _apply_rules(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply technical indicator rules to generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (0=sell, 1=hold, 2=buy)
        """
        predictions = np.ones(len(X), dtype=int)  # Default to HOLD (1)
        
        # Extract required features
        sma_5_10_cross = X['sma_5_10_cross'].values
        momentum_5 = X['momentum_5'].values
        price_to_sma_20 = X['price_to_sma_20'].values
        returns = X['returns'].values
        
        # BUY signals: Bullish conditions
        # - Fast MA above slow MA (positive crossover)
        # - Positive momentum
        # - Price above 20-day MA
        buy_condition = (
            (sma_5_10_cross > 0) &  # 5-day MA > 10-day MA
            (momentum_5 > 0) &  # Positive momentum
            (price_to_sma_20 > 1.0) &  # Price above 20-day MA
            (returns > 0)  # Positive recent returns
        )
        predictions[buy_condition] = 2
        
        # SELL signals: Bearish conditions
        # - Fast MA below slow MA (negative crossover)
        # - Negative momentum
        # - Price below 20-day MA
        sell_condition = (
            (sma_5_10_cross < 0) &  # 5-day MA < 10-day MA
            (momentum_5 < 0) &  # Negative momentum
            (price_to_sma_20 < 1.0) &  # Price below 20-day MA
            (returns < 0)  # Negative recent returns
        )
        predictions[sell_condition] = 0
        
        return predictions
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using technical indicator rules.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (0=sell, 1=hold, 2=buy)
        """
        self.check_trained()
        
        # Ensure correct feature columns
        X = X[self.feature_columns]
        
        # Apply rules
        predictions = self._apply_rules(X)
        
        return predictions
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get confidence scores for predictions.
        Confidence is based on historical accuracy of each signal type.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of confidence scores (0 to 1)
        """
        self.check_trained()
        
        # Ensure correct feature columns
        X = X[self.feature_columns]
        
        # Get predictions
        predictions = self._apply_rules(X)
        
        # Calculate signal strength (how strongly conditions are met)
        sma_5_10_cross = X['sma_5_10_cross'].values
        momentum_5 = X['momentum_5'].values
        price_to_sma_20 = X['price_to_sma_20'].values
        
        # Normalize signal strength to 0-1 range
        cross_strength = np.abs(sma_5_10_cross) / (np.abs(sma_5_10_cross).max() + 1e-10)
        momentum_strength = np.abs(momentum_5) / (np.abs(momentum_5).max() + 1e-10)
        price_strength = np.abs(price_to_sma_20 - 1.0) / (np.abs(price_to_sma_20 - 1.0).max() + 1e-10)
        
        # Average signal strength
        signal_strength = (cross_strength + momentum_strength + price_strength) / 3
        
        # Combine historical accuracy with signal strength
        confidence = np.zeros(len(predictions))
        for i, pred in enumerate(predictions):
            historical_accuracy = self.statistics.get(pred, 0.33)
            confidence[i] = 0.5 * historical_accuracy + 0.5 * signal_strength[i]
        
        return confidence
    
    def get_signal_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed breakdown of technical signals.
        
        Args:
            X: Feature DataFrame (single row)
            
        Returns:
            DataFrame with signal details
        """
        row = X.iloc[-1] if len(X) > 0 else X.iloc[0]
        
        details = {
            'Indicator': [
                '5-day MA vs 10-day MA',
                'Momentum (5-day)',
                'Price vs 20-day MA',
                'Recent Returns'
            ],
            'Value': [
                f"{row['sma_5_10_cross']:.2f}",
                f"{row['momentum_5']:.2f}",
                f"{row['price_to_sma_20']:.4f}",
                f"{row['returns']:.4f}"
            ],
            'Signal': [
                'Bullish' if row['sma_5_10_cross'] > 0 else 'Bearish',
                'Positive' if row['momentum_5'] > 0 else 'Negative',
                'Above' if row['price_to_sma_20'] > 1.0 else 'Below',
                'Positive' if row['returns'] > 0 else 'Negative'
            ]
        }
        
        return pd.DataFrame(details)


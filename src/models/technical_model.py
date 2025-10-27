"""Technical indicator-based trading model."""

import pandas as pd
import numpy as np
from typing import Dict

from src.models.base_model import BaseModel


class TechnicalIndicatorModel(BaseModel):
    """Rule-based model using technical indicators."""
    
    def __init__(self):
        """Initialize technical indicator model."""
        super().__init__('technical_indicators')
        self.rules = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0,
            'bb_lower_threshold': 0.02,
            'bb_upper_threshold': 0.02
        }
        self.is_trained = True  # Rule-based, no training needed
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No training needed for rule-based model."""
        self.feature_columns = X.columns.tolist()
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions based on technical indicators.
        
        Returns:
            0: sell, 1: hold, 2: buy
        """
        predictions = []
        
        for idx, row in X.iterrows():
            score = 0
            
            # RSI signals
            if 'rsi' in row.index:
                rsi = row['rsi']
                if rsi < self.rules['rsi_oversold']:
                    score += 2  # Strong buy signal
                elif rsi > self.rules['rsi_overbought']:
                    score -= 2  # Strong sell signal
            
            # MACD signals
            if 'macd_diff' in row.index:
                macd_diff = row['macd_diff']
                if macd_diff > self.rules['macd_threshold']:
                    score += 1  # Buy signal
                else:
                    score -= 1  # Sell signal
            
            # Bollinger Bands signals
            if all(col in row.index for col in ['close', 'bb_low', 'bb_high']):
                close = row['close']
                bb_low = row['bb_low']
                bb_high = row['bb_high']
                bb_range = bb_high - bb_low
                
                # Price near lower band (oversold)
                if (close - bb_low) / bb_range < self.rules['bb_lower_threshold']:
                    score += 1
                # Price near upper band (overbought)
                elif (bb_high - close) / bb_range < self.rules['bb_upper_threshold']:
                    score -= 1
            
            # Moving average crossover
            if all(col in row.index for col in ['sma_20', 'sma_50']):
                if row['sma_20'] > row['sma_50']:
                    score += 1  # Bullish
                else:
                    score -= 1  # Bearish
            
            # Stochastic signals
            if 'stoch_k' in row.index:
                stoch = row['stoch_k']
                if stoch < 20:
                    score += 1  # Oversold
                elif stoch > 80:
                    score -= 1  # Overbought
            
            # Convert score to action
            if score >= 2:
                predictions.append(2)  # Buy
            elif score <= -2:
                predictions.append(0)  # Sell
            else:
                predictions.append(1)  # Hold
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return pseudo-probabilities based on signal strength.
        """
        predictions = self.predict(X)
        n_samples = len(predictions)
        n_classes = 3
        
        proba = np.zeros((n_samples, n_classes))
        
        for i, pred in enumerate(predictions):
            # Strong signal: high confidence
            if pred == 0:  # Sell
                proba[i] = [0.7, 0.2, 0.1]
            elif pred == 2:  # Buy
                proba[i] = [0.1, 0.2, 0.7]
            else:  # Hold
                proba[i] = [0.2, 0.6, 0.2]
        
        return proba
    
    def get_signal_strength(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength for each prediction.
        
        Returns:
            Series with signal strength (-1 to 1)
        """
        strengths = []
        
        for idx, row in X.iterrows():
            score = 0
            max_score = 0
            
            # Calculate normalized score
            if 'rsi' in row.index:
                max_score += 2
                rsi = row['rsi']
                if rsi < self.rules['rsi_oversold']:
                    score += 2
                elif rsi > self.rules['rsi_overbought']:
                    score -= 2
            
            if 'macd_diff' in row.index:
                max_score += 1
                if row['macd_diff'] > self.rules['macd_threshold']:
                    score += 1
                else:
                    score -= 1
            
            if all(col in row.index for col in ['sma_20', 'sma_50']):
                max_score += 1
                if row['sma_20'] > row['sma_50']:
                    score += 1
                else:
                    score -= 1
            
            if 'stoch_k' in row.index:
                max_score += 1
                stoch = row['stoch_k']
                if stoch < 20:
                    score += 1
                elif stoch > 80:
                    score -= 1
            
            # Normalize to -1 to 1
            if max_score > 0:
                strength = score / max_score
            else:
                strength = 0
            
            strengths.append(strength)
        
        return pd.Series(strengths, index=X.index)

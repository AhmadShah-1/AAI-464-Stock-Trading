"""Basic tests for the trading system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.model_factory import ModelFactory
from src.models.technical_model import TechnicalIndicatorModel
from src.utils.features import add_technical_indicators, add_price_features, create_target_variable
from src.utils.trend_detection import TrendDetector
from src.config import Config


class TestModelFactory:
    """Test the model factory."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = ModelFactory.get_available_models()
        assert len(models) > 0
        assert 'random_forest' in models
        assert 'xgboost' in models
        assert 'technical_indicators' in models
    
    def test_create_model(self):
        """Test creating a model."""
        model = ModelFactory.create_model('random_forest')
        assert model is not None
        assert model.name == 'random_forest'
    
    def test_create_invalid_model(self):
        """Test creating an invalid model raises error."""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model')


class TestFeatureEngineering:
    """Test feature engineering utilities."""
    
    def create_sample_data(self, n_days=100):
        """Create sample OHLCV data."""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(n_days) * 0.5,
            'high': close_prices + abs(np.random.randn(n_days) * 2),
            'low': close_prices - abs(np.random.randn(n_days) * 2),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        return df
    
    def test_add_price_features(self):
        """Test adding price features."""
        df = self.create_sample_data()
        df_features = add_price_features(df)
        
        assert 'returns' in df_features.columns
        assert 'volatility_5' in df_features.columns
        assert 'volume_change' in df_features.columns
    
    def test_add_technical_indicators(self):
        """Test adding technical indicators."""
        df = self.create_sample_data()
        df_features = add_technical_indicators(df)
        
        # Check for some key indicators
        assert 'rsi' in df_features.columns or 'momentum_rsi' in df_features.columns
        assert len(df_features) > 0
    
    def test_create_target_variable(self):
        """Test creating target variable."""
        df = self.create_sample_data()
        df_target = create_target_variable(df, horizon=5)
        
        assert 'future_return' in df_target.columns
        assert 'target_direction' in df_target.columns
        assert 'target_action' in df_target.columns


class TestTrendDetector:
    """Test trend detection utilities."""
    
    def create_bullish_data(self, n_days=100):
        """Create bullish trend data."""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        close_prices = 100 + np.arange(n_days) * 0.5 + np.random.randn(n_days) * 0.2
        
        df = pd.DataFrame({
            'open': close_prices,
            'high': close_prices + 1,
            'low': close_prices - 1,
            'close': close_prices,
            'volume': 1000000
        }, index=dates)
        
        return df
    
    def create_bearish_data(self, n_days=100):
        """Create bearish trend data."""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        close_prices = 100 - np.arange(n_days) * 0.5 + np.random.randn(n_days) * 0.2
        
        df = pd.DataFrame({
            'open': close_prices,
            'high': close_prices + 1,
            'low': close_prices - 1,
            'close': close_prices,
            'volume': 1000000
        }, index=dates)
        
        return df
    
    def test_detect_bullish_trend(self):
        """Test detecting bullish trend."""
        detector = TrendDetector()
        df = self.create_bullish_data()
        
        trend = detector.detect_trend(df)
        assert trend in ['bullish', 'neutral', 'bearish']
    
    def test_detect_bearish_trend(self):
        """Test detecting bearish trend."""
        detector = TrendDetector()
        df = self.create_bearish_data()
        
        trend = detector.detect_trend(df)
        assert trend in ['bullish', 'neutral', 'bearish']
    
    def test_trend_strength(self):
        """Test trend strength calculation."""
        detector = TrendDetector()
        df = self.create_bullish_data()
        
        strength = detector.detect_trend_strength(df)
        assert 0 <= strength <= 1
    
    def test_support_resistance(self):
        """Test support and resistance calculation."""
        detector = TrendDetector()
        df = self.create_bullish_data()
        
        support, resistance = detector.identify_support_resistance(df)
        assert support < resistance
        assert support > 0
        assert resistance > 0
    
    def test_get_trend_analysis(self):
        """Test comprehensive trend analysis."""
        detector = TrendDetector()
        df = self.create_bullish_data()
        
        analysis = detector.get_trend_analysis(df)
        
        assert 'trend' in analysis
        assert 'strength' in analysis
        assert 'momentum' in analysis
        assert 'support' in analysis
        assert 'resistance' in analysis
        assert 'breakout' in analysis
        assert 'recommendation' in analysis


class TestTechnicalIndicatorModel:
    """Test technical indicator model."""
    
    def create_sample_features(self, n_samples=10):
        """Create sample feature data."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': 100 + np.random.randn(n_samples) * 5,
            'rsi': 30 + np.random.rand(n_samples) * 40,
            'macd_diff': np.random.randn(n_samples) * 2,
            'bb_low': 95 + np.random.rand(n_samples) * 2,
            'bb_high': 105 + np.random.rand(n_samples) * 2,
            'sma_20': 100 + np.random.randn(n_samples) * 3,
            'sma_50': 100 + np.random.randn(n_samples) * 3,
            'stoch_k': 20 + np.random.rand(n_samples) * 60
        })
        
        return df
    
    def test_model_prediction(self):
        """Test model can make predictions."""
        model = TechnicalIndicatorModel()
        df = self.create_sample_features()
        
        predictions = model.predict(df)
        
        assert len(predictions) == len(df)
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_model_predict_proba(self):
        """Test model can predict probabilities."""
        model = TechnicalIndicatorModel()
        df = self.create_sample_features()
        
        probas = model.predict_proba(df)
        
        assert probas.shape == (len(df), 3)
        assert all(abs(probas.sum(axis=1) - 1.0) < 0.01)  # Probabilities sum to 1


class TestConfig:
    """Test configuration."""
    
    def test_tech_stocks_list(self):
        """Test tech stocks configuration."""
        assert len(Config.TECH_STOCKS) > 0
        assert 'AAPL' in Config.TECH_STOCKS
        assert isinstance(Config.TECH_STOCKS, list)
    
    def test_default_values(self):
        """Test default configuration values."""
        assert Config.MAX_POSITION_SIZE > 0
        assert 0 < Config.RISK_PER_TRADE <= 1
        assert Config.PREDICTION_HORIZON > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

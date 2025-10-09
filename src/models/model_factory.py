"""Model factory for easy switching between prediction models."""

from typing import Dict, Type
from src.models.base_model import BaseModel
from src.models.ml_models import RandomForestModel, XGBoostModel, GradientBoostingModel
from src.models.technical_model import TechnicalIndicatorModel


class ModelFactory:
    """Factory for creating prediction models."""
    
    _models: Dict[str, Type[BaseModel]] = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'gradient_boosting': GradientBoostingModel,
        'technical_indicators': TechnicalIndicatorModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters
        
        Returns:
            Model instance
        
        Raises:
            ValueError: If model name is not recognized
        """
        if model_name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
        
        model_class = cls._models[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Model name
            model_class: Model class
        """
        cls._models[name] = model_class

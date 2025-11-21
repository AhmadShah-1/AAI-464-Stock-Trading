"""
Factory pattern for creating trading models.
Enables seamless switching between different model implementations.
"""

from models.base_model import BaseModel
from models.random_forest_model import RandomForestModel
from models.technical_indicator_model import TechnicalIndicatorModel


class ModelFactory:
    """Factory class for creating trading model instances."""
    
    # Registry of available models
    _models = {
        'random_forest': RandomForestModel,
        'rf': RandomForestModel,  # Shorthand alias
        'technical_indicator': TechnicalIndicatorModel,
        'ti': TechnicalIndicatorModel,  # Shorthand alias
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create ('random_forest' or 'technical_indicator')
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model_type is not recognized
        """
        model_type = model_type.lower().strip()
        
        if model_type not in cls._models:
            available = ', '.join(set(cls._models.keys()))
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types.
        
        Returns:
            List of model type strings
        """
        # Return only primary names (not aliases)
        return ['random_forest', 'technical_indicator']
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """
        Register a new model type (for extensibility).
        
        Args:
            name: Name to register the model under
            model_class: Model class (must inherit from BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must inherit from BaseModel")
        
        cls._models[name.lower()] = model_class
        print(f"Registered new model type: {name}")


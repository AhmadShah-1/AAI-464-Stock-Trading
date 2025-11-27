"""
Models package for stock trading prediction.

Put your models here, otherwise it won't show up in the menu
"""

from models.base_model import BaseModel
from models.random_forest_model import RandomForestModel

__all__ = [
    'BaseModel',
    'RandomForestModel',
]


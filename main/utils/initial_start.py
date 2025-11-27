'''
This file is used to display the header and the menu of the program
'''

import sys
import pandas as pd
import numpy as np
from datetime import datetime

from config import Config
from utils.alpaca_client import AlpacaClient
from utils.feature_engineering import FeatureEngineer
from utils.visualizer import TradingVisualizer
import models

def print_header(title: str):
    # Print a formatted section header
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# Prompt user to select Model
def select_model():
    print_header("MODEL SELECTION")

    # Get all available models from models/__init__.py
    model_names = models.__all__
    
    # Filter out BaseModel (it's abstract) and get actual model classes
    available_models = []
    for name in model_names:
        if name != 'BaseModel':  # Skip abstract base class
            model_class = getattr(models, name)
            available_models.append(model_class)

    # Display available models
    print("\nAvailable Models:")
    for index, model_class in enumerate(available_models):
        # Try to get a description from the model's docstring or name
        model_instance = model_class()
        print(f'  {index} : {model_instance.name}')
    
    while True:
        choice = input("\nEnter your choice: ").strip()
        
        if choice.isdigit() and int(choice) < len(available_models):
            return available_models[int(choice)]
        else:
            print("Invalid choice. Please enter a valid choice.")


# Display Prediction Results
def display_prediction_results(model, X_latest, symbol: str, latest_price: float):
    """
    Args:
        model: Trained model instance
        X_latest: Features for the latest data point
        symbol: Stock symbol
        latest_price: Current stock price
    """

    print_header("PREDICTION RESULTS")

    # Make prediction
    prediction = model.predict(X_latest)[0]  # Returns a list, so we take the first element
    confidence = model.get_confidence(X_latest)[0]
    action = model.get_action_name(prediction)
    
    # Display results
    print(f"\nStock: {symbol}")
    print(f"Current Price: ${latest_price:.2f}")
    print(f"Model: {model.name}")
    print(f"\nRECOMMENDATION: {action}")
    print(f"Confidence: {confidence:.2%}")
    
    # Display confidence interpretation
    if confidence >= Config.CONFIDENCE_THRESHOLD:
        confidence_level = "HIGH"
    elif confidence >= 0.4:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    print(f"Confidence Level: {confidence_level}")
    
    # Display additional details for Random Forest
    if hasattr(model, 'get_class_probabilities'):
        print("\nClass Probabilities:")
        probs = model.get_class_probabilities(X_latest)[0]
        print(f"  SELL: {probs[0]:.2%}")
        print(f"  HOLD: {probs[1]:.2%}")
        print(f"  BUY:  {probs[2]:.2%}")
    
    # Display technical signals for Technical Indicator model
    if hasattr(model, 'get_signal_details'):
        print("\nTechnical Signals:")
        signals = model.get_signal_details(X_latest)
        print(signals.to_string(index=False))
    
    # Trading decision
    print("\n" + "-" * 70)
    if confidence >= Config.CONFIDENCE_THRESHOLD:
        print(f"✓ Trading signal: {action} (Confidence meets threshold)")
    else:
        print(f"✗ Trading signal: HOLD (Confidence below threshold of {Config.CONFIDENCE_THRESHOLD:.0%})")
    print("-" * 70)
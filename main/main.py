"""
Main entry point for the stock trading prediction system.
Orchestrates data fetching, model training, and prediction generation.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import project modules
from config import Config
from utils.alpaca_client import AlpacaClient
from utils.feature_engineering import FeatureEngineer
from utils.model_factory import ModelFactory
from utils.visualizer import TradingVisualizer


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def select_model() -> str:
    """
    Prompt user to select a model type.
    
    Returns:
        Selected model type string
    """
    print_header("MODEL SELECTION")
    print("\nAvailable Models:")
    print("  1. Random Forest (ML-based ensemble method)")
    print("  2. Technical Indicator (Rule-based strategy)")
    
    available_models = ModelFactory.get_available_models()
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['1', 'rf', 'random_forest', 'random-forest']:
            print(f"\nSelected from command line: Random Forest")
            return 'random_forest'
        elif arg in ['2', 'ti', 'technical_indicator', 'technical-indicator']:
            print(f"\nSelected from command line: Technical Indicator")
            return 'technical_indicator'
    
    # Interactive selection
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            return 'random_forest'
        elif choice == '2':
            return 'technical_indicator'
        else:
            print("Invalid choice. Please enter 1 or 2.")


def display_prediction_results(model, X_latest, symbol: str, latest_price: float):
    """
    Display prediction results in a formatted way.
    
    Args:
        model: Trained model instance
        X_latest: Features for the latest data point
        symbol: Stock symbol
        latest_price: Current stock price
    """
    print_header("PREDICTION RESULTS")
    
    # Make prediction
    prediction = model.predict(X_latest)[0]
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


def main():
    """Main execution function."""
    try:
        # Display configuration
        Config.display()
        
        # Select model
        model_type = select_model()
        
        # Initialize Alpaca client
        print_header("DATA FETCHING")
        client = AlpacaClient()
        
        # Fetch historical data
        symbol = Config.DEFAULT_SYMBOL
        df = client.fetch_historical_data(symbol)
        
        print(f"\nData Summary:")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Total trading days: {len(df)}")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        
        # Feature engineering
        print_header("FEATURE ENGINEERING")
        engineer = FeatureEngineer(forward_days=5, threshold=0.02)
        df_features = engineer.prepare_data(df, include_target=True)
        
        print(f"Created {len(engineer.get_feature_columns())} features")
        print(f"Data after feature engineering: {len(df_features)} samples")
        
        # Prepare training data (all but last row for target variable safety)
        feature_cols = engineer.get_feature_columns()
        
        # Split: use most data for training, keep some recent data for testing
        train_size = int(len(df_features) * 0.8)
        
        X_train = df_features[feature_cols].iloc[:train_size]
        y_train = df_features['target'].iloc[:train_size]
        
        X_test = df_features[feature_cols].iloc[train_size:-5]  # Exclude last few for safety
        y_test = df_features['target'].iloc[train_size:-5]
        
        # Get latest data point for prediction (no target needed)
        df_latest = engineer.prepare_data(df.tail(30), include_target=False)
        X_latest = df_latest[feature_cols].tail(1)
        latest_price = df['close'].iloc[-1]
        
        # Create and train model
        print_header("MODEL TRAINING")
        model = ModelFactory.create_model(model_type)
        model.train(X_train, y_train)
        
        # Evaluate model on test data
        print_header("MODEL EVALUATION")
        metrics = model.evaluate(X_test, y_test)
        print(f"\nTest Set Performance:")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        
        if 'sell_accuracy' in metrics:
            print(f"  SELL Accuracy: {metrics['sell_accuracy']:.2%}")
        if 'hold_accuracy' in metrics:
            print(f"  HOLD Accuracy: {metrics['hold_accuracy']:.2%}")
        if 'buy_accuracy' in metrics:
            print(f"  BUY Accuracy: {metrics['buy_accuracy']:.2%}")
        
        # Generate visualizations
        print_header("GENERATING VISUALIZATIONS")
        visualizer = TradingVisualizer()
        
        # Get predictions and confidence for test set
        test_predictions = model.predict(X_test)
        test_confidence = model.get_confidence(X_test)
        
        # Get the corresponding data for visualization (need original timestamps and prices)
        viz_df = df_features.iloc[train_size:-5].copy()
        
        # Create visualization
        viz_path = visualizer.plot_trading_signals(
            df=viz_df,
            predictions=test_predictions,
            confidence=test_confidence,
            model_name=model.name,
            symbol=symbol,
            feature_cols=feature_cols
        )
        
        # Create summary report
        report_path = visualizer.create_summary_report(
            df=viz_df,
            predictions=test_predictions,
            confidence=test_confidence,
            model_name=model.name,
            symbol=symbol,
            test_metrics=metrics
        )
        
        print(f"\nVisualization files created in: {visualizer.output_dir}")
        
        # Make prediction on latest data
        display_prediction_results(model, X_latest, symbol, latest_price)
        
        # Final notes
        print_header("NOTES")
        print("• This system uses PAPER TRADING only (no real money at risk)")
        print("• Predictions are for educational purposes in pattern recognition")
        print("• Always perform your own analysis before making trading decisions")
        print(f"• Model trained on {train_size} samples, tested on {len(X_test)} samples")
        print(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "=" * 70)
        print("  Program completed successfully!")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


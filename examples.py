#!/usr/bin/env python3
"""
Example usage script demonstrating the trading system capabilities.

This script shows how to:
1. Initialize the system
2. Analyze stocks
3. Generate reports
"""

import os
from src.api.alpaca_client import AlpacaClient
from src.models.model_factory import ModelFactory
from src.strategies.trading_strategy import TradingStrategy
from src.reports.report_generator import ReportGenerator
from src.utils.trend_detection import TrendDetector


def example_1_basic_analysis():
    """Example 1: Basic stock analysis with technical indicators."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Analysis with Technical Indicators")
    print("="*60 + "\n")
    
    # Note: This example requires valid Alpaca API credentials
    # For demonstration, we'll show the code structure
    
    try:
        # Initialize components
        client = AlpacaClient()
        model = ModelFactory.create_model('technical_indicators')
        strategy = TradingStrategy(model, client, min_confidence=0.6)
        
        # Analyze a single stock
        print("Analyzing AAPL...")
        analysis = strategy.analyze_stock('AAPL')
        
        print(f"\nSymbol: {analysis['symbol']}")
        print(f"Action: {analysis['action'].upper()}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        print(f"Current Price: ${analysis.get('current_price', 0):.2f}")
        
        trend = analysis.get('trend_analysis', {})
        print(f"\nTrend: {trend.get('trend', 'N/A')}")
        print(f"Trend Strength: {trend.get('strength', 0):.2f}")
        print(f"Recommendation: {trend.get('recommendation', 'N/A')}")
        
    except Exception as e:
        print(f"Note: This example requires valid Alpaca API credentials")
        print(f"Error: {e}")


def example_2_multiple_stocks():
    """Example 2: Analyze multiple stocks and generate reports."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Stock Analysis")
    print("="*60 + "\n")
    
    try:
        # Initialize
        client = AlpacaClient()
        model = ModelFactory.create_model('random_forest')
        strategy = TradingStrategy(model, client)
        
        # Analyze multiple stocks
        symbols = ['AAPL', 'MSFT', 'NVDA']
        print(f"Analyzing {len(symbols)} stocks: {', '.join(symbols)}\n")
        
        analyses = strategy.scan_watchlist(symbols)
        
        # Generate reports
        report_gen = ReportGenerator()
        report_gen.print_summary(analyses)
        
        # Save reports
        json_path = report_gen.generate_analysis_report(analyses)
        print(f"\nReport saved to: {json_path}")
        
    except Exception as e:
        print(f"Note: This example requires valid Alpaca API credentials")
        print(f"Error: {e}")


def example_3_trend_detection():
    """Example 3: Standalone trend detection."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Trend Detection")
    print("="*60 + "\n")
    
    try:
        # Initialize
        client = AlpacaClient()
        detector = TrendDetector()
        
        # Get data
        symbol = 'AAPL'
        print(f"Fetching data for {symbol}...")
        df = client.get_historical_data(symbol, days=90)
        
        if not df.empty:
            # Detect trend
            trend_info = detector.get_trend_analysis(df)
            
            print(f"\nTrend Analysis for {symbol}:")
            print(f"Trend: {trend_info['trend']}")
            print(f"Strength: {trend_info['strength']:.2f}")
            print(f"Momentum: {trend_info['momentum']:.2%}")
            print(f"Support Level: ${trend_info['support']:.2f}")
            print(f"Resistance Level: ${trend_info['resistance']:.2f}")
            
            breakout = trend_info['breakout']
            if breakout['breakout']:
                print(f"\nBreakout Detected!")
                print(f"Direction: {breakout['direction']}")
                print(f"Strength: {breakout['strength']:.2%}")
        
    except Exception as e:
        print(f"Note: This example requires valid Alpaca API credentials")
        print(f"Error: {e}")


def example_4_model_comparison():
    """Example 4: Compare different models."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Comparison")
    print("="*60 + "\n")
    
    try:
        available_models = ModelFactory.get_available_models()
        print(f"Available Models: {', '.join(available_models)}\n")
        
        # Create instances of each model
        for model_name in available_models:
            model = ModelFactory.create_model(model_name)
            print(f"✓ {model_name}: {type(model).__name__}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# Stock Trading System - Usage Examples")
    print("#"*60)
    
    # Check for API credentials
    api_key = os.getenv('ALPACA_API_KEY', '')
    if not api_key:
        print("\nNote: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
        print("to run examples that interact with the API.\n")
    
    # Run examples
    example_4_model_comparison()  # This one works without API credentials
    example_1_basic_analysis()
    example_2_multiple_stocks()
    example_3_trend_detection()
    
    print("\n" + "#"*60)
    print("# Examples Complete")
    print("#"*60 + "\n")


if __name__ == '__main__':
    main()

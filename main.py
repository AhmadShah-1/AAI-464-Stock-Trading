#!/usr/bin/env python3
"""
Main application for AI-powered stock trading with Alpaca API.

This script provides functionality to:
- Analyze tech/AI stocks using machine learning models
- Generate buy/sell/hold recommendations
- Detect market trends (bull/bear runs)
- Generate comprehensive reports and statistics
"""

import argparse
import logging
import sys
from datetime import datetime

from src.config import Config
from src.api.alpaca_client import AlpacaClient
from src.models.model_factory import ModelFactory
from src.strategies.trading_strategy import TradingStrategy
from src.reports.report_generator import ReportGenerator
from src.utils.features import prepare_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_argparse():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='AI-Powered Stock Trading System for Tech/AI Stocks'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['analyze', 'train', 'backtest', 'trade'],
        default='analyze',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=Config.DEFAULT_MODEL,
        choices=ModelFactory.get_available_models(),
        help='Model to use for predictions'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Stock symbols to analyze (default: tech stocks from config)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute trades (requires TRADING_ENABLED=true in config)'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        default=True,
        help='Generate reports and visualizations'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum confidence threshold for trades'
    )
    
    return parser


def train_model(model_name: str, symbols: list = None):
    """
    Train a model on historical data.
    
    Args:
        model_name: Name of model to train
        symbols: List of stock symbols for training
    """
    logger.info(f"Training {model_name} model...")
    
    if symbols is None:
        symbols = Config.TECH_STOCKS
    
    # Initialize client
    client = AlpacaClient()
    
    # Collect training data
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        df = client.get_historical_data(symbol, days=730)  # 2 years
        
        if not df.empty:
            df_features = prepare_features(df, horizon=Config.PREDICTION_HORIZON)
            if not df_features.empty:
                all_data.append(df_features)
    
    if not all_data:
        logger.error("No training data collected")
        return None
    
    # Combine all data
    import pandas as pd
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Prepare features and target
    feature_cols = [col for col in combined_df.columns 
                   if not col.startswith('target') and col != 'future_return']
    X = combined_df[feature_cols]
    y = combined_df['target_action'].astype(int)
    
    # Drop any remaining NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
    
    # Create and train model
    model = ModelFactory.create_model(model_name)
    model.train(X, y)
    
    # Evaluate
    metrics = model.evaluate(X, y)
    logger.info(f"Training metrics: {metrics}")
    
    return model


def analyze_stocks(model, symbols: list = None, execute_trades: bool = False):
    """
    Analyze stocks and optionally execute trades.
    
    Args:
        model: Trained model
        symbols: List of stock symbols
        execute_trades: Whether to execute trades
    """
    logger.info("Starting stock analysis...")
    
    # Initialize components
    client = AlpacaClient()
    strategy = TradingStrategy(model, client)
    
    # Scan watchlist
    analyses = strategy.scan_watchlist(symbols)
    
    # Execute trades if requested
    if execute_trades and Config.TRADING_ENABLED:
        logger.info("Executing trades...")
        for analysis in analyses:
            if analysis.get('action') in ['buy', 'sell']:
                result = strategy.execute_trade(analysis)
                logger.info(f"Trade result for {analysis['symbol']}: {result}")
    
    return analyses


def generate_reports(analyses: list):
    """
    Generate comprehensive reports.
    
    Args:
        analyses: List of analysis results
    """
    logger.info("Generating reports...")
    
    report_gen = ReportGenerator()
    
    # Print summary to console
    report_gen.print_summary(analyses)
    
    # Generate JSON report
    json_path = report_gen.generate_analysis_report(analyses)
    logger.info(f"JSON report: {json_path}")
    
    # Generate visualizations
    try:
        viz_path = report_gen.generate_visualization(analyses)
        logger.info(f"Visualization: {viz_path}")
    except Exception as e:
        logger.warning(f"Could not generate visualization: {e}")
    
    # Generate portfolio report if trading is enabled
    if Config.TRADING_ENABLED:
        try:
            client = AlpacaClient()
            account_info = client.get_account_info()
            positions = client.get_positions()
            
            portfolio_path = report_gen.generate_portfolio_report(account_info, positions)
            logger.info(f"Portfolio report: {portfolio_path}")
        except Exception as e:
            logger.warning(f"Could not generate portfolio report: {e}")


def main():
    """Main application entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Validate configuration
    if not Config.validate():
        logger.error("Configuration validation failed. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    
    logger.info(f"Starting trading system in {args.mode} mode")
    logger.info(f"Using model: {args.model}")
    
    try:
        if args.mode == 'train':
            # Train model
            model = train_model(args.model, args.symbols)
            if model:
                logger.info("Model training completed successfully")
        
        elif args.mode == 'analyze':
            # Create or load model
            if args.model == 'technical_indicators':
                # Technical model doesn't need training
                model = ModelFactory.create_model(args.model)
            else:
                # Train ML model
                model = train_model(args.model, args.symbols or Config.TECH_STOCKS)
            
            if not model:
                logger.error("Failed to create/train model")
                sys.exit(1)
            
            # Analyze stocks
            analyses = analyze_stocks(model, args.symbols, args.execute)
            
            # Generate reports
            if args.report:
                generate_reports(analyses)
        
        elif args.mode == 'backtest':
            logger.info("Backtesting mode not yet implemented")
            # TODO: Implement backtesting
        
        elif args.mode == 'trade':
            logger.info("Live trading mode")
            # Similar to analyze but with trading enabled
            model = train_model(args.model, args.symbols or Config.TECH_STOCKS)
            if model:
                analyses = analyze_stocks(model, args.symbols, execute_trades=True)
                if args.report:
                    generate_reports(analyses)
        
        logger.info("Operation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

"""Trading strategy engine for executing buy/sell/hold decisions."""

import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from src.api.alpaca_client import AlpacaClient
from src.models.base_model import BaseModel
from src.utils.features import prepare_features
from src.utils.trend_detection import TrendDetector
from src.config import Config

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Main trading strategy engine."""
    
    def __init__(
        self,
        model: BaseModel,
        alpaca_client: AlpacaClient,
        min_confidence: float = 0.6,
        use_trend_filter: bool = True
    ):
        """
        Initialize trading strategy.
        
        Args:
            model: Prediction model
            alpaca_client: Alpaca API client
            min_confidence: Minimum confidence threshold for trades
            use_trend_filter: Whether to use trend detection filter
        """
        self.model = model
        self.alpaca_client = alpaca_client
        self.min_confidence = min_confidence
        self.use_trend_filter = use_trend_filter
        self.trend_detector = TrendDetector()
    
    def analyze_stock(self, symbol: str) -> Dict:
        """
        Analyze a stock and generate trading signal.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Analysis results with recommendation
        """
        try:
            # Fetch historical data
            df = self.alpaca_client.get_historical_data(symbol, days=365)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return {
                    'symbol': symbol,
                    'action': 'hold',
                    'confidence': 0.0,
                    'error': 'No data available'
                }
            
            # Prepare features
            df_features = prepare_features(df, horizon=Config.PREDICTION_HORIZON)
            
            if df_features.empty:
                logger.warning(f"Insufficient data for feature engineering: {symbol}")
                return {
                    'symbol': symbol,
                    'action': 'hold',
                    'confidence': 0.0,
                    'error': 'Insufficient data'
                }
            
            # Get latest features (excluding target columns)
            feature_cols = [col for col in df_features.columns 
                          if not col.startswith('target') and col != 'future_return']
            latest_features = df_features[feature_cols].iloc[-1:].copy()
            
            # Get model prediction
            prediction = self.model.predict(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]
            confidence = max(probabilities)
            
            # Map prediction to action
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            action = action_map.get(prediction, 'hold')
            
            # Get trend analysis
            trend_analysis = self.trend_detector.get_trend_analysis(df)
            
            # Apply trend filter
            if self.use_trend_filter:
                action = self._apply_trend_filter(action, trend_analysis)
            
            # Get current price
            current_price = df.iloc[-1]['close']
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'prediction': prediction,
                'probabilities': {
                    'sell': probabilities[0],
                    'hold': probabilities[1],
                    'buy': probabilities[2]
                },
                'current_price': current_price,
                'trend_analysis': trend_analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'hold',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _apply_trend_filter(self, action: str, trend_analysis: Dict) -> str:
        """
        Apply trend-based filter to trading action.
        
        Args:
            action: Original action
            trend_analysis: Trend analysis results
        
        Returns:
            Filtered action
        """
        trend = trend_analysis['trend']
        
        # Don't buy in strong bearish trend
        if action == 'buy' and trend == 'bearish':
            if trend_analysis['strength'] > 0.5:
                return 'hold'
        
        # Don't sell in strong bullish trend
        if action == 'sell' and trend == 'bullish':
            if trend_analysis['strength'] > 0.5:
                return 'hold'
        
        return action
    
    def execute_trade(self, analysis: Dict, position_size: Optional[float] = None) -> Dict:
        """
        Execute trade based on analysis.
        
        Args:
            analysis: Analysis results
            position_size: Position size in dollars (default: Config.MAX_POSITION_SIZE)
        
        Returns:
            Trade execution results
        """
        symbol = analysis['symbol']
        action = analysis['action']
        confidence = analysis.get('confidence', 0.0)
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            logger.info(f"Confidence {confidence:.2f} below threshold for {symbol}")
            return {
                'symbol': symbol,
                'executed': False,
                'reason': 'Low confidence'
            }
        
        # Skip hold actions
        if action == 'hold':
            return {
                'symbol': symbol,
                'executed': False,
                'reason': 'Hold recommendation'
            }
        
        # Calculate position size
        if position_size is None:
            position_size = Config.MAX_POSITION_SIZE
        
        current_price = analysis.get('current_price', 0)
        if current_price == 0:
            return {
                'symbol': symbol,
                'executed': False,
                'reason': 'Invalid price'
            }
        
        qty = int(position_size / current_price)
        
        if qty == 0:
            return {
                'symbol': symbol,
                'executed': False,
                'reason': 'Insufficient position size'
            }
        
        # Execute trade
        if action == 'buy':
            order = self.alpaca_client.place_order(symbol, qty, 'buy')
        elif action == 'sell':
            # Check if we have a position to sell
            positions = self.alpaca_client.get_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                return {
                    'symbol': symbol,
                    'executed': False,
                    'reason': 'No position to sell'
                }
            
            qty = min(qty, position['qty'])
            order = self.alpaca_client.place_order(symbol, qty, 'sell')
        else:
            return {
                'symbol': symbol,
                'executed': False,
                'reason': 'Invalid action'
            }
        
        if order:
            return {
                'symbol': symbol,
                'executed': True,
                'action': action,
                'qty': qty,
                'order': order
            }
        else:
            return {
                'symbol': symbol,
                'executed': False,
                'reason': 'Order placement failed'
            }
    
    def scan_watchlist(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan a list of stocks and generate recommendations.
        
        Args:
            symbols: List of stock symbols (default: Config.TECH_STOCKS)
        
        Returns:
            List of analysis results
        """
        if symbols is None:
            symbols = Config.TECH_STOCKS
        
        results = []
        
        for symbol in symbols:
            logger.info(f"Analyzing {symbol}...")
            analysis = self.analyze_stock(symbol)
            results.append(analysis)
        
        return results

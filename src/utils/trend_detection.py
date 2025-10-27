"""Trend detection utilities for identifying bull and bear markets."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class TrendDetector:
    """Detect market trends (bull/bear runs)."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize trend detector.
        
        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def detect_trend(self, df: pd.DataFrame) -> str:
        """
        Detect current market trend.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            Trend label: 'bullish', 'bearish', or 'neutral'
        """
        if df.empty or len(df) < self.long_window:
            return 'neutral'
        
        df = df.copy()
        
        # Calculate moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Golden cross (bullish) / Death cross (bearish)
        if latest['sma_short'] > latest['sma_long']:
            # Check if price is above both MAs
            if latest['close'] > latest['sma_short']:
                return 'bullish'
            else:
                return 'neutral'
        elif latest['sma_short'] < latest['sma_long']:
            # Check if price is below both MAs
            if latest['close'] < latest['sma_short']:
                return 'bearish'
            else:
                return 'neutral'
        else:
            return 'neutral'
    
    def detect_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength (0-1).
        
        Args:
            df: DataFrame with price data
        
        Returns:
            Trend strength value between 0 and 1
        """
        if df.empty or len(df) < self.long_window:
            return 0.5
        
        df = df.copy()
        
        # Calculate moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate divergence
        latest = df.iloc[-1]
        divergence = abs(latest['sma_short'] - latest['sma_long']) / latest['sma_long']
        
        # Normalize to 0-1 range
        strength = min(divergence / 0.1, 1.0)  # 10% divergence = max strength
        
        return strength
    
    def identify_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[float, float]:
        """
        Identify support and resistance levels.
        
        Args:
            df: DataFrame with price data
            window: Lookback window for calculation
        
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if df.empty or len(df) < window:
            return (0.0, 0.0)
        
        recent_data = df.tail(window)
        
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        
        return (support, resistance)
    
    def detect_breakout(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Detect price breakouts.
        
        Args:
            df: DataFrame with price data
            window: Lookback window
        
        Returns:
            Dictionary with breakout information
        """
        if df.empty or len(df) < window + 1:
            return {'breakout': False, 'direction': None, 'strength': 0.0}
        
        support, resistance = self.identify_support_resistance(df.iloc[:-1], window)
        current_price = df.iloc[-1]['close']
        
        breakout = False
        direction = None
        strength = 0.0
        
        # Check for breakout above resistance
        if current_price > resistance:
            breakout = True
            direction = 'bullish'
            strength = (current_price - resistance) / resistance
        # Check for breakdown below support
        elif current_price < support:
            breakout = True
            direction = 'bearish'
            strength = (support - current_price) / support
        
        return {
            'breakout': breakout,
            'direction': direction,
            'strength': min(strength, 1.0),
            'support': support,
            'resistance': resistance,
            'current_price': current_price
        }
    
    def get_trend_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive trend analysis.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            Dictionary with complete trend analysis
        """
        trend = self.detect_trend(df)
        strength = self.detect_trend_strength(df)
        support, resistance = self.identify_support_resistance(df)
        breakout_info = self.detect_breakout(df)
        
        # Calculate price momentum
        if len(df) >= 10:
            momentum = (df.iloc[-1]['close'] - df.iloc[-10]['close']) / df.iloc[-10]['close']
        else:
            momentum = 0.0
        
        return {
            'trend': trend,
            'strength': strength,
            'momentum': momentum,
            'support': support,
            'resistance': resistance,
            'breakout': breakout_info,
            'recommendation': self._get_recommendation(trend, strength, breakout_info)
        }
    
    def _get_recommendation(
        self,
        trend: str,
        strength: float,
        breakout_info: Dict
    ) -> str:
        """Generate trading recommendation based on trend analysis."""
        if trend == 'bullish' and strength > 0.6:
            return 'strong_buy'
        elif trend == 'bullish' and strength > 0.3:
            return 'buy'
        elif trend == 'bearish' and strength > 0.6:
            return 'strong_sell'
        elif trend == 'bearish' and strength > 0.3:
            return 'sell'
        elif breakout_info['breakout'] and breakout_info['direction'] == 'bullish':
            return 'buy'
        elif breakout_info['breakout'] and breakout_info['direction'] == 'bearish':
            return 'sell'
        else:
            return 'hold'

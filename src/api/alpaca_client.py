"""Alpaca API client for stock trading operations."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.config import Config

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Client for interacting with Alpaca Trading API."""
    
    def __init__(self):
        """Initialize Alpaca client with API credentials."""
        self.trading_client = TradingClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            paper=True
        )
        self.data_client = StockHistoricalDataClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY
        )
        logger.info("Alpaca client initialized successfully")
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise
    
    def get_historical_data(
        self,
        symbol: str,
        days: int = 365,
        timeframe: TimeFrame = TimeFrame.Day
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            timeframe: Data timeframe (Day, Hour, Minute)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Reset index to get symbol and timestamp as columns
            df = df.reset_index()
            
            # Rename columns for consistency
            df.columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('timestamp')
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market'
    ) -> Optional[Dict]:
        """
        Place a trading order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            order_type: Order type (default: 'market')
        
        Returns:
            Order details if successful, None otherwise
        """
        if not Config.TRADING_ENABLED:
            logger.warning("Trading is disabled in configuration")
            return None
        
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data=market_order_data)
            
            logger.info(f"Order placed: {side} {qty} shares of {symbol}")
            return {
                'id': str(order.id),
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'status': order.status.value,
                'submitted_at': order.submitted_at
            }
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """Get orders with optional status filter."""
        try:
            request_params = {}
            if status:
                request_params['status'] = OrderStatus[status.upper()]
            
            request = GetOrdersRequest(**request_params) if request_params else None
            orders = self.trading_client.get_orders(filter=request)
            
            return [
                {
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side.value,
                    'status': order.status.value,
                    'submitted_at': order.submitted_at
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

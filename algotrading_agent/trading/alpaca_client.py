import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.common.exceptions import APIError

from ..components.decision_engine import TradingPair


class AlpacaClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.paper_trading = config.get('paper_trading', True)
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials are required")
            
        # Initialize clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper_trading
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        self.logger = logging.getLogger("algotrading.alpaca_client")
        self.logger.info(f"Initialized Alpaca client (paper_trading={self.paper_trading})")
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                "account_id": account.id,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "status": account.status.value,
                "day_trade_count": account.day_trade_count,
                "pattern_day_trader": account.pattern_day_trader
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "quantity": int(pos.qty),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "side": pos.side.value if pos.side else "long"
                }
                for pos in positions
            ]
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
            
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                # Use mid price (average of bid and ask)
                return (float(quote.bid_price) + float(quote.ask_price)) / 2
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
            
    async def execute_trading_pair(self, pair: TradingPair) -> Dict[str, Any]:
        """Execute a trading pair (entry order with stop-loss and take-profit)"""
        try:
            # Get current price to validate
            current_price = await self.get_current_price(pair.symbol)
            if not current_price:
                raise ValueError(f"Could not get current price for {pair.symbol}")
                
            # Create bracket order (entry + stop-loss + take-profit)
            side = OrderSide.BUY if pair.action == "buy" else OrderSide.SELL
            
            # Prepare the main order request
            order_request = MarketOrderRequest(
                symbol=pair.symbol,
                qty=pair.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(
                    stop_price=pair.stop_loss
                ),
                take_profit=TakeProfitRequest(
                    limit_price=pair.take_profit
                )
            )
            
            # Submit the order
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(f"Submitted bracket order for {pair.symbol}: {order.id}")
            
            return {
                "order_id": order.id,
                "symbol": pair.symbol,
                "side": order.side.value,
                "quantity": int(order.qty),
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat(),
                "stop_loss_price": pair.stop_loss,
                "take_profit_price": pair.take_profit
            }
            
        except APIError as e:
            self.logger.error(f"Alpaca API error executing {pair.symbol}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing trading pair {pair.symbol}: {e}")
            raise
            
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            return {
                "order_id": order.id,
                "status": order.status.value,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": int(order.qty),
                "filled_qty": int(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
                "submitted_at": order.submitted_at.isoformat(),
                "filled_at": order.filled_at.isoformat() if order.filled_at else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order status {order_id}: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
            
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get orders with optional status filter"""
        try:
            orders = self.trading_client.get_orders()
            
            order_list = []
            for order in orders:
                if status and order.status.value != status:
                    continue
                    
                order_list.append({
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": int(order.qty),
                    "status": order.status.value,
                    "order_type": order.order_type.value,
                    "submitted_at": order.submitted_at.isoformat(),
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "filled_qty": int(order.filled_qty or 0),
                    "filled_avg_price": float(order.filled_avg_price or 0)
                })
                
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
            
    async def close_position(self, symbol: str, percentage: float = 100.0) -> Dict[str, Any]:
        """Close a position (partial or full)"""
        try:
            if percentage == 100.0:
                # Close entire position
                order = self.trading_client.close_position(symbol)
            else:
                # Close partial position
                positions = await self.get_positions()
                position = next((p for p in positions if p["symbol"] == symbol), None)
                
                if not position:
                    raise ValueError(f"No position found for {symbol}")
                    
                qty_to_close = int(abs(position["quantity"]) * (percentage / 100.0))
                side = OrderSide.SELL if position["quantity"] > 0 else OrderSide.BUY
                
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty_to_close,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_request)
                
            self.logger.info(f"Closed position for {symbol}: {order.id}")
            
            return {
                "order_id": order.id,
                "symbol": symbol,
                "percentage_closed": percentage,
                "status": order.status.value
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            raise
            
    async def get_portfolio_history(self, period: str = "1D") -> Dict[str, Any]:
        """Get portfolio performance history"""
        try:
            portfolio = self.trading_client.get_portfolio_history(period=period)
            
            return {
                "equity": [float(eq) for eq in portfolio.equity],
                "profit_loss": [float(pl) for pl in portfolio.profit_loss],
                "profit_loss_pct": [float(plp) for plp in portfolio.profit_loss_pct],
                "base_value": float(portfolio.base_value),
                "timeframe": portfolio.timeframe.value,
                "timestamp": [ts.isoformat() for ts in portfolio.timestamp]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return {}
            
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
            
    async def validate_trading_pair(self, pair: TradingPair) -> Dict[str, Any]:
        """Validate a trading pair before execution"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if symbol exists and is tradable
            current_price = await self.get_current_price(pair.symbol)
            if not current_price:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Symbol {pair.symbol} not found or not tradable")
                return validation_result
                
            # Check market hours
            if not self.is_market_open():
                validation_result["warnings"].append("Market is currently closed")
                
            # Check account buying power
            account = await self.get_account_info()
            position_value = current_price * pair.quantity
            
            if position_value > account["buying_power"]:
                validation_result["valid"] = False
                validation_result["errors"].append("Insufficient buying power")
                
            # Validate stop loss and take profit levels
            if pair.action == "buy":
                if pair.stop_loss >= current_price:
                    validation_result["errors"].append("Stop loss must be below current price for buy orders")
                    validation_result["valid"] = False
                if pair.take_profit <= current_price:
                    validation_result["errors"].append("Take profit must be above current price for buy orders")
                    validation_result["valid"] = False
            else:  # sell (short)
                if pair.stop_loss <= current_price:
                    validation_result["errors"].append("Stop loss must be above current price for sell orders")
                    validation_result["valid"] = False
                if pair.take_profit >= current_price:
                    validation_result["errors"].append("Take profit must be below current price for sell orders")
                    validation_result["valid"] = False
                    
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            
        return validation_result
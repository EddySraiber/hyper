import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, AssetClass
from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest
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
        
        # Initialize crypto data client for crypto trading
        self.crypto_data_client = CryptoHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        self.logger = logging.getLogger("algotrading.alpaca_client")
        self.logger.info(f"Initialized Alpaca client (paper_trading={self.paper_trading})")
        
        # Supported crypto symbols on Alpaca (using correct trading pair format)
        self.crypto_symbols = {
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'DOGE/USD', 'SOL/USD', 'AVAX/USD',
            'DOT/USD', 'LINK/USD', 'SHIB/USD', 'UNI/USD', 'AAVE/USD', 'BCH/USD',
            'CRV/USD', 'GRT/USD', 'MKR/USD', 'PEPE/USD', 'SUSHI/USD', 'XRP/USD',
            'XTZ/USD', 'YFI/USD', 'USDC/USD', 'USDT/USD',
            # Also include legacy format for detection
            'BTCUSD', 'ETHUSD', 'LTCUSD', 'DOGEUSD', 'SOLUSD', 'AVAXUSD', 
            'DOTUSD', 'LINKUSD', 'SHIBUSD', 'UNIUSD', 'AAVEUSD', 'BCHUSD',
            'CRVUSD', 'GRTUSD', 'MKRUSD', 'PEPEUSD', 'SUSHIUSD', 'XRPUSD', 
            'XTZUSD', 'YFIUSD'
        }
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a crypto asset"""
        # Normalize symbol format
        symbol = symbol.upper().replace('/', '')
        
        # Check against known crypto symbols
        return symbol in self.crypto_symbols
    
    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol format for Alpaca API (use trading pair format)"""
        symbol = symbol.upper()
        
        # Common crypto symbol mappings to trading pair format
        crypto_mappings = {
            'BTC': 'BTC/USD',
            'ETH': 'ETH/USD', 
            'LTC': 'LTC/USD',
            'DOGE': 'DOGE/USD',
            'SOL': 'SOL/USD',
            'AVAX': 'AVAX/USD',
            'DOT': 'DOT/USD',
            'LINK': 'LINK/USD',
            'SHIB': 'SHIB/USD',
            'UNI': 'UNI/USD',
            'AAVE': 'AAVE/USD',
            'BCH': 'BCH/USD',
            # Convert legacy format to trading pair format
            'BTCUSD': 'BTC/USD',
            'ETHUSD': 'ETH/USD',
            'LTCUSD': 'LTC/USD',
            'DOGEUSD': 'DOGE/USD',
            'SOLUSD': 'SOL/USD',
            'AVAXUSD': 'AVAX/USD',
            'DOTUSD': 'DOT/USD',
            'LINKUSD': 'LINK/USD',
            'SHIBUSD': 'SHIB/USD',
            'UNIUSD': 'UNI/USD'
        }
        
        # Return mapped symbol or original if already in correct format
        return crypto_mappings.get(symbol, symbol)
        
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
                "day_trade_count": getattr(account, 'day_trade_count', 0),
                "pattern_day_trader": getattr(account, 'pattern_day_trader', False)
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
        """Get current price for a symbol (stocks or crypto) with timeout protection"""
        try:
            # Normalize symbol and check if it's crypto
            normalized_symbol = self._normalize_crypto_symbol(symbol)
            is_crypto = self._is_crypto_symbol(normalized_symbol)
            
            if is_crypto:
                # Use crypto data client for crypto symbols
                request = CryptoLatestQuoteRequest(symbol_or_symbols=[normalized_symbol])
                
                quotes = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.crypto_data_client.get_crypto_latest_quote, request
                    ), timeout=10.0
                )
                
                if normalized_symbol in quotes:
                    quote = quotes[normalized_symbol]
                    # Use mid price (average of bid and ask)
                    return (float(quote.bid_price) + float(quote.ask_price)) / 2
            else:
                # Use stock data client for stock symbols
                request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                
                quotes = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.data_client.get_stock_latest_quote, request
                    ), timeout=10.0
                )
                
                if symbol in quotes:
                    quote = quotes[symbol]
                    # Use mid price (average of bid and ask)
                    return (float(quote.bid_price) + float(quote.ask_price)) / 2
                    
            return None
            
        except asyncio.TimeoutError:
            asset_type = "crypto" if self._is_crypto_symbol(symbol) else "stock"
            self.logger.warning(f"Timeout getting {asset_type} price for {symbol} - market may be closed")
            return None
        except Exception as e:
            asset_type = "crypto" if self._is_crypto_symbol(symbol) else "stock"  
            self.logger.error(f"Error getting {asset_type} price for {symbol}: {e}")
            return None
            
    async def execute_trading_pair(self, pair: TradingPair, price_flexibility_pct: float = 0.01) -> Dict[str, Any]:
        """Execute a trading pair (stocks or crypto) with price flexibility check"""
        try:
            # Normalize symbol for crypto
            trading_symbol = self._normalize_crypto_symbol(pair.symbol)
            is_crypto = self._is_crypto_symbol(trading_symbol)
            
            # Get current price to validate and check flexibility bounds
            current_price = await self.get_current_price(trading_symbol)
            if not current_price:
                raise ValueError(f"Could not get current price for {trading_symbol}")
            
            # Check if current price is acceptable (embrace better prices, reject worse ones)
            target_price = pair.entry_price
            
            # Log asset type for debugging
            asset_type = "crypto" if is_crypto else "stock"
            self.logger.info(f"Executing {asset_type} trading pair: {trading_symbol}")
            
            if pair.action == "buy":
                # For BUY orders: Accept current price if it's not TOO MUCH HIGHER than target
                # We're happy to buy cheaper, only reject if too expensive
                max_acceptable_price = target_price * (1 + price_flexibility_pct)
                if current_price > max_acceptable_price:
                    raise ValueError(f"Buy price ${current_price:.2f} too expensive (max: ${max_acceptable_price:.2f})")
                
                # Log if we're getting a better deal
                if current_price < target_price:
                    savings = target_price - current_price
                    savings_pct = (savings / target_price) * 100
                    self.logger.info(f"💰 Better entry price! {trading_symbol} @ ${current_price:.2f} "
                                   f"(target: ${target_price:.2f}, saving: ${savings:.2f} = {savings_pct:.1f}%)")
                else:
                    self.logger.info(f"Price check passed: {trading_symbol} @ ${current_price:.2f} "
                                   f"(target: ${target_price:.2f}, within +{price_flexibility_pct*100:.1f}%)")
                    
            else:  # sell (short)
                # For SELL orders: Accept current price if it's not TOO MUCH LOWER than target  
                # We're happy to sell higher, only reject if too low
                min_acceptable_price = target_price * (1 - price_flexibility_pct)
                if current_price < min_acceptable_price:
                    raise ValueError(f"Sell price ${current_price:.2f} too low (min: ${min_acceptable_price:.2f})")
                
                # Log if we're getting a better deal  
                if current_price > target_price:
                    bonus = current_price - target_price
                    bonus_pct = (bonus / target_price) * 100
                    self.logger.info(f"💰 Better exit price! {trading_symbol} @ ${current_price:.2f} "
                                   f"(target: ${target_price:.2f}, bonus: ${bonus:.2f} = {bonus_pct:.1f}%)")
                else:
                    self.logger.info(f"Price check passed: {trading_symbol} @ ${current_price:.2f} "
                                   f"(target: ${target_price:.2f}, within -{price_flexibility_pct*100:.1f}%)")
                
            # Create bracket order (entry + stop-loss + take-profit)
            side = OrderSide.BUY if pair.action == "buy" else OrderSide.SELL
            
            # CRITICAL: Force precise rounding for Alpaca API compliance
            stop_price_rounded = round(float(pair.stop_loss), 2)
            take_profit_rounded = round(float(pair.take_profit), 2)
            
            # Prepare the main order request (crypto vs stock differences)
            if is_crypto:
                # Crypto orders: Use notional amount (minimum $10) and no bracket orders initially
                notional_amount = pair.entry_price * pair.quantity
                if notional_amount < 10.0:
                    notional_amount = 10.0  # Alpaca minimum for crypto
                    self.logger.info(f"🚀 Adjusting crypto order to minimum $10: {trading_symbol}")
                
                order_request = MarketOrderRequest(
                    symbol=trading_symbol,
                    notional=notional_amount,  # Use notional for crypto
                    side=side,
                    time_in_force=TimeInForce.GTC  # Crypto uses GTC, not DAY
                )
                self.logger.info(f"🚀 CRYPTO ORDER: {side} ${notional_amount:.2f} of {trading_symbol}")
                
            else:
                # Stock orders: Use quantity and bracket orders
                order_request = MarketOrderRequest(
                    symbol=trading_symbol,
                    qty=pair.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(
                        stop_price=stop_price_rounded
                    ),
                    take_profit=TakeProfitRequest(
                        limit_price=take_profit_rounded
                    )
                )
            
            # Submit the order with timeout protection (15 seconds)
            order = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.trading_client.submit_order, order_request
                ), timeout=15.0
            )
            
            self.logger.info(f"Submitted bracket order for {trading_symbol}: {order.id}")
            
            return {
                "order_id": order.id,
                "symbol": trading_symbol,
                "side": order.side.value,
                "quantity": int(order.qty),
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat(),
                "stop_loss_price": pair.stop_loss,
                "take_profit_price": pair.take_profit
            }
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout executing order for {trading_symbol} - market may be closed")
            raise
        except APIError as e:
            self.logger.error(f"Alpaca API error executing {trading_symbol}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing trading pair {trading_symbol}: {e}")
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
            
    async def is_market_open(self, symbol: str = None) -> bool:
        """Check if market is currently open (crypto trades 24/7)"""
        try:
            # If symbol is crypto, it's always tradeable (24/7)
            if symbol and self._is_crypto_symbol(symbol):
                return True
                
            # For stocks, check market hours
            clock = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.trading_client.get_clock
                ), timeout=5.0
            )
            return clock.is_open
        except asyncio.TimeoutError:
            self.logger.warning("Timeout checking market status - assuming market closed")
            return False
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
            
    async def validate_trading_pair(self, pair: TradingPair) -> Dict[str, Any]:
        """Validate a trading pair (stocks or crypto) before execution"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Normalize symbol for crypto
            trading_symbol = self._normalize_crypto_symbol(pair.symbol)
            is_crypto = self._is_crypto_symbol(trading_symbol)
            
            # Check if symbol exists and is tradable
            current_price = await self.get_current_price(trading_symbol)
            if not current_price:
                asset_type = "crypto" if is_crypto else "stock"
                validation_result["valid"] = False
                validation_result["errors"].append(f"{asset_type.capitalize()} symbol {trading_symbol} not found or not tradable")
                return validation_result
                
            # Check market hours (crypto trades 24/7)
            if not await self.is_market_open(trading_symbol):
                validation_result["warnings"].append("Stock market is currently closed")
            elif is_crypto:
                validation_result["warnings"].append("Crypto trades 24/7 - no market hour restrictions")
                
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
        
    async def update_stop_loss(self, symbol: str, new_stop_price: float, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update stop-loss for an existing position
        
        Args:
            symbol: Stock symbol
            new_stop_price: New stop-loss price
            order_id: Specific order ID to update (optional, will find active stop if not provided)
        
        Returns:
            Dictionary with update result
        """
        try:
            # Find the current stop-loss order if order_id not provided
            if not order_id:
                orders = await self.get_orders("open")
                stop_orders = [
                    order for order in orders 
                    if order["symbol"] == symbol and "stop" in order.get("order_type", "").lower()
                ]
                
                if not stop_orders:
                    return {
                        "success": False,
                        "error": f"No active stop-loss order found for {symbol}",
                        "symbol": symbol
                    }
                
                # Use the most recent stop order
                order_id = stop_orders[0]["order_id"]
            
            # Cancel the existing stop order
            cancel_success = await self.cancel_order(order_id)
            if not cancel_success:
                return {
                    "success": False,
                    "error": f"Failed to cancel existing stop order {order_id}",
                    "symbol": symbol
                }
            
            # Get current position to determine quantity and side
            positions = await self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not position:
                return {
                    "success": False,
                    "error": f"No position found for {symbol}",
                    "symbol": symbol
                }
            
            # Create new stop order
            quantity = abs(position["quantity"])
            side = OrderSide.SELL if position["quantity"] > 0 else OrderSide.BUY
            
            stop_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC,  # Good Till Canceled
                order_class=OrderClass.SIMPLE,
                stop_price=new_stop_price
            )
            
            # Submit new stop order with timeout protection
            new_order = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.trading_client.submit_order, stop_request
                ), timeout=15.0
            )
            
            self.logger.info(f"📈 Updated stop-loss for {symbol}: ${new_stop_price:.2f} (Order: {new_order.id})")
            
            return {
                "success": True,
                "symbol": symbol,
                "new_order_id": new_order.id,
                "old_order_id": order_id,
                "new_stop_price": new_stop_price,
                "quantity": quantity,
                "side": side.value
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timeout updating stop-loss for {symbol}",
                "symbol": symbol
            }
        except Exception as e:
            self.logger.error(f"Error updating stop-loss for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }
    
    async def update_take_profit(self, symbol: str, new_take_profit_price: float, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update take-profit for an existing position
        
        Args:
            symbol: Stock symbol
            new_take_profit_price: New take-profit price
            order_id: Specific order ID to update (optional, will find active take-profit if not provided)
        
        Returns:
            Dictionary with update result
        """
        try:
            # Find the current take-profit order if order_id not provided
            if not order_id:
                orders = await self.get_orders("open")
                limit_orders = [
                    order for order in orders 
                    if order["symbol"] == symbol and order.get("order_type") == "limit"
                ]
                
                if not limit_orders:
                    return {
                        "success": False,
                        "error": f"No active take-profit order found for {symbol}",
                        "symbol": symbol
                    }
                
                # Use the most recent limit order (assuming it's take-profit)
                order_id = limit_orders[0]["order_id"]
            
            # Cancel the existing take-profit order
            cancel_success = await self.cancel_order(order_id)
            if not cancel_success:
                return {
                    "success": False,
                    "error": f"Failed to cancel existing take-profit order {order_id}",
                    "symbol": symbol
                }
            
            # Get current position to determine quantity and side
            positions = await self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not position:
                return {
                    "success": False,
                    "error": f"No position found for {symbol}",
                    "symbol": symbol
                }
            
            # Create new limit order for take-profit
            quantity = abs(position["quantity"])
            side = OrderSide.SELL if position["quantity"] > 0 else OrderSide.BUY
            
            limit_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC,  # Good Till Canceled
                limit_price=new_take_profit_price
            )
            
            # Submit new take-profit order with timeout protection
            new_order = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.trading_client.submit_order, limit_request
                ), timeout=15.0
            )
            
            self.logger.info(f"🎯 Updated take-profit for {symbol}: ${new_take_profit_price:.2f} (Order: {new_order.id})")
            
            return {
                "success": True,
                "symbol": symbol,
                "new_order_id": new_order.id,
                "old_order_id": order_id,
                "new_take_profit_price": new_take_profit_price,
                "quantity": quantity,
                "side": side.value
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timeout updating take-profit for {symbol}",
                "symbol": symbol
            }
        except Exception as e:
            self.logger.error(f"Error updating take-profit for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }
    
    async def update_position_parameters(self, symbol: str, new_stop_price: Optional[float] = None, 
                                       new_take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Update both stop-loss and take-profit for a position
        
        Args:
            symbol: Stock symbol
            new_stop_price: New stop-loss price (optional)
            new_take_profit_price: New take-profit price (optional)
        
        Returns:
            Dictionary with update results
        """
        results = {
            "symbol": symbol,
            "stop_loss_result": None,
            "take_profit_result": None,
            "success": True,
            "errors": []
        }
        
        # Update stop-loss if provided
        if new_stop_price is not None:
            stop_result = await self.update_stop_loss(symbol, new_stop_price)
            results["stop_loss_result"] = stop_result
            if not stop_result["success"]:
                results["success"] = False
                results["errors"].append(f"Stop-loss update failed: {stop_result.get('error')}")
        
        # Update take-profit if provided
        if new_take_profit_price is not None:
            tp_result = await self.update_take_profit(symbol, new_take_profit_price)
            results["take_profit_result"] = tp_result
            if not tp_result["success"]:
                results["success"] = False
                results["errors"].append(f"Take-profit update failed: {tp_result.get('error')}")
        
        if results["success"]:
            self.logger.info(f"✅ Successfully updated position parameters for {symbol}")
        else:
            self.logger.error(f"❌ Failed to update some parameters for {symbol}: {results['errors']}")
        
        return results
    
    async def get_position_with_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed position information including associated orders
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with position and order details
        """
        try:
            # Get position
            positions = await self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not position:
                return {
                    "symbol": symbol,
                    "has_position": False,
                    "position": None,
                    "orders": []
                }
            
            # Get associated orders
            all_orders = await self.get_orders()
            position_orders = [order for order in all_orders if order["symbol"] == symbol]
            
            # Categorize orders
            stop_orders = [order for order in position_orders if "stop" in order.get("order_type", "").lower()]
            limit_orders = [order for order in position_orders if order.get("order_type") == "limit"]
            
            # Calculate unrealized P&L percentage
            current_price = await self.get_current_price(symbol)
            unrealized_pl_pct = position["unrealized_plpc"] if position.get("unrealized_plpc") else 0
            
            return {
                "symbol": symbol,
                "has_position": True,
                "position": {
                    **position,
                    "current_price": current_price,
                    "unrealized_pl_pct": unrealized_pl_pct
                },
                "orders": {
                    "stop_loss_orders": stop_orders,
                    "take_profit_orders": limit_orders,
                    "all_orders": position_orders
                },
                "trailing_eligible": {
                    "can_trail_up": position["quantity"] > 0 and unrealized_pl_pct > 0,  # Long position in profit
                    "can_trail_down": position["quantity"] < 0 and unrealized_pl_pct > 0,  # Short position in profit
                    "current_profit_pct": unrealized_pl_pct
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position details for {symbol}: {e}")
            return {
                "symbol": symbol,
                "has_position": False,
                "position": None,
                "orders": [],
                "error": str(e)
            }
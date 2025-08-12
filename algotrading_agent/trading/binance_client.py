"""
Binance Spot API Client for Crypto Trading Integration

Provides async interface for Binance Spot trading with free tier support:
- 1200 requests/minute (free tier)
- Real-time market data
- Comprehensive order management
- Error handling and rate limiting
"""

import asyncio
import logging
import hashlib
import hmac
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp

from ..components.decision_engine import TradingPair


class BinanceClient:
    """
    Binance Spot API client with async support and rate limiting.
    
    Free tier features:
    - 1200 requests per minute
    - Real-time price data
    - Order management
    - Account information
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.testnet = config.get('testnet', True)  # Start with testnet
        
        # API endpoints
        if self.testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"
            
        self.logger = logging.getLogger("algotrading.binance_client")
        self.logger.info(f"Initialized Binance client (testnet={self.testnet})")
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 1200)  # requests per minute
        self.request_count = 0
        self.rate_limit_reset = time.time() + 60
        
        # Session for connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time >= self.rate_limit_reset:
            self.request_count = 0
            self.rate_limit_reset = current_time + 60
            
        # Check if we're at the limit
        if self.request_count >= self.rate_limit:
            wait_time = self.rate_limit_reset - current_time
            self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.rate_limit_reset = time.time() + 60
            
        self.request_count += 1
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to Binance API with rate limiting and error handling"""
        await self._check_rate_limit()
        
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        
        if params is None:
            params = {}
            
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        try:
            async with self.session.request(
                method, url, params=params, headers=headers, timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"Binance API error {response.status}: {error_text}")
                    raise Exception(f"Binance API error: {response.status} - {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            response = await self._make_request('GET', '/v3/account', signed=True)
            
            # Calculate total balance in USDT equivalent
            total_value = 0.0
            balances = []
            
            for balance in response.get('balances', []):
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:  # Only include non-zero balances
                    balances.append({
                        'asset': balance['asset'],
                        'free': free,
                        'locked': locked,
                        'total': total
                    })
                    
                    # For simplification, assume USDT = $1, BTC price lookup would be needed for real calculation
                    if balance['asset'] == 'USDT':
                        total_value += total
                    elif balance['asset'] == 'BTC':
                        # Would need price lookup in real implementation
                        btc_price = await self.get_current_price('BTCUSDT')
                        if btc_price:
                            total_value += total * btc_price
            
            return {
                'account_type': response.get('accountType', 'SPOT'),
                'can_trade': response.get('canTrade', False),
                'can_withdraw': response.get('canWithdraw', False),
                'can_deposit': response.get('canDeposit', False),
                'portfolio_value': total_value,
                'balances': balances,
                'maker_commission': response.get('makerCommission', 10),
                'taker_commission': response.get('takerCommission', 10),
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions (non-zero balances)"""
        try:
            account_info = await self.get_account_info()
            positions = []
            
            for balance in account_info['balances']:
                if balance['total'] > 0 and balance['asset'] != 'USDT':
                    # Get current price to calculate market value
                    symbol = f"{balance['asset']}USDT"
                    current_price = await self.get_current_price(symbol)
                    
                    if current_price:
                        market_value = balance['total'] * current_price
                        
                        positions.append({
                            'symbol': symbol,
                            'asset': balance['asset'],
                            'quantity': balance['total'],
                            'market_value': market_value,
                            'current_price': current_price,
                            'side': 'long'  # Spot trading is always long
                        })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            response = await self._make_request('GET', '/v3/ticker/price', {
                'symbol': symbol.upper()
            })
            return float(response['price'])
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book depth"""
        try:
            response = await self._make_request('GET', '/v3/depth', {
                'symbol': symbol.upper(),
                'limit': limit
            })
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return {}
    
    async def submit_order(self, trading_pair: TradingPair) -> Dict[str, Any]:
        """Submit a trading order"""
        try:
            # Convert stock trading pair to crypto format
            symbol = self._convert_symbol_format(trading_pair.symbol)
            side = 'BUY' if trading_pair.action.upper() == 'BUY' else 'SELL'
            
            # For simplicity, using market orders initially
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': trading_pair.shares,
                'newOrderRespType': 'FULL'
            }
            
            response = await self._make_request('POST', '/v3/order', order_params, signed=True)
            
            return {
                'success': True,
                'order_id': response['orderId'],
                'client_order_id': response['clientOrderId'],
                'symbol': response['symbol'],
                'status': response['status'],
                'filled_quantity': float(response.get('executedQty', 0)),
                'fills': response.get('fills', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            response = await self._make_request('DELETE', '/v3/order', {
                'symbol': symbol.upper(),
                'orderId': order_id
            }, signed=True)
            
            return {
                'success': True,
                'order_id': response['orderId'],
                'status': response['status']
            }
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            response = await self._make_request('GET', '/v3/order', {
                'symbol': symbol.upper(),
                'orderId': order_id
            }, signed=True)
            
            return {
                'order_id': response['orderId'],
                'symbol': response['symbol'],
                'status': response['status'],
                'side': response['side'],
                'type': response['type'],
                'quantity': float(response['origQty']),
                'filled_quantity': float(response['executedQty']),
                'price': float(response.get('price', 0)),
                'stop_price': float(response.get('stopPrice', 0)),
                'time': response['time']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {}
    
    async def get_trade_history(self, symbol: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            endpoint = '/v3/myTrades'
            params = {'limit': limit}
            
            if symbol:
                params['symbol'] = symbol.upper()
                
            response = await self._make_request('GET', endpoint, params, signed=True)
            
            trades = []
            for trade in response:
                trades.append({
                    'id': trade['id'],
                    'order_id': trade['orderId'],
                    'symbol': trade['symbol'],
                    'side': 'BUY' if trade['isBuyer'] else 'SELL',
                    'quantity': float(trade['qty']),
                    'price': float(trade['price']),
                    'commission': float(trade['commission']),
                    'commission_asset': trade['commissionAsset'],
                    'time': trade['time']
                })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert stock symbol to crypto trading pair format"""
        # This is a simplified conversion - in reality, you'd need a mapping
        # For now, assume symbols ending with common crypto suffixes
        if symbol.upper().endswith('USDT'):
            return symbol.upper()
        elif symbol.upper() in ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL']:
            return f"{symbol.upper()}USDT"
        else:
            # For stocks, might not be directly tradeable on Binance
            self.logger.warning(f"Symbol {symbol} may not be available on Binance")
            return f"{symbol.upper()}USDT"  # Fallback
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information and trading rules"""
        try:
            response = await self._make_request('GET', '/v3/exchangeInfo')
            
            for sym_info in response.get('symbols', []):
                if sym_info['symbol'] == symbol.upper():
                    return {
                        'symbol': sym_info['symbol'],
                        'status': sym_info['status'],
                        'base_asset': sym_info['baseAsset'],
                        'quote_asset': sym_info['quoteAsset'],
                        'filters': sym_info['filters'],
                        'permissions': sym_info.get('permissions', [])
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> Dict[str, Any]:
        """Close a position by selling the specified percentage"""
        try:
            # Get current position
            positions = await self.get_positions()
            position = None
            
            for pos in positions:
                if pos['symbol'] == symbol or pos['asset'] == symbol:
                    position = pos
                    break
            
            if not position:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}'
                }
            
            # Calculate quantity to sell
            sell_quantity = position['quantity'] * (percentage / 100.0)
            
            # Create sell order
            trading_pair = TradingPair(
                symbol=position['symbol'],
                action='SELL',
                shares=sell_quantity,
                confidence=1.0,
                stop_loss=None,
                take_profit=None
            )
            
            return await self.submit_order(trading_pair)
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_24hr_stats(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        try:
            response = await self._make_request('GET', '/v3/ticker/24hr', {
                'symbol': symbol.upper()
            })
            
            return {
                'symbol': response['symbol'],
                'price_change': float(response['priceChange']),
                'price_change_percent': float(response['priceChangePercent']),
                'weighted_avg_price': float(response['weightedAvgPrice']),
                'prev_close_price': float(response['prevClosePrice']),
                'last_price': float(response['lastPrice']),
                'bid_price': float(response['bidPrice']),
                'ask_price': float(response['askPrice']),
                'open_price': float(response['openPrice']),
                'high_price': float(response['highPrice']),
                'low_price': float(response['lowPrice']),
                'volume': float(response['volume']),
                'quote_volume': float(response['quoteVolume']),
                'count': int(response['count'])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting 24hr stats for {symbol}: {e}")
            return {}
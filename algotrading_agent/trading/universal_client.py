"""
Universal Trading Client - Multi-Asset Trading Interface

Routes trading operations across multiple asset classes and exchanges:
- Alpaca (stocks, ETFs)
- Binance (crypto)
- Coinbase (crypto)

Provides unified interface for the Enhanced Trade Manager while handling
asset-specific routing, validation, and optimization.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from .alpaca_client import AlpacaClient
from .binance_client import BinanceClient
from ..components.decision_engine import TradingPair


class AssetClass(Enum):
    """Asset class enumeration for routing decisions"""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"


class ExchangeType(Enum):
    """Exchange type enumeration"""
    ALPACA = "alpaca"
    BINANCE = "binance" 
    COINBASE = "coinbase"


class UniversalTradingClient:
    """
    Universal trading client that routes orders to appropriate exchanges
    based on asset class and optimization criteria.
    
    Features:
    - Automatic asset class detection
    - Intelligent exchange routing
    - Unified position management
    - Cross-asset portfolio optimization
    - Failover and redundancy
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("algotrading.universal_client")
        
        # Initialize exchange clients
        self._initialize_clients()
        
        # Asset class routing rules
        self.routing_rules = self._setup_routing_rules()
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.exchange_performance = {
            ExchangeType.ALPACA: {"trades": 0, "successes": 0},
            ExchangeType.BINANCE: {"trades": 0, "successes": 0},
            ExchangeType.COINBASE: {"trades": 0, "successes": 0}
        }
        
    def _initialize_clients(self):
        """Initialize all available exchange clients"""
        self.clients = {}
        
        # Initialize Alpaca client for stocks
        try:
            alpaca_config = self.config.get('alpaca', {})
            if alpaca_config.get('enabled', True):
                self.clients[ExchangeType.ALPACA] = AlpacaClient(alpaca_config)
                self.logger.info("✅ Alpaca client initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Alpaca client: {e}")
            
        # Initialize Binance client for crypto
        try:
            binance_config = self.config.get('binance', {})
            if binance_config.get('enabled', False):
                self.clients[ExchangeType.BINANCE] = BinanceClient(binance_config)
                self.logger.info("✅ Binance client initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Binance client: {e}")
            
        # TODO: Initialize Coinbase client
        # coinbase_config = self.config.get('coinbase', {})
        # if coinbase_config.get('enabled', False):
        #     self.clients[ExchangeType.COINBASE] = CoinbaseClient(coinbase_config)
        
        self.logger.info(f"Universal client initialized with {len(self.clients)} exchanges")
    
    def _setup_routing_rules(self) -> Dict[AssetClass, List[ExchangeType]]:
        """Setup routing rules for different asset classes"""
        return {
            AssetClass.STOCK: [ExchangeType.ALPACA],
            AssetClass.ETF: [ExchangeType.ALPACA],
            AssetClass.CRYPTO: [ExchangeType.BINANCE, ExchangeType.COINBASE],
            AssetClass.FOREX: []  # Not supported yet
        }
    
    def detect_asset_class(self, symbol: str) -> AssetClass:
        """
        Detect asset class from symbol format
        
        Examples:
        - AAPL -> STOCK
        - SPY -> ETF  
        - BTCUSDT -> CRYPTO
        - BTC -> CRYPTO
        """
        symbol_upper = symbol.upper()
        
        # Crypto patterns
        crypto_patterns = [
            'USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL', 
            'MATIC', 'AVAX', 'ATOM', 'LUNA', 'FTM', 'NEAR', 'ICP', 'VET',
            'LINK', 'UNI', 'AAVE', 'COMP', 'MKR', 'YFI', 'SUSHI', 'CRV'
        ]
        
        # Check if it's a crypto trading pair (ends with USDT, USDC, etc.)
        for pattern in ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH']:
            if symbol_upper.endswith(pattern):
                return AssetClass.CRYPTO
                
        # Check if it's a standalone crypto symbol
        for crypto in crypto_patterns:
            if symbol_upper == crypto:
                return AssetClass.CRYPTO
        
        # Common ETF patterns
        etf_patterns = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SLV', 'VTI', 'VOO']
        if symbol_upper in etf_patterns or 'ETF' in symbol_upper:
            return AssetClass.ETF
            
        # Default to stock
        return AssetClass.STOCK
    
    def get_optimal_exchange(self, asset_class: AssetClass, symbol: str) -> Optional[ExchangeType]:
        """
        Get optimal exchange for trading based on:
        - Asset class routing rules
        - Exchange availability
        - Historical performance
        - Current market conditions
        """
        available_exchanges = self.routing_rules.get(asset_class, [])
        
        # Filter by available clients
        available_exchanges = [
            exchange for exchange in available_exchanges 
            if exchange in self.clients
        ]
        
        if not available_exchanges:
            self.logger.error(f"No available exchanges for {asset_class.value}")
            return None
            
        # For now, use first available exchange
        # TODO: Implement intelligent routing based on:
        # - Fees and spread comparison
        # - Liquidity analysis
        # - Historical performance
        # - Current system load
        
        return available_exchanges[0]
    
    async def route_trade(self, trading_pair: TradingPair) -> Dict[str, Any]:
        """
        Route trade to optimal exchange based on asset class and conditions
        """
        self.total_trades += 1
        
        try:
            # Detect asset class
            asset_class = self.detect_asset_class(trading_pair.symbol)
            self.logger.info(f"🎯 Routing {trading_pair.symbol} as {asset_class.value}")
            
            # Get optimal exchange
            exchange = self.get_optimal_exchange(asset_class, trading_pair.symbol)
            if not exchange:
                raise Exception(f"No available exchange for {asset_class.value}")
                
            # Update performance tracking
            self.exchange_performance[exchange]["trades"] += 1
            
            # Route to appropriate client
            client = self.clients[exchange]
            
            if exchange == ExchangeType.ALPACA:
                result = await self._execute_alpaca_trade(client, trading_pair)
            elif exchange == ExchangeType.BINANCE:
                result = await self._execute_binance_trade(client, trading_pair)
            else:
                raise Exception(f"Unsupported exchange: {exchange}")
            
            # Track success
            if result.get('success', False):
                self.successful_trades += 1
                self.exchange_performance[exchange]["successes"] += 1
                
            return {
                **result,
                'exchange': exchange.value,
                'asset_class': asset_class.value,
                'routing_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.failed_trades += 1
            error_msg = f"Trade routing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'exchange': None,
                'asset_class': None,
                'routing_timestamp': datetime.utcnow().isoformat()
            }
    
    async def _execute_alpaca_trade(self, client: AlpacaClient, trading_pair: TradingPair) -> Dict[str, Any]:
        """Execute trade via Alpaca client"""
        try:
            result = await client.submit_order(trading_pair)
            return {
                'success': True,
                'order_id': result.get('order_id'),
                'message': 'Alpaca order submitted successfully',
                'raw_response': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Alpaca execution failed: {str(e)}"
            }
    
    async def _execute_binance_trade(self, client: BinanceClient, trading_pair: TradingPair) -> Dict[str, Any]:
        """Execute trade via Binance client"""
        try:
            result = await client.submit_order(trading_pair)
            return {
                'success': result.get('success', False),
                'order_id': result.get('order_id'),
                'message': 'Binance order submitted successfully' if result.get('success') else result.get('error'),
                'raw_response': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Binance execution failed: {str(e)}"
            }
    
    async def get_unified_positions(self) -> List[Dict[str, Any]]:
        """Get positions across all exchanges in unified format"""
        all_positions = []
        
        for exchange, client in self.clients.items():
            try:
                positions = await client.get_positions()
                
                for position in positions:
                    # Add exchange metadata
                    position['exchange'] = exchange.value
                    position['asset_class'] = self.detect_asset_class(position['symbol']).value
                    all_positions.append(position)
                    
            except Exception as e:
                self.logger.error(f"Error getting positions from {exchange.value}: {e}")
        
        return all_positions
    
    async def get_unified_account_info(self) -> Dict[str, Any]:
        """Get account information across all exchanges"""
        account_info = {
            'total_portfolio_value': 0.0,
            'total_cash': 0.0,
            'exchanges': {}
        }
        
        for exchange, client in self.clients.items():
            try:
                info = await client.get_account_info()
                account_info['exchanges'][exchange.value] = info
                
                # Aggregate totals (simplified - would need currency conversion in reality)
                account_info['total_portfolio_value'] += info.get('portfolio_value', 0.0)
                account_info['total_cash'] += info.get('cash', 0.0)
                
            except Exception as e:
                self.logger.error(f"Error getting account info from {exchange.value}: {e}")
                account_info['exchanges'][exchange.value] = {'error': str(e)}
        
        return account_info
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price using optimal exchange for the symbol"""
        asset_class = self.detect_asset_class(symbol)
        exchange = self.get_optimal_exchange(asset_class, symbol)
        
        if not exchange or exchange not in self.clients:
            return None
            
        try:
            client = self.clients[exchange]
            return await client.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> Dict[str, Any]:
        """Close position across exchanges"""
        # Determine which exchange holds the position
        positions = await self.get_unified_positions()
        target_position = None
        target_exchange = None
        
        for position in positions:
            if position['symbol'] == symbol:
                target_position = position
                target_exchange = position['exchange']
                break
        
        if not target_position:
            return {
                'success': False,
                'error': f'No position found for {symbol}'
            }
        
        try:
            exchange_type = ExchangeType(target_exchange)
            client = self.clients[exchange_type]
            return await client.close_position(symbol, percentage)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error closing position: {str(e)}'
            }
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel order - requires knowing which exchange it's on"""
        # For now, try each exchange until we find the order
        for exchange, client in self.clients.items():
            try:
                result = await client.cancel_order(symbol, order_id)
                if result.get('success', False):
                    return {
                        **result,
                        'exchange': exchange.value
                    }
            except Exception as e:
                continue  # Try next exchange
        
        return {
            'success': False,
            'error': f'Order {order_id} not found on any exchange'
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics across all exchanges"""
        success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        exchange_stats = {}
        for exchange, stats in self.exchange_performance.items():
            if stats["trades"] > 0:
                exchange_success_rate = (stats["successes"] / stats["trades"] * 100)
                exchange_stats[exchange.value] = {
                    "total_trades": stats["trades"],
                    "successful_trades": stats["successes"],
                    "success_rate": exchange_success_rate
                }
        
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'overall_success_rate': success_rate,
            'exchange_performance': exchange_stats,
            'active_exchanges': list(self.clients.keys())
        }
    
    def get_routing_status(self) -> Dict[str, Any]:
        """Get current routing configuration and status"""
        return {
            'available_exchanges': [exchange.value for exchange in self.clients.keys()],
            'routing_rules': {
                asset_class.value: [exchange.value for exchange in exchanges]
                for asset_class, exchanges in self.routing_rules.items()
            },
            'supported_asset_classes': [asset_class.value for asset_class in self.routing_rules.keys()],
            'performance_stats': self.get_performance_stats()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check across all exchanges"""
        health_status = {
            'overall_healthy': True,
            'exchanges': {}
        }
        
        for exchange, client in self.clients.items():
            try:
                # Try to get account info as health check
                await client.get_account_info()
                health_status['exchanges'][exchange.value] = {
                    'status': 'healthy',
                    'last_check': datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_status['exchanges'][exchange.value] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': datetime.utcnow().isoformat()
                }
                health_status['overall_healthy'] = False
        
        return health_status
"""
Express Execution Manager - High-speed trade execution system

Handles ultra-fast trade execution for momentum patterns and breaking news
with multiple speed lanes: Lightning (<5s), Express (<15s), Fast (<30s).
Optimized for capturing rapid market opportunities with minimal latency.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from ..core.base import ComponentBase
from ..trading.alpaca_client import AlpacaClient
from ..components.decision_engine import TradingPair
from .momentum_pattern_detector import PatternSignal, PatternType
from .breaking_news_velocity_tracker import NewsVelocitySignal, VelocityLevel


class ExecutionLane(Enum):
    """Execution speed lanes"""
    LIGHTNING = "lightning"  # <5 seconds - flash crashes, circuit breakers
    EXPRESS = "express"      # <15 seconds - breaking news, earnings surprises
    FAST = "fast"           # <30 seconds - volume breakouts, momentum
    STANDARD = "standard"    # <60 seconds - normal processing


@dataclass
class ExpressTrade:
    """Express trading order with speed requirements"""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    execution_lane: ExecutionLane
    speed_target_seconds: int
    
    # Triggers
    trigger_type: str  # "pattern", "news_velocity", "manual"
    trigger_data: Any  # PatternSignal or NewsVelocitySignal
    
    # Pricing
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    # Timing
    created_at: datetime = None
    submitted_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    execution_latency_ms: Optional[int] = None
    order_id: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "execution_lane": self.execution_lane.value,
            "speed_target_seconds": self.speed_target_seconds,
            "trigger_type": self.trigger_type,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_latency_ms": self.execution_latency_ms,
            "order_id": self.order_id,
            "success": self.success,
            "error_message": self.error_message
        }


class ExpressExecutionManager(ComponentBase):
    """
    High-speed trade execution manager with multiple speed lanes
    
    Handles rapid execution of trades triggered by momentum patterns and
    breaking news with latency targets as low as 5 seconds for critical opportunities.
    """
    
    def __init__(self, config: Dict[str, Any], alpaca_client: AlpacaClient, enhanced_trade_manager):
        super().__init__("express_execution_manager", config)
        self.alpaca_client = alpaca_client
        self.enhanced_trade_manager = enhanced_trade_manager
        
        # Configuration
        self.max_concurrent_trades = config.get("max_concurrent_trades", 10)
        self.enable_lightning_lane = config.get("enable_lightning_lane", True)
        self.enable_express_lane = config.get("enable_express_lane", True)
        self.enable_fast_lane = config.get("enable_fast_lane", True)
        
        # Speed targets (milliseconds)
        self.speed_targets = {
            ExecutionLane.LIGHTNING: 5000,   # 5 seconds
            ExecutionLane.EXPRESS: 15000,    # 15 seconds
            ExecutionLane.FAST: 30000,       # 30 seconds
            ExecutionLane.STANDARD: 60000    # 60 seconds
        }
        
        # Lane-specific configurations
        self.lane_configs = {
            ExecutionLane.LIGHTNING: {
                "skip_validation": True,      # Minimal validation for speed
                "pre_compute_prices": True,   # Pre-calculate stop/take profit
                "use_market_orders": True,    # Market orders for speed
                "max_position_size": 0.02     # 2% max for high-risk speed trades
            },
            ExecutionLane.EXPRESS: {
                "skip_validation": False,     # Basic validation
                "pre_compute_prices": True,   
                "use_market_orders": True,    
                "max_position_size": 0.03     # 3% max
            },
            ExecutionLane.FAST: {
                "skip_validation": False,     # Full validation
                "pre_compute_prices": False,  # Calculate prices normally
                "use_market_orders": False,   # Can use limit orders
                "max_position_size": 0.05     # 5% max
            },
            ExecutionLane.STANDARD: {
                "skip_validation": False,     
                "pre_compute_prices": False,  
                "use_market_orders": False,   
                "max_position_size": 0.05     # 5% max
            }
        }
        
        # Active trades tracking
        self.active_express_trades: Dict[str, ExpressTrade] = {}  # trade_id -> trade
        self.execution_queue: List[ExpressTrade] = []
        self.processing_trades: bool = False
        
        # Performance tracking
        self.total_express_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.average_latency_by_lane: Dict[ExecutionLane, float] = {}
        self.lane_success_rates: Dict[ExecutionLane, float] = {}
        
        # Pre-computed price cache for speed
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}  # symbol -> (price, timestamp)
        self.cache_expiry_seconds = 5  # 5 second price cache
        
        # Speed optimization
        self.initialized_clients: bool = False
        self.connection_pool_ready: bool = False
        
    async def start(self) -> None:
        """Start express execution manager"""
        self.logger.info("🚀 Starting Express Execution Manager")
        self.is_running = True
        
        # Initialize connection pools for speed
        await self._initialize_speed_optimizations()
        
        # Start background processing
        asyncio.create_task(self._process_execution_queue())
        
        self.logger.info(f"✅ Express execution manager started with {len(self.speed_targets)} speed lanes")
    
    async def stop(self) -> None:
        """Stop express execution manager"""
        self.logger.info("🛑 Stopping Express Execution Manager")
        self.is_running = False
        
        # Complete any active trades
        await self._complete_remaining_trades()
        
        self._log_performance_summary()
    
    async def _initialize_speed_optimizations(self):
        """Initialize optimizations for high-speed execution"""
        try:
            # Pre-warm connection pools
            self.logger.info("🔧 Initializing speed optimizations...")
            
            # Test connectivity
            await self.alpaca_client.get_account()
            self.connection_pool_ready = True
            
            # Pre-cache common symbols
            common_symbols = ["SPY", "QQQ", "AAPL", "TSLA", "BTCUSD"]
            for symbol in common_symbols:
                price = await self.alpaca_client.get_current_price(symbol)
                if price:
                    self.price_cache[symbol] = (price, datetime.utcnow())
            
            self.initialized_clients = True
            self.logger.info("✅ Speed optimizations initialized")
            
        except Exception as e:
            self.logger.error(f"Speed optimization initialization failed: {e}")
            self.connection_pool_ready = False
    
    async def execute_pattern_trade(self, pattern_signal: PatternSignal) -> Optional[ExpressTrade]:
        """Execute trade based on momentum pattern detection"""
        if not self.is_running:
            return None
        
        try:
            # Determine execution lane based on pattern type
            execution_lane = self._get_pattern_execution_lane(pattern_signal.pattern_type)
            
            if not self._is_lane_enabled(execution_lane):
                self.logger.warning(f"Execution lane {execution_lane.value} is disabled")
                return None
            
            # Determine trade direction
            side = "buy" if pattern_signal.direction == "bullish" else "sell"
            
            # Calculate position size based on lane config
            lane_config = self.lane_configs[execution_lane]
            max_position_pct = lane_config["max_position_size"]
            
            # Get account value for position sizing
            account = await self.alpaca_client.get_account()
            account_value = float(account.portfolio_value) if account else 100000
            
            # Calculate quantity
            current_price = await self._get_fast_price(pattern_signal.symbol)
            if not current_price:
                self.logger.error(f"Cannot get price for {pattern_signal.symbol}")
                return None
            
            position_value = account_value * max_position_pct
            quantity = max(1, int(position_value / current_price))
            
            # Create express trade
            express_trade = ExpressTrade(
                symbol=pattern_signal.symbol,
                side=side,
                quantity=quantity,
                execution_lane=execution_lane,
                speed_target_seconds=self.speed_targets[execution_lane] // 1000,
                trigger_type="pattern",
                trigger_data=pattern_signal,
                entry_price=current_price
            )
            
            # Pre-compute protective prices if configured
            if lane_config.get("pre_compute_prices", False):
                express_trade.stop_loss_price, express_trade.take_profit_price = self._pre_compute_protective_prices(
                    current_price, side, pattern_signal.volatility
                )
            
            # Add to execution queue
            self.execution_queue.append(express_trade)
            self.total_express_trades += 1
            
            self.logger.info(f"🚀 Queued {execution_lane.value} trade: {pattern_signal.symbol} {side} "
                           f"(pattern: {pattern_signal.pattern_type.value}, target: {express_trade.speed_target_seconds}s)")
            
            return express_trade
            
        except Exception as e:
            self.logger.error(f"Error creating pattern trade for {pattern_signal.symbol}: {e}")
            return None
    
    async def execute_news_velocity_trade(self, velocity_signal: NewsVelocitySignal) -> Optional[ExpressTrade]:
        """Execute trade based on breaking news velocity"""
        if not self.is_running or not velocity_signal.symbols:
            return None
        
        try:
            # Use the first symbol (most relevant)
            symbol = velocity_signal.symbols[0]
            
            # Determine execution lane based on velocity level
            execution_lane = self._get_velocity_execution_lane(velocity_signal.velocity_level)
            
            if not self._is_lane_enabled(execution_lane):
                return None
            
            # Determine trade direction based on sentiment and velocity
            if velocity_signal.financial_impact_score > 0.5:
                side = "buy"  # Positive news typically bullish
            else:
                side = "sell"  # Negative/uncertain news potentially bearish
            
            # Calculate position size
            lane_config = self.lane_configs[execution_lane]
            max_position_pct = lane_config["max_position_size"]
            
            account = await self.alpaca_client.get_account()
            account_value = float(account.portfolio_value) if account else 100000
            
            current_price = await self._get_fast_price(symbol)
            if not current_price:
                return None
            
            position_value = account_value * max_position_pct
            quantity = max(1, int(position_value / current_price))
            
            # Create express trade
            express_trade = ExpressTrade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                execution_lane=execution_lane,
                speed_target_seconds=self.speed_targets[execution_lane] // 1000,
                trigger_type="news_velocity",
                trigger_data=velocity_signal,
                entry_price=current_price
            )
            
            # Pre-compute protective prices for velocity-based trades
            express_trade.stop_loss_price, express_trade.take_profit_price = self._pre_compute_protective_prices(
                current_price, side, velocity_signal.velocity_score / 10  # Normalize velocity to volatility
            )
            
            self.execution_queue.append(express_trade)
            self.total_express_trades += 1
            
            self.logger.info(f"⚡ Queued {execution_lane.value} velocity trade: {symbol} {side} "
                           f"(velocity: {velocity_signal.velocity_level.value}, score: {velocity_signal.velocity_score:.1f})")
            
            return express_trade
            
        except Exception as e:
            self.logger.error(f"Error creating velocity trade: {e}")
            return None
    
    def _get_pattern_execution_lane(self, pattern_type: PatternType) -> ExecutionLane:
        """Determine execution lane based on pattern type"""
        if pattern_type in [PatternType.FLASH_CRASH, PatternType.FLASH_SURGE]:
            return ExecutionLane.LIGHTNING
        elif pattern_type in [PatternType.EARNINGS_SURPRISE, PatternType.NEWS_VELOCITY_SPIKE]:
            return ExecutionLane.EXPRESS
        elif pattern_type in [PatternType.VOLUME_BREAKOUT, PatternType.REVERSAL_PATTERN]:
            return ExecutionLane.FAST
        else:
            return ExecutionLane.STANDARD
    
    def _get_velocity_execution_lane(self, velocity_level: VelocityLevel) -> ExecutionLane:
        """Determine execution lane based on news velocity level"""
        if velocity_level == VelocityLevel.VIRAL:
            return ExecutionLane.LIGHTNING
        elif velocity_level == VelocityLevel.BREAKING:
            return ExecutionLane.EXPRESS
        elif velocity_level == VelocityLevel.TRENDING:
            return ExecutionLane.FAST
        else:
            return ExecutionLane.STANDARD
    
    def _is_lane_enabled(self, lane: ExecutionLane) -> bool:
        """Check if execution lane is enabled"""
        if lane == ExecutionLane.LIGHTNING:
            return self.enable_lightning_lane
        elif lane == ExecutionLane.EXPRESS:
            return self.enable_express_lane
        elif lane == ExecutionLane.FAST:
            return self.enable_fast_lane
        else:
            return True  # Standard lane always enabled
    
    async def _get_fast_price(self, symbol: str) -> Optional[float]:
        """Get price with caching for speed optimization"""
        current_time = datetime.utcnow()
        
        # Check cache first
        if symbol in self.price_cache:
            cached_price, cached_time = self.price_cache[symbol]
            if (current_time - cached_time).total_seconds() < self.cache_expiry_seconds:
                return cached_price
        
        # Get fresh price
        try:
            price = await self.alpaca_client.get_current_price(symbol)
            if price:
                self.price_cache[symbol] = (price, current_time)
            return price
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _pre_compute_protective_prices(self, entry_price: float, side: str, volatility: float) -> Tuple[Optional[float], Optional[float]]:
        """Pre-compute stop-loss and take-profit prices for speed"""
        try:
            # Adjust stops based on volatility
            base_stop_pct = 0.02  # 2% base stop
            base_profit_pct = 0.04  # 4% base profit target
            
            # Adjust for volatility
            stop_pct = base_stop_pct + (volatility * 0.5)  # Wider stops for volatile assets
            profit_pct = base_profit_pct + (volatility * 1.0)  # Higher targets for volatile assets
            
            if side == "buy":
                stop_loss_price = round(entry_price * (1 - stop_pct), 2)
                take_profit_price = round(entry_price * (1 + profit_pct), 2)
            else:  # sell/short
                stop_loss_price = round(entry_price * (1 + stop_pct), 2)
                take_profit_price = round(entry_price * (1 - profit_pct), 2)
            
            return stop_loss_price, take_profit_price
            
        except Exception as e:
            self.logger.error(f"Error pre-computing protective prices: {e}")
            return None, None
    
    async def _process_execution_queue(self):
        """Background task to process execution queue"""
        while self.is_running:
            try:
                if not self.processing_trades and self.execution_queue:
                    self.processing_trades = True
                    await self._execute_queued_trades()
                    self.processing_trades = False
                
                await asyncio.sleep(0.1)  # Very fast polling for low latency
                
            except Exception as e:
                self.logger.error(f"Execution queue processing error: {e}")
                self.processing_trades = False
                await asyncio.sleep(1)
    
    async def _execute_queued_trades(self):
        """Execute all queued trades with speed optimization"""
        if not self.execution_queue:
            return
        
        # Sort by speed target (fastest first)
        self.execution_queue.sort(key=lambda t: t.speed_target_seconds)
        
        # Process trades concurrently within limits
        concurrent_limit = min(len(self.execution_queue), self.max_concurrent_trades)
        trades_to_process = self.execution_queue[:concurrent_limit]
        self.execution_queue = self.execution_queue[concurrent_limit:]
        
        # Execute trades concurrently
        execution_tasks = [self._execute_single_trade(trade) for trade in trades_to_process]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            trade = trades_to_process[i]
            if isinstance(result, Exception):
                self.logger.error(f"Trade execution exception for {trade.symbol}: {result}")
                trade.success = False
                trade.error_message = str(result)
                self.failed_trades += 1
            else:
                if trade.success:
                    self.successful_trades += 1
                else:
                    self.failed_trades += 1
    
    async def _execute_single_trade(self, trade: ExpressTrade) -> ExpressTrade:
        """Execute a single express trade with timing"""
        start_time = time.time()
        trade.submitted_at = datetime.utcnow()
        
        try:
            lane_config = self.lane_configs[trade.execution_lane]
            
            # Create trading pair for enhanced trade manager
            trading_pair = TradingPair(
                symbol=trade.symbol,
                action=trade.side,
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss_price,
                take_profit=trade.take_profit_price,
                confidence=0.8,  # High confidence for express trades
                reasoning=f"Express {trade.execution_lane.value} trade triggered by {trade.trigger_type}"
            )
            
            # Execute through enhanced trade manager with speed optimizations
            if lane_config.get("skip_validation", False):
                # Skip some validations for lightning speed
                result = await self.enhanced_trade_manager.execute_trade_express(trading_pair)
            else:
                # Normal execution path
                result = await self.enhanced_trade_manager.execute_trade(trading_pair)
            
            # Record execution time
            execution_time = time.time() - start_time
            trade.execution_latency_ms = int(execution_time * 1000)
            trade.executed_at = datetime.utcnow()
            
            if result and result.get("success", False):
                trade.success = True
                trade.order_id = result.get("order_id")
                
                self.logger.info(f"✅ {trade.execution_lane.value.upper()} trade executed: {trade.symbol} "
                               f"in {trade.execution_latency_ms}ms (target: {trade.speed_target_seconds}s)")
            else:
                trade.success = False
                trade.error_message = result.get("error", "Unknown execution error") if result else "No result"
                
                self.logger.error(f"❌ {trade.execution_lane.value.upper()} trade failed: {trade.symbol} "
                                f"- {trade.error_message}")
            
            # Update performance tracking
            self._update_lane_performance(trade.execution_lane, trade.execution_latency_ms, trade.success)
            
            trade.completed_at = datetime.utcnow()
            return trade
            
        except Exception as e:
            trade.success = False
            trade.error_message = str(e)
            trade.execution_latency_ms = int((time.time() - start_time) * 1000)
            trade.completed_at = datetime.utcnow()
            
            self.logger.error(f"❌ Express trade execution error for {trade.symbol}: {e}")
            return trade
    
    def _update_lane_performance(self, lane: ExecutionLane, latency_ms: int, success: bool):
        """Update performance tracking for execution lane"""
        # Update average latency
        if lane not in self.average_latency_by_lane:
            self.average_latency_by_lane[lane] = latency_ms
        else:
            # Simple moving average
            self.average_latency_by_lane[lane] = (self.average_latency_by_lane[lane] + latency_ms) / 2
        
        # Update success rate (simplified tracking)
        if lane not in self.lane_success_rates:
            self.lane_success_rates[lane] = 1.0 if success else 0.0
        else:
            # Simple moving average
            current_rate = self.lane_success_rates[lane]
            new_rate = 1.0 if success else 0.0
            self.lane_success_rates[lane] = (current_rate + new_rate) / 2
    
    async def _complete_remaining_trades(self):
        """Complete any remaining trades during shutdown"""
        if self.execution_queue:
            self.logger.info(f"Completing {len(self.execution_queue)} remaining trades...")
            await self._execute_queued_trades()
    
    def get_active_trades(self) -> List[ExpressTrade]:
        """Get all active express trades"""
        return list(self.active_express_trades.values())
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get execution queue status"""
        return {
            "queued_trades": len(self.execution_queue),
            "processing": self.processing_trades,
            "active_trades": len(self.active_express_trades)
        }
    
    def _log_performance_summary(self):
        """Log express execution performance summary"""
        self.logger.info("🚀 EXPRESS EXECUTION MANAGER PERFORMANCE:")
        self.logger.info(f"  Total express trades: {self.total_express_trades}")
        self.logger.info(f"  Successful trades: {self.successful_trades}")
        self.logger.info(f"  Failed trades: {self.failed_trades}")
        
        if self.total_express_trades > 0:
            success_rate = (self.successful_trades / self.total_express_trades) * 100
            self.logger.info(f"  Overall success rate: {success_rate:.1f}%")
        
        for lane, latency in self.average_latency_by_lane.items():
            success_rate = self.lane_success_rates.get(lane, 0) * 100
            self.logger.info(f"  {lane.value}: {latency:.0f}ms avg, {success_rate:.1f}% success")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            "is_running": self.is_running,
            "connection_ready": self.connection_pool_ready,
            "total_express_trades": self.total_express_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "queued_trades": len(self.execution_queue),
            "processing_trades": self.processing_trades,
            "lane_performance": {
                lane.value: {
                    "avg_latency_ms": self.average_latency_by_lane.get(lane, 0),
                    "success_rate": self.lane_success_rates.get(lane, 0),
                    "enabled": self._is_lane_enabled(lane)
                }
                for lane in ExecutionLane
            },
            "speed_targets": {lane.value: target for lane, target in self.speed_targets.items()},
            "cached_prices": len(self.price_cache)
        }
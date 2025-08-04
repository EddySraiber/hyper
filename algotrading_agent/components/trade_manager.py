from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from ..core.base import PersistentComponent
from ..trading.alpaca_client import AlpacaClient
from .decision_engine import TradingPair


class TradeStatus(Enum):
    PENDING_ENTRY = "pending_entry"      # Waiting for entry order to fill
    ACTIVE_POSITION = "active_position"  # Position is open, monitoring for exit
    PENDING_EXIT = "pending_exit"        # Exit order submitted, waiting for fill
    COMPLETED = "completed"              # Trade fully completed
    FAILED = "failed"                    # Trade failed, needs cleanup
    CANCELLED = "cancelled"              # Trade manually cancelled


class ActiveTrade:
    def __init__(self, trading_pair: TradingPair, config: Dict[str, Any]):
        self.id = f"{trading_pair.symbol}_{int(datetime.utcnow().timestamp())}"
        self.symbol = trading_pair.symbol
        self.action = trading_pair.action  # "buy" or "sell"
        self.quantity = trading_pair.quantity
        
        # Entry details
        self.entry_target_price = trading_pair.entry_price
        self.entry_order_id: Optional[str] = None
        self.actual_entry_price: Optional[float] = None
        self.entry_filled_at: Optional[datetime] = None
        
        # Exit details
        self.exit_target_price = trading_pair.take_profit
        self.stop_loss_price = trading_pair.stop_loss
        self.exit_order_id: Optional[str] = None
        self.actual_exit_price: Optional[float] = None
        self.exit_filled_at: Optional[datetime] = None
        
        # Status tracking
        self.status = TradeStatus.PENDING_ENTRY
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self.retry_count = 0
        self.max_retries = config.get("max_retries", 3)
        
        # Trade metadata
        self.confidence = trading_pair.confidence
        self.reasoning = trading_pair.reasoning
        self.price_flexibility_pct = config.get("price_flexibility_pct", 0.01)  # 1% default
        
        # Performance tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "entry_target_price": self.entry_target_price,
            "entry_order_id": self.entry_order_id,
            "actual_entry_price": self.actual_entry_price,
            "entry_filled_at": self.entry_filled_at.isoformat() if self.entry_filled_at else None,
            "exit_target_price": self.exit_target_price,
            "stop_loss_price": self.stop_loss_price,
            "exit_order_id": self.exit_order_id,
            "actual_exit_price": self.actual_exit_price,
            "exit_filled_at": self.exit_filled_at.isoformat() if self.exit_filled_at else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "retry_count": self.retry_count,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl
        }
        
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current market price"""
        if self.actual_entry_price and self.status == TradeStatus.ACTIVE_POSITION:
            if self.action == "buy":
                self.unrealized_pnl = (current_price - self.actual_entry_price) * self.quantity
            else:  # sell (short)
                self.unrealized_pnl = (self.actual_entry_price - current_price) * self.quantity
        self.last_updated = datetime.utcnow()


class TradeManager(PersistentComponent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("trade_manager", config)
        
        # Configuration
        self.polling_interval = config.get("polling_interval", 60)  # seconds
        self.max_active_trades = config.get("max_active_trades", 10)
        self.price_flexibility_pct = config.get("price_flexibility_pct", 0.01)
        self.trade_timeout_hours = config.get("trade_timeout_hours", 24)
        
        # Trade queue storage
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.completed_trades: List[Dict[str, Any]] = []
        self.failed_trades: List[Dict[str, Any]] = []
        
        # External dependencies (injected by main app)
        self.alpaca_client: Optional[AlpacaClient] = None
        self.decision_engine = None  # For feedback on failed trades
        
        # Failure tracking
        self.failure_feedback: List[Dict[str, Any]] = []
        
        # Background task
        self._polling_task: Optional[asyncio.Task] = None
        
    def start(self) -> None:
        """Start the trade manager and polling service"""
        self.logger.info("Starting Trade Manager")
        self.is_running = True
        
        # Load persisted state
        self._load_memory()
        
        # Start background polling
        if not self._polling_task:
            self._polling_task = asyncio.create_task(self._polling_loop())
            
    def stop(self) -> None:
        """Stop the trade manager"""
        self.logger.info("Stopping Trade Manager")
        self.is_running = False
        
        # Cancel polling task
        if self._polling_task:
            self._polling_task.cancel()
            self._polling_task = None
            
        # Save state
        self._save_memory()
        
    def add_trade(self, trading_pair: TradingPair) -> bool:
        """Add a new trade to the queue for monitoring"""
        if len(self.active_trades) >= self.max_active_trades:
            self.logger.warning(f"Trade queue full ({self.max_active_trades}), cannot add new trade")
            return False
            
        trade = ActiveTrade(trading_pair, self.config)
        self.active_trades[trade.id] = trade
        
        self.logger.info(f"Added trade to queue: {trade.symbol} {trade.action} @ ${trade.entry_target_price:.2f}")
        self._save_memory()
        return True
        
    def process(self, data=None):
        """Process method required by ComponentBase - returns current queue status"""
        return self.get_queue_status()
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current trade queue status"""
        status_counts = {}
        for status in TradeStatus:
            status_counts[status.value] = sum(1 for t in self.active_trades.values() if t.status == status)
            
        total_unrealized_pnl = sum(t.unrealized_pnl for t in self.active_trades.values())
        
        return {
            "active_trades": len(self.active_trades),
            "max_capacity": self.max_active_trades,
            "status_breakdown": status_counts,
            "total_unrealized_pnl": total_unrealized_pnl,
            "completed_today": len([t for t in self.completed_trades 
                                  if datetime.fromisoformat(t["exit_filled_at"]).date() == datetime.utcnow().date()]),
            "failed_today": len([t for t in self.failed_trades 
                               if datetime.fromisoformat(t["last_updated"]).date() == datetime.utcnow().date()]),
            "pending_feedback": len(self.failure_feedback)
        }
        
    async def _polling_loop(self):
        """Background polling loop to monitor active trades"""
        self.logger.info(f"Starting trade polling loop (interval: {self.polling_interval}s)")
        
        while self.is_running:
            try:
                await self._poll_active_trades()
                await asyncio.sleep(self.polling_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(self.polling_interval)
                
        self.logger.info("Trade polling loop stopped")
        
    async def _poll_active_trades(self):
        """Poll all active trades for status updates"""
        if not self.alpaca_client or not self.active_trades:
            return
            
        # Check market hours - skip polling if markets closed
        try:
            market_open = await self.alpaca_client.is_market_open()
            if not market_open:
                return  # Skip polling during market closure
        except Exception as e:
            self.logger.warning(f"Could not check market status for polling: {e}")
            return
            
        self.logger.debug(f"Polling {len(self.active_trades)} active trades")
        
        trades_to_remove = []
        
        for trade_id, trade in self.active_trades.items():
            try:
                updated = await self._update_trade_status(trade)
                
                # Clean up completed/failed trades
                if trade.status in [TradeStatus.COMPLETED, TradeStatus.FAILED, TradeStatus.CANCELLED]:
                    trades_to_remove.append(trade_id)
                    
            except Exception as e:
                self.logger.error(f"Error updating trade {trade_id}: {e}")
                
        # Remove completed/failed trades from active queue
        for trade_id in trades_to_remove:
            trade = self.active_trades.pop(trade_id)
            
            if trade.status == TradeStatus.COMPLETED:
                self.completed_trades.append(trade.to_dict())
                self.logger.info(f"Trade completed: {trade.symbol} PnL: ${trade.realized_pnl:.2f}")
            elif trade.status == TradeStatus.FAILED:
                self.failed_trades.append(trade.to_dict())
                self.report_trade_failure(trade_id, "Trade execution failed", trade.reasoning)
                self.logger.warning(f"Trade failed: {trade.symbol} - {trade.reasoning}")
                
        if trades_to_remove:
            self._save_memory()
            
    async def _update_trade_status(self, trade: ActiveTrade) -> bool:
        """Update individual trade status - returns True if trade was updated"""
        # Check for timeout
        if datetime.utcnow() - trade.created_at > timedelta(hours=self.trade_timeout_hours):
            self.logger.warning(f"Trade {trade.id} timed out after {self.trade_timeout_hours} hours")
            trade.status = TradeStatus.FAILED
            return True
            
        # Update based on current status
        if trade.status == TradeStatus.PENDING_ENTRY:
            return await self._check_entry_order(trade)
        elif trade.status == TradeStatus.ACTIVE_POSITION:
            return await self._monitor_active_position(trade)
        elif trade.status == TradeStatus.PENDING_EXIT:
            return await self._check_exit_order(trade)
            
        return False
        
    async def _check_entry_order(self, trade: ActiveTrade) -> bool:
        """Check if entry order has filled"""
        if not trade.entry_order_id:
            # Entry order hasn't been submitted yet - this should be handled by main execution flow
            return False
            
        try:
            order_status = await self.alpaca_client.get_order_status(trade.entry_order_id)
            
            if order_status["status"] == "filled":
                trade.actual_entry_price = float(order_status["filled_avg_price"])
                trade.entry_filled_at = datetime.fromisoformat(order_status["filled_at"])
                trade.status = TradeStatus.ACTIVE_POSITION
                
                self.logger.info(f"Entry filled: {trade.symbol} @ ${trade.actual_entry_price:.2f}")
                return True
                
            elif order_status["status"] in ["cancelled", "expired", "rejected"]:
                trade.status = TradeStatus.FAILED
                self.logger.warning(f"Entry order failed: {trade.symbol} - {order_status['status']}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking entry order {trade.entry_order_id}: {e}")
            
        return False
        
    async def _monitor_active_position(self, trade: ActiveTrade) -> bool:
        """Monitor active position and update P&L"""
        try:
            # Get current price and update unrealized P&L
            current_price = await self.alpaca_client.get_current_price(trade.symbol)
            if current_price:
                trade.update_unrealized_pnl(current_price)
                
                # Check if stop-loss or take-profit should be adjusted (future enhancement)
                # This is where dynamic adjustment logic would go
                
        except Exception as e:
            self.logger.error(f"Error monitoring position {trade.symbol}: {e}")
            
        return False  # Position monitoring doesn't change status
        
    async def _check_exit_order(self, trade: ActiveTrade) -> bool:
        """Check if exit order has filled"""
        if not trade.exit_order_id:
            return False
            
        try:
            order_status = await self.alpaca_client.get_order_status(trade.exit_order_id)
            
            if order_status["status"] == "filled":
                trade.actual_exit_price = float(order_status["filled_avg_price"])
                trade.exit_filled_at = datetime.fromisoformat(order_status["filled_at"])
                trade.status = TradeStatus.COMPLETED
                
                # Calculate realized P&L
                if trade.action == "buy":
                    trade.realized_pnl = (trade.actual_exit_price - trade.actual_entry_price) * trade.quantity
                else:  # sell (short)
                    trade.realized_pnl = (trade.actual_entry_price - trade.actual_exit_price) * trade.quantity
                    
                self.logger.info(f"Exit filled: {trade.symbol} @ ${trade.actual_exit_price:.2f} PnL: ${trade.realized_pnl:.2f}")
                return True
                
            elif order_status["status"] in ["cancelled", "expired", "rejected"]:
                trade.status = TradeStatus.FAILED
                self.logger.warning(f"Exit order failed: {trade.symbol} - {order_status['status']}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking exit order {trade.exit_order_id}: {e}")
            
        return False
        
    def _get_memory_data(self) -> Dict[str, Any]:
        """Get data to persist"""
        return {
            "active_trades": {tid: trade.to_dict() for tid, trade in self.active_trades.items()},
            "completed_trades": self.completed_trades[-100:],  # Keep last 100
            "failed_trades": self.failed_trades[-50:]  # Keep last 50
        }
        
    def _set_memory_data(self, data: Dict[str, Any]) -> None:
        """Restore data from persistence"""
        # Restore active trades (convert back to ActiveTrade objects)
        active_data = data.get("active_trades", {})
        self.active_trades = {}
        
        # Note: Full restoration would require reconstructing ActiveTrade objects
        # For now, we'll just restore the basic data
        self.completed_trades = data.get("completed_trades", [])
        self.failed_trades = data.get("failed_trades", [])
        
        self.logger.info(f"Restored {len(self.active_trades)} active trades, "
                        f"{len(self.completed_trades)} completed, "
                        f"{len(self.failed_trades)} failed")
                        
    def report_trade_failure(self, trade_id: str, reason: str, error_details: str = "") -> None:
        """Report a trade failure for feedback to decision engine"""
        if trade_id in self.active_trades:
            trade = self.active_trades[trade_id]
            
            failure_report = {
                "trade_id": trade_id,
                "symbol": trade.symbol,
                "action": trade.action,
                "target_price": trade.entry_target_price,
                "failure_reason": reason,
                "error_details": error_details,
                "failure_time": datetime.utcnow().isoformat(),
                "retry_count": trade.retry_count,
                "confidence": trade.confidence
            }
            
            self.failure_feedback.append(failure_report)
            self.logger.warning(f"Trade failure reported: {trade.symbol} - {reason}")
            
    def get_failure_feedback(self) -> List[Dict[str, Any]]:
        """Get and clear failure feedback for decision engine"""
        feedback = self.failure_feedback.copy()
        self.failure_feedback.clear()
        return feedback
        
    def cancel_trade(self, trade_id: str, reason: str = "Manual cancellation") -> bool:
        """Cancel an active trade"""
        if trade_id not in self.active_trades:
            return False
            
        trade = self.active_trades[trade_id]
        trade.status = TradeStatus.CANCELLED
        
        # Try to cancel any pending orders
        if self.alpaca_client:
            if trade.entry_order_id:
                try:
                    asyncio.create_task(self._cancel_order_if_pending(trade.entry_order_id))
                except Exception as e:
                    self.logger.error(f"Error cancelling entry order {trade.entry_order_id}: {e}")
                    
            if trade.exit_order_id:
                try:
                    asyncio.create_task(self._cancel_order_if_pending(trade.exit_order_id))
                except Exception as e:
                    self.logger.error(f"Error cancelling exit order {trade.exit_order_id}: {e}")
        
        self.logger.info(f"Trade cancelled: {trade.symbol} - {reason}")
        return True
        
    async def _cancel_order_if_pending(self, order_id: str) -> None:
        """Cancel an order if it's still pending"""
        try:
            order_status = await self.alpaca_client.get_order_status(order_id)
            if order_status["status"] in ["new", "partially_filled", "pending_new"]:
                await self.alpaca_client.cancel_order(order_id)
                self.logger.info(f"Cancelled pending order: {order_id}")
        except Exception as e:
            self.logger.error(f"Error checking/cancelling order {order_id}: {e}")
            
    def retry_failed_trade(self, trade_id: str) -> bool:
        """Retry a failed trade (resets to pending entry)"""
        if trade_id not in self.active_trades:
            return False
            
        trade = self.active_trades[trade_id]
        if trade.status != TradeStatus.FAILED or trade.retry_count >= trade.max_retries:
            return False
            
        # Reset trade for retry
        trade.status = TradeStatus.PENDING_ENTRY
        trade.entry_order_id = None
        trade.retry_count += 1
        trade.last_updated = datetime.utcnow()
        
        self.logger.info(f"Retrying failed trade: {trade.symbol} (attempt {trade.retry_count}/{trade.max_retries})")
        return True
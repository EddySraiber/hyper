"""
BracketOrderManager - Atomic bracket order execution and monitoring

Ensures all trades are executed as complete units with guaranteed protective orders.
Never allows positions to exist without corresponding stop-loss and take-profit protection.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..components.decision_engine import TradingPair
from .alpaca_client import AlpacaClient


class BracketOrderStatus(Enum):
    """Bracket order lifecycle states"""
    PLANNED = "planned"           # Bracket order created but not submitted
    SUBMITTING = "submitting"     # Currently submitting to broker
    ACTIVE = "active"             # All orders active and protecting position
    PARTIALLY_FILLED = "partially_filled"  # Entry filled, protective orders active
    COMPLETED = "completed"       # Trade completed (take-profit or stop-loss hit)
    FAILED = "failed"            # Submission failed, no position created
    PROTECTION_FAILED = "protection_failed"  # Position exists but protection lost
    EMERGENCY_EXIT = "emergency_exit"  # Emergency liquidation in progress


class BracketOrder:
    """Represents a complete bracket order unit"""
    
    def __init__(self, trading_pair: TradingPair, bracket_id: str):
        self.trading_pair = trading_pair
        self.bracket_id = bracket_id
        self.status = BracketOrderStatus.PLANNED
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Order IDs
        self.entry_order_id: Optional[str] = None
        self.stop_loss_order_id: Optional[str] = None
        self.take_profit_order_id: Optional[str] = None
        
        # Execution tracking
        self.entry_filled_at: Optional[datetime] = None
        self.entry_fill_price: Optional[float] = None
        self.position_quantity: int = 0
        
        # Protection status
        self.has_active_stop_loss = False
        self.has_active_take_profit = False
        
        # Error tracking
        self.errors: List[str] = []
        self.retry_count = 0
        self.max_retries = 3
        
    def is_protected(self) -> bool:
        """Check if position has both protective orders active"""
        return self.has_active_stop_loss and self.has_active_take_profit
    
    def has_position(self) -> bool:
        """Check if this bracket has an active position"""
        return (self.position_quantity != 0 and 
                self.status not in [BracketOrderStatus.PLANNED, BracketOrderStatus.FAILED])
    
    def needs_protection(self) -> bool:
        """Check if position exists but lacks protection"""
        return self.has_position() and not self.is_protected()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring"""
        return {
            "bracket_id": self.bracket_id,
            "symbol": self.trading_pair.symbol,
            "action": self.trading_pair.action,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "entry_order_id": self.entry_order_id,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_id": self.take_profit_order_id,
            "position_quantity": self.position_quantity,
            "is_protected": self.is_protected(),
            "needs_protection": self.needs_protection(),
            "errors": self.errors,
            "retry_count": self.retry_count
        }


class BracketOrderManager:
    """
    Manages bracket orders with guaranteed position protection.
    
    Key principles:
    1. Never allow positions without protective orders
    2. Atomic bracket order execution (all-or-nothing)
    3. Continuous monitoring and protection recovery
    4. Emergency liquidation for unprotectable positions
    """
    
    def __init__(self, alpaca_client: AlpacaClient, config: Dict[str, Any]):
        self.alpaca_client = alpaca_client
        self.config = config
        self.logger = logging.getLogger("algotrading.bracket_order_manager")
        
        # Active bracket orders tracking
        self.brackets: Dict[str, BracketOrder] = {}
        
        # Configuration
        self.max_concurrent_brackets = config.get("max_concurrent_brackets", 10)
        self.protection_check_interval = config.get("protection_check_interval", 30)  # seconds
        self.emergency_liquidation_enabled = config.get("emergency_liquidation_enabled", True)
        self.max_protection_failures = config.get("max_protection_failures", 3)
        
        # Protection monitoring
        self.monitoring_active = False
        self.protection_failures: Dict[str, int] = {}  # Track failures per symbol
        
    async def submit_bracket_order(self, trading_pair: TradingPair) -> Tuple[bool, str, Optional[BracketOrder]]:
        """
        Submit atomic bracket order with guaranteed protection.
        
        Returns:
            success: bool - True if bracket successfully submitted
            message: str - Success/error message  
            bracket: BracketOrder - Bracket object if successful, None if failed
        """
        bracket_id = f"bracket_{trading_pair.symbol}_{int(datetime.utcnow().timestamp())}"
        bracket = BracketOrder(trading_pair, bracket_id)
        
        try:
            self.logger.info(f"🔄 Submitting bracket order: {trading_pair.symbol} {trading_pair.action}")
            bracket.status = BracketOrderStatus.SUBMITTING
            
            # Pre-validation
            validation_result = await self.alpaca_client.validate_trading_pair(trading_pair)
            if not validation_result["valid"]:
                error_msg = f"Validation failed: {', '.join(validation_result['errors'])}"
                bracket.errors.append(error_msg)
                bracket.status = BracketOrderStatus.FAILED
                return False, error_msg, bracket
            
            # Submit bracket order to Alpaca
            order_result = await self.alpaca_client.execute_trading_pair(trading_pair)
            
            # Store order IDs (Alpaca bracket orders include all three orders)
            # Handle both dictionary and object response formats
            if isinstance(order_result, dict):
                bracket.entry_order_id = order_result.get("order_id")
            else:
                # Handle object response (likely from Alpaca API directly)
                bracket.entry_order_id = getattr(order_result, 'id', str(order_result))
            
            if not bracket.entry_order_id:
                raise ValueError(f"Could not extract order_id from result: {order_result}")
            
            bracket.status = BracketOrderStatus.ACTIVE
            
            # Add to tracking
            self.brackets[bracket_id] = bracket
            
            self.logger.info(f"✅ Bracket order submitted successfully: {bracket_id}")
            return True, f"Bracket order submitted: {bracket_id}", bracket
            
        except Exception as e:
            error_msg = f"Bracket submission failed: {str(e)}"
            bracket.errors.append(error_msg)
            bracket.status = BracketOrderStatus.FAILED
            self.logger.error(f"❌ Bracket order failed: {trading_pair.symbol} - {error_msg}")
            return False, error_msg, bracket
    
    async def start_protection_monitoring(self):
        """Start continuous monitoring of bracket order protection"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.logger.info("🛡️  Starting bracket order protection monitoring")
        
        while self.monitoring_active:
            try:
                await self._monitor_all_brackets()
                await asyncio.sleep(self.protection_check_interval)
            except Exception as e:
                self.logger.error(f"Error in protection monitoring: {e}")
                await asyncio.sleep(5)  # Brief pause on errors
    
    def stop_protection_monitoring(self):
        """Stop protection monitoring"""
        self.monitoring_active = False
        self.logger.info("🛑 Stopping bracket order protection monitoring")
    
    async def _monitor_all_brackets(self):
        """Monitor all active brackets for protection status"""
        if not self.brackets:
            return
            
        self.logger.debug(f"🔍 Monitoring {len(self.brackets)} bracket orders")
        
        for bracket_id, bracket in list(self.brackets.items()):
            try:
                await self._monitor_bracket(bracket)
            except Exception as e:
                self.logger.error(f"Error monitoring bracket {bracket_id}: {e}")
    
    async def _monitor_bracket(self, bracket: BracketOrder):
        """Monitor individual bracket for protection status"""
        symbol = bracket.trading_pair.symbol
        
        # Get current position and orders
        position_data = await self.alpaca_client.get_position_with_orders(symbol)
        
        if not position_data["has_position"]:
            # No position exists - bracket may have completed or never filled
            if bracket.status in [BracketOrderStatus.ACTIVE, BracketOrderStatus.PARTIALLY_FILLED]:
                bracket.status = BracketOrderStatus.COMPLETED
                self.logger.info(f"✅ Bracket completed (no position): {bracket.bracket_id}")
            return
        
        # Position exists - update bracket tracking
        position = position_data["position"]
        bracket.position_quantity = position["quantity"]
        
        # Check protective orders
        orders = position_data["orders"]
        stop_orders = orders["stop_loss_orders"]
        limit_orders = orders["take_profit_orders"]
        
        # Update protection status
        bracket.has_active_stop_loss = len([o for o in stop_orders if o["status"] in ["new", "accepted"]]) > 0
        bracket.has_active_take_profit = len([o for o in limit_orders if o["status"] in ["new", "accepted"]]) > 0
        
        # Assess protection status
        if bracket.is_protected():
            # Fully protected - reset failure count
            if symbol in self.protection_failures:
                del self.protection_failures[symbol]
            
            if bracket.status == BracketOrderStatus.PROTECTION_FAILED:
                bracket.status = BracketOrderStatus.ACTIVE
                self.logger.info(f"✅ Protection restored for bracket: {bracket.bracket_id}")
        
        elif bracket.needs_protection():
            # Position exists but lacks protection - CRITICAL
            await self._handle_protection_failure(bracket)
    
    async def _handle_protection_failure(self, bracket: BracketOrder):
        """Handle critical protection failure"""
        symbol = bracket.trading_pair.symbol
        bracket.status = BracketOrderStatus.PROTECTION_FAILED
        
        # Track failure count
        self.protection_failures[symbol] = self.protection_failures.get(symbol, 0) + 1
        failure_count = self.protection_failures[symbol]
        
        self.logger.error(f"🚨 PROTECTION FAILURE: {bracket.bracket_id} - Attempt {failure_count}")
        
        # Attempt to restore protection
        if failure_count <= self.max_protection_failures:
            success = await self._restore_protection(bracket)
            if success:
                self.logger.info(f"✅ Protection restored: {bracket.bracket_id}")
                return
        
        # Protection restoration failed - emergency liquidation
        if self.emergency_liquidation_enabled:
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION: {symbol} - Cannot restore protection")
            await self._emergency_liquidate(bracket)
        else:
            self.logger.error(f"🚨 UNPROTECTED POSITION: {symbol} - Emergency liquidation disabled")
    
    async def _restore_protection(self, bracket: BracketOrder) -> bool:
        """Attempt to restore missing protective orders"""
        symbol = bracket.trading_pair.symbol
        pair = bracket.trading_pair
        
        try:
            # Create missing stop-loss
            if not bracket.has_active_stop_loss:
                stop_result = await self.alpaca_client.update_stop_loss(symbol, pair.stop_loss)
                if stop_result["success"]:
                    bracket.stop_loss_order_id = stop_result["new_order_id"]
                    bracket.has_active_stop_loss = True
                    self.logger.info(f"🛡️  Restored stop-loss: {symbol} @ ${pair.stop_loss}")
                else:
                    return False
            
            # Create missing take-profit
            if not bracket.has_active_take_profit:
                tp_result = await self.alpaca_client.update_take_profit(symbol, pair.take_profit)
                if tp_result["success"]:
                    bracket.take_profit_order_id = tp_result["new_order_id"]
                    bracket.has_active_take_profit = True
                    self.logger.info(f"🎯 Restored take-profit: {symbol} @ ${pair.take_profit}")
                else:
                    return False
            
            return bracket.is_protected()
            
        except Exception as e:
            self.logger.error(f"Failed to restore protection for {symbol}: {e}")
            return False
    
    async def _emergency_liquidate(self, bracket: BracketOrder):
        """Emergency liquidation of unprotectable position"""
        symbol = bracket.trading_pair.symbol
        bracket.status = BracketOrderStatus.EMERGENCY_EXIT
        
        try:
            # Close position immediately at market price
            result = await self.alpaca_client.close_position(symbol, percentage=100.0)
            
            bracket.status = BracketOrderStatus.COMPLETED
            self.logger.error(f"🚨 EMERGENCY EXIT COMPLETED: {symbol} - Order: {result.get('order_id')}")
            
        except Exception as e:
            self.logger.error(f"🚨 EMERGENCY EXIT FAILED: {symbol} - {e}")
            # This is the worst-case scenario - position exists but can't be closed or protected
    
    def get_bracket_status(self) -> Dict[str, Any]:
        """Get comprehensive bracket order status"""
        total_brackets = len(self.brackets)
        protected_count = sum(1 for b in self.brackets.values() if b.is_protected())
        unprotected_count = sum(1 for b in self.brackets.values() if b.needs_protection())
        
        return {
            "total_brackets": total_brackets,
            "protected_positions": protected_count,
            "unprotected_positions": unprotected_count,
            "protection_failures": dict(self.protection_failures),
            "monitoring_active": self.monitoring_active,
            "brackets": [b.to_dict() for b in self.brackets.values()],
            "status_summary": {
                "protected": protected_count,
                "needs_protection": unprotected_count,
                "healthy": protected_count > 0 and unprotected_count == 0
            }
        }
    
    def cleanup_completed_brackets(self):
        """Remove completed brackets to prevent memory growth"""
        completed_statuses = [BracketOrderStatus.COMPLETED, BracketOrderStatus.FAILED]
        
        # Keep completed brackets for 1 hour for monitoring
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        to_remove = []
        for bracket_id, bracket in self.brackets.items():
            if (bracket.status in completed_statuses and 
                bracket.updated_at < cutoff_time):
                to_remove.append(bracket_id)
        
        for bracket_id in to_remove:
            del self.brackets[bracket_id]
        
        if to_remove:
            self.logger.info(f"🧹 Cleaned up {len(to_remove)} completed brackets")
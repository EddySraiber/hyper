"""
TradeStateManager - Trade lifecycle state machine

Implements a comprehensive state machine for trade lifecycle management.
Ensures guaranteed progression through protected states with fail-safes.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..components.decision_engine import TradingPair


class TradeState(Enum):
    """Complete trade lifecycle states"""
    
    # Pre-execution states
    PLANNED = "planned"                    # Trade decision made, not yet submitted
    VALIDATING = "validating"              # Validating trade parameters
    SUBMITTING = "submitting"              # Submitting bracket order
    
    # Active states  
    ENTRY_PENDING = "entry_pending"        # Entry order submitted, waiting for fill
    ENTRY_FILLED = "entry_filled"          # Entry filled, protective orders active
    POSITION_ACTIVE = "position_active"    # Position established with protection
    
    # Protection management states
    PROTECTION_MONITORING = "protection_monitoring"  # Actively monitoring protection
    PROTECTION_UPDATING = "protection_updating"      # Updating protective orders
    PROTECTION_FAILED = "protection_failed"          # Lost protection, attempting recovery
    
    # Exit states
    EXITING_PROFIT = "exiting_profit"      # Take-profit triggered
    EXITING_LOSS = "exiting_loss"          # Stop-loss triggered  
    EXITING_MANUAL = "exiting_manual"      # Manual exit initiated
    EMERGENCY_LIQUIDATING = "emergency_liquidating"  # Emergency liquidation
    
    # Terminal states
    COMPLETED_PROFIT = "completed_profit"  # Successfully closed at profit
    COMPLETED_LOSS = "completed_loss"      # Successfully closed at loss
    COMPLETED_MANUAL = "completed_manual"  # Manually closed
    FAILED = "failed"                      # Trade failed to execute
    EMERGENCY_CLOSED = "emergency_closed"  # Emergency liquidation completed


class TradeStateTransition:
    """Represents a state transition with metadata"""
    
    def __init__(self, from_state: TradeState, to_state: TradeState, reason: str, 
                 timestamp: datetime = None):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,  
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }


class ManagedTrade:
    """A trade under complete state management"""
    
    def __init__(self, trade_id: str, trading_pair: TradingPair):
        self.trade_id = trade_id
        self.trading_pair = trading_pair
        self.state = TradeState.PLANNED
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Order tracking
        self.entry_order_id: Optional[str] = None
        self.stop_loss_order_id: Optional[str] = None
        self.take_profit_order_id: Optional[str] = None
        self.exit_order_id: Optional[str] = None
        
        # Position tracking
        self.position_quantity: int = 0
        self.entry_fill_price: Optional[float] = None
        self.exit_fill_price: Optional[float] = None
        
        # State management
        self.state_history: List[TradeStateTransition] = []
        self.last_protection_check: Optional[datetime] = None
        self.protection_check_failures = 0
        self.max_protection_failures = 3
        
        # Performance tracking
        self.realized_pnl: Optional[float] = None
        self.fees_paid: float = 0.0
        self.errors: List[str] = []
        
    def transition_to(self, new_state: TradeState, reason: str) -> bool:
        """Safely transition to new state with validation"""
        if not self._is_valid_transition(self.state, new_state):
            self.errors.append(f"Invalid transition: {self.state.value} -> {new_state.value}")
            return False
        
        transition = TradeStateTransition(self.state, new_state, reason)
        self.state_history.append(transition)
        self.state = new_state
        self.updated_at = datetime.utcnow()
        return True
    
    def _is_valid_transition(self, from_state: TradeState, to_state: TradeState) -> bool:
        """Validate state transition is allowed"""
        
        # Terminal states cannot transition  
        terminal_states = {
            TradeState.COMPLETED_PROFIT, TradeState.COMPLETED_LOSS,
            TradeState.COMPLETED_MANUAL, TradeState.FAILED, TradeState.EMERGENCY_CLOSED
        }
        if from_state in terminal_states:
            return False
        
        # Define valid transitions
        valid_transitions = {
            TradeState.PLANNED: {
                TradeState.VALIDATING, TradeState.FAILED
            },
            TradeState.VALIDATING: {
                TradeState.SUBMITTING, TradeState.FAILED
            },
            TradeState.SUBMITTING: {
                TradeState.ENTRY_PENDING, TradeState.FAILED
            },
            TradeState.ENTRY_PENDING: {
                TradeState.ENTRY_FILLED, TradeState.FAILED
            },
            TradeState.ENTRY_FILLED: {
                TradeState.POSITION_ACTIVE, TradeState.PROTECTION_FAILED
            },
            TradeState.POSITION_ACTIVE: {
                TradeState.PROTECTION_MONITORING, TradeState.EXITING_PROFIT,
                TradeState.EXITING_LOSS, TradeState.EXITING_MANUAL,
                TradeState.PROTECTION_FAILED
            },
            TradeState.PROTECTION_MONITORING: {
                TradeState.POSITION_ACTIVE, TradeState.PROTECTION_UPDATING,
                TradeState.PROTECTION_FAILED, TradeState.EXITING_PROFIT,
                TradeState.EXITING_LOSS, TradeState.EXITING_MANUAL
            },
            TradeState.PROTECTION_UPDATING: {
                TradeState.POSITION_ACTIVE, TradeState.PROTECTION_FAILED
            },
            TradeState.PROTECTION_FAILED: {
                TradeState.POSITION_ACTIVE, TradeState.EMERGENCY_LIQUIDATING
            },
            TradeState.EXITING_PROFIT: {
                TradeState.COMPLETED_PROFIT, TradeState.EMERGENCY_LIQUIDATING
            },
            TradeState.EXITING_LOSS: {
                TradeState.COMPLETED_LOSS, TradeState.EMERGENCY_LIQUIDATING
            },
            TradeState.EXITING_MANUAL: {
                TradeState.COMPLETED_MANUAL, TradeState.EMERGENCY_LIQUIDATING
            },
            TradeState.EMERGENCY_LIQUIDATING: {
                TradeState.EMERGENCY_CLOSED
            }
        }
        
        return to_state in valid_transitions.get(from_state, set())
    
    def is_active(self) -> bool:
        """Check if trade is in an active (non-terminal) state"""
        terminal_states = {
            TradeState.COMPLETED_PROFIT, TradeState.COMPLETED_LOSS,
            TradeState.COMPLETED_MANUAL, TradeState.FAILED, TradeState.EMERGENCY_CLOSED
        }
        return self.state not in terminal_states
    
    def has_position(self) -> bool:
        """Check if trade has an active position"""
        position_states = {
            TradeState.ENTRY_FILLED, TradeState.POSITION_ACTIVE,
            TradeState.PROTECTION_MONITORING, TradeState.PROTECTION_UPDATING,
            TradeState.PROTECTION_FAILED, TradeState.EXITING_PROFIT,
            TradeState.EXITING_LOSS, TradeState.EXITING_MANUAL,
            TradeState.EMERGENCY_LIQUIDATING
        }
        return self.state in position_states
    
    def needs_protection_check(self) -> bool:
        """Check if trade needs protection status verification"""
        if not self.has_position():
            return False
        
        # Check if enough time has passed since last check
        if self.last_protection_check:
            time_since_check = datetime.utcnow() - self.last_protection_check
            if time_since_check < timedelta(minutes=1):  # Don't check too frequently
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring/logging"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.trading_pair.symbol,
            "action": self.trading_pair.action,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "entry_order_id": self.entry_order_id,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_id": self.take_profit_order_id,
            "position_quantity": self.position_quantity,
            "entry_fill_price": self.entry_fill_price,
            "exit_fill_price": self.exit_fill_price,
            "realized_pnl": self.realized_pnl,
            "is_active": self.is_active(),
            "has_position": self.has_position(),
            "protection_check_failures": self.protection_check_failures,
            "errors": self.errors,
            "state_transitions": len(self.state_history)
        }


class TradeStateManager:
    """
    Complete trade lifecycle state management.
    
    Ensures every trade progresses through proper states with guaranteed protection.
    Provides comprehensive monitoring and recovery capabilities.
    """
    
    def __init__(self, alpaca_client, config: Dict[str, Any]):
        self.alpaca_client = alpaca_client
        self.config = config
        self.logger = logging.getLogger("algotrading.trade_state_manager")
        
        # Configuration
        self.state_check_interval = config.get("state_check_interval", 30)  # seconds
        self.protection_check_interval = config.get("protection_check_interval", 60)  # seconds
        self.max_trade_age_hours = config.get("max_trade_age_hours", 48)
        self.emergency_liquidation_enabled = config.get("emergency_liquidation_enabled", True)
        
        # Active trades
        self.active_trades: Dict[str, ManagedTrade] = {}
        self.monitoring_active = False
        
        # Statistics
        self.total_trades_managed = 0
        self.successful_completions = 0
        self.emergency_liquidations = 0
        self.failed_trades = 0
    
    async def start_trade_management(self):
        """Start comprehensive trade state management"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("🎯 Starting trade state management")
        
        while self.monitoring_active:
            try:
                await self._manage_all_trades()
                await asyncio.sleep(self.state_check_interval)
            except Exception as e:
                self.logger.error(f"Error in trade state management: {e}")
                await asyncio.sleep(5)
    
    def stop_trade_management(self):
        """Stop trade state management"""
        self.monitoring_active = False
        self.logger.info("🛑 Stopping trade state management")
    
    async def create_managed_trade(self, trading_pair: TradingPair) -> str:
        """Create a new managed trade with state tracking"""
        trade_id = f"trade_{trading_pair.symbol}_{int(datetime.utcnow().timestamp())}"
        
        managed_trade = ManagedTrade(trade_id, trading_pair)
        self.active_trades[trade_id] = managed_trade
        self.total_trades_managed += 1
        
        self.logger.info(f"🎯 Created managed trade: {trade_id} ({trading_pair.symbol} {trading_pair.action})")
        
        # Immediately start processing the trade
        await self._process_trade_state(managed_trade)
        
        return trade_id
    
    async def _manage_all_trades(self):
        """Manage state progression for all active trades"""
        if not self.active_trades:
            return
        
        self.logger.debug(f"🎯 Managing {len(self.active_trades)} active trades")
        
        for trade_id, trade in list(self.active_trades.items()):
            try:
                await self._process_trade_state(trade)
                
                # Clean up completed trades
                if not trade.is_active():
                    await self._finalize_trade(trade)
                    del self.active_trades[trade_id]
                    
            except Exception as e:
                self.logger.error(f"Error processing trade {trade_id}: {e}")
                trade.errors.append(f"State processing error: {str(e)}")
    
    async def _process_trade_state(self, trade: ManagedTrade):
        """Process individual trade based on current state"""
        
        if trade.state == TradeState.PLANNED:
            await self._handle_planned_state(trade)
        
        elif trade.state == TradeState.VALIDATING:
            await self._handle_validating_state(trade)
        
        elif trade.state == TradeState.SUBMITTING:
            await self._handle_submitting_state(trade)
        
        elif trade.state == TradeState.ENTRY_PENDING:
            await self._handle_entry_pending_state(trade)
        
        elif trade.state == TradeState.ENTRY_FILLED:
            await self._handle_entry_filled_state(trade)
        
        elif trade.state == TradeState.POSITION_ACTIVE:
            await self._handle_position_active_state(trade)
        
        elif trade.state == TradeState.PROTECTION_MONITORING:
            await self._handle_protection_monitoring_state(trade)
        
        elif trade.state == TradeState.PROTECTION_FAILED:
            await self._handle_protection_failed_state(trade)
        
        # Exit states are handled by order fills/cancellations
    
    async def _handle_planned_state(self, trade: ManagedTrade):
        """Handle PLANNED state - validate trade parameters"""
        trade.transition_to(TradeState.VALIDATING, "Starting validation")
    
    async def _handle_validating_state(self, trade: ManagedTrade):
        """Handle VALIDATING state - validate with broker"""
        try:
            validation = await self.alpaca_client.validate_trading_pair(trade.trading_pair)
            
            if validation["valid"]:
                trade.transition_to(TradeState.SUBMITTING, "Validation successful")
            else:
                errors = ', '.join(validation["errors"])
                trade.errors.extend(validation["errors"])
                trade.transition_to(TradeState.FAILED, f"Validation failed: {errors}")
                
        except Exception as e:
            trade.errors.append(f"Validation error: {str(e)}")
            trade.transition_to(TradeState.FAILED, f"Validation exception: {str(e)}")
    
    async def _handle_submitting_state(self, trade: ManagedTrade):
        """Handle SUBMITTING state - submit bracket order"""
        try:
            result = await self.alpaca_client.execute_trading_pair(trade.trading_pair)
            
            trade.entry_order_id = result["order_id"]
            trade.transition_to(TradeState.ENTRY_PENDING, "Bracket order submitted")
            
        except Exception as e:
            trade.errors.append(f"Submission error: {str(e)}")
            trade.transition_to(TradeState.FAILED, f"Submission failed: {str(e)}")
    
    async def _handle_entry_pending_state(self, trade: ManagedTrade):
        """Handle ENTRY_PENDING state - check for entry fill"""
        try:
            if not trade.entry_order_id:
                trade.transition_to(TradeState.FAILED, "No entry order ID")
                return
            
            order_status = await self.alpaca_client.get_order_status(trade.entry_order_id)
            
            if order_status["status"] == "filled":
                trade.entry_fill_price = order_status["filled_avg_price"]
                trade.position_quantity = order_status["filled_qty"]
                if trade.trading_pair.action == "sell":
                    trade.position_quantity = -trade.position_quantity
                
                trade.transition_to(TradeState.ENTRY_FILLED, "Entry order filled")
            
            elif order_status["status"] in ["cancelled", "rejected", "expired"]:
                trade.transition_to(TradeState.FAILED, f"Entry order {order_status['status']}")
                
        except Exception as e:
            trade.errors.append(f"Entry check error: {str(e)}")
    
    async def _handle_entry_filled_state(self, trade: ManagedTrade):
        """Handle ENTRY_FILLED state - verify protection is active"""
        # Give protection orders time to be established
        await asyncio.sleep(2)
        
        position_data = await self.alpaca_client.get_position_with_orders(trade.trading_pair.symbol)
        
        if self._verify_protection(position_data):
            trade.transition_to(TradeState.POSITION_ACTIVE, "Protection verified")
        else:
            trade.transition_to(TradeState.PROTECTION_FAILED, "Protection not established")
    
    async def _handle_position_active_state(self, trade: ManagedTrade):
        """Handle POSITION_ACTIVE state - start protection monitoring"""
        if trade.needs_protection_check():
            trade.transition_to(TradeState.PROTECTION_MONITORING, "Starting protection monitoring")
    
    async def _handle_protection_monitoring_state(self, trade: ManagedTrade):
        """Handle PROTECTION_MONITORING state - verify ongoing protection"""
        try:
            position_data = await self.alpaca_client.get_position_with_orders(trade.trading_pair.symbol)
            trade.last_protection_check = datetime.utcnow()
            
            if not position_data["has_position"]:
                # Position closed - determine exit reason
                await self._determine_exit_completion(trade)
            
            elif self._verify_protection(position_data):
                trade.transition_to(TradeState.POSITION_ACTIVE, "Protection verified")
                trade.protection_check_failures = 0
            
            else:
                trade.protection_check_failures += 1
                trade.transition_to(TradeState.PROTECTION_FAILED, 
                                  f"Protection lost (failure {trade.protection_check_failures})")
                
        except Exception as e:
            trade.errors.append(f"Protection check error: {str(e)}")
            trade.protection_check_failures += 1
    
    async def _handle_protection_failed_state(self, trade: ManagedTrade):
        """Handle PROTECTION_FAILED state - attempt recovery or liquidate"""
        if trade.protection_check_failures >= trade.max_protection_failures:
            if self.emergency_liquidation_enabled:
                trade.transition_to(TradeState.EMERGENCY_LIQUIDATING, "Too many protection failures")
                await self._emergency_liquidate_trade(trade)
            else:
                self.logger.error(f"🚨 Trade {trade.trade_id} cannot be protected!")
        else:
            # Attempt to restore protection
            success = await self._attempt_protection_restoration(trade)
            if success:
                trade.transition_to(TradeState.POSITION_ACTIVE, "Protection restored")
    
    def _verify_protection(self, position_data: Dict[str, Any]) -> bool:
        """Verify position has proper protective orders"""
        if not position_data["has_position"]:
            return True  # No position to protect
        
        orders = position_data["orders"]
        active_stop_orders = [o for o in orders["stop_loss_orders"] if o["status"] in ["new", "accepted"]]
        active_limit_orders = [o for o in orders["take_profit_orders"] if o["status"] in ["new", "accepted"]]
        
        return len(active_stop_orders) > 0 and len(active_limit_orders) > 0
    
    async def _attempt_protection_restoration(self, trade: ManagedTrade) -> bool:
        """Attempt to restore protection for a trade"""
        try:
            symbol = trade.trading_pair.symbol
            pair = trade.trading_pair
            
            # Update stop-loss and take-profit
            result = await self.alpaca_client.update_position_parameters(
                symbol, pair.stop_loss, pair.take_profit
            )
            
            return result["success"]
            
        except Exception as e:
            trade.errors.append(f"Protection restoration error: {str(e)}")
            return False
    
    async def _emergency_liquidate_trade(self, trade: ManagedTrade):
        """Emergency liquidation of trade"""
        try:
            result = await self.alpaca_client.close_position(trade.trading_pair.symbol)
            trade.exit_order_id = result["order_id"]
            self.emergency_liquidations += 1
            
            trade.transition_to(TradeState.EMERGENCY_CLOSED, "Emergency liquidation completed")
            
        except Exception as e:
            trade.errors.append(f"Emergency liquidation error: {str(e)}")
    
    async def _determine_exit_completion(self, trade: ManagedTrade):
        """Determine how position was closed and update state accordingly"""
        # This would analyze order history to determine if exit was via:
        # - Take-profit (COMPLETED_PROFIT)  
        # - Stop-loss (COMPLETED_LOSS)
        # - Manual close (COMPLETED_MANUAL)
        
        # For now, default to manual close
        trade.transition_to(TradeState.COMPLETED_MANUAL, "Position closed")
        self.successful_completions += 1
    
    async def _finalize_trade(self, trade: ManagedTrade):
        """Finalize completed trade"""
        self.logger.info(f"✅ Trade completed: {trade.trade_id} - Final state: {trade.state.value}")
        
        if trade.state == TradeState.FAILED:
            self.failed_trades += 1
    
    def get_trade_management_status(self) -> Dict[str, Any]:
        """Get comprehensive trade management status"""
        active_count = len(self.active_trades)
        position_trades = sum(1 for t in self.active_trades.values() if t.has_position())
        
        return {
            "monitoring_active": self.monitoring_active,
            "active_trades": active_count,
            "trades_with_positions": position_trades,
            "statistics": {
                "total_trades_managed": self.total_trades_managed,
                "successful_completions": self.successful_completions,
                "emergency_liquidations": self.emergency_liquidations,
                "failed_trades": self.failed_trades
            },
            "trades": [trade.to_dict() for trade in self.active_trades.values()],
            "health_status": {
                "all_trades_healthy": all(
                    t.state not in [TradeState.PROTECTION_FAILED, TradeState.EMERGENCY_LIQUIDATING]
                    for t in self.active_trades.values()
                ),
                "has_failed_protection": any(
                    t.state == TradeState.PROTECTION_FAILED
                    for t in self.active_trades.values()  
                ),
                "emergency_liquidations_active": any(
                    t.state == TradeState.EMERGENCY_LIQUIDATING
                    for t in self.active_trades.values()
                )
            }
        }
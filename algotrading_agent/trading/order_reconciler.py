"""
OrderReconciler - Position and order state reconciliation

Continuously reconciles positions with their protective orders to detect and fix gaps.
Ensures system state consistency between positions and orders.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from .alpaca_client import AlpacaClient


@dataclass
class PositionOrderState:
    """Represents the reconciled state of a position and its orders"""
    symbol: str
    has_position: bool
    position_quantity: int = 0
    position_value: float = 0.0
    unrealized_pl: float = 0.0
    
    # Order counts
    total_orders: int = 0
    active_stop_orders: int = 0
    active_limit_orders: int = 0
    pending_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    
    # Protection status
    has_stop_protection: bool = False
    has_limit_protection: bool = False
    is_fully_protected: bool = False
    
    # Reconciliation issues
    reconciliation_issues: List[str] = None
    needs_attention: bool = False
    
    def __post_init__(self):
        if self.reconciliation_issues is None:
            self.reconciliation_issues = []
        
        # Calculate derived fields
        self.is_fully_protected = self.has_stop_protection and self.has_limit_protection
        self.needs_attention = (self.has_position and not self.is_fully_protected) or len(self.reconciliation_issues) > 0


class OrderReconciler:
    """
    Reconciles positions with orders to maintain system state consistency.
    
    Key functions:
    1. Detect orphaned positions (positions without protective orders)
    2. Detect orphaned orders (orders without corresponding positions)
    3. Identify order state inconsistencies
    4. Provide reconciliation recommendations
    """
    
    def __init__(self, alpaca_client: AlpacaClient, config: Dict[str, Any]):
        self.alpaca_client = alpaca_client
        self.config = config
        self.logger = logging.getLogger("algotrading.order_reconciler")
        
        # Configuration
        self.reconciliation_interval = config.get("reconciliation_interval", 60)  # seconds
        self.stale_order_threshold_hours = config.get("stale_order_threshold_hours", 24)
        self.auto_cleanup_enabled = config.get("auto_cleanup_enabled", True)
        
        # State tracking
        self.position_states: Dict[str, PositionOrderState] = {}
        self.monitoring_active = False
        
        # Metrics
        self.reconciliation_count = 0
        self.issues_detected = 0
        self.issues_resolved = 0
        self.last_reconciliation: Optional[datetime] = None
    
    async def start_reconciliation_monitoring(self):
        """Start continuous position-order reconciliation"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("🔄 Starting position-order reconciliation monitoring")
        
        while self.monitoring_active:
            try:
                await self._perform_full_reconciliation()
                await asyncio.sleep(self.reconciliation_interval)
            except Exception as e:
                self.logger.error(f"Error in reconciliation monitoring: {e}")
                await asyncio.sleep(10)  # Longer pause on errors
    
    def stop_reconciliation_monitoring(self):
        """Stop reconciliation monitoring"""
        self.monitoring_active = False
        self.logger.info("🛑 Stopping position-order reconciliation monitoring")
    
    async def _perform_full_reconciliation(self):
        """Perform comprehensive position-order reconciliation"""
        self.reconciliation_count += 1
        self.last_reconciliation = datetime.now(timezone.utc)
        
        try:
            # Get all positions and orders
            positions = await self.alpaca_client.get_positions()
            all_orders = await self.alpaca_client.get_orders()
            
            self.logger.debug(f"🔄 Reconciling {len(positions)} positions with {len(all_orders)} orders")
            
            # Build reconciliation state
            current_symbols = set()
            new_issues = 0
            
            # Process each position
            for position in positions:
                symbol = position["symbol"]
                current_symbols.add(symbol)
                
                state = await self._reconcile_position(symbol, position, all_orders)
                self.position_states[symbol] = state
                
                if state.needs_attention:
                    new_issues += 1
                    if len(state.reconciliation_issues) > 0:
                        self.logger.warning(f"⚠️  Reconciliation issues for {symbol}: {', '.join(state.reconciliation_issues)}")
            
            # Clean up states for positions that no longer exist
            removed_symbols = set(self.position_states.keys()) - current_symbols
            for symbol in removed_symbols:
                self.logger.debug(f"✅ Removing reconciliation state for closed position: {symbol}")
                del self.position_states[symbol]
            
            # Check for orphaned orders (orders without positions)
            await self._check_for_orphaned_orders(all_orders, current_symbols)
            
            # Update metrics
            self.issues_detected += new_issues
            
            # Log summary
            total_issues = sum(1 for state in self.position_states.values() if state.needs_attention)
            if total_issues > 0:
                self.logger.warning(f"⚠️  Reconciliation found {total_issues} positions needing attention")
            elif self.reconciliation_count % 10 == 0:  # Log every 10th reconciliation when clean
                self.logger.info(f"✅ Reconciliation clean: {len(positions)} positions properly aligned")
                
        except Exception as e:
            self.logger.error(f"Full reconciliation failed: {e}")
    
    async def _reconcile_position(self, symbol: str, position: Dict[str, Any], 
                                all_orders: List[Dict[str, Any]]) -> PositionOrderState:
        """Reconcile individual position with its orders"""
        
        # Initialize state
        state = PositionOrderState(
            symbol=symbol,
            has_position=True,
            position_quantity=position["quantity"],
            position_value=position["market_value"],
            unrealized_pl=position["unrealized_pl"]
        )
        
        # Filter orders for this symbol
        symbol_orders = [order for order in all_orders if order["symbol"] == symbol]
        state.total_orders = len(symbol_orders)
        
        # Categorize orders by status and type
        active_orders = [o for o in symbol_orders if o["status"] in ["new", "accepted", "pending_new"]]
        filled_orders = [o for o in symbol_orders if o["status"] == "filled"]
        cancelled_orders = [o for o in symbol_orders if o["status"] in ["cancelled", "expired", "rejected"]]
        
        state.pending_orders = len(active_orders)
        state.filled_orders = len(filled_orders)
        state.cancelled_orders = len(cancelled_orders)
        
        # Check for protective orders
        stop_orders = [o for o in active_orders if "stop" in o.get("order_type", "").lower()]
        limit_orders = [o for o in active_orders if o.get("order_type") == "limit"]
        
        state.active_stop_orders = len(stop_orders)
        state.active_limit_orders = len(limit_orders)
        state.has_stop_protection = len(stop_orders) > 0
        state.has_limit_protection = len(limit_orders) > 0
        
        # Analyze for issues
        await self._analyze_position_issues(state, symbol_orders)
        
        return state
    
    async def _analyze_position_issues(self, state: PositionOrderState, orders: List[Dict[str, Any]]):
        """Analyze position for reconciliation issues"""
        issues = []
        
        # Critical: Position without stop-loss protection
        if state.has_position and not state.has_stop_protection:
            issues.append("Missing stop-loss protection")
        
        # Critical: Position without take-profit protection  
        if state.has_position and not state.has_limit_protection:
            issues.append("Missing take-profit protection")
        
        # Check for excessive orders
        if state.active_stop_orders > 1:
            issues.append(f"Multiple stop orders ({state.active_stop_orders})")
        
        if state.active_limit_orders > 1:
            issues.append(f"Multiple limit orders ({state.active_limit_orders})")
        
        # Check for stale orders
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.stale_order_threshold_hours)
        
        for order in orders:
            if order["status"] in ["new", "accepted"]:
                submitted_at = datetime.fromisoformat(order["submitted_at"].replace('Z', '+00:00'))
                if submitted_at < cutoff_time:
                    issues.append(f"Stale order: {order['order_id']} ({order['order_type']})")
        
        # Check for quantity mismatches in protective orders
        position_qty = abs(state.position_quantity)
        
        for order in orders:
            if order["status"] in ["new", "accepted"]:
                order_qty = order["quantity"]
                if order_qty != position_qty:
                    issues.append(f"Quantity mismatch: position {position_qty}, order {order_qty}")
        
        # Check for price reasonableness (orders too far from current price)
        current_price = await self.alpaca_client.get_current_price(state.symbol)
        if current_price:
            for order in orders:
                if order["status"] in ["new", "accepted"] and "stop" in order.get("order_type", "").lower():
                    # Check stop order is reasonable distance from current price
                    stop_price = order.get("stop_price") or order.get("limit_price", 0)
                    if stop_price > 0:
                        price_diff_pct = abs(stop_price - current_price) / current_price
                        if price_diff_pct > 0.2:  # More than 20% away
                            issues.append(f"Stop order too far from market: {price_diff_pct*100:.1f}%")
        
        state.reconciliation_issues = issues
    
    async def _check_for_orphaned_orders(self, all_orders: List[Dict[str, Any]], 
                                       position_symbols: set):
        """Check for orders that don't have corresponding positions"""
        active_orders = [o for o in all_orders if o["status"] in ["new", "accepted", "pending_new"]]
        
        orphaned_orders = []
        for order in active_orders:
            symbol = order["symbol"]
            if symbol not in position_symbols:
                orphaned_orders.append(order)
        
        if orphaned_orders:
            self.logger.warning(f"⚠️  Found {len(orphaned_orders)} orphaned orders:")
            for order in orphaned_orders:
                self.logger.warning(f"   Orphaned: {order['symbol']} {order['order_type']} "
                                  f"(Order ID: {order['order_id']})")
                
                # Auto-cleanup if enabled
                if self.auto_cleanup_enabled:
                    try:
                        success = await self.alpaca_client.cancel_order(order["order_id"])
                        if success:
                            self.logger.info(f"🧹 Auto-cancelled orphaned order: {order['order_id']}")
                            self.issues_resolved += 1
                    except Exception as e:
                        self.logger.error(f"Failed to cancel orphaned order {order['order_id']}: {e}")
    
    async def force_reconcile_symbol(self, symbol: str) -> Dict[str, Any]:
        """Force reconciliation for a specific symbol"""
        try:
            # Get fresh data
            all_orders = await self.alpaca_client.get_orders()
            positions = await self.alpaca_client.get_positions()
            
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if position:
                state = await self._reconcile_position(symbol, position, all_orders)
                self.position_states[symbol] = state
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "state": {
                        "has_position": state.has_position,
                        "is_fully_protected": state.is_fully_protected,
                        "issues": state.reconciliation_issues,
                        "needs_attention": state.needs_attention
                    }
                }
            else:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Position not found"
                }
                
        except Exception as e:
            self.logger.error(f"Force reconciliation failed for {symbol}: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e)
            }
    
    def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get comprehensive reconciliation status"""
        positions_needing_attention = [
            state for state in self.position_states.values() 
            if state.needs_attention
        ]
        
        unprotected_positions = [
            state for state in self.position_states.values()
            if state.has_position and not state.is_fully_protected
        ]
        
        return {
            "monitoring_active": self.monitoring_active,
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "statistics": {
                "reconciliation_count": self.reconciliation_count,
                "issues_detected": self.issues_detected,
                "issues_resolved": self.issues_resolved
            },
            "current_state": {
                "total_positions": len(self.position_states),
                "positions_needing_attention": len(positions_needing_attention),
                "unprotected_positions": len(unprotected_positions),
                "fully_protected_positions": len([s for s in self.position_states.values() if s.is_fully_protected])
            },
            "positions_detail": [
                {
                    "symbol": state.symbol,
                    "has_position": state.has_position,
                    "is_fully_protected": state.is_fully_protected,
                    "has_stop_protection": state.has_stop_protection,
                    "has_limit_protection": state.has_limit_protection,
                    "issues": state.reconciliation_issues,
                    "needs_attention": state.needs_attention
                }
                for state in self.position_states.values()
            ],
            "health_status": {
                "all_reconciled": len(positions_needing_attention) == 0,
                "has_unprotected": len(unprotected_positions) > 0,
                "needs_immediate_attention": len(positions_needing_attention) > 0
            }
        }
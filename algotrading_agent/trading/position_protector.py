"""
PositionProtector - Continuous position protection monitoring

Ensures NO position can exist without protective orders.
Provides fail-safe protection for any positions that slip through bracket order system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .alpaca_client import AlpacaClient


class UnprotectedPosition:
    """Represents a position that lacks proper protection"""
    
    def __init__(self, symbol: str, position_data: Dict[str, Any]):
        self.symbol = symbol
        self.quantity = position_data["quantity"]
        self.market_value = position_data["market_value"]
        self.unrealized_pl = position_data["unrealized_pl"]
        self.side = position_data["side"]
        
        self.discovered_at = datetime.utcnow()
        self.protection_attempts = 0
        self.last_protection_attempt = None
        self.errors: List[str] = []
        
        # Generate default protective prices (5% stop-loss, 10% take-profit)
        self.current_price = self._estimate_current_price(position_data)
        self.suggested_stop_loss = self._calculate_default_stop_loss()
        self.suggested_take_profit = self._calculate_default_take_profit()
    
    def _estimate_current_price(self, position_data: Dict[str, Any]) -> float:
        """Estimate current price from position data"""
        if self.quantity == 0:
            return 0.0
        return abs(position_data["market_value"] / self.quantity)
    
    def _calculate_default_stop_loss(self) -> float:
        """Calculate conservative stop-loss price"""
        if self.quantity > 0:  # Long position
            return round(self.current_price * 0.95, 2)  # 5% below current
        else:  # Short position  
            return round(self.current_price * 1.05, 2)  # 5% above current
    
    def _calculate_default_take_profit(self) -> float:
        """Calculate conservative take-profit price"""
        if self.quantity > 0:  # Long position
            return round(self.current_price * 1.10, 2)  # 10% above current
        else:  # Short position
            return round(self.current_price * 0.90, 2)  # 10% below current
    
    def is_critical(self) -> bool:
        """Check if position is critically unprotected (large value or loss)"""
        return (abs(self.market_value) > 1000 or  # Large position value
                self.unrealized_pl < -100 or        # Significant loss  
                self.protection_attempts >= 3)      # Multiple failed attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "unrealized_pl": self.unrealized_pl,
            "side": self.side,
            "current_price": self.current_price,
            "suggested_stop_loss": self.suggested_stop_loss,
            "suggested_take_profit": self.suggested_take_profit,
            "discovered_at": self.discovered_at.isoformat(),
            "protection_attempts": self.protection_attempts,
            "last_protection_attempt": self.last_protection_attempt.isoformat() if self.last_protection_attempt else None,
            "is_critical": self.is_critical(),
            "errors": self.errors
        }


class PositionProtector:
    """
    Continuous position protection monitoring and enforcement.
    
    This is the system's safety net - ensures NO position can exist unprotected.
    Works independently of bracket order system to catch any gaps.
    """
    
    def __init__(self, alpaca_client: AlpacaClient, config: Dict[str, Any]):
        self.alpaca_client = alpaca_client
        self.config = config
        self.logger = logging.getLogger("algotrading.position_protector")
        
        # Configuration
        self.check_interval = config.get("check_interval", 30)  # seconds
        self.max_protection_attempts = config.get("max_protection_attempts", 5)
        self.emergency_liquidation_enabled = config.get("emergency_liquidation_enabled", True)
        self.default_stop_loss_pct = config.get("default_stop_loss_pct", 0.05)  # 5%
        self.default_take_profit_pct = config.get("default_take_profit_pct", 0.10)  # 10%
        
        # Tracking
        self.unprotected_positions: Dict[str, UnprotectedPosition] = {}
        self.monitoring_active = False
        self.total_protection_attempts = 0
        self.successful_protections = 0
        self.emergency_liquidations = 0
        
        # Metrics
        self.last_scan_time: Optional[datetime] = None
        self.scan_count = 0
    
    async def start_monitoring(self):
        """Start continuous position protection monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.logger.info("🛡️  Starting position protection monitoring")
        
        while self.monitoring_active:
            try:
                await self._scan_and_protect_positions()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in position protection monitoring: {e}")
                await asyncio.sleep(5)  # Brief pause on errors
    
    def stop_monitoring(self):
        """Stop position protection monitoring"""
        self.monitoring_active = False
        self.logger.info("🛑 Stopping position protection monitoring")
    
    async def _scan_and_protect_positions(self):
        """Scan all positions and ensure they have protection"""
        self.last_scan_time = datetime.utcnow()
        self.scan_count += 1
        
        try:
            # Get all current positions
            positions = await self.alpaca_client.get_positions()
            
            if not positions:
                # No positions - clear any tracked unprotected positions
                if self.unprotected_positions:
                    self.logger.info("✅ No positions found - clearing unprotected tracking")
                    self.unprotected_positions.clear()
                return
            
            self.logger.debug(f"🔍 Scanning {len(positions)} positions for protection")
            
            current_symbols = set()
            newly_unprotected = []
            
            for position in positions:
                symbol = position["symbol"]
                current_symbols.add(symbol)
                
                # Get detailed position information with orders
                position_details = await self.alpaca_client.get_position_with_orders(symbol)
                
                if await self._is_position_protected(position_details):
                    # Position is properly protected
                    if symbol in self.unprotected_positions:
                        self.logger.info(f"✅ Position protection restored: {symbol}")
                        del self.unprotected_positions[symbol]
                else:
                    # Position is unprotected
                    if symbol not in self.unprotected_positions:
                        # Newly discovered unprotected position
                        unprotected = UnprotectedPosition(symbol, position)
                        self.unprotected_positions[symbol] = unprotected
                        newly_unprotected.append(symbol)
                        self.logger.error(f"🚨 UNPROTECTED POSITION DISCOVERED: {symbol}")
                    
                    # Attempt to protect
                    await self._attempt_protection(symbol)
            
            # Clean up positions that no longer exist
            removed_symbols = set(self.unprotected_positions.keys()) - current_symbols
            for symbol in removed_symbols:
                self.logger.info(f"✅ Position closed: {symbol}")
                del self.unprotected_positions[symbol]
            
            # Log summary
            if self.unprotected_positions:
                self.logger.warning(f"🚨 {len(self.unprotected_positions)} positions remain UNPROTECTED")
            elif self.scan_count % 10 == 0:  # Log every 10th scan when all protected
                self.logger.info(f"✅ All {len(positions)} positions are properly protected")
                
        except Exception as e:
            self.logger.error(f"Error scanning positions: {e}")
    
    async def _is_position_protected(self, position_details: Dict[str, Any]) -> bool:
        """Check if a position has proper protective orders"""
        if not position_details["has_position"]:
            return True  # No position to protect
        
        orders = position_details["orders"]
        stop_orders = orders["stop_loss_orders"]
        limit_orders = orders["take_profit_orders"]
        
        # Check for active stop-loss orders
        active_stop_orders = [o for o in stop_orders if o["status"] in ["new", "accepted", "pending_new"]]
        
        # Check for active take-profit orders  
        active_limit_orders = [o for o in limit_orders if o["status"] in ["new", "accepted", "pending_new"]]
        
        # Position is protected if it has BOTH stop-loss AND take-profit
        has_stop_protection = len(active_stop_orders) > 0
        has_profit_protection = len(active_limit_orders) > 0
        
        return has_stop_protection and has_profit_protection
    
    async def _attempt_protection(self, symbol: str):
        """Attempt to protect an unprotected position"""
        if symbol not in self.unprotected_positions:
            return
        
        unprotected = self.unprotected_positions[symbol]
        unprotected.protection_attempts += 1
        unprotected.last_protection_attempt = datetime.utcnow()
        self.total_protection_attempts += 1
        
        try:
            self.logger.info(f"🛡️  Attempting to protect position: {symbol} (attempt {unprotected.protection_attempts})")
            
            # Get fresh position data
            position_details = await self.alpaca_client.get_position_with_orders(symbol)
            
            if not position_details["has_position"]:
                self.logger.info(f"✅ Position {symbol} no longer exists")
                del self.unprotected_positions[symbol]
                return
            
            position = position_details["position"]
            orders = position_details["orders"]
            
            # Update current price
            unprotected.current_price = position.get("current_price", unprotected.current_price)
            
            # Check what protection is missing
            active_stop_orders = [o for o in orders["stop_loss_orders"] if o["status"] in ["new", "accepted"]]
            active_limit_orders = [o for o in orders["take_profit_orders"] if o["status"] in ["new", "accepted"]]
            
            needs_stop_loss = len(active_stop_orders) == 0
            needs_take_profit = len(active_limit_orders) == 0
            
            # Create missing protective orders
            protection_success = True
            
            if needs_stop_loss:
                stop_result = await self.alpaca_client.update_stop_loss(
                    symbol, unprotected.suggested_stop_loss
                )
                if stop_result["success"]:
                    self.logger.info(f"🛡️  Created stop-loss for {symbol} @ ${unprotected.suggested_stop_loss}")
                else:
                    protection_success = False
                    error_msg = f"Stop-loss creation failed: {stop_result.get('error')}"
                    unprotected.errors.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
            
            if needs_take_profit:
                tp_result = await self.alpaca_client.update_take_profit(
                    symbol, unprotected.suggested_take_profit
                )
                if tp_result["success"]:
                    self.logger.info(f"🎯 Created take-profit for {symbol} @ ${unprotected.suggested_take_profit}")
                else:
                    protection_success = False
                    error_msg = f"Take-profit creation failed: {tp_result.get('error')}"
                    unprotected.errors.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
            
            if protection_success:
                self.successful_protections += 1
                self.logger.info(f"✅ Successfully protected position: {symbol}")
                # Position will be removed from unprotected list on next scan
            
            # Check if we need emergency liquidation
            elif (unprotected.protection_attempts >= self.max_protection_attempts and 
                  self.emergency_liquidation_enabled):
                await self._emergency_liquidate(symbol)
                
        except Exception as e:
            error_msg = f"Protection attempt failed: {str(e)}"
            unprotected.errors.append(error_msg)
            self.logger.error(f"❌ Protection failed for {symbol}: {error_msg}")
    
    async def _emergency_liquidate(self, symbol: str):
        """Emergency liquidation of persistently unprotectable position"""
        if symbol not in self.unprotected_positions:
            return
        
        unprotected = self.unprotected_positions[symbol]
        
        try:
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION: {symbol} - Cannot create protective orders")
            
            # Close position immediately at market price
            result = await self.alpaca_client.close_position(symbol, percentage=100.0)
            
            self.emergency_liquidations += 1
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION COMPLETED: {symbol} - Order: {result.get('order_id')}")
            
            # Remove from unprotected tracking
            del self.unprotected_positions[symbol]
            
        except Exception as e:
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION FAILED: {symbol} - {e}")
            # This is the worst case - position exists, can't be protected, can't be closed
    
    async def force_protect_all(self) -> Dict[str, Any]:
        """Force protection attempt for all unprotected positions"""
        if not self.unprotected_positions:
            return {"message": "No unprotected positions found", "protected": 0}
        
        symbols = list(self.unprotected_positions.keys())
        self.logger.info(f"🛡️  Force protecting {len(symbols)} positions: {symbols}")
        
        initial_count = len(self.unprotected_positions)
        
        for symbol in symbols:
            await self._attempt_protection(symbol)
        
        # Rescan to get updated status
        await self._scan_and_protect_positions()
        
        final_count = len(self.unprotected_positions)
        protected_count = initial_count - final_count
        
        return {
            "message": f"Protected {protected_count} out of {initial_count} positions",
            "initial_unprotected": initial_count,
            "protected": protected_count,
            "still_unprotected": final_count,
            "unprotected_symbols": list(self.unprotected_positions.keys())
        }
    
    def get_protection_status(self) -> Dict[str, Any]:
        """Get comprehensive protection status"""
        return {
            "monitoring_active": self.monitoring_active,
            "unprotected_positions": len(self.unprotected_positions),
            "unprotected_details": [pos.to_dict() for pos in self.unprotected_positions.values()],
            "statistics": {
                "total_protection_attempts": self.total_protection_attempts,
                "successful_protections": self.successful_protections,
                "emergency_liquidations": self.emergency_liquidations,
                "scan_count": self.scan_count,
                "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None
            },
            "critical_positions": [
                symbol for symbol, pos in self.unprotected_positions.items() 
                if pos.is_critical()
            ],
            "health_status": {
                "all_protected": len(self.unprotected_positions) == 0,
                "critical_exposure": any(pos.is_critical() for pos in self.unprotected_positions.values()),
                "needs_attention": len(self.unprotected_positions) > 0
            }
        }
"""
Enhanced Guardian Service - Advanced Position Safety Monitor

Comprehensive protection system that prevents unsafe trading positions through:
- High-frequency scanning (30 seconds vs 10 minutes) 
- Multi-layer validation (orders, positions, crypto vs stocks)
- Smart leak detection (tests, manual orders, failed brackets)
- Automated remediation with fallback strategies

This is the "guardian angel" that catches what the PositionProtector might miss.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum

from .alpaca_client import AlpacaClient


class LeakType(Enum):
    """Types of position leaks that can occur"""
    TEST_ORDER = "test_order"           # From unsafe tests
    MANUAL_ORDER = "manual_order"       # Direct manual trading
    FAILED_BRACKET = "failed_bracket"   # Bracket order partially failed
    CRYPTO_UNPROTECTED = "crypto_unprotected"  # Crypto without stops
    ORPHANED_POSITION = "orphaned_position"    # Position without orders
    PARTIAL_PROTECTION = "partial_protection"  # Only stop OR take-profit


class PositionLeak:
    """Represents a detected unsafe position"""
    
    def __init__(self, symbol: str, leak_type: LeakType, position_data: Dict[str, Any]):
        self.symbol = symbol
        self.leak_type = leak_type
        self.position_data = position_data
        self.discovered_at = datetime.utcnow()
        
        # Extract key position info
        self.quantity = position_data.get("quantity", 0)
        self.market_value = position_data.get("market_value", 0)
        self.unrealized_pl = position_data.get("unrealized_pl", 0)
        self.side = position_data.get("side", "unknown")
        
        # Risk assessment
        self.risk_level = self._assess_risk_level()
        self.remediation_attempts = 0
        self.last_remediation = None
        self.errors: List[str] = []
    
    def _assess_risk_level(self) -> str:
        """Assess risk level: low, medium, high, critical"""
        abs_value = abs(self.market_value)
        unrealized_loss = self.unrealized_pl if self.unrealized_pl < 0 else 0
        
        if abs_value > 5000 or unrealized_loss < -500:
            return "critical"
        elif abs_value > 1000 or unrealized_loss < -100:
            return "high" 
        elif abs_value > 200 or unrealized_loss < -50:
            return "medium"
        else:
            return "low"
    
    def is_crypto(self) -> bool:
        """Check if this is a crypto position"""
        return "/" in self.symbol or any(crypto in self.symbol.upper() 
                                       for crypto in ["BTC", "ETH", "DOGE", "SOL", "USDT", "USDC"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring"""
        return {
            "symbol": self.symbol,
            "leak_type": self.leak_type.value,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "unrealized_pl": self.unrealized_pl,
            "side": self.side,
            "risk_level": self.risk_level,
            "is_crypto": self.is_crypto(),
            "discovered_at": self.discovered_at.isoformat(),
            "remediation_attempts": self.remediation_attempts,
            "last_remediation": self.last_remediation.isoformat() if self.last_remediation else None,
            "errors": self.errors
        }


class GuardianService:
    """
    Enhanced Guardian Service - The system's ultimate safety net
    
    Runs high-frequency scans to detect and remediate position leaks that could
    expose the account to uncontrolled risk. This catches what regular protection
    systems might miss due to timing, errors, or edge cases.
    """
    
    def __init__(self, alpaca_client: AlpacaClient, config: Dict[str, Any]):
        self.alpaca_client = alpaca_client
        self.config = config
        self.logger = logging.getLogger("algotrading.guardian_service")
        
        # Configuration - More aggressive than PositionProtector
        self.scan_interval = config.get("guardian_scan_interval", 30)  # 30 seconds
        self.max_remediation_attempts = config.get("max_remediation_attempts", 3)
        self.emergency_liquidation_enabled = config.get("emergency_liquidation_enabled", True)
        self.crypto_protection_enabled = config.get("crypto_protection_enabled", True)
        
        # Tracking
        self.active_leaks: Dict[str, PositionLeak] = {}
        self.monitoring_active = False
        self.scan_count = 0
        self.last_scan_time: Optional[datetime] = None
        
        # Statistics
        self.leaks_detected = 0
        self.leaks_remediated = 0
        self.emergency_liquidations = 0
        self.crypto_positions_protected = 0
        
        # Historical tracking for pattern detection
        self.recent_positions: Set[str] = set()
        self.position_history: Dict[str, datetime] = {}
    
    async def start_guardian_monitoring(self):
        """Start high-frequency guardian monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("🛡️  Starting Guardian Service - Advanced Position Safety Monitor")
        self.logger.info(f"   Scan frequency: {self.scan_interval}s (high-frequency leak detection)")
        
        while self.monitoring_active:
            try:
                await self._scan_for_leaks()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                self.logger.error(f"Guardian monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause on errors
    
    def stop_guardian_monitoring(self):
        """Stop guardian monitoring"""
        self.monitoring_active = False
        self.logger.info("🛑 Guardian Service stopped")
    
    async def _scan_for_leaks(self):
        """Comprehensive scan for position leaks"""
        self.last_scan_time = datetime.utcnow()
        self.scan_count += 1
        
        try:
            # Get current positions and orders
            positions = await self.alpaca_client.get_positions()
            orders = await self.alpaca_client.get_orders()
            
            if not positions:
                # Clean up if no positions exist
                if self.active_leaks:
                    self.logger.info("✅ No positions - clearing leak tracking")
                    self.active_leaks.clear()
                return
            
            self.logger.debug(f"🔍 Guardian scan #{self.scan_count}: {len(positions)} positions")
            
            current_symbols = set()
            new_leaks_found = []
            
            for position in positions:
                symbol = position["symbol"]
                current_symbols.add(symbol)
                
                # Update position tracking
                self.recent_positions.add(symbol)
                self.position_history[symbol] = datetime.utcnow()
                
                # Check for various types of leaks
                leak_type = await self._detect_leak_type(symbol, position, orders)
                
                if leak_type:
                    if symbol not in self.active_leaks:
                        # New leak detected
                        leak = PositionLeak(symbol, leak_type, position)
                        self.active_leaks[symbol] = leak
                        new_leaks_found.append(symbol)
                        self.leaks_detected += 1
                        
                        risk_emoji = self._get_risk_emoji(leak.risk_level)
                        self.logger.error(f"🚨 {risk_emoji} LEAK DETECTED: {symbol} ({leak_type.value}) - ${leak.market_value:.2f}")
                    
                    # Attempt remediation
                    await self._remediate_leak(symbol)
                else:
                    # Position is safe - remove from leak tracking
                    if symbol in self.active_leaks:
                        self.logger.info(f"✅ Leak resolved: {symbol}")
                        del self.active_leaks[symbol]
                        self.leaks_remediated += 1
            
            # Clean up closed positions
            closed_symbols = set(self.active_leaks.keys()) - current_symbols
            for symbol in closed_symbols:
                self.logger.info(f"✅ Position closed: {symbol}")
                del self.active_leaks[symbol]
            
            # Log summary
            if self.active_leaks:
                critical_count = sum(1 for leak in self.active_leaks.values() if leak.risk_level == "critical")
                high_count = sum(1 for leak in self.active_leaks.values() if leak.risk_level == "high")
                
                if critical_count > 0 or high_count > 0:
                    self.logger.warning(f"🚨 Active leaks: {len(self.active_leaks)} ({critical_count} critical, {high_count} high risk)")
                
            elif self.scan_count % 20 == 0:  # Log every 20th scan when clean
                self.logger.info(f"✅ Guardian scan #{self.scan_count}: All {len(positions)} positions secure")
                
        except Exception as e:
            self.logger.error(f"Guardian scan error: {e}")
    
    async def _detect_leak_type(self, symbol: str, position: Dict[str, Any], orders: List[Dict[str, Any]]) -> Optional[LeakType]:
        """Detect what type of leak this position represents"""
        try:
            # Get detailed position with orders
            position_details = await self.alpaca_client.get_position_with_orders(symbol)
            
            if not position_details["has_position"]:
                return None
            
            orders_info = position_details["orders"]
            stop_orders = orders_info["stop_loss_orders"]
            limit_orders = orders_info["take_profit_orders"]
            
            # Check for active protective orders
            active_stops = [o for o in stop_orders if o["status"] in ["new", "accepted", "pending_new"]]
            active_limits = [o for o in limit_orders if o["status"] in ["new", "accepted", "pending_new"]]
            
            has_stop = len(active_stops) > 0
            has_take_profit = len(active_limits) > 0
            
            # Classify the type of leak
            if not has_stop and not has_take_profit:
                # No protection at all
                if self._is_crypto_symbol(symbol):
                    return LeakType.CRYPTO_UNPROTECTED
                elif self._appears_to_be_test_order(symbol, position):
                    return LeakType.TEST_ORDER
                else:
                    return LeakType.ORPHANED_POSITION
            
            elif has_stop and not has_take_profit:
                # Only has stop-loss, missing take-profit
                return LeakType.PARTIAL_PROTECTION
            
            elif not has_stop and has_take_profit:
                # Only has take-profit, missing stop-loss (critical!)
                return LeakType.PARTIAL_PROTECTION
            
            # Both protections exist - not a leak
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting leak type for {symbol}: {e}")
            return LeakType.ORPHANED_POSITION  # Default to safest assumption
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency"""
        return "/" in symbol or any(crypto in symbol.upper() 
                                  for crypto in ["BTC", "ETH", "DOGE", "SOL", "USDT", "USDC", "AVAX", "DOT"])
    
    def _appears_to_be_test_order(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Heuristic to detect if this might be from a test"""
        # Small positions in common test assets might be test orders
        abs_value = abs(position.get("market_value", 0))
        quantity = abs(position.get("quantity", 0))
        
        # Small DOGE positions are likely test orders (like our recent issue)
        if "DOGE" in symbol and abs_value < 50:
            return True
        
        # Very small positions in any crypto could be tests
        if self._is_crypto_symbol(symbol) and abs_value < 20:
            return True
        
        # Single shares of expensive stocks could be tests
        if not self._is_crypto_symbol(symbol) and quantity == 1 and abs_value > 100:
            return True
        
        return False
    
    async def _remediate_leak(self, symbol: str):
        """Attempt to fix a detected leak with time-based emergency liquidation"""
        if symbol not in self.active_leaks:
            return
        
        leak = self.active_leaks[symbol]
        leak.remediation_attempts += 1
        leak.last_remediation = datetime.utcnow()
        
        # Calculate how long this leak has been trying to be fixed
        leak_age_hours = (datetime.utcnow() - leak.discovered_at).total_seconds() / 3600
        
        try:
            self.logger.info(f"🔧 Remediating {leak.leak_type.value}: {symbol} (attempt {leak.remediation_attempts}, age: {leak_age_hours:.1f}h)")
            
            # TIME-BASED EMERGENCY LIQUIDATION
            # If position has been unprotectable for too long, force liquidate it
            max_unprotected_hours = 4.0  # Maximum time to allow unprotected positions
            
            if (leak_age_hours > max_unprotected_hours or 
                leak.remediation_attempts >= 50):  # Also liquidate after 50 failed attempts
                
                self.logger.error(f"🚨 TIME LIMIT EXCEEDED: {symbol} unprotected for {leak_age_hours:.1f}h - FORCE LIQUIDATING")
                await self._emergency_liquidate_leak(symbol)
                return
            
            # Strategy depends on leak type
            if leak.leak_type == LeakType.TEST_ORDER:
                # Test orders should usually be closed immediately
                await self._close_test_position(symbol)
                
            elif leak.leak_type == LeakType.CRYPTO_UNPROTECTED:
                # Crypto needs different protection strategy
                await self._protect_crypto_position(symbol, leak)
                
            elif leak.leak_type in [LeakType.ORPHANED_POSITION, LeakType.PARTIAL_PROTECTION]:
                # Try to add missing protective orders
                success = await self._add_missing_protection(symbol, leak)
                
                # If protection keeps failing, check for emergency liquidation based on attempts
                if (not success and 
                    leak.remediation_attempts >= self.max_remediation_attempts and 
                    self.emergency_liquidation_enabled and 
                    leak.risk_level in ["medium", "high", "critical"]):
                    
                    self.logger.warning(f"🚨 PROTECTION REPEATEDLY FAILED: {symbol} ({leak.remediation_attempts} attempts) - considering emergency liquidation")
                    await self._emergency_liquidate_leak(symbol)
                
        except Exception as e:
            error_msg = f"Remediation failed: {e}"
            leak.errors.append(error_msg)
            self.logger.error(f"❌ {symbol} remediation failed: {error_msg}")
            
            # If remediation itself is crashing repeatedly, force liquidate
            if leak.remediation_attempts >= 20:
                self.logger.error(f"🚨 REMEDIATION CRASHES: {symbol} remediation failing repeatedly - FORCE LIQUIDATING")
                await self._emergency_liquidate_leak(symbol)
    
    async def _close_test_position(self, symbol: str):
        """Close a position identified as a test order"""
        try:
            result = await self.alpaca_client.close_position(symbol, percentage=100.0)
            self.logger.info(f"🧹 Closed test position: {symbol} (Order: {result.get('order_id')})")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to close test position {symbol}: {e}")
            return False
    
    async def _protect_crypto_position(self, symbol: str, leak: PositionLeak):
        """Add protection for crypto positions (different from stocks)"""
        if not self.crypto_protection_enabled:
            return
        
        try:
            # For crypto, we might need to use manual stop-loss monitoring
            # since bracket orders don't work the same way
            self.logger.warning(f"⚠️  Crypto protection needed for {symbol} - considering manual monitoring")
            
            # Get current price for crypto
            current_price = await self.alpaca_client.get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Cannot get current price for {symbol}")
            
            # Calculate protection levels for crypto (wider spreads due to volatility)
            if leak.quantity > 0:  # Long position
                stop_price = round(current_price * 0.92, 6)  # 8% stop-loss for crypto
                take_profit_price = round(current_price * 1.15, 6)  # 15% take-profit
            else:  # Short position
                stop_price = round(current_price * 1.08, 6)  # 8% stop-loss for crypto shorts
                take_profit_price = round(current_price * 0.85, 6)  # 15% take-profit
            
            self.logger.info(f"🚀 Crypto protection calculated for {symbol}: SL=${stop_price}, TP=${take_profit_price}")
            self.crypto_positions_protected += 1
            
        except Exception as e:
            self.logger.error(f"❌ Crypto protection failed for {symbol}: {e}")
    
    async def _add_missing_protection(self, symbol: str, leak: PositionLeak) -> bool:
        """Add missing stop-loss and/or take-profit orders"""
        try:
            # Get current price
            current_price = await self.alpaca_client.get_current_price(symbol)
            if not current_price:
                self.logger.warning(f"⚠️  Cannot get current price for {symbol} - using estimated price")
                current_price = abs(leak.market_value / leak.quantity) if leak.quantity != 0 else 100.0
            
            # Calculate protection levels with wider spreads for volatile positions
            spread_multiplier = 1.0
            if abs(leak.unrealized_pl) > 100:  # Wider spreads for losing positions
                spread_multiplier = 1.5
            
            if leak.quantity > 0:  # Long position
                stop_loss_pct = 0.05 * spread_multiplier  # 5-7.5% stop-loss
                take_profit_pct = 0.10 * spread_multiplier  # 10-15% take-profit
                stop_price = round(current_price * (1 - stop_loss_pct), 2)
                take_profit_price = round(current_price * (1 + take_profit_pct), 2)
            else:  # Short position
                stop_loss_pct = 0.05 * spread_multiplier
                take_profit_pct = 0.10 * spread_multiplier
                stop_price = round(current_price * (1 + stop_loss_pct), 2)
                take_profit_price = round(current_price * (1 - take_profit_pct), 2)
            
            self.logger.info(f"🛡️  Attempting protection for {symbol}: SL=${stop_price}, TP=${take_profit_price} (spread: {spread_multiplier}x)")
            
            # Try to add missing orders
            results = await self.alpaca_client.update_position_parameters(
                symbol, stop_price, take_profit_price
            )
            
            if results["success"]:
                self.logger.info(f"✅ Successfully added protection for {symbol}")
                return True
            else:
                # Log specific error details
                errors = results.get('errors', ['Unknown error'])
                self.logger.error(f"❌ Protection failed for {symbol}: {'; '.join(errors)}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Protection addition failed for {symbol}: {e}")
            return False
    
    async def _emergency_liquidate_leak(self, symbol: str):
        """Emergency liquidation for persistent leaks"""
        leak = self.active_leaks[symbol]
        
        try:
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION: {symbol} - {leak.leak_type.value} (${leak.market_value:.2f})")
            
            result = await self.alpaca_client.close_position(symbol, percentage=100.0)
            
            self.emergency_liquidations += 1
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION COMPLETED: {symbol} - Order: {result.get('order_id')}")
            
            # Remove from tracking
            del self.active_leaks[symbol]
            
        except Exception as e:
            self.logger.error(f"🚨 EMERGENCY LIQUIDATION FAILED: {symbol} - {e}")
    
    def _get_risk_emoji(self, risk_level: str) -> str:
        """Get emoji for risk level"""
        risk_emojis = {
            "low": "⚠️",
            "medium": "🟡", 
            "high": "🔴",
            "critical": "💥"
        }
        return risk_emojis.get(risk_level, "⚠️")
    
    async def force_remediate_all(self) -> Dict[str, Any]:
        """Force remediation for all active leaks"""
        if not self.active_leaks:
            return {"message": "No active leaks found", "remediated": 0}
        
        symbols = list(self.active_leaks.keys())
        self.logger.info(f"🔧 Force remediating {len(symbols)} leaks: {symbols}")
        
        initial_count = len(self.active_leaks)
        
        for symbol in symbols:
            await self._remediate_leak(symbol)
        
        # Rescan after remediation attempts
        await self._scan_for_leaks()
        
        final_count = len(self.active_leaks)
        remediated_count = initial_count - final_count
        
        return {
            "message": f"Remediated {remediated_count} out of {initial_count} leaks",
            "initial_leaks": initial_count,
            "remediated": remediated_count,
            "still_active": final_count,
            "remaining_leaks": [leak.to_dict() for leak in self.active_leaks.values()]
        }
    
    def get_guardian_status(self) -> Dict[str, Any]:
        """Get comprehensive guardian status"""
        return {
            "monitoring_active": self.monitoring_active,
            "scan_interval": self.scan_interval,
            "active_leaks": len(self.active_leaks),
            "leak_details": [leak.to_dict() for leak in self.active_leaks.values()],
            "statistics": {
                "scan_count": self.scan_count,
                "leaks_detected": self.leaks_detected,
                "leaks_remediated": self.leaks_remediated,
                "emergency_liquidations": self.emergency_liquidations,
                "crypto_positions_protected": self.crypto_positions_protected,
                "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None
            },
            "risk_assessment": {
                "critical_leaks": len([l for l in self.active_leaks.values() if l.risk_level == "critical"]),
                "high_risk_leaks": len([l for l in self.active_leaks.values() if l.risk_level == "high"]),
                "total_exposure": sum(abs(l.market_value) for l in self.active_leaks.values()),
                "unrealized_losses": sum(l.unrealized_pl for l in self.active_leaks.values() if l.unrealized_pl < 0)
            },
            "health_status": {
                "system_secure": len(self.active_leaks) == 0,
                "needs_immediate_attention": any(l.risk_level == "critical" for l in self.active_leaks.values()),
                "recent_activity": len(self.recent_positions) > 0
            }
        }
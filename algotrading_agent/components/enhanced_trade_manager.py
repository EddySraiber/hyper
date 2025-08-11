"""
Enhanced TradeManager - Integrates new order management architecture

Combines the new BracketOrderManager, PositionProtector, OrderReconciler, and TradeStateManager
to provide comprehensive trade execution with guaranteed position protection.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from ..core.base import PersistentComponent
from ..trading.alpaca_client import AlpacaClient
from ..trading.bracket_order_manager import BracketOrderManager
from ..trading.position_protector import PositionProtector
from ..trading.order_reconciler import OrderReconciler
from ..trading.trade_state_manager import TradeStateManager
from .decision_engine import TradingPair


class EnhancedTradeManager(PersistentComponent):
    """
    Next-generation trade manager with comprehensive position protection.
    
    Key features:
    - Guaranteed bracket order execution
    - Continuous position protection monitoring
    - Order-position reconciliation
    - Complete trade state management
    - Emergency fail-safes
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("enhanced_trade_manager", config)
        
        # Configuration
        self.max_concurrent_trades = config.get("max_concurrent_trades", 10)
        self.enable_legacy_compatibility = config.get("enable_legacy_compatibility", True)
        
        # External dependencies (injected by main app)
        self.alpaca_client: Optional[AlpacaClient] = None
        self.decision_engine = None
        
        # Core architecture components
        self.bracket_manager: Optional[BracketOrderManager] = None
        self.position_protector: Optional[PositionProtector] = None
        self.order_reconciler: Optional[OrderReconciler] = None
        self.state_manager: Optional[TradeStateManager] = None
        
        # Statistics and monitoring
        self.total_trades_processed = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.protection_recoveries = 0
        self.emergency_liquidations = 0
        
    def start(self) -> None:
        """Initialize all components and start protection systems"""
        if not self.alpaca_client:
            self.logger.error("Cannot start without Alpaca client")
            return
            
        self.logger.info("🚀 Starting Enhanced Trade Manager")
        self.is_running = True
        
        try:
            # Initialize core components
            self._initialize_components()
            
            # Start protection systems
            asyncio.create_task(self._start_protection_systems())
            
            self.logger.info("✅ Enhanced Trade Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start Enhanced Trade Manager: {e}")
            self.is_running = False
    
    def stop(self) -> None:
        """Stop all protection systems and save state"""
        self.logger.info("🛑 Stopping Enhanced Trade Manager")
        self.is_running = False
        
        # Stop all monitoring systems
        if self.bracket_manager:
            self.bracket_manager.stop_protection_monitoring()
        if self.position_protector:
            self.position_protector.stop_monitoring()
        if self.order_reconciler:
            self.order_reconciler.stop_reconciliation_monitoring()
        if self.state_manager:
            self.state_manager.stop_trade_management()
    
    def _initialize_components(self):
        """Initialize all architecture components"""
        self.logger.info("🔧 Initializing enhanced trade management components")
        
        # Initialize BracketOrderManager
        bracket_config = {
            "max_concurrent_brackets": self.max_concurrent_trades,
            "protection_check_interval": 30,
            "emergency_liquidation_enabled": True,
            "max_protection_failures": 3
        }
        self.bracket_manager = BracketOrderManager(self.alpaca_client, bracket_config)
        
        # Initialize PositionProtector
        protector_config = {
            "check_interval": 30,
            "max_protection_attempts": 5,
            "emergency_liquidation_enabled": True,
            "default_stop_loss_pct": 0.05,
            "default_take_profit_pct": 0.10
        }
        self.position_protector = PositionProtector(self.alpaca_client, protector_config)
        
        # Initialize OrderReconciler
        reconciler_config = {
            "reconciliation_interval": 60,
            "stale_order_threshold_hours": 24,
            "auto_cleanup_enabled": True
        }
        self.order_reconciler = OrderReconciler(self.alpaca_client, reconciler_config)
        
        # Initialize TradeStateManager
        state_config = {
            "state_check_interval": 30,
            "protection_check_interval": 60,
            "max_trade_age_hours": 48,
            "emergency_liquidation_enabled": True
        }
        self.state_manager = TradeStateManager(self.alpaca_client, state_config)
        
        self.logger.info("✅ All components initialized")
    
    async def _start_protection_systems(self):
        """Start all protection and monitoring systems"""
        self.logger.info("🛡️  Starting protection systems")
        
        try:
            # Start all monitoring systems concurrently
            await asyncio.gather(
                self.bracket_manager.start_protection_monitoring(),
                self.position_protector.start_monitoring(),
                self.order_reconciler.start_reconciliation_monitoring(),
                self.state_manager.start_trade_management()
            )
        except Exception as e:
            self.logger.error(f"Error starting protection systems: {e}")
    
    async def execute_trade(self, trading_pair: TradingPair) -> Dict[str, Any]:
        """
        Execute trade with comprehensive protection system.
        
        This is the main entry point for trade execution.
        """
        if not self.is_running:
            return self._create_execution_result(False, "Trade manager not running", None)
        
        self.total_trades_processed += 1
        
        try:
            self.logger.info(f"🎯 Executing trade: {trading_pair.symbol} {trading_pair.action}")
            
            # Method 1: Use BracketOrderManager for atomic execution
            success, message, bracket = await self.bracket_manager.submit_bracket_order(trading_pair)
            
            if success:
                # Method 2: Also create state-managed trade for comprehensive monitoring
                trade_id = await self.state_manager.create_managed_trade(trading_pair)
                
                self.successful_executions += 1
                self.logger.info(f"✅ Trade executed successfully: {message}")
                
                return self._create_execution_result(True, message, {
                    "bracket_id": bracket.bracket_id,
                    "trade_id": trade_id,
                    "entry_order_id": bracket.entry_order_id
                })
            else:
                self.failed_executions += 1
                self.logger.error(f"❌ Trade execution failed: {message}")
                
                return self._create_execution_result(False, message, None)
                
        except Exception as e:
            self.failed_executions += 1
            error_msg = f"Trade execution exception: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_execution_result(False, error_msg, None)
    
    async def force_protect_all_positions(self) -> Dict[str, Any]:
        """Force protection attempt for all unprotected positions"""
        self.logger.info("🛡️  Force protecting all positions")
        
        try:
            # Use PositionProtector to force protection
            result = await self.position_protector.force_protect_all()
            
            if result["protected"] > 0:
                self.protection_recoveries += result["protected"]
            
            return result
            
        except Exception as e:
            error_msg = f"Force protection failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    async def emergency_liquidate_unprotected(self) -> Dict[str, Any]:
        """Emergency liquidation of all unprotected positions"""
        self.logger.error("🚨 EMERGENCY LIQUIDATION INITIATED")
        
        try:
            # Get all unprotected positions
            protection_status = self.position_protector.get_protection_status()
            unprotected_symbols = [
                pos["symbol"] for pos in protection_status["unprotected_details"]
            ]
            
            if not unprotected_symbols:
                return {"message": "No unprotected positions found"}
            
            liquidation_results = []
            
            for symbol in unprotected_symbols:
                try:
                    result = await self.alpaca_client.close_position(symbol, percentage=100.0)
                    liquidation_results.append({
                        "symbol": symbol,
                        "success": True,
                        "order_id": result.get("order_id")
                    })
                    self.emergency_liquidations += 1
                    
                except Exception as e:
                    liquidation_results.append({
                        "symbol": symbol,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "message": f"Emergency liquidation attempted for {len(unprotected_symbols)} positions",
                "results": liquidation_results
            }
            
        except Exception as e:
            error_msg = f"Emergency liquidation failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status from all components"""
        if not self.is_running:
            return {"error": "Enhanced Trade Manager not running"}
        
        try:
            # Collect status from all components
            bracket_status = self.bracket_manager.get_bracket_status() if self.bracket_manager else {}
            protection_status = self.position_protector.get_protection_status() if self.position_protector else {}
            reconciliation_status = self.order_reconciler.get_reconciliation_status() if self.order_reconciler else {}
            state_status = self.state_manager.get_trade_management_status() if self.state_manager else {}
            
            # Calculate overall health
            all_protected = (
                bracket_status.get("status_summary", {}).get("healthy", False) and
                protection_status.get("health_status", {}).get("all_protected", False) and
                reconciliation_status.get("health_status", {}).get("all_reconciled", False) and
                state_status.get("health_status", {}).get("all_trades_healthy", False)
            )
            
            unprotected_count = (
                bracket_status.get("unprotected_positions", 0) +
                protection_status.get("unprotected_positions", 0)
            )
            
            return {
                "enhanced_trade_manager": {
                    "is_running": self.is_running,
                    "statistics": {
                        "total_trades_processed": self.total_trades_processed,
                        "successful_executions": self.successful_executions,
                        "failed_executions": self.failed_executions,
                        "protection_recoveries": self.protection_recoveries,
                        "emergency_liquidations": self.emergency_liquidations
                    },
                    "overall_health": {
                        "all_positions_protected": all_protected,
                        "unprotected_positions": unprotected_count,
                        "needs_attention": unprotected_count > 0 or not all_protected
                    }
                },
                "bracket_order_manager": bracket_status,
                "position_protector": protection_status,
                "order_reconciler": reconciliation_status,
                "trade_state_manager": state_status
            }
            
        except Exception as e:
            return {"error": f"Status collection failed: {str(e)}"}
    
    def get_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get critical alerts requiring immediate attention"""
        alerts = []
        
        try:
            if not self.is_running:
                alerts.append({
                    "level": "critical",
                    "component": "enhanced_trade_manager",
                    "message": "Enhanced Trade Manager is not running",
                    "action_required": "Start the system immediately"
                })
                return alerts
            
            # Check for unprotected positions
            protection_status = self.position_protector.get_protection_status()
            unprotected_count = protection_status.get("unprotected_positions", 0)
            
            if unprotected_count > 0:
                alerts.append({
                    "level": "critical",
                    "component": "position_protector", 
                    "message": f"{unprotected_count} positions are UNPROTECTED",
                    "action_required": "Force protect all positions or emergency liquidate"
                })
            
            # Check for failed bracket orders
            bracket_status = self.bracket_manager.get_bracket_status()
            protection_failures = bracket_status.get("protection_failures", {})
            
            if protection_failures:
                alerts.append({
                    "level": "critical",
                    "component": "bracket_order_manager",
                    "message": f"Protection failures detected: {list(protection_failures.keys())}",
                    "action_required": "Review and fix protection failures"
                })
            
            # Check for reconciliation issues
            reconciliation_status = self.order_reconciler.get_reconciliation_status()
            positions_needing_attention = reconciliation_status.get("current_state", {}).get("positions_needing_attention", 0)
            
            if positions_needing_attention > 0:
                alerts.append({
                    "level": "warning",
                    "component": "order_reconciler",
                    "message": f"{positions_needing_attention} positions need reconciliation",
                    "action_required": "Review order-position alignment"
                })
            
        except Exception as e:
            alerts.append({
                "level": "critical",
                "component": "enhanced_trade_manager",
                "message": f"Alert generation failed: {str(e)}",
                "action_required": "Check system status"
            })
        
        return alerts
    
    # Legacy compatibility methods
    def process(self, data=None):
        """Legacy compatibility method"""
        if self.enable_legacy_compatibility:
            return self.get_comprehensive_status()
        return {"error": "Legacy compatibility disabled"}
    
    def add_trade(self, trading_pair: TradingPair) -> bool:
        """Legacy compatibility method - delegates to execute_trade"""
        if self.enable_legacy_compatibility:
            # Schedule async execution
            future = asyncio.create_task(self.execute_trade(trading_pair))
            return True
        return False
    
    def _create_execution_result(self, success: bool, message: str, data: Any) -> Dict[str, Any]:
        """Create standardized execution result"""
        return {
            "success": success,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
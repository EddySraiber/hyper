"""
Unit tests for trade pairs safety mechanisms

Tests comprehensive trade lifecycle management, position protection,
and failure recovery scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

# Import components to test
from algotrading_agent.trading.bracket_order_manager import BracketOrderManager, BracketOrder, BracketOrderStatus
from algotrading_agent.trading.position_protector import PositionProtector, UnprotectedPosition
from algotrading_agent.trading.trade_state_manager import TradeStateManager, ManagedTrade, TradeState
from algotrading_agent.trading.order_reconciler import OrderReconciler, PositionOrderState
from algotrading_agent.components.enhanced_trade_manager import EnhancedTradeManager
from algotrading_agent.components.decision_engine import TradingPair


class MockAlpacaClient:
    """Mock Alpaca client for testing"""
    
    def __init__(self):
        self.positions = []
        self.orders = []
        self.order_status_responses = {}
        self.validation_responses = {"valid": True, "errors": [], "warnings": []}
        self.market_open = True
        self.current_prices = {}
        
    async def get_positions(self):
        return self.positions
    
    async def get_orders(self, status=None):
        if status:
            return [o for o in self.orders if o["status"] == status]
        return self.orders
    
    async def get_order_status(self, order_id):
        return self.order_status_responses.get(order_id, {"status": "new", "filled_qty": 0, "filled_avg_price": 0})
    
    async def execute_trading_pair(self, trading_pair):
        return {"order_id": f"order_{trading_pair.symbol}_{int(datetime.utcnow().timestamp())}"}
    
    async def validate_trading_pair(self, trading_pair):
        return self.validation_responses
    
    async def get_current_price(self, symbol):
        return self.current_prices.get(symbol, 100.0)
    
    async def is_market_open(self):
        return self.market_open
    
    async def get_position_with_orders(self, symbol):
        position = next((p for p in self.positions if p["symbol"] == symbol), None)
        symbol_orders = [o for o in self.orders if o["symbol"] == symbol]
        
        return {
            "has_position": position is not None,
            "position": position,
            "orders": {
                "stop_loss_orders": [o for o in symbol_orders if "stop" in o.get("order_type", "").lower()],
                "take_profit_orders": [o for o in symbol_orders if o.get("order_type") == "limit"],
                "all_orders": symbol_orders
            }
        }
    
    async def update_stop_loss(self, symbol, stop_price):
        return {"success": True, "new_order_id": f"stop_{symbol}_{int(datetime.utcnow().timestamp())}"}
    
    async def update_take_profit(self, symbol, take_profit_price):
        return {"success": True, "new_order_id": f"tp_{symbol}_{int(datetime.utcnow().timestamp())}"}
    
    async def close_position(self, symbol, percentage=100.0):
        return {"order_id": f"close_{symbol}_{int(datetime.utcnow().timestamp())}"}
    
    async def cancel_order(self, order_id):
        return True


def create_test_trading_pair(symbol="AAPL", action="buy", quantity=10):
    """Create a test trading pair"""
    return TradingPair(
        symbol=symbol,
        action=action,
        quantity=quantity,
        entry_price=150.0,
        stop_loss=142.5,  # 5% below for buy
        take_profit=165.0,  # 10% above for buy
        confidence=0.75,
        reasoning="Test trade"
    )


class TestBracketOrderManager:
    """Test bracket order management system"""
    
    @pytest.fixture
    def alpaca_client(self):
        return MockAlpacaClient()
    
    @pytest.fixture
    def bracket_manager(self, alpaca_client):
        config = {
            "max_concurrent_brackets": 5,
            "protection_check_interval": 30,
            "emergency_liquidation_enabled": True,
            "max_protection_failures": 3
        }
        return BracketOrderManager(alpaca_client, config)
    
    def test_bracket_order_creation(self):
        """Test bracket order object creation"""
        trading_pair = create_test_trading_pair()
        bracket = BracketOrder(trading_pair, "test_bracket_1")
        
        assert bracket.trading_pair.symbol == "AAPL"
        assert bracket.status == BracketOrderStatus.PLANNED
        assert not bracket.is_protected()
        assert not bracket.has_position()
        assert not bracket.needs_protection()
    
    @pytest.mark.asyncio
    async def test_successful_bracket_submission(self, bracket_manager, alpaca_client):
        """Test successful bracket order submission"""
        trading_pair = create_test_trading_pair()
        
        success, message, bracket = await bracket_manager.submit_bracket_order(trading_pair)
        
        assert success
        assert bracket is not None
        assert bracket.status == BracketOrderStatus.ACTIVE
        assert bracket.entry_order_id is not None
    
    @pytest.mark.asyncio
    async def test_bracket_validation_failure(self, bracket_manager, alpaca_client):
        """Test bracket submission with validation failure"""
        alpaca_client.validation_responses = {
            "valid": False,
            "errors": ["Insufficient buying power"],
            "warnings": []
        }
        
        trading_pair = create_test_trading_pair()
        success, message, bracket = await bracket_manager.submit_bracket_order(trading_pair)
        
        assert not success
        assert bracket.status == BracketOrderStatus.FAILED
        assert "validation failed" in message.lower()
    
    @pytest.mark.asyncio
    async def test_protection_monitoring(self, bracket_manager, alpaca_client):
        """Test protection status monitoring"""
        # Setup position without protection
        alpaca_client.positions = [{
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 0.0,
            "side": "long"
        }]
        
        # Create bracket order
        trading_pair = create_test_trading_pair()
        success, message, bracket = await bracket_manager.submit_bracket_order(trading_pair)
        
        # Monitor bracket (should detect unprotected position)
        await bracket_manager._monitor_bracket(bracket)
        
        assert bracket.needs_protection()


class TestPositionProtector:
    """Test position protection system"""
    
    @pytest.fixture
    def alpaca_client(self):
        return MockAlpacaClient()
    
    @pytest.fixture
    def position_protector(self, alpaca_client):
        config = {
            "check_interval": 30,
            "max_protection_attempts": 5,
            "emergency_liquidation_enabled": True,
            "default_stop_loss_pct": 0.05,
            "default_take_profit_pct": 0.10
        }
        return PositionProtector(alpaca_client, config)
    
    def test_unprotected_position_creation(self):
        """Test creation of unprotected position object"""
        position_data = {
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 50.0,
            "side": "long"
        }
        
        unprotected = UnprotectedPosition("AAPL", position_data)
        
        assert unprotected.symbol == "AAPL"
        assert unprotected.quantity == 10
        assert unprotected.suggested_stop_loss == 142.5  # 5% below 150
        assert unprotected.suggested_take_profit == 165.0  # 10% above 150
    
    @pytest.mark.asyncio
    async def test_unprotected_position_detection(self, position_protector, alpaca_client):
        """Test detection of unprotected positions"""
        # Setup unprotected position
        alpaca_client.positions = [{
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 0.0,
            "side": "long"
        }]
        
        alpaca_client.orders = []  # No protective orders
        
        await position_protector._scan_and_protect_positions()
        
        assert len(position_protector.unprotected_positions) == 1
        assert "AAPL" in position_protector.unprotected_positions
    
    @pytest.mark.asyncio
    async def test_protection_attempt(self, position_protector, alpaca_client):
        """Test attempt to protect unprotected position"""
        # Setup unprotected position
        position_data = {
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 0.0,
            "side": "long"
        }
        
        unprotected = UnprotectedPosition("AAPL", position_data)
        position_protector.unprotected_positions["AAPL"] = unprotected
        
        await position_protector._attempt_protection("AAPL")
        
        assert unprotected.protection_attempts == 1
        assert position_protector.total_protection_attempts == 1
    
    @pytest.mark.asyncio
    async def test_emergency_liquidation(self, position_protector, alpaca_client):
        """Test emergency liquidation of unprotectable position"""
        # Setup position that fails protection attempts
        position_data = {
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": -200.0,  # Large loss
            "side": "long"
        }
        
        unprotected = UnprotectedPosition("AAPL", position_data)
        unprotected.protection_attempts = 5  # Max attempts reached
        position_protector.unprotected_positions["AAPL"] = unprotected
        
        await position_protector._emergency_liquidate("AAPL")
        
        assert position_protector.emergency_liquidations == 1
        assert "AAPL" not in position_protector.unprotected_positions


class TestTradeStateManager:
    """Test trade state management system"""
    
    @pytest.fixture
    def alpaca_client(self):
        return MockAlpacaClient()
    
    @pytest.fixture
    def state_manager(self, alpaca_client):
        config = {
            "state_check_interval": 30,
            "protection_check_interval": 60,
            "max_trade_age_hours": 48,
            "emergency_liquidation_enabled": True
        }
        return TradeStateManager(alpaca_client, config)
    
    def test_managed_trade_creation(self):
        """Test managed trade object creation"""
        trading_pair = create_test_trading_pair()
        trade = ManagedTrade("test_trade_1", trading_pair)
        
        assert trade.state == TradeState.PLANNED
        assert trade.is_active()
        assert not trade.has_position()
        assert not trade.needs_protection_check()
    
    def test_state_transitions(self):
        """Test valid state transitions"""
        trading_pair = create_test_trading_pair()
        trade = ManagedTrade("test_trade_1", trading_pair)
        
        # Valid transitions
        assert trade.transition_to(TradeState.VALIDATING, "Starting validation")
        assert trade.state == TradeState.VALIDATING
        
        assert trade.transition_to(TradeState.SUBMITTING, "Validation passed")
        assert trade.state == TradeState.SUBMITTING
        
        # Invalid transition (should fail)
        assert not trade.transition_to(TradeState.COMPLETED_PROFIT, "Invalid transition")
        assert trade.state == TradeState.SUBMITTING  # Should remain unchanged
    
    @pytest.mark.asyncio
    async def test_trade_lifecycle_progression(self, state_manager, alpaca_client):
        """Test complete trade lifecycle progression"""
        trading_pair = create_test_trading_pair()
        trade_id = await state_manager.create_managed_trade(trading_pair)
        
        trade = state_manager.active_trades[trade_id]
        
        # Should progress through validation and submission
        assert trade.state in [TradeState.VALIDATING, TradeState.SUBMITTING, TradeState.ENTRY_PENDING]
    
    @pytest.mark.asyncio
    async def test_protection_failure_handling(self, state_manager, alpaca_client):
        """Test handling of protection failures"""
        trading_pair = create_test_trading_pair()
        trade = ManagedTrade("test_trade_1", trading_pair)
        trade.state = TradeState.PROTECTION_FAILED
        trade.protection_check_failures = 3  # Max failures reached
        
        state_manager.active_trades["test_trade_1"] = trade
        
        await state_manager._handle_protection_failed_state(trade)
        
        # Should trigger emergency liquidation
        assert trade.state == TradeState.EMERGENCY_LIQUIDATING


class TestOrderReconciler:
    """Test order-position reconciliation system"""
    
    @pytest.fixture
    def alpaca_client(self):
        return MockAlpacaClient()
    
    @pytest.fixture
    def order_reconciler(self, alpaca_client):
        config = {
            "reconciliation_interval": 60,
            "stale_order_threshold_hours": 24,
            "auto_cleanup_enabled": True
        }
        return OrderReconciler(alpaca_client, config)
    
    @pytest.mark.asyncio
    async def test_position_order_reconciliation(self, order_reconciler, alpaca_client):
        """Test reconciliation of positions with orders"""
        # Setup position with incomplete protection
        position = {
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 0.0,
            "side": "long"
        }
        
        orders = [
            {"symbol": "AAPL", "order_type": "stop_loss", "status": "new", "quantity": 10, "submitted_at": datetime.utcnow().isoformat()}
            # Missing take-profit order
        ]
        
        alpaca_client.positions = [position]
        alpaca_client.orders = orders
        
        state = await order_reconciler._reconcile_position("AAPL", position, orders)
        
        assert state.has_position
        assert state.has_stop_protection
        assert not state.has_limit_protection
        assert not state.is_fully_protected
        assert state.needs_attention
        assert "Missing take-profit protection" in state.reconciliation_issues
    
    @pytest.mark.asyncio
    async def test_orphaned_order_detection(self, order_reconciler, alpaca_client):
        """Test detection of orphaned orders"""
        # Setup orders without corresponding positions
        alpaca_client.positions = []  # No positions
        alpaca_client.orders = [
            {"symbol": "AAPL", "order_type": "limit", "status": "new", "order_id": "orphaned_1"},
            {"symbol": "MSFT", "order_type": "stop_loss", "status": "accepted", "order_id": "orphaned_2"}
        ]
        
        await order_reconciler._check_for_orphaned_orders(alpaca_client.orders, set())
        
        # Should detect and auto-cleanup orphaned orders
        assert order_reconciler.issues_resolved == 2  # Both orders cancelled


class TestEnhancedTradeManager:
    """Test enhanced trade manager integration"""
    
    @pytest.fixture
    def alpaca_client(self):
        return MockAlpacaClient()
    
    @pytest.fixture
    def enhanced_manager(self, alpaca_client):
        config = {
            "max_concurrent_trades": 10,
            "enable_legacy_compatibility": True
        }
        manager = EnhancedTradeManager(config)
        manager.alpaca_client = alpaca_client
        return manager
    
    def test_component_initialization(self, enhanced_manager):
        """Test initialization of all components"""
        enhanced_manager.start()
        
        assert enhanced_manager.is_running
        assert enhanced_manager.bracket_manager is not None
        assert enhanced_manager.position_protector is not None
        assert enhanced_manager.order_reconciler is not None
        assert enhanced_manager.state_manager is not None
    
    @pytest.mark.asyncio
    async def test_integrated_trade_execution(self, enhanced_manager):
        """Test integrated trade execution"""
        enhanced_manager.start()
        trading_pair = create_test_trading_pair()
        
        result = await enhanced_manager.execute_trade(trading_pair)
        
        assert result["success"]
        assert "bracket_id" in result["data"]
        assert "trade_id" in result["data"]
        assert enhanced_manager.successful_executions == 1
    
    @pytest.mark.asyncio
    async def test_critical_alerts_generation(self, enhanced_manager, alpaca_client):
        """Test generation of critical alerts"""
        enhanced_manager.start()
        
        # Simulate unprotected positions
        alpaca_client.positions = [{
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": -100.0,
            "side": "long"
        }]
        
        # Let position protector detect unprotected position
        await enhanced_manager.position_protector._scan_and_protect_positions()
        
        alerts = enhanced_manager.get_critical_alerts()
        
        assert len(alerts) > 0
        assert any("UNPROTECTED" in alert["message"] for alert in alerts)


class TestFailureRecoveryScenarios:
    """Test failure recovery and edge cases"""
    
    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self):
        """Test recovery from network timeouts during order submission"""
        alpaca_client = MockAlpacaClient()
        
        # Mock network timeout
        async def timeout_execute_trading_pair(trading_pair):
            raise asyncio.TimeoutError("Network timeout")
        
        alpaca_client.execute_trading_pair = timeout_execute_trading_pair
        
        config = {"max_concurrent_brackets": 5}
        bracket_manager = BracketOrderManager(alpaca_client, config)
        
        trading_pair = create_test_trading_pair()
        success, message, bracket = await bracket_manager.submit_bracket_order(trading_pair)
        
        assert not success
        assert bracket.status == BracketOrderStatus.FAILED
        assert "timeout" in message.lower() or "exception" in message.lower()
    
    @pytest.mark.asyncio
    async def test_partial_fill_handling(self):
        """Test handling of partially filled orders"""
        alpaca_client = MockAlpacaClient()
        alpaca_client.order_status_responses = {
            "test_order": {
                "status": "partially_filled",
                "filled_qty": 5,  # Only half filled
                "filled_avg_price": 150.0,
                "quantity": 10
            }
        }
        
        config = {"state_check_interval": 30}
        state_manager = TradeStateManager(alpaca_client, config)
        
        trading_pair = create_test_trading_pair()
        trade = ManagedTrade("test_trade", trading_pair)
        trade.entry_order_id = "test_order"
        trade.state = TradeState.ENTRY_PENDING
        
        # Should remain in pending state for partial fills
        updated = await state_manager._check_entry_order(trade)
        assert not updated  # No state change for partial fill
        assert trade.state == TradeState.ENTRY_PENDING
    
    @pytest.mark.asyncio
    async def test_stale_order_cleanup(self):
        """Test cleanup of stale orders"""
        alpaca_client = MockAlpacaClient()
        
        # Create stale order (25 hours old)
        stale_time = datetime.utcnow() - timedelta(hours=25)
        alpaca_client.orders = [{
            "symbol": "AAPL",
            "order_type": "limit",
            "status": "new",
            "order_id": "stale_order",
            "submitted_at": stale_time.isoformat() + "Z"
        }]
        
        config = {"stale_order_threshold_hours": 24, "auto_cleanup_enabled": True}
        reconciler = OrderReconciler(alpaca_client, config)
        
        position = {"symbol": "AAPL", "quantity": 10, "market_value": 1500.0, "unrealized_pl": 0.0}
        state = await reconciler._reconcile_position("AAPL", position, alpaca_client.orders)
        
        assert "Stale order" in str(state.reconciliation_issues)
    
    @pytest.mark.asyncio
    async def test_market_closure_handling(self):
        """Test handling of trades during market closure"""
        alpaca_client = MockAlpacaClient()
        alpaca_client.market_open = False
        
        config = {"protection_check_interval": 30}
        protector = PositionProtector(alpaca_client, config)
        
        # Should skip protection attempts during market closure
        alpaca_client.positions = [{
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 0.0,
            "side": "long"
        }]
        
        await protector._scan_and_protect_positions()
        # Should not create unprotected position during market closure
        # (Implementation would need to check market hours)


# Integration test scenarios
class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_protected_trade_lifecycle(self):
        """Test complete lifecycle of a properly protected trade"""
        alpaca_client = MockAlpacaClient()
        
        # Setup successful order progression
        alpaca_client.order_status_responses = {
            "entry_order": {"status": "filled", "filled_qty": 10, "filled_avg_price": 150.0, "filled_at": datetime.utcnow().isoformat()},
            "stop_order": {"status": "new", "quantity": 10},
            "tp_order": {"status": "new", "quantity": 10}
        }
        
        # Enhanced trade manager setup
        config = {"max_concurrent_trades": 5}
        manager = EnhancedTradeManager(config)
        manager.alpaca_client = alpaca_client
        manager.start()
        
        trading_pair = create_test_trading_pair()
        
        # Execute trade
        result = await manager.execute_trade(trading_pair)
        assert result["success"]
        
        # Simulate position creation with protection
        alpaca_client.positions = [{
            "symbol": "AAPL",
            "quantity": 10,
            "market_value": 1500.0,
            "unrealized_pl": 0.0,
            "side": "long"
        }]
        
        alpaca_client.orders = [
            {"symbol": "AAPL", "order_type": "stop_loss", "status": "new", "quantity": 10, "submitted_at": datetime.utcnow().isoformat()},
            {"symbol": "AAPL", "order_type": "limit", "status": "new", "quantity": 10, "submitted_at": datetime.utcnow().isoformat()}
        ]
        
        # Check comprehensive status
        status = manager.get_comprehensive_status()
        
        assert status["enhanced_trade_manager"]["is_running"]
        assert status["enhanced_trade_manager"]["statistics"]["successful_executions"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
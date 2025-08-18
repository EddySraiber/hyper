#!/usr/bin/env python3
"""
Comprehensive Bracket Order Manager Testing

Tests atomic bracket order operations, failure modes,
and integration with position protection systems.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from algotrading_agent.trading.bracket_order_manager import BracketOrderManager, BracketOrder
from algotrading_agent.trading.alpaca_client import AlpacaClient


class TestBracketOrderManagerComprehensive:
    """Comprehensive Bracket Order Manager testing"""
    
    @pytest.fixture
    async def bracket_manager(self):
        """Create Bracket Order Manager for testing"""
        config = {
            "default_stop_loss_pct": 0.05,
            "default_take_profit_pct": 0.10,
            "bracket_timeout_seconds": 60,
            "max_retry_attempts": 3
        }
        
        mock_client = AsyncMock(spec=AlpacaClient)
        return BracketOrderManager(config, mock_client)
    
    @pytest.mark.asyncio
    async def test_atomic_bracket_creation(self, bracket_manager):
        """Test atomic creation of complete bracket orders"""
        bracket_order = BracketOrder(
            symbol="AAPL",
            quantity=100,
            side="buy",
            entry_price=150.00,
            stop_loss_price=142.50,
            take_profit_price=165.00
        )
        
        # Mock successful order submissions
        bracket_manager.alpaca_client.submit_order.side_effect = [
            {"id": "entry_123", "status": "new"},
            {"id": "stop_456", "status": "new"},
            {"id": "profit_789", "status": "new"}
        ]
        
        result = await bracket_manager.create_bracket_order(bracket_order)
        
        assert result.success == True
        assert result.entry_order_id == "entry_123"
        assert result.stop_loss_order_id == "stop_456"
        assert result.take_profit_order_id == "profit_789"
        assert bracket_manager.alpaca_client.submit_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_partial_failure_rollback(self, bracket_manager):
        """Test rollback when bracket order creation partially fails"""
        bracket_order = BracketOrder(
            symbol="TSLA",
            quantity=50,
            side="sell",
            entry_price=250.00,
            stop_loss_price=262.50,
            take_profit_price=225.00
        )
        
        # Simulate partial failure (entry succeeds, stop loss fails)
        bracket_manager.alpaca_client.submit_order.side_effect = [
            {"id": "entry_abc", "status": "new"},    # Entry succeeds
            Exception("Stop loss order failed"),     # Stop loss fails
        ]
        
        bracket_manager.alpaca_client.cancel_order.return_value = {"status": "cancelled"}
        
        result = await bracket_manager.create_bracket_order(bracket_order)
        
        assert result.success == False
        assert result.error_message is not None
        # Should have cancelled the entry order
        bracket_manager.alpaca_client.cancel_order.assert_called_with("entry_abc")
    
    @pytest.mark.asyncio
    async def test_partial_fill_scenarios(self, bracket_manager):
        """Test handling of partial fills in bracket orders"""
        bracket_order = BracketOrder(
            symbol="NVDA",
            quantity=100,
            side="buy",
            entry_price=500.00,
            stop_loss_price=475.00,
            take_profit_price=550.00
        )
        
        # Simulate partial fill on entry order
        bracket_manager.alpaca_client.submit_order.side_effect = [
            {"id": "entry_def", "status": "partially_filled", "filled_qty": 60},
            {"id": "stop_ghi", "status": "new"},
            {"id": "profit_jkl", "status": "new"}
        ]
        
        result = await bracket_manager.create_bracket_order(bracket_order)
        
        # Should adjust protective order quantities to match filled amount
        assert result.success == True
        # Verify stop loss and take profit orders were adjusted for 60 shares
        stop_loss_call = bracket_manager.alpaca_client.submit_order.call_args_list[1]
        take_profit_call = bracket_manager.alpaca_client.submit_order.call_args_list[2]
        
        assert stop_loss_call[1]["qty"] == 60
        assert take_profit_call[1]["qty"] == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

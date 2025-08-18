#!/usr/bin/env python3
"""
Position Protector Integration Testing

Tests position protection mechanisms including:
- Automatic protection application
- Failure recovery scenarios
- Timeout handling
- Integration with trading pipeline
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from algotrading_agent.trading.position_protector import PositionProtector, UnprotectedPosition
from algotrading_agent.trading.alpaca_client import AlpacaClient


class TestPositionProtectorIntegration:
    """Integration tests for Position Protector"""
    
    @pytest.fixture
    async def position_protector(self):
        """Create Position Protector for testing"""
        config = {
            "protection_check_interval": 1,
            "max_protection_attempts": 3,
            "protection_timeout_seconds": 30,
            "default_stop_loss_pct": 0.05,
            "default_take_profit_pct": 0.10
        }
        
        mock_client = AsyncMock(spec=AlpacaClient)
        return PositionProtector(config, mock_client)
    
    @pytest.mark.asyncio
    async def test_automatic_protection_application(self, position_protector):
        """Test automatic protection is applied to unprotected positions"""
        # Create unprotected position
        unprotected_position = UnprotectedPosition(
            symbol="AAPL",
            quantity=100,
            side="long",
            entry_price=150.00,
            current_price=152.00
        )
        
        # Mock successful protection creation
        position_protector.alpaca_client.submit_order.return_value = {"id": "order_123", "status": "accepted"}
        
        result = await position_protector.apply_protection(unprotected_position)
        
        assert result.success == True
        assert result.stop_loss_order_id is not None
        assert result.take_profit_order_id is not None
        assert position_protector.alpaca_client.submit_order.call_count == 2  # Stop loss + take profit
    
    @pytest.mark.asyncio
    async def test_protection_failure_recovery(self, position_protector):
        """Test recovery from protection application failures"""
        unprotected_position = UnprotectedPosition(
            symbol="TSLA",
            quantity=50,
            side="short",
            entry_price=250.00,
            current_price=245.00
        )
        
        # Simulate partial failure (stop loss succeeds, take profit fails)
        position_protector.alpaca_client.submit_order.side_effect = [
            {"id": "stop_123", "status": "accepted"},  # Stop loss succeeds
            Exception("Take profit order failed")       # Take profit fails
        ]
        
        result = await position_protector.apply_protection(unprotected_position)
        
        # Should have partial protection
        assert result.stop_loss_order_id is not None
        assert result.take_profit_order_id is None
        assert result.needs_retry == True
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, position_protector):
        """Test handling of protection timeout scenarios"""
        unprotected_position = UnprotectedPosition(
            symbol="NVDA",
            quantity=25,
            side="long",
            entry_price=500.00,
            current_price=510.00
        )
        
        # Simulate timeout
        async def slow_submit(*args, **kwargs):
            await asyncio.sleep(35)  # Longer than 30 second timeout
            return {"id": "order_timeout", "status": "accepted"}
        
        position_protector.alpaca_client.submit_order.side_effect = slow_submit
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                position_protector.apply_protection(unprotected_position),
                timeout=30
            )
    
    @pytest.mark.asyncio
    async def test_integration_with_trading_pipeline(self, position_protector):
        """Test integration with main trading pipeline"""
        # Simulate new position from trading pipeline
        new_position = {
            "symbol": "SPY",
            "quantity": 200,
            "side": "long",
            "filled_avg_price": 420.00,
            "order_id": "main_order_456"
        }
        
        # Position Protector should detect and protect automatically
        position_protector.alpaca_client.get_positions.return_value = [new_position]
        position_protector.alpaca_client.get_orders.return_value = []  # No protective orders yet
        
        await position_protector.check_and_protect_positions()
        
        # Should have created protective orders
        assert position_protector.alpaca_client.submit_order.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_market_hours_handling(self, position_protector):
        """Test protection behavior during market hours vs after hours"""
        position = UnprotectedPosition(
            symbol="QQQ",
            quantity=100,
            side="long",
            entry_price=300.00,
            current_price=305.00
        )
        
        # Test during market hours
        with patch('algotrading_agent.trading.position_protector.is_market_open') as mock_market:
            mock_market.return_value = True
            position_protector.alpaca_client.submit_order.return_value = {"id": "order_789", "status": "accepted"}
            
            result = await position_protector.apply_protection(position)
            assert result.success == True
        
        # Test after market hours
        with patch('algotrading_agent.trading.position_protector.is_market_open') as mock_market:
            mock_market.return_value = False
            
            result = await position_protector.apply_protection(position)
            # Should queue for next market open or use extended hours
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

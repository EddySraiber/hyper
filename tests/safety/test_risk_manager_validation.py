#!/usr/bin/env python3
"""
Risk Manager Validation Testing

Tests portfolio protection, position sizing,
and risk limit enforcement mechanisms.
"""

import pytest
from unittest.mock import MagicMock, patch

from algotrading_agent.components.risk_manager import RiskManager


class TestRiskManagerValidation:
    """Risk Manager validation tests"""
    
    @pytest.fixture
    def risk_manager(self):
        """Create Risk Manager for testing"""
        config = {
            "max_portfolio_value": 100000,
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.02,
            "stop_loss_pct": 0.05,
            "position_concentration_limit": 0.10
        }
        return RiskManager(config)
    
    def test_position_sizing_validation(self, risk_manager):
        """Test position sizing stays within risk limits"""
        portfolio_value = 100000
        
        # Test normal position sizing
        position_size = risk_manager.calculate_position_size("AAPL", 150.00, portfolio_value)
        max_allowed = portfolio_value * 0.05  # 5% limit
        
        assert position_size * 150.00 <= max_allowed
    
    def test_daily_loss_limit_enforcement(self, risk_manager):
        """Test daily loss limit prevents over-exposure"""
        # Simulate existing daily loss
        risk_manager.daily_pnl = -1800  # $1,800 loss (1.8% of $100k)
        
        # Should reject new risky trade that could exceed 2% daily limit
        trade_decision = {
            "symbol": "TSLA",
            "action": "buy",
            "confidence": 0.7,
            "potential_loss": 500  # Would push total loss to 2.3%
        }
        
        validation = risk_manager.validate_trade_decision(trade_decision, 100000)
        assert validation.approved == False
        assert "daily loss limit" in validation.rejection_reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

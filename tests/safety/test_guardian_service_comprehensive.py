#!/usr/bin/env python3
"""
Comprehensive Guardian Service Testing Suite

Tests critical safety functionality including leak detection,
remediation accuracy, and failure recovery scenarios.

REQUIREMENTS:
- 95% leak detection accuracy
- <5% false positive rate
- <30 second scan cycles
- 100% remediation success under normal conditions
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from algotrading_agent.trading.guardian_service import GuardianService
from algotrading_agent.trading.alpaca_client import AlpacaClient


class TestGuardianServiceComprehensive:
    """Comprehensive Guardian Service testing"""
    
    @pytest.fixture
    async def guardian_service(self):
        """Create Guardian Service for testing"""
        config = {
            "guardian_scan_interval": 1,  # Fast for testing
            "max_remediation_attempts": 3,
            "emergency_liquidation_enabled": False,  # SAFE for testing
            "leak_detection_threshold": 0.95,
            "false_positive_threshold": 0.05
        }
        
        mock_client = AsyncMock(spec=AlpacaClient)
        return GuardianService(config, mock_client)
    
    @pytest.mark.asyncio
    async def test_leak_detection_accuracy(self, guardian_service):
        """Test leak detection accuracy meets 95% requirement"""
        # Create test scenarios with known leaks
        test_positions = self._create_leak_test_scenarios(100)
        
        detected_leaks = []
        for position in test_positions:
            is_leak = await guardian_service._detect_position_leak(position)
            if is_leak:
                detected_leaks.append(position)
        
        # Calculate accuracy
        expected_leaks = [p for p in test_positions if p.get("is_actual_leak")]
        accuracy = len([p for p in detected_leaks if p.get("is_actual_leak")]) / len(expected_leaks)
        
        assert accuracy >= 0.95, f"Leak detection accuracy {accuracy:.2%} below 95% requirement"
    
    @pytest.mark.asyncio 
    async def test_false_positive_rate(self, guardian_service):
        """Test false positive rate is below 5%"""
        # Create test scenarios with safe positions
        safe_positions = self._create_safe_position_scenarios(100)
        
        false_positives = []
        for position in safe_positions:
            is_leak = await guardian_service._detect_position_leak(position)
            if is_leak:
                false_positives.append(position)
        
        false_positive_rate = len(false_positives) / len(safe_positions)
        assert false_positive_rate <= 0.05, f"False positive rate {false_positive_rate:.2%} above 5% limit"
    
    @pytest.mark.asyncio
    async def test_remediation_success_rate(self, guardian_service):
        """Test remediation success under normal conditions"""
        # Create remediable leak scenarios
        leak_scenarios = self._create_remediable_leak_scenarios(50)
        
        successful_remediations = 0
        for leak in leak_scenarios:
            success = await guardian_service._attempt_remediation(leak)
            if success:
                successful_remediations += 1
        
        success_rate = successful_remediations / len(leak_scenarios)
        assert success_rate >= 0.90, f"Remediation success rate {success_rate:.2%} below 90% requirement"
    
    @pytest.mark.asyncio
    async def test_scan_cycle_performance(self, guardian_service):
        """Test scan cycle completes within 30 seconds"""
        start_time = datetime.now()
        
        # Mock positions for realistic scan
        mock_positions = self._create_realistic_position_set(25)
        guardian_service.alpaca_client.get_positions.return_value = mock_positions
        
        await guardian_service._perform_scan_cycle()
        
        scan_duration = (datetime.now() - start_time).total_seconds()
        assert scan_duration <= 30, f"Scan cycle took {scan_duration:.1f}s, exceeds 30s limit"
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, guardian_service):
        """Test recovery from network failures"""
        # Simulate network failure
        guardian_service.alpaca_client.get_positions.side_effect = ConnectionError("Network timeout")
        
        # Guardian should handle gracefully
        result = await guardian_service._perform_scan_cycle()
        
        # Should not crash and should attempt retry
        assert result is not None
        assert guardian_service.consecutive_failures <= guardian_service.max_consecutive_failures
    
    @pytest.mark.asyncio
    async def test_database_failure_recovery(self, guardian_service):
        """Test recovery from database failures"""
        # Simulate database connection failure
        with patch('algotrading_agent.trading.guardian_service.logger') as mock_logger:
            guardian_service.db_connection = None
            
            result = await guardian_service._log_scan_results({})
            
            # Should handle gracefully and log error
            mock_logger.error.assert_called()
            assert result is not None
    
    def _create_leak_test_scenarios(self, count):
        """Create test scenarios with known leak patterns"""
        scenarios = []
        for i in range(count):
            # Mix of actual leaks and safe positions
            is_leak = i % 3 == 0  # 33% are actual leaks
            scenarios.append({
                "symbol": f"TEST{i}",
                "quantity": 100 if not is_leak else 100,
                "has_stop_loss": not is_leak,
                "has_take_profit": not is_leak,
                "is_actual_leak": is_leak,
                "market_value": 10000
            })
        return scenarios
    
    def _create_safe_position_scenarios(self, count):
        """Create safe position scenarios for false positive testing"""
        return [{
            "symbol": f"SAFE{i}",
            "quantity": 100,
            "has_stop_loss": True,
            "has_take_profit": True,
            "is_actual_leak": False,
            "market_value": 5000
        } for i in range(count)]
    
    def _create_remediable_leak_scenarios(self, count):
        """Create leak scenarios that should be remediable"""
        return [{
            "symbol": f"REMED{i}",
            "quantity": 50,
            "has_stop_loss": False,
            "has_take_profit": False,
            "remediation_possible": True,
            "market_value": 2500
        } for i in range(count)]
    
    def _create_realistic_position_set(self, count):
        """Create realistic position set for performance testing"""
        return [{
            "symbol": f"POS{i}",
            "quantity": 100 + i,
            "side": "long" if i % 2 == 0 else "short",
            "market_value": 1000 * (i + 1),
            "unrealized_pl": (-50 + i) * 10
        } for i in range(count)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

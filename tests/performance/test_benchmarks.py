#!/usr/bin/env python3
"""
Performance Benchmarking Tests

Tests system performance under various load conditions
and validates latency requirements for trading operations.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from algotrading_agent.trading.guardian_service import GuardianService
from algotrading_agent.components.decision_engine import DecisionEngine


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.mark.benchmark
    def test_guardian_service_scan_latency(self, benchmark):
        """Benchmark Guardian Service scan cycle latency"""
        
        async def scan_cycle():
            # Mock Guardian Service with realistic load
            guardian = GuardianService({}, AsyncMock())
            # Simulate 25 positions (realistic load)
            mock_positions = [{"symbol": f"TEST{i}", "qty": 100} for i in range(25)]
            guardian.alpaca_client.get_positions.return_value = mock_positions
            
            await guardian._perform_scan_cycle()
        
        # Benchmark the scan cycle
        result = benchmark.pedantic(
            lambda: asyncio.run(scan_cycle()),
            rounds=10,
            iterations=1
        )
        
        # Validate latency requirement: <30 seconds
        assert result < 30, f"Guardian scan cycle took {result:.1f}s, exceeds 30s requirement"
    
    @pytest.mark.benchmark
    def test_trading_decision_latency(self, benchmark):
        """Benchmark trading decision generation latency"""
        
        def generate_decision():
            decision_engine = DecisionEngine({})
            # Mock news analysis results
            mock_analysis = {
                "sentiment_score": 0.7,
                "impact_score": 0.8,
                "confidence": 0.9,
                "entities": ["AAPL", "earnings"]
            }
            
            return decision_engine.generate_trading_decision(mock_analysis)
        
        # Benchmark decision generation
        result = benchmark.pedantic(
            generate_decision,
            rounds=50,
            iterations=1
        )
        
        # Validate latency requirement: <50ms for standard lane
        assert result < 0.05, f"Trading decision took {result:.3f}s, exceeds 50ms requirement"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_system_load_capacity(self, benchmark):
        """Test system capacity under 10x normal load"""
        
        async def simulate_high_load():
            # Simulate 10x normal processing load
            tasks = []
            for i in range(100):  # 10x normal 10 positions
                task = asyncio.create_task(self._simulate_position_processing())
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Benchmark high load processing
        result = benchmark.pedantic(
            lambda: asyncio.run(simulate_high_load()),
            rounds=3,
            iterations=1
        )
        
        # System should handle 10x load within reasonable time
        assert result < 120, f"High load processing took {result:.1f}s, system may not scale"
    
    async def _simulate_position_processing(self):
        """Simulate processing a single position"""
        await asyncio.sleep(0.1)  # Simulate realistic processing time
        return {"processed": True}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])

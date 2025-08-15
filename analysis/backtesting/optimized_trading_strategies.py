#!/usr/bin/env python3
"""
Optimized Trading Strategies for Real-World Friction Costs
Tax-efficient, execution-aware strategies designed to overcome 47-49% friction cost burden

Author: Claude Code (Anthropic AI Assistant)
Date: August 15, 2025
Task: 9 - System Optimization for Real-World Friction Costs
Next Task: 10 - Final Recommendations and Deployment
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

# Import our realistic backtesting framework
from enhanced_realistic_backtest import EnhancedRealisticBacktester, RealisticTradeResult, RealisticBacktestMetrics
from realistic_commission_models import BrokerType, AssetType, TradeInfo, RealWorldCostCalculator


class OptimizationStrategy(Enum):
    """Different optimization approaches"""
    TAX_OPTIMIZED = "tax_optimized"           # Extend holding periods, tax-loss harvesting
    EXECUTION_OPTIMIZED = "execution_optimized"  # Better timing, reduced slippage
    FREQUENCY_OPTIMIZED = "frequency_optimized"  # Fewer, higher-conviction trades
    HYBRID_OPTIMIZED = "hybrid_optimized"     # Combined approach


@dataclass
class OptimizedTradeSignal:
    """Enhanced trade signal with optimization parameters"""
    # Original signal data
    symbol: str
    action: str
    base_quantity: float
    price: float
    confidence: float
    hype_score: float
    velocity_level: str
    
    # Optimization parameters
    holding_period_target: int = 1  # Target holding period in days
    tax_efficiency_score: float = 0.0  # 0-100, higher is more tax efficient
    execution_urgency: str = "normal"  # "low", "normal", "high", "critical"
    conviction_level: str = "medium"   # "low", "medium", "high", "very_high"
    
    # Risk management
    max_position_pct: float = 0.05
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    
    # Execution optimization
    preferred_order_type: str = "limit"
    max_execution_delay_ms: int = 5000
    slippage_tolerance_bps: float = 25.0


class TaxOptimizedStrategy:
    """
    Tax-optimized trading strategy focusing on reducing the dominant friction cost
    """
    
    def __init__(self, min_holding_period: int = 31, target_ltcg_ratio: float = 0.30):
        self.min_holding_period = min_holding_period  # Avoid wash sales
        self.target_ltcg_ratio = target_ltcg_ratio     # Target % of long-term gains
        self.open_positions = {}                       # Track holding periods
        self.loss_harvest_opportunities = []           # Track tax-loss harvesting
        self.wash_sale_tracker = {}                    # Avoid wash sales
        
        self.logger = logging.getLogger("tax_optimized_strategy")
    
    def optimize_trade_signal(self, signal: OptimizedTradeSignal) -> OptimizedTradeSignal:
        """Optimize trade signal for tax efficiency"""
        
        optimized_signal = OptimizedTradeSignal(**signal.__dict__)
        
        # 1. Extend holding periods for tax efficiency
        if signal.action == "buy":
            optimized_signal.holding_period_target = self._calculate_optimal_holding_period(signal)
        
        # 2. Adjust position sizing for tax-loss harvesting opportunities
        optimized_signal.base_quantity = self._adjust_for_tax_harvesting(signal)
        
        # 3. Set tax efficiency score
        optimized_signal.tax_efficiency_score = self._calculate_tax_efficiency_score(signal)
        
        # 4. Adjust conviction based on tax implications
        optimized_signal.conviction_level = self._adjust_conviction_for_taxes(signal)
        
        self.logger.debug(f"Tax optimized {signal.symbol}: "
                         f"holding period {optimized_signal.holding_period_target}d, "
                         f"tax efficiency {optimized_signal.tax_efficiency_score:.1f}")
        
        return optimized_signal
    
    def _calculate_optimal_holding_period(self, signal: OptimizedTradeSignal) -> int:
        """Calculate optimal holding period balancing tax efficiency and alpha decay"""
        
        # Base holding period on hype score and velocity
        if signal.velocity_level == "viral" and signal.hype_score >= 8.0:
            # High urgency - accept short-term tax treatment
            return max(1, min(7, signal.confidence * 10))  # 1-7 days max
        elif signal.velocity_level == "breaking" and signal.hype_score >= 5.0:
            # Medium urgency - extend if high confidence
            return max(7, min(30, signal.confidence * 50)) if signal.confidence > 0.7 else 7
        elif signal.confidence >= 0.8:
            # High confidence - extend to long-term if possible
            return 365 + 1  # Just over 1 year for long-term treatment
        else:
            # Standard signals - balance tax efficiency with alpha decay
            return max(self.min_holding_period, int(signal.confidence * 90))  # Up to 3 months
    
    def _adjust_for_tax_harvesting(self, signal: OptimizedTradeSignal) -> float:
        """Adjust position size to enable tax-loss harvesting"""
        
        base_quantity = signal.base_quantity
        
        # Check for loss harvesting opportunities
        if signal.symbol in self.open_positions:
            for position in self.open_positions[signal.symbol]:
                if position['unrealized_pnl'] < -1000:  # Significant loss
                    # Reduce new position size to avoid wash sale
                    base_quantity *= 0.7
                    self.logger.info(f"Reduced position size for {signal.symbol} to avoid wash sale")
        
        # Reserve capacity for tax-loss harvesting
        if len(self.loss_harvest_opportunities) > 5:
            base_quantity *= 0.8  # Reserve 20% capacity
        
        return base_quantity
    
    def _calculate_tax_efficiency_score(self, signal: OptimizedTradeSignal) -> float:
        """Calculate tax efficiency score (0-100, higher better)"""
        
        score = 50.0  # Base score
        
        # Reward longer holding periods
        if signal.holding_period_target >= 365:
            score += 30  # Long-term capital gains treatment
        elif signal.holding_period_target >= 90:
            score += 15  # Medium-term
        elif signal.holding_period_target >= 31:
            score += 10  # Avoid wash sales
        
        # Reward high-conviction trades (fewer trades = better tax treatment)
        if signal.conviction_level == "very_high":
            score += 15
        elif signal.conviction_level == "high":
            score += 10
        elif signal.conviction_level == "medium":
            score += 5
        
        # Penalize high-frequency characteristics
        if signal.velocity_level == "viral":
            score -= 20  # Likely short-term
        elif signal.velocity_level == "breaking":
            score -= 10
        
        return max(0, min(100, score))
    
    def _adjust_conviction_for_taxes(self, signal: OptimizedTradeSignal) -> str:
        """Adjust conviction level considering tax implications"""
        
        base_conviction = signal.conviction_level
        
        # If we're planning a long holding period, increase conviction requirement
        if signal.holding_period_target >= 365:
            conviction_map = {
                "low": "low",      # Don't upgrade low conviction for long-term
                "medium": "high",   # Upgrade medium to high for long-term
                "high": "very_high", # Upgrade high to very_high for long-term
                "very_high": "very_high"
            }
            return conviction_map.get(base_conviction, base_conviction)
        
        return base_conviction


class ExecutionOptimizedStrategy:
    """
    Execution-optimized strategy focusing on reducing slippage and improving fill quality
    """
    
    def __init__(self, max_slippage_bps: float = 25.0, target_quality_score: float = 75.0):
        self.max_slippage_bps = max_slippage_bps
        self.target_quality_score = target_quality_score
        self.market_conditions = {}  # Track current market conditions
        self.execution_history = []  # Track execution quality
        
        self.logger = logging.getLogger("execution_optimized_strategy")
    
    def optimize_trade_signal(self, signal: OptimizedTradeSignal) -> OptimizedTradeSignal:
        """Optimize trade signal for better execution"""
        
        optimized_signal = OptimizedTradeSignal(**signal.__dict__)
        
        # 1. Choose optimal order type
        optimized_signal.preferred_order_type = self._choose_order_type(signal)
        
        # 2. Set execution timing
        optimized_signal.max_execution_delay_ms = self._calculate_optimal_delay(signal)
        
        # 3. Adjust position sizing for market impact
        optimized_signal.base_quantity = self._adjust_for_market_impact(signal)
        
        # 4. Set slippage tolerance
        optimized_signal.slippage_tolerance_bps = self._calculate_slippage_tolerance(signal)
        
        # 5. Adjust execution urgency
        optimized_signal.execution_urgency = self._determine_execution_urgency(signal)
        
        self.logger.debug(f"Execution optimized {signal.symbol}: "
                         f"{optimized_signal.preferred_order_type} order, "
                         f"max delay {optimized_signal.max_execution_delay_ms}ms")
        
        return optimized_signal
    
    def _choose_order_type(self, signal: OptimizedTradeSignal) -> str:
        """Choose optimal order type based on urgency and market conditions"""
        
        # High urgency or volatile conditions -> market order
        if (signal.velocity_level in ["viral", "breaking"] or 
            signal.hype_score >= 7.0):
            return "market"
        
        # Normal conditions -> limit order for better execution
        return "limit"
    
    def _calculate_optimal_delay(self, signal: OptimizedTradeSignal) -> int:
        """Calculate optimal execution delay balancing speed and quality"""
        
        if signal.velocity_level == "viral":
            return 2000   # 2 seconds max for viral events
        elif signal.velocity_level == "breaking":
            return 5000   # 5 seconds for breaking news
        elif signal.velocity_level == "trending":
            return 10000  # 10 seconds for trending
        else:
            return 30000  # 30 seconds for normal signals
    
    def _adjust_for_market_impact(self, signal: OptimizedTradeSignal) -> float:
        """Adjust position size to minimize market impact"""
        
        base_quantity = signal.base_quantity
        
        # Estimate daily volume for the symbol (simplified)
        estimated_daily_volume = self._estimate_daily_volume(signal.symbol)
        
        if estimated_daily_volume > 0:
            # Position size should be <1% of daily volume to minimize impact
            trade_value = base_quantity * signal.price
            daily_value = estimated_daily_volume * signal.price
            
            if trade_value > daily_value * 0.01:  # >1% of daily volume
                reduction_factor = (daily_value * 0.01) / trade_value
                base_quantity *= reduction_factor
                self.logger.info(f"Reduced {signal.symbol} position by {(1-reduction_factor)*100:.1f}% to minimize market impact")
        
        return base_quantity
    
    def _estimate_daily_volume(self, symbol: str) -> float:
        """Estimate daily volume for a symbol"""
        # Simplified volume estimates
        volume_estimates = {
            "AAPL": 50_000_000, "TSLA": 30_000_000, "SPY": 80_000_000,
            "MSFT": 25_000_000, "GOOGL": 20_000_000, "AMZN": 15_000_000,
            "BTCUSD": 2_000_000, "ETHUSD": 800_000, "DOGEUSD": 50_000
        }
        return volume_estimates.get(symbol, 1_000_000)  # Default 1M
    
    def _calculate_slippage_tolerance(self, signal: OptimizedTradeSignal) -> float:
        """Calculate acceptable slippage tolerance"""
        
        base_tolerance = 25.0  # 25 basis points base
        
        # Higher tolerance for urgent trades
        if signal.velocity_level == "viral":
            return min(100.0, base_tolerance * 2.0)
        elif signal.velocity_level == "breaking":
            return min(75.0, base_tolerance * 1.5)
        
        # Lower tolerance for normal trades
        return base_tolerance
    
    def _determine_execution_urgency(self, signal: OptimizedTradeSignal) -> str:
        """Determine execution urgency level"""
        
        if signal.velocity_level == "viral" and signal.hype_score >= 8.0:
            return "critical"
        elif signal.velocity_level == "breaking" and signal.hype_score >= 5.0:
            return "high"
        elif signal.velocity_level == "trending":
            return "normal"
        else:
            return "low"


class FrequencyOptimizedStrategy:
    """
    Frequency-optimized strategy focusing on fewer, higher-conviction trades
    """
    
    def __init__(self, min_confidence: float = 0.7, max_trades_per_day: int = 10):
        self.min_confidence = min_confidence
        self.max_trades_per_day = max_trades_per_day
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        self.conviction_history = []
        
        self.logger = logging.getLogger("frequency_optimized_strategy")
    
    def should_take_trade(self, signal: OptimizedTradeSignal) -> bool:
        """Determine if we should take this trade based on frequency optimization"""
        
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
        
        # Check daily trade limit
        if self.daily_trade_count >= self.max_trades_per_day:
            self.logger.info(f"Daily trade limit reached ({self.max_trades_per_day}), skipping {signal.symbol}")
            return False
        
        # Check minimum confidence threshold
        if signal.confidence < self.min_confidence:
            self.logger.debug(f"Confidence too low for {signal.symbol}: {signal.confidence:.3f} < {self.min_confidence}")
            return False
        
        # Additional conviction filtering
        if not self._meets_conviction_requirements(signal):
            return False
        
        return True
    
    def optimize_trade_signal(self, signal: OptimizedTradeSignal) -> OptimizedTradeSignal:
        """Optimize trade signal for frequency reduction"""
        
        if not self.should_take_trade(signal):
            return None  # Don't take this trade
        
        optimized_signal = OptimizedTradeSignal(**signal.__dict__)
        
        # Increase position size for fewer trades (within limits)
        optimized_signal.base_quantity = self._adjust_position_for_frequency(signal)
        
        # Adjust conviction level based on filtering
        optimized_signal.conviction_level = self._upgrade_conviction(signal)
        
        # Track this trade
        self.daily_trade_count += 1
        self.conviction_history.append({
            'date': datetime.now(),
            'symbol': signal.symbol,
            'confidence': signal.confidence,
            'conviction': optimized_signal.conviction_level
        })
        
        self.logger.info(f"Frequency optimized trade {self.daily_trade_count}/{self.max_trades_per_day}: "
                        f"{signal.symbol} (confidence: {signal.confidence:.3f})")
        
        return optimized_signal
    
    def _meets_conviction_requirements(self, signal: OptimizedTradeSignal) -> bool:
        """Check if signal meets conviction requirements"""
        
        # Require higher standards for high-frequency signals
        if signal.velocity_level == "viral":
            return signal.hype_score >= 8.5 and signal.confidence >= 0.8
        elif signal.velocity_level == "breaking":
            return signal.hype_score >= 6.0 and signal.confidence >= 0.75
        elif signal.velocity_level == "trending":
            return signal.confidence >= 0.7
        else:
            return signal.confidence >= 0.75  # Higher bar for normal signals
    
    def _adjust_position_for_frequency(self, signal: OptimizedTradeSignal) -> float:
        """Adjust position size for fewer trades strategy"""
        
        base_quantity = signal.base_quantity
        
        # Increase position size for high-conviction trades (within risk limits)
        if signal.confidence >= 0.9:
            multiplier = 1.5  # 50% larger positions
        elif signal.confidence >= 0.8:
            multiplier = 1.3  # 30% larger positions
        elif signal.confidence >= 0.7:
            multiplier = 1.1  # 10% larger positions
        else:
            multiplier = 1.0
        
        # Don't exceed maximum position percentage
        max_quantity = (signal.max_position_pct * 100000) / signal.price  # Assume $100K account
        
        return min(base_quantity * multiplier, max_quantity)
    
    def _upgrade_conviction(self, signal: OptimizedTradeSignal) -> str:
        """Upgrade conviction level for filtered trades"""
        
        # Since we're filtering for higher quality, upgrade conviction
        conviction_map = {
            "low": "medium",
            "medium": "high", 
            "high": "very_high",
            "very_high": "very_high"
        }
        
        return conviction_map.get(signal.conviction_level, signal.conviction_level)


class HybridOptimizedStrategy:
    """
    Hybrid optimization strategy combining all approaches
    """
    
    def __init__(self, 
                 tax_weight: float = 0.5,
                 execution_weight: float = 0.3,
                 frequency_weight: float = 0.2):
        
        self.tax_weight = tax_weight
        self.execution_weight = execution_weight
        self.frequency_weight = frequency_weight
        
        # Initialize sub-strategies
        self.tax_strategy = TaxOptimizedStrategy(min_holding_period=31, target_ltcg_ratio=0.3)
        self.execution_strategy = ExecutionOptimizedStrategy(max_slippage_bps=20.0)
        self.frequency_strategy = FrequencyOptimizedStrategy(min_confidence=0.75, max_trades_per_day=8)
        
        self.logger = logging.getLogger("hybrid_optimized_strategy")
    
    def optimize_trade_signal(self, signal: OptimizedTradeSignal) -> Optional[OptimizedTradeSignal]:
        """Apply hybrid optimization combining all strategies"""
        
        # 1. First check if frequency strategy allows this trade
        frequency_optimized = self.frequency_strategy.optimize_trade_signal(signal)
        if frequency_optimized is None:
            return None  # Trade filtered out by frequency optimization
        
        # 2. Apply tax optimization
        tax_optimized = self.tax_strategy.optimize_trade_signal(frequency_optimized)
        
        # 3. Apply execution optimization
        execution_optimized = self.execution_strategy.optimize_trade_signal(tax_optimized)
        
        # 4. Calculate combined optimization score
        combined_score = self._calculate_combined_score(execution_optimized)
        
        self.logger.info(f"Hybrid optimized {signal.symbol}: "
                        f"score {combined_score:.1f}, "
                        f"holding {execution_optimized.holding_period_target}d, "
                        f"conviction {execution_optimized.conviction_level}")
        
        return execution_optimized
    
    def _calculate_combined_score(self, signal: OptimizedTradeSignal) -> float:
        """Calculate combined optimization score"""
        
        # Tax efficiency component
        tax_score = signal.tax_efficiency_score
        
        # Execution quality component (simplified)
        execution_score = 100 - signal.slippage_tolerance_bps  # Lower slippage = higher score
        execution_score = max(0, min(100, execution_score))
        
        # Frequency component (based on conviction)
        conviction_scores = {
            "low": 25, "medium": 50, "high": 75, "very_high": 100
        }
        frequency_score = conviction_scores.get(signal.conviction_level, 50)
        
        # Weighted combination
        combined_score = (
            tax_score * self.tax_weight +
            execution_score * self.execution_weight +
            frequency_score * self.frequency_weight
        )
        
        return combined_score


# Testing and demonstration functions
def generate_optimized_sample_signals(num_signals: int = 50) -> List[OptimizedTradeSignal]:
    """Generate sample trade signals for optimization testing"""
    
    import random
    
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "SPY", "BTCUSD", "ETHUSD"]
    velocity_levels = ["normal", "trending", "breaking", "viral"]
    actions = ["buy", "sell"]
    conviction_levels = ["low", "medium", "high", "very_high"]
    
    signals = []
    
    for i in range(num_signals):
        symbol = random.choice(symbols)
        is_crypto = symbol.endswith('USD')
        
        # Generate correlated parameters
        hype_score = random.uniform(1, 10)
        if hype_score >= 8:
            velocity_level = "viral"
            confidence = random.uniform(0.7, 0.95)
            conviction = random.choice(["high", "very_high"])
        elif hype_score >= 5:
            velocity_level = "breaking"
            confidence = random.uniform(0.6, 0.85)
            conviction = random.choice(["medium", "high"])
        elif hype_score >= 3:
            velocity_level = "trending"
            confidence = random.uniform(0.5, 0.75)
            conviction = random.choice(["medium", "high"])
        else:
            velocity_level = "normal"
            confidence = random.uniform(0.3, 0.65)
            conviction = random.choice(["low", "medium"])
        
        signal = OptimizedTradeSignal(
            symbol=symbol,
            action=random.choice(actions),
            base_quantity=random.uniform(0.1, 10) if is_crypto else random.randint(10, 1000),
            price=random.uniform(0.01, 10000) if is_crypto else random.uniform(10, 500),
            confidence=confidence,
            hype_score=hype_score,
            velocity_level=velocity_level,
            conviction_level=conviction
        )
        
        signals.append(signal)
    
    return signals


async def test_optimization_strategies():
    """Test all optimization strategies"""
    print("🎯 Testing Optimization Strategies")
    print("=" * 70)
    
    # Generate sample signals
    sample_signals = generate_optimized_sample_signals(100)
    
    # Initialize strategies
    strategies = {
        "Tax Optimized": TaxOptimizedStrategy(),
        "Execution Optimized": ExecutionOptimizedStrategy(), 
        "Frequency Optimized": FrequencyOptimizedStrategy(),
        "Hybrid Optimized": HybridOptimizedStrategy()
    }
    
    for strategy_name, strategy in strategies.items():
        print(f"\n🧪 Testing {strategy_name} Strategy:")
        print("-" * 50)
        
        optimized_signals = []
        filtered_count = 0
        
        for signal in sample_signals[:20]:  # Test first 20 signals
            if hasattr(strategy, 'should_take_trade') and not strategy.should_take_trade(signal):
                filtered_count += 1
                continue
                
            optimized = strategy.optimize_trade_signal(signal)
            if optimized is not None:
                optimized_signals.append(optimized)
        
        # Display results
        print(f"  Original signals: 20")
        print(f"  Filtered out: {filtered_count}")
        print(f"  Optimized signals: {len(optimized_signals)}")
        
        if optimized_signals:
            avg_holding_period = sum(s.holding_period_target for s in optimized_signals) / len(optimized_signals)
            avg_tax_efficiency = sum(s.tax_efficiency_score for s in optimized_signals) / len(optimized_signals)
            
            print(f"  Avg holding period: {avg_holding_period:.1f} days")
            print(f"  Avg tax efficiency: {avg_tax_efficiency:.1f}/100")
            
            # Show conviction distribution
            conviction_dist = {}
            for signal in optimized_signals:
                conviction_dist[signal.conviction_level] = conviction_dist.get(signal.conviction_level, 0) + 1
            
            print(f"  Conviction distribution: {conviction_dist}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(test_optimization_strategies())
    
    print("\n✅ Optimization strategies implemented and tested!")
    print("📁 Ready for integration into enhanced backtesting framework")
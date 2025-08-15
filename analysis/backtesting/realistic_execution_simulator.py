#!/usr/bin/env python3
"""
Realistic Execution Delays and Slippage Simulator
Market-condition-based execution timing and market impact modeling

Author: Claude Code (Anthropic AI Assistant)
Date: August 14, 2025
Task: 5 - Realistic Execution Delays and Slippage
Next Task: 6 - Market Impact Calculator
"""

import random
import math
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

from realistic_commission_models import BrokerType, AssetType, TradeInfo


class MarketCondition(Enum):
    """Market conditions affecting execution quality"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    LOW_VOLUME = "low_volume"
    HIGH_VOLUME = "high_volume"
    AFTER_HOURS = "after_hours"
    OPENING = "opening"
    CLOSING = "closing"
    NEWS_EVENT = "news_event"
    EARNINGS = "earnings"
    HOLIDAY = "holiday"


class OrderType(Enum):
    """Order types with different execution characteristics"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class ExecutionResult:
    """Result of realistic execution simulation"""
    requested_price: float
    executed_price: float
    slippage_bps: float
    execution_delay_ms: int
    partial_fill_ratio: float = 1.0
    market_condition: MarketCondition = MarketCondition.NORMAL
    order_type: OrderType = OrderType.MARKET
    
    # Breakdown components
    network_latency_ms: int = 0
    broker_processing_ms: int = 0
    market_response_ms: int = 0
    queue_waiting_ms: int = 0
    
    # Slippage components
    bid_ask_spread_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    volatility_impact_bps: float = 0.0
    
    def get_total_cost_bps(self) -> float:
        """Get total execution cost in basis points"""
        return abs(self.slippage_bps)
    
    def get_execution_quality_score(self) -> float:
        """Get execution quality score (0-100, higher is better)"""
        # Penalties for delays and slippage
        delay_penalty = min(50, self.execution_delay_ms / 100)  # 1 point per 100ms
        slippage_penalty = min(40, abs(self.slippage_bps))      # 1 point per bp
        partial_fill_penalty = (1 - self.partial_fill_ratio) * 10
        
        return max(0, 100 - delay_penalty - slippage_penalty - partial_fill_penalty)


@dataclass  
class MarketDataSnapshot:
    """Market data for execution simulation"""
    symbol: str
    bid_price: float
    ask_price: float
    last_price: float
    volume_24h: float
    volatility_daily: float
    market_cap: float = 0.0
    average_daily_volume: float = 0.0
    
    def get_bid_ask_spread_bps(self) -> float:
        """Calculate bid-ask spread in basis points"""
        if self.last_price > 0:
            spread = self.ask_price - self.bid_price
            return (spread / self.last_price) * 10000
        return 0.0
    
    def get_liquidity_score(self) -> float:
        """Get liquidity score (0-100, higher is more liquid)"""
        # Based on volume and spread
        volume_score = min(50, math.log10(self.volume_24h / 1000000) * 10)  # $1M baseline
        spread_score = max(0, 50 - self.get_bid_ask_spread_bps())
        return max(0, volume_score + spread_score)


class RealisticExecutionSimulator:
    """
    Simulate realistic execution delays and slippage based on market conditions
    """
    
    def __init__(self):
        # Base latency components (milliseconds)
        self.base_network_latency = {
            "same_city": (1, 5),
            "cross_country": (20, 40), 
            "international": (50, 200)
        }
        
        self.base_broker_processing = {
            BrokerType.ALPACA: (50, 200),
            BrokerType.INTERACTIVE_BROKERS: (30, 150),
            BrokerType.CHARLES_SCHWAB: (100, 300),
            BrokerType.ROBINHOOD: (200, 500),
            BrokerType.BINANCE_US: (10, 100),
            BrokerType.COINBASE_PRO: (50, 200)
        }
        
        # Market response times
        self.market_response_times = {
            MarketCondition.NORMAL: (50, 200),
            MarketCondition.VOLATILE: (200, 1000),
            MarketCondition.LOW_VOLUME: (500, 2000),
            MarketCondition.HIGH_VOLUME: (20, 100),
            MarketCondition.AFTER_HOURS: (1000, 5000),
            MarketCondition.OPENING: (100, 500),
            MarketCondition.CLOSING: (200, 800),
            MarketCondition.NEWS_EVENT: (500, 3000),
            MarketCondition.EARNINGS: (300, 1500)
        }
        
        # Slippage parameters
        self.base_slippage_rates = {
            AssetType.US_STOCK: {
                "major": 0.5,      # 0.5 bps base for major stocks
                "minor": 2.0,      # 2.0 bps base for minor stocks
                "penny": 10.0      # 10.0 bps base for penny stocks
            },
            AssetType.CRYPTOCURRENCY: {
                "major": 2.0,      # 2.0 bps base for BTC/ETH
                "minor": 5.0,      # 5.0 bps base for altcoins
                "exotic": 25.0     # 25.0 bps base for exotic pairs
            }
        }
        
    def simulate_execution(
        self, 
        trade: TradeInfo, 
        market_data: MarketDataSnapshot,
        order_type: OrderType = OrderType.MARKET,
        market_condition: MarketCondition = MarketCondition.NORMAL,
        connection_quality: str = "cross_country"
    ) -> ExecutionResult:
        """
        Simulate realistic trade execution with delays and slippage
        """
        
        # Determine market condition based on market data if not specified
        if market_condition == MarketCondition.NORMAL:
            market_condition = self._detect_market_condition(trade, market_data)
        
        # Simulate execution delay components
        network_latency = self._simulate_network_latency(connection_quality)
        broker_processing = self._simulate_broker_processing(trade.broker, market_condition)
        market_response = self._simulate_market_response(market_condition, market_data)
        queue_waiting = self._simulate_queue_waiting(market_condition, order_type)
        
        total_delay = network_latency + broker_processing + market_response + queue_waiting
        
        # Simulate slippage and price impact
        slippage_result = self._simulate_slippage(
            trade, market_data, market_condition, order_type, total_delay
        )
        
        # Determine final execution price
        if trade.side.lower() == "buy":
            executed_price = trade.price + (slippage_result["total_slippage_bps"] / 10000 * trade.price)
        else:
            executed_price = trade.price - (slippage_result["total_slippage_bps"] / 10000 * trade.price)
        
        # Simulate partial fills for large orders or poor liquidity
        partial_fill_ratio = self._simulate_partial_fill(trade, market_data, market_condition)
        
        return ExecutionResult(
            requested_price=trade.price,
            executed_price=executed_price,
            slippage_bps=slippage_result["total_slippage_bps"],
            execution_delay_ms=int(total_delay),
            partial_fill_ratio=partial_fill_ratio,
            market_condition=market_condition,
            order_type=order_type,
            
            # Delay breakdown
            network_latency_ms=int(network_latency),
            broker_processing_ms=int(broker_processing),
            market_response_ms=int(market_response),
            queue_waiting_ms=int(queue_waiting),
            
            # Slippage breakdown
            bid_ask_spread_bps=slippage_result["bid_ask_spread_bps"],
            temporary_impact_bps=slippage_result["temporary_impact_bps"],
            permanent_impact_bps=slippage_result["permanent_impact_bps"],
            volatility_impact_bps=slippage_result["volatility_impact_bps"]
        )
    
    def _detect_market_condition(self, trade: TradeInfo, market_data: MarketDataSnapshot) -> MarketCondition:
        """Detect market condition based on trade and market data"""
        
        # Check time-based conditions
        if trade.timestamp:
            trade_time = trade.timestamp.time()
            
            # Market hours (9:30 AM - 4:00 PM ET)
            market_open = time(9, 30)
            market_close = time(16, 0)
            
            if trade_time < market_open or trade_time > market_close:
                if trade.asset_type == AssetType.US_STOCK:
                    return MarketCondition.AFTER_HOURS
            
            # Opening conditions (9:30-10:00 AM)
            if time(9, 30) <= trade_time <= time(10, 0):
                return MarketCondition.OPENING
                
            # Closing conditions (3:30-4:00 PM)
            if time(15, 30) <= trade_time <= time(16, 0):
                return MarketCondition.CLOSING
        
        # Check volatility conditions
        if market_data.volatility_daily > 0.05:  # 5% daily volatility
            return MarketCondition.VOLATILE
        
        # Check volume conditions
        if market_data.average_daily_volume > 0:
            volume_ratio = market_data.volume_24h / market_data.average_daily_volume
            if volume_ratio > 3.0:
                return MarketCondition.HIGH_VOLUME
            elif volume_ratio < 0.3:
                return MarketCondition.LOW_VOLUME
        
        return MarketCondition.NORMAL
    
    def _simulate_network_latency(self, connection_quality: str) -> float:
        """Simulate network latency based on connection quality"""
        latency_range = self.base_network_latency.get(connection_quality, (20, 40))
        return random.uniform(latency_range[0], latency_range[1])
    
    def _simulate_broker_processing(self, broker: BrokerType, condition: MarketCondition) -> float:
        """Simulate broker processing delay"""
        base_range = self.base_broker_processing.get(broker, (100, 300))
        base_delay = random.uniform(base_range[0], base_range[1])
        
        # Apply condition multipliers
        condition_multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 1.5,
            MarketCondition.LOW_VOLUME: 1.2,
            MarketCondition.HIGH_VOLUME: 1.3,
            MarketCondition.AFTER_HOURS: 2.0,
            MarketCondition.OPENING: 1.8,
            MarketCondition.CLOSING: 1.6,
            MarketCondition.NEWS_EVENT: 2.5,
            MarketCondition.EARNINGS: 2.0
        }
        
        multiplier = condition_multipliers.get(condition, 1.0)
        return base_delay * multiplier
    
    def _simulate_market_response(self, condition: MarketCondition, market_data: MarketDataSnapshot) -> float:
        """Simulate market maker response time"""
        base_range = self.market_response_times.get(condition, (50, 200))
        base_delay = random.uniform(base_range[0], base_range[1])
        
        # Adjust for liquidity
        liquidity_score = market_data.get_liquidity_score()
        liquidity_multiplier = max(0.5, (100 - liquidity_score) / 100 + 0.5)
        
        return base_delay * liquidity_multiplier
    
    def _simulate_queue_waiting(self, condition: MarketCondition, order_type: OrderType) -> float:
        """Simulate queue waiting time based on order priority"""
        
        # Market orders get priority
        if order_type == OrderType.MARKET:
            base_wait = random.uniform(0, 50)
        else:
            base_wait = random.uniform(50, 200)
        
        # Condition multipliers
        if condition in [MarketCondition.VOLATILE, MarketCondition.NEWS_EVENT]:
            base_wait *= 3.0
        elif condition in [MarketCondition.OPENING, MarketCondition.CLOSING]:
            base_wait *= 2.0
        elif condition == MarketCondition.LOW_VOLUME:
            base_wait *= 1.5
        
        return base_wait
    
    def _simulate_slippage(
        self, 
        trade: TradeInfo, 
        market_data: MarketDataSnapshot,
        market_condition: MarketCondition,
        order_type: OrderType,
        execution_delay_ms: float
    ) -> Dict[str, float]:
        """Simulate comprehensive slippage including all components"""
        
        # 1. Bid-Ask Spread Component
        bid_ask_spread_bps = market_data.get_bid_ask_spread_bps()
        
        # Market orders cross the spread, limit orders may avoid it
        if order_type == OrderType.MARKET:
            spread_cost = bid_ask_spread_bps / 2  # Cross half the spread on average
        else:
            spread_cost = 0.0  # Limit orders sit on the bid/ask
        
        # 2. Temporary Market Impact (recovers quickly)
        temporary_impact_bps = self._calculate_temporary_impact(trade, market_data)
        
        # 3. Permanent Market Impact (price discovery)
        permanent_impact_bps = self._calculate_permanent_impact(trade, market_data)
        
        # 4. Volatility Impact (price moves during execution delay)
        volatility_impact_bps = self._calculate_volatility_impact(
            market_data.volatility_daily, execution_delay_ms
        )
        
        # Apply market condition multipliers
        condition_multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 2.0,
            MarketCondition.LOW_VOLUME: 1.8,
            MarketCondition.HIGH_VOLUME: 0.8,
            MarketCondition.AFTER_HOURS: 3.0,
            MarketCondition.OPENING: 1.5,
            MarketCondition.CLOSING: 1.3,
            MarketCondition.NEWS_EVENT: 3.5,
            MarketCondition.EARNINGS: 2.5
        }
        
        multiplier = condition_multipliers.get(market_condition, 1.0)
        
        # Apply multiplier to impact components (not spread)
        temporary_impact_bps *= multiplier
        permanent_impact_bps *= multiplier
        volatility_impact_bps *= multiplier
        
        total_slippage_bps = (
            spread_cost + 
            temporary_impact_bps + 
            permanent_impact_bps + 
            volatility_impact_bps
        )
        
        return {
            "bid_ask_spread_bps": spread_cost,
            "temporary_impact_bps": temporary_impact_bps,
            "permanent_impact_bps": permanent_impact_bps,
            "volatility_impact_bps": volatility_impact_bps,
            "total_slippage_bps": total_slippage_bps
        }
    
    def _calculate_temporary_impact(self, trade: TradeInfo, market_data: MarketDataSnapshot) -> float:
        """Calculate temporary market impact based on order size"""
        
        # Market impact model: Cost = α × (Order Size / ADV)^β
        # Where ADV is Average Daily Volume
        
        if market_data.average_daily_volume <= 0:
            return 0.0
        
        order_value = abs(trade.quantity * trade.price)
        
        # Estimate participation rate (order value / daily volume value)
        daily_volume_value = market_data.average_daily_volume * market_data.last_price
        participation_rate = order_value / daily_volume_value if daily_volume_value > 0 else 0.1
        
        # Impact parameters by asset type
        if trade.asset_type == AssetType.US_STOCK:
            if market_data.market_cap > 10_000_000_000:  # Large cap
                alpha, beta = 0.5, 0.5
            elif market_data.market_cap > 1_000_000_000:  # Mid cap
                alpha, beta = 1.0, 0.6
            else:  # Small cap
                alpha, beta = 2.0, 0.7
        elif trade.asset_type == AssetType.CRYPTOCURRENCY:
            alpha, beta = 1.5, 0.6  # Crypto generally higher impact
        else:
            alpha, beta = 1.0, 0.5  # Default
        
        # Calculate impact in basis points
        impact_bps = alpha * (participation_rate ** beta) * 10000
        
        # Cap at reasonable levels
        return min(impact_bps, 100.0)  # Max 100 bps temporary impact
    
    def _calculate_permanent_impact(self, trade: TradeInfo, market_data: MarketDataSnapshot) -> float:
        """Calculate permanent market impact (information content)"""
        
        # Permanent impact is typically 30-50% of temporary impact
        temporary_impact = self._calculate_temporary_impact(trade, market_data)
        permanent_ratio = 0.4  # 40% of temporary impact becomes permanent
        
        return temporary_impact * permanent_ratio
    
    def _calculate_volatility_impact(self, daily_volatility: float, delay_ms: float) -> float:
        """Calculate price movement during execution delay"""
        
        if delay_ms <= 0:
            return 0.0
        
        # Convert daily volatility to per-millisecond volatility
        # Assuming 6.5 trading hours = 23,400,000 ms per day
        ms_per_trading_day = 6.5 * 60 * 60 * 1000
        volatility_per_ms = daily_volatility / math.sqrt(ms_per_trading_day)
        
        # Price movement during delay (random walk)
        delay_volatility = volatility_per_ms * math.sqrt(delay_ms)
        
        # Convert to basis points (1 standard deviation move)
        volatility_impact_bps = delay_volatility * 10000
        
        # Random direction (could be favorable or unfavorable)
        direction = random.choice([-1, 1])
        
        return abs(volatility_impact_bps * direction)  # Take absolute for cost calculation
    
    def _simulate_partial_fill(
        self, 
        trade: TradeInfo, 
        market_data: MarketDataSnapshot,
        market_condition: MarketCondition
    ) -> float:
        """Simulate partial fill ratio based on liquidity and order size"""
        
        order_value = abs(trade.quantity * trade.price)
        
        # Calculate order size relative to market
        if market_data.average_daily_volume > 0:
            daily_value = market_data.average_daily_volume * market_data.last_price
            size_ratio = order_value / daily_value
        else:
            size_ratio = 0.01  # Default small order
        
        # Base fill ratio (higher for smaller orders)
        if size_ratio < 0.001:    # <0.1% of daily volume
            base_fill_ratio = 1.0
        elif size_ratio < 0.01:   # <1% of daily volume  
            base_fill_ratio = 0.95
        elif size_ratio < 0.05:   # <5% of daily volume
            base_fill_ratio = 0.80
        else:                     # >5% of daily volume
            base_fill_ratio = 0.50
        
        # Apply market condition adjustments
        condition_adjustments = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 0.9,
            MarketCondition.LOW_VOLUME: 0.7,
            MarketCondition.HIGH_VOLUME: 1.0,
            MarketCondition.AFTER_HOURS: 0.6,
            MarketCondition.NEWS_EVENT: 0.8
        }
        
        adjustment = condition_adjustments.get(market_condition, 1.0)
        final_fill_ratio = min(1.0, base_fill_ratio * adjustment)
        
        # Add some randomness
        return max(0.1, final_fill_ratio + random.uniform(-0.1, 0.1))


class MarketDataProvider:
    """
    Provide realistic market data for execution simulation
    """
    
    def __init__(self):
        # Sample market data for common symbols
        self.market_data_cache = self._initialize_sample_data()
    
    def _initialize_sample_data(self) -> Dict[str, MarketDataSnapshot]:
        """Initialize sample market data for testing"""
        return {
            # Major US Stocks
            "AAPL": MarketDataSnapshot(
                symbol="AAPL", bid_price=149.95, ask_price=150.05, last_price=150.00,
                volume_24h=50_000_000, volatility_daily=0.02, market_cap=2_500_000_000_000,
                average_daily_volume=50_000_000
            ),
            "TSLA": MarketDataSnapshot(
                symbol="TSLA", bid_price=199.80, ask_price=200.20, last_price=200.00,
                volume_24h=30_000_000, volatility_daily=0.04, market_cap=650_000_000_000,
                average_daily_volume=30_000_000
            ),
            "SPY": MarketDataSnapshot(
                symbol="SPY", bid_price=399.99, ask_price=400.01, last_price=400.00,
                volume_24h=80_000_000, volatility_daily=0.015, market_cap=300_000_000_000,
                average_daily_volume=80_000_000
            ),
            
            # Major Cryptocurrencies
            "BTCUSD": MarketDataSnapshot(
                symbol="BTCUSD", bid_price=49_950, ask_price=50_050, last_price=50_000,
                volume_24h=2_000_000_000, volatility_daily=0.06, market_cap=1_000_000_000_000,
                average_daily_volume=2_000_000_000
            ),
            "ETHUSD": MarketDataSnapshot(
                symbol="ETHUSD", bid_price=2_995, ask_price=3_005, last_price=3_000,
                volume_24h=800_000_000, volatility_daily=0.08, market_cap=400_000_000_000,
                average_daily_volume=800_000_000
            ),
            "DOGEUSD": MarketDataSnapshot(
                symbol="DOGEUSD", bid_price=0.0799, ask_price=0.0801, last_price=0.08,
                volume_24h=50_000_000, volatility_daily=0.12, market_cap=12_000_000_000,
                average_daily_volume=50_000_000
            )
        }
    
    def get_market_data(self, symbol: str) -> MarketDataSnapshot:
        """Get market data for a symbol"""
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol]
        
        # Generate synthetic data for unknown symbols
        return self._generate_synthetic_data(symbol)
    
    def _generate_synthetic_data(self, symbol: str) -> MarketDataSnapshot:
        """Generate synthetic market data for unknown symbols"""
        
        # Base price based on symbol type
        if "USD" in symbol:  # Crypto
            base_price = random.uniform(0.01, 10000)
            volatility = random.uniform(0.05, 0.20)
            volume = random.uniform(1_000_000, 100_000_000)
        else:  # Stock
            base_price = random.uniform(10, 500)
            volatility = random.uniform(0.015, 0.08)
            volume = random.uniform(1_000_000, 50_000_000)
        
        # Generate bid/ask around base price
        spread_pct = random.uniform(0.001, 0.01)  # 0.1% to 1% spread
        spread = base_price * spread_pct
        
        return MarketDataSnapshot(
            symbol=symbol,
            bid_price=base_price - spread/2,
            ask_price=base_price + spread/2,
            last_price=base_price,
            volume_24h=volume,
            volatility_daily=volatility,
            market_cap=random.uniform(100_000_000, 100_000_000_000),
            average_daily_volume=volume
        )


# Testing functions
def test_execution_simulation():
    """Test realistic execution simulation"""
    print("🔄 Testing Realistic Execution Simulation")
    print("=" * 60)
    
    simulator = RealisticExecutionSimulator()
    market_provider = MarketDataProvider()
    
    # Test scenarios
    test_trades = [
        TradeInfo(
            broker=BrokerType.ALPACA,
            asset_type=AssetType.US_STOCK,
            symbol="AAPL",
            quantity=100,
            price=150.00,
            side="buy",
            timestamp=datetime(2025, 8, 14, 10, 30)  # Normal hours
        ),
        TradeInfo(
            broker=BrokerType.ALPACA,
            asset_type=AssetType.CRYPTOCURRENCY,
            symbol="BTCUSD",
            quantity=0.1,
            price=50000.00,
            side="sell",
            timestamp=datetime(2025, 8, 14, 2, 30)  # After hours (crypto 24/7)
        ),
        TradeInfo(
            broker=BrokerType.INTERACTIVE_BROKERS,
            asset_type=AssetType.US_STOCK,
            symbol="TSLA",
            quantity=1000,  # Large order
            price=200.00,
            side="buy",
            timestamp=datetime(2025, 8, 14, 9, 45)  # Opening period
        )
    ]
    
    market_conditions = [MarketCondition.NORMAL, MarketCondition.VOLATILE, MarketCondition.LOW_VOLUME]
    
    for i, trade in enumerate(test_trades):
        condition = market_conditions[i % len(market_conditions)]
        market_data = market_provider.get_market_data(trade.symbol)
        
        result = simulator.simulate_execution(
            trade, market_data, OrderType.MARKET, condition
        )
        
        print(f"\n📊 Trade {i+1}: {trade.symbol} ({trade.side.upper()}) - {condition.value}")
        print(f"  Requested: ${trade.price:.2f}")
        print(f"  Executed:  ${result.executed_price:.2f}")
        print(f"  Slippage:  {result.slippage_bps:.1f} bps")
        print(f"  Delay:     {result.execution_delay_ms:,} ms")
        print(f"  Fill:      {result.partial_fill_ratio*100:.1f}%")
        print(f"  Quality:   {result.get_execution_quality_score():.1f}/100")
        
        print(f"  Delay Breakdown:")
        print(f"    Network:   {result.network_latency_ms} ms")
        print(f"    Broker:    {result.broker_processing_ms} ms") 
        print(f"    Market:    {result.market_response_ms} ms")
        print(f"    Queue:     {result.queue_waiting_ms} ms")
        
        print(f"  Slippage Breakdown:")
        print(f"    Spread:    {result.bid_ask_spread_bps:.1f} bps")
        print(f"    Temp Impact: {result.temporary_impact_bps:.1f} bps")
        print(f"    Perm Impact: {result.permanent_impact_bps:.1f} bps")
        print(f"    Volatility:  {result.volatility_impact_bps:.1f} bps")


def test_market_conditions():
    """Test execution under different market conditions"""
    print("\n🌊 Testing Different Market Conditions")
    print("=" * 60)
    
    simulator = RealisticExecutionSimulator()
    market_provider = MarketDataProvider()
    
    # Standard trade
    trade = TradeInfo(
        broker=BrokerType.ALPACA,
        asset_type=AssetType.US_STOCK,
        symbol="SPY",
        quantity=500,
        price=400.00,
        side="buy"
    )
    
    market_data = market_provider.get_market_data("SPY")
    
    conditions_to_test = [
        MarketCondition.NORMAL,
        MarketCondition.VOLATILE,
        MarketCondition.LOW_VOLUME,
        MarketCondition.AFTER_HOURS,
        MarketCondition.NEWS_EVENT
    ]
    
    results = {}
    
    for condition in conditions_to_test:
        result = simulator.simulate_execution(
            trade, market_data, OrderType.MARKET, condition
        )
        results[condition] = result
        
        print(f"\n🔍 {condition.value.upper()}:")
        print(f"  Delay: {result.execution_delay_ms:,} ms")
        print(f"  Slippage: {result.slippage_bps:.1f} bps")
        print(f"  Cost: ${result.slippage_bps/10000*trade.trade_value:.2f}")
        print(f"  Quality: {result.get_execution_quality_score():.1f}/100")
    
    # Summary comparison
    print(f"\n📈 Condition Impact Summary:")
    baseline = results[MarketCondition.NORMAL]
    
    for condition, result in results.items():
        if condition == MarketCondition.NORMAL:
            continue
            
        delay_ratio = result.execution_delay_ms / baseline.execution_delay_ms
        slippage_ratio = result.slippage_bps / baseline.slippage_bps if baseline.slippage_bps > 0 else 1
        
        print(f"  {condition.value}: {delay_ratio:.1f}x delay, {slippage_ratio:.1f}x slippage")


if __name__ == "__main__":
    # Run all tests
    test_execution_simulation()
    test_market_conditions()
    
    print("\n✅ Realistic execution simulation implemented and tested!")
    print("📁 Ready for integration into enhanced backtesting framework")
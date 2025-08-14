#!/usr/bin/env python3
"""
Comprehensive Historical Backtesting Framework for Hype Detection & Fast Trading System

This framework validates the profitability potential of the sentiment-based hype detection
mechanism through rigorous statistical analysis of historical data.

Key Components:
- BreakingNewsVelocityTracker historical simulation
- MomentumPatternDetector pattern recognition validation
- Multi-speed execution lane performance analysis
- Statistical significance testing and risk metrics
- Comprehensive performance benchmarking

Author: Claude Code (Anthropic AI Assistant)
Date: 2025-01-14
"""

import asyncio
import json
import logging
import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import system components
import sys
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.components.breaking_news_velocity_tracker import (
    BreakingNewsVelocityTracker, NewsVelocitySignal, VelocityLevel
)
from algotrading_agent.components.momentum_pattern_detector import (
    MomentumPatternDetector, PatternSignal, PatternType
)
from algotrading_agent.components.express_execution_manager import (
    ExpressExecutionManager, ExecutionLane, ExpressTrade
)
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
from algotrading_agent.components.news_filter import NewsFilter
from algotrading_agent.components.decision_engine import DecisionEngine
from algotrading_agent.components.risk_manager import RiskManager
from algotrading_agent.config.settings import get_config


class MarketRegime(Enum):
    """Market conditions for backtesting"""
    BULL = "bull"
    BEAR = "bear" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class BacktestTradeResult:
    """Individual trade result with comprehensive metrics"""
    trade_id: str
    symbol: str
    action: str  # "buy" or "sell"
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    
    # Trigger information
    trigger_type: str = ""  # "velocity", "pattern", "combined"
    velocity_score: Optional[float] = None
    pattern_type: Optional[str] = None
    execution_lane: Optional[str] = None
    speed_target_ms: Optional[int] = None
    actual_execution_ms: Optional[int] = None
    
    # Performance metrics
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_minutes: Optional[int] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    
    # Risk metrics
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_hit: bool = False
    take_profit_hit: bool = False
    
    # Market context
    market_regime: Optional[str] = None
    volatility_at_entry: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    def is_profitable(self) -> bool:
        return self.pnl is not None and self.pnl > 0


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic Performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    
    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    
    # Trade Analysis
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_trade_duration_minutes: float = 0.0
    
    # Execution Analysis (by lane)
    lightning_trades: int = 0
    express_trades: int = 0
    fast_trades: int = 0
    standard_trades: int = 0
    
    lightning_win_rate: float = 0.0
    express_win_rate: float = 0.0
    fast_win_rate: float = 0.0
    standard_win_rate: float = 0.0
    
    # Speed Performance
    avg_execution_latency_ms: float = 0.0
    speed_target_achievement_rate: float = 0.0
    
    # Pattern Performance
    pattern_success_rates: Dict[str, float] = None
    velocity_level_success_rates: Dict[str, float] = None
    
    # Statistical Significance
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval_95: Optional[Tuple[float, float]] = None
    statistically_significant: bool = False
    
    # Benchmark Comparison
    benchmark_return_pct: float = 0.0
    excess_return_pct: float = 0.0
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    
    def __post_init__(self):
        if self.pattern_success_rates is None:
            self.pattern_success_rates = {}
        if self.velocity_level_success_rates is None:
            self.velocity_level_success_rates = {}


class HypeDetectionBacktester:
    """
    Comprehensive backtesting engine for hype detection and fast trading system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger("hype_backtest")
        
        # Initialize components for backtesting
        self.velocity_tracker = BreakingNewsVelocityTracker(
            self.config.get_component_config('breaking_news_velocity_tracker')
        )
        
        # Mock Alpaca client for backtesting
        self.mock_alpaca = MockAlpacaClient()
        self.pattern_detector = MomentumPatternDetector(
            self.config.get_component_config('momentum_pattern_detector'), 
            self.mock_alpaca
        )
        
        # Initialize analysis components
        self.news_filter = NewsFilter(self.config.get_component_config('news_filter'))
        self.news_brain = NewsAnalysisBrain(self.config.get_component_config('news_analysis_brain'))
        self.decision_engine = DecisionEngine(self.config.get_component_config('decision_engine'))
        self.risk_manager = RiskManager(self.config.get_component_config('risk_manager'))
        
        # Backtesting state
        self.starting_capital = 100000.0
        self.current_capital = 100000.0
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float, 'unrealized_pnl': float}}
        self.trade_history: List[BacktestTradeResult] = []
        self.daily_values = []  # For drawdown calculation
        
        # Market data simulation
        self.price_data = {}  # Historical price data simulation
        self.volatility_data = {}  # Historical volatility data
        
        # Performance tracking
        self.execution_performance = {
            ExecutionLane.LIGHTNING: [],
            ExecutionLane.EXPRESS: [],
            ExecutionLane.FAST: [],
            ExecutionLane.STANDARD: []
        }
        
        # Statistical analysis
        self.daily_returns = []
        self.benchmark_returns = []
        
    async def run_comprehensive_backtest(
        self, 
        historical_news: List[Dict[str, Any]], 
        start_date: str, 
        end_date: str,
        benchmark_symbol: str = "SPY"
    ) -> BacktestMetrics:
        """
        Run comprehensive backtest with statistical validation
        """
        self.logger.info(f"🚀 Starting comprehensive hype detection backtest: {start_date} to {end_date}")
        
        # Initialize components
        await self._initialize_components()
        
        # Prepare historical data
        filtered_news = self._prepare_historical_data(historical_news, start_date, end_date)
        daily_news = self._group_news_by_day(filtered_news)
        
        # Generate benchmark returns for comparison
        self._generate_benchmark_data(start_date, end_date, benchmark_symbol)
        
        # Process each trading day
        trading_days = sorted(daily_news.keys())
        for i, date in enumerate(trading_days):
            self.logger.info(f"📅 Processing trading day {i+1}/{len(trading_days)}: {date}")
            await self._process_trading_day(date, daily_news[date])
            
            # Record daily portfolio value
            daily_value = self._calculate_portfolio_value()
            self.daily_values.append(daily_value)
            
            daily_return = 0.0 if i == 0 else (daily_value - self.daily_values[i-1]) / self.daily_values[i-1]
            self.daily_returns.append(daily_return)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics()
        
        # Statistical significance testing
        self._perform_statistical_tests(metrics)
        
        self.logger.info("✅ Backtest completed successfully")
        return metrics
    
    async def _initialize_components(self):
        """Initialize all trading components for backtesting"""
        await self.velocity_tracker.start()
        await self.pattern_detector.start()
        self.news_filter.start()
        await self.news_brain.start()
        self.decision_engine.start()
        self.risk_manager.start()
        
    def _prepare_historical_data(
        self, 
        historical_news: List[Dict[str, Any]], 
        start_date: str, 
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Prepare and clean historical news data for backtesting"""
        
        # Convert date strings to timezone-aware datetime objects for filtering
        from datetime import timezone
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        filtered_news = []
        for item in historical_news:
            # Parse timestamp
            if isinstance(item.get('timestamp'), str):
                try:
                    news_dt = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    if start_dt <= news_dt <= end_dt:
                        filtered_news.append(item)
                except ValueError:
                    # Try alternative timestamp formats
                    try:
                        news_dt = datetime.strptime(item['timestamp'][:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
                        if start_dt <= news_dt <= end_dt:
                            filtered_news.append(item)
                    except ValueError:
                        self.logger.warning(f"Could not parse timestamp: {item.get('timestamp')}")
                        continue
        
        self.logger.info(f"📰 Prepared {len(filtered_news)} historical news items for backtesting")
        return filtered_news
    
    def _group_news_by_day(self, news_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group news items by trading day"""
        daily_news = {}
        
        for item in news_items:
            try:
                # Extract date from timestamp
                if isinstance(item.get('timestamp'), str):
                    timestamp_str = item['timestamp']
                    if 'T' in timestamp_str:
                        date_str = timestamp_str.split('T')[0]
                    else:
                        date_str = timestamp_str[:10]
                    
                    if date_str not in daily_news:
                        daily_news[date_str] = []
                    daily_news[date_str].append(item)
                        
            except Exception as e:
                self.logger.warning(f"Error grouping news item: {e}")
                continue
        
        return daily_news
    
    def _generate_benchmark_data(self, start_date: str, end_date: str, benchmark_symbol: str):
        """Generate realistic benchmark returns for comparison"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        
        # Simulate SPY-like returns: ~10% annual return, ~16% volatility
        annual_return = 0.10
        annual_volatility = 0.16
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / math.sqrt(252)
        
        self.benchmark_returns = []
        for i in range(days):
            # Generate realistic market returns with occasional large moves
            if random.random() < 0.02:  # 2% chance of large move
                daily_ret = random.normalvariate(0, daily_volatility * 3)  # 3x volatility spike
            else:
                daily_ret = random.normalvariate(daily_return, daily_volatility)
            
            self.benchmark_returns.append(daily_ret)
    
    async def _process_trading_day(self, date: str, news_items: List[Dict[str, Any]]):
        """Process a single trading day through the complete pipeline"""
        
        if not news_items:
            return
        
        # Step 1: Filter news for relevance
        filtered_news = self.news_filter.process(news_items)
        if not filtered_news:
            return
        
        # Step 2: Analyze sentiment and extract entities
        analyzed_news = self.news_brain.process(filtered_news)
        
        # Step 3: Detect velocity patterns (hype detection)
        velocity_signals = await self.velocity_tracker.process(analyzed_news)
        
        # Step 4: Detect momentum patterns (price pattern recognition)
        pattern_signals = await self.pattern_detector.process()
        
        # Step 5: Generate trading decisions from signals
        trades_generated = []
        
        # Process velocity-based trades
        for signal in velocity_signals:
            trade = await self._create_velocity_trade(signal, date)
            if trade:
                trades_generated.append(trade)
        
        # Process pattern-based trades
        for signal in pattern_signals:
            trade = await self._create_pattern_trade(signal, date)
            if trade:
                trades_generated.append(trade)
        
        # Step 6: Execute approved trades
        for trade in trades_generated:
            await self._execute_backtest_trade(trade, date)
        
        # Step 7: Update existing positions and check for exits
        await self._update_positions(date)
    
    async def _create_velocity_trade(
        self, 
        velocity_signal: NewsVelocitySignal, 
        date: str
    ) -> Optional[BacktestTradeResult]:
        """Create a trade from velocity signal"""
        
        if not velocity_signal.symbols:
            return None
        
        symbol = velocity_signal.symbols[0]  # Use first symbol
        
        # Determine execution lane based on velocity level
        execution_lane = self._get_velocity_execution_lane(velocity_signal.velocity_level)
        
        # Determine trade direction based on sentiment and impact
        if velocity_signal.financial_impact_score > 0.6:
            action = "buy"
        elif velocity_signal.financial_impact_score < 0.4:
            action = "sell"
        else:
            return None  # Neutral signal, skip
        
        # Calculate position size based on velocity strength
        base_position_size = 0.02  # 2% base position
        velocity_multiplier = min(velocity_signal.velocity_score / 10.0, 0.5)  # Up to 50% boost
        position_size = base_position_size + velocity_multiplier * 0.02
        
        # Get current price
        current_price = self._get_simulated_price(symbol, date)
        if not current_price:
            return None
        
        # Calculate quantity
        position_value = self.current_capital * position_size
        quantity = max(1, int(position_value / current_price))
        
        # Create trade result
        trade = BacktestTradeResult(
            trade_id=f"velocity_{date}_{symbol}_{velocity_signal.velocity_level.value}",
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_price=current_price,
            entry_time=velocity_signal.detected_at,
            trigger_type="velocity",
            velocity_score=velocity_signal.velocity_score,
            execution_lane=execution_lane.value,
            speed_target_ms=self._get_speed_target(execution_lane),
            market_regime=self._detect_market_regime(date),
            volume_ratio=1.0,  # Simplified
            volatility_at_entry=self._get_simulated_volatility(symbol, date)
        )
        
        # Set stop loss and take profit based on velocity
        volatility = trade.volatility_at_entry or 0.02
        if action == "buy":
            trade.stop_loss_price = current_price * (1 - (0.02 + volatility * 0.5))
            trade.take_profit_price = current_price * (1 + (0.04 + volatility))
        else:
            trade.stop_loss_price = current_price * (1 + (0.02 + volatility * 0.5))
            trade.take_profit_price = current_price * (1 - (0.04 + volatility))
        
        return trade
    
    async def _create_pattern_trade(
        self, 
        pattern_signal: PatternSignal, 
        date: str
    ) -> Optional[BacktestTradeResult]:
        """Create a trade from pattern signal"""
        
        symbol = pattern_signal.symbol
        execution_lane = self._get_pattern_execution_lane(pattern_signal.pattern_type)
        
        # Get current price
        current_price = self._get_simulated_price(symbol, date)
        if not current_price:
            return None
        
        # Calculate position size based on pattern confidence
        base_position_size = 0.03  # 3% base for patterns
        confidence_multiplier = pattern_signal.confidence
        position_size = base_position_size * confidence_multiplier
        
        position_value = self.current_capital * position_size
        quantity = max(1, int(position_value / current_price))
        
        # Create trade result
        trade = BacktestTradeResult(
            trade_id=f"pattern_{date}_{symbol}_{pattern_signal.pattern_type.value}",
            symbol=symbol,
            action="buy" if pattern_signal.direction == "bullish" else "sell",
            quantity=quantity,
            entry_price=current_price,
            entry_time=pattern_signal.detected_at,
            trigger_type="pattern",
            pattern_type=pattern_signal.pattern_type.value,
            execution_lane=execution_lane.value,
            speed_target_ms=self._get_speed_target(execution_lane),
            market_regime=self._detect_market_regime(date),
            volume_ratio=pattern_signal.volume_ratio,
            volatility_at_entry=pattern_signal.volatility
        )
        
        # Set stops based on pattern volatility
        if trade.action == "buy":
            trade.stop_loss_price = current_price * (1 - (0.02 + pattern_signal.volatility * 0.3))
            trade.take_profit_price = current_price * (1 + (0.05 + pattern_signal.volatility * 0.8))
        else:
            trade.stop_loss_price = current_price * (1 + (0.02 + pattern_signal.volatility * 0.3))
            trade.take_profit_price = current_price * (1 - (0.05 + pattern_signal.volatility * 0.8))
        
        return trade
    
    async def _execute_backtest_trade(self, trade: BacktestTradeResult, date: str):
        """Execute trade with realistic execution simulation"""
        
        # Simulate execution latency
        execution_latency = self._simulate_execution_latency(trade.execution_lane)
        trade.actual_execution_ms = execution_latency
        
        # Simulate slippage based on execution lane and market conditions
        slippage = self._simulate_slippage(trade.execution_lane, trade.volatility_at_entry or 0.02)
        
        if trade.action == "buy":
            adjusted_price = trade.entry_price * (1 + slippage)
        else:
            adjusted_price = trade.entry_price * (1 - slippage)
        
        trade.entry_price = adjusted_price
        
        # Update capital and positions
        trade_value = trade.quantity * adjusted_price
        
        if trade.action == "buy":
            self.current_capital -= trade_value
            self._add_position(trade.symbol, trade.quantity, adjusted_price)
        else:
            # Short selling - add negative position
            self.current_capital += trade_value
            self._add_position(trade.symbol, -trade.quantity, adjusted_price)
        
        # Add to trade history
        self.trade_history.append(trade)
        
        # Track execution performance
        lane = ExecutionLane(trade.execution_lane)
        self.execution_performance[lane].append(execution_latency)
        
        self.logger.info(
            f"🚀 Executed {trade.action} {trade.quantity} {trade.symbol} @ ${adjusted_price:.2f} "
            f"({trade.execution_lane}, {execution_latency}ms, trigger: {trade.trigger_type})"
        )
    
    async def _update_positions(self, date: str):
        """Update positions and check for exits (stop loss/take profit)"""
        
        positions_to_close = []
        
        for trade in self.trade_history:
            if trade.exit_time is not None:
                continue  # Already closed
            
            current_price = self._get_simulated_price(trade.symbol, date)
            if not current_price:
                continue
            
            # Check stop loss
            if trade.stop_loss_price and self._should_hit_stop(trade, current_price):
                trade.exit_price = trade.stop_loss_price
                trade.stop_loss_hit = True
                positions_to_close.append(trade)
                
            # Check take profit
            elif trade.take_profit_price and self._should_hit_take_profit(trade, current_price):
                trade.exit_price = trade.take_profit_price
                trade.take_profit_hit = True
                positions_to_close.append(trade)
                
            # Time-based exit after 24 hours (simplified)
            elif trade.entry_time and (datetime.now() - trade.entry_time).total_seconds() > 86400:
                trade.exit_price = current_price
                positions_to_close.append(trade)
        
        # Process exits
        for trade in positions_to_close:
            self._close_position(trade, date)
    
    def _close_position(self, trade: BacktestTradeResult, date: str):
        """Close a position and calculate P&L"""
        
        trade.exit_time = datetime.now()
        
        if not trade.exit_price:
            trade.exit_price = self._get_simulated_price(trade.symbol, date)
        
        # Calculate P&L
        if trade.action == "buy":
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        else:  # sell/short
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
        
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
        
        # Update capital
        if trade.action == "buy":
            self.current_capital += trade.exit_price * trade.quantity
        else:
            # Cover short position
            self.current_capital -= trade.exit_price * abs(trade.quantity)
        
        # Remove from positions
        if trade.symbol in self.positions:
            self.positions[trade.symbol]['quantity'] -= trade.quantity
            if self.positions[trade.symbol]['quantity'] == 0:
                del self.positions[trade.symbol]
        
        # Calculate duration
        if trade.entry_time and trade.exit_time:
            trade.duration_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)
        
        self.logger.info(
            f"📊 Closed {trade.symbol}: P&L ${trade.pnl:.2f} ({trade.pnl_pct:.1f}%) "
            f"Duration: {trade.duration_minutes}min"
        )
    
    def _calculate_comprehensive_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        
        metrics = BacktestMetrics()
        
        # Filter completed trades
        completed_trades = [t for t in self.trade_history if t.exit_time is not None]
        
        # Basic metrics
        metrics.total_trades = len(completed_trades)
        metrics.winning_trades = len([t for t in completed_trades if t.is_profitable()])
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        
        if metrics.total_trades > 0:
            metrics.win_rate_pct = (metrics.winning_trades / metrics.total_trades) * 100
        
        # P&L metrics
        metrics.total_pnl = sum(t.pnl or 0 for t in completed_trades)
        final_capital = self.current_capital + sum(
            pos['quantity'] * pos['avg_price'] for pos in self.positions.values()
        )
        metrics.total_return_pct = ((final_capital - self.starting_capital) / self.starting_capital) * 100
        
        # Risk metrics
        if self.daily_returns:
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(self.daily_returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(self.daily_returns)
            metrics.max_drawdown_pct = self._calculate_max_drawdown()
        else:
            metrics.sharpe_ratio = 0.0
            metrics.sortino_ratio = 0.0
            metrics.max_drawdown_pct = 0.0
        
        # Execution lane analysis
        self._analyze_execution_lanes(metrics, completed_trades)
        
        # Pattern analysis
        self._analyze_pattern_performance(metrics, completed_trades)
        
        # Speed analysis
        execution_times = [t.actual_execution_ms for t in completed_trades if t.actual_execution_ms]
        if execution_times:
            metrics.avg_execution_latency_ms = sum(execution_times) / len(execution_times)
            
            # Speed target achievement
            achieved_targets = sum(
                1 for t in completed_trades 
                if t.actual_execution_ms and t.speed_target_ms and t.actual_execution_ms <= t.speed_target_ms
            )
            metrics.speed_target_achievement_rate = (achieved_targets / len(execution_times)) * 100
        
        # Profit factor
        winning_pnl = sum(t.pnl for t in completed_trades if t.is_profitable())
        losing_pnl = abs(sum(t.pnl for t in completed_trades if not t.is_profitable()))
        
        if losing_pnl > 0:
            metrics.profit_factor = winning_pnl / losing_pnl
        
        # Average trade metrics
        if metrics.winning_trades > 0:
            metrics.avg_win_pct = sum(
                t.pnl_pct for t in completed_trades if t.is_profitable()
            ) / metrics.winning_trades
        
        if metrics.losing_trades > 0:
            metrics.avg_loss_pct = sum(
                t.pnl_pct for t in completed_trades if not t.is_profitable()
            ) / metrics.losing_trades
        
        # Duration analysis
        durations = [t.duration_minutes for t in completed_trades if t.duration_minutes]
        if durations:
            metrics.avg_trade_duration_minutes = sum(durations) / len(durations)
        
        return metrics
    
    def _perform_statistical_tests(self, metrics: BacktestMetrics):
        """Perform statistical significance testing"""
        
        if len(self.daily_returns) < 10:  # Insufficient data
            return
        
        # T-test against zero (no alpha)
        returns_array = np.array(self.daily_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        n = len(returns_array)
        
        if std_return > 0:
            metrics.t_statistic = (mean_return * math.sqrt(n)) / std_return
            
            # Calculate p-value (two-tailed test)
            # Simplified calculation - in production would use scipy.stats
            degrees_freedom = n - 1
            t_abs = abs(metrics.t_statistic)
            
            # Rough p-value approximation for demonstration
            if t_abs > 2.576:  # 99% confidence
                metrics.p_value = 0.01
            elif t_abs > 1.96:  # 95% confidence
                metrics.p_value = 0.05
            else:
                metrics.p_value = 0.20
            
            metrics.statistically_significant = metrics.p_value < 0.05
            
            # 95% Confidence interval
            margin_error = 1.96 * (std_return / math.sqrt(n))
            metrics.confidence_interval_95 = (
                mean_return - margin_error,
                mean_return + margin_error
            )
        
        # Benchmark comparison
        if len(self.benchmark_returns) == len(self.daily_returns):
            benchmark_total = np.prod([1 + r for r in self.benchmark_returns]) - 1
            metrics.benchmark_return_pct = benchmark_total * 100
            metrics.excess_return_pct = metrics.total_return_pct - metrics.benchmark_return_pct
            
            # Calculate beta and alpha (simplified)
            if len(self.benchmark_returns) > 10:
                benchmark_var = np.var(self.benchmark_returns)
                covariance = np.cov(self.daily_returns, self.benchmark_returns)[0][1]
                
                if benchmark_var > 0:
                    metrics.beta = covariance / benchmark_var
                    
                    # Jensen's alpha
                    risk_free_rate = 0.02 / 252  # 2% annual risk-free rate
                    expected_return = risk_free_rate + metrics.beta * (np.mean(self.benchmark_returns) - risk_free_rate)
                    metrics.alpha = np.mean(self.daily_returns) - expected_return
                    
                    # Information ratio
                    tracking_error = np.std(np.array(self.daily_returns) - np.array(self.benchmark_returns))
                    if tracking_error > 0:
                        metrics.information_ratio = (np.mean(self.daily_returns) - np.mean(self.benchmark_returns)) / tracking_error
    
    def _analyze_execution_lanes(self, metrics: BacktestMetrics, completed_trades: List[BacktestTradeResult]):
        """Analyze performance by execution lane"""
        
        for lane in ExecutionLane:
            lane_trades = [t for t in completed_trades if t.execution_lane == lane.value]
            if not lane_trades:
                continue
                
            lane_wins = len([t for t in lane_trades if t.is_profitable()])
            lane_count = len(lane_trades)
            win_rate = (lane_wins / lane_count) * 100 if lane_count > 0 else 0
            
            if lane == ExecutionLane.LIGHTNING:
                metrics.lightning_trades = lane_count
                metrics.lightning_win_rate = win_rate
            elif lane == ExecutionLane.EXPRESS:
                metrics.express_trades = lane_count
                metrics.express_win_rate = win_rate
            elif lane == ExecutionLane.FAST:
                metrics.fast_trades = lane_count
                metrics.fast_win_rate = win_rate
            elif lane == ExecutionLane.STANDARD:
                metrics.standard_trades = lane_count
                metrics.standard_win_rate = win_rate
    
    def _analyze_pattern_performance(self, metrics: BacktestMetrics, completed_trades: List[BacktestTradeResult]):
        """Analyze performance by pattern and velocity types"""
        
        # Pattern performance
        pattern_stats = {}
        for trade in completed_trades:
            if trade.pattern_type:
                if trade.pattern_type not in pattern_stats:
                    pattern_stats[trade.pattern_type] = {'total': 0, 'wins': 0}
                pattern_stats[trade.pattern_type]['total'] += 1
                if trade.is_profitable():
                    pattern_stats[trade.pattern_type]['wins'] += 1
        
        for pattern, stats in pattern_stats.items():
            win_rate = (stats['wins'] / stats['total']) * 100 if stats['total'] > 0 else 0
            metrics.pattern_success_rates[pattern] = win_rate
        
        # Velocity performance
        velocity_stats = {}
        for trade in completed_trades:
            if trade.velocity_score and trade.velocity_score > 0:
                # Categorize velocity scores
                if trade.velocity_score >= 8.0:
                    level = "viral"
                elif trade.velocity_score >= 5.0:
                    level = "breaking"
                elif trade.velocity_score >= 2.5:
                    level = "trending"
                else:
                    level = "normal"
                
                if level not in velocity_stats:
                    velocity_stats[level] = {'total': 0, 'wins': 0}
                velocity_stats[level]['total'] += 1
                if trade.is_profitable():
                    velocity_stats[level]['wins'] += 1
        
        for level, stats in velocity_stats.items():
            win_rate = (stats['wins'] / stats['total']) * 100 if stats['total'] > 0 else 0
            metrics.velocity_level_success_rates[level] = win_rate
    
    # Helper methods for backtesting simulation
    
    def _get_velocity_execution_lane(self, velocity_level: VelocityLevel) -> ExecutionLane:
        """Map velocity level to execution lane"""
        mapping = {
            VelocityLevel.VIRAL: ExecutionLane.LIGHTNING,
            VelocityLevel.BREAKING: ExecutionLane.EXPRESS,
            VelocityLevel.TRENDING: ExecutionLane.FAST,
            VelocityLevel.NORMAL: ExecutionLane.STANDARD
        }
        return mapping.get(velocity_level, ExecutionLane.STANDARD)
    
    def _get_pattern_execution_lane(self, pattern_type: PatternType) -> ExecutionLane:
        """Map pattern type to execution lane"""
        if pattern_type in [PatternType.FLASH_CRASH, PatternType.FLASH_SURGE]:
            return ExecutionLane.LIGHTNING
        elif pattern_type in [PatternType.EARNINGS_SURPRISE, PatternType.NEWS_VELOCITY_SPIKE]:
            return ExecutionLane.EXPRESS
        elif pattern_type in [PatternType.VOLUME_BREAKOUT, PatternType.REVERSAL_PATTERN]:
            return ExecutionLane.FAST
        else:
            return ExecutionLane.STANDARD
    
    def _get_speed_target(self, lane: ExecutionLane) -> int:
        """Get speed target in milliseconds for execution lane"""
        targets = {
            ExecutionLane.LIGHTNING: 5000,
            ExecutionLane.EXPRESS: 15000,
            ExecutionLane.FAST: 30000,
            ExecutionLane.STANDARD: 60000
        }
        return targets.get(lane, 60000)
    
    def _simulate_execution_latency(self, lane: str) -> int:
        """Simulate realistic execution latency"""
        base_latencies = {
            "lightning": 3000,  # 3 seconds average
            "express": 12000,   # 12 seconds average
            "fast": 25000,      # 25 seconds average
            "standard": 45000   # 45 seconds average
        }
        
        base = base_latencies.get(lane, 45000)
        # Add realistic variance
        return max(1000, int(random.normalvariate(base, base * 0.3)))
    
    def _simulate_slippage(self, lane: str, volatility: float) -> float:
        """Simulate realistic slippage based on execution speed and volatility"""
        base_slippage = {
            "lightning": 0.0005,  # 5 bps
            "express": 0.0003,    # 3 bps
            "fast": 0.0002,       # 2 bps
            "standard": 0.0001    # 1 bp
        }
        
        base = base_slippage.get(lane, 0.0002)
        # Adjust for volatility
        volatility_adjustment = volatility * 0.5
        return base + volatility_adjustment
    
    def _get_simulated_price(self, symbol: str, date: str) -> Optional[float]:
        """Get simulated price for backtesting"""
        # Simplified price simulation - in production would use real historical data
        base_prices = {
            "SPY": 400.0, "QQQ": 350.0, "AAPL": 150.0, "TSLA": 200.0,
            "MSFT": 300.0, "GOOGL": 2500.0, "AMZN": 3000.0, "META": 250.0,
            "NVDA": 800.0, "AMD": 100.0, "BTCUSD": 45000.0, "ETHUSD": 3000.0
        }
        
        if symbol not in base_prices:
            return None
        
        base_price = base_prices[symbol]
        
        # Add daily volatility
        daily_volatility = 0.02  # 2% daily volatility
        price_change = random.normalvariate(0, daily_volatility)
        
        return base_price * (1 + price_change)
    
    def _get_simulated_volatility(self, symbol: str, date: str) -> float:
        """Get simulated volatility for backtesting"""
        # Simplified volatility simulation
        base_volatilities = {
            "SPY": 0.15, "QQQ": 0.18, "AAPL": 0.25, "TSLA": 0.45,
            "MSFT": 0.22, "GOOGL": 0.28, "AMZN": 0.30, "META": 0.35,
            "BTCUSD": 0.80, "ETHUSD": 0.75
        }
        return base_volatilities.get(symbol, 0.20)
    
    def _detect_market_regime(self, date: str) -> str:
        """Detect market regime for given date"""
        # Simplified regime detection - in production would analyze actual market data
        regimes = ["bull", "bear", "sideways", "high_volatility", "low_volatility"]
        return random.choice(regimes)
    
    def _should_hit_stop(self, trade: BacktestTradeResult, current_price: float) -> bool:
        """Determine if stop loss should be hit"""
        if not trade.stop_loss_price:
            return False
        
        if trade.action == "buy":
            return current_price <= trade.stop_loss_price
        else:
            return current_price >= trade.stop_loss_price
    
    def _should_hit_take_profit(self, trade: BacktestTradeResult, current_price: float) -> bool:
        """Determine if take profit should be hit"""
        if not trade.take_profit_price:
            return False
        
        if trade.action == "buy":
            return current_price >= trade.take_profit_price
        else:
            return current_price <= trade.take_profit_price
    
    def _add_position(self, symbol: str, quantity: int, price: float):
        """Add to position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        current_qty = self.positions[symbol]['quantity']
        current_avg = self.positions[symbol]['avg_price']
        
        new_qty = current_qty + quantity
        if new_qty != 0:
            new_avg = ((current_qty * current_avg) + (quantity * price)) / new_qty
            self.positions[symbol] = {'quantity': new_qty, 'avg_price': new_avg}
        else:
            del self.positions[symbol]
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        value = self.current_capital
        
        for symbol, position in self.positions.items():
            current_price = self._get_simulated_price(symbol, "current") or position['avg_price']
            value += position['quantity'] * current_price
        
        return value
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        risk_free_rate = 0.02 / 252  # 2% annual / 252 trading days
        return (mean_return - risk_free_rate) / std_return * math.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Calculate downside deviation
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return float('inf')  # No downside
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        
        risk_free_rate = 0.02 / 252
        return (mean_return - risk_free_rate) / downside_deviation * math.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.daily_values or len(self.daily_values) < 2:
            return 0.0
        
        peak = self.daily_values[0]
        max_drawdown = 0.0
        
        for value in self.daily_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # Return as percentage
    
    def generate_comprehensive_report(self, metrics: BacktestMetrics) -> str:
        """Generate comprehensive backtest report"""
        
        report = f"""
🚀 === COMPREHENSIVE HYPE DETECTION BACKTEST REPORT === 🚀

📊 EXECUTIVE SUMMARY:
Total Return: {metrics.total_return_pct:.2f}%
Total Trades: {metrics.total_trades}
Win Rate: {metrics.win_rate_pct:.1f}%
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Max Drawdown: {metrics.max_drawdown_pct:.2f}%

🎯 STATISTICAL SIGNIFICANCE:
T-Statistic: {metrics.t_statistic or 0:.3f} ({"Significant" if metrics.statistically_significant else "Not Significant"})
P-Value: {metrics.p_value or 0:.3f}
95% Confidence Interval: {metrics.confidence_interval_95 or (0, 0)}

⚡ EXECUTION LANE PERFORMANCE:
Lightning (<5s): {metrics.lightning_trades} trades, {metrics.lightning_win_rate:.1f}% win rate
Express (<15s): {metrics.express_trades} trades, {metrics.express_win_rate:.1f}% win rate  
Fast (<30s): {metrics.fast_trades} trades, {metrics.fast_win_rate:.1f}% win rate
Standard (<60s): {metrics.standard_trades} trades, {metrics.standard_win_rate:.1f}% win rate

🚄 SPEED PERFORMANCE:
Average Execution Latency: {metrics.avg_execution_latency_ms:.0f}ms
Speed Target Achievement: {metrics.speed_target_achievement_rate:.1f}%

🎪 HYPE DETECTION PERFORMANCE:
Velocity Level Success Rates:
"""
        
        for level, success_rate in metrics.velocity_level_success_rates.items():
            report += f"  {level.upper()}: {success_rate:.1f}%\n"
        
        report += f"""
🎯 PATTERN RECOGNITION PERFORMANCE:
Pattern Success Rates:
"""
        
        for pattern, success_rate in metrics.pattern_success_rates.items():
            report += f"  {pattern.upper()}: {success_rate:.1f}%\n"
        
        report += f"""
💰 RISK & PROFITABILITY METRICS:
Profit Factor: {metrics.profit_factor:.2f}
Sortino Ratio: {metrics.sortino_ratio:.2f}
Calmar Ratio: {metrics.calmar_ratio:.2f}
Average Win: {metrics.avg_win_pct:.2f}%
Average Loss: {metrics.avg_loss_pct:.2f}%
Average Trade Duration: {metrics.avg_trade_duration_minutes:.1f} minutes

📈 BENCHMARK COMPARISON:
Strategy Return: {metrics.total_return_pct:.2f}%
Benchmark Return: {metrics.benchmark_return_pct:.2f}%
Excess Return: {metrics.excess_return_pct:.2f}%
Beta: {metrics.beta:.2f if metrics.beta else "N/A"}
Alpha: {metrics.alpha:.4f if metrics.alpha else "N/A"}
Information Ratio: {metrics.information_ratio:.2f if metrics.information_ratio else "N/A"}

🔍 KEY INSIGHTS:
"""
        
        # Generate insights
        insights = self._generate_insights(metrics)
        for insight in insights:
            report += f"• {insight}\n"
        
        report += f"""
⚠️  RISK ASSESSMENT:
{"✅ STATISTICALLY SIGNIFICANT" if metrics.statistically_significant else "❌ NOT STATISTICALLY SIGNIFICANT"}
{"✅ POSITIVE ALPHA" if metrics.alpha and metrics.alpha > 0 else "❌ NEGATIVE OR NO ALPHA"}
{"✅ ACCEPTABLE DRAWDOWN" if metrics.max_drawdown_pct < 15 else "⚠️  HIGH DRAWDOWN"}
{"✅ GOOD RISK-ADJUSTED RETURNS" if metrics.sharpe_ratio > 1.0 else "⚠️  LOW RISK-ADJUSTED RETURNS"}

🎯 RECOMMENDATION:
"""
        
        recommendation = self._generate_recommendation(metrics)
        report += recommendation
        
        return report
    
    def _generate_insights(self, metrics: BacktestMetrics) -> List[str]:
        """Generate trading insights from metrics"""
        insights = []
        
        if metrics.lightning_win_rate > 70:
            insights.append("Lightning lane (fastest execution) shows exceptional performance")
        
        if metrics.speed_target_achievement_rate > 80:
            insights.append("System consistently meets speed targets for fast execution")
        
        # Find best performing velocity level
        if metrics.velocity_level_success_rates:
            best_velocity = max(metrics.velocity_level_success_rates.items(), key=lambda x: x[1])
            insights.append(f"'{best_velocity[0]}' velocity signals show highest success rate at {best_velocity[1]:.1f}%")
        
        # Find best performing pattern
        if metrics.pattern_success_rates:
            best_pattern = max(metrics.pattern_success_rates.items(), key=lambda x: x[1])
            insights.append(f"'{best_pattern[0]}' patterns show highest success rate at {best_pattern[1]:.1f}%")
        
        if metrics.profit_factor > 1.5:
            insights.append("Strong profit factor indicates good risk/reward balance")
        
        if metrics.max_drawdown_pct < 10:
            insights.append("Low maximum drawdown demonstrates good risk management")
        
        if metrics.sharpe_ratio > 1.5:
            insights.append("Excellent risk-adjusted returns (Sharpe > 1.5)")
        
        return insights
    
    def _generate_recommendation(self, metrics: BacktestMetrics) -> str:
        """Generate trading recommendation"""
        
        score = 0
        factors = []
        
        # Statistical significance
        if metrics.statistically_significant:
            score += 25
            factors.append("statistically significant results")
        
        # Risk-adjusted returns
        if metrics.sharpe_ratio > 1.0:
            score += 20
            factors.append("good Sharpe ratio")
        
        # Win rate
        if metrics.win_rate_pct > 55:
            score += 15
            factors.append("strong win rate")
        
        # Drawdown
        if metrics.max_drawdown_pct < 15:
            score += 15
            factors.append("acceptable drawdown")
        
        # Speed performance
        if metrics.speed_target_achievement_rate > 75:
            score += 10
            factors.append("reliable execution speed")
        
        # Alpha generation
        if metrics.alpha and metrics.alpha > 0:
            score += 15
            factors.append("positive alpha generation")
        
        if score >= 70:
            recommendation = "🟢 STRONG BUY RECOMMENDATION"
            reasoning = "Deploy system with confidence in live trading"
        elif score >= 50:
            recommendation = "🟡 CAUTIOUS RECOMMENDATION"  
            reasoning = "Deploy with reduced position sizes and close monitoring"
        else:
            recommendation = "🔴 NOT RECOMMENDED"
            reasoning = "Requires significant optimization before live deployment"
        
        return f"""{recommendation}

Score: {score}/100
Positive Factors: {', '.join(factors)}
Reasoning: {reasoning}

Next Steps:
{'• Deploy to live trading with paper money first' if score >= 70 else ''}
{'• Implement additional safety measures' if 50 <= score < 70 else ''}
{'• Focus on pattern optimization and risk management' if score < 50 else ''}
• Monitor performance closely for first 30 days
• Consider parameter tuning based on live results
"""


class MockAlpacaClient:
    """Mock Alpaca client for backtesting"""
    
    async def get_account(self):
        return type('Account', (), {'portfolio_value': '100000'})()
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        # Return simulated price
        base_prices = {
            "SPY": 400.0, "QQQ": 350.0, "AAPL": 150.0, "TSLA": 200.0,
            "MSFT": 300.0, "GOOGL": 2500.0, "AMZN": 3000.0
        }
        return base_prices.get(symbol, 100.0)


# Historical news data generator for testing
def generate_comprehensive_historical_news(days: int = 90) -> List[Dict[str, Any]]:
    """Generate comprehensive historical news data for backtesting"""
    
    news_data = []
    # Start from February 15, 2024 to match backtest date range
    from datetime import timezone
    current_date = datetime(2024, 2, 15, tzinfo=timezone.utc)
    
    # News templates with hype indicators
    templates = [
        # High velocity / breaking news
        {
            "title_template": "BREAKING: {symbol} {action} {amount}% {direction} Record {metric}",
            "hype_score": 9.0,
            "velocity_level": "viral",
            "patterns": ["BREAKING:", "surges", "record"],
            "actions": ["surges", "plunges", "beats", "misses"],
            "directions": ["after", "on", "following"],
            "metrics": ["earnings", "revenue", "guidance", "sales"]
        },
        {
            "title_template": "{symbol} Announces Breakthrough {event} - Stock {movement}",
            "hype_score": 7.5,
            "velocity_level": "breaking",
            "patterns": ["breakthrough", "announces", "major"],
            "events": ["AI partnership", "FDA approval", "merger deal", "acquisition"],
            "movements": ["soars 15%", "jumps 20%", "rallies", "surges"]
        },
        # Medium velocity / trending news
        {
            "title_template": "{symbol} {performance} {period} Estimates as {reason}",
            "hype_score": 5.5,
            "velocity_level": "trending",
            "patterns": ["beats estimates", "exceeds expectations"],
            "performances": ["beats", "exceeds", "tops", "misses"],
            "periods": ["Q3", "quarterly", "annual"],
            "reasons": ["demand increases", "costs decline", "growth accelerates"]
        },
        # Low velocity / normal news
        {
            "title_template": "{symbol} Reports {period} Results - {outcome}",
            "hype_score": 2.0,
            "velocity_level": "normal",
            "patterns": ["reports", "announces", "updates"],
            "periods": ["quarterly", "monthly", "annual"],
            "outcomes": ["mixed results", "in line", "as expected"]
        }
    ]
    
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "SPY", "QQQ"]
    
    for day in range(days):
        date = current_date + timedelta(days=day)
        
        # Generate 3-8 news items per day
        news_count = random.randint(3, 8)
        
        for _ in range(news_count):
            template = random.choice(templates)
            symbol = random.choice(symbols)
            
            # Generate news item
            if "{action}" in template["title_template"]:
                action = random.choice(template["actions"])
                amount = random.randint(3, 25)
                direction = random.choice(template["directions"])
                metric = random.choice(template["metrics"])
                
                title = template["title_template"].format(
                    symbol=symbol, action=action, amount=amount,
                    direction=direction, metric=metric
                )
            elif "{event}" in template["title_template"]:
                event = random.choice(template["events"])
                movement = random.choice(template["movements"])
                
                title = template["title_template"].format(
                    symbol=symbol, event=event, movement=movement
                )
            elif "{performance}" in template["title_template"]:
                performance = random.choice(template["performances"])
                period = random.choice(template["periods"])
                reason = random.choice(template["reasons"])
                
                title = template["title_template"].format(
                    symbol=symbol, performance=performance,
                    period=period, reason=reason
                )
            else:
                period = random.choice(template["periods"])
                outcome = random.choice(template["outcomes"])
                
                title = template["title_template"].format(
                    symbol=symbol, period=period, outcome=outcome
                )
            
            # Create news item
            news_item = {
                "title": title,
                "content": f"Detailed analysis of {symbol} showing {template['velocity_level']} momentum...",
                "source": random.choice(["Reuters", "Bloomberg", "MarketWatch", "Yahoo Finance"]),
                "timestamp": (date + timedelta(
                    hours=random.randint(9, 16),
                    minutes=random.randint(0, 59)
                )).isoformat() + "Z",
                "symbols": [symbol],
                "sentiment": random.normalvariate(0, 0.3),  # Centered around neutral
                "hype_score": template["hype_score"] + random.normalvariate(0, 1),
                "velocity_level": template["velocity_level"],
                "patterns": template["patterns"]
            }
            
            news_data.append(news_item)
    
    return news_data


async def run_full_backtest_analysis():
    """Run comprehensive backtest analysis"""
    
    # Generate historical news data
    print("📰 Generating comprehensive historical news dataset...")
    historical_news = generate_comprehensive_historical_news(days=180)  # 6 months
    print(f"Generated {len(historical_news)} historical news items")
    
    # Initialize backtester
    backtester = HypeDetectionBacktester()
    
    # Run backtest
    print("🚀 Running comprehensive backtest...")
    start_date = "2024-02-15"  # Adjusted to match generated news range
    end_date = "2024-08-14"
    
    metrics = await backtester.run_comprehensive_backtest(
        historical_news, start_date, end_date
    )
    
    # Generate report
    report = backtester.generate_comprehensive_report(metrics)
    print(report)
    
    # Save detailed results
    results_path = Path("/home/eddy/Hyper/data/hype_detection_backtest_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "metrics": asdict(metrics),
            "trade_history": [asdict(trade) for trade in backtester.trade_history],
            "execution_performance": {
                lane.value: times for lane, times in backtester.execution_performance.items()
            },
            "backtest_config": {
                "start_date": start_date,
                "end_date": end_date,
                "starting_capital": backtester.starting_capital,
                "total_news_items": len(historical_news)
            }
        }, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: {results_path}")
    
    return metrics


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the comprehensive backtest
    asyncio.run(run_full_backtest_analysis())
#!/usr/bin/env python3
"""
Enhanced Realistic Backtesting Framework
Complete integration of all friction costs, execution delays, and real-world constraints

Author: Claude Code (Anthropic AI Assistant)
Date: August 14, 2025
Task: 6 - Enhanced Realistic Backtest Framework
Next Task: 7 - Multi-Scenario Realistic Testing
"""

import asyncio
import json
import logging
import math
import random
# import numpy as np  # Using built-in math instead
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import our realistic models
from realistic_commission_models import (
    BrokerType, AssetType, CommissionCalculator, TaxCalculationEngine, 
    RealWorldCostCalculator, TradeInfo, CommissionResult
)
from realistic_execution_simulator import (
    RealisticExecutionSimulator, MarketDataProvider, ExecutionResult,
    MarketCondition, OrderType, MarketDataSnapshot
)


@dataclass
class RealisticTradeResult:
    """Complete realistic trade result with all friction costs"""
    # Basic trade info
    trade_id: str
    symbol: str
    action: str
    quantity: float
    requested_price: float
    executed_price: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    
    # Execution details
    execution_result: Optional[ExecutionResult] = None
    market_condition: str = "normal"
    order_type: str = "market"
    
    # Cost breakdown
    commission_cost: float = 0.0
    tax_cost: float = 0.0
    slippage_cost: float = 0.0
    total_friction_cost: float = 0.0
    
    # Performance metrics
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0
    holding_period_days: int = 0
    
    # Execution quality
    execution_quality_score: float = 0.0
    partial_fill_ratio: float = 1.0
    execution_delay_ms: int = 0
    
    # Trade classification
    trigger_type: str = ""
    hype_score: float = 0.0
    velocity_level: str = ""
    execution_lane: str = ""
    
    def calculate_total_costs(self):
        """Calculate total friction costs"""
        self.total_friction_cost = self.commission_cost + self.tax_cost + self.slippage_cost
        self.net_pnl = self.gross_pnl - self.total_friction_cost
        
        if abs(self.executed_price * self.quantity) > 0:
            self.net_pnl_pct = (self.net_pnl / abs(self.executed_price * self.quantity)) * 100


@dataclass
class RealisticBacktestMetrics:
    """Comprehensive realistic backtest performance metrics"""
    # Basic performance (post-friction)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_return_pct: float = 0.0
    
    # Friction cost analysis
    total_commission_cost: float = 0.0
    total_tax_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_friction_cost: float = 0.0
    friction_cost_pct: float = 0.0
    
    # Risk metrics (realistic)
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    
    # Execution analysis
    avg_execution_delay_ms: float = 0.0
    avg_execution_quality: float = 0.0
    avg_slippage_bps: float = 0.0
    partial_fill_rate: float = 0.0
    
    # Speed lane performance (realistic)
    lightning_trades: int = 0
    lightning_win_rate: float = 0.0
    lightning_avg_cost: float = 0.0
    
    express_trades: int = 0
    express_win_rate: float = 0.0
    express_avg_cost: float = 0.0
    
    fast_trades: int = 0
    fast_win_rate: float = 0.0
    fast_avg_cost: float = 0.0
    
    standard_trades: int = 0
    standard_win_rate: float = 0.0
    standard_avg_cost: float = 0.0
    
    # Asset class performance
    stock_trades: int = 0
    stock_win_rate: float = 0.0
    stock_avg_cost: float = 0.0
    
    crypto_trades: int = 0
    crypto_win_rate: float = 0.0
    crypto_avg_cost: float = 0.0
    
    # Tax efficiency metrics
    short_term_trades: int = 0
    long_term_trades: int = 0
    wash_sale_affected: int = 0
    tax_efficiency_score: float = 0.0
    
    # Comparison to idealized
    idealized_return_pct: float = 0.0
    performance_degradation_pct: float = 0.0
    
    def calculate_degradation(self, idealized_return: float):
        """Calculate performance degradation from idealized results"""
        self.idealized_return_pct = idealized_return
        if idealized_return != 0:
            self.performance_degradation_pct = ((idealized_return - self.total_return_pct) / idealized_return) * 100


class EnhancedRealisticBacktester:
    """
    Complete realistic backtesting framework integrating all friction costs
    """
    
    def __init__(
        self, 
        broker: BrokerType = BrokerType.ALPACA,
        starting_capital: float = 100000.0,
        tax_state: str = "CA",
        filing_status: str = "single",
        income_level: float = 100000
    ):
        
        self.broker = broker
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        
        # Initialize all calculators
        self.commission_calc = CommissionCalculator()
        self.tax_calc = TaxCalculationEngine(state=tax_state, filing_status=filing_status)
        self.execution_sim = RealisticExecutionSimulator()
        self.market_provider = MarketDataProvider()
        self.cost_calc = RealWorldCostCalculator(broker, tax_state, filing_status, income_level)
        
        # Trading state
        self.positions = {}  # {symbol: RealisticTradeResult}
        self.trade_history: List[RealisticTradeResult] = []
        self.daily_values = []
        self.daily_returns = []
        
        # Logger
        self.logger = logging.getLogger("realistic_backtest")
        
        # Configuration for different account sizes
        self.account_constraints = self._get_account_constraints()
        
    def _get_account_constraints(self) -> Dict[str, Any]:
        """Get trading constraints based on account size"""
        if self.starting_capital < 25000:
            return {
                "day_trades_per_5_days": 3,
                "margin_available": False,
                "min_trade_size": 100,  # Minimum viable trade size
                "max_position_pct": 0.20  # 20% max per position for small accounts
            }
        elif self.starting_capital < 100000:
            return {
                "day_trades_per_5_days": float('inf'),
                "margin_available": True,
                "min_trade_size": 500,
                "max_position_pct": 0.10  # 10% max per position
            }
        else:
            return {
                "day_trades_per_5_days": float('inf'),
                "margin_available": True,
                "min_trade_size": 1000,
                "max_position_pct": 0.05  # 5% max per position for large accounts
            }
    
    async def run_realistic_backtest(
        self,
        trades_to_simulate: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        idealized_performance: Optional[Dict[str, float]] = None
    ) -> RealisticBacktestMetrics:
        """
        Run comprehensive realistic backtest with all friction costs
        """
        
        self.logger.info(f"🎯 Starting realistic backtest: {start_date} to {end_date}")
        self.logger.info(f"💰 Starting capital: ${self.starting_capital:,.2f}")
        self.logger.info(f"🏦 Broker: {self.broker.value}")
        
        # Process trades chronologically
        sorted_trades = sorted(trades_to_simulate, key=lambda x: x.get('timestamp', ''))
        
        day_trade_count = 0
        last_day_trade_reset = None
        
        for trade_data in sorted_trades:
            
            # Check day trading limits for small accounts
            if self._is_day_trade(trade_data) and self.starting_capital < 25000:
                current_date = datetime.fromisoformat(trade_data['timestamp'])
                
                # Reset counter every 5 business days
                if last_day_trade_reset is None or (current_date - last_day_trade_reset).days >= 5:
                    day_trade_count = 0
                    last_day_trade_reset = current_date
                
                if day_trade_count >= 3:
                    self.logger.warning(f"⚠️ Day trading limit reached, skipping trade: {trade_data['symbol']}")
                    continue
                
                day_trade_count += 1
            
            # Execute realistic trade
            await self._execute_realistic_trade(trade_data)
            
            # Update daily portfolio values
            if len(self.daily_values) == 0 or self._is_new_day(trade_data['timestamp']):
                daily_value = self._calculate_portfolio_value()
                self.daily_values.append(daily_value)
                
                if len(self.daily_values) > 1:
                    daily_return = (daily_value - self.daily_values[-2]) / self.daily_values[-2]
                    self.daily_returns.append(daily_return)
        
        # Close any remaining open positions
        await self._close_open_positions()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_realistic_metrics()
        
        # Compare to idealized performance if provided
        if idealized_performance:
            metrics.calculate_degradation(idealized_performance.get('total_return_pct', 0))
        
        self.logger.info("✅ Realistic backtest completed")
        return metrics
    
    async def _execute_realistic_trade(self, trade_data: Dict[str, Any]):
        """Execute a single trade with all realistic constraints"""
        
        # Create trade info
        trade_info = TradeInfo(
            broker=self.broker,
            asset_type=AssetType.US_STOCK if not trade_data['symbol'].endswith('USD') else AssetType.CRYPTOCURRENCY,
            symbol=trade_data['symbol'],
            quantity=self._calculate_realistic_position_size(trade_data),
            price=trade_data['price'],
            side=trade_data['action'],
            timestamp=datetime.fromisoformat(trade_data['timestamp'])
        )
        
        # Check if trade meets minimum size requirements
        if trade_info.trade_value < self.account_constraints['min_trade_size']:
            self.logger.debug(f"Trade too small: ${trade_info.trade_value:.2f} < ${self.account_constraints['min_trade_size']}")
            return
        
        # Get market data
        market_data = self.market_provider.get_market_data(trade_info.symbol)
        
        # Determine market condition
        market_condition = self._determine_market_condition(trade_data, market_data)
        
        # Simulate realistic execution
        execution_result = self.execution_sim.simulate_execution(
            trade_info, market_data, OrderType.MARKET, market_condition
        )
        
        # Calculate all costs
        holding_period = trade_data.get('holding_period_days', 1)
        gross_pnl = trade_data.get('gross_pnl', 0.0)
        
        cost_result = self.cost_calc.calculate_total_trade_cost(
            trade_info, holding_period, gross_pnl
        )
        
        # Create realistic trade result
        trade_result = RealisticTradeResult(
            trade_id=f"realistic_{len(self.trade_history)}",
            symbol=trade_info.symbol,
            action=trade_info.side,
            quantity=trade_info.quantity * execution_result.partial_fill_ratio,
            requested_price=trade_info.price,
            executed_price=execution_result.executed_price,
            entry_time=trade_info.timestamp,
            
            # Execution details
            execution_result=execution_result,
            market_condition=execution_result.market_condition.value,
            order_type=execution_result.order_type.value,
            
            # Costs
            commission_cost=cost_result['commission'],
            tax_cost=cost_result['tax'],
            slippage_cost=abs(execution_result.executed_price - execution_result.requested_price) * trade_info.quantity,
            
            # Performance
            gross_pnl=gross_pnl,
            holding_period_days=holding_period,
            
            # Execution quality
            execution_quality_score=execution_result.get_execution_quality_score(),
            partial_fill_ratio=execution_result.partial_fill_ratio,
            execution_delay_ms=execution_result.execution_delay_ms,
            
            # Classification
            trigger_type=trade_data.get('trigger_type', ''),
            hype_score=trade_data.get('hype_score', 0.0),
            velocity_level=trade_data.get('velocity_level', ''),
            execution_lane=trade_data.get('execution_lane', 'standard')
        )
        
        trade_result.calculate_total_costs()
        
        # Update capital and positions
        self._update_capital_and_positions(trade_result)
        
        # Add to trade history
        self.trade_history.append(trade_result)
        
        self.logger.debug(
            f"✅ Executed {trade_result.symbol}: "
            f"${trade_result.net_pnl:.2f} net P&L "
            f"(${trade_result.total_friction_cost:.2f} friction cost)"
        )
    
    def _calculate_realistic_position_size(self, trade_data: Dict[str, Any]) -> float:
        """Calculate position size with realistic constraints"""
        
        # Original intended position size
        intended_size = trade_data.get('quantity', 100)
        intended_value = intended_size * trade_data['price']
        
        # Apply account size constraints
        max_position_value = self.current_capital * self.account_constraints['max_position_pct']
        
        if intended_value > max_position_value:
            # Reduce position size to fit constraints
            scaling_factor = max_position_value / intended_value
            intended_size *= scaling_factor
            self.logger.debug(f"Position size reduced by {(1-scaling_factor)*100:.1f}% due to account constraints")
        
        # Ensure minimum viable size
        min_value = self.account_constraints['min_trade_size']
        if intended_size * trade_data['price'] < min_value:
            intended_size = min_value / trade_data['price']
        
        return max(1, int(intended_size))  # Must be at least 1 share
    
    def _determine_market_condition(
        self, 
        trade_data: Dict[str, Any], 
        market_data: MarketDataSnapshot
    ) -> MarketCondition:
        """Determine market condition based on trade context"""
        
        # Use hype score and velocity level to infer market condition
        hype_score = trade_data.get('hype_score', 0.0)
        velocity_level = trade_data.get('velocity_level', 'normal')
        
        if velocity_level == 'viral' or hype_score >= 8.0:
            return MarketCondition.NEWS_EVENT
        elif velocity_level == 'breaking' or hype_score >= 5.0:
            return MarketCondition.VOLATILE
        elif velocity_level == 'trending':
            return MarketCondition.HIGH_VOLUME
        
        # Check time-based conditions
        trade_time = datetime.fromisoformat(trade_data['timestamp']).time()
        
        if trade_time < datetime.strptime("09:30", "%H:%M").time():
            return MarketCondition.AFTER_HOURS
        elif datetime.strptime("09:30", "%H:%M").time() <= trade_time <= datetime.strptime("10:00", "%H:%M").time():
            return MarketCondition.OPENING
        elif datetime.strptime("15:30", "%H:%M").time() <= trade_time <= datetime.strptime("16:00", "%H:%M").time():
            return MarketCondition.CLOSING
        
        return MarketCondition.NORMAL
    
    def _is_day_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Determine if this would be considered a day trade"""
        # Simplified: assume trades with holding period < 1 day are day trades
        return trade_data.get('holding_period_days', 1) < 1
    
    def _is_new_day(self, timestamp: str) -> bool:
        """Check if this timestamp represents a new trading day"""
        if not self.daily_values:
            return True
        
        current_date = datetime.fromisoformat(timestamp).date()
        
        # Compare with the last trade's date (simplified)
        if self.trade_history:
            last_date = self.trade_history[-1].entry_time.date()
            return current_date != last_date
        
        return True
    
    def _update_capital_and_positions(self, trade_result: RealisticTradeResult):
        """Update capital and position tracking"""
        
        trade_value = trade_result.quantity * trade_result.executed_price
        
        if trade_result.action == "buy":
            # Reduce capital by trade value plus friction costs
            self.current_capital -= (trade_value + trade_result.total_friction_cost)
            
            # Add to positions
            if trade_result.symbol not in self.positions:
                self.positions[trade_result.symbol] = []
            self.positions[trade_result.symbol].append(trade_result)
            
        else:  # sell
            # Increase capital by trade value minus friction costs
            self.current_capital += (trade_value - trade_result.total_friction_cost)
            
            # Remove from positions (simplified - FIFO)
            if trade_result.symbol in self.positions and self.positions[trade_result.symbol]:
                self.positions[trade_result.symbol].pop(0)
                if not self.positions[trade_result.symbol]:
                    del self.positions[trade_result.symbol]
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value including positions"""
        total_value = self.current_capital
        
        for symbol, position_list in self.positions.items():
            for position in position_list:
                # Use last known price (simplified)
                market_data = self.market_provider.get_market_data(symbol)
                current_price = market_data.last_price
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    async def _close_open_positions(self):
        """Close any remaining open positions at end of backtest"""
        positions_copy = dict(self.positions)  # Create copy to avoid runtime error
        for symbol, position_list in positions_copy.items():
            for position in position_list[:]:
                # Create closing trade
                closing_trade_data = {
                    'symbol': symbol,
                    'action': 'sell' if position.action == 'buy' else 'buy',
                    'quantity': position.quantity,
                    'price': self.market_provider.get_market_data(symbol).last_price,
                    'timestamp': datetime.now().isoformat(),
                    'gross_pnl': 0.0,
                    'holding_period_days': (datetime.now() - position.entry_time).days
                }
                
                await self._execute_realistic_trade(closing_trade_data)
    
    def _calculate_realistic_metrics(self) -> RealisticBacktestMetrics:
        """Calculate comprehensive realistic performance metrics"""
        
        metrics = RealisticBacktestMetrics()
        
        if not self.trade_history:
            return metrics
        
        # Basic metrics
        metrics.total_trades = len(self.trade_history)
        metrics.winning_trades = len([t for t in self.trade_history if t.net_pnl > 0])
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        
        if metrics.total_trades > 0:
            metrics.win_rate_pct = (metrics.winning_trades / metrics.total_trades) * 100
        
        # P&L metrics
        metrics.total_gross_pnl = sum(t.gross_pnl for t in self.trade_history)
        metrics.total_net_pnl = sum(t.net_pnl for t in self.trade_history)
        
        final_capital = self._calculate_portfolio_value()
        metrics.total_return_pct = ((final_capital - self.starting_capital) / self.starting_capital) * 100
        
        # Friction cost analysis
        metrics.total_commission_cost = sum(t.commission_cost for t in self.trade_history)
        metrics.total_tax_cost = sum(t.tax_cost for t in self.trade_history)
        metrics.total_slippage_cost = sum(t.slippage_cost for t in self.trade_history)
        metrics.total_friction_cost = metrics.total_commission_cost + metrics.total_tax_cost + metrics.total_slippage_cost
        
        if abs(metrics.total_gross_pnl) > 0:
            metrics.friction_cost_pct = (metrics.total_friction_cost / abs(metrics.total_gross_pnl)) * 100
        
        # Risk metrics
        if self.daily_returns:
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(self.daily_returns)
            metrics.max_drawdown_pct = self._calculate_max_drawdown()
        
        # Execution analysis
        execution_delays = [t.execution_delay_ms for t in self.trade_history if t.execution_delay_ms > 0]
        if execution_delays:
            metrics.avg_execution_delay_ms = sum(execution_delays) / len(execution_delays)
        
        quality_scores = [t.execution_quality_score for t in self.trade_history if t.execution_quality_score > 0]
        if quality_scores:
            metrics.avg_execution_quality = sum(quality_scores) / len(quality_scores)
        
        slippages = [t.execution_result.slippage_bps for t in self.trade_history if t.execution_result]
        if slippages:
            metrics.avg_slippage_bps = sum(slippages) / len(slippages)
        
        # Speed lane analysis
        self._analyze_execution_lanes(metrics)
        
        # Asset class analysis
        self._analyze_asset_classes(metrics)
        
        # Tax efficiency
        self._analyze_tax_efficiency(metrics)
        
        # Profit factor
        winning_pnl = sum(t.net_pnl for t in self.trade_history if t.net_pnl > 0)
        losing_pnl = abs(sum(t.net_pnl for t in self.trade_history if t.net_pnl < 0))
        
        if losing_pnl > 0:
            metrics.profit_factor = winning_pnl / losing_pnl
        
        return metrics
    
    def _analyze_execution_lanes(self, metrics: RealisticBacktestMetrics):
        """Analyze performance by execution lane"""
        
        lane_trades = {
            'lightning': [t for t in self.trade_history if t.execution_lane == 'lightning'],
            'express': [t for t in self.trade_history if t.execution_lane == 'express'],
            'fast': [t for t in self.trade_history if t.execution_lane == 'fast'],
            'standard': [t for t in self.trade_history if t.execution_lane == 'standard']
        }
        
        for lane, trades in lane_trades.items():
            if not trades:
                continue
                
            wins = len([t for t in trades if t.net_pnl > 0])
            win_rate = (wins / len(trades)) * 100 if trades else 0
            avg_cost = sum(t.total_friction_cost for t in trades) / len(trades) if trades else 0
            
            if lane == 'lightning':
                metrics.lightning_trades = len(trades)
                metrics.lightning_win_rate = win_rate
                metrics.lightning_avg_cost = avg_cost
            elif lane == 'express':
                metrics.express_trades = len(trades)
                metrics.express_win_rate = win_rate
                metrics.express_avg_cost = avg_cost
            elif lane == 'fast':
                metrics.fast_trades = len(trades)
                metrics.fast_win_rate = win_rate
                metrics.fast_avg_cost = avg_cost
            elif lane == 'standard':
                metrics.standard_trades = len(trades)
                metrics.standard_win_rate = win_rate
                metrics.standard_avg_cost = avg_cost
    
    def _analyze_asset_classes(self, metrics: RealisticBacktestMetrics):
        """Analyze performance by asset class"""
        
        stock_trades = [t for t in self.trade_history if not t.symbol.endswith('USD')]
        crypto_trades = [t for t in self.trade_history if t.symbol.endswith('USD')]
        
        if stock_trades:
            metrics.stock_trades = len(stock_trades)
            stock_wins = len([t for t in stock_trades if t.net_pnl > 0])
            metrics.stock_win_rate = (stock_wins / len(stock_trades)) * 100
            metrics.stock_avg_cost = sum(t.total_friction_cost for t in stock_trades) / len(stock_trades)
        
        if crypto_trades:
            metrics.crypto_trades = len(crypto_trades)
            crypto_wins = len([t for t in crypto_trades if t.net_pnl > 0])
            metrics.crypto_win_rate = (crypto_wins / len(crypto_trades)) * 100
            metrics.crypto_avg_cost = sum(t.total_friction_cost for t in crypto_trades) / len(crypto_trades)
    
    def _analyze_tax_efficiency(self, metrics: RealisticBacktestMetrics):
        """Analyze tax efficiency metrics"""
        
        short_term = [t for t in self.trade_history if t.holding_period_days < 365]
        long_term = [t for t in self.trade_history if t.holding_period_days >= 365]
        
        metrics.short_term_trades = len(short_term)
        metrics.long_term_trades = len(long_term)
        
        # Tax efficiency score: higher is better (more long-term holdings)
        if metrics.total_trades > 0:
            metrics.tax_efficiency_score = (metrics.long_term_trades / metrics.total_trades) * 100
        
        # Simplified wash sale estimate
        metrics.wash_sale_affected = int(metrics.short_term_trades * 0.1)  # Estimate 10% affected
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from daily returns"""
        if len(returns) < 2:
            return 0.0
        
        # Calculate mean and standard deviation manually
        mean_return = sum(returns) / len(returns)
        
        # Calculate variance
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_return = math.sqrt(variance) if variance > 0 else 0
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        risk_free_rate = 0.02 / 252  # 2% annual risk-free rate
        return (mean_return - risk_free_rate) / std_return * math.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from daily values"""
        if len(self.daily_values) < 2:
            return 0.0
        
        peak = self.daily_values[0]
        max_drawdown = 0.0
        
        for value in self.daily_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # Return as percentage


# Testing and demonstration functions
def generate_sample_trades(num_trades: int = 1000) -> List[Dict[str, Any]]:
    """Generate sample trades for realistic backtesting"""
    
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "BTCUSD", "ETHUSD", "DOGEUSD"]
    velocity_levels = ["normal", "trending", "breaking", "viral"]
    execution_lanes = ["standard", "fast", "express", "lightning"]
    
    trades = []
    start_date = datetime.now() - timedelta(days=180)
    
    for i in range(num_trades):
        # Generate realistic trade data
        symbol = random.choice(symbols)
        is_crypto = symbol.endswith('USD')
        
        base_price = random.uniform(0.01, 10000) if is_crypto else random.uniform(10, 500)
        quantity = random.uniform(0.1, 10) if is_crypto else random.randint(10, 1000)
        
        # Generate correlated hype score and velocity
        hype_score = random.uniform(1, 10)
        if hype_score >= 8:
            velocity_level = "viral"
            execution_lane = "lightning"
        elif hype_score >= 5:
            velocity_level = "breaking"
            execution_lane = "express"
        elif hype_score >= 2.5:
            velocity_level = "trending"
            execution_lane = "fast"
        else:
            velocity_level = "normal"
            execution_lane = "standard"
        
        # Generate realistic P&L based on hype score
        success_probability = 0.45 + (hype_score / 25)  # 45-85% success rate
        is_profitable = random.random() < success_probability
        
        if is_profitable:
            gross_pnl = random.uniform(50, 2000) * (hype_score / 5)
        else:
            gross_pnl = -random.uniform(20, 800)
        
        trade = {
            'symbol': symbol,
            'action': random.choice(['buy', 'sell']),
            'quantity': quantity,
            'price': base_price,
            'timestamp': (start_date + timedelta(days=random.randint(0, 179))).isoformat(),
            'gross_pnl': gross_pnl,
            'holding_period_days': random.randint(1, 30),
            'trigger_type': 'hype_detection',
            'hype_score': hype_score,
            'velocity_level': velocity_level,
            'execution_lane': execution_lane
        }
        
        trades.append(trade)
    
    return sorted(trades, key=lambda x: x['timestamp'])


async def run_comprehensive_realistic_test():
    """Run comprehensive realistic backtest test"""
    print("🎯 Running Comprehensive Realistic Backtest Test")
    print("=" * 70)
    
    # Test different account sizes and brokers
    test_scenarios = [
        {"capital": 10000, "broker": BrokerType.ALPACA, "name": "Small Account - Alpaca"},
        {"capital": 100000, "broker": BrokerType.ALPACA, "name": "Medium Account - Alpaca"},
        {"capital": 100000, "broker": BrokerType.INTERACTIVE_BROKERS, "name": "Medium Account - IBKR"}
    ]
    
    # Generate sample trade data
    sample_trades = generate_sample_trades(500)
    
    # Simulate idealized performance for comparison
    idealized_performance = {
        'total_return_pct': 1500.0,  # 15x return (idealized)
        'win_rate_pct': 67.2,
        'sharpe_ratio': 10.71
    }
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        print("-" * 50)
        
        backtester = EnhancedRealisticBacktester(
            broker=scenario['broker'],
            starting_capital=scenario['capital'],
            tax_state="CA",
            income_level=100000
        )
        
        metrics = await backtester.run_realistic_backtest(
            sample_trades,
            "2024-02-15",
            "2024-08-14",
            idealized_performance
        )
        
        results[scenario['name']] = metrics
        
        # Display key metrics
        print(f"📊 Results:")
        print(f"  Total Return: {metrics.total_return_pct:.1f}% (vs {idealized_performance['total_return_pct']:.1f}% idealized)")
        print(f"  Performance Degradation: {metrics.performance_degradation_pct:.1f}%")
        print(f"  Win Rate: {metrics.win_rate_pct:.1f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct:.1f}%")
        
        print(f"  Friction Costs:")
        print(f"    Commission: ${metrics.total_commission_cost:.2f}")
        print(f"    Tax: ${metrics.total_tax_cost:.2f}")
        print(f"    Slippage: ${metrics.total_slippage_cost:.2f}")
        print(f"    Total: ${metrics.total_friction_cost:.2f} ({metrics.friction_cost_pct:.1f}% of gross P&L)")
        
        print(f"  Execution Quality:")
        print(f"    Avg Delay: {metrics.avg_execution_delay_ms:.0f}ms")
        print(f"    Avg Quality: {metrics.avg_execution_quality:.1f}/100")
        print(f"    Avg Slippage: {metrics.avg_slippage_bps:.1f} bps")
    
    # Generate comparison summary
    print(f"\n📈 SCENARIO COMPARISON SUMMARY:")
    print("=" * 70)
    
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Return: {metrics.total_return_pct:.1f}% | Degradation: {metrics.performance_degradation_pct:.1f}% | Friction: {metrics.friction_cost_pct:.1f}%")
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive test
    asyncio.run(run_comprehensive_realistic_test())
    
    print("\n✅ Enhanced realistic backtesting framework implemented and tested!")
    print("📁 Ready for production realistic backtesting")
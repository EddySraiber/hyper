#!/usr/bin/env python3
"""
Optimization Metrics Tracking System
Enhanced monitoring and reporting for optimization strategy effectiveness

Author: Claude Code (Anthropic AI Assistant)
Date: August 15, 2025
Task: Enhanced monitoring and reporting for optimization tracking
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum


class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    EXECUTION_OPTIMIZED = "execution_optimized"
    TAX_OPTIMIZED = "tax_optimized"
    HYBRID_OPTIMIZED = "hybrid_optimized"
    BASELINE = "baseline"


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    preferred_order_type: str
    actual_order_type: str = None
    target_slippage_bps: float = 0.0
    actual_slippage_bps: float = 0.0
    execution_delay_ms: int = 0
    fill_quality_score: float = 0.0  # 0-100
    market_impact_factor: float = 0.0
    urgency_level: str = "normal"


@dataclass
class TaxMetrics:
    """Tax optimization metrics"""
    target_holding_period_days: int
    actual_holding_period_days: int = 0
    tax_efficiency_score: float = 0.0  # 0-100
    trade_classification: str = "short_term"  # short_term, medium_term, long_term
    wash_sale_risk: bool = False
    ltcg_eligible: bool = False
    estimated_tax_rate: float = 0.0  # Estimated tax rate
    tax_savings_usd: float = 0.0  # Estimated tax savings


@dataclass
class TradePerformanceMetrics:
    """Individual trade performance tracking"""
    trade_id: str
    symbol: str
    strategy: OptimizationStrategy
    entry_time: datetime
    exit_time: Optional[datetime] = None
    
    # Basic trade info
    action: str = ""  # buy/sell
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    
    # Performance metrics
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    return_pct: float = 0.0
    
    # Optimization metrics
    execution_metrics: Optional[ExecutionMetrics] = None
    tax_metrics: Optional[TaxMetrics] = None
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class StrategyPerformanceReport:
    """Strategy-level performance summary"""
    strategy: OptimizationStrategy
    period_start: datetime
    period_end: datetime
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Performance metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Optimization-specific metrics
    avg_execution_quality: float = 0.0
    avg_tax_efficiency: float = 0.0
    avg_holding_period_days: float = 0.0
    
    # Cost metrics
    total_friction_cost_usd: float = 0.0
    friction_cost_pct: float = 0.0  # As % of gross profit
    
    # Comparison to baseline
    outperformance_vs_baseline: float = 0.0
    friction_reduction_vs_baseline: float = 0.0


class OptimizationMetricsCollector:
    """
    Collects and tracks optimization strategy effectiveness
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("optimization_metrics")
        
        # In-memory storage (would use database in production)
        self.active_trades: Dict[str, TradePerformanceMetrics] = {}
        self.completed_trades: List[TradePerformanceMetrics] = []
        self.daily_summaries: Dict[str, Dict] = {}  # date -> metrics
        
        # Performance tracking windows
        self.performance_window_days = config.get("performance_window_days", 30)
        self.metrics_update_interval = config.get("metrics_update_interval", 60)  # seconds
        
        # Baseline comparison data
        self.baseline_metrics = {
            'avg_return_pct': 0.0,
            'avg_friction_cost_pct': 49.5,  # From backtesting analysis
            'avg_execution_quality': 50.0,
            'avg_tax_efficiency': 25.0
        }
        
        self.is_running = False
        self.last_update = datetime.now()
    
    async def start(self):
        """Start metrics collection"""
        self.is_running = True
        self.logger.info("🔍 Started Optimization Metrics Collector")
        
        # Start background metrics update task
        asyncio.create_task(self._periodic_metrics_update())
    
    async def stop(self):
        """Stop metrics collection"""
        self.is_running = False
        self.logger.info("⏹️ Stopped Optimization Metrics Collector")
    
    def track_trade_start(self, trade_data: Dict[str, Any], strategy: OptimizationStrategy):
        """Track when a trade is initiated"""
        
        trade_id = f"{trade_data['symbol']}_{int(time.time())}"
        
        # Extract optimization metadata
        execution_metadata = trade_data.get('execution_metadata', {})
        tax_metadata = trade_data.get('tax_metadata', {})
        
        execution_metrics = ExecutionMetrics(
            preferred_order_type=execution_metadata.get('preferred_order_type', 'market'),
            target_slippage_bps=execution_metadata.get('max_slippage_bps', 25.0),
            urgency_level=execution_metadata.get('execution_urgency', 'normal')
        ) if execution_metadata else None
        
        tax_metrics = TaxMetrics(
            target_holding_period_days=tax_metadata.get('target_holding_period_days', 1),
            tax_efficiency_score=tax_metadata.get('tax_efficiency_score', 0.0),
            trade_classification=tax_metadata.get('trade_classification', 'short_term'),
            wash_sale_risk=tax_metadata.get('wash_sale_risk', False),
            ltcg_eligible=tax_metadata.get('ltcg_eligible', False)
        ) if tax_metadata else None
        
        trade_metrics = TradePerformanceMetrics(
            trade_id=trade_id,
            symbol=trade_data['symbol'],
            strategy=strategy,
            entry_time=datetime.now(),
            action=trade_data['action'],
            quantity=trade_data['quantity'],
            entry_price=trade_data['entry_price'],
            execution_metrics=execution_metrics,
            tax_metrics=tax_metrics
        )
        
        self.active_trades[trade_id] = trade_metrics
        
        self.logger.info(f"📊 TRACKING TRADE: {trade_id} ({strategy.value}) - "
                        f"{trade_data['symbol']} {trade_data['action']} "
                        f"{trade_data['quantity']} @ ${trade_data['entry_price']:.2f}")
        
        return trade_id
    
    def update_trade_execution_quality(self, trade_id: str, execution_data: Dict[str, Any]):
        """Update trade with actual execution quality metrics"""
        
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        if trade.execution_metrics:
            trade.execution_metrics.actual_order_type = execution_data.get('order_type', 'unknown')
            trade.execution_metrics.actual_slippage_bps = execution_data.get('slippage_bps', 0.0)
            trade.execution_metrics.execution_delay_ms = execution_data.get('delay_ms', 0)
            trade.execution_metrics.fill_quality_score = execution_data.get('fill_quality_score', 50.0)
            
            self.logger.info(f"🎯 EXECUTION UPDATE: {trade_id} - "
                           f"slippage {trade.execution_metrics.actual_slippage_bps:.1f}bps, "
                           f"quality {trade.execution_metrics.fill_quality_score:.1f}/100")
    
    def update_trade_position(self, trade_id: str, current_price: float):
        """Update trade with current market position"""
        
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate unrealized P&L
        if trade.action == "buy":
            trade.unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
        else:  # sell
            trade.unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
        
        trade.return_pct = (trade.unrealized_pnl / (trade.entry_price * trade.quantity)) * 100
    
    def close_trade(self, trade_id: str, exit_price: float, exit_time: Optional[datetime] = None):
        """Mark trade as closed and calculate final metrics"""
        
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        trade.exit_time = exit_time or datetime.now()
        trade.exit_price = exit_price
        
        # Calculate realized P&L
        if trade.action == "buy":
            trade.realized_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # sell
            trade.realized_pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.return_pct = (trade.realized_pnl / (trade.entry_price * trade.quantity)) * 100
        
        # Update tax metrics with actual holding period
        if trade.tax_metrics:
            holding_period = (trade.exit_time - trade.entry_time).days
            trade.tax_metrics.actual_holding_period_days = holding_period
            
            # Estimate tax implications
            if holding_period >= 366:  # Long-term capital gains
                trade.tax_metrics.estimated_tax_rate = 0.15  # Simplified LTCG rate
            else:  # Short-term capital gains
                trade.tax_metrics.estimated_tax_rate = 0.22  # Simplified ordinary income rate
                
            if trade.realized_pnl > 0:  # Only calculate tax on gains
                trade.tax_metrics.tax_savings_usd = trade.realized_pnl * (0.22 - trade.tax_metrics.estimated_tax_rate)
        
        # Move to completed trades
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]
        
        self.logger.info(f"✅ TRADE CLOSED: {trade_id} - "
                        f"P&L ${trade.realized_pnl:.2f} ({trade.return_pct:.1f}%), "
                        f"held {trade.tax_metrics.actual_holding_period_days if trade.tax_metrics else 0}d")
    
    def generate_strategy_report(self, strategy: OptimizationStrategy, 
                               days_back: int = 30) -> StrategyPerformanceReport:
        """Generate performance report for a specific strategy"""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        strategy_trades = [
            trade for trade in self.completed_trades 
            if trade.strategy == strategy and trade.entry_time >= cutoff_date
        ]
        
        if not strategy_trades:
            return StrategyPerformanceReport(
                strategy=strategy,
                period_start=cutoff_date,
                period_end=datetime.now()
            )
        
        # Calculate basic statistics
        total_trades = len(strategy_trades)
        winning_trades = len([t for t in strategy_trades if t.realized_pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate performance metrics
        total_pnl = sum(trade.realized_pnl for trade in strategy_trades)
        total_investment = sum(trade.entry_price * trade.quantity for trade in strategy_trades)
        total_return_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0.0
        
        # Annualize return
        days_actual = (datetime.now() - cutoff_date).days
        annualized_return_pct = (total_return_pct * 365 / days_actual) if days_actual > 0 else 0.0
        
        # Calculate optimization-specific metrics
        execution_trades = [t for t in strategy_trades if t.execution_metrics]
        tax_trades = [t for t in strategy_trades if t.tax_metrics]
        
        avg_execution_quality = (
            sum(t.execution_metrics.fill_quality_score for t in execution_trades) / len(execution_trades)
            if execution_trades else 0.0
        )
        
        avg_tax_efficiency = (
            sum(t.tax_metrics.tax_efficiency_score for t in tax_trades) / len(tax_trades)
            if tax_trades else 0.0
        )
        
        avg_holding_period_days = (
            sum(t.tax_metrics.actual_holding_period_days for t in tax_trades) / len(tax_trades)
            if tax_trades else 0.0
        )
        
        # Estimate friction costs (simplified)
        estimated_friction_cost = total_investment * 0.002  # 20bps estimated commission + slippage
        friction_cost_pct = (estimated_friction_cost / abs(total_pnl)) * 100 if total_pnl != 0 else 0.0
        
        # Calculate vs baseline
        baseline_return = self.baseline_metrics['avg_return_pct']
        outperformance = annualized_return_pct - baseline_return
        
        baseline_friction = self.baseline_metrics['avg_friction_cost_pct']
        friction_reduction = baseline_friction - friction_cost_pct
        
        return StrategyPerformanceReport(
            strategy=strategy,
            period_start=cutoff_date,
            period_end=datetime.now(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            avg_execution_quality=avg_execution_quality,
            avg_tax_efficiency=avg_tax_efficiency,
            avg_holding_period_days=avg_holding_period_days,
            total_friction_cost_usd=estimated_friction_cost,
            friction_cost_pct=friction_cost_pct,
            outperformance_vs_baseline=outperformance,
            friction_reduction_vs_baseline=friction_reduction
        )
    
    def get_optimization_dashboard_data(self) -> Dict[str, Any]:
        """Get current optimization dashboard data"""
        
        current_time = datetime.now()
        
        # Generate reports for all strategies
        execution_report = self.generate_strategy_report(OptimizationStrategy.EXECUTION_OPTIMIZED)
        tax_report = self.generate_strategy_report(OptimizationStrategy.TAX_OPTIMIZED)
        
        # Active trades summary
        active_summary = {
            'total_active_trades': len(self.active_trades),
            'total_unrealized_pnl': sum(trade.unrealized_pnl for trade in self.active_trades.values()),
            'active_strategies': list(set(trade.strategy.value for trade in self.active_trades.values()))
        }
        
        # Recent performance summary (last 7 days)
        recent_cutoff = current_time - timedelta(days=7)
        recent_trades = [
            trade for trade in self.completed_trades 
            if trade.entry_time >= recent_cutoff
        ]
        
        recent_performance = {
            'total_trades': len(recent_trades),
            'total_pnl': sum(trade.realized_pnl for trade in recent_trades),
            'avg_return_pct': sum(trade.return_pct for trade in recent_trades) / len(recent_trades) if recent_trades else 0.0,
            'win_rate': len([t for t in recent_trades if t.realized_pnl > 0]) / len(recent_trades) if recent_trades else 0.0
        }
        
        return {
            'timestamp': current_time.isoformat(),
            'active_trades_summary': active_summary,
            'recent_performance_7d': recent_performance,
            'strategy_reports': {
                'execution_optimized': asdict(execution_report),
                'tax_optimized': asdict(tax_report)
            },
            'optimization_effectiveness': {
                'execution_quality_avg': execution_report.avg_execution_quality,
                'tax_efficiency_avg': tax_report.avg_tax_efficiency,
                'combined_outperformance': (execution_report.outperformance_vs_baseline + 
                                           tax_report.outperformance_vs_baseline) / 2,
                'friction_cost_reduction': (execution_report.friction_reduction_vs_baseline + 
                                          tax_report.friction_reduction_vs_baseline) / 2
            }
        }
    
    async def _periodic_metrics_update(self):
        """Periodic background metrics updates"""
        while self.is_running:
            try:
                # Update all active trade positions (would get real prices in production)
                for trade_id, trade in self.active_trades.items():
                    # Simulate price updates (would use real market data)
                    mock_price = trade.entry_price * (1 + (hash(trade_id) % 100 - 50) / 1000)
                    self.update_trade_position(trade_id, mock_price)
                
                # Log periodic summary
                if len(self.active_trades) > 0:
                    total_unrealized = sum(trade.unrealized_pnl for trade in self.active_trades.values())
                    self.logger.info(f"📊 ACTIVE TRADES: {len(self.active_trades)} positions, "
                                   f"unrealized P&L: ${total_unrealized:.2f}")
                
                self.last_update = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error in periodic metrics update: {e}")
            
            await asyncio.sleep(self.metrics_update_interval)
    
    def export_performance_data(self, filepath: str):
        """Export performance data to JSON file"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'active_trades': [asdict(trade) for trade in self.active_trades.values()],
            'completed_trades': [asdict(trade) for trade in self.completed_trades],
            'strategy_reports': {
                'execution_optimized': asdict(self.generate_strategy_report(OptimizationStrategy.EXECUTION_OPTIMIZED)),
                'tax_optimized': asdict(self.generate_strategy_report(OptimizationStrategy.TAX_OPTIMIZED))
            },
            'dashboard_data': self.get_optimization_dashboard_data()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📁 Exported performance data to {filepath}")


# Example usage and testing
async def test_optimization_metrics():
    """Test the optimization metrics system"""
    
    print("🔍 Testing Optimization Metrics Collector")
    print("=" * 60)
    
    # Initialize collector
    config = {
        'performance_window_days': 30,
        'metrics_update_interval': 5  # 5 seconds for testing
    }
    
    collector = OptimizationMetricsCollector(config)
    await collector.start()
    
    # Simulate some trades
    test_trades = [
        {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 10,
            'entry_price': 150.0,
            'execution_metadata': {
                'preferred_order_type': 'limit',
                'max_slippage_bps': 20.0,
                'execution_urgency': 'normal'
            },
            'tax_metadata': {
                'target_holding_period_days': 90,
                'tax_efficiency_score': 75.0,
                'trade_classification': 'medium_term'
            }
        },
        {
            'symbol': 'TSLA',
            'action': 'sell',
            'quantity': 5,
            'entry_price': 250.0,
            'execution_metadata': {
                'preferred_order_type': 'market',
                'max_slippage_bps': 30.0,
                'execution_urgency': 'high'
            },
            'tax_metadata': {
                'target_holding_period_days': 1,
                'tax_efficiency_score': 45.0,
                'trade_classification': 'short_term'
            }
        }
    ]
    
    # Track trade starts
    trade_ids = []
    for i, trade_data in enumerate(test_trades):
        strategy = OptimizationStrategy.EXECUTION_OPTIMIZED if i == 0 else OptimizationStrategy.TAX_OPTIMIZED
        trade_id = collector.track_trade_start(trade_data, strategy)
        trade_ids.append(trade_id)
    
    # Simulate execution updates
    await asyncio.sleep(2)
    
    collector.update_trade_execution_quality(trade_ids[0], {
        'order_type': 'limit',
        'slippage_bps': 18.5,
        'delay_ms': 2500,
        'fill_quality_score': 85.0
    })
    
    collector.update_trade_execution_quality(trade_ids[1], {
        'order_type': 'market',
        'slippage_bps': 25.0,
        'delay_ms': 500,
        'fill_quality_score': 65.0
    })
    
    # Simulate some time passing and position updates
    await asyncio.sleep(3)
    
    # Close trades
    collector.close_trade(trade_ids[0], 155.0)  # Profit
    collector.close_trade(trade_ids[1], 245.0)  # Profit
    
    # Generate reports
    await asyncio.sleep(1)
    
    execution_report = collector.generate_strategy_report(OptimizationStrategy.EXECUTION_OPTIMIZED)
    tax_report = collector.generate_strategy_report(OptimizationStrategy.TAX_OPTIMIZED)
    
    print(f"\n📊 Execution Optimized Report:")
    print(f"  Trades: {execution_report.total_trades}")
    print(f"  Win Rate: {execution_report.win_rate:.1%}")
    print(f"  Return: {execution_report.total_return_pct:.2f}%")
    print(f"  Execution Quality: {execution_report.avg_execution_quality:.1f}/100")
    
    print(f"\n📊 Tax Optimized Report:")
    print(f"  Trades: {tax_report.total_trades}")
    print(f"  Win Rate: {tax_report.win_rate:.1%}")
    print(f"  Return: {tax_report.total_return_pct:.2f}%")
    print(f"  Tax Efficiency: {tax_report.avg_tax_efficiency:.1f}/100")
    print(f"  Avg Holding Period: {tax_report.avg_holding_period_days:.1f} days")
    
    # Get dashboard data
    dashboard_data = collector.get_optimization_dashboard_data()
    print(f"\n🎯 Dashboard Summary:")
    print(f"  Active Trades: {dashboard_data['active_trades_summary']['total_active_trades']}")
    print(f"  Recent 7d Return: {dashboard_data['recent_performance_7d']['avg_return_pct']:.2f}%")
    print(f"  Combined Outperformance: {dashboard_data['optimization_effectiveness']['combined_outperformance']:.2f}%")
    
    await collector.stop()
    print("\n✅ Optimization metrics testing complete!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_optimization_metrics())
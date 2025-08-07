from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import time
import logging
import asyncio
import threading
from ..core.base import ComponentBase


@dataclass
class TradingMetrics:
    """Core trading performance metrics"""
    
    # Performance Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Position Metrics
    active_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    portfolio_value: float = 100000.0
    available_cash: float = 100000.0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    risk_utilization: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    
    # Trading Activity
    trades_per_hour: float = 0.0
    avg_trade_duration: float = 0.0
    avg_win_amount: float = 0.0
    avg_loss_amount: float = 0.0
    
    # News & Signal Metrics
    news_processed: int = 0
    signals_generated: int = 0
    signal_accuracy: float = 0.0
    news_to_trade_ratio: float = 0.0
    
    # System Health
    system_uptime: float = 0.0
    api_response_time: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Enhanced Features
    trailing_stops_active: int = 0
    trailing_stops_triggered: int = 0
    enhanced_signals_used: int = 0
    ai_analysis_success_rate: float = 0.0
    
    # Time-based metrics
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if self.losing_trades == 0 or self.avg_loss_amount == 0:
            return float('inf') if self.winning_trades > 0 else 0.0
        
        gross_profit = self.winning_trades * self.avg_win_amount
        gross_loss = abs(self.losing_trades * self.avg_loss_amount)
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if self.total_trades == 0:
            return 0.0
            
        avg_return = self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0
        # Simplified calculation - in production would need historical returns
        return max(avg_return - risk_free_rate, 0) if avg_return > 0 else 0.0
    
    def to_prometheus_metrics(self) -> Dict[str, Any]:
        """Convert to Prometheus-compatible metrics"""
        return {
            # Trading Performance
            'trading_total_trades': self.total_trades,
            'trading_winning_trades': self.winning_trades,
            'trading_losing_trades': self.losing_trades,
            'trading_win_rate_percent': self.win_rate(),
            'trading_total_pnl_usd': self.total_pnl,
            'trading_realized_pnl_usd': self.realized_pnl,
            'trading_unrealized_pnl_usd': self.unrealized_pnl,
            'trading_profit_factor': min(self.profit_factor(), 10.0),  # Cap for display
            'trading_sharpe_ratio': self.sharpe_ratio(),
            
            # Portfolio Metrics
            'portfolio_value_usd': self.portfolio_value,
            'portfolio_cash_usd': self.available_cash,
            'portfolio_active_positions': self.active_positions,
            'portfolio_long_positions': self.long_positions,
            'portfolio_short_positions': self.short_positions,
            
            # Risk Metrics
            'risk_max_drawdown_percent': self.max_drawdown * 100,
            'risk_current_drawdown_percent': self.current_drawdown * 100,
            'risk_utilization_percent': self.risk_utilization * 100,
            'risk_var_95_usd': self.var_95,
            
            # Activity Metrics
            'activity_trades_per_hour': self.trades_per_hour,
            'activity_avg_trade_duration_minutes': self.avg_trade_duration,
            'activity_news_processed': self.news_processed,
            'activity_signals_generated': self.signals_generated,
            'activity_news_to_trade_ratio': self.news_to_trade_ratio,
            
            # System Health
            'system_uptime_hours': self.system_uptime,
            'system_api_response_ms': self.api_response_time * 1000,
            'system_error_rate_percent': self.error_rate * 100,
            'system_memory_usage_mb': self.memory_usage_mb,
            
            # Enhanced Features
            'features_trailing_stops_active': self.trailing_stops_active,
            'features_trailing_stops_triggered': self.trailing_stops_triggered,
            'features_enhanced_signals_used': self.enhanced_signals_used,
            'features_ai_analysis_success_rate_percent': self.ai_analysis_success_rate * 100,
        }


class MetricsCollector(ComponentBase):
    """
    Collects and aggregates trading system metrics for observability
    
    Features:
    - Real-time trading performance metrics
    - System health monitoring
    - Historical data storage
    - Prometheus export format
    - Configurable collection intervals
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("metrics_collector", config)
        
        # Configuration
        self.collection_interval = config.get('collection_interval', 30)  # 30 seconds
        self.history_retention_hours = config.get('history_retention_hours', 168)  # 1 week
        self.enable_prometheus_export = config.get('enable_prometheus_export', True)
        self.prometheus_port = config.get('prometheus_port', 9090)
        
        # Metrics storage
        self.current_metrics = TradingMetrics()
        self.metrics_history: List[TradingMetrics] = []
        self.symbol_metrics: Dict[str, Dict] = defaultdict(dict)
        
        # System state tracking
        self.start_time = datetime.utcnow()
        self.last_collection_time = self.start_time
        self.collection_errors = 0
        
        # Performance counters
        self.trade_count_last_hour = 0
        self.news_count_last_hour = 0
        self.error_count_last_hour = 0
        
        # Dependencies (will be injected)
        self.alpaca_client = None
        self.risk_manager = None
        self.trailing_stop_manager = None
        self.decision_engine = None
        
        # Background collection thread
        self.collection_thread = None
        self.stop_event = threading.Event()
        
    def start(self) -> None:
        self.logger.info("Starting Metrics Collector")
        self.is_running = True
        
        # Start background collection
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        if self.enable_prometheus_export:
            self.logger.info(f"Metrics will be available for Prometheus scraping on port {self.prometheus_port}")
        
    def stop(self) -> None:
        self.logger.info("Stopping Metrics Collector")
        self.is_running = False
        self.stop_event.set()
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _collection_loop(self):
        """Background thread for periodic metrics collection"""
        while not self.stop_event.wait(self.collection_interval):
            try:
                asyncio.run(self._collect_metrics())
            except Exception as e:
                self.collection_errors += 1
                self.logger.error(f"Error in metrics collection: {e}")
    
    async def _collect_metrics(self):
        """Collect all system metrics"""
        collection_start = time.time()
        
        try:
            # Collect trading metrics
            await self._collect_trading_metrics()
            
            # Collect system health metrics
            self._collect_system_metrics()
            
            # Update timestamp
            self.current_metrics.timestamp = datetime.utcnow()
            
            # Store in history
            self._store_metrics_history()
            
            # Clean old data
            self._cleanup_old_metrics()
            
            # Update collection timing
            self.last_collection_time = datetime.utcnow()
            
        except Exception as e:
            self.collection_errors += 1
            self.logger.error(f"Error collecting metrics: {e}")
        
        # Track collection performance
        collection_time = time.time() - collection_start
        self.logger.debug(f"Metrics collection completed in {collection_time:.3f}s")
    
    async def _collect_trading_metrics(self):
        """Collect trading performance and portfolio metrics"""
        
        # Portfolio metrics from Alpaca
        if self.alpaca_client:
            try:
                account_info = await self.alpaca_client.get_account_info()
                positions = await self.alpaca_client.get_positions()
                
                self.current_metrics.portfolio_value = account_info.get('portfolio_value', 0)
                self.current_metrics.available_cash = account_info.get('cash', 0)
                
                # Position metrics
                long_pos = sum(1 for pos in positions if pos.get('quantity', 0) > 0)
                short_pos = sum(1 for pos in positions if pos.get('quantity', 0) < 0)
                
                self.current_metrics.active_positions = len(positions)
                self.current_metrics.long_positions = long_pos
                self.current_metrics.short_positions = short_pos
                
                # P&L metrics
                total_unrealized = sum(pos.get('unrealized_pl', 0) for pos in positions)
                self.current_metrics.unrealized_pnl = total_unrealized
                
            except Exception as e:
                self.logger.warning(f"Error collecting trading metrics from Alpaca: {e}")
        
        # Risk metrics from RiskManager
        if self.risk_manager:
            try:
                portfolio_status = self.risk_manager.get_portfolio_status()
                self.current_metrics.risk_utilization = portfolio_status.get('risk_utilization', 0)
                # Add more risk metrics as available
                
            except Exception as e:
                self.logger.warning(f"Error collecting risk metrics: {e}")
        
        # Trailing stops metrics
        if self.trailing_stop_manager:
            try:
                trailing_status = self.trailing_stop_manager.get_trailing_stop_status()
                self.current_metrics.trailing_stops_active = len(trailing_status)
                
            except Exception as e:
                self.logger.warning(f"Error collecting trailing stop metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system health and performance metrics"""
        import psutil
        
        # System uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        self.current_metrics.system_uptime = uptime
        
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.current_metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # Error rate (simple implementation)
        total_operations = max(self.current_metrics.news_processed + self.current_metrics.total_trades, 1)
        self.current_metrics.error_rate = self.collection_errors / total_operations
    
    def _store_metrics_history(self):
        """Store current metrics in history"""
        # Create a copy of current metrics
        import copy
        metrics_snapshot = copy.deepcopy(self.current_metrics)
        self.metrics_history.append(metrics_snapshot)
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.history_retention_hours)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    def record_trade(self, trade_result: Dict[str, Any]):
        """Record a completed trade"""
        self.current_metrics.total_trades += 1
        
        pnl = trade_result.get('pnl', 0)
        self.current_metrics.total_pnl += pnl
        
        if pnl > 0:
            self.current_metrics.winning_trades += 1
            # Update average win
            current_total_wins = self.current_metrics.avg_win_amount * (self.current_metrics.winning_trades - 1)
            self.current_metrics.avg_win_amount = (current_total_wins + pnl) / self.current_metrics.winning_trades
        else:
            self.current_metrics.losing_trades += 1
            # Update average loss
            current_total_losses = self.current_metrics.avg_loss_amount * (self.current_metrics.losing_trades - 1)
            self.current_metrics.avg_loss_amount = (current_total_losses + abs(pnl)) / self.current_metrics.losing_trades
    
    def record_news_processed(self, count: int = 1):
        """Record news items processed"""
        self.current_metrics.news_processed += count
    
    def record_signal_generated(self, signal_data: Dict[str, Any]):
        """Record a trading signal generated"""
        self.current_metrics.signals_generated += 1
        
        # Update news-to-trade ratio
        if self.current_metrics.news_processed > 0:
            self.current_metrics.news_to_trade_ratio = (
                self.current_metrics.signals_generated / self.current_metrics.news_processed
            )
    
    def record_api_call(self, response_time: float, success: bool = True):
        """Record API call performance"""
        # Update rolling average of response time
        current_avg = self.current_metrics.api_response_time
        self.current_metrics.api_response_time = (current_avg * 0.9 + response_time * 0.1)
        
        if not success:
            self.collection_errors += 1
    
    def get_current_metrics(self) -> TradingMetrics:
        """Get current metrics snapshot"""
        return self.current_metrics
    
    def get_metrics_history(self, hours: int = 24) -> List[TradingMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not self.enable_prometheus_export:
            return ""
        
        metrics = self.current_metrics.to_prometheus_metrics()
        prometheus_output = []
        
        for metric_name, value in metrics.items():
            prometheus_output.append(f"# TYPE {metric_name} gauge")
            prometheus_output.append(f"{metric_name} {value}")
        
        # Add metadata
        prometheus_output.insert(0, "# HELP Trading system metrics")
        prometheus_output.append(f"# Last updated: {self.current_metrics.timestamp}")
        
        return '\n'.join(prometheus_output)
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get human-readable trading summary"""
        metrics = self.current_metrics
        
        return {
            "performance": {
                "total_trades": metrics.total_trades,
                "win_rate": f"{metrics.win_rate():.1f}%",
                "total_pnl": f"${metrics.total_pnl:.2f}",
                "profit_factor": f"{metrics.profit_factor():.2f}",
                "sharpe_ratio": f"{metrics.sharpe_ratio():.2f}"
            },
            "portfolio": {
                "value": f"${metrics.portfolio_value:,.2f}",
                "cash": f"${metrics.available_cash:,.2f}",
                "positions": metrics.active_positions,
                "long_positions": metrics.long_positions,
                "short_positions": metrics.short_positions
            },
            "risk": {
                "current_drawdown": f"{metrics.current_drawdown*100:.1f}%",
                "max_drawdown": f"{metrics.max_drawdown*100:.1f}%",
                "risk_utilization": f"{metrics.risk_utilization*100:.1f}%"
            },
            "system": {
                "uptime": f"{metrics.system_uptime:.1f}h",
                "memory_usage": f"{metrics.memory_usage_mb:.1f}MB",
                "api_response_time": f"{metrics.api_response_time*1000:.0f}ms",
                "error_rate": f"{metrics.error_rate*100:.2f}%"
            },
            "enhanced_features": {
                "trailing_stops_active": metrics.trailing_stops_active,
                "enhanced_signals": metrics.enhanced_signals_used,
                "ai_success_rate": f"{metrics.ai_analysis_success_rate*100:.1f}%"
            }
        }
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method (called by system)"""
        if not self.is_running:
            return {"status": "not_running"}
        
        return {
            "status": "collecting",
            "last_collection": self.last_collection_time.isoformat(),
            "collection_errors": self.collection_errors,
            "metrics_count": len(self.metrics_history)
        }
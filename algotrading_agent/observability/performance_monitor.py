"""
Real-time Performance Monitor - Continuous tracking of trading system performance

Features:
- Real-time Sharpe ratio calculation and tracking
- Risk-adjusted return monitoring
- Performance alerts and thresholds
- Rolling window statistics
- Performance degradation detection
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from collections import deque
from dataclasses import dataclass
import math


@dataclass 
class PerformanceSnapshot:
    """Snapshot of current performance metrics"""
    timestamp: datetime
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    avg_trade_duration_hours: float
    
    # Risk metrics
    var_95: float
    expected_shortfall: float
    
    # Recent performance (last N trades)
    recent_win_rate: float
    recent_sharpe: float
    recent_avg_return: float


class PerformanceMonitor:
    """
    Real-time performance monitoring and alerting system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__name__)
        
        # Configuration
        self.monitoring_enabled = config.get("monitoring_enabled", True)
        self.update_interval = config.get("update_interval", 60)  # seconds
        self.alert_thresholds = config.get("alert_thresholds", {})
        self.rolling_window_size = config.get("rolling_window_size", 50)  # trades
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        self.trading_days_per_year = config.get("trading_days_per_year", 252)
        
        # Alert thresholds
        self.min_sharpe_alert = self.alert_thresholds.get("min_sharpe", 0.5)
        self.max_drawdown_alert = self.alert_thresholds.get("max_drawdown", 0.15)  # 15%
        self.min_win_rate_alert = self.alert_thresholds.get("min_win_rate", 0.40)  # 40%
        self.consecutive_losses_alert = self.alert_thresholds.get("consecutive_losses", 5)
        
        # Data storage
        self.trade_history: deque = deque(maxlen=1000)  # Last 1000 trades
        self.performance_history: List[PerformanceSnapshot] = []
        self.rolling_returns: deque = deque(maxlen=self.rolling_window_size)
        
        # State tracking
        self.current_snapshot: Optional[PerformanceSnapshot] = None
        self.last_update = datetime.now(timezone.utc)
        self.consecutive_losses = 0
        self.peak_portfolio_value = 0.0
        self.alerts_triggered: List[str] = []
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        self.logger.info("Performance Monitor initialized")
    
    async def start_monitoring(self, trading_client):
        """Start real-time performance monitoring"""
        if not self.monitoring_enabled:
            self.logger.info("Performance monitoring disabled")
            return
        
        self.trading_client = trading_client
        self.is_monitoring = True
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("🎯 Real-time performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_performance_metrics(self):
        """Update all performance metrics"""
        try:
            # Get fresh trading data
            account = await self.trading_client.get_account_info()
            positions = await self.trading_client.get_positions()
            orders = await self.trading_client.get_orders()
            
            # Calculate current metrics
            snapshot = await self._calculate_performance_snapshot(account, positions, orders)
            
            # Update rolling data
            if self.current_snapshot:
                pnl_change = snapshot.total_pnl - self.current_snapshot.total_pnl
                if pnl_change != 0:  # Only add when there's actual trading activity
                    self.rolling_returns.append(pnl_change)
            
            self.current_snapshot = snapshot
            self.performance_history.append(snapshot)
            self.last_update = snapshot.timestamp
            
            # Check for alerts
            await self._check_performance_alerts(snapshot)
            
            # Log periodic updates
            if len(self.performance_history) % 10 == 0:
                self.logger.info(f"📊 Performance Update: Sharpe: {snapshot.sharpe_ratio:.2f}, "
                               f"Win Rate: {snapshot.win_rate:.1%}, Drawdown: {snapshot.current_drawdown:.1%}")
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    async def _calculate_performance_snapshot(self, account, positions, orders) -> PerformanceSnapshot:
        """Calculate comprehensive performance snapshot"""
        timestamp = datetime.now(timezone.utc)
        
        # Basic account metrics
        portfolio_value = float(account.get('portfolio_value', 0))
        cash = float(account.get('cash', 0))
        total_pnl = portfolio_value - 100000  # Assuming $100k starting value
        
        # Update peak for drawdown calculation
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        
        # Trade-based metrics (if we have trade history)
        total_trades = len(self.trade_history)
        win_rate = self._calculate_win_rate(self.trade_history) if self.trade_history else 0.0
        
        # Sharpe ratio from rolling returns
        sharpe_ratio = self._calculate_rolling_sharpe() if len(self.rolling_returns) > 5 else 0.0
        
        # Risk metrics
        volatility = float(np.std(list(self.rolling_returns))) if len(self.rolling_returns) > 1 else 0.0
        max_drawdown = self._calculate_max_drawdown()
        var_95 = self._calculate_var(list(self.rolling_returns), 0.95) if self.rolling_returns else 0.0
        expected_shortfall = self._calculate_expected_shortfall(list(self.rolling_returns), 0.95) if self.rolling_returns else 0.0
        
        # Trade duration analysis
        avg_trade_duration = self._calculate_avg_trade_duration(positions) if positions else 0.0
        
        # Recent performance (last 20 trades)
        recent_trades = list(self.trade_history)[-20:] if len(self.trade_history) >= 20 else list(self.trade_history)
        recent_win_rate = self._calculate_win_rate(recent_trades) if recent_trades else win_rate
        recent_returns = list(self.rolling_returns)[-20:] if len(self.rolling_returns) >= 20 else list(self.rolling_returns)
        recent_sharpe = self._calculate_sharpe_from_returns(recent_returns) if len(recent_returns) > 5 else sharpe_ratio
        recent_avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            total_trades=total_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            avg_trade_duration_hours=avg_trade_duration,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            recent_win_rate=recent_win_rate,
            recent_sharpe=recent_sharpe,
            recent_avg_return=recent_avg_return
        )
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trade list"""
        if not trades:
            return 0.0
        
        wins = sum(1 for trade in trades if float(trade.get('realized_pnl', 0)) > 0)
        return wins / len(trades)
    
    def _calculate_rolling_sharpe(self) -> float:
        """Calculate Sharpe ratio from rolling returns"""
        if len(self.rolling_returns) < 2:
            return 0.0
        
        returns = list(self.rolling_returns)
        return self._calculate_sharpe_from_returns(returns)
    
    def _calculate_sharpe_from_returns(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from return series"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualize assuming daily returns
        excess_return = mean_return - (self.risk_free_rate / self.trading_days_per_year)
        sharpe = (excess_return / std_return) * math.sqrt(self.trading_days_per_year)
        
        return sharpe
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from performance history"""
        if not self.performance_history:
            return 0.0
        
        pnls = [snapshot.total_pnl for snapshot in self.performance_history]
        cumulative = np.array(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / np.maximum(running_max, 1)  # Avoid division by zero
        
        return float(np.max(drawdown))
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_expected_shortfall(self, returns: List[float], confidence: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if not returns:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        tail_losses = [r for r in returns if r <= var]
        
        return float(np.mean(tail_losses)) if tail_losses else 0.0
    
    def _calculate_avg_trade_duration(self, positions: List[Dict[str, Any]]) -> float:
        """Estimate average trade duration from current positions"""
        if not positions:
            return 0.0
        
        # This is a simplified estimation - in a full implementation,
        # we'd track actual trade open/close times
        return 24.0  # Placeholder: assume 24 hours average
    
    async def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance-based alerts"""
        alerts = []
        
        # Sharpe ratio alert
        if snapshot.sharpe_ratio < self.min_sharpe_alert:
            alerts.append(f"Low Sharpe ratio: {snapshot.sharpe_ratio:.2f} below threshold {self.min_sharpe_alert}")
        
        # Drawdown alert
        if snapshot.current_drawdown > self.max_drawdown_alert:
            alerts.append(f"High drawdown: {snapshot.current_drawdown:.1%} exceeds threshold {self.max_drawdown_alert:.1%}")
        
        # Win rate alert
        if snapshot.win_rate < self.min_win_rate_alert and snapshot.total_trades >= 10:
            alerts.append(f"Low win rate: {snapshot.win_rate:.1%} below threshold {self.min_win_rate_alert:.1%}")
        
        # Recent performance degradation
        if (snapshot.recent_sharpe < snapshot.sharpe_ratio * 0.5 and 
            snapshot.total_trades >= 20):
            alerts.append(f"Recent performance degradation: Recent Sharpe {snapshot.recent_sharpe:.2f} vs Overall {snapshot.sharpe_ratio:.2f}")
        
        # Log new alerts
        for alert in alerts:
            if alert not in self.alerts_triggered:
                self.logger.warning(f"🚨 PERFORMANCE ALERT: {alert}")
                self.alerts_triggered.append(alert)
        
        # Clear resolved alerts
        self.alerts_triggered = [alert for alert in self.alerts_triggered if alert in alerts]
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade for performance tracking"""
        self.trade_history.append(trade_data)
        
        # Track consecutive losses
        pnl = float(trade_data.get('realized_pnl', 0))
        if pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check for consecutive loss alert
        if self.consecutive_losses >= self.consecutive_losses_alert:
            self.logger.warning(f"🚨 CONSECUTIVE LOSSES ALERT: {self.consecutive_losses} losses in a row")
    
    def get_current_performance(self) -> Optional[PerformanceSnapshot]:
        """Get current performance snapshot"""
        return self.current_snapshot
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of current performance"""
        if not self.current_snapshot:
            return {"status": "No performance data available"}
        
        snapshot = self.current_snapshot
        
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "total_trades": snapshot.total_trades,
            "win_rate": f"{snapshot.win_rate:.1%}",
            "sharpe_ratio": f"{snapshot.sharpe_ratio:.2f}",
            "total_pnl": f"${snapshot.total_pnl:.2f}",
            "current_drawdown": f"{snapshot.current_drawdown:.1%}",
            "max_drawdown": f"{snapshot.max_drawdown:.1%}",
            "var_95": f"{snapshot.var_95:.4f}",
            "recent_win_rate": f"{snapshot.recent_win_rate:.1%}",
            "recent_sharpe": f"{snapshot.recent_sharpe:.2f}",
            "consecutive_losses": self.consecutive_losses,
            "active_alerts": len(self.alerts_triggered),
            "last_update": self.last_update.isoformat()
        }
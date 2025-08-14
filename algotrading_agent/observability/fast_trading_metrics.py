"""
Fast Trading Performance Metrics - Speed and momentum trading analytics

Tracks latency, execution speed, pattern recognition accuracy, and velocity
signal performance for high-speed momentum trading operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from ..core.base import ComponentBase


@dataclass
class SpeedMetric:
    """Individual speed measurement"""
    timestamp: datetime
    operation: str          # "pattern_detection", "velocity_tracking", "express_execution"
    symbol: str
    latency_ms: int
    success: bool
    execution_lane: Optional[str] = None
    trigger_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "symbol": self.symbol,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "execution_lane": self.execution_lane,
            "trigger_type": self.trigger_type
        }


@dataclass
class PatternAccuracyMetric:
    """Pattern detection accuracy tracking"""
    pattern_type: str
    symbol: str
    detected_at: datetime
    confidence: float
    prediction: str         # "bullish", "bearish", "neutral"
    actual_outcome: Optional[str] = None    # Actual price movement
    outcome_timestamp: Optional[datetime] = None
    accuracy_score: Optional[float] = None  # 0.0 to 1.0
    profit_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "symbol": self.symbol,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
            "prediction": self.prediction,
            "actual_outcome": self.actual_outcome,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "accuracy_score": self.accuracy_score,
            "profit_loss": self.profit_loss
        }


@dataclass
class VelocitySignalMetric:
    """News velocity signal performance"""
    signal_id: str
    velocity_level: str     # "viral", "breaking", "trending", "normal"
    velocity_score: float
    symbols: List[str]
    detected_at: datetime
    trade_executed: bool = False
    execution_latency_ms: Optional[int] = None
    trade_outcome: Optional[str] = None     # "profit", "loss", "break_even"
    profit_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "velocity_level": self.velocity_level,
            "velocity_score": self.velocity_score,
            "symbols": self.symbols,
            "detected_at": self.detected_at.isoformat(),
            "trade_executed": self.trade_executed,
            "execution_latency_ms": self.execution_latency_ms,
            "trade_outcome": self.trade_outcome,
            "profit_loss": self.profit_loss
        }


class FastTradingMetrics(ComponentBase):
    """
    Performance metrics and analytics for fast trading system
    
    Tracks latency, accuracy, and profitability of momentum patterns,
    velocity signals, and express execution across all speed lanes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("fast_trading_metrics", config)
        
        # Configuration
        self.metrics_retention_hours = config.get("metrics_retention_hours", 168)  # 1 week
        self.speed_targets = config.get("speed_targets", {
            "lightning": 5000,   # 5 seconds
            "express": 15000,    # 15 seconds
            "fast": 30000,       # 30 seconds
            "standard": 60000    # 60 seconds
        })
        
        # Metrics storage
        self.speed_metrics: deque = deque(maxlen=10000)
        self.pattern_accuracy_metrics: deque = deque(maxlen=5000)
        self.velocity_signal_metrics: deque = deque(maxlen=5000)
        
        # Real-time aggregations
        self.speed_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_operations": 0,
            "successful_operations": 0,
            "total_latency_ms": 0,
            "min_latency_ms": float('inf'),
            "max_latency_ms": 0,
            "recent_latencies": deque(maxlen=100)
        })
        
        self.pattern_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_detections": 0,
            "accurate_predictions": 0,
            "total_confidence": 0.0,
            "total_profit": 0.0,
            "profitable_trades": 0,
            "losing_trades": 0
        })
        
        self.velocity_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_signals": 0,
            "traded_signals": 0,
            "successful_trades": 0,
            "total_velocity_score": 0.0,
            "total_profit": 0.0
        })
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.total_fast_trades = 0
        self.profitable_fast_trades = 0
        self.total_fast_profit = 0.0
        
        # Lane performance tracking
        self.lane_performance: Dict[str, Dict[str, Any]] = {
            "lightning": {"trades": 0, "successes": 0, "total_latency": 0, "profit": 0.0},
            "express": {"trades": 0, "successes": 0, "total_latency": 0, "profit": 0.0},
            "fast": {"trades": 0, "successes": 0, "total_latency": 0, "profit": 0.0},
            "standard": {"trades": 0, "successes": 0, "total_latency": 0, "profit": 0.0}
        }
    
    async def start(self) -> None:
        """Start fast trading metrics collection"""
        self.logger.info("📊 Starting Fast Trading Metrics")
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        self.logger.info("✅ Fast trading metrics collection started")
    
    async def stop(self) -> None:
        """Stop metrics collection and log summary"""
        self.logger.info("🛑 Stopping Fast Trading Metrics")
        self.is_running = False
        
        self._log_comprehensive_summary()
    
    def record_speed_metric(self, operation: str, symbol: str, latency_ms: int, 
                           success: bool, execution_lane: Optional[str] = None,
                           trigger_type: Optional[str] = None):
        """Record a speed/latency measurement"""
        metric = SpeedMetric(
            timestamp=datetime.utcnow(),
            operation=operation,
            symbol=symbol,
            latency_ms=latency_ms,
            success=success,
            execution_lane=execution_lane,
            trigger_type=trigger_type
        )
        
        self.speed_metrics.append(metric)
        
        # Update real-time stats
        stats = self.speed_stats[operation]
        stats["total_operations"] += 1
        if success:
            stats["successful_operations"] += 1
        
        stats["total_latency_ms"] += latency_ms
        stats["min_latency_ms"] = min(stats["min_latency_ms"], latency_ms)
        stats["max_latency_ms"] = max(stats["max_latency_ms"], latency_ms)
        stats["recent_latencies"].append(latency_ms)
        
        # Update lane performance
        if execution_lane and execution_lane in self.lane_performance:
            lane_stats = self.lane_performance[execution_lane]
            lane_stats["trades"] += 1
            lane_stats["total_latency"] += latency_ms
            if success:
                lane_stats["successes"] += 1
        
        # Log speed violations
        target_latency = self.speed_targets.get(execution_lane, 60000) if execution_lane else 60000
        if latency_ms > target_latency:
            self.logger.warning(f"⚠️  Speed target missed: {operation} took {latency_ms}ms "
                              f"(target: {target_latency}ms) for {symbol}")
    
    def record_pattern_accuracy(self, pattern_type: str, symbol: str, 
                               confidence: float, prediction: str):
        """Record a pattern detection for accuracy tracking"""
        metric = PatternAccuracyMetric(
            pattern_type=pattern_type,
            symbol=symbol,
            detected_at=datetime.utcnow(),
            confidence=confidence,
            prediction=prediction
        )
        
        self.pattern_accuracy_metrics.append(metric)
        
        # Update stats
        stats = self.pattern_stats[pattern_type]
        stats["total_detections"] += 1
        stats["total_confidence"] += confidence
        
        self.logger.debug(f"📈 Pattern recorded: {pattern_type} for {symbol} "
                         f"(confidence: {confidence:.2f}, prediction: {prediction})")
    
    def update_pattern_outcome(self, pattern_type: str, symbol: str, 
                              detected_at: datetime, actual_outcome: str,
                              profit_loss: float = 0.0):
        """Update pattern accuracy with actual outcome"""
        # Find matching pattern metric
        for metric in reversed(list(self.pattern_accuracy_metrics)):
            if (metric.pattern_type == pattern_type and 
                metric.symbol == symbol and 
                abs((metric.detected_at - detected_at).total_seconds()) < 300):  # Within 5 minutes
                
                metric.actual_outcome = actual_outcome
                metric.outcome_timestamp = datetime.utcnow()
                metric.profit_loss = profit_loss
                
                # Calculate accuracy score
                if metric.prediction == actual_outcome:
                    metric.accuracy_score = 1.0
                elif (metric.prediction in ["bullish", "bearish"] and 
                      actual_outcome in ["bullish", "bearish"] and 
                      metric.prediction != actual_outcome):
                    metric.accuracy_score = 0.0  # Wrong direction
                else:
                    metric.accuracy_score = 0.5  # Partial accuracy
                
                # Update stats
                stats = self.pattern_stats[pattern_type]
                if metric.accuracy_score >= 0.5:
                    stats["accurate_predictions"] += 1
                
                stats["total_profit"] += profit_loss
                if profit_loss > 0:
                    stats["profitable_trades"] += 1
                elif profit_loss < 0:
                    stats["losing_trades"] += 1
                
                self.logger.info(f"✅ Pattern outcome updated: {pattern_type} for {symbol} "
                               f"(accuracy: {metric.accuracy_score:.1f}, P&L: ${profit_loss:.2f})")
                break
    
    def record_velocity_signal(self, signal_id: str, velocity_level: str, 
                              velocity_score: float, symbols: List[str]):
        """Record a velocity signal for performance tracking"""
        metric = VelocitySignalMetric(
            signal_id=signal_id,
            velocity_level=velocity_level,
            velocity_score=velocity_score,
            symbols=symbols,
            detected_at=datetime.utcnow()
        )
        
        self.velocity_signal_metrics.append(metric)
        
        # Update stats
        stats = self.velocity_stats[velocity_level]
        stats["total_signals"] += 1
        stats["total_velocity_score"] += velocity_score
        
        self.logger.debug(f"⚡ Velocity signal recorded: {velocity_level} "
                         f"(score: {velocity_score:.1f}, symbols: {', '.join(symbols)})")
    
    def update_velocity_trade_outcome(self, signal_id: str, execution_latency_ms: int,
                                     trade_outcome: str, profit_loss: float = 0.0):
        """Update velocity signal with trade execution outcome"""
        # Find matching velocity metric
        for metric in reversed(list(self.velocity_signal_metrics)):
            if metric.signal_id == signal_id:
                metric.trade_executed = True
                metric.execution_latency_ms = execution_latency_ms
                metric.trade_outcome = trade_outcome
                metric.profit_loss = profit_loss
                
                # Update stats
                stats = self.velocity_stats[metric.velocity_level]
                stats["traded_signals"] += 1
                stats["total_profit"] += profit_loss
                
                if trade_outcome == "profit":
                    stats["successful_trades"] += 1
                
                # Update overall fast trading stats
                self.total_fast_trades += 1
                self.total_fast_profit += profit_loss
                if profit_loss > 0:
                    self.profitable_fast_trades += 1
                
                self.logger.info(f"💰 Velocity trade completed: {metric.velocity_level} "
                               f"(latency: {execution_latency_ms}ms, outcome: {trade_outcome}, "
                               f"P&L: ${profit_loss:.2f})")
                break
    
    def record_express_trade(self, symbol: str, execution_lane: str, 
                            latency_ms: int, success: bool, profit_loss: float = 0.0):
        """Record express trade execution"""
        self.record_speed_metric("express_execution", symbol, latency_ms, success, execution_lane)
        
        # Update lane performance profit
        if execution_lane in self.lane_performance:
            self.lane_performance[execution_lane]["profit"] += profit_loss
        
        # Update overall stats
        self.total_fast_trades += 1
        self.total_fast_profit += profit_loss
        if profit_loss > 0:
            self.profitable_fast_trades += 1
    
    def get_speed_performance(self) -> Dict[str, Any]:
        """Get speed performance analytics"""
        performance = {}
        
        for operation, stats in self.speed_stats.items():
            if stats["total_operations"] > 0:
                avg_latency = stats["total_latency_ms"] / stats["total_operations"]
                success_rate = stats["successful_operations"] / stats["total_operations"] * 100
                
                # Calculate percentiles from recent latencies
                recent_latencies = sorted(list(stats["recent_latencies"]))
                if recent_latencies:
                    p50 = recent_latencies[len(recent_latencies) // 2]
                    p95 = recent_latencies[int(len(recent_latencies) * 0.95)]
                    p99 = recent_latencies[int(len(recent_latencies) * 0.99)]
                else:
                    p50 = p95 = p99 = 0
                
                performance[operation] = {
                    "total_operations": stats["total_operations"],
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                    "min_latency_ms": stats["min_latency_ms"],
                    "max_latency_ms": stats["max_latency_ms"],
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "p99_latency_ms": p99
                }
        
        return performance
    
    def get_pattern_accuracy(self) -> Dict[str, Any]:
        """Get pattern detection accuracy analytics"""
        accuracy = {}
        
        for pattern_type, stats in self.pattern_stats.items():
            if stats["total_detections"] > 0:
                accuracy_rate = (stats["accurate_predictions"] / stats["total_detections"] * 100 
                               if stats["total_detections"] > 0 else 0)
                avg_confidence = stats["total_confidence"] / stats["total_detections"]
                
                total_trades = stats["profitable_trades"] + stats["losing_trades"]
                win_rate = (stats["profitable_trades"] / total_trades * 100 
                           if total_trades > 0 else 0)
                avg_profit = stats["total_profit"] / total_trades if total_trades > 0 else 0
                
                accuracy[pattern_type] = {
                    "total_detections": stats["total_detections"],
                    "accuracy_rate": accuracy_rate,
                    "avg_confidence": avg_confidence,
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "avg_profit_per_trade": avg_profit,
                    "total_profit": stats["total_profit"]
                }
        
        return accuracy
    
    def get_velocity_performance(self) -> Dict[str, Any]:
        """Get velocity signal performance analytics"""
        performance = {}
        
        for velocity_level, stats in self.velocity_stats.items():
            if stats["total_signals"] > 0:
                trade_rate = (stats["traded_signals"] / stats["total_signals"] * 100)
                success_rate = (stats["successful_trades"] / stats["traded_signals"] * 100 
                               if stats["traded_signals"] > 0 else 0)
                avg_velocity_score = stats["total_velocity_score"] / stats["total_signals"]
                avg_profit = (stats["total_profit"] / stats["traded_signals"] 
                             if stats["traded_signals"] > 0 else 0)
                
                performance[velocity_level] = {
                    "total_signals": stats["total_signals"],
                    "traded_signals": stats["traded_signals"],
                    "trade_rate": trade_rate,
                    "success_rate": success_rate,
                    "avg_velocity_score": avg_velocity_score,
                    "avg_profit_per_trade": avg_profit,
                    "total_profit": stats["total_profit"]
                }
        
        return performance
    
    def get_lane_performance(self) -> Dict[str, Any]:
        """Get execution lane performance analytics"""
        performance = {}
        
        for lane, stats in self.lane_performance.items():
            if stats["trades"] > 0:
                success_rate = stats["successes"] / stats["trades"] * 100
                avg_latency = stats["total_latency"] / stats["trades"]
                avg_profit = stats["profit"] / stats["trades"]
                
                # Check if meeting speed targets
                target_latency = self.speed_targets.get(lane, 60000)
                meets_target = avg_latency <= target_latency
                
                performance[lane] = {
                    "total_trades": stats["trades"],
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                    "target_latency_ms": target_latency,
                    "meets_speed_target": meets_target,
                    "avg_profit_per_trade": avg_profit,
                    "total_profit": stats["profit"]
                }
        
        return performance
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """Get overall fast trading performance summary"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600  # hours
        
        win_rate = (self.profitable_fast_trades / self.total_fast_trades * 100 
                   if self.total_fast_trades > 0 else 0)
        avg_profit_per_trade = (self.total_fast_profit / self.total_fast_trades 
                               if self.total_fast_trades > 0 else 0)
        trades_per_hour = self.total_fast_trades / uptime if uptime > 0 else 0
        
        return {
            "uptime_hours": uptime,
            "total_fast_trades": self.total_fast_trades,
            "profitable_trades": self.profitable_fast_trades,
            "win_rate": win_rate,
            "total_profit": self.total_fast_profit,
            "avg_profit_per_trade": avg_profit_per_trade,
            "trades_per_hour": trades_per_hour,
            "speed_performance": self.get_speed_performance(),
            "pattern_accuracy": self.get_pattern_accuracy(),
            "velocity_performance": self.get_velocity_performance(),
            "lane_performance": self.get_lane_performance()
        }
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean speed metrics
        self.speed_metrics = deque([m for m in self.speed_metrics if m.timestamp > cutoff_time], 
                                  maxlen=10000)
        
        # Clean pattern metrics
        self.pattern_accuracy_metrics = deque([m for m in self.pattern_accuracy_metrics 
                                             if m.detected_at > cutoff_time], maxlen=5000)
        
        # Clean velocity metrics
        self.velocity_signal_metrics = deque([m for m in self.velocity_signal_metrics 
                                            if m.detected_at > cutoff_time], maxlen=5000)
    
    def _log_comprehensive_summary(self):
        """Log comprehensive performance summary"""
        performance = self.get_overall_performance()
        
        self.logger.info("🚀 FAST TRADING PERFORMANCE SUMMARY:")
        self.logger.info(f"  Uptime: {performance['uptime_hours']:.1f} hours")
        self.logger.info(f"  Total fast trades: {performance['total_fast_trades']}")
        self.logger.info(f"  Win rate: {performance['win_rate']:.1f}%")
        self.logger.info(f"  Total profit: ${performance['total_profit']:.2f}")
        self.logger.info(f"  Avg profit per trade: ${performance['avg_profit_per_trade']:.2f}")
        self.logger.info(f"  Trades per hour: {performance['trades_per_hour']:.1f}")
        
        # Log lane performance
        self.logger.info("📊 EXECUTION LANE PERFORMANCE:")
        for lane, perf in performance["lane_performance"].items():
            target_met = "✅" if perf["meets_speed_target"] else "❌"
            self.logger.info(f"  {lane}: {perf['total_trades']} trades, "
                           f"{perf['avg_latency_ms']:.0f}ms avg {target_met}, "
                           f"{perf['success_rate']:.1f}% success, "
                           f"${perf['avg_profit_per_trade']:.2f} avg profit")
        
        # Log top pattern types
        self.logger.info("🎯 TOP PATTERN ACCURACY:")
        pattern_perf = performance["pattern_accuracy"]
        for pattern_type in sorted(pattern_perf.keys(), 
                                 key=lambda x: pattern_perf[x]["accuracy_rate"], reverse=True)[:3]:
            perf = pattern_perf[pattern_type]
            self.logger.info(f"  {pattern_type}: {perf['accuracy_rate']:.1f}% accuracy, "
                           f"{perf['total_detections']} detections, "
                           f"${perf['total_profit']:.2f} total profit")
        
        # Log velocity performance
        self.logger.info("⚡ VELOCITY SIGNAL PERFORMANCE:")
        velocity_perf = performance["velocity_performance"]
        for level in ["viral", "breaking", "trending", "normal"]:
            if level in velocity_perf:
                perf = velocity_perf[level]
                self.logger.info(f"  {level}: {perf['total_signals']} signals, "
                               f"{perf['trade_rate']:.1f}% traded, "
                               f"{perf['success_rate']:.1f}% success")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive fast trading metrics status"""
        return {
            "is_running": self.is_running,
            "total_speed_metrics": len(self.speed_metrics),
            "total_pattern_metrics": len(self.pattern_accuracy_metrics),
            "total_velocity_metrics": len(self.velocity_signal_metrics),
            "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600,
            "performance_summary": self.get_overall_performance()
        }
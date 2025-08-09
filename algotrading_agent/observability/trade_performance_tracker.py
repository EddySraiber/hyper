"""
Trade Performance Tracker
Comprehensive tracking of trade outcomes, decision quality, and performance attribution
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TradeResult(Enum):
    """Trade outcome classification"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    ACTIVE = "active"
    CANCELLED = "cancelled"


class FailureReason(Enum):
    """Reasons why trades fail"""
    WEAK_SENTIMENT = "weak_sentiment"
    MARKET_OVERREACTION = "market_overreaction"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TIMING_ISSUE = "timing_issue"
    VOLUME_ISSUE = "volume_issue"
    NEWS_MISINTERPRETATION = "news_misinterpretation"
    MARKET_REVERSAL = "market_reversal"
    EXTERNAL_FACTOR = "external_factor"


@dataclass
class NewsContext:
    """News context that triggered the trade"""
    headline: str
    source: str
    sentiment_score: float
    impact_score: float
    timestamp: datetime
    symbols_mentioned: List[str]
    category: str = "general"  # earnings, merger, breaking, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass 
class TradeDecision:
    """Decision context for the trade"""
    symbol: str
    direction: str  # buy, sell, hold
    confidence: float
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    decision_factors: Dict[str, float]  # sentiment: 0.7, impact: 0.8, etc.
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TradeExecution:
    """Trade execution details"""
    order_id: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    quantity: int = 1
    fees: float = 0.0
    slippage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()
        return data


@dataclass
class TradeOutcome:
    """Complete trade outcome analysis"""
    trade_id: str
    news_context: NewsContext
    decision: TradeDecision
    execution: TradeExecution
    
    # Results
    result: TradeResult
    pnl_absolute: float = 0.0
    pnl_percentage: float = 0.0
    duration_minutes: int = 0
    
    # Analysis
    success_factors: List[str] = field(default_factory=list)
    failure_reasons: List[FailureReason] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    # Attribution
    news_accuracy: float = 0.0  # How well news predicted outcome
    decision_quality: float = 0.0  # How good was the decision process
    execution_quality: float = 0.0  # How well was it executed
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['news_context'] = self.news_context.to_dict()
        data['decision'] = self.decision.to_dict()
        data['execution'] = self.execution.to_dict()
        data['result'] = self.result.value
        data['failure_reasons'] = [reason.value for reason in self.failure_reasons]
        data['created_at'] = self.created_at.isoformat()
        return data


class TradePerformanceTracker:
    """
    Comprehensive trade performance tracking system
    Tracks entire pipeline: News → Decision → Execution → Outcome
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get('data_dir', '/app/data'))
        self.trades_file = self.data_dir / 'trade_outcomes.json'
        self.max_trades = config.get('max_trades', 1000)
        
        # In-memory storage
        self.trades: List[TradeOutcome] = []
        self.active_trades: Dict[str, TradeOutcome] = {}
        
        # Performance metrics
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Load existing data
        self._load_trades()
        self._calculate_performance_stats()
    
    def _load_trades(self):
        """Load trade outcomes from file"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    
                for item in data:
                    # Reconstruct TradeOutcome object
                    news_context = NewsContext(
                        headline=item['news_context']['headline'],
                        source=item['news_context']['source'],
                        sentiment_score=item['news_context']['sentiment_score'],
                        impact_score=item['news_context']['impact_score'],
                        timestamp=datetime.fromisoformat(item['news_context']['timestamp']),
                        symbols_mentioned=item['news_context']['symbols_mentioned'],
                        category=item['news_context'].get('category', 'general')
                    )
                    
                    decision = TradeDecision(
                        symbol=item['decision']['symbol'],
                        direction=item['decision']['direction'],
                        confidence=item['decision']['confidence'],
                        position_size=item['decision']['position_size'],
                        stop_loss=item['decision'].get('stop_loss'),
                        take_profit=item['decision'].get('take_profit'),
                        decision_factors=item['decision']['decision_factors'],
                        timestamp=datetime.fromisoformat(item['decision']['timestamp'])
                    )
                    
                    execution = TradeExecution(
                        order_id=item['execution']['order_id'],
                        entry_price=item['execution']['entry_price'],
                        entry_time=datetime.fromisoformat(item['execution']['entry_time']),
                        exit_price=item['execution'].get('exit_price'),
                        exit_time=datetime.fromisoformat(item['execution']['exit_time']) if item['execution'].get('exit_time') else None,
                        quantity=item['execution']['quantity'],
                        fees=item['execution']['fees'],
                        slippage=item['execution']['slippage']
                    )
                    
                    trade = TradeOutcome(
                        trade_id=item['trade_id'],
                        news_context=news_context,
                        decision=decision,
                        execution=execution,
                        result=TradeResult(item['result']),
                        pnl_absolute=item['pnl_absolute'],
                        pnl_percentage=item['pnl_percentage'],
                        duration_minutes=item['duration_minutes'],
                        success_factors=item['success_factors'],
                        failure_reasons=[FailureReason(reason) for reason in item['failure_reasons']],
                        lessons_learned=item['lessons_learned'],
                        news_accuracy=item['news_accuracy'],
                        decision_quality=item['decision_quality'],
                        execution_quality=item['execution_quality'],
                        created_at=datetime.fromisoformat(item['created_at'])
                    )
                    
                    self.trades.append(trade)
                    
                    # Add to active trades if still active
                    if trade.result == TradeResult.ACTIVE:
                        self.active_trades[trade.trade_id] = trade
                
                logger.info(f"Loaded {len(self.trades)} trade outcomes")
                
        except Exception as e:
            logger.error(f"Error loading trade outcomes: {e}")
            self.trades = []
    
    def _save_trades(self):
        """Save trade outcomes to file"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Keep only recent trades
            recent_trades = self.trades[-self.max_trades:]
            
            data = [trade.to_dict() for trade in recent_trades]
            
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving trade outcomes: {e}")
    
    def start_trade(
        self,
        trade_id: str,
        news_context: NewsContext,
        decision: TradeDecision,
        execution: TradeExecution
    ) -> TradeOutcome:
        """Start tracking a new trade"""
        
        trade = TradeOutcome(
            trade_id=trade_id,
            news_context=news_context,
            decision=decision,
            execution=execution,
            result=TradeResult.ACTIVE
        )
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        
        logger.info(f"Started tracking trade {trade_id}: {decision.symbol} {decision.direction}")
        
        return trade
    
    def complete_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        fees: float = 0.0
    ) -> Optional[TradeOutcome]:
        """Complete a trade and analyze the outcome"""
        
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades")
            return None
        
        trade = self.active_trades[trade_id]
        
        # Update execution details
        trade.execution.exit_price = exit_price
        trade.execution.exit_time = exit_time
        trade.execution.fees += fees
        
        # Calculate P&L
        entry_price = trade.execution.entry_price
        quantity = trade.execution.quantity
        direction_multiplier = 1 if trade.decision.direction == 'buy' else -1
        
        price_change = (exit_price - entry_price) * direction_multiplier
        trade.pnl_absolute = (price_change * quantity) - trade.execution.fees
        trade.pnl_percentage = (price_change / entry_price) * 100
        
        # Calculate duration
        duration = exit_time - trade.execution.entry_time
        trade.duration_minutes = int(duration.total_seconds() / 60)
        
        # Determine result
        if abs(trade.pnl_absolute) < 1.0:  # Within $1
            trade.result = TradeResult.BREAKEVEN
        elif trade.pnl_absolute > 0:
            trade.result = TradeResult.WIN
        else:
            trade.result = TradeResult.LOSS
        
        # Analyze the trade
        self._analyze_trade_outcome(trade)
        
        # Move from active to completed
        del self.active_trades[trade_id]
        self.trades.append(trade)
        
        # Update performance stats
        self._calculate_performance_stats()
        
        # Save to file
        self._save_trades()
        
        logger.info(f"Completed trade {trade_id}: {trade.result.value} "
                   f"P&L=${trade.pnl_absolute:.2f} ({trade.pnl_percentage:.1f}%)")
        
        return trade
    
    def _analyze_trade_outcome(self, trade: TradeOutcome):
        """Analyze why a trade succeeded or failed"""
        
        # Calculate quality scores
        trade.news_accuracy = self._calculate_news_accuracy(trade)
        trade.decision_quality = self._calculate_decision_quality(trade)
        trade.execution_quality = self._calculate_execution_quality(trade)
        
        # Identify success factors or failure reasons
        if trade.result == TradeResult.WIN:
            self._identify_success_factors(trade)
        elif trade.result == TradeResult.LOSS:
            self._identify_failure_reasons(trade)
        
        # Generate lessons learned
        self._generate_lessons_learned(trade)
    
    def _calculate_news_accuracy(self, trade: TradeOutcome) -> float:
        """Calculate how accurately the news predicted the outcome"""
        
        sentiment = trade.news_context.sentiment_score
        actual_return = trade.pnl_percentage
        
        # Check if sentiment direction matched actual direction
        sentiment_positive = sentiment > 0.1
        actual_positive = actual_return > 0
        
        direction_match = sentiment_positive == actual_positive
        
        # Score based on direction match and magnitude correlation
        if direction_match:
            # Good prediction - score based on sentiment strength vs actual return
            magnitude_score = min(abs(sentiment) / 1.0, 1.0)  # Normalize sentiment
            return 0.6 + (0.4 * magnitude_score)  # 60-100%
        else:
            # Wrong direction - low score
            return max(0.1, 0.4 - abs(sentiment) * 0.3)  # 10-40%
    
    def _calculate_decision_quality(self, trade: TradeOutcome) -> float:
        """Calculate decision quality based on process and outcome"""
        
        quality_score = 0.0
        
        # Confidence appropriateness (50% weight)
        confidence = trade.decision.confidence
        outcome_success = trade.result == TradeResult.WIN
        
        if outcome_success and confidence > 0.7:
            quality_score += 0.5  # High confidence, good outcome
        elif outcome_success and confidence > 0.4:
            quality_score += 0.3  # Medium confidence, good outcome  
        elif not outcome_success and confidence < 0.5:
            quality_score += 0.3  # Low confidence, bad outcome (good caution)
        
        # Risk management (30% weight)
        has_stop_loss = trade.decision.stop_loss is not None
        has_take_profit = trade.decision.take_profit is not None
        
        if has_stop_loss:
            quality_score += 0.15
        if has_take_profit:
            quality_score += 0.15
        
        # Position sizing (20% weight)
        position_size = trade.decision.position_size
        if 0.01 <= position_size <= 0.05:  # 1-5% of portfolio
            quality_score += 0.2
        elif position_size > 0.1:  # Too large
            quality_score -= 0.1
        
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_execution_quality(self, trade: TradeOutcome) -> float:
        """Calculate execution quality (slippage, timing, fees)"""
        
        quality_score = 0.8  # Start with good base score
        
        # Slippage penalty
        slippage_pct = abs(trade.execution.slippage) / trade.execution.entry_price * 100
        if slippage_pct > 0.5:  # More than 0.5% slippage
            quality_score -= slippage_pct * 0.1
        
        # Fee efficiency
        fees_pct = trade.execution.fees / (trade.execution.entry_price * trade.execution.quantity) * 100
        if fees_pct > 0.1:  # More than 0.1% in fees
            quality_score -= fees_pct * 0.5
        
        # Duration appropriateness (for swing trades)
        if trade.duration_minutes < 5:  # Too quick, might be noise
            quality_score -= 0.1
        elif trade.duration_minutes > 1440:  # More than 24 hours
            quality_score -= 0.05
        
        return min(1.0, max(0.1, quality_score))
    
    def _identify_success_factors(self, trade: TradeOutcome):
        """Identify what made the trade successful"""
        
        factors = []
        
        if trade.news_accuracy > 0.8:
            factors.append("Strong news signal")
        
        if trade.decision.confidence > 0.8:
            factors.append("High confidence decision")
            
        if trade.news_context.category == "breaking":
            factors.append("Breaking news advantage")
            
        if trade.duration_minutes < 60:
            factors.append("Quick execution")
            
        if trade.pnl_percentage > 5:
            factors.append("Strong market movement")
            
        trade.success_factors = factors
    
    def _identify_failure_reasons(self, trade: TradeOutcome):
        """Identify why the trade failed"""
        
        reasons = []
        
        if trade.news_accuracy < 0.4:
            reasons.append(FailureReason.NEWS_MISINTERPRETATION)
            
        if trade.decision.confidence < 0.3:
            reasons.append(FailureReason.WEAK_SENTIMENT)
            
        if trade.execution.exit_price == trade.decision.stop_loss:
            reasons.append(FailureReason.STOP_LOSS_TRIGGERED)
            
        if trade.duration_minutes < 10:
            reasons.append(FailureReason.TIMING_ISSUE)
            
        if abs(trade.pnl_percentage) > 10:  # Large adverse move
            reasons.append(FailureReason.MARKET_OVERREACTION)
            
        trade.failure_reasons = reasons
    
    def _generate_lessons_learned(self, trade: TradeOutcome):
        """Generate lessons learned from the trade"""
        
        lessons = []
        
        if trade.result == TradeResult.LOSS:
            if FailureReason.WEAK_SENTIMENT in trade.failure_reasons:
                lessons.append("Require higher sentiment confidence for similar trades")
                
            if FailureReason.STOP_LOSS_TRIGGERED in trade.failure_reasons:
                lessons.append("Consider wider stop-loss for this symbol/volatility")
                
            if trade.news_accuracy < 0.3:
                lessons.append(f"Be cautious of {trade.news_context.source} for {trade.decision.symbol}")
        
        elif trade.result == TradeResult.WIN:
            if "Strong news signal" in trade.success_factors:
                lessons.append(f"Continue leveraging {trade.news_context.source} signals")
                
            if "Breaking news advantage" in trade.success_factors:
                lessons.append("Prioritize breaking news trades")
        
        trade.lessons_learned = lessons
    
    def _calculate_performance_stats(self):
        """Calculate overall performance statistics"""
        
        if not self.trades:
            return
        
        completed_trades = [t for t in self.trades if t.result in [TradeResult.WIN, TradeResult.LOSS, TradeResult.BREAKEVEN]]
        
        if not completed_trades:
            return
        
        # Basic stats
        self.performance_stats['total_trades'] = len(completed_trades)
        winning_trades = [t for t in completed_trades if t.result == TradeResult.WIN]
        losing_trades = [t for t in completed_trades if t.result == TradeResult.LOSS]
        
        self.performance_stats['winning_trades'] = len(winning_trades)
        self.performance_stats['losing_trades'] = len(losing_trades)
        self.performance_stats['win_rate'] = len(winning_trades) / len(completed_trades) * 100
        
        # P&L stats
        total_pnl = sum(t.pnl_absolute for t in completed_trades)
        self.performance_stats['total_pnl'] = total_pnl
        
        if winning_trades:
            self.performance_stats['avg_win'] = np.mean([t.pnl_absolute for t in winning_trades])
            self.performance_stats['largest_win'] = max(t.pnl_absolute for t in winning_trades)
        
        if losing_trades:
            self.performance_stats['avg_loss'] = np.mean([t.pnl_absolute for t in losing_trades])
            self.performance_stats['largest_loss'] = min(t.pnl_absolute for t in losing_trades)
        
        # Profit factor
        gross_profit = sum(t.pnl_absolute for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl_absolute for t in losing_trades)) if losing_trades else 1
        self.performance_stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Simple Sharpe ratio approximation
        if len(completed_trades) > 1:
            returns = [t.pnl_percentage for t in completed_trades]
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            self.performance_stats['sharpe_ratio'] = (avg_return / return_std) if return_std > 0 else 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        # Recent performance (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_trades = [t for t in self.trades if t.created_at > cutoff_time]
        
        # Failure analysis
        failure_analysis = self._analyze_failures()
        
        # Source performance
        source_performance = self._analyze_source_performance()
        
        return {
            'overall_stats': self.performance_stats,
            'recent_performance': {
                'trades_24h': len(recent_trades),
                'pnl_24h': sum(t.pnl_absolute for t in recent_trades),
                'win_rate_24h': (sum(1 for t in recent_trades if t.result == TradeResult.WIN) / 
                                len(recent_trades) * 100) if recent_trades else 0
            },
            'failure_analysis': failure_analysis,
            'source_performance': source_performance,
            'active_trades': len(self.active_trades),
            'total_tracked_trades': len(self.trades)
        }
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze common failure patterns"""
        
        failed_trades = [t for t in self.trades if t.result == TradeResult.LOSS]
        
        if not failed_trades:
            return {}
        
        # Count failure reasons
        reason_counts = {}
        for trade in failed_trades:
            for reason in trade.failure_reasons:
                reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1
        
        # Calculate percentages
        total_failures = len(failed_trades)
        failure_breakdown = {
            reason: {
                'count': count,
                'percentage': count / total_failures * 100
            }
            for reason, count in reason_counts.items()
        }
        
        return {
            'total_failed_trades': total_failures,
            'failure_breakdown': failure_breakdown,
            'avg_loss_amount': np.mean([t.pnl_absolute for t in failed_trades]),
            'common_lessons': self._get_common_lessons(failed_trades)
        }
    
    def _analyze_source_performance(self) -> Dict[str, Any]:
        """Analyze performance by news source"""
        
        source_stats = {}
        
        for trade in self.trades:
            source = trade.news_context.source
            
            if source not in source_stats:
                source_stats[source] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'avg_accuracy': 0.0
                }
            
            stats = source_stats[source]
            stats['total_trades'] += 1
            
            if trade.result == TradeResult.WIN:
                stats['winning_trades'] += 1
            
            stats['total_pnl'] += trade.pnl_absolute
            stats['avg_accuracy'] += trade.news_accuracy
        
        # Calculate percentages and averages
        for source, stats in source_stats.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] * 100
                stats['avg_accuracy'] = stats['avg_accuracy'] / stats['total_trades'] * 100
        
        return source_stats
    
    def _get_common_lessons(self, failed_trades: List[TradeOutcome]) -> List[str]:
        """Extract most common lessons learned from failures"""
        
        lesson_counts = {}
        
        for trade in failed_trades:
            for lesson in trade.lessons_learned:
                lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
        
        # Return top 5 most common lessons
        sorted_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)
        return [lesson for lesson, count in sorted_lessons[:5]]
    
    def get_recent_trades(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent trades for dashboard display"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_trades = [t for t in self.trades if t.created_at > cutoff_time]
        
        # Sort by creation time, most recent first
        recent_trades.sort(key=lambda x: x.created_at, reverse=True)
        
        # Format for display
        formatted_trades = []
        for trade in recent_trades:
            formatted_trades.append({
                'symbol': trade.decision.symbol,
                'direction': trade.decision.direction,
                'entry_price': f"${trade.execution.entry_price:.2f}",
                'exit_price': f"${trade.execution.exit_price:.2f}" if trade.execution.exit_price else "Active",
                'pnl': f"${trade.pnl_absolute:+.2f}",
                'pnl_pct': f"{trade.pnl_percentage:+.1f}%",
                'duration': f"{trade.duration_minutes}m",
                'result': trade.result.value,
                'news_trigger': trade.news_context.headline[:50] + "...",
                'confidence': f"{trade.decision.confidence:.1%}",
                'news_accuracy': f"{trade.news_accuracy:.1%}"
            })
        
        return formatted_trades
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data needed for comprehensive trading dashboard"""
        
        return {
            'performance_summary': self.get_performance_summary(),
            'recent_trades': self.get_recent_trades(24),
            'active_trades': [
                {
                    'trade_id': trade_id,
                    'symbol': trade.decision.symbol,
                    'direction': trade.decision.direction,
                    'entry_price': trade.execution.entry_price,
                    'current_pnl': 0.0,  # Would need current price to calculate
                    'duration_minutes': int((datetime.now() - trade.execution.entry_time).total_seconds() / 60),
                    'news_trigger': trade.news_context.headline[:50] + "..."
                }
                for trade_id, trade in self.active_trades.items()
            ]
        }
"""
Decision Analysis Engine
Real-time analysis of trading decisions and their quality
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DecisionPattern:
    """Pattern in decision making"""
    pattern_id: str
    description: str
    conditions: Dict[str, Any]
    success_rate: float
    avg_return: float
    sample_size: int
    confidence_level: float


@dataclass
class DecisionInsight:
    """Insight from decision analysis"""
    insight_type: str  # "strength", "weakness", "opportunity", "threat"
    title: str
    description: str
    evidence: List[str]
    recommendation: str
    priority: int  # 1-5, 5 being highest
    created_at: datetime = field(default_factory=datetime.now)


class DecisionAnalyzer:
    """
    Analyzes trading decisions in real-time to identify patterns,
    strengths, weaknesses, and opportunities for improvement
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get('data_dir', '/app/data'))
        
        # Analysis windows
        self.short_window = config.get('short_analysis_window', 24)  # hours
        self.medium_window = config.get('medium_analysis_window', 168)  # 1 week
        self.long_window = config.get('long_analysis_window', 720)  # 1 month
        
        # Decision tracking
        self.decision_history: List[Dict[str, Any]] = []
        self.patterns: List[DecisionPattern] = []
        self.insights: List[DecisionInsight] = []
        
        # Real-time decision metrics
        self.current_session_stats = {
            'decisions_made': 0,
            'high_confidence_decisions': 0,
            'news_triggered_decisions': 0,
            'breaking_news_decisions': 0,
            'avg_confidence': 0.0,
            'decision_distribution': {
                'buy': 0,
                'sell': 0,
                'hold': 0
            }
        }
    
    def analyze_decision(
        self,
        decision_data: Dict[str, Any],
        news_context: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a trading decision in real-time
        Returns decision quality assessment and recommendations
        """
        
        analysis_result = {
            'decision_id': decision_data.get('id', f"decision_{datetime.now().timestamp()}"),
            'timestamp': datetime.now().isoformat(),
            'quality_score': 0.0,
            'risk_score': 0.0,
            'opportunity_score': 0.0,
            'recommendations': [],
            'warnings': [],
            'confidence_assessment': '',
            'similar_decisions': []
        }
        
        # Assess decision quality
        quality_scores = self._assess_decision_quality(decision_data, news_context, market_context)
        analysis_result['quality_score'] = quality_scores['overall']
        analysis_result['quality_breakdown'] = quality_scores
        
        # Assess risk level
        risk_assessment = self._assess_decision_risk(decision_data, market_context)
        analysis_result['risk_score'] = risk_assessment['score']
        analysis_result['risk_factors'] = risk_assessment['factors']
        
        # Identify opportunity potential
        opportunity_assessment = self._assess_opportunity_potential(decision_data, news_context)
        analysis_result['opportunity_score'] = opportunity_assessment['score']
        analysis_result['opportunity_factors'] = opportunity_assessment['factors']
        
        # Generate recommendations
        analysis_result['recommendations'] = self._generate_decision_recommendations(
            decision_data, quality_scores, risk_assessment, opportunity_assessment
        )
        
        # Check for warnings
        analysis_result['warnings'] = self._check_decision_warnings(
            decision_data, news_context, market_context
        )
        
        # Assess confidence appropriateness
        analysis_result['confidence_assessment'] = self._assess_confidence_level(
            decision_data, quality_scores, risk_assessment
        )
        
        # Find similar historical decisions
        analysis_result['similar_decisions'] = self._find_similar_decisions(decision_data, news_context)
        
        # Update session stats
        self._update_session_stats(decision_data)
        
        # Store decision for pattern analysis
        self.decision_history.append({
            'decision_data': decision_data,
            'news_context': news_context,
            'market_context': market_context,
            'analysis_result': analysis_result,
            'timestamp': datetime.now()
        })
        
        return analysis_result
    
    def _assess_decision_quality(
        self, 
        decision_data: Dict[str, Any], 
        news_context: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess the quality of a trading decision"""
        
        scores = {}
        
        # Sentiment alignment (25%)
        sentiment_score = news_context.get('sentiment_score', 0)
        decision_direction = decision_data.get('direction', 'hold')
        
        if decision_direction == 'buy' and sentiment_score > 0.1:
            scores['sentiment_alignment'] = min(1.0, sentiment_score * 2)
        elif decision_direction == 'sell' and sentiment_score < -0.1:
            scores['sentiment_alignment'] = min(1.0, abs(sentiment_score) * 2)
        elif decision_direction == 'hold':
            scores['sentiment_alignment'] = 1.0 - abs(sentiment_score)
        else:
            scores['sentiment_alignment'] = 0.2  # Poor alignment
        
        # News quality (20%)
        impact_score = news_context.get('impact_score', 0)
        source_reliability = self._get_source_reliability(news_context.get('source', ''))
        news_freshness = self._calculate_news_freshness(news_context.get('timestamp'))
        
        scores['news_quality'] = (
            impact_score * 0.5 + 
            source_reliability * 0.3 + 
            news_freshness * 0.2
        )
        
        # Confidence appropriateness (20%)
        confidence = decision_data.get('confidence', 0)
        expected_confidence = self._calculate_expected_confidence(news_context, market_context)
        confidence_diff = abs(confidence - expected_confidence)
        scores['confidence_appropriateness'] = max(0.0, 1.0 - confidence_diff * 2)
        
        # Risk management (20%)
        has_stop_loss = decision_data.get('stop_loss') is not None
        has_take_profit = decision_data.get('take_profit') is not None
        position_size = decision_data.get('position_size', 0)
        
        risk_mgmt_score = 0.0
        if has_stop_loss:
            risk_mgmt_score += 0.4
        if has_take_profit:
            risk_mgmt_score += 0.3
        if 0.01 <= position_size <= 0.05:  # Reasonable position size
            risk_mgmt_score += 0.3
        
        scores['risk_management'] = risk_mgmt_score
        
        # Timing quality (15%)
        market_hours = market_context.get('market_open', False)
        volatility = market_context.get('volatility', 0.5)
        volume = market_context.get('volume', 0.5)
        
        timing_score = 0.7  # Base score
        if market_hours:
            timing_score += 0.2
        if 0.3 <= volatility <= 0.7:  # Good volatility range
            timing_score += 0.1
        
        scores['timing_quality'] = min(1.0, timing_score)
        
        # Calculate overall score
        weights = {
            'sentiment_alignment': 0.25,
            'news_quality': 0.20,
            'confidence_appropriateness': 0.20,
            'risk_management': 0.20,
            'timing_quality': 0.15
        }
        
        scores['overall'] = sum(scores[key] * weights[key] for key in weights.keys())
        
        return scores
    
    def _assess_decision_risk(self, decision_data: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level of the decision"""
        
        risk_factors = []
        risk_score = 0.0
        
        # Position size risk
        position_size = decision_data.get('position_size', 0)
        if position_size > 0.05:  # More than 5%
            risk_factors.append(f"Large position size: {position_size:.1%}")
            risk_score += 0.3
        elif position_size > 0.1:  # More than 10%
            risk_factors.append(f"Very large position size: {position_size:.1%}")
            risk_score += 0.5
        
        # Confidence risk
        confidence = decision_data.get('confidence', 0)
        if confidence < 0.3:
            risk_factors.append(f"Low confidence decision: {confidence:.1%}")
            risk_score += 0.2
        
        # Market volatility risk
        volatility = market_context.get('volatility', 0.5)
        if volatility > 0.8:
            risk_factors.append("High market volatility")
            risk_score += 0.2
        
        # Missing risk controls
        if not decision_data.get('stop_loss'):
            risk_factors.append("No stop-loss protection")
            risk_score += 0.3
        
        # Symbol-specific risk
        symbol = decision_data.get('symbol', '')
        if symbol in ['TSLA', 'GME', 'AMC']:  # High volatility stocks
            risk_factors.append(f"High volatility symbol: {symbol}")
            risk_score += 0.1
        
        return {
            'score': min(1.0, risk_score),
            'factors': risk_factors,
            'level': self._categorize_risk_level(risk_score)
        }
    
    def _assess_opportunity_potential(self, decision_data: Dict[str, Any], news_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess opportunity potential of the decision"""
        
        opportunity_factors = []
        opportunity_score = 0.0
        
        # Breaking news opportunity
        if news_context.get('category') == 'breaking':
            opportunity_factors.append("Breaking news catalyst")
            opportunity_score += 0.3
        
        # High impact news
        impact_score = news_context.get('impact_score', 0)
        if impact_score > 0.8:
            opportunity_factors.append(f"High impact news (score: {impact_score:.1f})")
            opportunity_score += 0.2
        
        # Strong sentiment
        sentiment_score = abs(news_context.get('sentiment_score', 0))
        if sentiment_score > 0.7:
            opportunity_factors.append(f"Strong sentiment signal ({sentiment_score:.1f})")
            opportunity_score += 0.2
        
        # Multiple confirming signals
        decision_factors = decision_data.get('decision_factors', {})
        strong_factors = sum(1 for score in decision_factors.values() if score > 0.7)
        if strong_factors >= 3:
            opportunity_factors.append(f"Multiple strong signals ({strong_factors})")
            opportunity_score += 0.2
        
        # Earnings/catalyst events
        if any(keyword in news_context.get('headline', '').lower() 
               for keyword in ['earnings', 'beats', 'guidance', 'merger']):
            opportunity_factors.append("Earnings/catalyst event")
            opportunity_score += 0.1
        
        return {
            'score': min(1.0, opportunity_score),
            'factors': opportunity_factors,
            'level': self._categorize_opportunity_level(opportunity_score)
        }
    
    def _generate_decision_recommendations(
        self, 
        decision_data: Dict[str, Any], 
        quality_scores: Dict[str, float], 
        risk_assessment: Dict[str, Any], 
        opportunity_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations for the decision"""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_scores.get('sentiment_alignment', 0) < 0.5:
            recommendations.append("Consider waiting for stronger sentiment alignment")
        
        if quality_scores.get('news_quality', 0) < 0.4:
            recommendations.append("Verify news from additional reliable sources")
        
        if quality_scores.get('risk_management', 0) < 0.6:
            if not decision_data.get('stop_loss'):
                recommendations.append("Add stop-loss protection")
            if not decision_data.get('take_profit'):
                recommendations.append("Set take-profit target")
        
        # Risk-based recommendations
        if risk_assessment['score'] > 0.7:
            recommendations.append("Consider reducing position size due to high risk")
        
        if risk_assessment['score'] > 0.5 and decision_data.get('confidence', 0) < 0.6:
            recommendations.append("Avoid trade due to high risk and low confidence")
        
        # Opportunity-based recommendations
        if opportunity_assessment['score'] > 0.8:
            recommendations.append("Strong opportunity - consider increasing conviction")
        
        if opportunity_assessment['score'] < 0.3:
            recommendations.append("Limited opportunity - consider passing on this trade")
        
        # Position sizing recommendations
        position_size = decision_data.get('position_size', 0)
        confidence = decision_data.get('confidence', 0)
        
        optimal_size = min(0.05, confidence * 0.1)  # Max 5%, scaled by confidence
        if abs(position_size - optimal_size) > 0.02:
            recommendations.append(f"Consider adjusting position size to {optimal_size:.1%}")
        
        return recommendations
    
    def _check_decision_warnings(
        self, 
        decision_data: Dict[str, Any], 
        news_context: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> List[str]:
        """Check for warning conditions"""
        
        warnings = []
        
        # High risk warnings
        position_size = decision_data.get('position_size', 0)
        if position_size > 0.1:
            warnings.append("⚠️ VERY LARGE POSITION SIZE - Consider reducing")
        
        # Low quality warnings
        sentiment_score = abs(news_context.get('sentiment_score', 0))
        if sentiment_score < 0.2 and decision_data.get('confidence', 0) > 0.7:
            warnings.append("⚠️ High confidence with weak sentiment - Double check analysis")
        
        # Market condition warnings
        if not market_context.get('market_open', True):
            warnings.append("⚠️ Markets closed - Trade will execute at next open")
        
        volatility = market_context.get('volatility', 0.5)
        if volatility > 0.9:
            warnings.append("⚠️ Extremely high volatility - Expect large price swings")
        
        # News quality warnings
        source_reliability = self._get_source_reliability(news_context.get('source', ''))
        if source_reliability < 0.5:
            warnings.append("⚠️ Unreliable news source - Verify information")
        
        news_age = self._calculate_news_freshness(news_context.get('timestamp'))
        if news_age < 0.3:  # Very old news
            warnings.append("⚠️ Stale news - Market may have already reacted")
        
        return warnings
    
    def _assess_confidence_level(
        self, 
        decision_data: Dict[str, Any], 
        quality_scores: Dict[str, float], 
        risk_assessment: Dict[str, Any]
    ) -> str:
        """Assess if confidence level is appropriate"""
        
        confidence = decision_data.get('confidence', 0)
        quality = quality_scores.get('overall', 0)
        risk = risk_assessment['score']
        
        # Expected confidence range based on quality and risk
        expected_confidence = quality * (1 - risk * 0.5)
        
        if abs(confidence - expected_confidence) < 0.2:
            return "✅ Appropriate confidence level"
        elif confidence > expected_confidence + 0.2:
            return "⚠️ Overconfident - Consider reducing confidence"
        else:
            return "📈 Conservative - Could increase confidence"
    
    def _find_similar_decisions(self, decision_data: Dict[str, Any], news_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar historical decisions"""
        
        similar_decisions = []
        current_symbol = decision_data.get('symbol', '')
        current_direction = decision_data.get('direction', '')
        current_sentiment = news_context.get('sentiment_score', 0)
        
        for historical in self.decision_history[-50:]:  # Last 50 decisions
            hist_decision = historical['decision_data']
            hist_news = historical['news_context']
            
            # Check similarity criteria
            symbol_match = hist_decision.get('symbol') == current_symbol
            direction_match = hist_decision.get('direction') == current_direction
            sentiment_similar = abs(hist_news.get('sentiment_score', 0) - current_sentiment) < 0.3
            
            if sum([symbol_match, direction_match, sentiment_similar]) >= 2:
                similar_decisions.append({
                    'timestamp': historical['timestamp'].isoformat(),
                    'symbol': hist_decision.get('symbol'),
                    'direction': hist_decision.get('direction'),
                    'confidence': hist_decision.get('confidence'),
                    'quality_score': historical['analysis_result'].get('quality_score'),
                    'outcome': 'Unknown'  # Would need to track actual outcomes
                })
        
        return similar_decisions[-5:]  # Return last 5 similar decisions
    
    def _update_session_stats(self, decision_data: Dict[str, Any]):
        """Update current session statistics"""
        
        self.current_session_stats['decisions_made'] += 1
        
        confidence = decision_data.get('confidence', 0)
        if confidence > 0.7:
            self.current_session_stats['high_confidence_decisions'] += 1
        
        direction = decision_data.get('direction', 'hold')
        self.current_session_stats['decision_distribution'][direction] += 1
        
        # Update average confidence
        total_decisions = self.current_session_stats['decisions_made']
        current_avg = self.current_session_stats['avg_confidence']
        self.current_session_stats['avg_confidence'] = (
            (current_avg * (total_decisions - 1) + confidence) / total_decisions
        )
    
    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for news source"""
        
        source_scores = {
            'Reuters': 0.9,
            'Bloomberg': 0.9,
            'Wall Street Journal': 0.9,
            'MarketWatch': 0.8,
            'Yahoo Finance': 0.7,
            'CNBC': 0.8,
            'Seeking Alpha': 0.6,
        }
        
        return source_scores.get(source, 0.5)  # Default to medium reliability
    
    def _calculate_news_freshness(self, news_timestamp) -> float:
        """Calculate how fresh the news is (1.0 = very fresh, 0.0 = very old)"""
        
        if not news_timestamp:
            return 0.5
        
        if isinstance(news_timestamp, str):
            news_time = datetime.fromisoformat(news_timestamp.replace('Z', '+00:00'))
        else:
            news_time = news_timestamp
        
        age_minutes = (datetime.now() - news_time.replace(tzinfo=None)).total_seconds() / 60
        
        if age_minutes < 15:  # Very fresh
            return 1.0
        elif age_minutes < 60:  # Fresh
            return 0.8
        elif age_minutes < 240:  # Moderately fresh (4 hours)
            return 0.6
        elif age_minutes < 1440:  # Day old
            return 0.4
        else:  # Old news
            return 0.2
    
    def _calculate_expected_confidence(self, news_context: Dict[str, Any], market_context: Dict[str, Any]) -> float:
        """Calculate what confidence level should be based on context"""
        
        base_confidence = 0.5
        
        # Adjust based on sentiment strength
        sentiment_strength = abs(news_context.get('sentiment_score', 0))
        base_confidence += sentiment_strength * 0.3
        
        # Adjust based on news impact
        impact_score = news_context.get('impact_score', 0)
        base_confidence += impact_score * 0.2
        
        # Adjust based on market conditions
        if market_context.get('market_open', False):
            base_confidence += 0.1
        
        volatility = market_context.get('volatility', 0.5)
        if volatility > 0.8:  # High volatility reduces confidence
            base_confidence -= 0.2
        
        return min(1.0, max(0.1, base_confidence))
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        elif risk_score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _categorize_opportunity_level(self, opportunity_score: float) -> str:
        """Categorize opportunity level"""
        if opportunity_score < 0.2:
            return "Limited"
        elif opportunity_score < 0.5:
            return "Moderate"
        elif opportunity_score < 0.8:
            return "Strong"
        else:
            return "Exceptional"
    
    def generate_session_insights(self) -> List[DecisionInsight]:
        """Generate insights from current session"""
        
        insights = []
        stats = self.current_session_stats
        
        # Decision frequency insight
        if stats['decisions_made'] > 10:
            insights.append(DecisionInsight(
                insight_type="opportunity",
                title="High Decision Volume",
                description=f"Generated {stats['decisions_made']} decisions this session",
                evidence=[f"Average confidence: {stats['avg_confidence']:.1%}"],
                recommendation="Monitor decision quality to avoid overtrading",
                priority=3
            ))
        
        # Confidence pattern insight
        if stats['avg_confidence'] < 0.4:
            insights.append(DecisionInsight(
                insight_type="weakness",
                title="Low Average Confidence",
                description="Decisions showing low confidence levels",
                evidence=[f"Average confidence: {stats['avg_confidence']:.1%}"],
                recommendation="Wait for stronger signals or reduce position sizes",
                priority=4
            ))
        
        # Direction bias insight
        total_directional = stats['decision_distribution']['buy'] + stats['decision_distribution']['sell']
        if total_directional > 0:
            buy_pct = stats['decision_distribution']['buy'] / total_directional
            if buy_pct > 0.8:
                insights.append(DecisionInsight(
                    insight_type="threat",
                    title="Strong Bullish Bias",
                    description="Most decisions are bullish",
                    evidence=[f"{buy_pct:.1%} of decisions are buy orders"],
                    recommendation="Check for confirmation bias, consider bearish opportunities",
                    priority=3
                ))
        
        return insights
    
    def get_decision_analytics(self) -> Dict[str, Any]:
        """Get comprehensive decision analytics for dashboard"""
        
        return {
            'session_stats': self.current_session_stats,
            'recent_insights': self.generate_session_insights(),
            'decision_patterns': self._analyze_decision_patterns(),
            'quality_trends': self._analyze_quality_trends(),
            'risk_distribution': self._analyze_risk_distribution()
        }
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in recent decisions"""
        
        recent_decisions = self.decision_history[-20:] if self.decision_history else []
        
        if not recent_decisions:
            return {}
        
        # Analyze patterns
        patterns = {
            'high_quality_decisions': 0,
            'risky_decisions': 0,
            'overconfident_decisions': 0,
            'common_symbols': {},
            'time_distribution': {}
        }
        
        for decision in recent_decisions:
            analysis = decision['analysis_result']
            decision_data = decision['decision_data']
            
            if analysis['quality_score'] > 0.8:
                patterns['high_quality_decisions'] += 1
            
            if analysis['risk_score'] > 0.7:
                patterns['risky_decisions'] += 1
            
            confidence = decision_data.get('confidence', 0)
            quality = analysis['quality_score']
            if confidence > quality + 0.3:
                patterns['overconfident_decisions'] += 1
            
            # Track symbol frequency
            symbol = decision_data.get('symbol', 'Unknown')
            patterns['common_symbols'][symbol] = patterns['common_symbols'].get(symbol, 0) + 1
        
        return patterns
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze trends in decision quality"""
        
        recent_decisions = self.decision_history[-10:] if self.decision_history else []
        
        if not recent_decisions:
            return {}
        
        quality_scores = [d['analysis_result']['quality_score'] for d in recent_decisions]
        
        return {
            'current_quality': quality_scores[-1] if quality_scores else 0,
            'avg_quality': sum(quality_scores) / len(quality_scores),
            'quality_trend': self._calculate_trend(quality_scores),
            'improvement_needed': sum(1 for score in quality_scores if score < 0.6)
        }
    
    def _analyze_risk_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of risk levels"""
        
        recent_decisions = self.decision_history[-20:] if self.decision_history else []
        
        if not recent_decisions:
            return {}
        
        risk_levels = {'Low': 0, 'Medium': 0, 'High': 0, 'Very High': 0}
        
        for decision in recent_decisions:
            risk_assessment = decision['analysis_result'].get('risk_factors', {})
            risk_level = self._categorize_risk_level(decision['analysis_result']['risk_score'])
            risk_levels[risk_level] += 1
        
        total = sum(risk_levels.values())
        risk_distribution = {
            level: (count / total * 100) if total > 0 else 0
            for level, count in risk_levels.items()
        }
        
        return {
            'distribution': risk_distribution,
            'high_risk_percentage': risk_distribution['High'] + risk_distribution['Very High']
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "Insufficient data"
        
        recent_avg = sum(values[-3:]) / 3
        earlier_avg = sum(values[:-3]) / (len(values) - 3) if len(values) > 3 else values[0]
        
        if recent_avg > earlier_avg + 0.1:
            return "Improving"
        elif recent_avg < earlier_avg - 0.1:
            return "Declining"
        else:
            return "Stable"
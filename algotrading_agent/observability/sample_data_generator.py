"""
Sample Data Generator for Trading Performance Dashboard
Generates realistic sample trade data for demonstration and testing
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random
import json
from pathlib import Path

from .trade_performance_tracker import (
    TradePerformanceTracker, NewsContext, TradeDecision, 
    TradeExecution, TradeResult, FailureReason
)
from .decision_analyzer import DecisionAnalyzer
from .correlation_tracker import CorrelationTracker

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generates realistic sample data for testing the trading dashboard"""
    
    def __init__(self, data_dir: str = '/app/data'):
        self.data_dir = Path(data_dir)
        
        # Sample data pools
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'SPY', 'QQQ', 'NVDA', 'GM']
        self.news_sources = ['Reuters', 'Bloomberg', 'MarketWatch', 'Yahoo Finance', 'CNBC', 'Seeking Alpha']
        
        self.sample_headlines = [
            "{symbol} beats Q4 earnings expectations",
            "{symbol} announces major partnership deal",
            "{symbol} shares surge on breakthrough product announcement",
            "{symbol} misses revenue estimates in latest quarter",
            "{symbol} faces regulatory scrutiny over market practices",
            "{symbol} CEO announces surprise resignation",
            "{symbol} reports record quarterly revenue growth",
            "{symbol} stock downgraded by major analyst firm",
            "{symbol} launches innovative AI-powered service",
            "{symbol} disappoints with weak guidance for next quarter"
        ]
        
        # Market scenarios with different success rates
        self.scenarios = [
            {'type': 'bullish_market', 'win_rate': 0.75, 'avg_return': 2.5},
            {'type': 'bearish_market', 'win_rate': 0.35, 'avg_return': -1.8},
            {'type': 'volatile_market', 'win_rate': 0.55, 'avg_return': 0.8},
            {'type': 'stable_market', 'win_rate': 0.65, 'avg_return': 1.2}
        ]
    
    def generate_sample_trades(
        self, 
        num_trades: int = 50,
        days_back: int = 30,
        tracker: TradePerformanceTracker = None
    ) -> List[Dict[str, Any]]:
        """Generate sample completed trades"""
        
        if not tracker:
            tracker = TradePerformanceTracker({'data_dir': str(self.data_dir)})
        
        trades_generated = []
        
        for i in range(num_trades):
            # Generate trade timestamp (random within last N days)
            days_ago = random.uniform(0, days_back)
            trade_time = datetime.now() - timedelta(days=days_ago)
            
            # Choose random symbol and scenario
            symbol = random.choice(self.symbols)
            scenario = random.choice(self.scenarios)
            
            # Generate news context
            headline_template = random.choice(self.sample_headlines)
            headline = headline_template.format(symbol=symbol)
            
            news_context = NewsContext(
                headline=headline,
                source=random.choice(self.news_sources),
                sentiment_score=self._generate_sentiment_score(headline),
                impact_score=random.uniform(0.3, 1.0),
                timestamp=trade_time - timedelta(minutes=random.randint(5, 60)),
                symbols_mentioned=[symbol],
                category=random.choice(['earnings', 'general', 'breaking', 'analyst'])
            )
            
            # Generate decision
            direction = self._determine_direction(news_context.sentiment_score, scenario)
            confidence = self._generate_confidence(news_context, scenario)
            
            decision = TradeDecision(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                position_size=random.uniform(0.01, 0.05),
                stop_loss=None if random.random() > 0.7 else random.uniform(95, 99),
                take_profit=None if random.random() > 0.8 else random.uniform(101, 110),
                decision_factors={
                    'sentiment': news_context.sentiment_score,
                    'impact': news_context.impact_score,
                    'recency': random.uniform(0.5, 1.0)
                },
                timestamp=trade_time
            )
            
            # Generate execution
            entry_price = random.uniform(50, 500)  # Mock price range
            execution = TradeExecution(
                order_id=f"order_{i:04d}_{symbol}",
                entry_price=entry_price,
                entry_time=trade_time + timedelta(minutes=random.randint(1, 5)),
                fees=random.uniform(0, 2.0),
                slippage=random.uniform(-0.5, 0.5)
            )
            
            # Determine outcome based on scenario
            will_win = random.random() < scenario['win_rate']
            duration_minutes = random.randint(15, 480)  # 15 minutes to 8 hours
            
            # Calculate exit price and complete trade
            if will_win:
                price_change_pct = random.uniform(0.5, abs(scenario['avg_return']) * 2)
                if direction == 'sell':  # Short position
                    price_change_pct = -price_change_pct
            else:
                price_change_pct = random.uniform(-abs(scenario['avg_return']) * 2, -0.5)
                if direction == 'sell':  # Short position
                    price_change_pct = -price_change_pct
            
            exit_price = entry_price * (1 + price_change_pct / 100)
            exit_time = execution.entry_time + timedelta(minutes=duration_minutes)
            
            # Complete the trade
            completed_trade = tracker.start_trade(
                trade_id=f"trade_{i:04d}_{symbol}_{trade_time.strftime('%Y%m%d')}",
                news_context=news_context,
                decision=decision,
                execution=execution
            )
            
            # Complete it immediately with calculated outcome
            final_trade = tracker.complete_trade(
                completed_trade.trade_id,
                exit_price=exit_price,
                exit_time=exit_time,
                fees=random.uniform(0, 2.0)
            )
            
            if final_trade:
                trades_generated.append({
                    'trade_id': final_trade.trade_id,
                    'symbol': symbol,
                    'result': final_trade.result.value,
                    'pnl': final_trade.pnl_absolute,
                    'pnl_pct': final_trade.pnl_percentage,
                    'duration': final_trade.duration_minutes,
                    'scenario': scenario['type']
                })
        
        logger.info(f"Generated {len(trades_generated)} sample trades")
        return trades_generated
    
    def generate_sample_decisions(
        self,
        num_decisions: int = 30,
        analyzer: DecisionAnalyzer = None
    ) -> List[Dict[str, Any]]:
        """Generate sample decision analysis data"""
        
        if not analyzer:
            analyzer = DecisionAnalyzer({'data_dir': str(self.data_dir)})
        
        decisions_generated = []
        
        for i in range(num_decisions):
            symbol = random.choice(self.symbols)
            scenario = random.choice(self.scenarios)
            
            # Generate decision data
            sentiment_score = random.uniform(-1.0, 1.0)
            direction = 'buy' if sentiment_score > 0.1 else 'sell' if sentiment_score < -0.1 else 'hold'
            
            decision_data = {
                'id': f"decision_{i:04d}",
                'symbol': symbol,
                'direction': direction,
                'confidence': random.uniform(0.2, 0.9),
                'position_size': random.uniform(0.01, 0.06),
                'stop_loss': random.uniform(95, 99) if random.random() > 0.3 else None,
                'take_profit': random.uniform(101, 115) if random.random() > 0.4 else None,
                'decision_factors': {
                    'sentiment': sentiment_score,
                    'impact': random.uniform(0.3, 1.0),
                    'recency': random.uniform(0.4, 1.0)
                }
            }
            
            # Generate news context
            news_context = {
                'headline': random.choice(self.sample_headlines).format(symbol=symbol),
                'source': random.choice(self.news_sources),
                'sentiment_score': sentiment_score,
                'impact_score': decision_data['decision_factors']['impact'],
                'timestamp': (datetime.now() - timedelta(minutes=random.randint(5, 300))).isoformat(),
                'category': random.choice(['earnings', 'general', 'breaking'])
            }
            
            # Generate market context
            market_context = {
                'market_open': random.choice([True, False]),
                'volatility': random.uniform(0.2, 0.9),
                'volume': random.uniform(0.3, 1.0)
            }
            
            # Analyze the decision
            analysis_result = analyzer.analyze_decision(decision_data, news_context, market_context)
            
            decisions_generated.append({
                'decision_id': decision_data['id'],
                'symbol': symbol,
                'quality_score': analysis_result['quality_score'],
                'risk_score': analysis_result['risk_score'],
                'confidence': decision_data['confidence'],
                'direction': direction
            })
        
        logger.info(f"Generated {len(decisions_generated)} sample decisions")
        return decisions_generated
    
    def generate_sample_correlations(
        self,
        num_correlations: int = 40,
        tracker: CorrelationTracker = None
    ) -> List[Dict[str, Any]]:
        """Generate sample correlation test data"""
        
        if not tracker:
            tracker = CorrelationTracker({'data_dir': str(self.data_dir)})
        
        correlations_generated = []
        
        # Generate realistic correlation scenarios
        scenarios = [
            {'sentiment_range': (0.3, 0.8), 'direction': 'up', 'success_rate': 0.75},
            {'sentiment_range': (-0.8, -0.3), 'direction': 'down', 'success_rate': 0.70},
            {'sentiment_range': (-0.2, 0.2), 'direction': 'neutral', 'success_rate': 0.60},
            {'sentiment_range': (0.1, 0.3), 'direction': 'up', 'success_rate': 0.55},  # Weak signals
            {'sentiment_range': (-0.3, -0.1), 'direction': 'down', 'success_rate': 0.52}
        ]
        
        for i in range(num_correlations):
            symbol = random.choice(self.symbols)
            scenario = random.choice(scenarios)
            
            # Generate sentiment in specified range
            sentiment = random.uniform(scenario['sentiment_range'][0], scenario['sentiment_range'][1])
            predicted_direction = scenario['direction']
            
            # Determine actual outcome based on success rate
            if random.random() < scenario['success_rate']:
                # Successful prediction
                if predicted_direction == 'up':
                    actual_change = random.uniform(1.0, 8.0)
                elif predicted_direction == 'down':
                    actual_change = random.uniform(-8.0, -1.0)
                else:  # neutral
                    actual_change = random.uniform(-1.5, 1.5)
            else:
                # Failed prediction
                if predicted_direction == 'up':
                    actual_change = random.uniform(-6.0, -0.5)
                elif predicted_direction == 'down':
                    actual_change = random.uniform(0.5, 6.0)
                else:  # neutral
                    actual_change = random.uniform(-5.0, 5.0)
                    if abs(actual_change) < 2.0:  # Make it a more significant move
                        actual_change = actual_change * 3
            
            # Add to correlation tracker
            tracker.add_test_result(
                symbol=symbol,
                sentiment_score=sentiment,
                predicted_direction=predicted_direction,
                actual_price_change=actual_change,
                confidence_score=abs(sentiment),
                test_type='generated_sample'
            )
            
            correlations_generated.append({
                'symbol': symbol,
                'sentiment': sentiment,
                'predicted': predicted_direction,
                'actual_change': actual_change,
                'correct': abs(actual_change) > 2.0 and (
                    (predicted_direction == 'up' and actual_change > 0) or
                    (predicted_direction == 'down' and actual_change < 0) or
                    (predicted_direction == 'neutral' and abs(actual_change) <= 2.0)
                )
            })
        
        logger.info(f"Generated {len(correlations_generated)} sample correlations")
        return correlations_generated
    
    def _generate_sentiment_score(self, headline: str) -> float:
        """Generate realistic sentiment score based on headline content"""
        
        positive_words = ['beats', 'surge', 'breakthrough', 'record', 'growth', 'partnership']
        negative_words = ['misses', 'disappoints', 'scrutiny', 'resignation', 'downgraded']
        
        sentiment = 0.0
        headline_lower = headline.lower()
        
        for word in positive_words:
            if word in headline_lower:
                sentiment += random.uniform(0.3, 0.7)
        
        for word in negative_words:
            if word in headline_lower:
                sentiment -= random.uniform(0.3, 0.7)
        
        # Add some random noise
        sentiment += random.uniform(-0.2, 0.2)
        
        # Clamp to valid range
        return max(-1.0, min(1.0, sentiment))
    
    def _determine_direction(self, sentiment: float, scenario: Dict[str, Any]) -> str:
        """Determine trade direction based on sentiment and market scenario"""
        
        if abs(sentiment) < 0.1:
            return 'hold'
        elif sentiment > 0:
            return 'buy'
        else:
            return 'sell'
    
    def _generate_confidence(self, news_context: NewsContext, scenario: Dict[str, Any]) -> float:
        """Generate realistic confidence score"""
        
        base_confidence = abs(news_context.sentiment_score) * 0.6
        impact_boost = news_context.impact_score * 0.3
        scenario_modifier = random.uniform(0.8, 1.2)  # Market scenario affects confidence
        
        confidence = (base_confidence + impact_boost) * scenario_modifier
        
        # Add some randomness and clamp
        confidence += random.uniform(-0.1, 0.1)
        return max(0.1, min(0.95, confidence))
    
    def generate_full_sample_dataset(self, output_summary: bool = True) -> Dict[str, Any]:
        """Generate a complete sample dataset for all components"""
        
        logger.info("Generating comprehensive sample dataset...")
        
        # Initialize trackers
        trade_tracker = TradePerformanceTracker({'data_dir': str(self.data_dir)})
        decision_analyzer = DecisionAnalyzer({'data_dir': str(self.data_dir)})
        correlation_tracker = CorrelationTracker({'data_dir': str(self.data_dir)})
        
        # Generate data
        trades = self.generate_sample_trades(num_trades=60, tracker=trade_tracker)
        decisions = self.generate_sample_decisions(num_decisions=40, analyzer=decision_analyzer)
        correlations = self.generate_sample_correlations(num_correlations=50, tracker=correlation_tracker)
        
        # Create summary
        summary = {
            'trades_generated': len(trades),
            'decisions_generated': len(decisions),
            'correlations_generated': len(correlations),
            'trade_results': {
                'wins': sum(1 for t in trades if t['result'] == 'win'),
                'losses': sum(1 for t in trades if t['result'] == 'loss'),
                'total_pnl': sum(t['pnl'] for t in trades)
            },
            'decision_quality': {
                'avg_quality': sum(d['quality_score'] for d in decisions) / len(decisions),
                'high_quality': sum(1 for d in decisions if d['quality_score'] > 0.7)
            },
            'correlation_accuracy': {
                'correct_predictions': sum(1 for c in correlations if c['correct']),
                'accuracy_rate': sum(1 for c in correlations if c['correct']) / len(correlations) * 100
            }
        }
        
        if output_summary:
            logger.info("Sample dataset generation complete:")
            logger.info(f"  📊 Generated {summary['trades_generated']} trades")
            logger.info(f"  📈 Win rate: {summary['trade_results']['wins']/(summary['trade_results']['wins'] + summary['trade_results']['losses']) * 100:.1f}%")
            logger.info(f"  💰 Total P&L: ${summary['trade_results']['total_pnl']:.2f}")
            logger.info(f"  🧠 Average decision quality: {summary['decision_quality']['avg_quality']:.1%}")
            logger.info(f"  🎯 Correlation accuracy: {summary['correlation_accuracy']['accuracy_rate']:.1f}%")
        
        return {
            'summary': summary,
            'components': {
                'trade_tracker': trade_tracker,
                'decision_analyzer': decision_analyzer,
                'correlation_tracker': correlation_tracker
            }
        }
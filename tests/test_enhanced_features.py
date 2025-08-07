#!/usr/bin/env python3
"""
Test script for enhanced NewsImpactScorer and DecisionEngine features
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.components.news_impact_scorer import NewsImpactScorer
from algotrading_agent.components.decision_engine import DecisionEngine
from algotrading_agent.config.settings import get_config

def create_test_news_item(title, content, sentiment_score=0.0, age_hours=1, source="Reuters"):
    """Create a test news item with the specified parameters"""
    published_time = datetime.utcnow() - timedelta(hours=age_hours)
    
    return {
        "title": title,
        "content": content,
        "source": source,
        "published": published_time,
        "timestamp": published_time.isoformat(),
        "sentiment": {
            "polarity": sentiment_score,
            "confidence": abs(sentiment_score) * 0.8 + 0.2  # Confidence based on sentiment strength
        },
        "entities": {
            "tickers": ["AAPL"],  # Default to Apple for testing
            "companies": ["Apple Inc"]
        },
        "impact_score": 0.7,  # Default moderate impact
        "filter_score": 0.6   # Default filter score
    }

def test_temporal_dynamics():
    """Test temporal dynamics calculation"""
    print("\n🕐 Testing Temporal Dynamics...")
    
    config = get_config()
    scorer = NewsImpactScorer(config.get('news_impact_scorer', {}))
    
    # Test cases with different timing scenarios
    test_cases = [
        {
            "name": "Breaking Flash News",
            "item": create_test_news_item(
                "BREAKING: Apple announces revolutionary breakthrough in AI technology",
                "Apple Inc. just announced a major breakthrough that could revolutionize the AI industry.",
                sentiment_score=0.8,
                age_hours=0.25,  # 15 minutes old
                source="Reuters"
            )
        },
        {
            "name": "Recent Positive News", 
            "item": create_test_news_item(
                "Apple beats earnings estimates with record revenue",
                "Apple reported strong quarterly results, beating analyst estimates by 15%.",
                sentiment_score=0.6,
                age_hours=2,
                source="Bloomberg"
            )
        },
        {
            "name": "Stale Negative News",
            "item": create_test_news_item(
                "Apple faces regulatory challenges in Europe",
                "European regulators continue to investigate Apple's market practices.",
                sentiment_score=-0.4,
                age_hours=30,  # Over a day old
                source="CNBC"
            )
        }
    ]
    
    for test_case in test_cases:
        item = test_case["item"]
        sentiment_strength = abs(item["sentiment"]["polarity"])
        
        temporal_result = scorer.calculate_temporal_dynamics(item, sentiment_strength)
        
        print(f"\n  📊 {test_case['name']}:")
        print(f"    Age: {temporal_result['age_hours']:.1f} hours")
        print(f"    Hype Window: {temporal_result['hype_window']['type']}")
        print(f"    Peak Detection: {temporal_result['peak_detection']['window']}")
        print(f"    Decay Pattern: {temporal_result['decay_info']['pattern']}")
        print(f"    Temporal Multiplier: {temporal_result['temporal_multiplier']:.3f}")

def test_strength_correlation():
    """Test strength correlation calculation"""
    print("\n💪 Testing Strength Correlation...")
    
    config = get_config()
    scorer = NewsImpactScorer(config.get('news_impact_scorer', {}))
    
    test_cases = [
        {
            "name": "High Sentiment + High Hype",
            "sentiment": 0.85,
            "hype": 0.9,
            "symbols": ["AAPL", "TSLA"]
        },
        {
            "name": "Moderate Sentiment + Low Hype",
            "sentiment": 0.4,
            "hype": 0.2,
            "symbols": ["SPY"]
        },
        {
            "name": "Strong Negative Sentiment",
            "sentiment": -0.75,
            "hype": 0.6,
            "symbols": ["NVDA"]
        }
    ]
    
    for test_case in test_cases:
        result = scorer.calculate_strength_correlation(
            test_case["sentiment"], 
            test_case["hype"], 
            test_case["symbols"]
        )
        
        print(f"\n  📈 {test_case['name']}:")
        print(f"    Expected Velocity: {result['velocity_mapping']['expected_velocity']:.1f}")
        print(f"    Volume Multiplier: {result['volume_correlation']['volume_multiplier']:.1f}x")
        print(f"    Volatility Level: {result['volatility_patterns']['volatility_level']}")
        print(f"    Overall Strength Score: {result['overall_strength_score']:.3f}")

def test_market_context():
    """Test market context calculation"""
    print("\n🏢 Testing Market Context...")
    
    config = get_config()
    scorer = NewsImpactScorer(config.get('news_impact_scorer', {}))
    
    test_cases = [
        {
            "name": "Tech Sector Bull News",
            "item": create_test_news_item(
                "Tech stocks surge as AI breakthrough drives massive rally",
                "Technology companies are experiencing unprecedented growth driven by artificial intelligence innovations.",
                sentiment_score=0.7
            ),
            "symbols": ["AAPL", "MSFT", "GOOGL"]
        },
        {
            "name": "Energy Sector Bear News",
            "item": create_test_news_item(
                "Oil prices plunge as recession fears mount",
                "Energy stocks are under pressure as crude oil prices drop amid economic uncertainty.",
                sentiment_score=-0.6
            ),
            "symbols": ["XOM", "CVX"]
        },
        {
            "name": "General Market News",
            "item": create_test_news_item(
                "Federal Reserve holds interest rates steady",
                "The Fed decided to maintain current interest rate levels in today's meeting.",
                sentiment_score=0.1
            ),
            "symbols": ["SPY", "QQQ"]
        }
    ]
    
    for test_case in test_cases:
        result = scorer.calculate_market_context(
            test_case["item"],
            test_case["symbols"]
        )
        
        print(f"\n  🎯 {test_case['name']}:")
        print(f"    Detected Sectors: {result['sector_analysis']['detected_sectors']}")
        print(f"    Market Regime: {result['market_regime']['regime']}")
        print(f"    Time Impact: {result['time_impact']['time_period']} ({result['time_impact']['multiplier']:.1f}x)")
        print(f"    Context Score: {result['context_score']:.3f}")
        print(f"    Context Multiplier: {result['context_multiplier']:.3f}")

def test_decision_engine_integration():
    """Test DecisionEngine integration with enhanced features"""
    print("\n🧠 Testing Decision Engine Integration...")
    
    config = get_config()
    scorer = NewsImpactScorer(config.get('news_impact_scorer', {}))
    decision_engine = DecisionEngine(config.get('decision_engine', {}))
    
    # Inject the scorer into the decision engine
    decision_engine.news_impact_scorer = scorer
    
    # Create test news items
    test_news = [
        create_test_news_item(
            "BREAKING: Apple announces record-breaking AI chip breakthrough",
            "Apple's new AI chip delivers unprecedented performance gains, beating all competitors by 300%.",
            sentiment_score=0.9,
            age_hours=0.5,  # Very fresh breaking news
            source="Reuters"
        ),
        create_test_news_item(
            "Apple stock surges on strong quarterly results",
            "AAPL shares jumped 8% in after-hours trading following stellar earnings report.",
            sentiment_score=0.7,
            age_hours=1,
            source="Bloomberg"
        )
    ]
    
    # Test signal strength calculation with enhanced features
    print(f"  📡 Testing Enhanced Signal Strength Calculation...")
    signal_strength = decision_engine._calculate_signal_strength(test_news)
    print(f"    Enhanced Signal Strength: {signal_strength:.4f}")
    
    # Test confidence calculation with enhanced features
    confidence = decision_engine._calculate_confidence(test_news, signal_strength)
    print(f"    Enhanced Confidence: {confidence:.4f}")
    
    # Test enhanced target adjustments
    enhanced_adjustments = decision_engine._get_enhanced_target_adjustments(test_news, signal_strength)
    if enhanced_adjustments:
        print(f"    Take Profit Multiplier: {enhanced_adjustments['take_profit_multiplier']:.2f}")
        print(f"    Stop Loss Multiplier: {enhanced_adjustments['stop_loss_multiplier']:.2f}")
        print(f"    Reasoning: {enhanced_adjustments.get('reasoning', 'None')}")
    else:
        print(f"    No enhanced adjustments calculated")

def main():
    """Run all tests"""
    print("🚀 Starting Enhanced Features Test Suite")
    print("=" * 60)
    
    try:
        test_temporal_dynamics()
        test_strength_correlation()
        test_market_context()
        test_decision_engine_integration()
        
        print("\n" + "=" * 60)
        print("✅ All Enhanced Features Tests Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Test AI Integration with Sample News Data
"""

import asyncio
import logging
import sys
import os

# Add project to path
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.config.settings import get_config
from algotrading_agent.components.ai_analyzer import AIAnalyzer
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain

# Sample news data for testing
SAMPLE_NEWS = [
    {
        "title": "Apple Inc. beats Q3 earnings expectations with record iPhone sales",
        "content": "Apple Inc. reported strong Q3 earnings that exceeded analyst expectations, driven by record iPhone sales and robust services revenue. The company posted earnings per share of $1.52 vs expected $1.39.",
        "symbol": "AAPL",
        "timestamp": "2024-08-05T10:00:00Z",
        "source": "Reuters"
    },
    {
        "title": "Tesla stock plunges on disappointing delivery numbers",
        "content": "Tesla shares dropped 8% in pre-market trading after the company reported lower-than-expected vehicle deliveries for Q3. Analysts express concerns about increasing competition in the EV market.",
        "symbol": "TSLA", 
        "timestamp": "2024-08-05T09:30:00Z",
        "source": "MarketWatch"
    },
    {
        "title": "Microsoft announces breakthrough in quantum computing research",
        "content": "Microsoft revealed significant advances in quantum computing technology that could revolutionize cloud computing and AI applications. The breakthrough involves new error-correction techniques.",
        "symbol": "MSFT",
        "timestamp": "2024-08-05T11:15:00Z",
        "source": "Bloomberg"
    }
]

async def test_ai_integration():
    """Test the AI integration with sample news data"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing AI Integration")
    print("=" * 50)
    
    try:
        # Load configuration
        config = get_config()
        print(f"✅ Configuration loaded")
        
        # Test 1: Traditional Analysis Only
        print("\n📊 Test 1: Traditional Analysis (TextBlob)")
        print("-" * 40)
        
        # Disable AI for baseline test
        config_no_ai = {'ai_analyzer': {'enabled': False}}
        
        brain_traditional = NewsAnalysisBrain(config_no_ai)
        start_result = brain_traditional.start()
        if start_result:  # If it returns a coroutine, await it
            await start_result
        
        process_result = brain_traditional.process(SAMPLE_NEWS.copy())
        if hasattr(process_result, '__await__'):  # Check if it's awaitable
            traditional_results = await process_result
        else:
            traditional_results = process_result
        
        for i, item in enumerate(traditional_results, 1):
            sentiment = item.get('sentiment', {})
            print(f"  News {i}: {item['title'][:50]}...")
            print(f"    Traditional Sentiment: {sentiment.get('polarity', 0):.3f} ({sentiment.get('label', 'unknown')})")
            print(f"    Confidence: {sentiment.get('confidence', 0):.3f}")
            print()
        
        stop_result = brain_traditional.stop()
        if stop_result:  # If it returns a coroutine, await it
            await stop_result
        
        # Test 2: AI-Enhanced Analysis (Fallback Mode)
        print("\n🤖 Test 2: AI-Enhanced Analysis (Fallback Mode)")
        print("-" * 50)
        
        # Enable AI but don't provide API key (will use fallback)
        config_ai_fallback = {
            'ai_analyzer': {
                'enabled': True,
                'provider': 'openai',
                'api_key': '',  # Empty key triggers fallback
                'fallback_enabled': True
            }
        }
        
        brain_ai_fallback = NewsAnalysisBrain(config_ai_fallback)
        await brain_ai_fallback.start()
        
        ai_fallback_results = await brain_ai_fallback.process(SAMPLE_NEWS.copy())
        
        for i, item in enumerate(ai_fallback_results, 1):
            print(f"  News {i}: {item['title'][:50]}...")
            print(f"    AI Provider: {item.get('ai_provider', 'N/A')}")
            print(f"    AI Sentiment: {item.get('ai_sentiment', 0):.3f}")
            print(f"    AI Confidence: {item.get('ai_confidence', 0):.3f}")
            print(f"    Traditional Sentiment: {item.get('traditional_sentiment', 0):.3f}")
            print(f"    Combined Sentiment: {item.get('combined_sentiment', 0):.3f}")
            print(f"    Analysis Method: {item.get('analysis_method', 'unknown')}")
            print()
            
        await brain_ai_fallback.stop()
        
        # Test 3: Mock AI Analysis
        print("\n🎯 Test 3: Direct AI Analyzer Test")
        print("-" * 40)
        
        ai_analyzer = AIAnalyzer(config_ai_fallback)
        await ai_analyzer.start()
        
        # This will use fallback analysis since no real API key
        ai_direct_results = await ai_analyzer.analyze_news_batch(SAMPLE_NEWS.copy())
        
        for i, item in enumerate(ai_direct_results, 1):
            print(f"  News {i}: {item['title'][:50]}...")
            print(f"    Market Sentiment: {item.get('market_sentiment', 0):.3f}")
            print(f"    Volatility Prediction: {item.get('volatility_prediction', 0):.3f}")
            print(f"    Time Horizon: {item.get('time_horizon', 'unknown')}")
            trading_signals = item.get('trading_signals', {})
            print(f"    Trading Action: {trading_signals.get('action', 'unknown')}")
            print(f"    Action Strength: {trading_signals.get('strength', 0):.3f}")
            print(f"    Risk Factors: {len(item.get('risk_factors', []))}")
            print()
            
        await ai_analyzer.stop()
        
        # Test 4: Configuration Validation
        print("\n⚙️  Test 4: Configuration Validation")
        print("-" * 40)
        
        ai_config = config.get('ai_analyzer', {})
        print(f"  AI Enabled: {ai_config.get('enabled', False)}")
        print(f"  Provider: {ai_config.get('provider', 'none')}")
        print(f"  Model: {ai_config.get('model', 'none')}")
        print(f"  Timeout: {ai_config.get('timeout', 0)}s")
        print(f"  Max Retries: {ai_config.get('max_retries', 0)}")
        print(f"  Fallback Enabled: {ai_config.get('fallback_enabled', False)}")
        
        print(f"\n✅ All tests completed successfully!")
        print("\n🚀 Ready to enable AI analysis:")
        print("   1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("   2. Update config/default.yml: ai_analyzer.enabled = true")
        print("   3. Restart the trading system")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ai_integration())
    sys.exit(0 if success else 1)
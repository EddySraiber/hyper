#!/usr/bin/env python3
"""
Test OpenAI AI Integration
"""

import asyncio
import logging
import sys
import os

# Add project to path
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.components.ai_analyzer import AIAnalyzer

# Sample news for testing
SAMPLE_NEWS = [
    {
        "title": "Apple Inc. beats Q3 earnings expectations with record iPhone sales",
        "content": "Apple Inc. reported strong Q3 earnings that exceeded analyst expectations, driven by record iPhone sales and robust services revenue. The company posted earnings per share of $1.52 vs expected $1.39.",
        "symbol": "AAPL",
        "timestamp": "2024-08-05T10:00:00Z"
    },
    {
        "title": "Tesla stock plunges on disappointing delivery numbers",
        "content": "Tesla shares dropped 8% in pre-market trading after the company reported lower-than-expected vehicle deliveries for Q3. Analysts express concerns about increasing competition in the EV market.",
        "symbol": "TSLA", 
        "timestamp": "2024-08-05T09:30:00Z"
    }
]

async def test_openai_integration():
    """Test the OpenAI AI integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 Testing OpenAI AI Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    print(f"✅ Found OpenAI API key: {api_key[:8]}...")
    
    try:
        # Test configuration
        config = {
            'ai_analyzer': {
                'enabled': True,
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'api_key': '',  # Will use env var
                'timeout': 30,
                'fallback_enabled': True
            }
        }
        
        print("\n🤖 Initializing OpenAI AI Analyzer")
        ai_analyzer = AIAnalyzer(config)
        await ai_analyzer.start()
        
        print(f"   Provider: {ai_analyzer.provider}")
        print(f"   Model: {ai_analyzer.model}")
        print(f"   API Key: {ai_analyzer.api_key[:8]}...")
        
        print("\n📊 Analyzing sample news with OpenAI...")
        results = await ai_analyzer.analyze_news_batch(SAMPLE_NEWS.copy())
        
        for i, item in enumerate(results, 1):
            print(f"\n📰 News {i}: {item['title'][:60]}...")
            print(f"   🤖 AI Provider: {item.get('ai_provider', 'N/A')}")
            print(f"   💭 Market Sentiment: {item.get('market_sentiment', 0):.3f}")
            print(f"   🎯 Confidence: {item.get('confidence_score', 0):.3f}")
            print(f"   📈 Volatility Prediction: {item.get('volatility_prediction', 0):.3f}")
            print(f"   ⏰ Time Horizon: {item.get('time_horizon', 'unknown')}")
            print(f"   🌍 Market Impact: {item.get('market_impact', 'unknown')}")
            
            trading_signals = item.get('trading_signals', {})
            print(f"   📊 Trading Action: {trading_signals.get('action', 'unknown')}")
            print(f"   💪 Action Strength: {trading_signals.get('strength', 0):.3f}")
            
            if trading_signals.get('entry_price_target'):
                print(f"   🎯 Entry Target: ${trading_signals.get('entry_price_target')}")
            if trading_signals.get('stop_loss_suggestion'):
                print(f"   🛑 Stop Loss: ${trading_signals.get('stop_loss_suggestion')}")
            if trading_signals.get('take_profit_suggestion'):
                print(f"   💰 Take Profit: ${trading_signals.get('take_profit_suggestion')}")
            
            risk_factors = item.get('risk_factors', [])
            print(f"   ⚠️  Risk Factors: {len(risk_factors)}")
            if risk_factors and len(risk_factors) > 0:
                for risk in risk_factors[:2]:  # Show first 2 risks
                    print(f"      - {risk}")
            
            insights = item.get('key_insights', [])
            print(f"   💡 Key Insights: {len(insights)}")
            if insights and len(insights) > 0:
                for insight in insights[:2]:  # Show first 2 insights
                    print(f"      - {insight}")
        
        await ai_analyzer.stop()
        
        print(f"\n✅ OpenAI integration test successful!")
        print("\n🚀 To enable in trading system:")
        print("   1. Update config/default.yml: ai_analyzer.enabled = true")
        print("   2. Restart with: docker-compose restart algotrading-agent")
        print("\n📈 Expected improvements:")
        print("   - More accurate sentiment analysis")
        print("   - Better volatility predictions") 
        print("   - Specific entry/exit price targets")
        print("   - Risk factor identification")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openai_integration())
    sys.exit(0 if success else 1)
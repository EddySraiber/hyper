#!/usr/bin/env python3
"""
Test Groq AI Integration
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
        "content": "Apple Inc. reported strong Q3 earnings that exceeded analyst expectations, driven by record iPhone sales and robust services revenue.",
        "symbol": "AAPL",
        "timestamp": "2024-08-05T10:00:00Z"
    }
]

async def test_groq_integration():
    """Test the Groq AI integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🦾 Testing Groq AI Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("❌ GROQ_API_KEY environment variable not set!")
        print("   Get your free API key from: https://console.groq.com/")
        print("   Then run: export GROQ_API_KEY='your-key-here'")
        return False
    
    print(f"✅ Found Groq API key: {api_key[:8]}...")
    
    try:
        # Test configuration
        config = {
            'ai_analyzer': {
                'enabled': True,
                'provider': 'groq',
                'model': 'llama3-8b-8192',
                'api_key': '',  # Will use env var
                'timeout': 30,
                'fallback_enabled': True
            }
        }
        
        print("\n🤖 Initializing Groq AI Analyzer")
        ai_analyzer = AIAnalyzer(config)
        await ai_analyzer.start()
        
        print(f"   Provider: {ai_analyzer.provider}")
        print(f"   Model: {ai_analyzer.model}")
        print(f"   API Key: {ai_analyzer.api_key[:8]}...")
        
        print("\n📊 Analyzing sample news with Groq...")
        results = await ai_analyzer.analyze_news_batch(SAMPLE_NEWS.copy())
        
        for i, item in enumerate(results, 1):
            print(f"\n📰 News {i}: {item['title'][:60]}...")
            print(f"   🤖 AI Provider: {item.get('ai_provider', 'N/A')}")
            print(f"   💭 Market Sentiment: {item.get('market_sentiment', 0):.3f}")
            print(f"   🎯 Confidence: {item.get('confidence_score', 0):.3f}")
            print(f"   📈 Volatility Prediction: {item.get('volatility_prediction', 0):.3f}")
            print(f"   ⏰ Time Horizon: {item.get('time_horizon', 'unknown')}")
            
            trading_signals = item.get('trading_signals', {})
            print(f"   📊 Trading Action: {trading_signals.get('action', 'unknown')}")
            print(f"   💪 Action Strength: {trading_signals.get('strength', 0):.3f}")
            
            risk_factors = item.get('risk_factors', [])
            print(f"   ⚠️  Risk Factors: {len(risk_factors)}")
            if risk_factors:
                for risk in risk_factors[:3]:  # Show first 3 risks
                    print(f"      - {risk}")
            
            insights = item.get('key_insights', [])
            print(f"   💡 Key Insights: {len(insights)}")
            if insights:
                for insight in insights[:2]:  # Show first 2 insights
                    print(f"      - {insight}")
        
        await ai_analyzer.stop()
        
        print(f"\n✅ Groq integration test successful!")
        print("\n🚀 To enable in trading system:")
        print("   1. Update config/default.yml: ai_analyzer.enabled = true")
        print("   2. Ensure GROQ_API_KEY is set in environment")
        print("   3. Restart with: docker-compose restart algotrading-agent")
        
        return True
        
    except Exception as e:
        print(f"❌ Groq test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_groq_integration())
    sys.exit(0 if success else 1)
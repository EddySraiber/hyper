#!/usr/bin/env python3
"""
Test Configurable AI System with Multiple Providers
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

async def test_configurable_ai():
    """Test the configurable AI system with different providers"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 Testing Configurable AI System")
    print("=" * 60)
    
    # Check available API keys
    available_providers = []
    if os.getenv("GROQ_API_KEY"):
        available_providers.append("groq")
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("openai")  
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")
    
    print(f"💳 Available API Keys: {', '.join(available_providers) if available_providers else 'None (will use fallback)'}")
    
    # Test configurations for different providers
    test_configs = [
        {
            "name": "Groq Primary with Fallback Chain", 
            "provider": "groq",
            "fallback_chain": ["groq", "openai", "anthropic", "traditional"]
        },
        {
            "name": "OpenAI Primary with Groq Fallback",
            "provider": "openai", 
            "fallback_chain": ["openai", "groq", "traditional"]
        },
        {
            "name": "Local AI (Mock)",
            "provider": "local",
            "fallback_chain": ["local", "traditional"]
        }
    ]
    
    for test_config in test_configs:
        print(f"\n🧪 Test: {test_config['name']}")
        print("-" * 50)
        
        # Build full configuration
        config = {
            'ai_analyzer': {
                'enabled': True,
                'provider': test_config['provider'],
                'fallback_chain': test_config['fallback_chain'],
                'providers': {
                    'openai': {
                        'enabled': True,
                        'model': 'gpt-3.5-turbo',
                        'api_key_env': 'OPENAI_API_KEY',
                        'base_url': 'https://api.openai.com/v1/chat/completions',
                        'timeout': 30,
                        'max_tokens': 800,
                        'temperature': 0.3
                    },
                    'groq': {
                        'enabled': True,
                        'model': 'llama3-8b-8192', 
                        'api_key_env': 'GROQ_API_KEY',
                        'base_url': 'https://api.groq.com/openai/v1/chat/completions',
                        'timeout': 30,
                        'max_tokens': 800,
                        'temperature': 0.3
                    },
                    'anthropic': {
                        'enabled': True,
                        'model': 'claude-3-sonnet-20240229',
                        'api_key_env': 'ANTHROPIC_API_KEY',
                        'base_url': 'https://api.anthropic.com/v1/messages',
                        'timeout': 30,
                        'max_tokens': 800
                    },
                    'local': {
                        'enabled': True,
                        'model': 'llama3',
                        'base_url': 'http://localhost:11434/api/generate',
                        'timeout': 60
                    }
                },
                'max_retries': 2,
                'fallback_enabled': True
            }
        }
        
        try:
            ai_analyzer = AIAnalyzer(config)
            await ai_analyzer.start()
            
            print(f"   🎯 Primary Provider: {ai_analyzer.primary_provider}")
            print(f"   🔄 Fallback Chain: {ai_analyzer.fallback_chain}")
            print(f"   ⚙️  Loaded Providers: {list(ai_analyzer.provider_configs.keys())}")
            
            # Test analysis
            results = await ai_analyzer.analyze_news_batch(SAMPLE_NEWS.copy())
            
            for item in results:
                print(f"\n   📰 News: {item['title'][:50]}...")
                print(f"      🤖 AI Provider: {item.get('ai_provider', 'N/A')}")
                print(f"      🧠 AI Model: {item.get('ai_model', 'N/A')}")
                print(f"      💭 Market Sentiment: {item.get('market_sentiment', 0):.3f}")
                print(f"      🎯 Confidence: {item.get('confidence_score', 0):.3f}")
                print(f"      📈 Volatility: {item.get('volatility_prediction', 0):.3f}")
                print(f"      📊 Trading Action: {item.get('trading_signals', {}).get('action', 'N/A')}")
                
                risk_factors = item.get('risk_factors', [])
                if risk_factors:
                    print(f"      ⚠️  Top Risk: {risk_factors[0]}")
            
            await ai_analyzer.stop()
            print(f"   ✅ Test completed successfully")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            continue
    
    print(f"\n🎯 Configuration Examples:")
    print("-" * 30)
    print("📝 To use Groq as primary:")
    print("   ai_analyzer.provider = 'groq'")
    print("   GROQ_API_KEY=your-groq-key")
    
    print("\n📝 To use OpenAI as primary:")
    print("   ai_analyzer.provider = 'openai'")
    print("   OPENAI_API_KEY=your-openai-key")
    
    print("\n📝 To customize fallback chain:")
    print("   ai_analyzer.fallback_chain = ['groq', 'openai', 'traditional']")
    
    print("\n📝 To change models:")
    print("   providers.groq.model = 'llama3-70b-8192'")
    print("   providers.openai.model = 'gpt-4'")
    
    print(f"\n✅ Configurable AI system test completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_configurable_ai())
    sys.exit(0 if success else 1)
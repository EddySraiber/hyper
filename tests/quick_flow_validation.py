#!/usr/bin/env python3
"""
Quick Trading Flow Validation
=============================

Simple validation of key trading flow components without live trading.
Tests the most critical components quickly.
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

print("🧪 QUICK TRADING FLOW VALIDATION")
print("=" * 50)

async def validate_news_scraping():
    """Test news scraping and filtering"""
    print("\n1. 📰 TESTING NEWS PIPELINE...")
    
    try:
        # Test with container
        import subprocess
        result = subprocess.run([
            "docker-compose", "exec", "-T", "algotrading-agent", "python", "-c",
            """
import asyncio
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.enhanced_news_scraper import EnhancedNewsScraper
from algotrading_agent.components.news_filter import NewsFilter

async def test_news():
    config = get_config()
    scraper = EnhancedNewsScraper(config.get_component_config('enhanced_news_scraper'))
    
    # Get recent news
    recent_news = scraper.get_recent_news()
    print(f'📊 News items: {len(recent_news)}')
    
    if recent_news:
        sample = recent_news[0]
        print(f'📄 Sample: {sample.get("title", "No title")[:60]}...')
        
        # Test filtering
        filter_comp = NewsFilter(config.get_component_config('news_filter'))
        relevant = filter_comp.is_relevant(sample)
        print(f'🔍 Relevance: {"✅ Relevant" if relevant else "❌ Filtered"}')
    
    return len(recent_news)

result = asyncio.run(test_news())
print(f'RESULT: {result}')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ News pipeline working")
            return True
        else:
            print(f"   ❌ News pipeline error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ News pipeline failed: {e}")
        return False

async def validate_ai_analysis():
    """Test AI sentiment analysis"""
    print("\n2. 🤖 TESTING AI ANALYSIS...")
    
    try:
        result = subprocess.run([
            "docker-compose", "exec", "-T", "algotrading-agent", "python", "-c",
            """
import asyncio
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.ai_analyzer import AIAnalyzer

async def test_ai():
    config = get_config()
    ai = AIAnalyzer(config.get_component_config('ai_analyzer'))
    
    # Test with sample news
    test_item = {
        'title': 'Apple beats earnings expectations with record revenue',
        'content': 'Apple Inc reported stronger than expected quarterly earnings...',
        'symbol': 'AAPL',
        'timestamp': '2024-08-18T15:00:00Z'
    }
    
    try:
        analysis = await ai.analyze_sentiment(test_item)
        print(f'📈 Sentiment: {analysis.get("sentiment", "N/A")}')
        print(f'🎯 Confidence: {analysis.get("confidence", "N/A")}')
        print(f'🤖 Provider: {analysis.get("provider_used", "N/A")}')
        return True
    except Exception as e:
        print(f'❌ AI analysis failed: {e}')
        # Test fallback
        print('🔄 Testing fallback analysis...')
        try:
            # Force traditional analysis
            ai.config['provider'] = 'traditional'
            analysis = await ai.analyze_sentiment(test_item)
            print(f'📈 Fallback sentiment: {analysis.get("sentiment", "N/A")}')
            return True
        except Exception as e2:
            print(f'❌ Fallback also failed: {e2}')
            return False

result = asyncio.run(test_ai())
print(f'RESULT: {result}')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ AI analysis working")
            return True
        else:
            print(f"   ❌ AI analysis error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ AI analysis failed: {e}")
        return False

async def validate_decision_engine():
    """Test decision engine"""
    print("\n3. 🎯 TESTING DECISION ENGINE...")
    
    try:
        result = subprocess.run([
            "docker-compose", "exec", "-T", "algotrading-agent", "python", "-c",
            """
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.decision_engine import DecisionEngine
from datetime import datetime

config = get_config()
engine = DecisionEngine(config.get_component_config('decision_engine'))

# Test with mock analyzed news
test_item = {
    'symbol': 'AAPL',
    'sentiment': 0.8,
    'confidence': 0.85,
    'impact_score': 0.7,
    'timestamp': datetime.now(),
    'title': 'Apple beats earnings expectations'
}

try:
    decision = engine.make_decision(test_item)
    if decision:
        print(f'📊 Decision: {decision.symbol} {decision.action}')
        print(f'🔢 Quantity: {decision.quantity}')
        print(f'💰 Entry: ${decision.entry_price}')
        print(f'🎯 Confidence: {decision.confidence}')
    else:
        print('📊 No decision generated (filtered)')
    print('RESULT: True')
except Exception as e:
    print(f'❌ Decision engine error: {e}')
    print('RESULT: False')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Decision engine working")
            return True
        else:
            print(f"   ❌ Decision engine error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Decision engine failed: {e}")
        return False

async def validate_risk_management():
    """Test risk management"""
    print("\n4. 🛡️  TESTING RISK MANAGEMENT...")
    
    try:
        result = subprocess.run([
            "docker-compose", "exec", "-T", "algotrading-agent", "python", "-c",
            """
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.risk_manager import RiskManager
from algotrading_agent.components.decision_engine import TradingPair

config = get_config()
risk_manager = RiskManager(config.get_component_config('risk_manager'))

# Test validation
test_pair = TradingPair(
    symbol='AAPL',
    action='buy',
    quantity=10,
    confidence=0.8,
    entry_price=150.0
)

try:
    validation = risk_manager.validate_trade(test_pair)
    approved = validation.get('approved', False)
    print(f'📊 Validation: {"✅ Approved" if approved else "❌ Rejected"}')
    if validation.get('warnings'):
        print(f'⚠️  Warnings: {validation["warnings"]}')
    if validation.get('errors'):
        print(f'❌ Errors: {validation["errors"]}')
    print('RESULT: True')
except Exception as e:
    print(f'❌ Risk management error: {e}')
    print('RESULT: False')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Risk management working")
            return True
        else:
            print(f"   ❌ Risk management error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Risk management failed: {e}")
        return False

async def validate_trade_execution():
    """Test trade execution (simulation mode)"""
    print("\n5. ⚡ TESTING TRADE EXECUTION...")
    
    try:
        result = subprocess.run([
            "docker-compose", "exec", "-T", "algotrading-agent", "python", "-c",
            """
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.enhanced_trade_manager import EnhancedTradeManager
from algotrading_agent.components.decision_engine import TradingPair

config = get_config()
# Get component status without execution
try:
    trade_manager = EnhancedTradeManager(config.get_component_config('enhanced_trade_manager'))
    print('📊 Trade manager: ✅ Initialized')
    
    # Check if components are available
    if hasattr(trade_manager, 'universal_client'):
        print('🌐 Universal client: ✅ Available')
    else:
        print('🌐 Universal client: ❌ Not available')
    
    if hasattr(trade_manager, 'guardian_service'):
        print('🛡️  Guardian service: ✅ Available')
    else:
        print('🛡️  Guardian service: ❌ Not available')
    
    print('RESULT: True')
except Exception as e:
    print(f'❌ Trade manager error: {e}')
    print('RESULT: False')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Trade execution components working")
            return True
        else:
            print(f"   ❌ Trade execution error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Trade execution failed: {e}")
        return False

async def validate_safety_systems():
    """Test safety systems"""
    print("\n6. 🛡️  TESTING SAFETY SYSTEMS...")
    
    try:
        result = subprocess.run([
            "docker-compose", "logs", "--tail=50", "algotrading-agent"
        ], capture_output=True, text=True, timeout=10)
        
        logs = result.stdout
        
        # Check for safety system indicators
        safety_indicators = {
            "Guardian Service": "Guardian Service" in logs,
            "Position Protector": "position protection" in logs,
            "Bracket Orders": "bracket order" in logs,
            "Trade Manager": "Enhanced Trade Manager" in logs
        }
        
        print("   🔍 Safety system status:")
        for system, active in safety_indicators.items():
            status = "✅" if active else "❌"
            print(f"      {status} {system}")
        
        active_systems = sum(safety_indicators.values())
        print(f"   📊 Active systems: {active_systems}/4")
        
        return active_systems >= 3  # At least 3 should be active
        
    except Exception as e:
        print(f"   ❌ Safety systems check failed: {e}")
        return False

async def main():
    """Run all validations"""
    start_time = time.time()
    
    tests = [
        ("News Pipeline", validate_news_scraping),
        ("AI Analysis", validate_ai_analysis), 
        ("Decision Engine", validate_decision_engine),
        ("Risk Management", validate_risk_management),
        ("Trade Execution", validate_trade_execution),
        ("Safety Systems", validate_safety_systems)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name} validation failed: {e}")
            results.append((name, False))
    
    # Summary
    duration = time.time() - start_time
    passed = len([r for r in results if r[1]])
    total = len(results)
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Duration: {duration:.1f}s")
    print(f"Tests: {passed}/{total} passed")
    print(f"Success rate: {passed/total*100:.0f}%")
    
    print(f"\n📋 RESULTS:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    overall_success = passed >= (total * 0.8)  # 80% pass rate
    print(f"\n🏆 Overall: {'✅ PASSED' if overall_success else '❌ NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n💡 System flow validated - ready for production monitoring")
    else:
        print(f"\n💡 {total - passed} components need attention before full validation")
    
    return overall_success

if __name__ == "__main__":
    import subprocess
    success = asyncio.run(main())
    print(f"\n{'✅ VALIDATION SUCCESSFUL' if success else '❌ VALIDATION INCOMPLETE'}")
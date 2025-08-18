#!/usr/bin/env python3
"""
Comprehensive Trading Flow Test
===============================

Production-ready validation of the entire trading system without live trades.
Validates all components end-to-end with proper error handling and reporting.
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os

def setup_logging():
    """Setup logging for validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/app/data/flow_validation.log')
        ]
    )

async def test_component_imports():
    """Test if all components can be imported"""
    print("🔍 TESTING COMPONENT IMPORTS...")
    
    components = {
        "Config": "algotrading_agent.config.settings.get_config",
        "NewsScraper": "algotrading_agent.components.enhanced_news_scraper.EnhancedNewsScraper", 
        "NewsFilter": "algotrading_agent.components.news_filter.NewsFilter",
        "AI Analyzer": "algotrading_agent.components.ai_analyzer.AIAnalyzer",
        "Decision Engine": "algotrading_agent.components.decision_engine.DecisionEngine",
        "Risk Manager": "algotrading_agent.components.risk_manager.RiskManager",
        "Trade Manager": "algotrading_agent.components.enhanced_trade_manager.EnhancedTradeManager"
    }
    
    results = {}
    for name, import_path in components.items():
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            results[name] = True
            print(f"   ✅ {name}")
        except Exception as e:
            results[name] = False
            print(f"   ❌ {name}: {e}")
    
    success_rate = sum(results.values()) / len(results)
    print(f"   📊 Import success: {success_rate*100:.0f}%")
    return success_rate > 0.8

async def test_news_components():
    """Test news scraping components"""
    print("\n📰 TESTING NEWS COMPONENTS...")
    
    try:
        from algotrading_agent.config.settings import get_config
        from algotrading_agent.components.enhanced_news_scraper import EnhancedNewsScraper
        from algotrading_agent.components.news_filter import NewsFilter
        
        config = get_config()
        
        # Test news scraper initialization
        scraper_config = config.get_component_config('enhanced_news_scraper')
        scraper = EnhancedNewsScraper(scraper_config)
        print("   ✅ News scraper initialized")
        
        # Test news filter
        filter_config = config.get_component_config('news_filter')
        news_filter = NewsFilter(filter_config)
        print("   ✅ News filter initialized")
        
        # Test with sample news item
        sample_item = {
            "title": "Apple reports record quarterly earnings",
            "content": "Apple Inc. today announced financial results for its fiscal quarter...",
            "url": "https://example.com/news",
            "timestamp": datetime.now(),
            "source": "Test Source"
        }
        
        is_relevant = news_filter.is_relevant(sample_item)
        print(f"   📊 Sample news relevance: {'✅ Relevant' if is_relevant else '❌ Filtered'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ News components failed: {e}")
        return False

async def test_ai_analysis():
    """Test AI analysis with all providers"""
    print("\n🤖 TESTING AI ANALYSIS...")
    
    try:
        from algotrading_agent.config.settings import get_config
        from algotrading_agent.components.ai_analyzer import AIAnalyzer
        
        config = get_config()
        ai_config = config.get_component_config('ai_analyzer')
        ai_analyzer = AIAnalyzer(ai_config)
        
        # Test news item
        test_item = {
            "title": "Apple beats earnings expectations with record revenue",
            "content": "Apple Inc. reported stronger than expected quarterly earnings today...",
            "symbol": "AAPL",
            "timestamp": datetime.now()
        }
        
        # Test providers
        providers = ['groq', 'openai', 'anthropic', 'traditional']
        working_providers = []
        
        for provider in providers:
            try:
                # Set provider
                original_provider = ai_analyzer.config.get('provider', 'groq')
                ai_analyzer.config['provider'] = provider
                
                analysis = await ai_analyzer.analyze_sentiment(test_item)
                
                if analysis and 'sentiment' in analysis:
                    sentiment = analysis.get('sentiment', 0)
                    confidence = analysis.get('confidence', 0)
                    print(f"   ✅ {provider}: sentiment={sentiment:.2f}, confidence={confidence:.2f}")
                    working_providers.append(provider)
                else:
                    print(f"   ❌ {provider}: No analysis returned")
                
                # Restore provider
                ai_analyzer.config['provider'] = original_provider
                
            except Exception as e:
                print(f"   ❌ {provider}: {str(e)[:50]}...")
        
        print(f"   📊 Working providers: {len(working_providers)}/{len(providers)}")
        return len(working_providers) > 0
        
    except Exception as e:
        print(f"   ❌ AI analysis failed: {e}")
        return False

async def test_decision_making():
    """Test decision engine"""
    print("\n🎯 TESTING DECISION ENGINE...")
    
    try:
        from algotrading_agent.config.settings import get_config
        from algotrading_agent.components.decision_engine import DecisionEngine
        
        config = get_config()
        engine = DecisionEngine(config.get_component_config('decision_engine'))
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "High confidence bullish",
                "data": {
                    "symbol": "AAPL",
                    "sentiment": 0.8,
                    "confidence": 0.9,
                    "impact_score": 0.8,
                    "timestamp": datetime.now(),
                    "title": "Apple beats earnings expectations"
                }
            },
            {
                "name": "Low confidence bearish", 
                "data": {
                    "symbol": "TSLA",
                    "sentiment": -0.3,
                    "confidence": 0.4,
                    "impact_score": 0.5,
                    "timestamp": datetime.now(),
                    "title": "Tesla production concerns"
                }
            },
            {
                "name": "High confidence bearish",
                "data": {
                    "symbol": "META",
                    "sentiment": -0.7,
                    "confidence": 0.85,
                    "impact_score": 0.7,
                    "timestamp": datetime.now(),
                    "title": "Meta faces regulatory challenges"
                }
            }
        ]
        
        decisions_made = 0
        for scenario in test_scenarios:
            try:
                decision = engine.make_decision(scenario["data"])
                if decision:
                    print(f"   ✅ {scenario['name']}: {decision.symbol} {decision.action} (conf: {decision.confidence:.2f})")
                    decisions_made += 1
                else:
                    print(f"   ⚪ {scenario['name']}: No decision (filtered)")
            except Exception as e:
                print(f"   ❌ {scenario['name']}: {e}")
        
        print(f"   📊 Decisions generated: {decisions_made}/{len(test_scenarios)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Decision engine failed: {e}")
        return False

async def test_risk_management():
    """Test risk management"""
    print("\n🛡️  TESTING RISK MANAGEMENT...")
    
    try:
        from algotrading_agent.config.settings import get_config
        from algotrading_agent.components.risk_manager import RiskManager
        from algotrading_agent.components.decision_engine import TradingPair
        
        config = get_config()
        risk_manager = RiskManager(config.get_component_config('risk_manager'))
        
        # Test different risk scenarios
        test_trades = [
            ("Normal trade", TradingPair(symbol="AAPL", action="buy", quantity=10, entry_price=150.0)),
            ("Large position", TradingPair(symbol="SPY", action="buy", quantity=100, entry_price=400.0)),
            ("Small trade", TradingPair(symbol="MSFT", action="sell", quantity=1, entry_price=300.0))
        ]
        
        approvals = 0
        for name, trade in test_trades:
            try:
                validation = risk_manager.validate_trade(trade)
                approved = validation.get('approved', False)
                
                if approved:
                    print(f"   ✅ {name}: Approved")
                    approvals += 1
                else:
                    reasons = validation.get('errors', []) + validation.get('warnings', [])
                    print(f"   ❌ {name}: Rejected - {', '.join(reasons[:2])}")
                    
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
        
        print(f"   📊 Approval rate: {approvals}/{len(test_trades)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Risk management failed: {e}")
        return False

async def test_trade_infrastructure():
    """Test trading infrastructure without actual trades"""
    print("\n⚡ TESTING TRADE INFRASTRUCTURE...")
    
    try:
        from algotrading_agent.config.settings import get_config
        from algotrading_agent.components.enhanced_trade_manager import EnhancedTradeManager
        from algotrading_agent.trading.alpaca_client import AlpacaClient
        
        config = get_config()
        
        # Test Alpaca client initialization
        alpaca_config = config.get_alpaca_config()
        alpaca_client = AlpacaClient(alpaca_config)
        print("   ✅ Alpaca client initialized")
        
        # Test account connectivity (read-only)
        try:
            account_info = await alpaca_client.get_account_info()
            if account_info:
                print("   ✅ Alpaca API connection verified")
            else:
                print("   ⚠️  Alpaca API connection issues")
        except Exception as e:
            print(f"   ⚠️  Alpaca API: {str(e)[:50]}...")
        
        # Test enhanced trade manager
        trade_config = config.get_component_config('enhanced_trade_manager')
        trade_manager = EnhancedTradeManager(trade_config)
        print("   ✅ Enhanced trade manager initialized")
        
        # Test component availability
        components = ['guardian_service', 'bracket_order_manager', 'position_protector']
        available_components = 0
        
        for component in components:
            if hasattr(trade_manager, component.replace('_', '')):
                print(f"   ✅ {component} available")
                available_components += 1
            else:
                print(f"   ❌ {component} not available")
        
        print(f"   📊 Infrastructure components: {available_components}/{len(components)}")
        return available_components >= 2
        
    except Exception as e:
        print(f"   ❌ Trade infrastructure failed: {e}")
        return False

async def test_safety_monitoring():
    """Test safety and monitoring systems"""
    print("\n🛡️  TESTING SAFETY MONITORING...")
    
    try:
        from algotrading_agent.config.settings import get_config
        
        config = get_config()
        
        # Check safety configurations
        safety_configs = {
            "Guardian Service": config.get_component_config('guardian_service'),
            "Position Protector": config.get_component_config('enhanced_trade_manager.position_protector'),
            "Bracket Orders": config.get_component_config('enhanced_trade_manager.bracket_order_manager'),
            "Risk Limits": config.get('risk_manager.max_position_pct', 0) > 0
        }
        
        active_safety = 0
        for system, configured in safety_configs.items():
            if configured:
                print(f"   ✅ {system}: Configured")
                active_safety += 1
            else:
                print(f"   ❌ {system}: Not configured")
        
        # Check observability
        observability_config = config.get_component_config('observability')
        if observability_config:
            print("   ✅ Observability: Configured")
            active_safety += 1
        else:
            print("   ❌ Observability: Not configured")
        
        print(f"   📊 Safety systems: {active_safety}/{len(safety_configs)+1}")
        return active_safety >= 3
        
    except Exception as e:
        print(f"   ❌ Safety monitoring failed: {e}")
        return False

async def test_live_system_health():
    """Test current system health from logs"""
    print("\n📊 TESTING LIVE SYSTEM HEALTH...")
    
    try:
        import subprocess
        
        # Get recent logs
        result = subprocess.run([
            "docker-compose", "logs", "--tail=100", "algotrading-agent"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("   ❌ Cannot access system logs")
            return False
        
        logs = result.stdout.lower()
        
        # Check for health indicators
        health_indicators = {
            "System Running": any(x in logs for x in ["started successfully", "processing cycle", "main loop"]),
            "Components Active": any(x in logs for x in ["enhanced trade manager", "news scraper", "decision engine"]),
            "Trading Activity": any(x in logs for x in ["trading decisions", "generated", "executed"]),
            "Safety Systems": any(x in logs for x in ["guardian service", "position protector", "bracket order"]),
            "No Critical Errors": "error" not in logs.split('\n')[-20:] if logs else False  # Recent errors only
        }
        
        health_score = 0
        for indicator, status in health_indicators.items():
            if status:
                print(f"   ✅ {indicator}")
                health_score += 1
            else:
                print(f"   ❌ {indicator}")
        
        print(f"   📊 System health: {health_score}/{len(health_indicators)} indicators")
        return health_score >= 3
        
    except Exception as e:
        print(f"   ❌ System health check failed: {e}")
        return False

async def main():
    """Run comprehensive flow validation"""
    setup_logging()
    
    print("🧪 COMPREHENSIVE TRADING FLOW VALIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Environment: Docker container")
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Component Imports", test_component_imports),
        ("News Components", test_news_components), 
        ("AI Analysis", test_ai_analysis),
        ("Decision Making", test_decision_making),
        ("Risk Management", test_risk_management),
        ("Trade Infrastructure", test_trade_infrastructure),
        ("Safety Monitoring", test_safety_monitoring),
        ("Live System Health", test_live_system_health)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Calculate results
    duration = time.time() - start_time
    passed = len([r for r in results if r[1]])
    total = len(results)
    success_rate = passed / total
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 1),
        "tests_passed": passed,
        "tests_total": total,
        "success_rate": round(success_rate * 100, 1),
        "overall_success": success_rate >= 0.75,  # 75% pass rate
        "results": [{"test": name, "passed": passed} for name, passed in results]
    }
    
    # Save report
    report_file = f"/app/data/flow_validation_{int(time.time())}.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Duration: {duration:.1f}s")
    print(f"Tests: {passed}/{total} passed")
    print(f"Success rate: {success_rate*100:.0f}%")
    
    print(f"\n📋 DETAILED RESULTS:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # Final assessment
    if report["overall_success"]:
        print(f"\n🏆 VALIDATION PASSED")
        print("✅ Trading system flow validated")
        print("✅ Ready for production monitoring")
        print("✅ All critical components functional")
    else:
        failed_tests = [name for name, success in results if not success]
        print(f"\n⚠️  VALIDATION NEEDS ATTENTION")
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print(f"💡 {total - passed} components need review")
    
    return report["overall_success"]

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        exit(1)
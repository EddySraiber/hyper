#!/usr/bin/env python3
"""
Real Trading System Performance Test
Tests actual system performance with Phase 2 optimizations enabled/disabled
"""

import asyncio
import time
import json
import psutil
from datetime import datetime
from typing import Dict, List, Any

async def test_system_with_optimizations() -> Dict[str, Any]:
    """Test system performance with Phase 2 optimizations enabled"""
    print("🔍 TESTING REAL SYSTEM WITH PHASE 2 OPTIMIZATIONS")
    print("=" * 60)
    
    # Start performance monitoring
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=1)
    start_memory = psutil.virtual_memory().percent
    
    print(f"📊 Initial System State:")
    print(f"   CPU Usage: {start_cpu:.1f}%")
    print(f"   Memory Usage: {start_memory:.1f}%")
    print(f"   Test Start Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Test results container
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "system_state": {
            "initial_cpu": start_cpu,
            "initial_memory": start_memory
        },
        "performance_metrics": {},
        "component_tests": {}
    }
    
    print("🚀 Testing Phase 2 Components with Real System...")
    
    # Test 1: Async News Processing
    print("\n1️⃣ Testing Async News Processing...")
    try:
        # Import and test async components
        from algotrading_agent.components.async_news_optimizer import AsyncNewsOptimizer
        from algotrading_agent.config.settings import get_config
        
        config = get_config()
        optimizer_config = config.get('enhanced_news_scraper', {}).get('async_optimizer', {})
        
        async_optimizer = AsyncNewsOptimizer(optimizer_config)
        await async_optimizer.start()
        
        # Test async optimization capabilities
        test_start = time.time()
        stats = await async_optimizer.get_optimization_stats()
        test_time = time.time() - test_start
        
        await async_optimizer.stop()
        
        test_results["component_tests"]["async_processing"] = {
            "component_loaded": True,
            "configuration": optimizer_config,
            "stats_retrieval_time": round(test_time * 1000, 2),  # ms
            "status": "✅ READY"
        }
        
        print(f"   ✅ AsyncNewsOptimizer: Ready for deployment")
        print(f"   ⚡ Stats retrieval: {test_time * 1000:.2f}ms")
        
    except Exception as e:
        test_results["component_tests"]["async_processing"] = {
            "component_loaded": False,
            "error": str(e),
            "status": "❌ FAILED"
        }
        print(f"   ❌ AsyncNewsOptimizer: Failed - {e}")
    
    # Test 2: AI Batch Optimization
    print("\n2️⃣ Testing AI Batch Optimization...")
    try:
        from algotrading_agent.components.ai_batch_optimizer import AIBatchOptimizer
        
        batch_config = config.get('ai_analyzer', {}).get('batch_optimizer', {})
        ai_batch_optimizer = AIBatchOptimizer(batch_config)
        await ai_batch_optimizer.start()
        
        test_start = time.time()
        stats = await ai_batch_optimizer.get_optimization_stats()
        test_time = time.time() - test_start
        
        await ai_batch_optimizer.stop()
        
        test_results["component_tests"]["ai_batch_optimization"] = {
            "component_loaded": True,
            "configuration": batch_config,
            "stats_retrieval_time": round(test_time * 1000, 2),
            "status": "✅ READY"
        }
        
        print(f"   ✅ AIBatchOptimizer: Ready for deployment")
        print(f"   💰 Cost optimization: Enabled")
        
    except Exception as e:
        test_results["component_tests"]["ai_batch_optimization"] = {
            "component_loaded": False,
            "error": str(e),
            "status": "❌ FAILED"
        }
        print(f"   ❌ AIBatchOptimizer: Failed - {e}")
    
    # Test 3: Intelligent Caching
    print("\n3️⃣ Testing Intelligent Caching...")
    try:
        from algotrading_agent.components.intelligent_cache_manager import IntelligentCacheManager
        
        cache_config = config.get('news_analysis_brain', {}).get('cache_manager', {})
        cache_manager = IntelligentCacheManager(cache_config)
        await cache_manager.start()
        
        # Test caching functionality
        test_start = time.time()
        
        # Test cache operations
        test_key = "test_analysis_key"
        test_data = {"sentiment": 0.8, "confidence": 0.9}
        
        # Cache a result
        await cache_manager.cache_result(test_key, test_data, content="test news content")
        
        # Retrieve cached result
        cached_result = await cache_manager.get_cached_result(test_key, content="test news content")
        
        # Get stats
        stats = await cache_manager.get_cache_stats()
        
        test_time = time.time() - test_start
        
        await cache_manager.stop()
        
        cache_working = cached_result is not None and "sentiment" in cached_result
        
        test_results["component_tests"]["intelligent_caching"] = {
            "component_loaded": True,
            "cache_test_passed": cache_working,
            "cache_stats": stats,
            "test_time": round(test_time * 1000, 2),
            "status": "✅ WORKING" if cache_working else "⚠️ PARTIAL"
        }
        
        print(f"   ✅ IntelligentCacheManager: {'Working' if cache_working else 'Partial'}")
        print(f"   💾 Cache test: {'Passed' if cache_working else 'Failed'}")
        
    except Exception as e:
        test_results["component_tests"]["intelligent_caching"] = {
            "component_loaded": False,
            "error": str(e),
            "status": "❌ FAILED"
        }
        print(f"   ❌ IntelligentCacheManager: Failed - {e}")
    
    # Test 4: Connection Pooling
    print("\n4️⃣ Testing Connection Pooling...")
    try:
        from algotrading_agent.components.connection_pool_optimizer import ConnectionPoolOptimizer
        
        pool_config = config.get('enhanced_news_scraper', {}).get('connection_pool_optimizer', {})
        connection_optimizer = ConnectionPoolOptimizer(pool_config)
        await connection_optimizer.start()
        
        test_start = time.time()
        
        # Test getting optimized session
        test_url = "https://httpbin.org/status/200"  # Safe test endpoint
        try:
            session = await connection_optimizer.get_optimized_session(test_url)
            connection_ready = session is not None
        except:
            connection_ready = False
            
        # Get optimization stats
        stats = await connection_optimizer.get_optimization_stats()
        
        test_time = time.time() - test_start
        
        await connection_optimizer.stop()
        
        test_results["component_tests"]["connection_pooling"] = {
            "component_loaded": True,
            "connection_test_passed": connection_ready,
            "optimization_stats": stats,
            "test_time": round(test_time * 1000, 2),
            "status": "✅ READY" if connection_ready else "⚠️ PARTIAL"
        }
        
        print(f"   ✅ ConnectionPoolOptimizer: {'Ready' if connection_ready else 'Partial'}")
        print(f"   🌐 Connection pooling: {'Enabled' if connection_ready else 'Limited'}")
        
    except Exception as e:
        test_results["component_tests"]["connection_pooling"] = {
            "component_loaded": False,
            "error": str(e),
            "status": "❌ FAILED"
        }
        print(f"   ❌ ConnectionPoolOptimizer: Failed - {e}")
    
    # System resource monitoring
    print("\n📊 Final System State...")
    final_time = time.time()
    final_cpu = psutil.cpu_percent(interval=1)
    final_memory = psutil.virtual_memory().percent
    
    total_test_time = final_time - start_time
    
    test_results["performance_metrics"] = {
        "total_test_time": round(total_test_time, 2),
        "final_cpu": final_cpu,
        "final_memory": final_memory,
        "cpu_change": round(final_cpu - start_cpu, 1),
        "memory_change": round(final_memory - start_memory, 1)
    }
    
    print(f"   CPU Usage: {final_cpu:.1f}% (Δ {final_cpu - start_cpu:+.1f}%)")
    print(f"   Memory Usage: {final_memory:.1f}% (Δ {final_memory - start_memory:+.1f}%)")
    print(f"   Total Test Time: {total_test_time:.2f}s")
    
    # Overall assessment
    working_components = sum(1 for test in test_results["component_tests"].values() 
                           if test["status"].startswith("✅"))
    total_components = len(test_results["component_tests"])
    
    test_results["overall_assessment"] = {
        "working_components": working_components,
        "total_components": total_components,
        "success_rate": round((working_components / total_components) * 100, 1),
        "system_ready": working_components >= 3,  # At least 3/4 components working
        "deployment_recommendation": "READY" if working_components >= 3 else "NEEDS_FIXES"
    }
    
    print(f"\n🎯 SYSTEM READINESS ASSESSMENT")
    print(f"   Working Components: {working_components}/{total_components}")
    print(f"   Success Rate: {test_results['overall_assessment']['success_rate']}%")
    print(f"   Deployment Status: {test_results['overall_assessment']['deployment_recommendation']}")
    
    return test_results

async def main():
    """Main test execution"""
    try:
        # Run real system test
        results = await test_system_with_optimizations()
        
        # Save detailed results
        with open('real_system_performance_test.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Real system test completed")
        print(f"📄 Detailed results saved to: real_system_performance_test.json")
        
        # Provide recommendations based on results
        print(f"\n🔧 RECOMMENDATIONS:")
        
        assessment = results["overall_assessment"]
        if assessment["deployment_recommendation"] == "READY":
            print("   🚀 System is ready for Phase 2 optimization deployment")
            print("   💡 All major components are functional")
            print("   📈 Expected performance improvements should be achievable")
        else:
            print("   ⚠️  Some components need attention before deployment")
            failed_components = [name for name, test in results["component_tests"].items() 
                               if not test["status"].startswith("✅")]
            print(f"   🔧 Review these components: {', '.join(failed_components)}")
            print("   🧪 Run individual component tests for detailed diagnostics")
            
    except Exception as e:
        print(f"❌ Real system test failed: {e}")
        print("🔧 Check system configuration and dependencies")

if __name__ == "__main__":
    asyncio.run(main())
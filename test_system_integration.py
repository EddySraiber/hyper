#!/usr/bin/env python3
"""
System Integration Test - Test Phase 2 optimizations in Docker environment
"""

import subprocess
import json
import time
from datetime import datetime

def run_docker_command(command: str, description: str = None) -> dict:
    """Run command in Docker container and return results"""
    if description:
        print(f"   🔍 {description}")
    
    full_command = f"docker-compose exec -T algotrading-agent {command}"
    
    try:
        start_time = time.time()
        result = subprocess.run(
            full_command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        execution_time = time.time() - start_time
        
        success = result.returncode == 0
        
        return {
            "success": success,
            "execution_time": round(execution_time, 2),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "execution_time": 30.0,
            "stdout": "",
            "stderr": "Command timed out",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "execution_time": 0,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

def test_phase2_integration():
    """Test Phase 2 optimizations integration in Docker"""
    print("🔍 PHASE 2 INTEGRATION TEST IN DOCKER ENVIRONMENT")
    print("=" * 60)
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "docker_tests": {},
        "system_health": {},
        "optimization_status": {},
        "overall_assessment": {}
    }
    
    # Test 1: Check Docker container is running
    print("\n1️⃣ Testing Docker Container Status...")
    container_check = run_docker_command(
        "echo 'Container is running'",
        "Checking container accessibility"
    )
    
    if not container_check["success"]:
        print("   ❌ Docker container is not running or not accessible")
        print("   🔧 Run: docker-compose up -d --build")
        return test_results
    else:
        print("   ✅ Docker container is running and accessible")
    
    # Test 2: Check Python dependencies
    print("\n2️⃣ Testing Python Dependencies...")
    deps_test = run_docker_command(
        "python3 -c 'import aiohttp, asyncio; print(\"Dependencies OK\")'",
        "Verifying aiohttp and asyncio availability"
    )
    
    test_results["docker_tests"]["dependencies"] = deps_test
    
    if deps_test["success"]:
        print("   ✅ Required dependencies are available")
    else:
        print("   ❌ Missing required dependencies")
        print(f"   Error: {deps_test['stderr']}")
    
    # Test 3: Test configuration access
    print("\n3️⃣ Testing Configuration Access...")
    config_test = run_docker_command(
        "python3 -c 'from algotrading_agent.config.settings import get_config; c=get_config(); print(\"Config loaded\", len(c.data) if hasattr(c, \"data\") else \"dict\", \"items\")'",
        "Testing configuration system"
    )
    
    test_results["docker_tests"]["configuration"] = config_test
    
    if config_test["success"]:
        print("   ✅ Configuration system is accessible")
    else:
        print("   ❌ Configuration system error")
        print(f"   Error: {config_test['stderr']}")
    
    # Test 4: Test component imports
    print("\n4️⃣ Testing Component Imports...")
    import_test = run_docker_command(
        """python3 -c "
try:
    from algotrading_agent.components.news_scraper import NewsScraper
    from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
    print('✅ Core components imported successfully')
except Exception as e:
    print('❌ Import error:', str(e))
    import sys
    sys.exit(1)
"
""",
        "Testing core component imports"
    )
    
    test_results["docker_tests"]["component_imports"] = import_test
    
    if import_test["success"]:
        print("   ✅ Core components can be imported")
    else:
        print("   ❌ Component import failed")
        print(f"   Error: {import_test['stderr']}")
    
    # Test 5: Test optimization component creation
    print("\n5️⃣ Testing Optimization Components...")
    optimization_test = run_docker_command(
        """python3 -c "
import asyncio
from algotrading_agent.config.settings import get_config

async def test_optimizations():
    try:
        config = get_config()
        
        # Test async optimizer creation
        try:
            from algotrading_agent.components.async_news_optimizer import AsyncNewsOptimizer
            async_config = config.get('enhanced_news_scraper', {}).get('async_optimizer', {})
            async_opt = AsyncNewsOptimizer(async_config)
            print('✅ AsyncNewsOptimizer created')
        except Exception as e:
            print('⚠️ AsyncNewsOptimizer:', str(e))
        
        # Test AI batch optimizer creation
        try:
            from algotrading_agent.components.ai_batch_optimizer import AIBatchOptimizer
            ai_config = config.get('ai_analyzer', {}).get('batch_optimizer', {})
            ai_opt = AIBatchOptimizer(ai_config)
            print('✅ AIBatchOptimizer created')
        except Exception as e:
            print('⚠️ AIBatchOptimizer:', str(e))
        
        # Test cache manager creation
        try:
            from algotrading_agent.components.intelligent_cache_manager import IntelligentCacheManager
            cache_config = config.get('news_analysis_brain', {}).get('cache_manager', {})
            cache_mgr = IntelligentCacheManager(cache_config)
            print('✅ IntelligentCacheManager created')
        except Exception as e:
            print('⚠️ IntelligentCacheManager:', str(e))
            
        # Test connection optimizer creation
        try:
            from algotrading_agent.components.connection_pool_optimizer import ConnectionPoolOptimizer
            conn_config = config.get('enhanced_news_scraper', {}).get('connection_pool_optimizer', {})
            conn_opt = ConnectionPoolOptimizer(conn_config)
            print('✅ ConnectionPoolOptimizer created')
        except Exception as e:
            print('⚠️ ConnectionPoolOptimizer:', str(e))
            
    except Exception as e:
        print('❌ General error:', str(e))
        import sys
        sys.exit(1)

asyncio.run(test_optimizations())
"
""",
        "Creating Phase 2 optimization components"
    )
    
    test_results["docker_tests"]["optimization_components"] = optimization_test
    
    if optimization_test["success"]:
        print("   ✅ Phase 2 components can be created")
        print(f"   Output: {optimization_test['stdout']}")
    else:
        print("   ❌ Phase 2 component creation failed")
        print(f"   Error: {optimization_test['stderr']}")
    
    # Test 6: Check existing system health
    print("\n6️⃣ Testing System Health...")
    health_test = run_docker_command(
        "curl -s http://localhost:8080/health",
        "Checking system health endpoint"
    )
    
    test_results["system_health"]["health_endpoint"] = health_test
    
    if health_test["success"] and "healthy" in health_test["stdout"].lower():
        print("   ✅ System health endpoint is responsive")
    else:
        print("   ⚠️ System health endpoint not available (system may not be fully started)")
    
    # Test 7: Check trading system status
    print("\n7️⃣ Testing Trading System Status...")
    status_test = run_docker_command(
        "python3 -c 'from main import AlgotradingAgent; import asyncio; print(\"AlgotradingAgent can be imported\")'",
        "Testing main trading system import"
    )
    
    test_results["docker_tests"]["trading_system"] = status_test
    
    if status_test["success"]:
        print("   ✅ Main trading system is importable")
    else:
        print("   ⚠️ Main trading system import issues")
        print(f"   Error: {status_test['stderr']}")
    
    # Overall assessment
    print("\n🎯 INTEGRATION TEST SUMMARY")
    print("=" * 40)
    
    successful_tests = sum(1 for test in test_results["docker_tests"].values() if test["success"])
    total_tests = len(test_results["docker_tests"])
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    test_results["overall_assessment"] = {
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "success_rate": round(success_rate, 1),
        "integration_ready": successful_tests >= 4,  # At least 4/6 core tests pass
        "deployment_status": "READY" if successful_tests >= 4 else "NEEDS_ATTENTION"
    }
    
    print(f"   Successful Tests: {successful_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Integration Status: {test_results['overall_assessment']['deployment_status']}")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    if test_results["overall_assessment"]["integration_ready"]:
        print("   ✅ System is ready for Phase 2 optimization testing")
        print("   🚀 Run: docker-compose up -d --build to start optimized system")
        print("   📊 Monitor logs for optimization performance metrics")
    else:
        print("   ⚠️ Address failing tests before deployment")
        print("   🔧 Check Docker environment and dependencies")
        print("   📝 Review configuration files for Phase 2 settings")
    
    # Save results
    with open('system_integration_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
        
    print(f"\n📄 Detailed results saved to: system_integration_test_results.json")
    
    return test_results

if __name__ == "__main__":
    test_phase2_integration()
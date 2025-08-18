#!/usr/bin/env python3
"""
Actual System Performance Test
Test the real trading system to measure current performance vs theoretical expectations
"""

import subprocess
import time
import json
import re
from datetime import datetime

def run_docker_logs_test(duration_minutes=2):
    """Monitor system performance for specified duration"""
    print(f"🔍 MONITORING SYSTEM PERFORMANCE FOR {duration_minutes} MINUTES")
    print("=" * 60)
    
    # Start monitoring
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    performance_data = {
        "test_start": datetime.fromtimestamp(start_time).isoformat(),
        "test_duration_minutes": duration_minutes,
        "news_processing": {
            "items_processed": 0,
            "sources_scraped": 0,
            "processing_times": [],
            "errors": 0
        },
        "analysis_performance": {
            "analysis_completed": 0,
            "sentiment_analyses": 0,
            "ai_analyses": 0,
            "cache_hits": 0
        },
        "system_health": {
            "component_starts": 0,
            "component_errors": 0,
            "memory_usage": [],
            "cpu_usage": []
        },
        "trading_activity": {
            "decisions_generated": 0,
            "trades_executed": 0,
            "successful_trades": 0
        }
    }
    
    print(f"📊 Starting performance monitoring...")
    print(f"   Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Monitor logs in real-time
    log_command = "docker-compose logs -f --tail=0 algotrading-agent"
    
    try:
        process = subprocess.Popen(
            log_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        print("   📄 Monitoring system logs...")
        
        while time.time() < end_time:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
                
            line = line.strip()
            current_time = time.time()
            
            # Parse different types of log entries
            if "Scraped" in line and "items from" in line:
                # News scraping activity
                match = re.search(r"Scraped (\d+) items from ([\w\s]+)", line)
                if match:
                    items = int(match.group(1))
                    performance_data["news_processing"]["items_processed"] += items
                    performance_data["news_processing"]["sources_scraped"] += 1
                    
            elif "Analyzed" in line and "news items" in line:
                # News analysis activity
                match = re.search(r"Analyzed (\d+) news items", line)
                if match:
                    items = int(match.group(1))
                    performance_data["analysis_performance"]["analysis_completed"] += items
                    
            elif "Generated" in line and "trading decisions" in line:
                # Trading decision generation
                match = re.search(r"Generated (\d+) trading decisions", line)
                if match:
                    decisions = int(match.group(1))
                    performance_data["trading_activity"]["decisions_generated"] += decisions
                    
            elif "AI-enhanced analysis" in line:
                performance_data["analysis_performance"]["ai_analyses"] += 1
                
            elif "sentiment analysis" in line and "enabled" in line:
                performance_data["analysis_performance"]["sentiment_analyses"] += 1
                
            elif "Cache hit" in line or "cached" in line:
                performance_data["analysis_performance"]["cache_hits"] += 1
                
            elif "Starting" in line and any(comp in line for comp in ["Scraper", "Analysis", "Decision", "Risk", "Statistical"]):
                performance_data["system_health"]["component_starts"] += 1
                
            elif "ERROR" in line or "Exception" in line:
                performance_data["system_health"]["component_errors"] += 1
                
            # Print progress every 30 seconds
            if int(current_time) % 30 == 0:
                elapsed = (current_time - start_time) / 60
                print(f"   ⏱️ {elapsed:.1f}min: {performance_data['news_processing']['items_processed']} items, "
                      f"{performance_data['trading_activity']['decisions_generated']} decisions")
        
        # Stop the log monitoring
        process.terminate()
        
    except KeyboardInterrupt:
        print("\n   ⚠️ Monitoring interrupted by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"   ❌ Monitoring error: {e}")
        if 'process' in locals():
            process.terminate()
    
    # Calculate performance metrics
    actual_duration = (time.time() - start_time) / 60
    performance_data["actual_duration_minutes"] = round(actual_duration, 2)
    
    return performance_data

def analyze_performance_results(performance_data):
    """Analyze performance results and compare to theoretical expectations"""
    print(f"\n📊 PERFORMANCE ANALYSIS RESULTS")
    print("=" * 50)
    
    duration = performance_data["actual_duration_minutes"]
    
    # News processing analysis
    news_data = performance_data["news_processing"]
    items_per_minute = news_data["items_processed"] / max(duration, 0.1)
    sources_per_minute = news_data["sources_scraped"] / max(duration, 0.1)
    
    print(f"📰 News Processing Performance:")
    print(f"   Total Items Processed: {news_data['items_processed']}")
    print(f"   Sources Scraped: {news_data['sources_scraped']}")
    print(f"   Processing Rate: {items_per_minute:.1f} items/minute")
    print(f"   Source Rate: {sources_per_minute:.1f} sources/minute")
    print(f"   Error Rate: {news_data['errors']} errors")
    
    # Analysis performance
    analysis_data = performance_data["analysis_performance"]
    analysis_per_minute = analysis_data["analysis_completed"] / max(duration, 0.1)
    
    print(f"\n🧠 Analysis Performance:")
    print(f"   Analyses Completed: {analysis_data['analysis_completed']}")
    print(f"   Analysis Rate: {analysis_per_minute:.1f} analyses/minute")
    print(f"   AI Analyses: {analysis_data['ai_analyses']}")
    print(f"   Sentiment Analyses: {analysis_data['sentiment_analyses']}")
    print(f"   Cache Hits: {analysis_data['cache_hits']}")
    
    # Trading activity
    trading_data = performance_data["trading_activity"]
    decisions_per_minute = trading_data["decisions_generated"] / max(duration, 0.1)
    
    print(f"\n💼 Trading Performance:")
    print(f"   Decisions Generated: {trading_data['decisions_generated']}")
    print(f"   Decision Rate: {decisions_per_minute:.1f} decisions/minute")
    print(f"   Trades Executed: {trading_data['trades_executed']}")
    
    # System health
    health_data = performance_data["system_health"]
    print(f"\n🏥 System Health:")
    print(f"   Component Starts: {health_data['component_starts']}")
    print(f"   Component Errors: {health_data['component_errors']}")
    print(f"   Error Rate: {(health_data['component_errors'] / max(health_data['component_starts'], 1)) * 100:.1f}%")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    
    # Expected vs actual performance
    expected_items_per_minute = 10  # Conservative baseline expectation
    expected_decisions_per_minute = 2  # Conservative baseline expectation
    
    news_performance = (items_per_minute / expected_items_per_minute) * 100 if expected_items_per_minute > 0 else 0
    decision_performance = (decisions_per_minute / expected_decisions_per_minute) * 100 if expected_decisions_per_minute > 0 else 0
    
    print(f"   News Processing: {news_performance:.1f}% of expected baseline")
    print(f"   Decision Making: {decision_performance:.1f}% of expected baseline")
    
    # Cache effectiveness (if any cache hits detected)
    if analysis_data["cache_hits"] > 0:
        cache_rate = (analysis_data["cache_hits"] / max(analysis_data["analysis_completed"], 1)) * 100
        print(f"   Cache Hit Rate: {cache_rate:.1f}%")
    
    # Overall system assessment
    overall_performance = (news_performance + decision_performance) / 2
    print(f"   Overall Performance: {overall_performance:.1f}% of baseline expectations")
    
    # System status
    if health_data["component_errors"] == 0 and health_data["component_starts"] > 0:
        system_status = "✅ HEALTHY"
    elif health_data["component_errors"] < health_data["component_starts"]:
        system_status = "⚠️ PARTIALLY FUNCTIONAL"
    else:
        system_status = "❌ NEEDS ATTENTION"
        
    print(f"   System Status: {system_status}")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    
    if overall_performance >= 100:
        print("   🏆 System performing at or above baseline expectations")
        print("   🚀 Phase 2 optimizations likely to provide additional benefits")
    elif overall_performance >= 75:
        print("   ✅ System performing adequately")
        print("   📈 Phase 2 optimizations recommended for enhanced performance")
    elif overall_performance >= 50:
        print("   ⚠️ System performance below expectations")
        print("   🔧 Check system configuration and resource allocation")
    else:
        print("   ❌ System performance significantly below expectations")
        print("   🔍 Investigate system issues before implementing optimizations")
    
    return {
        "performance_metrics": {
            "news_items_per_minute": round(items_per_minute, 2),
            "analyses_per_minute": round(analysis_per_minute, 2),
            "decisions_per_minute": round(decisions_per_minute, 2),
            "cache_hit_rate": round((analysis_data["cache_hits"] / max(analysis_data["analysis_completed"], 1)) * 100, 1),
            "error_rate": round((health_data["component_errors"] / max(health_data["component_starts"], 1)) * 100, 1)
        },
        "performance_vs_baseline": {
            "news_processing": round(news_performance, 1),
            "decision_making": round(decision_performance, 1),
            "overall": round(overall_performance, 1)
        },
        "system_status": system_status,
        "optimization_readiness": overall_performance >= 50
    }

def main():
    """Main performance test execution"""
    print("🚀 ACTUAL SYSTEM PERFORMANCE TEST")
    print("Testing real trading system performance...")
    print()
    
    # Check if system is running
    try:
        result = subprocess.run(
            "docker-compose ps algotrading-agent",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if "Up" not in result.stdout:
            print("❌ Trading system is not running")
            print("🔧 Start with: docker-compose up -d --build")
            return
            
    except Exception as e:
        print(f"❌ Could not check system status: {e}")
        return
    
    # Run performance monitoring
    try:
        performance_data = run_docker_logs_test(duration_minutes=2)  # 2-minute test
        analysis_results = analyze_performance_results(performance_data)
        
        # Combine results
        final_results = {
            "test_timestamp": datetime.now().isoformat(),
            "raw_performance_data": performance_data,
            "analysis_results": analysis_results
        }
        
        # Save results
        with open('actual_system_performance_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n✅ Performance test completed successfully")
        print(f"📄 Detailed results saved to: actual_system_performance_results.json")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")

if __name__ == "__main__":
    main()
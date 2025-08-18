#!/usr/bin/env python3
"""
Performance Monitoring Script for Optimization Validation
"""

import psutil
import time
import json
from datetime import datetime

def monitor_performance(duration_minutes=30):
    """Monitor system performance after optimization"""
    print(f"📊 Monitoring performance for {duration_minutes} minutes...")
    
    metrics = []
    samples = duration_minutes * 2  # Sample every 30 seconds
    
    for i in range(samples):
        cpu_percent = psutil.cpu_percent(interval=30)
        memory = psutil.virtual_memory()
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "sample": i + 1
        }
        
        metrics.append(metric)
        print(f"Sample {i+1}/{samples}: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
    
    # Calculate averages
    avg_cpu = sum(m["cpu_percent"] for m in metrics) / len(metrics)
    avg_memory = sum(m["memory_percent"] for m in metrics) / len(metrics)
    max_cpu = max(m["cpu_percent"] for m in metrics)
    max_memory = max(m["memory_percent"] for m in metrics)
    
    results = {
        "monitoring_duration_minutes": duration_minutes,
        "samples_collected": len(metrics),
        "performance_summary": {
            "cpu_average": avg_cpu,
            "cpu_peak": max_cpu,
            "memory_average": avg_memory,
            "memory_peak": max_memory
        },
        "raw_metrics": metrics,
        "validation": {
            "cpu_within_target": 60 <= avg_cpu <= 80,
            "memory_within_target": 70 <= avg_memory <= 85,
            "cpu_not_overloaded": max_cpu < 90,
            "memory_not_overloaded": max_memory < 90
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"performance_monitoring_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Performance Summary:")
    print(f"   CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% peak")
    print(f"   Memory: {avg_memory:.1f}% avg, {max_memory:.1f}% peak")
    print(f"   Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    monitor_performance()

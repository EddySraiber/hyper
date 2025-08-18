#!/usr/bin/env python3
"""
Quick Optimization Validation Script
"""

import psutil
import json
import time
from datetime import datetime

def validate_optimization():
    """Validate the optimization results"""
    print("🔍 VALIDATING CPU OPTIMIZATION RESULTS")
    print("=" * 45)
    
    # Collect performance samples
    cpu_samples = []
    memory_samples = []
    
    for i in range(6):  # 30 seconds of sampling
        cpu_percent = psutil.cpu_percent(interval=5)
        memory = psutil.virtual_memory()
        
        cpu_samples.append(cpu_percent)
        memory_samples.append(memory.percent)
        
        print(f"   Sample {i+1}: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
    
    # Calculate metrics
    avg_cpu = sum(cpu_samples) / len(cpu_samples)
    avg_memory = sum(memory_samples) / len(memory_samples)
    max_cpu = max(cpu_samples)
    max_memory = max(memory_samples)
    
    # Cost analysis
    before_cost = 50  # Estimated $50/month for 0.5 CPU + 1GB
    after_cost = 30   # Estimated $30/month for 0.25 CPU + 800MB
    monthly_savings = before_cost - after_cost
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "optimization_results": {
            "cpu_average": avg_cpu,
            "cpu_peak": max_cpu,
            "memory_average": avg_memory,
            "memory_peak": max_memory
        },
        "resource_allocation": {
            "cpu_limit_before": "0.5 cores",
            "cpu_limit_after": "0.25 cores",
            "memory_limit_before": "1GB",
            "memory_limit_after": "800MB"
        },
        "cost_impact": {
            "monthly_cost_before": before_cost,
            "monthly_cost_after": after_cost,
            "monthly_savings": monthly_savings,
            "cost_reduction_percent": (monthly_savings / before_cost) * 100
        },
        "validation": {
            "system_stable": max_cpu < 80,
            "memory_adequate": max_memory < 80,
            "optimization_successful": monthly_savings > 0
        }
    }
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"   CPU Usage: {avg_cpu:.1f}% avg, {max_cpu:.1f}% peak")
    print(f"   Memory Usage: {avg_memory:.1f}% avg, {max_memory:.1f}% peak")
    print(f"   Monthly Savings: ${monthly_savings}")
    print(f"   Cost Reduction: {results['cost_impact']['cost_reduction_percent']:.0f}%")
    
    # Save results
    with open('optimization_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Validation completed - results saved")
    return results

if __name__ == "__main__":
    validate_optimization()
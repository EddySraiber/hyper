#!/usr/bin/env python3
"""
Performance Optimization Implementation Tool

Implements Phase 1 CPU rightsizing and performance optimization
based on FinOps analysis results.
"""

import os
import json
import time
import psutil
from datetime import datetime
from typing import Dict, List, Any

class PerformanceOptimizer:
    """Implements performance optimizations based on analysis"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.optimization_results = {}
        
    def analyze_current_state(self):
        """Analyze current system state before optimization"""
        print("🔍 ANALYZING CURRENT SYSTEM STATE")
        print("=" * 45)
        
        # System metrics
        cpu_count = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        # Sample current performance
        cpu_samples = []
        for i in range(5):
            cpu_percent = psutil.cpu_percent(interval=2)
            cpu_samples.append(cpu_percent)
            
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        
        self.baseline_metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_cores": cpu_count,
            "memory_total_gb": memory.total / (1024**3),
            "cpu_utilization_percent": avg_cpu,
            "memory_utilization_percent": memory.percent,
            "estimated_monthly_cost": self._calculate_monthly_cost(cpu_count, memory.total / (1024**3))
        }
        
        print(f"📊 Current Configuration:")
        print(f"   CPU Cores: {cpu_count}")
        print(f"   Memory: {memory.total / (1024**3):.1f}GB")
        print(f"   CPU Utilization: {avg_cpu:.1f}%")
        print(f"   Memory Utilization: {memory.percent:.1f}%")
        print(f"   Estimated Monthly Cost: ${self.baseline_metrics['estimated_monthly_cost']:.2f}")
        
        return self.baseline_metrics
        
    def calculate_optimization_recommendations(self):
        """Calculate specific optimization recommendations"""
        print(f"\n💡 CALCULATING OPTIMIZATION RECOMMENDATIONS")
        print("=" * 50)
        
        current_cpu = self.baseline_metrics["cpu_cores"]
        current_memory = self.baseline_metrics["memory_total_gb"]
        current_cpu_util = self.baseline_metrics["cpu_utilization_percent"]
        current_memory_util = self.baseline_metrics["memory_utilization_percent"]
        
        # CPU optimization calculation
        target_cpu_utilization = 70  # Target 70% utilization
        optimal_cores = max(2, int((current_cpu_util / target_cpu_utilization) * current_cpu))
        
        # Memory optimization calculation  
        target_memory_utilization = 80  # Target 80% utilization
        optimal_memory = max(4, (current_memory_util / target_memory_utilization) * current_memory)
        
        # Cost calculations
        current_cost = self._calculate_monthly_cost(current_cpu, current_memory)
        optimized_cost = self._calculate_monthly_cost(optimal_cores, optimal_memory)
        monthly_savings = current_cost - optimized_cost
        
        recommendations = {
            "current_configuration": {
                "cpu_cores": current_cpu,
                "memory_gb": current_memory,
                "monthly_cost": current_cost
            },
            "optimized_configuration": {
                "cpu_cores": optimal_cores,
                "memory_gb": optimal_memory,
                "monthly_cost": optimized_cost
            },
            "optimization_impact": {
                "cpu_reduction_percent": ((current_cpu - optimal_cores) / current_cpu) * 100,
                "memory_reduction_percent": ((current_memory - optimal_memory) / current_memory) * 100,
                "monthly_savings": monthly_savings,
                "annual_savings": monthly_savings * 12,
                "cost_reduction_percent": (monthly_savings / current_cost) * 100
            },
            "performance_impact": {
                "expected_cpu_utilization": min(90, current_cpu_util * (current_cpu / optimal_cores)),
                "expected_memory_utilization": min(90, current_memory_util * (current_memory / optimal_memory)),
                "safety_margin": optimal_cores / (current_cpu_util / 100 * current_cpu),
                "risk_level": "low" if optimal_cores >= 2 else "medium"
            }
        }
        
        print(f"🎯 Optimization Recommendations:")
        print(f"   CPU: {current_cpu} → {optimal_cores} cores ({recommendations['optimization_impact']['cpu_reduction_percent']:.0f}% reduction)")
        print(f"   Memory: {current_memory:.1f}GB → {optimal_memory:.1f}GB ({recommendations['optimization_impact']['memory_reduction_percent']:.0f}% reduction)")
        print(f"   Monthly Cost: ${current_cost:.2f} → ${optimized_cost:.2f}")
        print(f"   Monthly Savings: ${monthly_savings:.2f} ({recommendations['optimization_impact']['cost_reduction_percent']:.0f}% reduction)")
        print(f"   Safety Margin: {recommendations['performance_impact']['safety_margin']:.1f}x current peak usage")
        
        return recommendations
        
    def generate_docker_optimization_config(self, recommendations: Dict):
        """Generate Docker configuration for optimization"""
        print(f"\n🐳 GENERATING DOCKER OPTIMIZATION CONFIG")
        print("=" * 45)
        
        optimal_config = recommendations["optimized_configuration"]
        
        # Generate Docker Compose override
        docker_compose_override = f"""version: '3.8'

services:
  algotrading-agent:
    deploy:
      resources:
        limits:
          cpus: '{optimal_config['cpu_cores']}.0'
          memory: {optimal_config['memory_gb']:.0f}G
        reservations:
          cpus: '{optimal_config['cpu_cores'] * 0.5:.1f}'
          memory: {optimal_config['memory_gb'] * 0.7:.1f}G
    environment:
      - OPTIMIZATION_MODE=enabled
      - CPU_CORES={optimal_config['cpu_cores']}
      - MEMORY_GB={optimal_config['memory_gb']:.0f}
      - PERFORMANCE_MONITORING=true
"""
        
        # Generate Docker run command
        docker_run_command = f"""docker run -d \\
  --name algotrading-optimized \\
  --cpus="{optimal_config['cpu_cores']}.0" \\
  --memory="{optimal_config['memory_gb']:.0f}g" \\
  --restart=unless-stopped \\
  -v $(pwd)/config:/app/config \\
  -v $(pwd)/data:/app/data \\
  -v $(pwd)/logs:/app/logs \\
  algotrading-agent:latest"""
        
        config_files = {
            "docker-compose.override.yml": docker_compose_override,
            "docker_run_optimized.sh": docker_run_command,
            "optimization_config.json": json.dumps(recommendations, indent=2, default=str)
        }
        
        # Save configuration files
        os.makedirs("analysis/performance/optimization_configs", exist_ok=True)
        
        for filename, content in config_files.items():
            filepath = f"analysis/performance/optimization_configs/{filename}"
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Created: {filepath}")
            
        return config_files
        
    def create_monitoring_script(self):
        """Create performance monitoring script for validation"""
        monitoring_script = '''#!/usr/bin/env python3
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
    
    print(f"\\n📊 Performance Summary:")
    print(f"   CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% peak")
    print(f"   Memory: {avg_memory:.1f}% avg, {max_memory:.1f}% peak")
    print(f"   Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    monitor_performance()
'''
        
        script_path = "analysis/performance/optimization_configs/monitor_performance.py"
        with open(script_path, 'w') as f:
            f.write(monitoring_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"✅ Created monitoring script: {script_path}")
        return script_path
        
    def create_implementation_guide(self, recommendations: Dict):
        """Create step-by-step implementation guide"""
        guide_content = f"""# Performance Optimization Implementation Guide

## 📋 Pre-Implementation Checklist

### 1. Backup Current Configuration
```bash
# Backup current Docker configuration
cp docker-compose.yml docker-compose.yml.backup

# Backup current container settings
docker inspect algotrading-agent > container_config_backup.json

# Create system snapshot
docker commit algotrading-agent algotrading-agent:pre-optimization
```

### 2. Prepare Monitoring
```bash
# Set up monitoring for validation
python3 analysis/performance/optimization_configs/monitor_performance.py &
MONITOR_PID=$!
echo $MONITOR_PID > monitor.pid
```

## 🚀 Implementation Steps

### Phase 1: Container Optimization (Immediate)

#### Step 1: Stop Current Container
```bash
docker-compose down
```

#### Step 2: Apply Optimized Configuration
```bash
# Copy the optimized docker-compose override
cp analysis/performance/optimization_configs/docker-compose.override.yml ./

# Start with optimized configuration
docker-compose up -d
```

#### Step 3: Validate Performance
```bash
# Monitor for 30 minutes
python3 analysis/performance/optimization_configs/monitor_performance.py

# Check container resource usage
docker stats algotrading-agent --no-stream
```

### Phase 2: Performance Validation (24 hours)

#### Expected Results:
- **CPU Utilization**: {recommendations['performance_impact']['expected_cpu_utilization']:.1f}% average
- **Memory Utilization**: {recommendations['performance_impact']['expected_memory_utilization']:.1f}% average
- **Monthly Cost**: ${recommendations['optimized_configuration']['monthly_cost']:.2f}
- **Monthly Savings**: ${recommendations['optimization_impact']['monthly_savings']:.2f}

#### Validation Criteria:
- ✅ CPU utilization 60-80%
- ✅ Memory utilization 70-85%
- ✅ Trading latency <100ms
- ✅ Guardian Service scans <30s
- ✅ System stability 99.9%+

### Phase 3: Monitoring & Alerting (Ongoing)

#### Set Up Continuous Monitoring
```bash
# Add monitoring to crontab
echo "*/15 * * * * python3 /app/analysis/performance/optimization_configs/monitor_performance.py" | crontab -

# Set up alerts for performance issues
# (Implement based on your monitoring infrastructure)
```

## ⚠️ Rollback Procedure

If any issues occur:

```bash
# Stop optimized container
docker-compose down

# Restore original configuration
cp docker-compose.yml.backup docker-compose.yml
rm docker-compose.override.yml

# Start with original configuration
docker-compose up -d

# Or restore from snapshot
docker stop algotrading-agent
docker rm algotrading-agent
docker run --name algotrading-agent algotrading-agent:pre-optimization
```

## 📊 Expected Benefits

### Immediate (Week 1):
- ✅ **Cost Reduction**: {recommendations['optimization_impact']['cost_reduction_percent']:.0f}% monthly savings
- ✅ **Resource Efficiency**: Better CPU and memory utilization
- ✅ **Performance**: Maintained trading performance with lower costs

### Medium-term (Month 1):
- ✅ **Optimized Operations**: More efficient resource allocation
- ✅ **Cost Predictability**: Stable, lower monthly costs
- ✅ **Scaling Readiness**: Better foundation for future growth

## 🎯 Success Metrics

Monitor these KPIs for 30 days:

- **CPU Utilization**: Target 60-80%
- **Memory Utilization**: Target 70-85%
- **Monthly Cost**: Target ${recommendations['optimized_configuration']['monthly_cost']:.2f}
- **Trading Performance**: Maintain <100ms latency
- **System Reliability**: Maintain 99.9%+ uptime

## 📞 Support

If you encounter issues:
1. Check the monitoring results in generated JSON files
2. Validate container logs: `docker-compose logs algotrading-agent`
3. Use rollback procedure if necessary
4. Monitor performance metrics for anomalies

---

**Total Expected Monthly Savings**: ${recommendations['optimization_impact']['monthly_savings']:.2f} ({recommendations['optimization_impact']['cost_reduction_percent']:.0f}% reduction)
**Annual Savings**: ${recommendations['optimization_impact']['annual_savings']:.2f}
**Risk Level**: {recommendations['performance_impact']['risk_level'].upper()}
"""
        
        guide_path = "analysis/performance/optimization_configs/IMPLEMENTATION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"✅ Created implementation guide: {guide_path}")
        return guide_path
        
    def _calculate_monthly_cost(self, cpu_cores: int, memory_gb: float) -> float:
        """Calculate estimated monthly cost"""
        # Cost model based on typical cloud pricing
        cpu_cost_per_core_month = 15.0  # $15 per core per month
        memory_cost_per_gb_month = 3.0   # $3 per GB per month
        
        return (cpu_cores * cpu_cost_per_core_month) + (memory_gb * memory_cost_per_gb_month)
        
    def run_complete_optimization_analysis(self):
        """Run complete optimization analysis and generate implementation materials"""
        print("🚀 PERFORMANCE OPTIMIZATION IMPLEMENTATION TOOL")
        print("=" * 55)
        
        try:
            # Phase 1: Analyze current state
            baseline = self.analyze_current_state()
            
            # Phase 2: Calculate recommendations
            recommendations = self.calculate_optimization_recommendations()
            
            # Phase 3: Generate implementation materials
            docker_configs = self.generate_docker_optimization_config(recommendations)
            monitoring_script = self.create_monitoring_script()
            implementation_guide = self.create_implementation_guide(recommendations)
            
            # Phase 4: Summary and next steps
            print(f"\n✅ OPTIMIZATION ANALYSIS COMPLETE!")
            print("=" * 40)
            
            impact = recommendations["optimization_impact"]
            print(f"💰 Financial Impact:")
            print(f"   Monthly Savings: ${impact['monthly_savings']:.2f}")
            print(f"   Annual Savings: ${impact['annual_savings']:.2f}")
            print(f"   Cost Reduction: {impact['cost_reduction_percent']:.0f}%")
            
            performance = recommendations["performance_impact"]
            print(f"\n📊 Performance Impact:")
            print(f"   Expected CPU Utilization: {performance['expected_cpu_utilization']:.1f}%")
            print(f"   Expected Memory Utilization: {performance['expected_memory_utilization']:.1f}%")
            print(f"   Safety Margin: {performance['safety_margin']:.1f}x current usage")
            print(f"   Risk Level: {performance['risk_level'].upper()}")
            
            print(f"\n📋 Implementation Materials Created:")
            print(f"   • Docker configuration files")
            print(f"   • Performance monitoring script")
            print(f"   • Step-by-step implementation guide")
            print(f"   • Rollback procedures")
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Review implementation guide: analysis/performance/optimization_configs/IMPLEMENTATION_GUIDE.md")
            print(f"   2. Backup current configuration")
            print(f"   3. Apply optimized Docker configuration")
            print(f"   4. Monitor performance for 24-48 hours")
            print(f"   5. Validate ${impact['monthly_savings']:.2f}/month cost savings")
            
            return {
                "baseline_metrics": baseline,
                "optimization_recommendations": recommendations,
                "implementation_files": {
                    "docker_configs": docker_configs,
                    "monitoring_script": monitoring_script,
                    "implementation_guide": implementation_guide
                }
            }
            
        except Exception as e:
            print(f"❌ Optimization analysis failed: {e}")
            raise


def main():
    """Execute performance optimization implementation tool"""
    optimizer = PerformanceOptimizer()
    results = optimizer.run_complete_optimization_analysis()
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"analysis/performance/optimization_implementation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Complete results saved to: {results_file}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick CPU Usage Analysis for FinOps Performance Assessment

Rapid CPU usage analysis across trading system components with
immediate optimization recommendations.
"""

import psutil
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class QuickCPUAnalyzer:
    """Quick CPU usage analyzer for immediate FinOps insights"""
    
    def __init__(self):
        self.cost_per_cpu_core_hour = 0.05  # $0.05 per core hour
        self.cost_per_gb_memory_hour = 0.02  # $0.02 per GB hour
        
    def analyze_current_usage(self) -> Dict[str, Any]:
        """Analyze current CPU and memory usage"""
        print("🔍 QUICK CPU USAGE ANALYSIS")
        print("=" * 40)
        
        # System information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        
        print(f"💻 System Configuration:")
        print(f"   CPU Cores: {cpu_count} logical, {cpu_physical} physical")
        print(f"   Memory: {memory.total / (1024**3):.1f}GB total")
        
        # Sample CPU usage over 30 seconds
        print(f"\n📊 Sampling CPU usage for 30 seconds...")
        cpu_samples = []
        memory_samples = []
        
        for i in range(6):  # 6 samples over 30 seconds
            cpu_percent = psutil.cpu_percent(interval=5)
            memory_percent = memory.percent
            cpu_samples.append(cpu_percent)
            memory_samples.append(memory_percent)
            print(f"   Sample {i+1}: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
        
        # Calculate averages
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        max_memory = max(memory_samples)
        
        # Process analysis
        print(f"\n🔍 Analyzing running processes...")
        processes = []
        trading_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0.1 or proc_info['memory_info'].rss > 10 * 1024 * 1024:  # >10MB
                    processes.append({
                        'name': proc_info['name'],
                        'cpu_percent': proc_info['cpu_percent'],
                        'memory_mb': proc_info['memory_info'].rss / (1024**2),
                        'cmdline': ' '.join(proc_info['cmdline'][:3]) if proc_info['cmdline'] else ''
                    })
                    
                    # Identify trading-related processes
                    cmdline = ' '.join(proc_info['cmdline']) if proc_info['cmdline'] else ''
                    if any(keyword in cmdline.lower() for keyword in 
                          ['python', 'algotrading', 'trading', 'main.py', 'docker']):
                        trading_processes.append({
                            'name': proc_info['name'],
                            'cpu_percent': proc_info['cpu_percent'],
                            'memory_mb': proc_info['memory_info'].rss / (1024**2),
                            'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Cost analysis
        cpu_cores_used = (avg_cpu / 100) * cpu_count
        memory_gb_used = avg_memory / 100 * (memory.total / (1024**3))
        
        current_hourly_cost = (cpu_cores_used * self.cost_per_cpu_core_hour + 
                              memory_gb_used * self.cost_per_gb_memory_hour)
        
        # Optimization analysis
        optimization_opportunities = self._analyze_optimization_opportunities(
            avg_cpu, max_cpu, avg_memory, cpu_count, memory.total / (1024**3), current_hourly_cost
        )
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "system_config": {
                "cpu_cores": cpu_count,
                "physical_cores": cpu_physical,
                "memory_total_gb": memory.total / (1024**3)
            },
            "usage_analysis": {
                "cpu_average_percent": avg_cpu,
                "cpu_peak_percent": max_cpu,
                "memory_average_percent": avg_memory,
                "memory_peak_percent": max_memory,
                "cpu_cores_utilized": cpu_cores_used,
                "memory_gb_utilized": memory_gb_used
            },
            "cost_analysis": {
                "current_hourly_cost": current_hourly_cost,
                "estimated_daily_cost": current_hourly_cost * 24,
                "estimated_monthly_cost": current_hourly_cost * 24 * 30,
                "cpu_cost_percentage": (cpu_cores_used * self.cost_per_cpu_core_hour / current_hourly_cost) * 100 if current_hourly_cost > 0 else 0,
                "memory_cost_percentage": (memory_gb_used * self.cost_per_gb_memory_hour / current_hourly_cost) * 100 if current_hourly_cost > 0 else 0
            },
            "process_analysis": {
                "total_processes_analyzed": len(processes),
                "trading_related_processes": len(trading_processes),
                "top_cpu_processes": sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5],
                "top_memory_processes": sorted(processes, key=lambda x: x['memory_mb'], reverse=True)[:5],
                "trading_processes": trading_processes
            },
            "optimization_opportunities": optimization_opportunities
        }
        
        return results
    
    def _analyze_optimization_opportunities(self, avg_cpu: float, max_cpu: float, 
                                          avg_memory: float, cpu_count: int, 
                                          memory_gb: float, current_cost: float) -> List[Dict]:
        """Analyze optimization opportunities"""
        opportunities = []
        
        # CPU rightsizing analysis
        if avg_cpu < 30:
            # Calculate optimal CPU allocation
            target_utilization = 70  # Target 70% utilization
            optimal_cores = max(2, int((avg_cpu / target_utilization) * cpu_count))
            cpu_reduction = cpu_count - optimal_cores
            cpu_savings = cpu_reduction * self.cost_per_cpu_core_hour * 24 * 30
            
            opportunities.append({
                "type": "cpu_rightsizing",
                "priority": "high",
                "current_state": f"{cpu_count} cores at {avg_cpu:.1f}% utilization",
                "recommended_state": f"{optimal_cores} cores at ~{target_utilization}% utilization",
                "description": f"Reduce CPU allocation by {cpu_reduction} cores ({cpu_reduction/cpu_count*100:.0f}% reduction)",
                "estimated_monthly_savings": cpu_savings,
                "implementation_effort": "medium",
                "risk_level": "low",
                "confidence": 0.9
            })
        
        # Memory rightsizing analysis  
        if avg_memory < 50:
            target_memory_utilization = 75
            optimal_memory = max(2, (avg_memory / target_memory_utilization) * memory_gb)
            memory_reduction = memory_gb - optimal_memory
            memory_savings = memory_reduction * self.cost_per_gb_memory_hour * 24 * 30
            
            opportunities.append({
                "type": "memory_rightsizing",
                "priority": "medium",
                "current_state": f"{memory_gb:.1f}GB at {avg_memory:.1f}% utilization",
                "recommended_state": f"{optimal_memory:.1f}GB at ~{target_memory_utilization}% utilization",
                "description": f"Reduce memory allocation by {memory_reduction:.1f}GB ({memory_reduction/memory_gb*100:.0f}% reduction)",
                "estimated_monthly_savings": memory_savings,
                "implementation_effort": "low",
                "risk_level": "low",
                "confidence": 0.85
            })
        
        # Instance type optimization
        if avg_cpu < 20 and avg_memory < 40:
            total_savings = current_cost * 0.6 * 24 * 30  # 60% cost reduction potential
            opportunities.append({
                "type": "instance_optimization",
                "priority": "high",
                "current_state": f"Over-provisioned instance (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)",
                "recommended_state": "Smaller instance type with auto-scaling",
                "description": "Move to smaller instance type with burst capability",
                "estimated_monthly_savings": total_savings,
                "implementation_effort": "high",
                "risk_level": "medium",
                "confidence": 0.75
            })
        
        # Auto-scaling opportunity
        if max_cpu - avg_cpu > 20:  # High variability
            autoscale_savings = current_cost * 0.3 * 24 * 30  # 30% savings potential
            opportunities.append({
                "type": "auto_scaling",
                "priority": "medium",
                "current_state": f"Fixed capacity with high variability (avg: {avg_cpu:.1f}%, peak: {max_cpu:.1f}%)",
                "recommended_state": "Auto-scaling based on demand",
                "description": "Implement auto-scaling to match resource allocation with demand",
                "estimated_monthly_savings": autoscale_savings,
                "implementation_effort": "high",
                "risk_level": "medium",
                "confidence": 0.7
            })
        
        return sorted(opportunities, key=lambda x: x["estimated_monthly_savings"], reverse=True)
    
    def print_analysis_summary(self, results: Dict[str, Any]):
        """Print analysis summary"""
        print(f"\n📊 ANALYSIS RESULTS:")
        print("=" * 30)
        
        usage = results["usage_analysis"]
        cost = results["cost_analysis"]
        processes = results["process_analysis"]
        
        print(f"🖥️  System Utilization:")
        print(f"   CPU: {usage['cpu_average_percent']:.1f}% avg, {usage['cpu_peak_percent']:.1f}% peak")
        print(f"   Memory: {usage['memory_average_percent']:.1f}% avg, {usage['memory_peak_percent']:.1f}% peak")
        print(f"   Cores used: {usage['cpu_cores_utilized']:.2f} / {results['system_config']['cpu_cores']}")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Hourly: ${cost['current_hourly_cost']:.4f}")
        print(f"   Daily: ${cost['estimated_daily_cost']:.2f}")
        print(f"   Monthly: ${cost['estimated_monthly_cost']:.2f}")
        print(f"   CPU cost: {cost['cpu_cost_percentage']:.1f}%")
        print(f"   Memory cost: {cost['memory_cost_percentage']:.1f}%")
        
        print(f"\n🔍 Process Analysis:")
        print(f"   Total processes: {processes['total_processes_analyzed']}")
        print(f"   Trading processes: {processes['trading_related_processes']}")
        
        if processes['top_cpu_processes']:
            print(f"   Top CPU consumer: {processes['top_cpu_processes'][0]['name']} ({processes['top_cpu_processes'][0]['cpu_percent']:.1f}%)")
        
        opportunities = results["optimization_opportunities"]
        print(f"\n🚀 Optimization Opportunities ({len(opportunities)}):")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"   {i}. {opp['type']} - ${opp['estimated_monthly_savings']:.2f}/month")
            print(f"      {opp['description']}")
        
        if opportunities:
            total_savings = sum(opp['estimated_monthly_savings'] for opp in opportunities)
            print(f"\n💡 Total potential monthly savings: ${total_savings:.2f}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = f"analysis/performance/quick_cpu_analysis_{timestamp}.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {json_file}")
        except Exception as e:
            print(f"⚠️  Could not save JSON results: {e}")
        
        # Save markdown summary
        md_file = f"analysis/performance/quick_cpu_summary_{timestamp}.md"
        try:
            with open(md_file, 'w') as f:
                f.write("# Quick CPU Usage Analysis Summary\n\n")
                f.write(f"**Generated**: {results['timestamp']}\n\n")
                
                usage = results["usage_analysis"]
                cost = results["cost_analysis"]
                
                f.write("## System Utilization\n")
                f.write(f"- CPU: {usage['cpu_average_percent']:.1f}% average, {usage['cpu_peak_percent']:.1f}% peak\n")
                f.write(f"- Memory: {usage['memory_average_percent']:.1f}% average, {usage['memory_peak_percent']:.1f}% peak\n")
                f.write(f"- CPU cores utilized: {usage['cpu_cores_utilized']:.2f} / {results['system_config']['cpu_cores']}\n\n")
                
                f.write("## Cost Analysis\n")
                f.write(f"- Monthly cost: ${cost['estimated_monthly_cost']:.2f}\n")
                f.write(f"- CPU cost percentage: {cost['cpu_cost_percentage']:.1f}%\n")
                f.write(f"- Memory cost percentage: {cost['memory_cost_percentage']:.1f}%\n\n")
                
                f.write("## Top Optimization Opportunities\n")
                for i, opp in enumerate(results["optimization_opportunities"][:5], 1):
                    f.write(f"{i}. **{opp['type']}** ({opp['priority']} priority)\n")
                    f.write(f"   - {opp['description']}\n")
                    f.write(f"   - Monthly savings: ${opp['estimated_monthly_savings']:.2f}\n")
                    f.write(f"   - Risk: {opp['risk_level']}, Confidence: {opp['confidence']:.0%}\n\n")
                
            print(f"📄 Summary saved to: {md_file}")
        except Exception as e:
            print(f"⚠️  Could not save markdown summary: {e}")


def main():
    """Execute quick CPU analysis"""
    print("🚀 QUICK CPU USAGE ANALYSIS FOR FINOPS")
    print("=" * 45)
    
    analyzer = QuickCPUAnalyzer()
    
    try:
        results = analyzer.analyze_current_usage()
        analyzer.print_analysis_summary(results)
        analyzer.save_results(results)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📋 Key insights and optimization opportunities identified")
        print(f"📊 Detailed results saved for further analysis")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
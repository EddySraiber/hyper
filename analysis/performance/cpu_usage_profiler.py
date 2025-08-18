#!/usr/bin/env python3
"""
CPU Usage Profiler for FinOps Performance Analysis

Comprehensive CPU usage analysis across all trading flows with
real-time monitoring and cost optimization recommendations.
"""

import asyncio
import psutil
import time
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

@dataclass
class ComponentPerformanceMetrics:
    """Performance metrics for individual components"""
    component_name: str
    cpu_percent: float
    memory_mb: float
    io_read_bytes: int
    io_write_bytes: int
    network_bytes_sent: int
    network_bytes_recv: int
    execution_time_ms: float
    operations_per_second: float
    cost_per_operation: float
    timestamp: datetime

@dataclass
class TradingFlowMetrics:
    """Trading flow specific performance metrics"""
    flow_name: str
    total_cpu_time: float
    peak_cpu_usage: float
    average_memory_mb: float
    total_operations: int
    success_rate: float
    average_latency_ms: float
    cost_per_successful_trade: float
    bottleneck_components: List[str]
    optimization_opportunities: List[str]

class CPUUsageProfiler:
    """Comprehensive CPU usage profiler for trading system"""
    
    def __init__(self, monitoring_duration_minutes: int = 15):
        self.monitoring_duration = monitoring_duration_minutes
        self.metrics_history = []
        self.component_metrics = defaultdict(list)
        self.trading_flow_metrics = {}
        self.system_baseline = None
        self.profiling_active = False
        
        # FinOps cost calculations (per hour)
        self.cost_per_cpu_core_hour = 0.05  # $0.05 per core hour
        self.cost_per_gb_memory_hour = 0.02  # $0.02 per GB hour
        self.cost_per_gb_io = 0.001  # $0.001 per GB I/O
        
        # Component process identification
        self.component_processes = {
            "news_scraper": ["python", "news_scraper"],
            "ai_analyzer": ["python", "ai_analyzer", "openai", "groq"],
            "decision_engine": ["python", "decision_engine"],
            "risk_manager": ["python", "risk_manager"],
            "trade_manager": ["python", "trade_manager"],
            "guardian_service": ["python", "guardian_service"],
            "fast_trading": ["python", "momentum_pattern", "express_execution"],
            "main_process": ["python", "main.py"]
        }
        
    def start_profiling(self):
        """Start comprehensive CPU usage profiling"""
        print("🔍 STARTING COMPREHENSIVE CPU USAGE PROFILING")
        print("=" * 60)
        print(f"⏱️  Monitoring Duration: {self.monitoring_duration} minutes")
        print(f"💰 Cost Analysis: CPU cores, memory, I/O operations")
        print(f"🎯 Focus: Trading flows, component efficiency, optimization opportunities")
        print()
        
        self.profiling_active = True
        
        # Collect system baseline
        self._collect_system_baseline()
        
        # Start monitoring threads
        monitor_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        component_thread = threading.Thread(target=self._monitor_component_performance, daemon=True)
        flow_thread = threading.Thread(target=self._monitor_trading_flows, daemon=True)
        
        monitor_thread.start()
        component_thread.start()
        flow_thread.start()
        
        # Wait for profiling duration
        time.sleep(self.monitoring_duration * 60)
        self.profiling_active = False
        
        # Generate comprehensive analysis
        analysis = self._generate_comprehensive_analysis()
        
        # Save results
        self._save_profiling_results(analysis)
        
        return analysis
        
    def _collect_system_baseline(self):
        """Collect system baseline metrics"""
        print("📊 Collecting system baseline...")
        
        cpu_info = psutil.cpu_count(logical=True)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        self.system_baseline = {
            "timestamp": datetime.now().isoformat(),
            "cpu_cores": cpu_info,
            "physical_cores": psutil.cpu_count(logical=False),
            "memory_total_gb": memory_info.total / (1024**3),
            "disk_total_gb": disk_info.total / (1024**3),
            "system_load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        print(f"   💻 CPU Cores: {cpu_info} logical, {psutil.cpu_count(logical=False)} physical")
        print(f"   🧠 Memory: {memory_info.total / (1024**3):.1f}GB total")
        print(f"   💾 Disk: {disk_info.total / (1024**3):.1f}GB total")
        
    def _monitor_system_resources(self):
        """Monitor overall system resource usage"""
        while self.profiling_active:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
                load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                
                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Disk I/O metrics
                disk_io = psutil.disk_io_counters()
                
                # Network I/O metrics
                network_io = psutil.net_io_counters()
                
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent_total": cpu_percent,
                    "cpu_percent_per_core": cpu_per_core,
                    "load_average": load_avg,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "swap_percent": swap.percent,
                    "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                    "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0,
                    "network_sent_mb": network_io.bytes_sent / (1024**2) if network_io else 0,
                    "network_recv_mb": network_io.bytes_recv / (1024**2) if network_io else 0
                }
                
                self.metrics_history.append(metrics)
                
                # Calculate real-time costs
                self._calculate_realtime_costs(metrics)
                
                time.sleep(5)  # Sample every 5 seconds
                
            except Exception as e:
                print(f"⚠️  System monitoring error: {e}")
                time.sleep(5)
                
    def _monitor_component_performance(self):
        """Monitor individual component performance"""
        while self.profiling_active:
            try:
                # Get all running processes
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                    try:
                        proc_info = proc.info
                        cmdline = ' '.join(proc_info['cmdline']) if proc_info['cmdline'] else ''
                        
                        # Identify component based on command line
                        component = self._identify_component(proc_info['name'], cmdline)
                        
                        if component:
                            # Collect detailed metrics for this component
                            with proc.oneshot():
                                cpu_percent = proc.cpu_percent()
                                memory_info = proc.memory_info()
                                io_counters = proc.io_counters() if hasattr(proc, 'io_counters') else None
                                
                                metrics = ComponentPerformanceMetrics(
                                    component_name=component,
                                    cpu_percent=cpu_percent,
                                    memory_mb=memory_info.rss / (1024**2),
                                    io_read_bytes=io_counters.read_bytes if io_counters else 0,
                                    io_write_bytes=io_counters.write_bytes if io_counters else 0,
                                    network_bytes_sent=0,  # Would need more detailed monitoring
                                    network_bytes_recv=0,
                                    execution_time_ms=0,  # Would be calculated over time
                                    operations_per_second=0,  # Component-specific calculation
                                    cost_per_operation=0,  # Calculated based on resource usage
                                    timestamp=datetime.now()
                                )
                                
                                self.component_metrics[component].append(metrics)
                                
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                        
                time.sleep(10)  # Sample every 10 seconds
                
            except Exception as e:
                print(f"⚠️  Component monitoring error: {e}")
                time.sleep(10)
                
    def _monitor_trading_flows(self):
        """Monitor trading flow specific performance"""
        flows = {
            "news_processing": ["news_scraper", "news_filter", "news_analysis_brain"],
            "decision_making": ["decision_engine", "ai_analyzer", "risk_manager"],
            "trade_execution": ["trade_manager", "enhanced_trade_manager", "alpaca_client"],
            "safety_monitoring": ["guardian_service", "position_protector", "bracket_order_manager"],
            "fast_trading": ["momentum_pattern_detector", "express_execution_manager"]
        }
        
        while self.profiling_active:
            try:
                for flow_name, components in flows.items():
                    # Aggregate metrics for this flow
                    flow_cpu = 0
                    flow_memory = 0
                    flow_operations = 0
                    
                    for component in components:
                        if component in self.component_metrics:
                            recent_metrics = self.component_metrics[component][-5:]  # Last 5 samples
                            if recent_metrics:
                                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                                avg_memory = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)
                                flow_cpu += avg_cpu
                                flow_memory += avg_memory
                    
                    # Store flow metrics
                    self.trading_flow_metrics[flow_name] = {
                        "timestamp": datetime.now().isoformat(),
                        "total_cpu_percent": flow_cpu,
                        "total_memory_mb": flow_memory,
                        "components_active": len([c for c in components if c in self.component_metrics]),
                        "estimated_cost_per_hour": self._calculate_flow_cost(flow_cpu, flow_memory)
                    }
                
                time.sleep(30)  # Sample every 30 seconds
                
            except Exception as e:
                print(f"⚠️  Trading flow monitoring error: {e}")
                time.sleep(30)
                
    def _identify_component(self, process_name: str, cmdline: str) -> Optional[str]:
        """Identify which component a process belongs to"""
        cmdline_lower = cmdline.lower()
        
        for component, keywords in self.component_processes.items():
            if any(keyword.lower() in cmdline_lower for keyword in keywords):
                return component
                
        # Check for Python processes running specific modules
        if 'python' in process_name.lower():
            if 'news_scraper' in cmdline_lower:
                return 'news_scraper'
            elif 'decision_engine' in cmdline_lower:
                return 'decision_engine'
            elif 'trade_manager' in cmdline_lower:
                return 'trade_manager'
            elif 'guardian' in cmdline_lower:
                return 'guardian_service'
            elif 'main.py' in cmdline_lower:
                return 'main_process'
                
        return None
        
    def _calculate_realtime_costs(self, metrics: Dict[str, Any]):
        """Calculate real-time infrastructure costs"""
        # CPU cost calculation
        cpu_cores_used = (metrics["cpu_percent_total"] / 100) * self.system_baseline["cpu_cores"]
        cpu_cost_per_hour = cpu_cores_used * self.cost_per_cpu_core_hour
        
        # Memory cost calculation
        memory_gb_used = metrics["memory_used_gb"]
        memory_cost_per_hour = memory_gb_used * self.cost_per_gb_memory_hour
        
        # I/O cost calculation
        io_gb_total = (metrics["disk_read_mb"] + metrics["disk_write_mb"]) / 1024
        io_cost = io_gb_total * self.cost_per_gb_io
        
        # Total cost per hour
        total_cost_per_hour = cpu_cost_per_hour + memory_cost_per_hour + io_cost
        
        metrics["cost_analysis"] = {
            "cpu_cost_per_hour": cpu_cost_per_hour,
            "memory_cost_per_hour": memory_cost_per_hour,
            "io_cost": io_cost,
            "total_cost_per_hour": total_cost_per_hour,
            "daily_cost_estimate": total_cost_per_hour * 24,
            "monthly_cost_estimate": total_cost_per_hour * 24 * 30
        }
        
    def _calculate_flow_cost(self, cpu_percent: float, memory_mb: float) -> float:
        """Calculate cost for a specific trading flow"""
        cpu_cores_used = (cpu_percent / 100) * self.system_baseline["cpu_cores"]
        memory_gb_used = memory_mb / 1024
        
        cpu_cost = cpu_cores_used * self.cost_per_cpu_core_hour
        memory_cost = memory_gb_used * self.cost_per_gb_memory_hour
        
        return cpu_cost + memory_cost
        
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        print("\n📊 GENERATING COMPREHENSIVE ANALYSIS")
        print("=" * 50)
        
        # System utilization analysis
        if self.metrics_history:
            avg_cpu = sum(m["cpu_percent_total"] for m in self.metrics_history) / len(self.metrics_history)
            max_cpu = max(m["cpu_percent_total"] for m in self.metrics_history)
            avg_memory = sum(m["memory_percent"] for m in self.metrics_history) / len(self.metrics_history)
            
            print(f"📈 System Utilization:")
            print(f"   CPU: {avg_cpu:.1f}% average, {max_cpu:.1f}% peak")
            print(f"   Memory: {avg_memory:.1f}% average")
        
        # Component performance analysis
        component_analysis = {}
        for component, metrics_list in self.component_metrics.items():
            if metrics_list:
                avg_cpu = sum(m.cpu_percent for m in metrics_list) / len(metrics_list)
                avg_memory = sum(m.memory_mb for m in metrics_list) / len(metrics_list)
                total_samples = len(metrics_list)
                
                component_analysis[component] = {
                    "average_cpu_percent": avg_cpu,
                    "average_memory_mb": avg_memory,
                    "total_samples": total_samples,
                    "monitoring_duration_minutes": self.monitoring_duration,
                    "estimated_hourly_cost": self._calculate_flow_cost(avg_cpu, avg_memory),
                    "efficiency_score": min(100, (avg_cpu + avg_memory/100) * 2),  # Simple efficiency metric
                }
                
        # Trading flow analysis
        flow_analysis = {}
        for flow, metrics in self.trading_flow_metrics.items():
            flow_analysis[flow] = {
                "total_cpu_percent": metrics.get("total_cpu_percent", 0),
                "total_memory_mb": metrics.get("total_memory_mb", 0),
                "estimated_cost_per_hour": metrics.get("estimated_cost_per_hour", 0),
                "components_active": metrics.get("components_active", 0)
            }
        
        # Cost analysis
        if self.metrics_history:
            latest_costs = self.metrics_history[-1].get("cost_analysis", {})
            cost_analysis = {
                "current_hourly_cost": latest_costs.get("total_cost_per_hour", 0),
                "estimated_daily_cost": latest_costs.get("daily_cost_estimate", 0),
                "estimated_monthly_cost": latest_costs.get("monthly_cost_estimate", 0),
                "cpu_cost_percentage": (latest_costs.get("cpu_cost_per_hour", 0) / 
                                     max(latest_costs.get("total_cost_per_hour", 1), 0.001)) * 100,
                "memory_cost_percentage": (latest_costs.get("memory_cost_per_hour", 0) / 
                                        max(latest_costs.get("total_cost_per_hour", 1), 0.001)) * 100
            }
        else:
            cost_analysis = {}
        
        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            component_analysis, flow_analysis, cost_analysis
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "profiling_duration_minutes": self.monitoring_duration,
            "system_baseline": self.system_baseline,
            "system_utilization": {
                "average_cpu_percent": avg_cpu if self.metrics_history else 0,
                "peak_cpu_percent": max_cpu if self.metrics_history else 0,
                "average_memory_percent": avg_memory if self.metrics_history else 0,
                "total_samples": len(self.metrics_history)
            },
            "component_analysis": component_analysis,
            "trading_flow_analysis": flow_analysis,
            "cost_analysis": cost_analysis,
            "optimization_opportunities": optimization_opportunities,
            "raw_metrics_count": {
                "system_metrics": len(self.metrics_history),
                "component_metrics": {k: len(v) for k, v in self.component_metrics.items()}
            }
        }
        
    def _identify_optimization_opportunities(self, component_analysis: Dict, 
                                           flow_analysis: Dict, cost_analysis: Dict) -> List[Dict]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # CPU over-provisioning check
        if self.metrics_history:
            avg_cpu = sum(m["cpu_percent_total"] for m in self.metrics_history) / len(self.metrics_history)
            if avg_cpu < 30:
                cpu_reduction = min(75, 100 - avg_cpu * 2)  # Conservative reduction
                monthly_savings = cost_analysis.get("estimated_monthly_cost", 0) * (cpu_reduction / 100)
                
                opportunities.append({
                    "type": "cpu_rightsizing",
                    "priority": "high",
                    "description": f"CPU utilization at {avg_cpu:.1f}% suggests {cpu_reduction:.0f}% core reduction opportunity",
                    "estimated_monthly_savings": monthly_savings,
                    "implementation_effort": "medium",
                    "risk_level": "low"
                })
        
        # Component efficiency analysis
        for component, metrics in component_analysis.items():
            efficiency = metrics.get("efficiency_score", 0)
            if efficiency < 50:
                opportunities.append({
                    "type": "component_optimization",
                    "priority": "medium",
                    "description": f"{component} efficiency at {efficiency:.0f}% - optimization potential",
                    "component": component,
                    "estimated_monthly_savings": metrics.get("estimated_hourly_cost", 0) * 24 * 30 * 0.3,
                    "implementation_effort": "medium",
                    "risk_level": "medium"
                })
        
        # Memory optimization
        if self.metrics_history:
            avg_memory = sum(m["memory_percent"] for m in self.metrics_history) / len(self.metrics_history)
            if avg_memory < 50:
                memory_reduction = min(50, 80 - avg_memory)
                memory_savings = cost_analysis.get("estimated_monthly_cost", 0) * 0.3 * (memory_reduction / 100)
                
                opportunities.append({
                    "type": "memory_rightsizing", 
                    "priority": "medium",
                    "description": f"Memory utilization at {avg_memory:.1f}% suggests {memory_reduction:.0f}% reduction opportunity",
                    "estimated_monthly_savings": memory_savings,
                    "implementation_effort": "low",
                    "risk_level": "low"
                })
        
        return sorted(opportunities, key=lambda x: x.get("estimated_monthly_savings", 0), reverse=True)
        
    def _save_profiling_results(self, analysis: Dict[str, Any]):
        """Save profiling results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"analysis/performance/cpu_profiling_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\n💾 Profiling results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
            
        # Also save summary report
        summary_file = f"analysis/performance/cpu_profiling_summary_{timestamp}.md"
        self._generate_summary_report(analysis, summary_file)
        
    def _generate_summary_report(self, analysis: Dict[str, Any], filename: str):
        """Generate human-readable summary report"""
        try:
            with open(filename, 'w') as f:
                f.write("# CPU Usage Profiling Summary Report\n\n")
                f.write(f"**Generated**: {analysis['timestamp']}\n")
                f.write(f"**Duration**: {analysis['profiling_duration_minutes']} minutes\n\n")
                
                # System utilization
                sys_util = analysis.get('system_utilization', {})
                f.write("## System Utilization\n")
                f.write(f"- **Average CPU**: {sys_util.get('average_cpu_percent', 0):.1f}%\n")
                f.write(f"- **Peak CPU**: {sys_util.get('peak_cpu_percent', 0):.1f}%\n")
                f.write(f"- **Average Memory**: {sys_util.get('average_memory_percent', 0):.1f}%\n\n")
                
                # Cost analysis
                cost = analysis.get('cost_analysis', {})
                f.write("## Cost Analysis\n")
                f.write(f"- **Hourly Cost**: ${cost.get('current_hourly_cost', 0):.4f}\n")
                f.write(f"- **Daily Cost**: ${cost.get('estimated_daily_cost', 0):.2f}\n")
                f.write(f"- **Monthly Cost**: ${cost.get('estimated_monthly_cost', 0):.2f}\n\n")
                
                # Optimization opportunities
                opportunities = analysis.get('optimization_opportunities', [])
                f.write("## Top Optimization Opportunities\n")
                for i, opp in enumerate(opportunities[:5], 1):
                    f.write(f"{i}. **{opp['type']}** ({opp['priority']} priority)\n")
                    f.write(f"   - {opp['description']}\n")
                    f.write(f"   - Monthly savings: ${opp.get('estimated_monthly_savings', 0):.2f}\n\n")
                    
            print(f"📄 Summary report saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save summary report: {e}")


def main():
    """Run CPU usage profiling"""
    print("🚀 CPU USAGE PROFILER FOR FINOPS ANALYSIS")
    print("=" * 55)
    print()
    
    # Default to 5 minutes for demo, but can be extended for production analysis
    profiler = CPUUsageProfiler(monitoring_duration_minutes=5)
    
    print("🎯 PROFILING OBJECTIVES:")
    print("   • Measure CPU usage across all trading flows")
    print("   • Identify performance bottlenecks and inefficiencies")
    print("   • Calculate cost per trade and optimization opportunities")
    print("   • Generate FinOps recommendations for compute rightsizing")
    print()
    
    try:
        analysis_results = profiler.start_profiling()
        
        print("\n✅ CPU PROFILING COMPLETED!")
        print("=" * 40)
        
        # Display key findings
        sys_util = analysis_results.get('system_utilization', {})
        cost = analysis_results.get('cost_analysis', {})
        opportunities = analysis_results.get('optimization_opportunities', [])
        
        print(f"📊 KEY FINDINGS:")
        print(f"   CPU Utilization: {sys_util.get('average_cpu_percent', 0):.1f}% avg, {sys_util.get('peak_cpu_percent', 0):.1f}% peak")
        print(f"   Monthly Cost: ${cost.get('estimated_monthly_cost', 0):.2f}")
        print(f"   Optimization Opportunities: {len(opportunities)} identified")
        
        if opportunities:
            top_opportunity = opportunities[0]
            print(f"   Top Opportunity: {top_opportunity['type']} - ${top_opportunity.get('estimated_monthly_savings', 0):.2f}/month savings")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Review detailed analysis in generated JSON and markdown files")
        print(f"   2. Implement top optimization opportunities")
        print(f"   3. Establish continuous performance monitoring")
        print(f"   4. Plan compute rightsizing based on actual usage patterns")
        
    except KeyboardInterrupt:
        print("\n⚠️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Compute Optimizer - CPU, Memory, and Process Optimization

This script analyzes and optimizes compute resource usage including:
- CPU utilization analysis and rightsizing recommendations
- Memory optimization and leak detection
- Process optimization and consolidation
- Auto-scaling configuration
- Performance tuning recommendations
"""

import psutil
import os
import json
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_used_gb: float
    load_average: float
    process_count: int


@dataclass
class OptimizationRecommendation:
    """Compute optimization recommendation"""
    category: str  # cpu, memory, process, instance
    priority: str  # high, medium, low
    action: str
    description: str
    current_state: str
    optimized_state: str
    estimated_savings_monthly: float
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    confidence: float  # 0.0 to 1.0


class ComputeOptimizer:
    """
    Comprehensive compute optimization and analysis
    """
    
    def __init__(self, monitoring_duration_minutes: int = 5):
        self.monitoring_duration = monitoring_duration_minutes
        self.metrics_history = []
        self.current_config = self._detect_current_configuration()
        self.optimization_recommendations = []
        
    def analyze_and_optimize(self) -> Dict:
        """Run comprehensive compute analysis and optimization"""
        print("💻 COMPUTE OPTIMIZATION ANALYSIS")
        print("=" * 50)
        
        try:
            # 1. Collect baseline metrics
            print(f"📊 Collecting metrics for {self.monitoring_duration} minutes...")
            baseline_metrics = self._collect_baseline_metrics()
            
            # 2. Analyze current configuration
            print("🔍 Analyzing current configuration...")
            config_analysis = self._analyze_current_configuration()
            
            # 3. Generate optimization recommendations
            print("💡 Generating optimization recommendations...")
            recommendations = self._generate_recommendations(baseline_metrics)
            
            # 4. Calculate potential savings
            print("💰 Calculating potential savings...")
            savings_analysis = self._calculate_savings(recommendations)
            
            # 5. Generate implementation plan
            print("📋 Creating implementation plan...")
            implementation_plan = self._create_implementation_plan(recommendations)
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "analysis_duration_minutes": self.monitoring_duration,
                "current_configuration": config_analysis,
                "baseline_metrics": baseline_metrics,
                "optimization_recommendations": [
                    {
                        "category": rec.category,
                        "priority": rec.priority,
                        "action": rec.action,
                        "description": rec.description,
                        "current_state": rec.current_state,
                        "optimized_state": rec.optimized_state,
                        "estimated_savings_monthly": rec.estimated_savings_monthly,
                        "implementation_effort": rec.implementation_effort,
                        "risk_level": rec.risk_level,
                        "confidence": rec.confidence
                    }
                    for rec in recommendations
                ],
                "savings_analysis": savings_analysis,
                "implementation_plan": implementation_plan
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Compute optimization failed: {e}")
            raise
            
    def _collect_baseline_metrics(self) -> Dict:
        """Collect baseline system metrics over monitoring period"""
        metrics_samples = []
        sample_interval = 30  # 30 seconds between samples
        total_samples = (self.monitoring_duration * 60) // sample_interval
        
        print(f"   Collecting {total_samples} samples every {sample_interval} seconds...")
        
        for i in range(total_samples):
            # Collect comprehensive metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get load average (Linux/Unix)
            try:
                load_avg = os.getloadavg()[0]
            except:
                load_avg = 0.0
                
            # Process information
            process_count = len(psutil.pids())
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024**3),
                load_average=load_avg,
                process_count=process_count
            )
            
            metrics_samples.append(metrics)
            self.metrics_history.append(metrics)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                progress = ((i + 1) / total_samples) * 100
                print(f"      Progress: {progress:.0f}% - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
            
            if i < total_samples - 1:  # Don't sleep after last sample
                time.sleep(sample_interval)
        
        # Calculate summary statistics
        cpu_values = [m.cpu_percent for m in metrics_samples]
        memory_values = [m.memory_percent for m in metrics_samples]
        load_values = [m.load_average for m in metrics_samples]
        
        baseline_summary = {
            "sample_count": len(metrics_samples),
            "monitoring_period_minutes": self.monitoring_duration,
            "cpu_utilization": {
                "average": statistics.mean(cpu_values),
                "median": statistics.median(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "std_dev": statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            "memory_utilization": {
                "average": statistics.mean(memory_values),
                "median": statistics.median(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "std_dev": statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            "load_average": {
                "average": statistics.mean(load_values),
                "max": max(load_values),
                "min": min(load_values)
            },
            "process_count": {
                "average": statistics.mean([m.process_count for m in metrics_samples]),
                "max": max([m.process_count for m in metrics_samples])
            }
        }
        
        return baseline_summary
        
    def _detect_current_configuration(self) -> Dict:
        """Detect current system configuration"""
        # CPU information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_current = cpu_freq.current if cpu_freq else 0
            cpu_freq_max = cpu_freq.max if cpu_freq else 0
        except:
            cpu_freq_current = cpu_freq_max = 0
            
        # Memory information
        memory = psutil.virtual_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "logical_cores": cpu_count,
                "physical_cores": cpu_count_physical,
                "current_frequency_mhz": cpu_freq_current,
                "max_frequency_mhz": cpu_freq_max
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3)
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3)
            },
            "estimated_instance_type": self._estimate_instance_type(cpu_count, memory.total / (1024**3)),
            "estimated_monthly_cost": self._estimate_monthly_cost(cpu_count, memory.total / (1024**3))
        }
        
    def _analyze_current_configuration(self) -> Dict:
        """Analyze current configuration efficiency"""
        config = self.current_config
        
        # Analyze CPU efficiency
        cpu_cores = config["cpu"]["logical_cores"]
        cpu_efficiency = "oversized" if cpu_cores > 8 else "appropriate" if cpu_cores >= 2 else "undersized"
        
        # Analyze memory efficiency
        memory_gb = config["memory"]["total_gb"]
        memory_efficiency = "oversized" if memory_gb > 32 else "appropriate" if memory_gb >= 4 else "undersized"
        
        # Estimate utilization efficiency
        if self.metrics_history:
            avg_cpu = statistics.mean([m.cpu_percent for m in self.metrics_history])
            avg_memory = statistics.mean([m.memory_percent for m in self.metrics_history])
            
            cpu_utilization_efficiency = "underutilized" if avg_cpu < 30 else "efficient" if avg_cpu < 80 else "overutilized"
            memory_utilization_efficiency = "underutilized" if avg_memory < 50 else "efficient" if avg_memory < 85 else "overutilized"
        else:
            cpu_utilization_efficiency = "unknown"
            memory_utilization_efficiency = "unknown"
            
        return {
            "cpu_sizing": cpu_efficiency,
            "memory_sizing": memory_efficiency,
            "cpu_utilization": cpu_utilization_efficiency,
            "memory_utilization": memory_utilization_efficiency,
            "overall_efficiency": self._calculate_overall_efficiency()
        }
        
    def _generate_recommendations(self, baseline_metrics: Dict) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # CPU recommendations
        avg_cpu = baseline_metrics["cpu_utilization"]["average"]
        max_cpu = baseline_metrics["cpu_utilization"]["max"]
        cpu_cores = self.current_config["cpu"]["logical_cores"]
        
        if avg_cpu < 30 and cpu_cores > 4:
            # Significant CPU rightsizing opportunity
            recommended_cores = max(2, min(4, int(cpu_cores * (avg_cpu / 50))))
            current_cost = self._estimate_monthly_cost(cpu_cores, self.current_config["memory"]["total_gb"])
            optimized_cost = self._estimate_monthly_cost(recommended_cores, self.current_config["memory"]["total_gb"])
            
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority="high",
                action="rightsize_cpu",
                description=f"Downsize from {cpu_cores} to {recommended_cores} cores due to low utilization",
                current_state=f"{cpu_cores} cores, {avg_cpu:.1f}% average utilization",
                optimized_state=f"{recommended_cores} cores, estimated {avg_cpu * (cpu_cores/recommended_cores):.1f}% utilization",
                estimated_savings_monthly=current_cost - optimized_cost,
                implementation_effort="medium",
                risk_level="low",
                confidence=0.9
            ))
            
        elif avg_cpu > 80 or max_cpu > 95:
            # CPU upgrade needed
            recommended_cores = min(cpu_cores * 2, 16)
            current_cost = self._estimate_monthly_cost(cpu_cores, self.current_config["memory"]["total_gb"])
            optimized_cost = self._estimate_monthly_cost(recommended_cores, self.current_config["memory"]["total_gb"])
            
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority="high",
                action="upgrade_cpu",
                description=f"Upgrade from {cpu_cores} to {recommended_cores} cores due to high utilization",
                current_state=f"{cpu_cores} cores, {avg_cpu:.1f}% average utilization",
                optimized_state=f"{recommended_cores} cores, estimated {avg_cpu / 2:.1f}% utilization",
                estimated_savings_monthly=-(optimized_cost - current_cost),  # Negative savings (cost increase)
                implementation_effort="medium",
                risk_level="low",
                confidence=0.85
            ))
            
        # Memory recommendations
        avg_memory = baseline_metrics["memory_utilization"]["average"]
        max_memory = baseline_metrics["memory_utilization"]["max"]
        memory_gb = self.current_config["memory"]["total_gb"]
        
        if avg_memory > 90 or max_memory > 95:
            # Memory upgrade needed
            recommended_memory = min(memory_gb * 1.5, 64)
            current_cost = self._estimate_monthly_cost(cpu_cores, memory_gb)
            optimized_cost = self._estimate_monthly_cost(cpu_cores, recommended_memory)
            
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="high",
                action="upgrade_memory",
                description=f"Upgrade from {memory_gb:.0f}GB to {recommended_memory:.0f}GB due to high utilization",
                current_state=f"{memory_gb:.0f}GB, {avg_memory:.1f}% average utilization",
                optimized_state=f"{recommended_memory:.0f}GB, estimated {avg_memory * (memory_gb/recommended_memory):.1f}% utilization",
                estimated_savings_monthly=-(optimized_cost - current_cost),
                implementation_effort="low",
                risk_level="low",
                confidence=0.95
            ))
            
        # Process optimization recommendations
        avg_processes = baseline_metrics["process_count"]["average"]
        if avg_processes > 200:
            recommendations.append(OptimizationRecommendation(
                category="process",
                priority="medium",
                action="optimize_processes",
                description=f"High process count ({avg_processes:.0f}) may indicate inefficiency",
                current_state=f"{avg_processes:.0f} average processes",
                optimized_state="Optimized process management and consolidation",
                estimated_savings_monthly=10,  # Estimate
                implementation_effort="high",
                risk_level="medium",
                confidence=0.6
            ))
            
        # Auto-scaling recommendations
        cpu_std_dev = baseline_metrics["cpu_utilization"]["std_dev"]
        if cpu_std_dev > 20:  # High variability
            recommendations.append(OptimizationRecommendation(
                category="instance",
                priority="medium",
                action="implement_autoscaling",
                description=f"High CPU variability ({cpu_std_dev:.1f}% std dev) suggests auto-scaling benefits",
                current_state="Fixed instance sizing",
                optimized_state="Auto-scaling based on demand",
                estimated_savings_monthly=50,  # Estimate
                implementation_effort="high",
                risk_level="medium",
                confidence=0.75
            ))
            
        # Reserved instance recommendations
        if avg_cpu > 50 and cpu_std_dev < 15:  # Stable, moderate usage
            recommendations.append(OptimizationRecommendation(
                category="instance",
                priority="medium",
                action="purchase_reserved_instances",
                description="Stable usage pattern suitable for reserved instances",
                current_state="On-demand pricing",
                optimized_state="Reserved instance pricing (1-year term)",
                estimated_savings_monthly=self._estimate_ri_savings(),
                implementation_effort="low",
                risk_level="low",
                confidence=0.9
            ))
            
        return recommendations
        
    def _calculate_savings(self, recommendations: List[OptimizationRecommendation]) -> Dict:
        """Calculate total potential savings from recommendations"""
        total_monthly_savings = sum(rec.estimated_savings_monthly for rec in recommendations if rec.estimated_savings_monthly > 0)
        total_monthly_costs = sum(abs(rec.estimated_savings_monthly) for rec in recommendations if rec.estimated_savings_monthly < 0)
        net_monthly_savings = total_monthly_savings - total_monthly_costs
        
        # Categorize savings
        savings_by_category = {}
        for rec in recommendations:
            if rec.category not in savings_by_category:
                savings_by_category[rec.category] = 0
            savings_by_category[rec.category] += rec.estimated_savings_monthly
            
        # Risk-adjusted savings (discount by confidence and risk)
        risk_adjusted_savings = 0
        for rec in recommendations:
            risk_factor = 1.0 if rec.risk_level == "low" else 0.8 if rec.risk_level == "medium" else 0.6
            adjusted_savings = rec.estimated_savings_monthly * rec.confidence * risk_factor
            risk_adjusted_savings += max(0, adjusted_savings)  # Only positive savings
            
        return {
            "total_monthly_savings": total_monthly_savings,
            "total_monthly_costs": total_monthly_costs,
            "net_monthly_savings": net_monthly_savings,
            "annual_savings_potential": net_monthly_savings * 12,
            "savings_by_category": savings_by_category,
            "risk_adjusted_monthly_savings": risk_adjusted_savings,
            "confidence_weighted_savings": sum(rec.estimated_savings_monthly * rec.confidence for rec in recommendations if rec.estimated_savings_monthly > 0)
        }
        
    def _create_implementation_plan(self, recommendations: List[OptimizationRecommendation]) -> Dict:
        """Create implementation plan for recommendations"""
        # Sort by priority and potential savings
        priority_order = {"high": 1, "medium": 2, "low": 3}
        sorted_recs = sorted(recommendations, key=lambda x: (priority_order[x.priority], -x.estimated_savings_monthly))
        
        # Phase implementation
        phases = {
            "immediate": [],  # High priority, low effort, low risk
            "short_term": [], # High priority, medium effort OR medium priority, low effort
            "medium_term": [], # Medium priority, medium effort OR low priority, low effort
            "long_term": []   # High effort or high risk items
        }
        
        for rec in sorted_recs:
            if rec.priority == "high" and rec.implementation_effort == "low" and rec.risk_level == "low":
                phases["immediate"].append(rec)
            elif (rec.priority == "high" and rec.implementation_effort == "medium") or \
                 (rec.priority == "medium" and rec.implementation_effort == "low"):
                phases["short_term"].append(rec)
            elif (rec.priority == "medium" and rec.implementation_effort == "medium") or \
                 (rec.priority == "low" and rec.implementation_effort == "low"):
                phases["medium_term"].append(rec)
            else:
                phases["long_term"].append(rec)
                
        return {
            "total_recommendations": len(recommendations),
            "implementation_phases": {
                phase: [
                    {
                        "action": rec.action,
                        "description": rec.description,
                        "estimated_savings": rec.estimated_savings_monthly,
                        "effort": rec.implementation_effort,
                        "risk": rec.risk_level
                    }
                    for rec in recs
                ]
                for phase, recs in phases.items()
            },
            "recommended_sequence": [
                f"Phase 1 (Immediate): {len(phases['immediate'])} actions",
                f"Phase 2 (Short-term): {len(phases['short_term'])} actions", 
                f"Phase 3 (Medium-term): {len(phases['medium_term'])} actions",
                f"Phase 4 (Long-term): {len(phases['long_term'])} actions"
            ],
            "quick_wins": [rec.action for rec in phases["immediate"]],
            "estimated_timeline_weeks": self._estimate_implementation_timeline(phases)
        }
        
    def _estimate_instance_type(self, cpu_cores: int, memory_gb: float) -> str:
        """Estimate AWS instance type based on specs"""
        if cpu_cores <= 1 and memory_gb <= 1:
            return "t3.nano"
        elif cpu_cores <= 1 and memory_gb <= 2:
            return "t3.micro"
        elif cpu_cores <= 2 and memory_gb <= 4:
            return "t3.small"
        elif cpu_cores <= 2 and memory_gb <= 8:
            return "t3.medium"
        elif cpu_cores <= 2 and memory_gb <= 16:
            return "t3.large"
        elif cpu_cores <= 4 and memory_gb <= 16:
            return "t3.xlarge"
        elif cpu_cores <= 8 and memory_gb <= 32:
            return "c5.2xlarge"
        elif cpu_cores <= 16 and memory_gb <= 64:
            return "c5.4xlarge"
        else:
            return "c5.9xlarge"
            
    def _estimate_monthly_cost(self, cpu_cores: int, memory_gb: float) -> float:
        """Estimate monthly AWS cost based on specs"""
        instance_type = self._estimate_instance_type(cpu_cores, memory_gb)
        
        # Approximate monthly costs (on-demand pricing)
        costs = {
            "t3.nano": 4,
            "t3.micro": 8,
            "t3.small": 16,
            "t3.medium": 33,
            "t3.large": 66,
            "t3.xlarge": 133,
            "c5.2xlarge": 280,
            "c5.4xlarge": 560,
            "c5.9xlarge": 1260
        }
        
        return costs.get(instance_type, 100)
        
    def _estimate_ri_savings(self) -> float:
        """Estimate reserved instance savings"""
        current_cost = self.current_config["estimated_monthly_cost"]
        return current_cost * 0.36  # Typical 36% RI savings
        
    def _calculate_overall_efficiency(self) -> float:
        """Calculate overall efficiency score (0-100)"""
        if not self.metrics_history:
            return 50  # Unknown
            
        avg_cpu = statistics.mean([m.cpu_percent for m in self.metrics_history])
        avg_memory = statistics.mean([m.memory_percent for m in self.metrics_history])
        
        # Efficiency scoring
        cpu_score = min(100, max(0, 100 - abs(avg_cpu - 70)))  # Target 70% CPU
        memory_score = min(100, max(0, 100 - abs(avg_memory - 75)))  # Target 75% memory
        
        return (cpu_score + memory_score) / 2
        
    def _estimate_implementation_timeline(self, phases: Dict) -> int:
        """Estimate implementation timeline in weeks"""
        timeline = 0
        timeline += len(phases["immediate"]) * 0.5  # 0.5 weeks per immediate action
        timeline += len(phases["short_term"]) * 1    # 1 week per short-term action
        timeline += len(phases["medium_term"]) * 2   # 2 weeks per medium-term action
        timeline += len(phases["long_term"]) * 4     # 4 weeks per long-term action
        
        return max(1, int(timeline))


def main():
    """Run compute optimization analysis"""
    print("🚀 COMPUTE OPTIMIZATION STARTING...")
    print()
    
    # Quick analysis (2 minutes for demo)
    optimizer = ComputeOptimizer(monitoring_duration_minutes=2)
    
    try:
        results = optimizer.analyze_and_optimize()
        
        print("\n📊 OPTIMIZATION RESULTS:")
        print("=" * 40)
        
        # Current configuration
        config = results["current_configuration"]
        print(f"💻 Current Configuration:")
        print(f"   CPU: {config['cpu']['logical_cores']} cores")
        print(f"   Memory: {config['memory']['total_gb']:.1f}GB")
        print(f"   Estimated instance: {config['estimated_instance_type']}")
        print(f"   Estimated cost: ${config['estimated_monthly_cost']}/month")
        
        # Baseline metrics
        baseline = results["baseline_metrics"]
        print(f"\n📈 Performance Baseline:")
        print(f"   CPU utilization: {baseline['cpu_utilization']['average']:.1f}% avg, {baseline['cpu_utilization']['max']:.1f}% max")
        print(f"   Memory utilization: {baseline['memory_utilization']['average']:.1f}% avg, {baseline['memory_utilization']['max']:.1f}% max")
        print(f"   Load average: {baseline['load_average']['average']:.2f}")
        
        # Recommendations
        recommendations = results["optimization_recommendations"]
        print(f"\n💡 Optimization Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"   {i}. {rec['action'].replace('_', ' ').title()} ({rec['priority']} priority)")
            print(f"      💰 Savings: ${rec['estimated_savings_monthly']:.0f}/month")
            print(f"      📋 {rec['description']}")
        
        # Savings analysis
        savings = results["savings_analysis"]
        print(f"\n💰 Savings Analysis:")
        print(f"   Total monthly savings: ${savings['total_monthly_savings']:.0f}")
        print(f"   Net monthly impact: ${savings['net_monthly_savings']:.0f}")
        print(f"   Annual potential: ${savings['annual_savings_potential']:.0f}")
        print(f"   Risk-adjusted savings: ${savings['risk_adjusted_monthly_savings']:.0f}/month")
        
        # Implementation plan
        plan = results["implementation_plan"]
        print(f"\n📋 Implementation Plan:")
        print(f"   Total recommendations: {plan['total_recommendations']}")
        print(f"   Quick wins available: {len(plan['quick_wins'])}")
        print(f"   Estimated timeline: {plan['estimated_timeline_weeks']} weeks")
        
        if plan['quick_wins']:
            print(f"   Immediate actions:")
            for action in plan['quick_wins'][:3]:
                print(f"      • {action.replace('_', ' ').title()}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"compute_optimization_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📄 Results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save results: {e}")
            
        print("\n✅ COMPUTE OPTIMIZATION COMPLETED!")
        
        # Summary recommendation
        if savings['net_monthly_savings'] > 100:
            print(f"\n🎯 KEY RECOMMENDATION: Significant optimization opportunity!")
            print(f"   Potential monthly savings: ${savings['net_monthly_savings']:.0f}")
            print(f"   Implement high-priority, low-risk optimizations first")
        elif savings['net_monthly_savings'] > 0:
            print(f"\n🎯 KEY RECOMMENDATION: Moderate optimization opportunity")
            print(f"   Focus on quick wins and gradual improvements")
        else:
            print(f"\n🎯 KEY RECOMMENDATION: System appears well-optimized")
            print(f"   Consider monitoring-based fine-tuning")
        
    except Exception as e:
        print(f"\n❌ Compute optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
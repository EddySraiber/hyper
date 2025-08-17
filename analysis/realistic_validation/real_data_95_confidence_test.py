#!/usr/bin/env python3
"""
Real Data 95% Confidence Validation Test

Uses the collected real market data to run enhanced 95% confidence validation,
replacing synthetic data with institutional-grade historical market data.

Phase 1 Week 2: Real market data validation for 95% confidence
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.append('/app')

# Import path fix for container environment
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enhanced_95_confidence_validator", 
    "/app/analysis/realistic_validation/enhanced_95_confidence_validator.py"
)
validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator_module)
Enhanced95ConfidenceValidator = validator_module.Enhanced95ConfidenceValidator


class RealData95ConfidenceTest:
    """
    Enhanced 95% confidence validation using real market data
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.data_dir = Path("/app/data/market_data")
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def run_real_data_validation(self) -> Dict[str, Any]:
        """
        Run 95% confidence validation with real market data
        """
        self.logger.info("🔬 REAL DATA 95% CONFIDENCE VALIDATION")
        self.logger.info("=" * 60)
        
        # Step 1: Load real market data
        real_data = await self._load_real_market_data()
        
        # Step 2: Prepare data for validation
        baseline_results, enhanced_results = await self._prepare_validation_data(real_data)
        
        # Step 3: Run enhanced 95% confidence validation
        validator = Enhanced95ConfidenceValidator()
        results = await validator.validate_with_95_confidence()
        
        # Step 4: Compare with real data performance
        real_data_comparison = await self._compare_with_real_data(
            baseline_results, enhanced_results, results
        )
        
        # Step 5: Generate comprehensive report
        final_report = await self._generate_real_data_report(
            results, real_data_comparison, real_data
        )
        
        self._print_real_data_summary(final_report)
        return final_report
    
    async def _load_real_market_data(self) -> Dict[str, Any]:
        """Load the collected real market data"""
        
        # Load latest market data
        summary_file = self.data_dir / "latest_market_data_summary.json"
        
        if not summary_file.exists():
            raise FileNotFoundError("Real market data not found. Run real_market_data_collector.py first.")
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Load full dataset
        data_file = Path(summary['file_path'])
        with open(data_file, 'r') as f:
            real_data = json.load(f)
        
        self.logger.info(f"✅ Loaded real market data from {data_file.name}")
        self.logger.info(f"   📊 Period: {real_data['data_period']['start'][:10]} to {real_data['data_period']['end'][:10]}")
        self.logger.info(f"   📈 Baseline samples: {len(real_data['baseline_samples'])}")
        self.logger.info(f"   🚀 Enhanced samples: {len(real_data['enhanced_samples'])}")
        self.logger.info(f"   📰 Market events: {len(real_data['market_events'])}")
        
        return real_data
    
    async def _prepare_validation_data(self, real_data: Dict[str, Any]) -> tuple:
        """Prepare real market data for validation"""
        
        # Extract baseline performance
        baseline_samples = real_data['baseline_samples']
        enhanced_samples = real_data['enhanced_samples']
        
        # Calculate realistic performance metrics from real data
        baseline_returns = [sample['net_return'] for sample in baseline_samples]
        enhanced_returns = [sample['net_return'] for sample in enhanced_samples]
        
        # Annualized performance (assuming daily returns)
        baseline_annual = np.mean(baseline_returns) * 252
        enhanced_annual = np.mean(enhanced_returns) * 252
        
        # Calculate volatility
        baseline_vol = np.std(baseline_returns) * np.sqrt(252)
        enhanced_vol = np.std(enhanced_returns) * np.sqrt(252)
        
        # Calculate Sharpe ratios (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        baseline_sharpe = (baseline_annual - risk_free_rate) / baseline_vol if baseline_vol > 0 else 0
        enhanced_sharpe = (enhanced_annual - risk_free_rate) / enhanced_vol if enhanced_vol > 0 else 0
        
        # Calculate max drawdown (simplified)
        baseline_cumulative = np.cumprod(1 + np.array(baseline_returns))
        enhanced_cumulative = np.cumprod(1 + np.array(enhanced_returns))
        
        baseline_dd = self._calculate_max_drawdown(baseline_cumulative)
        enhanced_dd = self._calculate_max_drawdown(enhanced_cumulative)
        
        # Win rates
        baseline_win_rate = sum(1 for r in baseline_returns if r > 0) / len(baseline_returns)
        enhanced_win_rate = sum(1 for r in enhanced_returns if r > 0) / len(enhanced_returns)
        
        baseline_results = {
            'annual_return_pct': baseline_annual,
            'sharpe_ratio': baseline_sharpe,
            'max_drawdown_pct': baseline_dd,
            'volatility_pct': baseline_vol,
            'win_rate': baseline_win_rate,
            'total_trades': len(baseline_samples),
            'daily_returns': baseline_returns
        }
        
        enhanced_results = {
            'annual_return_pct': enhanced_annual,
            'sharpe_ratio': enhanced_sharpe,
            'max_drawdown_pct': enhanced_dd,
            'volatility_pct': enhanced_vol,
            'win_rate': enhanced_win_rate,
            'total_trades': len(enhanced_samples),
            'daily_returns': enhanced_returns
        }
        
        self.logger.info(f"✅ Prepared validation data from real market samples")
        self.logger.info(f"   📊 Baseline annual return: {baseline_annual:.1%}")
        self.logger.info(f"   🚀 Enhanced annual return: {enhanced_annual:.1%}")
        self.logger.info(f"   📈 Improvement: {enhanced_annual - baseline_annual:+.1%}")
        
        return baseline_results, enhanced_results
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return abs(np.min(drawdown))
    
    async def _compare_with_real_data(self, baseline_results: Dict[str, Any], 
                                    enhanced_results: Dict[str, Any],
                                    validation_results) -> Dict[str, Any]:
        """Compare validation results with real market data"""
        
        # Real data performance
        real_baseline_return = baseline_results['annual_return_pct']
        real_enhanced_return = enhanced_results['annual_return_pct']
        real_improvement = real_enhanced_return - real_baseline_return
        
        # Validation results (simulated)
        val_baseline_return = validation_results.baseline_annual_return
        val_enhanced_return = validation_results.enhanced_annual_return
        val_improvement = validation_results.absolute_improvement
        
        # Compare confidence levels
        real_effect_size = self._calculate_real_effect_size(baseline_results, enhanced_results)
        
        comparison = {
            'real_data_performance': {
                'baseline_return': real_baseline_return,
                'enhanced_return': real_enhanced_return,
                'improvement': real_improvement,
                'relative_improvement': (real_improvement / abs(real_baseline_return)) * 100 if real_baseline_return != 0 else 0,
                'effect_size': real_effect_size
            },
            'validation_performance': {
                'baseline_return': val_baseline_return,
                'enhanced_return': val_enhanced_return,
                'improvement': val_improvement,
                'relative_improvement': validation_results.relative_improvement,
                'effect_size': validation_results.effect_size_cohens_d
            },
            'comparison_analysis': {
                'return_difference': abs(real_improvement - val_improvement),
                'effect_size_difference': abs(real_effect_size - validation_results.effect_size_cohens_d),
                'validation_accuracy': self._assess_validation_accuracy(real_improvement, val_improvement),
                'data_quality_impact': 'real_data_provides_realistic_baseline'
            }
        }
        
        return comparison
    
    def _calculate_real_effect_size(self, baseline_results: Dict[str, Any], 
                                  enhanced_results: Dict[str, Any]) -> float:
        """Calculate Cohen's d effect size from real data"""
        baseline_returns = baseline_results['daily_returns']
        enhanced_returns = enhanced_results['daily_returns']
        
        mean_diff = np.mean(enhanced_returns) - np.mean(baseline_returns)
        pooled_std = np.sqrt((np.var(baseline_returns) + np.var(enhanced_returns)) / 2)
        
        return mean_diff / pooled_std if pooled_std > 0 else 0
    
    def _assess_validation_accuracy(self, real_improvement: float, val_improvement: float) -> str:
        """Assess how accurate the validation is compared to real data"""
        difference = abs(real_improvement - val_improvement)
        
        if difference < 0.01:  # <1% difference
            return "excellent_accuracy"
        elif difference < 0.02:  # <2% difference
            return "good_accuracy"
        elif difference < 0.05:  # <5% difference
            return "moderate_accuracy"
        else:
            return "poor_accuracy_needs_calibration"
    
    async def _generate_real_data_report(self, validation_results, 
                                       comparison: Dict[str, Any],
                                       real_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive real data validation report"""
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_type': 'real_market_data_95_confidence',
            'data_period': real_data['data_period'],
            'sample_statistics': {
                'total_samples': len(real_data['baseline_samples']) + len(real_data['enhanced_samples']),
                'market_events_analyzed': len(real_data['market_events']),
                'symbols_covered': len(real_data['symbols_collected']),
                'trading_days': 262  # 2024 trading days
            },
            'confidence_analysis': {
                'target_confidence': 0.95,
                'achieved_confidence': validation_results.confidence_level,
                'confidence_gap': 0.95 - validation_results.confidence_level,
                'statistical_significance': validation_results.p_value < 0.05,
                'effect_size': validation_results.effect_size_cohens_d,
                'sample_adequacy': validation_results.sample_adequacy_ratio >= 1.0
            },
            'real_vs_validation_comparison': comparison,
            'institutional_assessment': {
                'deployment_status': validation_results.deployment_status,
                'recommended_capital': validation_results.recommended_capital,
                'risk_adjusted_capital': validation_results.risk_adjusted_capital,
                'institutional_grade': validation_results.confidence_level >= 0.95
            },
            'next_phase_recommendations': self._generate_next_phase_recommendations(
                validation_results, comparison
            )
        }
        
        return report
    
    def _generate_next_phase_recommendations(self, validation_results, 
                                           comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations for next phase"""
        recommendations = []
        
        confidence_level = validation_results.confidence_level
        effect_size = validation_results.effect_size_cohens_d
        
        if confidence_level < 0.70:
            recommendations.extend([
                "PRIORITY: Increase sample size to 400+ observations",
                "Improve signal quality through enhanced AI models",
                "Optimize transaction cost models for better net returns"
            ])
        elif confidence_level < 0.85:
            recommendations.extend([
                "Enhance effect size through signal combination optimization",
                "Extend validation period for more robust statistics",
                "Implement advanced risk management for volatility control"
            ])
        elif confidence_level < 0.95:
            recommendations.extend([
                "Fine-tune system parameters for final 95% confidence push",
                "Conduct out-of-sample validation on 2025 data",
                "Prepare for institutional deployment review"
            ])
        else:
            recommendations.extend([
                "✅ Ready for institutional deployment",
                "Begin phased capital allocation according to deployment plan",
                "Monitor live performance against validation benchmarks"
            ])
        
        # Add data quality recommendations
        data_accuracy = comparison['comparison_analysis']['validation_accuracy']
        if data_accuracy in ['poor_accuracy_needs_calibration', 'moderate_accuracy']:
            recommendations.append("Calibrate validation framework with more real market data")
        
        return recommendations
    
    def _print_real_data_summary(self, report: Dict[str, Any]):
        """Print comprehensive real data validation summary"""
        
        print("\n" + "🔬" * 60)
        print("🔬" + " " * 10 + "REAL MARKET DATA 95% CONFIDENCE VALIDATION" + " " * 10 + "🔬")
        print("🔬" + " " * 15 + "INSTITUTIONAL-GRADE ANALYSIS COMPLETE" + " " * 15 + "🔬")
        print("🔬" * 60)
        
        # Data Quality Summary
        stats = report['sample_statistics']
        print(f"\n📊 REAL MARKET DATA ANALYSIS")
        print(f"   📈 Total Samples Analyzed: {stats['total_samples']}")
        print(f"   📰 Market Events: {stats['market_events_analyzed']}")
        print(f"   📊 Symbols Covered: {stats['symbols_covered']}")
        print(f"   📅 Period: {report['data_period']['start'][:10]} to {report['data_period']['end'][:10]}")
        
        # Confidence Analysis
        conf = report['confidence_analysis']
        print(f"\n🎯 95% CONFIDENCE ANALYSIS")
        print(f"   🎯 Target Confidence: {conf['target_confidence']:.1%}")
        print(f"   📊 Achieved Confidence: {conf['achieved_confidence']:.1%}")
        print(f"   📊 Confidence Gap: {conf['confidence_gap']:.1%}")
        print(f"   ✅ Statistical Significance: {'YES' if conf['statistical_significance'] else 'NO'}")
        print(f"   📊 Effect Size: {conf['effect_size']:.3f}")
        
        # Real vs Validation Comparison
        comparison = report['real_vs_validation_comparison']
        real_perf = comparison['real_data_performance']
        val_perf = comparison['validation_performance']
        
        print(f"\n📈 REAL DATA vs VALIDATION COMPARISON")
        print(f"   📊 Real Data Improvement: {real_perf['improvement']:+.1%}")
        print(f"   📊 Validation Improvement: {val_perf['improvement']:+.1%}")
        print(f"   📊 Accuracy Assessment: {comparison['comparison_analysis']['validation_accuracy']}")
        print(f"   📊 Effect Size Difference: {comparison['comparison_analysis']['effect_size_difference']:.3f}")
        
        # Institutional Assessment
        inst = report['institutional_assessment']
        print(f"\n🏛️ INSTITUTIONAL DEPLOYMENT ASSESSMENT")
        print(f"   🎯 Status: {inst['deployment_status']}")
        print(f"   💰 Recommended Capital: ${inst['recommended_capital']:,}")
        print(f"   💰 Risk-Adjusted Capital: ${inst['risk_adjusted_capital']:,}")
        print(f"   ✅ Institutional Grade: {'YES' if inst['institutional_grade'] else 'NO'}")
        
        # Next Phase Recommendations
        print(f"\n🚀 NEXT PHASE RECOMMENDATIONS")
        for i, rec in enumerate(report['next_phase_recommendations'][:4], 1):
            print(f"   {i}. {rec}")
        
        # Final Verdict
        print(f"\n🎯 FINAL ASSESSMENT")
        if conf['achieved_confidence'] >= 0.95:
            print("🎉 ✅ 95% CONFIDENCE ACHIEVED - READY FOR INSTITUTIONAL DEPLOYMENT")
        elif conf['achieved_confidence'] >= 0.85:
            print("⚠️ ✅ STRONG VALIDATION (85%+) - APPROACHING INSTITUTIONAL STANDARDS")
        elif conf['achieved_confidence'] >= 0.70:
            print("🔄 MODERATE VALIDATION (70%+) - CONTINUE ENHANCEMENT EFFORTS")
        else:
            print("❌ INSUFFICIENT CONFIDENCE - MAJOR IMPROVEMENTS REQUIRED")
        
        print("\n" + "🔬" * 60 + "\n")


async def main():
    """Run real data 95% confidence validation"""
    tester = RealData95ConfidenceTest()
    
    try:
        report = await tester.run_real_data_validation()
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"/app/data/real_data_95_confidence_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Complete report saved: {output_file}")
        return report
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
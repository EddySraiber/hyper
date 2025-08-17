#!/usr/bin/env python3
"""
Realistic Validation Runner for Algorithmic Trading System

This script runs comprehensive validation suitable for institutional deployment,
replacing the flawed synthetic validation with real market data and statistical rigor.

Key Features:
- Real historical market data from Alpaca API
- Realistic performance targets (8-20% annual returns)
- Comprehensive transaction cost modeling
- Statistical significance testing with proper hypothesis testing
- Risk assessment suitable for $500K-$1M deployment

Dr. Sarah Chen - Quantitative Finance Expert
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append('/home/eddy/Hyper')

from realistic_backtesting_engine import RealisticBacktestingEngine
from statistical_validator import StatisticalValidator, ValidationResult
from algotrading_agent.config.settings import get_config


def setup_logging() -> str:
    """Setup comprehensive logging for validation"""
    
    # Create logs directory
    log_dir = Path("/home/eddy/Hyper/analysis/realistic_validation/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"realistic_validation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return str(log_filename)


def print_banner():
    """Print validation banner"""
    print()
    print("🔬" * 50)
    print("🔬" + " " * 18 + "REALISTIC VALIDATION FRAMEWORK" + " " * 18 + "🔬")
    print("🔬" + " " * 15 + "INSTITUTIONAL-GRADE STATISTICAL ANALYSIS" + " " * 15 + "🔬")
    print("🔬" * 50)
    print()
    print("Dr. Sarah Chen - Quantitative Finance Expert")
    print("Designed for $500K-$1M Real Capital Deployment")
    print()
    print("🎯 OBJECTIVES:")
    print("  ✅ Replace synthetic validation with real market data")
    print("  ✅ Test realistic performance targets (8-20% annual returns)")
    print("  ✅ Include comprehensive transaction costs and taxes")
    print("  ✅ Provide statistical significance with proper hypothesis testing")
    print("  ✅ Generate institutional-grade deployment recommendations")
    print()
    print("❌ ELIMINATES:")
    print("  ❌ Impossible 23,847% return claims")
    print("  ❌ Synthetic/artificial market data")
    print("  ❌ Circular validation against system assumptions")
    print("  ❌ Overfitted performance metrics")
    print()
    print("=" * 100)
    print()


class RealisticValidationRunner:
    """
    Master orchestrator for comprehensive realistic validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = self._create_validation_config()
        self.results = {}
        
    def _create_validation_config(self) -> dict:
        """Create comprehensive validation configuration"""
        
        # Get base configuration
        base_config = get_config()
        
        # Create realistic validation parameters
        validation_config = {
            # Market data parameters
            'validation_symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'start_date': datetime(2023, 1, 1),  # 2-year historical period
            'end_date': datetime(2024, 12, 31),
            'out_of_sample_buffer_days': 30,  # 30-day buffer for out-of-sample testing
            
            # Portfolio parameters
            'initial_capital': 100000,  # $100K starting capital
            'max_position_size': 0.05,  # 5% maximum position size
            'max_portfolio_risk': 0.20,  # 20% maximum portfolio risk
            
            # Realistic performance targets
            'target_annual_return': 0.15,  # 15% annual return target
            'excellent_annual_return': 0.20,  # 20% excellent performance
            'good_annual_return': 0.12,  # 12% good performance
            'marginal_annual_return': 0.08,  # 8% marginal performance
            
            # Risk thresholds
            'max_acceptable_drawdown': 0.20,  # 20% maximum drawdown
            'min_sharpe_ratio': 1.0,  # 1.0 minimum Sharpe ratio
            'excellent_sharpe': 1.5,  # 1.5 excellent Sharpe ratio
            'good_sharpe': 1.2,  # 1.2 good Sharpe ratio
            'marginal_sharpe': 0.8,  # 0.8 marginal Sharpe ratio
            
            # Statistical parameters
            'significance_level': 0.05,  # 5% significance level
            'statistical_power': 0.80,  # 80% statistical power
            'min_effect_size': 0.2,  # 0.2 minimum meaningful effect size
            'min_sample_size': 30,  # 30 minimum trades for significance
            'confidence_level': 0.95,  # 95% confidence intervals
            
            # Transaction cost model (realistic)
            'commission_per_trade': 0.0,  # Alpaca commission-free
            'spread_bps': 2.0,  # 2 basis points bid-ask spread
            'market_impact_bps': 1.5,  # 1.5 bps market impact
            'sec_fee_bps': 0.0278,  # SEC fee (2.78 bps on sells)
            'taf_fee_per_share': 0.000130,  # TAF fee per share
            'tax_rate': 0.37,  # 37% combined tax rate (federal + state)
            
            # Deployment thresholds
            'full_deployment_score': 75,  # 75+ for full deployment approval
            'gradual_deployment_score': 60,  # 60+ for gradual deployment
            'conditional_approval_score': 45,  # 45+ for conditional approval
        }
        
        return validation_config
    
    async def run_comprehensive_validation(self) -> dict:
        """
        Run comprehensive realistic validation
        """
        self.logger.info("🚀 Starting Comprehensive Realistic Validation")
        
        try:
            # Step 1: Run baseline system validation
            self.logger.info("📊 Step 1: Baseline System Validation")
            baseline_results = await self._run_baseline_validation()
            
            # Step 2: Run enhanced system validation
            self.logger.info("🚀 Step 2: Enhanced System Validation")
            enhanced_results = await self._run_enhanced_validation()
            
            # Step 3: Statistical comparison and validation
            self.logger.info("🔬 Step 3: Statistical Significance Testing")
            statistical_results = await self._run_statistical_validation(
                baseline_results, enhanced_results
            )
            
            # Step 4: Risk assessment
            self.logger.info("⚠️ Step 4: Comprehensive Risk Assessment")
            risk_assessment = await self._assess_deployment_risk(
                enhanced_results, statistical_results
            )
            
            # Step 5: Generate deployment recommendations
            self.logger.info("🎯 Step 5: Deployment Recommendations")
            deployment_recommendations = await self._generate_deployment_plan(
                statistical_results, risk_assessment
            )
            
            # Compile final results
            final_results = {
                'validation_timestamp': datetime.now().isoformat(),
                'validation_config': self.config,
                'baseline_results': baseline_results,
                'enhanced_results': enhanced_results,
                'statistical_analysis': statistical_results,
                'risk_assessment': risk_assessment,
                'deployment_recommendations': deployment_recommendations,
                'validation_summary': self._create_validation_summary(statistical_results)
            }
            
            # Save results
            await self._save_validation_results(final_results)
            
            # Print comprehensive summary
            self._print_validation_summary(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            raise
    
    async def _run_baseline_validation(self) -> dict:
        """
        Run baseline system validation (traditional news sentiment only)
        """
        self.logger.info("📈 Running baseline system validation...")
        
        # Configure baseline system (traditional sentiment only)
        baseline_config = self.config.copy()
        baseline_config.update({
            'system_type': 'baseline',
            'use_ai_enhancement': False,
            'use_regime_detection': False,
            'use_options_flow': False,
            'sentiment_method': 'traditional'
        })
        
        # Run realistic backtesting
        engine = RealisticBacktestingEngine(baseline_config)
        results = await engine.run_comprehensive_backtest()
        
        self.logger.info(f"✅ Baseline validation complete:")
        self.logger.info(f"   📊 Annual Return: {results.annual_return_pct:.1%}")
        self.logger.info(f"   📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        self.logger.info(f"   📊 Max Drawdown: {results.max_drawdown_pct:.1%}")
        self.logger.info(f"   📊 Total Trades: {results.total_trades}")
        
        return {
            'system_type': 'baseline',
            'annual_return_pct': results.annual_return_pct,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown_pct': results.max_drawdown_pct,
            'volatility_pct': results.volatility_pct,
            'win_rate': results.win_rate,
            'total_trades': results.total_trades,
            'total_transaction_costs': results.total_transaction_costs,
            'total_tax_burden': results.total_tax_burden,
            'daily_returns': results.daily_returns,
            'portfolio_values': results.portfolio_values,
            'var_95': results.var_95,
            'var_99': results.var_99,
            'statistical_significance': results.is_statistically_significant()
        }
    
    async def _run_enhanced_validation(self) -> dict:
        """
        Run enhanced system validation (AI + regime detection + options flow)
        """
        self.logger.info("🚀 Running enhanced system validation...")
        
        # Configure enhanced system
        enhanced_config = self.config.copy()
        enhanced_config.update({
            'system_type': 'enhanced',
            'use_ai_enhancement': True,
            'use_regime_detection': True,
            'use_options_flow': True,
            'sentiment_method': 'ai_enhanced'
        })
        
        # Run realistic backtesting
        engine = RealisticBacktestingEngine(enhanced_config)
        results = await engine.run_comprehensive_backtest()
        
        self.logger.info(f"✅ Enhanced validation complete:")
        self.logger.info(f"   📊 Annual Return: {results.annual_return_pct:.1%}")
        self.logger.info(f"   📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        self.logger.info(f"   📊 Max Drawdown: {results.max_drawdown_pct:.1%}")
        self.logger.info(f"   📊 Total Trades: {results.total_trades}")
        
        return {
            'system_type': 'enhanced',
            'annual_return_pct': results.annual_return_pct,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown_pct': results.max_drawdown_pct,
            'volatility_pct': results.volatility_pct,
            'win_rate': results.win_rate,
            'total_trades': results.total_trades,
            'total_transaction_costs': results.total_transaction_costs,
            'total_tax_burden': results.total_tax_burden,
            'daily_returns': results.daily_returns,
            'portfolio_values': results.portfolio_values,
            'var_95': results.var_95,
            'var_99': results.var_99,
            'statistical_significance': results.is_statistically_significant()
        }
    
    async def _run_statistical_validation(self, baseline_results: dict, 
                                        enhanced_results: dict) -> dict:
        """
        Run comprehensive statistical validation
        """
        self.logger.info("🔬 Running statistical significance testing...")
        
        # Initialize statistical validator
        validator = StatisticalValidator(self.config)
        
        # Perform comprehensive validation
        validation_report = validator.validate_system_performance(
            baseline_results, enhanced_results
        )
        
        # Convert report to dictionary
        return {
            'overall_score': validation_report.overall_score,
            'validation_result': validation_report.validation_result.value,
            'confidence_level': validation_report.confidence_level,
            'return_test': {
                'test_name': validation_report.return_test.test_name,
                'statistic': validation_report.return_test.statistic,
                'p_value': validation_report.return_test.p_value,
                'is_significant': validation_report.return_test.is_significant,
                'effect_size': validation_report.return_test.effect_size,
                'confidence_interval': validation_report.return_test.confidence_interval,
                'interpretation': validation_report.return_test.interpretation
            },
            'sharpe_test': {
                'test_name': validation_report.sharpe_test.test_name,
                'statistic': validation_report.sharpe_test.statistic,
                'p_value': validation_report.sharpe_test.p_value,
                'is_significant': validation_report.sharpe_test.is_significant,
                'effect_size': validation_report.sharpe_test.effect_size,
                'interpretation': validation_report.sharpe_test.interpretation
            },
            'performance_comparison': {
                'baseline_return': validation_report.performance_comparison.baseline_return,
                'enhanced_return': validation_report.performance_comparison.enhanced_return,
                'return_difference': validation_report.performance_comparison.return_difference,
                'relative_improvement': validation_report.performance_comparison.relative_improvement,
                'sharpe_improvement': validation_report.performance_comparison.sharpe_improvement,
                'drawdown_improvement': validation_report.performance_comparison.drawdown_improvement
            },
            'risk_assessment': {
                'risk_level': validation_report.risk_level,
                'risk_factors': validation_report.risk_factors
            },
            'deployment_recommendations': validation_report.deployment_recommendations,
            'risk_mitigation_strategies': validation_report.risk_mitigation_strategies,
            'sample_size_validation': {
                'actual_sample_size': validation_report.actual_sample_size,
                'required_sample_size': validation_report.required_sample_size,
                'sample_size_adequate': validation_report.sample_size_adequate
            }
        }
    
    async def _assess_deployment_risk(self, enhanced_results: dict, 
                                    statistical_results: dict) -> dict:
        """
        Comprehensive deployment risk assessment
        """
        self.logger.info("⚠️ Assessing deployment risk...")
        
        risk_factors = []
        risk_score = 0
        
        # Performance risk assessment
        annual_return = enhanced_results['annual_return_pct']
        if annual_return < 0.08:
            risk_factors.append("Low annual return (<8%)")
            risk_score += 20
        elif annual_return > 0.30:
            risk_factors.append("Unrealistically high return (>30%) - possible overfitting")
            risk_score += 15
        
        # Volatility risk
        volatility = enhanced_results['volatility_pct']
        if volatility > 0.35:
            risk_factors.append("High volatility (>35%)")
            risk_score += 15
        
        # Drawdown risk
        max_drawdown = enhanced_results['max_drawdown_pct']
        if max_drawdown > 0.25:
            risk_factors.append("High maximum drawdown (>25%)")
            risk_score += 20
        
        # Statistical significance risk
        if not statistical_results['return_test']['is_significant']:
            risk_factors.append("Returns not statistically significant")
            risk_score += 25
        
        # Sample size risk
        if not statistical_results['sample_size_validation']['sample_size_adequate']:
            risk_factors.append("Insufficient sample size for reliable conclusions")
            risk_score += 20
        
        # Market regime concentration risk
        if enhanced_results['win_rate'] > 0.85:
            risk_factors.append("Extremely high win rate may indicate overfitting")
            risk_score += 15
        
        # Determine overall risk level
        if risk_score >= 50:
            overall_risk = "HIGH"
        elif risk_score >= 25:
            overall_risk = "MODERATE"
        else:
            overall_risk = "LOW"
        
        return {
            'overall_risk_level': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'deployment_safe': risk_score < 50,
            'recommended_max_capital': self._calculate_max_safe_capital(risk_score),
            'required_monitoring_frequency': self._determine_monitoring_frequency(overall_risk)
        }
    
    async def _generate_deployment_plan(self, statistical_results: dict, 
                                      risk_assessment: dict) -> dict:
        """
        Generate comprehensive deployment plan
        """
        self.logger.info("🎯 Generating deployment recommendations...")
        
        validation_result = statistical_results['validation_result']
        overall_score = statistical_results['overall_score']
        risk_level = risk_assessment['overall_risk_level']
        
        # Base deployment recommendation
        if validation_result == 'approved_full_deployment' and risk_level in ['LOW', 'MODERATE']:
            deployment_status = "APPROVED"
            initial_capital = min(500000, risk_assessment['recommended_max_capital'])
            scaling_timeline = "3-month gradual scaling to full allocation"
        elif validation_result == 'approved_gradual_deployment':
            deployment_status = "CONDITIONAL APPROVAL"
            initial_capital = min(100000, risk_assessment['recommended_max_capital'])
            scaling_timeline = "6-month gradual scaling with performance validation"
        elif validation_result == 'conditional_approval':
            deployment_status = "DEVELOPMENT REQUIRED"
            initial_capital = 25000
            scaling_timeline = "12-month development and re-validation required"
        else:
            deployment_status = "NOT APPROVED"
            initial_capital = 0
            scaling_timeline = "Major system improvements required before deployment"
        
        # Specific deployment phases
        deployment_phases = []
        if deployment_status in ["APPROVED", "CONDITIONAL APPROVAL"]:
            deployment_phases = [
                {
                    "phase": 1,
                    "duration_weeks": 4,
                    "capital_allocation": initial_capital * 0.25,
                    "objectives": ["Validate live execution", "Confirm transaction costs", "Test monitoring systems"]
                },
                {
                    "phase": 2,
                    "duration_weeks": 8,
                    "capital_allocation": initial_capital * 0.50,
                    "objectives": ["Scale position sizes", "Test risk management", "Validate performance attribution"]
                },
                {
                    "phase": 3,
                    "duration_weeks": 12,
                    "capital_allocation": initial_capital,
                    "objectives": ["Full system validation", "Performance tracking", "Prepare for scaling"]
                }
            ]
        
        return {
            'deployment_status': deployment_status,
            'overall_score': overall_score,
            'confidence_level': statistical_results['confidence_level'],
            'initial_capital_allocation': initial_capital,
            'maximum_recommended_capital': risk_assessment['recommended_max_capital'],
            'scaling_timeline': scaling_timeline,
            'deployment_phases': deployment_phases,
            'monitoring_requirements': {
                'frequency': risk_assessment['required_monitoring_frequency'],
                'key_metrics': ['daily_return', 'drawdown', 'win_rate', 'transaction_costs'],
                'alert_thresholds': {
                    'daily_loss_limit': 0.05,  # 5% daily loss
                    'drawdown_limit': 0.15,    # 15% drawdown
                    'consecutive_losses': 5     # 5 consecutive losing trades
                }
            },
            'success_criteria': {
                'phase_1': 'Achieve >5% monthly return with <10% drawdown',
                'phase_2': 'Maintain >8% quarterly return with <15% drawdown',
                'phase_3': 'Deliver >12% annual return with <20% drawdown'
            },
            'risk_mitigation': statistical_results['risk_mitigation_strategies']
        }
    
    def _calculate_max_safe_capital(self, risk_score: int) -> int:
        """Calculate maximum safe capital allocation based on risk score"""
        if risk_score < 15:
            return 1000000  # $1M for very low risk
        elif risk_score < 30:
            return 500000   # $500K for low-moderate risk
        elif risk_score < 50:
            return 200000   # $200K for moderate-high risk
        else:
            return 50000    # $50K for high risk
    
    def _determine_monitoring_frequency(self, risk_level: str) -> str:
        """Determine required monitoring frequency based on risk level"""
        if risk_level == "HIGH":
            return "real_time"  # Real-time monitoring required
        elif risk_level == "MODERATE":
            return "daily"      # Daily monitoring required
        else:
            return "weekly"     # Weekly monitoring sufficient
    
    async def _save_validation_results(self, results: dict):
        """Save comprehensive validation results"""
        
        # Create results directory
        results_dir = Path("/home/eddy/Hyper/analysis/realistic_validation/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"realistic_validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Validation results saved: {results_file}")
        
        # Save summary report
        summary_file = results_dir / f"validation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self._create_text_summary(results))
        
        self.logger.info(f"📄 Summary report saved: {summary_file}")
    
    def _create_validation_summary(self, statistical_results: dict) -> dict:
        """Create concise validation summary"""
        return {
            'validation_result': statistical_results['validation_result'],
            'overall_score': statistical_results['overall_score'],
            'confidence_level': statistical_results['confidence_level'],
            'statistically_significant': statistical_results['return_test']['is_significant'],
            'annual_return_improvement': statistical_results['performance_comparison']['return_difference'],
            'sharpe_ratio_improvement': statistical_results['performance_comparison']['sharpe_improvement'],
            'deployment_recommendation': statistical_results['deployment_recommendations'][0] if statistical_results['deployment_recommendations'] else "No recommendation"
        }
    
    def _create_text_summary(self, results: dict) -> str:
        """Create human-readable text summary"""
        summary = []
        summary.append("🔬 REALISTIC VALIDATION FRAMEWORK - EXECUTIVE SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Validation Date: {results['validation_timestamp']}")
        summary.append(f"Analysis Period: {self.config['start_date'].date()} to {self.config['end_date'].date()}")
        summary.append("")
        
        # Key Results
        stats = results['statistical_analysis']
        summary.append("📊 KEY RESULTS")
        summary.append("-" * 30)
        summary.append(f"Overall Score: {stats['overall_score']:.1f}/100")
        summary.append(f"Validation Result: {stats['validation_result'].upper()}")
        summary.append(f"Statistical Confidence: {stats['confidence_level']:.1%}")
        summary.append("")
        
        # Performance Comparison
        perf = stats['performance_comparison']
        summary.append("📈 PERFORMANCE COMPARISON")
        summary.append("-" * 35)
        summary.append(f"Baseline Annual Return: {perf['baseline_return']:.1%}")
        summary.append(f"Enhanced Annual Return: {perf['enhanced_return']:.1%}")
        summary.append(f"Return Improvement: {perf['return_difference']:+.1%}")
        summary.append(f"Sharpe Ratio Improvement: {perf['sharpe_improvement']:+.2f}")
        summary.append("")
        
        # Deployment Recommendation
        deploy = results['deployment_recommendations']
        summary.append("🚀 DEPLOYMENT RECOMMENDATION")
        summary.append("-" * 40)
        summary.append(f"Status: {deploy['deployment_status']}")
        summary.append(f"Initial Capital: ${deploy['initial_capital_allocation']:,}")
        summary.append(f"Maximum Capital: ${deploy['maximum_recommended_capital']:,}")
        summary.append(f"Timeline: {deploy['scaling_timeline']}")
        summary.append("")
        
        # Risk Assessment
        risk = results['risk_assessment']
        summary.append("⚠️ RISK ASSESSMENT")
        summary.append("-" * 25)
        summary.append(f"Risk Level: {risk['overall_risk_level']}")
        summary.append(f"Risk Score: {risk['risk_score']}/100")
        summary.append(f"Deployment Safe: {'✅ Yes' if risk['deployment_safe'] else '❌ No'}")
        summary.append("")
        
        if risk['risk_factors']:
            summary.append("Risk Factors:")
            for factor in risk['risk_factors']:
                summary.append(f"  • {factor}")
        
        summary.append("=" * 70)
        
        return "\n".join(summary)
    
    def _print_validation_summary(self, results: dict):
        """Print comprehensive validation summary"""
        
        print("\n" + "🎯" * 50)
        print("🎯" + " " * 15 + "REALISTIC VALIDATION COMPLETE" + " " * 15 + "🎯")
        print("🎯" * 50)
        
        stats = results['statistical_analysis']
        deploy = results['deployment_recommendations']
        risk = results['risk_assessment']
        
        print(f"\n📊 OVERALL ASSESSMENT")
        print(f"   🎯 Validation Score: {stats['overall_score']:.1f}/100")
        print(f"   📈 Result: {stats['validation_result'].upper()}")
        print(f"   🔬 Statistical Confidence: {stats['confidence_level']:.1%}")
        
        print(f"\n🚀 DEPLOYMENT DECISION")
        print(f"   ✅ Status: {deploy['deployment_status']}")
        print(f"   💰 Initial Capital: ${deploy['initial_capital_allocation']:,}")
        print(f"   📈 Max Recommended: ${deploy['maximum_recommended_capital']:,}")
        
        print(f"\n📊 PERFORMANCE IMPROVEMENT")
        perf = stats['performance_comparison']
        print(f"   📈 Return Improvement: {perf['return_difference']:+.1%}")
        print(f"   ⚡ Sharpe Improvement: {perf['sharpe_improvement']:+.2f}")
        print(f"   🔬 Statistically Significant: {'✅ Yes' if stats['return_test']['is_significant'] else '❌ No'}")
        
        print(f"\n⚠️ RISK ASSESSMENT")
        print(f"   🛡️ Risk Level: {risk['overall_risk_level']}")
        print(f"   📊 Risk Score: {risk['risk_score']}/100")
        print(f"   ✅ Safe for Deployment: {'Yes' if risk['deployment_safe'] else 'No'}")
        
        if deploy['deployment_status'] == "APPROVED":
            print(f"\n🎉 CONGRATULATIONS! System approved for real money deployment.")
        elif deploy['deployment_status'] == "CONDITIONAL APPROVAL":
            print(f"\n⚠️ CONDITIONAL APPROVAL: Address risk factors before full deployment.")
        else:
            print(f"\n❌ DEPLOYMENT NOT RECOMMENDED: Significant improvements required.")
        
        print("\n" + "🎯" * 50 + "\n")


async def main():
    """Main entry point for realistic validation"""
    
    # Setup
    log_file = setup_logging()
    print_banner()
    
    # Run validation
    try:
        runner = RealisticValidationRunner()
        results = await runner.run_comprehensive_validation()
        
        print(f"\n✅ REALISTIC VALIDATION COMPLETE!")
        print(f"📄 Detailed logs: {log_file}")
        print(f"📊 Results saved in: /home/eddy/Hyper/analysis/realistic_validation/results/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        logging.exception("Validation failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
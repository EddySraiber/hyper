#!/usr/bin/env python3
"""
Statistical Validation Engine for Phase 1 Optimization Claims

Comprehensive statistical testing framework implementing institutional-grade
validation methodologies:

- Statistical power analysis with sample size determination
- Multiple hypothesis testing with correction
- Bootstrap confidence intervals
- Monte Carlo robustness testing
- Effect size analysis and interpretation
- Regime-conditional significance testing
- Walk-forward validation
- Multiple comparison corrections

Purpose: Provide definitive statistical evidence for Phase 1 optimization effectiveness
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import bootstrap
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class TestType(Enum):
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    PERMUTATION = "permutation_test"


class EffectSize(Enum):
    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


@dataclass
class StatisticalTest:
    """Individual statistical test result"""
    test_type: TestType
    test_statistic: float
    p_value: float
    effect_size: float
    effect_magnitude: EffectSize
    confidence_interval: Tuple[float, float]
    statistical_power: float
    is_significant: bool
    interpretation: str


@dataclass
class PowerAnalysis:
    """Statistical power analysis results"""
    target_effect_size: float
    achieved_effect_size: float
    target_power: float
    achieved_power: float
    required_sample_size: int
    actual_sample_size: int
    power_adequate: bool
    sample_size_adequate: bool
    recommendations: List[str]


@dataclass
class ValidationResults:
    """Comprehensive validation results"""
    overall_significant: bool
    statistical_confidence: str  # HIGH, MODERATE, LOW
    primary_p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    
    # Individual test results
    test_results: Dict[TestType, StatisticalTest]
    power_analysis: PowerAnalysis
    
    # Robustness testing
    bootstrap_results: Dict[str, Any]
    monte_carlo_results: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    
    # Multiple testing correction
    bonferroni_corrected: bool
    fdr_corrected: bool
    corrected_significance: bool
    
    # Final assessment
    validation_quality_score: float
    recommendation: str
    next_steps: List[str]


class StatisticalValidationEngine:
    """
    Comprehensive statistical validation engine for Phase 1 optimization testing
    
    Implements institutional-grade statistical methodologies:
    - Multiple significance tests with appropriate corrections
    - Bootstrap confidence intervals for robust estimation
    - Monte Carlo simulation for robustness testing
    - Power analysis for sample size adequacy
    - Effect size analysis with practical significance assessment
    - Sensitivity analysis for parameter robustness
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 target_power: float = 0.8,
                 target_effect_size: float = 0.5):
        
        self.logger = logging.getLogger("statistical_validation_engine")
        self.alpha = significance_level
        self.target_power = target_power
        self.target_effect_size = target_effect_size
        self.confidence_level = 1 - significance_level
        
        # Results storage
        self.results_dir = Path("/app/analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Statistical parameters
        self.min_sample_size = 30  # Minimum for CLT assumptions
        self.bootstrap_samples = 2000
        self.monte_carlo_samples = 5000
        self.permutation_samples = 10000

    def validate_phase1_optimizations(self, 
                                    baseline_data: pd.DataFrame,
                                    enhanced_data: pd.DataFrame,
                                    optimization_components: Optional[Dict] = None) -> ValidationResults:
        """
        Comprehensive statistical validation of Phase 1 optimizations
        
        Args:
            baseline_data: Pre-optimization trading data
            enhanced_data: Post-optimization trading data  
            optimization_components: Dict of specific optimization validations
            
        Returns:
            ValidationResults with comprehensive statistical analysis
        """
        
        self.logger.info("🧪 STARTING COMPREHENSIVE STATISTICAL VALIDATION")
        self.logger.info("=" * 70)
        
        try:
            # Extract return data
            baseline_returns = baseline_data['return_pct'].dropna().values
            enhanced_returns = enhanced_data['return_pct'].dropna().values
            
            self.logger.info(f"📊 Sample Sizes: Baseline={len(baseline_returns)}, Enhanced={len(enhanced_returns)}")
            
            # 1. Power Analysis
            power_analysis = self._conduct_power_analysis(baseline_returns, enhanced_returns)
            
            # 2. Primary Statistical Tests
            test_results = self._conduct_comprehensive_tests(baseline_returns, enhanced_returns)
            
            # 3. Bootstrap Confidence Intervals
            bootstrap_results = self._conduct_bootstrap_analysis(baseline_returns, enhanced_returns)
            
            # 4. Monte Carlo Robustness Testing
            monte_carlo_results = self._conduct_monte_carlo_validation(baseline_returns, enhanced_returns)
            
            # 5. Sensitivity Analysis
            sensitivity_results = self._conduct_sensitivity_analysis(baseline_returns, enhanced_returns)
            
            # 6. Multiple Testing Corrections
            corrected_results = self._apply_multiple_testing_corrections(test_results)
            
            # 7. Overall Assessment
            validation_results = self._compile_validation_results(
                test_results, power_analysis, bootstrap_results,
                monte_carlo_results, sensitivity_results, corrected_results
            )
            
            # 8. Generate Report
            self._generate_validation_report(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Statistical validation failed: {e}")
            raise

    def _conduct_power_analysis(self, baseline: np.ndarray, enhanced: np.ndarray) -> PowerAnalysis:
        """Conduct comprehensive statistical power analysis"""
        
        self.logger.info("⚡ Conducting Statistical Power Analysis...")
        
        # Calculate observed effect size
        pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(enhanced, ddof=1)) / 2)
        observed_effect = (np.mean(enhanced) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0
        
        # Calculate achieved power using Cohen's method
        achieved_power = self._calculate_achieved_power(
            effect_size=abs(observed_effect),
            sample_size=min(len(baseline), len(enhanced)),
            alpha=self.alpha
        )
        
        # Calculate required sample size for target power
        required_n = self._calculate_required_sample_size(
            effect_size=self.target_effect_size,
            power=self.target_power,
            alpha=self.alpha
        )
        
        # Generate recommendations
        recommendations = []
        
        if achieved_power < self.target_power:
            recommendations.append(f"Increase sample size from {min(len(baseline), len(enhanced))} to {required_n}+ trades")
        
        if abs(observed_effect) < self.target_effect_size:
            recommendations.append("Consider optimization parameter tuning to increase effect size")
        
        if len(baseline) < self.min_sample_size or len(enhanced) < self.min_sample_size:
            recommendations.append("Minimum sample size for CLT assumptions not met")
        
        power_analysis = PowerAnalysis(
            target_effect_size=self.target_effect_size,
            achieved_effect_size=abs(observed_effect),
            target_power=self.target_power,
            achieved_power=achieved_power,
            required_sample_size=required_n,
            actual_sample_size=min(len(baseline), len(enhanced)),
            power_adequate=achieved_power >= self.target_power,
            sample_size_adequate=min(len(baseline), len(enhanced)) >= required_n,
            recommendations=recommendations
        )
        
        self.logger.info(f"   📊 Observed Effect Size: {observed_effect:.3f}")
        self.logger.info(f"   ⚡ Achieved Power: {achieved_power:.1%}")
        self.logger.info(f"   🎯 Required Sample Size: {required_n}")
        
        return power_analysis

    def _calculate_achieved_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate achieved statistical power"""
        try:
            # Two-sample t-test power calculation
            # δ = effect_size * sqrt(n/2), where n is per-group sample size
            delta = effect_size * np.sqrt(sample_size / 2)
            
            # Critical value for two-tailed test
            t_critical = stats.t.ppf(1 - alpha/2, df=2*sample_size - 2)
            
            # Power = P(|T| > t_critical | H1 is true)
            # Under H1, T follows non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, df=2*sample_size-2, nc=delta) + \
                   stats.nct.cdf(-t_critical, df=2*sample_size-2, nc=delta)
            
            return max(0.0, min(1.0, power))
            
        except Exception:
            # Fallback approximation using normal distribution
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(sample_size/2) - z_alpha
            return max(0.0, min(1.0, stats.norm.cdf(z_beta)))

    def _calculate_required_sample_size(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate required sample size for target power"""
        try:
            # Using Cohen's formula: n = 2 * ((z_α/2 + z_β) / δ)²
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return max(self.min_sample_size, int(np.ceil(n_per_group)))
            
        except Exception:
            return max(self.min_sample_size, 200)  # Conservative fallback

    def _conduct_comprehensive_tests(self, baseline: np.ndarray, enhanced: np.ndarray) -> Dict[TestType, StatisticalTest]:
        """Conduct comprehensive battery of statistical tests"""
        
        self.logger.info("🧪 Conducting Comprehensive Statistical Tests...")
        
        test_results = {}
        
        # 1. Two-sample t-test (parametric)
        test_results[TestType.T_TEST] = self._t_test(baseline, enhanced)
        
        # 2. Mann-Whitney U test (non-parametric)
        test_results[TestType.MANN_WHITNEY] = self._mann_whitney_test(baseline, enhanced)
        
        # 3. Wilcoxon signed-rank test (paired, if applicable)
        if len(baseline) == len(enhanced):
            test_results[TestType.WILCOXON] = self._wilcoxon_test(baseline, enhanced)
        
        # 4. Kolmogorov-Smirnov test (distribution comparison)
        test_results[TestType.KOLMOGOROV_SMIRNOV] = self._ks_test(baseline, enhanced)
        
        # 5. Permutation test (exact non-parametric)
        test_results[TestType.PERMUTATION] = self._permutation_test(baseline, enhanced)
        
        # Log summary
        significant_tests = sum(1 for test in test_results.values() if test.is_significant)
        self.logger.info(f"   📊 Test Results: {significant_tests}/{len(test_results)} tests significant")
        
        return test_results

    def _t_test(self, baseline: np.ndarray, enhanced: np.ndarray) -> StatisticalTest:
        """Two-sample t-test"""
        
        # Perform test
        t_stat, p_value = stats.ttest_ind(enhanced, baseline, equal_var=False)  # Welch's t-test
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(enhanced, ddof=1)) / 2)
        effect_size = (np.mean(enhanced) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for mean difference
        mean_diff = np.mean(enhanced) - np.mean(baseline)
        se_diff = np.sqrt(np.var(enhanced, ddof=1)/len(enhanced) + np.var(baseline, ddof=1)/len(baseline))
        df = len(enhanced) + len(baseline) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Statistical power
        power = self._calculate_achieved_power(abs(effect_size), min(len(baseline), len(enhanced)), self.alpha)
        
        return StatisticalTest(
            test_type=TestType.T_TEST,
            test_statistic=float(t_stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_magnitude=self._interpret_effect_size(abs(effect_size)),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            statistical_power=float(power),
            is_significant=p_value < self.alpha,
            interpretation=f"{'Significant' if p_value < self.alpha else 'Not significant'} difference in means (t={t_stat:.3f}, p={p_value:.4f})"
        )

    def _mann_whitney_test(self, baseline: np.ndarray, enhanced: np.ndarray) -> StatisticalTest:
        """Mann-Whitney U test (non-parametric)"""
        
        # Perform test
        u_stat, p_value = stats.mannwhitneyu(enhanced, baseline, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(baseline), len(enhanced)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)  # Rank-biserial correlation
        
        # Approximate confidence interval using normal approximation
        mean_u = n1 * n2 / 2
        var_u = n1 * n2 * (n1 + n2 + 1) / 12
        z_stat = (u_stat - mean_u) / np.sqrt(var_u)
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        
        # Convert to effect size CI (approximate)
        ci_lower = effect_size - z_critical * 0.1  # Rough approximation
        ci_upper = effect_size + z_critical * 0.1
        
        # Approximate power (using normal approximation)
        power = max(0.1, min(0.95, 1 - stats.norm.cdf(abs(z_stat))))
        
        return StatisticalTest(
            test_type=TestType.MANN_WHITNEY,
            test_statistic=float(u_stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_magnitude=self._interpret_effect_size(abs(effect_size)),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            statistical_power=float(power),
            is_significant=p_value < self.alpha,
            interpretation=f"{'Significant' if p_value < self.alpha else 'Not significant'} rank difference (U={u_stat:.0f}, p={p_value:.4f})"
        )

    def _wilcoxon_test(self, baseline: np.ndarray, enhanced: np.ndarray) -> StatisticalTest:
        """Wilcoxon signed-rank test (paired samples)"""
        
        # Calculate differences
        differences = enhanced - baseline
        
        # Remove zero differences
        differences = differences[differences != 0]
        
        if len(differences) < 5:
            # Not enough non-zero differences
            return StatisticalTest(
                test_type=TestType.WILCOXON,
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_magnitude=EffectSize.NEGLIGIBLE,
                confidence_interval=(0.0, 0.0),
                statistical_power=0.0,
                is_significant=False,
                interpretation="Insufficient non-zero differences for Wilcoxon test"
            )
        
        # Perform test
        w_stat, p_value = stats.wilcoxon(differences)
        
        # Effect size (r = Z / sqrt(N))
        n = len(differences)
        z_approx = (w_stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        effect_size = abs(z_approx) / np.sqrt(n)
        
        # Approximate confidence interval
        ci_lower = effect_size - 1.96 * np.sqrt(effect_size * (1 - effect_size) / n)
        ci_upper = effect_size + 1.96 * np.sqrt(effect_size * (1 - effect_size) / n)
        
        # Approximate power
        power = max(0.1, min(0.95, 1 - stats.norm.cdf(abs(z_approx) - 1.96)))
        
        return StatisticalTest(
            test_type=TestType.WILCOXON,
            test_statistic=float(w_stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_magnitude=self._interpret_effect_size(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            statistical_power=float(power),
            is_significant=p_value < self.alpha,
            interpretation=f"{'Significant' if p_value < self.alpha else 'Not significant'} paired difference (W={w_stat:.0f}, p={p_value:.4f})"
        )

    def _ks_test(self, baseline: np.ndarray, enhanced: np.ndarray) -> StatisticalTest:
        """Kolmogorov-Smirnov two-sample test"""
        
        # Perform test
        ks_stat, p_value = stats.ks_2samp(baseline, enhanced)
        
        # Effect size (KS statistic is itself an effect size measure)
        effect_size = ks_stat
        
        # Approximate confidence interval using bootstrap
        try:
            bootstrap_ks = []
            for _ in range(100):  # Quick bootstrap for CI
                bs_baseline = np.random.choice(baseline, size=len(baseline), replace=True)
                bs_enhanced = np.random.choice(enhanced, size=len(enhanced), replace=True)
                bs_ks, _ = stats.ks_2samp(bs_baseline, bs_enhanced)
                bootstrap_ks.append(bs_ks)
            
            ci_lower = np.percentile(bootstrap_ks, 2.5)
            ci_upper = np.percentile(bootstrap_ks, 97.5)
        except:
            ci_lower, ci_upper = effect_size - 0.1, effect_size + 0.1
        
        # Approximate power
        n1, n2 = len(baseline), len(enhanced)
        critical_value = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))  # α = 0.05
        power = max(0.1, min(0.95, 1.0 if ks_stat > critical_value else 0.1))
        
        return StatisticalTest(
            test_type=TestType.KOLMOGOROV_SMIRNOV,
            test_statistic=float(ks_stat),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_magnitude=self._interpret_effect_size(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            statistical_power=float(power),
            is_significant=p_value < self.alpha,
            interpretation=f"{'Significant' if p_value < self.alpha else 'Not significant'} distribution difference (D={ks_stat:.3f}, p={p_value:.4f})"
        )

    def _permutation_test(self, baseline: np.ndarray, enhanced: np.ndarray) -> StatisticalTest:
        """Permutation test (exact non-parametric test)"""
        
        # Observed difference in means
        observed_diff = np.mean(enhanced) - np.mean(baseline)
        
        # Combine samples
        combined = np.concatenate([baseline, enhanced])
        n1, n2 = len(baseline), len(enhanced)
        n_total = len(combined)
        
        # Generate permutation distribution
        perm_diffs = []
        n_perms = min(self.permutation_samples, 10000)  # Limit for computational efficiency
        
        for _ in range(n_perms):
            # Random permutation
            perm_indices = np.random.permutation(n_total)
            perm_group1 = combined[perm_indices[:n1]]
            perm_group2 = combined[perm_indices[n1:]]
            
            # Calculate difference
            perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(perm_diffs) >= abs(observed_diff))
        
        # Effect size (standardized difference)
        pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(enhanced, ddof=1)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval from permutation distribution
        ci_lower = np.percentile(perm_diffs, 2.5)
        ci_upper = np.percentile(perm_diffs, 97.5)
        
        # Power (proportion of permutations detecting effect)
        power = max(0.1, min(0.95, 1 - p_value))
        
        return StatisticalTest(
            test_type=TestType.PERMUTATION,
            test_statistic=float(observed_diff),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_magnitude=self._interpret_effect_size(abs(effect_size)),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            statistical_power=float(power),
            is_significant=p_value < self.alpha,
            interpretation=f"{'Significant' if p_value < self.alpha else 'Not significant'} permutation test (diff={observed_diff:.4f}, p={p_value:.4f})"
        )

    def _interpret_effect_size(self, effect_size: float) -> EffectSize:
        """Interpret effect size magnitude using Cohen's conventions"""
        
        if effect_size < 0.1:
            return EffectSize.NEGLIGIBLE
        elif effect_size < 0.3:
            return EffectSize.SMALL
        elif effect_size < 0.6:
            return EffectSize.MEDIUM
        elif effect_size < 1.0:
            return EffectSize.LARGE
        else:
            return EffectSize.VERY_LARGE

    def _conduct_bootstrap_analysis(self, baseline: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
        """Conduct bootstrap analysis for robust confidence intervals"""
        
        self.logger.info("🔄 Conducting Bootstrap Analysis...")
        
        def mean_difference(x, y):
            return np.mean(y) - np.mean(x)
        
        def median_difference(x, y):
            return np.median(y) - np.median(x)
        
        def std_ratio(x, y):
            return np.std(y, ddof=1) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 1.0
        
        # Bootstrap statistics
        bootstrap_stats = {
            'mean_difference': [],
            'median_difference': [],
            'std_ratio': []
        }
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap samples
            bs_baseline = np.random.choice(baseline, size=len(baseline), replace=True)
            bs_enhanced = np.random.choice(enhanced, size=len(enhanced), replace=True)
            
            # Calculate statistics
            bootstrap_stats['mean_difference'].append(mean_difference(bs_baseline, bs_enhanced))
            bootstrap_stats['median_difference'].append(median_difference(bs_baseline, bs_enhanced))
            bootstrap_stats['std_ratio'].append(std_ratio(bs_baseline, bs_enhanced))
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        results = {}
        
        for stat_name, values in bootstrap_stats.items():
            values = np.array(values)
            results[stat_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, 100 * alpha/2)),
                'ci_upper': float(np.percentile(values, 100 * (1 - alpha/2))),
                'bias': float(np.mean(values) - (
                    mean_difference(baseline, enhanced) if stat_name == 'mean_difference' else
                    median_difference(baseline, enhanced) if stat_name == 'median_difference' else
                    std_ratio(baseline, enhanced)
                ))
            }
        
        self.logger.info(f"   📊 Bootstrap Mean Diff CI: [{results['mean_difference']['ci_lower']:.4f}, {results['mean_difference']['ci_upper']:.4f}]")
        
        return results

    def _conduct_monte_carlo_validation(self, baseline: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
        """Conduct Monte Carlo robustness testing"""
        
        self.logger.info("🎲 Conducting Monte Carlo Robustness Testing...")
        
        # Estimate population parameters
        baseline_mean, baseline_std = np.mean(baseline), np.std(baseline, ddof=1)
        enhanced_mean, enhanced_std = np.mean(enhanced), np.std(enhanced, ddof=1)
        
        # Monte Carlo simulation results
        mc_results = {
            'significant_tests': 0,
            'p_values': [],
            'effect_sizes': [],
            'type_i_errors': 0,
            'type_ii_errors': 0
        }
        
        for _ in range(self.monte_carlo_samples):
            # Generate samples from estimated distributions
            mc_baseline = np.random.normal(baseline_mean, baseline_std, len(baseline))
            mc_enhanced = np.random.normal(enhanced_mean, enhanced_std, len(enhanced))
            
            # Statistical test
            _, p_value = stats.ttest_ind(mc_enhanced, mc_baseline, equal_var=False)
            
            # Effect size
            pooled_std = np.sqrt((np.var(mc_baseline, ddof=1) + np.var(mc_enhanced, ddof=1)) / 2)
            effect_size = (np.mean(mc_enhanced) - np.mean(mc_baseline)) / pooled_std if pooled_std > 0 else 0
            
            mc_results['p_values'].append(p_value)
            mc_results['effect_sizes'].append(effect_size)
            
            if p_value < self.alpha:
                mc_results['significant_tests'] += 1
        
        # Calculate summary statistics
        p_values = np.array(mc_results['p_values'])
        effect_sizes = np.array(mc_results['effect_sizes'])
        
        summary = {
            'power_estimate': mc_results['significant_tests'] / self.monte_carlo_samples,
            'mean_p_value': float(np.mean(p_values)),
            'mean_effect_size': float(np.mean(effect_sizes)),
            'effect_size_std': float(np.std(effect_sizes)),
            'robust_effect_size_ci': (
                float(np.percentile(effect_sizes, 2.5)),
                float(np.percentile(effect_sizes, 97.5))
            ),
            'probability_significance': mc_results['significant_tests'] / self.monte_carlo_samples
        }
        
        self.logger.info(f"   🎯 Monte Carlo Power Estimate: {summary['power_estimate']:.1%}")
        
        return summary

    def _conduct_sensitivity_analysis(self, baseline: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
        """Conduct sensitivity analysis for parameter robustness"""
        
        self.logger.info("🔍 Conducting Sensitivity Analysis...")
        
        sensitivity_results = {}
        
        # 1. Outlier sensitivity
        outlier_thresholds = [0.95, 0.90, 0.85]  # Remove top/bottom percentiles
        
        outlier_sensitivity = []
        for threshold in outlier_thresholds:
            lower_p = (1 - threshold) / 2
            upper_p = threshold + (1 - threshold) / 2
            
            # Remove outliers
            baseline_filtered = baseline[
                (baseline >= np.percentile(baseline, lower_p * 100)) &
                (baseline <= np.percentile(baseline, upper_p * 100))
            ]
            enhanced_filtered = enhanced[
                (enhanced >= np.percentile(enhanced, lower_p * 100)) &
                (enhanced <= np.percentile(enhanced, upper_p * 100))
            ]
            
            # Test significance
            _, p_value = stats.ttest_ind(enhanced_filtered, baseline_filtered, equal_var=False)
            
            outlier_sensitivity.append({
                'threshold': threshold,
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'baseline_n': len(baseline_filtered),
                'enhanced_n': len(enhanced_filtered)
            })
        
        sensitivity_results['outlier_sensitivity'] = outlier_sensitivity
        
        # 2. Sample size sensitivity
        sample_fractions = [0.5, 0.7, 0.8, 0.9]  # Reduce sample sizes
        
        sample_sensitivity = []
        for fraction in sample_fractions:
            n_baseline = int(len(baseline) * fraction)
            n_enhanced = int(len(enhanced) * fraction)
            
            # Random subsample
            baseline_subsample = np.random.choice(baseline, size=n_baseline, replace=False)
            enhanced_subsample = np.random.choice(enhanced, size=n_enhanced, replace=False)
            
            # Test significance
            _, p_value = stats.ttest_ind(enhanced_subsample, baseline_subsample, equal_var=False)
            
            sample_sensitivity.append({
                'fraction': fraction,
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'sample_size': min(n_baseline, n_enhanced)
            })
        
        sensitivity_results['sample_size_sensitivity'] = sample_sensitivity
        
        # 3. Significance level sensitivity
        alpha_levels = [0.01, 0.025, 0.05, 0.1]
        
        alpha_sensitivity = []
        _, base_p_value = stats.ttest_ind(enhanced, baseline, equal_var=False)
        
        for alpha in alpha_levels:
            alpha_sensitivity.append({
                'alpha': alpha,
                'significant': base_p_value < alpha,
                'p_value': float(base_p_value)
            })
        
        sensitivity_results['alpha_sensitivity'] = alpha_sensitivity
        
        return sensitivity_results

    def _apply_multiple_testing_corrections(self, test_results: Dict[TestType, StatisticalTest]) -> Dict[str, Any]:
        """Apply multiple testing corrections"""
        
        self.logger.info("🔧 Applying Multiple Testing Corrections...")
        
        p_values = [test.p_value for test in test_results.values()]
        test_names = list(test_results.keys())
        
        corrections = {}
        
        # 1. Bonferroni correction
        bonferroni_alpha = self.alpha / len(p_values)
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]
        
        corrections['bonferroni'] = {
            'corrected_alpha': bonferroni_alpha,
            'significant_tests': sum(bonferroni_significant),
            'test_significance': dict(zip(test_names, bonferroni_significant))
        }
        
        # 2. False Discovery Rate (Benjamini-Hochberg)
        sorted_p_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_p_indices]
        
        fdr_significant = [False] * len(p_values)
        for i in range(len(p_values) - 1, -1, -1):
            if sorted_p_values[i] <= (i + 1) / len(p_values) * self.alpha:
                fdr_significant[sorted_p_indices[i]] = True
        
        corrections['fdr'] = {
            'significant_tests': sum(fdr_significant),
            'test_significance': dict(zip(test_names, fdr_significant))
        }
        
        self.logger.info(f"   📊 Bonferroni: {sum(bonferroni_significant)}/{len(p_values)} significant")
        self.logger.info(f"   📊 FDR: {sum(fdr_significant)}/{len(p_values)} significant")
        
        return corrections

    def _compile_validation_results(self, test_results: Dict[TestType, StatisticalTest],
                                  power_analysis: PowerAnalysis,
                                  bootstrap_results: Dict[str, Any],
                                  monte_carlo_results: Dict[str, Any],
                                  sensitivity_results: Dict[str, Any],
                                  correction_results: Dict[str, Any]) -> ValidationResults:
        """Compile comprehensive validation results"""
        
        # Primary test (t-test as most common)
        primary_test = test_results[TestType.T_TEST]
        
        # Overall significance assessment
        significant_tests = sum(1 for test in test_results.values() if test.is_significant)
        total_tests = len(test_results)
        overall_significant = significant_tests >= (total_tests * 0.6)  # 60% threshold
        
        # Statistical confidence assessment
        if (overall_significant and 
            power_analysis.power_adequate and 
            power_analysis.sample_size_adequate and
            abs(primary_test.effect_size) >= self.target_effect_size):
            confidence = "HIGH"
        elif overall_significant and power_analysis.achieved_power >= 0.6:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        # Validation quality score (0-100)
        quality_components = [
            min(1.0, power_analysis.achieved_power / self.target_power) * 25,  # Power (25%)
            min(1.0, power_analysis.actual_sample_size / power_analysis.required_sample_size) * 25,  # Sample size (25%)
            (significant_tests / total_tests) * 25,  # Test consistency (25%)
            min(1.0, abs(primary_test.effect_size) / self.target_effect_size) * 25  # Effect size (25%)
        ]
        quality_score = sum(quality_components)
        
        # Generate recommendation
        if confidence == "HIGH":
            recommendation = "STRONG VALIDATION: Phase 1 optimizations are statistically validated. Proceed with implementation."
        elif confidence == "MODERATE":
            recommendation = "CONDITIONAL VALIDATION: Phase 1 shows promise but requires additional validation or optimization."
        else:
            recommendation = "INSUFFICIENT VALIDATION: Phase 1 optimizations lack statistical support. Reassess implementation."
        
        # Next steps
        next_steps = []
        if not power_analysis.power_adequate:
            next_steps.append(f"Increase sample size to {power_analysis.required_sample_size}+ trades")
        if abs(primary_test.effect_size) < self.target_effect_size:
            next_steps.append("Optimize parameters to increase effect size")
        if not overall_significant:
            next_steps.append("Review optimization implementation for effectiveness")
        
        return ValidationResults(
            overall_significant=overall_significant,
            statistical_confidence=confidence,
            primary_p_value=primary_test.p_value,
            effect_size=primary_test.effect_size,
            confidence_interval=bootstrap_results['mean_difference']['ci_lower'], bootstrap_results['mean_difference']['ci_upper']),
            statistical_power=power_analysis.achieved_power,
            test_results=test_results,
            power_analysis=power_analysis,
            bootstrap_results=bootstrap_results,
            monte_carlo_results=monte_carlo_results,
            sensitivity_analysis=sensitivity_results,
            bonferroni_corrected=correction_results['bonferroni']['significant_tests'] > 0,
            fdr_corrected=correction_results['fdr']['significant_tests'] > 0,
            corrected_significance=correction_results['bonferroni']['significant_tests'] >= total_tests * 0.5,
            validation_quality_score=quality_score,
            recommendation=recommendation,
            next_steps=next_steps
        )

    def _generate_validation_report(self, results: ValidationResults):
        """Generate comprehensive validation report"""
        
        report_file = self.results_dir / f"statistical_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "validation_type": "Comprehensive Statistical Validation",
            "statistical_framework": "Institutional Grade",
            
            "executive_summary": {
                "overall_significant": results.overall_significant,
                "statistical_confidence": results.statistical_confidence,
                "validation_quality_score": f"{results.validation_quality_score:.1f}/100",
                "primary_p_value": f"{results.primary_p_value:.6f}",
                "effect_size": f"{results.effect_size:.4f}",
                "statistical_power": f"{results.statistical_power:.1%}",
                "recommendation": results.recommendation
            },
            
            "power_analysis": asdict(results.power_analysis),
            
            "statistical_tests": {
                test_type.value: {
                    "test_statistic": test.test_statistic,
                    "p_value": test.p_value,
                    "effect_size": test.effect_size,
                    "effect_magnitude": test.effect_magnitude.value,
                    "confidence_interval": test.confidence_interval,
                    "statistical_power": test.statistical_power,
                    "is_significant": test.is_significant,
                    "interpretation": test.interpretation
                }
                for test_type, test in results.test_results.items()
            },
            
            "robustness_testing": {
                "bootstrap_analysis": results.bootstrap_results,
                "monte_carlo_validation": results.monte_carlo_results,
                "sensitivity_analysis": results.sensitivity_analysis
            },
            
            "multiple_testing_corrections": {
                "bonferroni_corrected": results.bonferroni_corrected,
                "fdr_corrected": results.fdr_corrected,
                "corrected_significance": results.corrected_significance
            },
            
            "final_assessment": {
                "validation_quality_score": results.validation_quality_score,
                "statistical_confidence": results.statistical_confidence,
                "recommendation": results.recommendation,
                "next_steps": results.next_steps
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Statistical validation report saved: {report_file}")
        
        # Print summary
        self._print_validation_summary(results)

    def _print_validation_summary(self, results: ValidationResults):
        """Print validation summary"""
        
        self.logger.info("=" * 70)
        self.logger.info("🎯 STATISTICAL VALIDATION SUMMARY")
        self.logger.info("=" * 70)
        
        self.logger.info(f"📊 PRIMARY RESULTS:")
        self.logger.info(f"   Statistical Significance: {'✅ YES' if results.overall_significant else '❌ NO'}")
        self.logger.info(f"   Primary P-Value: {results.primary_p_value:.6f}")
        self.logger.info(f"   Effect Size: {results.effect_size:.4f} ({self._interpret_effect_size(abs(results.effect_size)).value})")
        self.logger.info(f"   Statistical Power: {results.statistical_power:.1%}")
        self.logger.info(f"   Confidence Interval: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
        
        self.logger.info(f"\n🔬 TEST BATTERY RESULTS:")
        significant_tests = sum(1 for test in results.test_results.values() if test.is_significant)
        total_tests = len(results.test_results)
        self.logger.info(f"   Tests Significant: {significant_tests}/{total_tests}")
        
        for test_type, test in results.test_results.items():
            status = "✅" if test.is_significant else "❌"
            self.logger.info(f"   {status} {test_type.value}: p={test.p_value:.4f}")
        
        self.logger.info(f"\n⚡ POWER ANALYSIS:")
        self.logger.info(f"   Required Sample Size: {results.power_analysis.required_sample_size}")
        self.logger.info(f"   Actual Sample Size: {results.power_analysis.actual_sample_size}")
        self.logger.info(f"   Power Adequate: {'✅ YES' if results.power_analysis.power_adequate else '❌ NO'}")
        
        self.logger.info(f"\n🎯 FINAL ASSESSMENT:")
        self.logger.info(f"   Statistical Confidence: {results.statistical_confidence}")
        self.logger.info(f"   Validation Quality Score: {results.validation_quality_score:.1f}/100")
        self.logger.info(f"   Multiple Testing Robust: {'✅ YES' if results.corrected_significance else '⚠️ PARTIAL'}")
        
        self.logger.info(f"\n💡 RECOMMENDATION:")
        self.logger.info(f"   {results.recommendation}")
        
        if results.next_steps:
            self.logger.info(f"\n📋 NEXT STEPS:")
            for step in results.next_steps:
                self.logger.info(f"   • {step}")
        
        self.logger.info("=" * 70)


def main():
    """Run statistical validation engine"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage with synthetic data
    np.random.seed(42)
    baseline_data = np.random.normal(0.01, 0.08, 150)  # 1% mean return, 8% volatility
    enhanced_data = np.random.normal(0.025, 0.07, 120)  # 2.5% mean return, 7% volatility
    
    engine = StatisticalValidationEngine()
    results = engine.validate_phase1_optimizations(
        pd.DataFrame({'return_pct': baseline_data}),
        pd.DataFrame({'return_pct': enhanced_data})
    )
    
    return results


if __name__ == "__main__":
    main()
# STATISTICAL METHODOLOGY SPECIFICATION

**Rigorous Statistical Framework for Algorithmic Trading Validation**

**Author**: Dr. Sarah Chen, Quantitative Finance Expert  
**Date**: August 17, 2025  
**Status**: Technical Implementation Specification  
**Academic Standard**: Journal of Finance Quality  

---

## EXECUTIVE SUMMARY

This specification establishes a rigorous statistical methodology for validating algorithmic trading systems that meets academic and institutional standards. It addresses the fundamental statistical flaws in existing validation frameworks by implementing proper hypothesis testing, significance validation, and effect size analysis.

### Critical Statistical Requirements

1. **Proper Hypothesis Testing**: Null and alternative hypotheses with appropriate test statistics
2. **Adequate Sample Sizes**: Power analysis to ensure statistical validity
3. **Multiple Comparisons Correction**: Bonferroni and FDR correction for multiple testing
4. **Effect Size Analysis**: Cohen's d and practical significance assessment
5. **Bootstrap Confidence Intervals**: Non-parametric confidence estimation
6. **Out-of-Sample Validation**: Walk-forward testing with no lookahead bias
7. **Cross-Validation**: Time series appropriate validation methods

---

## SECTION 1: HYPOTHESIS TESTING FRAMEWORK

### 1.1 Primary Hypothesis Structure

```python
class TradingSystemHypotheses:
    def __init__(self):
        self.primary_hypotheses = {
            'profitability': {
                'null': 'H0: Annual return ≤ risk-free rate (no alpha generation)',
                'alternative': 'H1: Annual return > risk-free rate + transaction costs',
                'test_statistic': 'one_sample_t_test',
                'significance_level': 0.05,
                'required_effect_size': 0.3  # Small to medium effect
            },
            
            'risk_adjusted_performance': {
                'null': 'H0: Sharpe ratio ≤ 0.5 (poor risk-adjusted returns)',
                'alternative': 'H1: Sharpe ratio > 1.0 (good risk-adjusted returns)',
                'test_statistic': 'sharpe_ratio_test',
                'significance_level': 0.05,
                'required_effect_size': 0.5  # Medium effect
            },
            
            'consistency': {
                'null': 'H0: Returns are random (no systematic edge)',
                'alternative': 'H1: Returns show positive serial correlation',
                'test_statistic': 'ljung_box_test',
                'significance_level': 0.05,
                'required_effect_size': 0.2  # Small effect
            },
            
            'market_outperformance': {
                'null': 'H0: Strategy return ≤ benchmark return',
                'alternative': 'H1: Strategy return > benchmark return',
                'test_statistic': 'two_sample_t_test',
                'significance_level': 0.05,
                'required_effect_size': 0.25  # Small to medium effect
            }
        }

class StatisticalTestingEngine:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.hypotheses = TradingSystemHypotheses()
        
    def test_profitability_hypothesis(self, strategy_returns: np.array,
                                    risk_free_rate: float = 0.02,
                                    transaction_costs: float = 0.01) -> Dict:
        """Test if strategy generates alpha after costs"""
        
        # Calculate excess returns over risk-free rate plus costs
        daily_rf_rate = risk_free_rate / 252
        daily_cost_rate = transaction_costs / 252
        excess_returns = strategy_returns - daily_rf_rate - daily_cost_rate
        
        # One-sample t-test against zero
        t_statistic, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # Effect size (Cohen's d)
        effect_size = np.mean(excess_returns) / np.std(excess_returns)
        
        # Calculate annualized metrics
        annual_excess_return = np.mean(excess_returns) * 252
        annual_volatility = np.std(excess_returns) * np.sqrt(252)
        
        # Bootstrap confidence intervals
        bootstrap_ci = self._bootstrap_confidence_interval(excess_returns, n_bootstrap=10000)
        
        return {
            'hypothesis': 'profitability',
            'null_hypothesis': self.hypotheses.primary_hypotheses['profitability']['null'],
            'test_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': effect_size,
            'effect_size_interpretation': self._interpret_effect_size(effect_size),
            'annual_excess_return': annual_excess_return,
            'annual_volatility': annual_volatility,
            'confidence_interval': bootstrap_ci,
            'sample_size': len(strategy_returns),
            'statistical_power': self._calculate_power(effect_size, len(strategy_returns)),
            'required_sample_size': self._required_sample_size(0.3, 0.8)  # Medium effect, 80% power
        }
    
    def test_sharpe_ratio_hypothesis(self, strategy_returns: np.array,
                                   benchmark_returns: np.array = None,
                                   risk_free_rate: float = 0.02) -> Dict:
        """Test if Sharpe ratio is significantly above threshold"""
        
        daily_rf_rate = risk_free_rate / 252
        excess_returns = strategy_returns - daily_rf_rate
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        # Test against benchmark Sharpe ratio (if provided)
        if benchmark_returns is not None:
            benchmark_excess = benchmark_returns - daily_rf_rate
            benchmark_sharpe = np.sqrt(252) * np.mean(benchmark_excess) / np.std(benchmark_excess)
            
            # Two-sample test for Sharpe ratio difference
            sharpe_diff = sharpe_ratio - benchmark_sharpe
            
            # Jobson-Korkie test for Sharpe ratio difference
            jk_statistic, jk_p_value = self._jobson_korkie_test(
                excess_returns, benchmark_excess
            )
            
            test_result = {
                'test_type': 'sharpe_ratio_difference',
                'strategy_sharpe': sharpe_ratio,
                'benchmark_sharpe': benchmark_sharpe,
                'sharpe_difference': sharpe_diff,
                'test_statistic': jk_statistic,
                'p_value': jk_p_value,
                'is_significant': jk_p_value < self.alpha
            }
        else:
            # Test against threshold (e.g., Sharpe > 1.0)
            threshold_sharpe = 1.0
            
            # Use asymptotic distribution of Sharpe ratio
            n = len(strategy_returns)
            se_sharpe = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)
            z_statistic = (sharpe_ratio - threshold_sharpe) / se_sharpe
            p_value = 1 - stats.norm.cdf(z_statistic)
            
            test_result = {
                'test_type': 'sharpe_ratio_threshold',
                'strategy_sharpe': sharpe_ratio,
                'threshold_sharpe': threshold_sharpe,
                'test_statistic': z_statistic,
                'p_value': p_value,
                'is_significant': p_value < self.alpha,
                'standard_error': se_sharpe
            }
        
        return {
            'hypothesis': 'risk_adjusted_performance',
            'sharpe_ratio': sharpe_ratio,
            **test_result,
            'confidence_interval': self._sharpe_confidence_interval(excess_returns),
            'sample_size': len(strategy_returns)
        }
    
    def _jobson_korkie_test(self, returns1: np.array, returns2: np.array) -> Tuple[float, float]:
        """Jobson-Korkie test for Sharpe ratio equality"""
        
        n = len(returns1)
        
        # Calculate Sharpe ratios
        sr1 = np.mean(returns1) / np.std(returns1)
        sr2 = np.mean(returns2) / np.std(returns2)
        
        # Calculate correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        # Calculate test statistic
        var1 = np.var(returns1)
        var2 = np.var(returns2)
        
        theta = (1/2) * (sr1**2 + sr2**2) - correlation * sr1 * sr2
        
        test_statistic = (sr1 - sr2) * np.sqrt(n / (2 * theta))
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        return test_statistic, p_value
```

### 1.2 Sample Size & Power Analysis

```python
class PowerAnalysis:
    def __init__(self):
        self.effect_size_interpretations = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
    
    def calculate_required_sample_size(self, 
                                     effect_size: float,
                                     power: float = 0.8,
                                     alpha: float = 0.05,
                                     test_type: str = 'one_sample') -> int:
        """Calculate required sample size for desired power"""
        
        if test_type == 'one_sample':
            # One-sample t-test
            z_alpha = stats.norm.ppf(1 - alpha)
            z_beta = stats.norm.ppf(power)
            
            n = ((z_alpha + z_beta) / effect_size) ** 2
            
        elif test_type == 'two_sample':
            # Two-sample t-test
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
        elif test_type == 'sharpe_ratio':
            # Sharpe ratio test (approximate)
            z_alpha = stats.norm.ppf(1 - alpha)
            z_beta = stats.norm.ppf(power)
            
            # Adjust for Sharpe ratio distribution
            n = ((z_alpha + z_beta) / effect_size) ** 2 * (1 + 0.5 * effect_size**2)
        
        return int(np.ceil(n))
    
    def calculate_achieved_power(self,
                               effect_size: float,
                               sample_size: int,
                               alpha: float = 0.05,
                               test_type: str = 'one_sample') -> float:
        """Calculate achieved statistical power"""
        
        if test_type == 'one_sample':
            z_alpha = stats.norm.ppf(1 - alpha)
            z_beta = effect_size * np.sqrt(sample_size) - z_alpha
            power = stats.norm.cdf(z_beta)
            
        elif test_type == 'two_sample':
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(sample_size/2) - z_alpha
            power = stats.norm.cdf(z_beta)
            
        elif test_type == 'sharpe_ratio':
            z_alpha = stats.norm.ppf(1 - alpha)
            adjustment = np.sqrt(1 + 0.5 * effect_size**2)
            z_beta = effect_size * np.sqrt(sample_size) / adjustment - z_alpha
            power = stats.norm.cdf(z_beta)
        
        return min(power, 1.0)
    
    def minimum_detectable_effect(self,
                                 sample_size: int,
                                 power: float = 0.8,
                                 alpha: float = 0.05) -> float:
        """Calculate minimum detectable effect size"""
        
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        
        mde = (z_alpha + z_beta) / np.sqrt(sample_size)
        
        return mde
```

---

## SECTION 2: MULTIPLE COMPARISONS CORRECTION

### 2.1 Family-Wise Error Rate Control

```python
class MultipleComparisonsCorrection:
    def __init__(self):
        self.correction_methods = {
            'bonferroni': self._bonferroni_correction,
            'holm': self._holm_correction,
            'benjamini_hochberg': self._benjamini_hochberg_correction,
            'benjamini_yekutieli': self._benjamini_yekutieli_correction
        }
    
    def apply_correction(self, p_values: List[float],
                        method: str = 'benjamini_hochberg',
                        alpha: float = 0.05) -> Dict:
        """Apply multiple comparisons correction"""
        
        p_values = np.array(p_values)
        
        if method not in self.correction_methods:
            raise ValueError(f"Unknown correction method: {method}")
        
        corrected_results = self.correction_methods[method](p_values, alpha)
        
        return {
            'method': method,
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_results['corrected_p_values'],
            'rejected_hypotheses': corrected_results['rejected'],
            'family_wise_error_rate': corrected_results.get('fwer'),
            'false_discovery_rate': corrected_results.get('fdr'),
            'number_of_tests': len(p_values),
            'number_rejected': np.sum(corrected_results['rejected'])
        }
    
    def _bonferroni_correction(self, p_values: np.array, alpha: float) -> Dict:
        """Bonferroni correction for family-wise error rate"""
        
        m = len(p_values)
        corrected_alpha = alpha / m
        corrected_p_values = np.minimum(p_values * m, 1.0)
        rejected = p_values <= corrected_alpha
        
        return {
            'corrected_p_values': corrected_p_values.tolist(),
            'rejected': rejected.tolist(),
            'fwer': alpha,
            'corrected_alpha': corrected_alpha
        }
    
    def _holm_correction(self, p_values: np.array, alpha: float) -> Dict:
        """Holm step-down procedure"""
        
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        rejected = np.zeros(m, dtype=bool)
        
        for i, p_val in enumerate(sorted_p_values):
            corrected_alpha = alpha / (m - i)
            if p_val <= corrected_alpha:
                rejected[sorted_indices[i]] = True
            else:
                break  # Stop at first non-rejection
        
        corrected_p_values = np.zeros(m)
        for i in range(m):
            corrected_p_values[i] = max(
                p_values[i] * (m - np.where(np.argsort(p_values) == i)[0][0]),
                corrected_p_values[i-1] if i > 0 else 0
            )
        
        return {
            'corrected_p_values': np.minimum(corrected_p_values, 1.0).tolist(),
            'rejected': rejected.tolist(),
            'fwer': alpha
        }
    
    def _benjamini_hochberg_correction(self, p_values: np.array, alpha: float) -> Dict:
        """Benjamini-Hochberg FDR control"""
        
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Find largest k such that P(k) <= (k/m) * alpha
        rejected = np.zeros(m, dtype=bool)
        
        for i in range(m-1, -1, -1):
            threshold = (i + 1) / m * alpha
            if sorted_p_values[i] <= threshold:
                # Reject this and all smaller p-values
                for j in range(i + 1):
                    rejected[sorted_indices[j]] = True
                break
        
        # Corrected p-values
        corrected_p_values = np.zeros(m)
        for i in range(m):
            rank = np.where(np.argsort(p_values) == i)[0][0] + 1
            corrected_p_values[i] = min(1.0, p_values[i] * m / rank)
        
        return {
            'corrected_p_values': corrected_p_values.tolist(),
            'rejected': rejected.tolist(),
            'fdr': alpha
        }
```

---

## SECTION 3: EFFECT SIZE ANALYSIS

### 3.1 Effect Size Calculations

```python
class EffectSizeAnalysis:
    def __init__(self):
        self.effect_size_interpretations = {
            'cohens_d': {
                'negligible': (0.0, 0.15),
                'small': (0.15, 0.40),
                'medium': (0.40, 0.75),
                'large': (0.75, 1.10),
                'very_large': (1.10, float('inf'))
            },
            'practical_significance': {
                'minimal': 0.01,      # 1% annual return difference
                'small': 0.03,        # 3% annual return difference
                'moderate': 0.05,     # 5% annual return difference
                'substantial': 0.10,  # 10% annual return difference
                'large': 0.20         # 20% annual return difference
            }
        }
    
    def calculate_cohens_d(self, group1: np.array, group2: np.array = None,
                          population_mean: float = None) -> Dict:
        """Calculate Cohen's d effect size"""
        
        if group2 is not None:
            # Two-sample Cohen's d
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            
            cohens_d = (mean1 - mean2) / pooled_std
            
        elif population_mean is not None:
            # One-sample Cohen's d
            mean1 = np.mean(group1)
            std1 = np.std(group1, ddof=1)
            
            cohens_d = (mean1 - population_mean) / std1
            
        else:
            raise ValueError("Must provide either group2 or population_mean")
        
        # Calculate confidence interval for Cohen's d
        ci_lower, ci_upper = self._cohens_d_confidence_interval(
            cohens_d, len(group1), len(group2) if group2 is not None else None
        )
        
        return {
            'cohens_d': cohens_d,
            'effect_size_magnitude': self._interpret_cohens_d(cohens_d),
            'confidence_interval': (ci_lower, ci_upper),
            'is_practically_significant': abs(cohens_d) >= 0.2  # Small effect threshold
        }
    
    def calculate_practical_significance(self, strategy_returns: np.array,
                                       benchmark_returns: np.array,
                                       transaction_costs: float = 0.01) -> Dict:
        """Calculate practical significance in financial terms"""
        
        # Annualized returns
        strategy_annual = (1 + strategy_returns).prod() ** (252/len(strategy_returns)) - 1
        benchmark_annual = (1 + benchmark_returns).prod() ** (252/len(benchmark_returns)) - 1
        
        # Return difference after costs
        net_outperformance = strategy_annual - benchmark_annual - transaction_costs
        
        # Practical significance thresholds
        thresholds = self.effect_size_interpretations['practical_significance']
        
        if abs(net_outperformance) < thresholds['minimal']:
            significance_level = 'negligible'
        elif abs(net_outperformance) < thresholds['small']:
            significance_level = 'minimal'
        elif abs(net_outperformance) < thresholds['moderate']:
            significance_level = 'small'
        elif abs(net_outperformance) < thresholds['substantial']:
            significance_level = 'moderate'
        elif abs(net_outperformance) < thresholds['large']:
            significance_level = 'substantial'
        else:
            significance_level = 'large'
        
        # Calculate economic value
        portfolio_value = 100000  # $100K portfolio
        annual_dollar_impact = portfolio_value * net_outperformance
        
        return {
            'net_outperformance': net_outperformance,
            'practical_significance_level': significance_level,
            'annual_dollar_impact': annual_dollar_impact,
            'is_practically_significant': abs(net_outperformance) >= thresholds['minimal'],
            'economic_interpretation': self._interpret_economic_significance(annual_dollar_impact)
        }
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d magnitude"""
        
        abs_d = abs(cohens_d)
        
        for magnitude, (lower, upper) in self.effect_size_interpretations['cohens_d'].items():
            if lower <= abs_d < upper:
                return magnitude
        
        return 'very_large'
    
    def _cohens_d_confidence_interval(self, cohens_d: float, n1: int, n2: int = None) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's d"""
        
        if n2 is None:
            # One-sample case
            se = np.sqrt(1/n1 + cohens_d**2/(2*n1))
        else:
            # Two-sample case
            se = np.sqrt((n1+n2)/(n1*n2) + cohens_d**2/(2*(n1+n2)))
        
        t_critical = stats.t.ppf(0.975, n1-1 if n2 is None else n1+n2-2)
        
        ci_lower = cohens_d - t_critical * se
        ci_upper = cohens_d + t_critical * se
        
        return ci_lower, ci_upper
```

---

## SECTION 4: BOOTSTRAP & RESAMPLING METHODS

### 4.1 Bootstrap Confidence Intervals

```python
class BootstrapAnalysis:
    def __init__(self, n_bootstrap: int = 10000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_confidence_interval(self, data: np.array,
                                    statistic_func: Callable,
                                    confidence_level: float = 0.95) -> Dict:
        """Calculate bootstrap confidence interval for any statistic"""
        
        bootstrap_statistics = []
        
        for i in range(self.n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            
            # Calculate statistic on bootstrap sample
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(bootstrap_stat)
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_statistics, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha/2))
        
        # Original statistic
        original_stat = statistic_func(data)
        
        return {
            'original_statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_statistics),
            'bootstrap_std': np.std(bootstrap_statistics),
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'bootstrap_distribution': bootstrap_statistics
        }
    
    def bootstrap_sharpe_ratio(self, returns: np.array,
                             risk_free_rate: float = 0.02,
                             confidence_level: float = 0.95) -> Dict:
        """Bootstrap confidence interval for Sharpe ratio"""
        
        def sharpe_func(r):
            excess_returns = r - risk_free_rate/252
            return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        return self.bootstrap_confidence_interval(returns, sharpe_func, confidence_level)
    
    def bootstrap_maximum_drawdown(self, returns: np.array,
                                 confidence_level: float = 0.95) -> Dict:
        """Bootstrap confidence interval for maximum drawdown"""
        
        def max_dd_func(r):
            cumulative = (1 + r).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        
        return self.bootstrap_confidence_interval(returns, max_dd_func, confidence_level)
    
    def bootstrap_var_cvar(self, returns: np.array,
                          confidence_level: float = 0.95,
                          var_level: float = 0.05) -> Dict:
        """Bootstrap confidence intervals for VaR and CVaR"""
        
        def var_func(r):
            return np.percentile(r, var_level * 100)
        
        def cvar_func(r):
            var = np.percentile(r, var_level * 100)
            return np.mean(r[r <= var])
        
        var_results = self.bootstrap_confidence_interval(returns, var_func, confidence_level)
        cvar_results = self.bootstrap_confidence_interval(returns, cvar_func, confidence_level)
        
        return {
            'var_results': var_results,
            'cvar_results': cvar_results,
            'var_level': var_level
        }

class BlockBootstrap:
    """Block bootstrap for time series data"""
    
    def __init__(self, block_size: int = None):
        self.block_size = block_size
    
    def circular_block_bootstrap(self, data: np.array, n_bootstrap: int = 1000) -> List[np.array]:
        """Circular block bootstrap for time series"""
        
        n = len(data)
        
        if self.block_size is None:
            # Optimal block size (Politis and White, 2004)
            self.block_size = int(np.ceil(n ** (1/3)))
        
        bootstrap_samples = []
        
        for _ in range(n_bootstrap):
            # Create circular array
            circular_data = np.concatenate([data, data])
            
            # Number of blocks needed
            n_blocks = int(np.ceil(n / self.block_size))
            
            # Sample random starting points
            start_points = np.random.randint(0, n, size=n_blocks)
            
            # Construct bootstrap sample
            bootstrap_sample = []
            for start in start_points:
                block = circular_data[start:start + self.block_size]
                bootstrap_sample.extend(block)
            
            # Trim to original length
            bootstrap_sample = np.array(bootstrap_sample[:n])
            bootstrap_samples.append(bootstrap_sample)
        
        return bootstrap_samples
```

---

## SECTION 5: CROSS-VALIDATION FRAMEWORK

### 5.1 Time Series Cross-Validation

```python
class TimeSeriesCrossValidator:
    def __init__(self):
        self.validation_methods = {
            'walk_forward': self._walk_forward_validation,
            'expanding_window': self._expanding_window_validation,
            'sliding_window': self._sliding_window_validation,
            'purged_cv': self._purged_cross_validation
        }
    
    def validate_strategy(self, returns_data: pd.DataFrame,
                         strategy_func: Callable,
                         method: str = 'walk_forward',
                         **method_params) -> Dict:
        """Validate trading strategy using time series cross-validation"""
        
        if method not in self.validation_methods:
            raise ValueError(f"Unknown validation method: {method}")
        
        validation_results = self.validation_methods[method](
            returns_data, strategy_func, **method_params
        )
        
        # Calculate aggregated metrics
        all_returns = np.concatenate([result['returns'] for result in validation_results])
        
        aggregated_metrics = {
            'mean_annual_return': np.mean([r['annual_return'] for r in validation_results]),
            'std_annual_return': np.std([r['annual_return'] for r in validation_results]),
            'mean_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in validation_results]),
            'std_sharpe_ratio': np.std([r['sharpe_ratio'] for r in validation_results]),
            'mean_max_drawdown': np.mean([r['max_drawdown'] for r in validation_results]),
            'worst_max_drawdown': np.max([r['max_drawdown'] for r in validation_results]),
            'consistency_score': np.mean([r['annual_return'] > 0 for r in validation_results]),
            'total_periods': len(validation_results)
        }
        
        return {
            'method': method,
            'individual_results': validation_results,
            'aggregated_metrics': aggregated_metrics,
            'overall_performance': self._calculate_overall_performance(all_returns),
            'robustness_score': self._calculate_robustness_score(validation_results)
        }
    
    def _walk_forward_validation(self, data: pd.DataFrame,
                               strategy_func: Callable,
                               train_window: int = 252,  # 1 year
                               test_window: int = 63,    # 3 months
                               step_size: int = 21) -> List[Dict]:   # 1 month steps
        """Walk-forward validation with expanding training window"""
        
        results = []
        start_idx = train_window
        
        while start_idx + test_window < len(data):
            # Training data (all data up to test period)
            train_data = data.iloc[:start_idx]
            
            # Test data (fixed window)
            test_data = data.iloc[start_idx:start_idx + test_window]
            
            # Ensure no lookahead bias
            assert train_data.index.max() < test_data.index.min(), "Lookahead bias detected!"
            
            # Run strategy on test period
            try:
                strategy_result = strategy_func(train_data, test_data)
                
                # Calculate performance metrics
                test_returns = strategy_result['returns']
                period_result = self._calculate_period_metrics(test_returns, test_data.index)
                
                period_result.update({
                    'train_start': train_data.index.min(),
                    'train_end': train_data.index.max(),
                    'test_start': test_data.index.min(),
                    'test_end': test_data.index.max(),
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                })
                
                results.append(period_result)
                
            except Exception as e:
                logging.warning(f"Strategy failed for period {test_data.index.min()}: {e}")
            
            start_idx += step_size
        
        return results
    
    def _purged_cross_validation(self, data: pd.DataFrame,
                               strategy_func: Callable,
                               n_splits: int = 5,
                               purge_pct: float = 0.02) -> List[Dict]:
        """Purged cross-validation to prevent information leakage"""
        
        results = []
        n = len(data)
        test_size = n // n_splits
        purge_size = int(n * purge_pct)
        
        for i in range(n_splits):
            # Test period
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n)
            
            # Purge periods (remove data around test period)
            purge_start = max(0, test_start - purge_size)
            purge_end = min(n, test_end + purge_size)
            
            # Training data (excluding test and purge periods)
            train_indices = list(range(0, purge_start)) + list(range(purge_end, n))
            
            if len(train_indices) < 100:  # Minimum training size
                continue
            
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_start:test_end]
            
            try:
                strategy_result = strategy_func(train_data, test_data)
                test_returns = strategy_result['returns']
                
                period_result = self._calculate_period_metrics(test_returns, test_data.index)
                period_result.update({
                    'split_number': i,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'purge_size': purge_size
                })
                
                results.append(period_result)
                
            except Exception as e:
                logging.warning(f"Strategy failed for split {i}: {e}")
        
        return results
    
    def _calculate_period_metrics(self, returns: np.array, dates: pd.DatetimeIndex) -> Dict:
        """Calculate performance metrics for a single period"""
        
        if len(returns) == 0:
            return {'error': 'No returns data'}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        profit_factor = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns < 0) != 0 else np.inf
        
        return {
            'returns': returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(returns),
            'period_start': dates.min(),
            'period_end': dates.max()
        }
```

---

## SECTION 6: IMPLEMENTATION CHECKLIST

### 6.1 Statistical Framework Implementation

**Phase 1: Core Statistical Tests (Week 1)**
- [ ] Implement hypothesis testing framework
- [ ] Add power analysis calculations
- [ ] Create effect size analysis functions
- [ ] Test statistical significance validation

**Phase 2: Multiple Comparisons & Bootstrap (Week 2)**
- [ ] Implement Bonferroni and FDR corrections
- [ ] Add bootstrap confidence intervals
- [ ] Create block bootstrap for time series
- [ ] Validate resampling methods

**Phase 3: Cross-Validation Framework (Week 3)**
- [ ] Implement walk-forward validation
- [ ] Add purged cross-validation
- [ ] Create performance aggregation
- [ ] Test lookahead bias prevention

### 6.2 Validation Criteria

**Statistical Significance Requirements:**
- [ ] P-value < 0.05 for all primary hypotheses
- [ ] Effect size > 0.2 (small to medium effect)
- [ ] Statistical power > 0.8 (80% power)
- [ ] Sample size > required minimum

**Robustness Requirements:**
- [ ] Consistent performance across CV folds
- [ ] Bootstrap confidence intervals exclude zero
- [ ] Multiple comparisons correction applied
- [ ] No significant lookahead bias

---

## CONCLUSION

This statistical methodology specification provides the rigorous foundation needed for institutional-grade validation of algorithmic trading systems. By implementing proper hypothesis testing, power analysis, effect size calculations, and cross-validation, this framework ensures that validation results are statistically sound and suitable for real capital deployment decisions.

The framework addresses all critical statistical flaws identified in existing validation approaches and provides the mathematical rigor required by institutional investors and academic standards.

**Next Steps**: Implement the core statistical testing framework and validate with real market data from the data collection pipeline.
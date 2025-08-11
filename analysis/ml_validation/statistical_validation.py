#!/usr/bin/env python3
"""
Statistical Validation Framework for Algorithmic Trading System
"""

import numpy as np
import json
import statistics
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import scipy.stats as stats


class StatisticalValidation:
    """
    Comprehensive statistical validation framework for trading system performance.
    """
    
    def __init__(self, trade_data_file: str = "/home/eddy/Hyper/data/trade_outcomes.json"):
        self.trade_data_file = trade_data_file
        self.trades = self._load_trades()
        self.initial_capital = 100000
        
    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trade data from JSON file"""
        try:
            with open(self.trade_data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Trade data file {self.trade_data_file} not found")
            return []
            
    def validate_sample_size(self, desired_confidence: float = 0.95, 
                           margin_of_error: float = 0.05) -> Dict[str, Any]:
        """
        Validate whether we have sufficient sample size for statistical significance.
        
        Args:
            desired_confidence: Desired confidence level (default 95%)
            margin_of_error: Acceptable margin of error (default 5%)
        """
        current_sample_size = len(self.trades)
        
        # For proportion estimates (win rate), calculate required sample size
        # Using conservative p=0.5 for maximum required sample size
        z_score = stats.norm.ppf(1 - (1 - desired_confidence) / 2)
        required_sample_size = int(np.ceil((z_score ** 2 * 0.5 * 0.5) / (margin_of_error ** 2)))
        
        # Calculate actual margin of error with current sample size
        if current_sample_size > 0:
            actual_margin_of_error = np.sqrt((0.5 * 0.5 * z_score ** 2) / current_sample_size)
        else:
            actual_margin_of_error = float('inf')
        
        return {
            "current_sample_size": current_sample_size,
            "required_sample_size": required_sample_size,
            "is_sufficient": current_sample_size >= required_sample_size,
            "statistical_power": min(current_sample_size / required_sample_size, 1.0),
            "actual_margin_of_error": actual_margin_of_error,
            "desired_margin_of_error": margin_of_error,
            "confidence_level": desired_confidence
        }
    
    def test_performance_significance(self, null_hypothesis_win_rate: float = 0.5) -> Dict[str, Any]:
        """
        Test whether system performance is significantly different from random.
        
        Args:
            null_hypothesis_win_rate: Expected win rate under null hypothesis (default 50%)
        """
        if not self.trades:
            return {"error": "No trade data available"}
        
        # Calculate actual win rate
        wins = sum(1 for trade in self.trades if trade.get('result') == 'win')
        total_trades = len(self.trades)
        actual_win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Binomial test for win rate significance
        p_value = stats.binom_test(wins, total_trades, null_hypothesis_win_rate, alternative='two-sided')
        
        # Calculate confidence interval for win rate
        if total_trades > 0:
            ci_lower, ci_upper = stats.binom.interval(0.95, total_trades, actual_win_rate)
            ci_lower = ci_lower / total_trades
            ci_upper = ci_upper / total_trades
        else:
            ci_lower = ci_upper = 0
        
        return {
            "actual_win_rate": actual_win_rate,
            "null_hypothesis_win_rate": null_hypothesis_win_rate,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_interval_95": [ci_lower, ci_upper],
            "effect_size": actual_win_rate - null_hypothesis_win_rate,
            "interpretation": self._interpret_significance(p_value, actual_win_rate - null_hypothesis_win_rate)
        }
    
    def validate_sentiment_accuracy(self, claimed_accuracy: float = 0.80) -> Dict[str, Any]:
        """
        Validate claimed sentiment analysis accuracy against actual performance.
        """
        accuracies = [
            trade.get('news_accuracy', 0) for trade in self.trades 
            if 'news_accuracy' in trade and trade['news_accuracy'] is not None
        ]
        
        if not accuracies:
            return {"error": "No sentiment accuracy data available"}
        
        actual_accuracy = statistics.mean(accuracies)
        
        # One-sample t-test against claimed accuracy
        t_stat, p_value = stats.ttest_1samp(accuracies, claimed_accuracy)
        
        # Calculate confidence interval
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(accuracies) - 1, 
            loc=actual_accuracy, 
            scale=stats.sem(accuracies)
        )
        
        return {
            "claimed_accuracy": claimed_accuracy,
            "actual_accuracy": actual_accuracy,
            "sample_size": len(accuracies),
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_claim_supported": p_value > 0.05 and actual_accuracy >= claimed_accuracy * 0.95,
            "confidence_interval_95": [ci_lower, ci_upper],
            "accuracy_difference": actual_accuracy - claimed_accuracy,
            "interpretation": self._interpret_accuracy_test(p_value, actual_accuracy, claimed_accuracy)
        }
    
    def calculate_robust_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics with proper statistical measures.
        """
        if not self.trades:
            return {"error": "No trade data available"}
        
        # Extract P&L values
        pnl_values = [trade.get('pnl_absolute', 0) for trade in self.trades]
        
        # Basic metrics
        total_pnl = sum(pnl_values)
        wins = sum(1 for pnl in pnl_values if pnl > 0)
        losses = sum(1 for pnl in pnl_values if pnl < 0)
        win_rate = wins / len(pnl_values)
        
        # Risk metrics
        sharpe_ratio = self._calculate_corrected_sharpe_ratio(pnl_values)
        max_drawdown = self._calculate_corrected_max_drawdown(pnl_values)
        
        # Confidence intervals
        win_rate_ci = self._calculate_proportion_ci(wins, len(pnl_values))
        avg_pnl_ci = self._calculate_mean_ci(pnl_values)
        
        return {
            "total_trades": len(pnl_values),
            "total_pnl": total_pnl,
            "average_pnl": statistics.mean(pnl_values),
            "win_rate": win_rate,
            "win_rate_ci_95": win_rate_ci,
            "average_pnl_ci_95": avg_pnl_ci,
            "sharpe_ratio_corrected": sharpe_ratio,
            "max_drawdown_corrected": max_drawdown,
            "profit_factor": self._calculate_profit_factor(pnl_values),
            "statistical_validation": {
                "sample_size_adequate": len(pnl_values) >= 100,
                "normality_test": self._test_normality(pnl_values),
                "stationarity_test": self._test_stationarity(pnl_values)
            }
        }
    
    def monte_carlo_validation(self, n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation to validate system robustness.
        """
        if not self.trades:
            return {"error": "No trade data available"}
        
        pnl_values = [trade.get('pnl_absolute', 0) for trade in self.trades]
        
        # Bootstrap sampling for confidence intervals
        simulation_results = []
        
        for _ in range(n_simulations):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(pnl_values, size=len(pnl_values), replace=True)
            
            # Calculate metrics for this sample
            total_pnl = sum(bootstrap_sample)
            win_rate = sum(1 for pnl in bootstrap_sample if pnl > 0) / len(bootstrap_sample)
            sharpe = self._calculate_corrected_sharpe_ratio(bootstrap_sample)
            max_dd = self._calculate_corrected_max_drawdown(bootstrap_sample)
            
            simulation_results.append({
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            })
        
        # Calculate confidence intervals from simulations
        metrics = ['total_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown']
        confidence_intervals = {}
        
        for metric in metrics:
            values = [result[metric] for result in simulation_results]
            confidence_intervals[metric] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'ci_95': [np.percentile(values, 2.5), np.percentile(values, 97.5)],
                'ci_99': [np.percentile(values, 0.5), np.percentile(values, 99.5)]
            }
        
        return {
            "n_simulations": n_simulations,
            "confidence_intervals": confidence_intervals,
            "robustness_score": self._calculate_robustness_score(confidence_intervals)
        }
    
    def _calculate_corrected_sharpe_ratio(self, pnl_values: List[float], 
                                        risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio with proper risk-free rate and annualization"""
        if len(pnl_values) < 2:
            return 0.0
        
        returns = [pnl / self.initial_capital for pnl in pnl_values]
        
        if statistics.stdev(returns) == 0:
            return 0.0
        
        daily_risk_free = risk_free_rate / 252
        excess_returns = [r - daily_risk_free for r in returns]
        
        mean_excess = statistics.mean(excess_returns)
        std_returns = statistics.stdev(returns)
        
        # Annualize (approximate)
        sharpe_ratio = (mean_excess / std_returns) * np.sqrt(252) if std_returns != 0 else 0
        
        return sharpe_ratio
    
    def _calculate_corrected_max_drawdown(self, pnl_values: List[float]) -> float:
        """Calculate maximum drawdown correctly"""
        if not pnl_values:
            return 0.0
        
        cumulative_pnl = [self.initial_capital]
        running_total = self.initial_capital
        
        for pnl in pnl_values:
            running_total += pnl
            cumulative_pnl.append(running_total)
        
        max_dd = 0.0
        peak = cumulative_pnl[0]
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            
            if peak > 0:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_profit_factor(self, pnl_values: List[float]) -> float:
        """Calculate profit factor"""
        wins = [pnl for pnl in pnl_values if pnl > 0]
        losses = [abs(pnl) for pnl in pnl_values if pnl < 0]
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1
        
        return total_wins / total_losses if total_losses > 0 else float('inf')
    
    def _calculate_proportion_ci(self, successes: int, total: int, 
                               confidence: float = 0.95) -> List[float]:
        """Calculate confidence interval for a proportion"""
        if total == 0:
            return [0, 0]
        
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)
        p = successes / total
        
        margin = z * np.sqrt(p * (1 - p) / total)
        
        return [max(0, p - margin), min(1, p + margin)]
    
    def _calculate_mean_ci(self, values: List[float], confidence: float = 0.95) -> List[float]:
        """Calculate confidence interval for a mean"""
        if len(values) < 2:
            return [0, 0]
        
        mean_val = statistics.mean(values)
        sem = stats.sem(values)
        
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean_val, scale=sem)
        
        return list(ci)
    
    def _test_normality(self, values: List[float]) -> Dict[str, Any]:
        """Test if returns are normally distributed"""
        if len(values) < 8:
            return {"test": "insufficient_data"}
        
        statistic, p_value = stats.shapiro(values)
        
        return {
            "test": "shapiro_wilk",
            "statistic": statistic,
            "p_value": p_value,
            "is_normal": p_value > 0.05
        }
    
    def _test_stationarity(self, values: List[float]) -> Dict[str, Any]:
        """Test if returns are stationary (simplified)"""
        if len(values) < 10:
            return {"test": "insufficient_data"}
        
        # Simple test: compare first half vs second half means
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]
        
        statistic, p_value = stats.ttest_ind(first_half, second_half)
        
        return {
            "test": "mean_stationarity",
            "statistic": statistic,
            "p_value": p_value,
            "is_stationary": p_value > 0.05
        }
    
    def _calculate_robustness_score(self, confidence_intervals: Dict[str, Any]) -> float:
        """Calculate overall robustness score"""
        scores = []
        
        # Sharpe ratio stability (higher is better)
        sharpe_std = confidence_intervals['sharpe_ratio']['std']
        sharpe_score = 1 / (1 + sharpe_std) if sharpe_std > 0 else 1
        scores.append(sharpe_score)
        
        # Win rate stability (lower variance is better)
        win_rate_std = confidence_intervals['win_rate']['std']
        win_rate_score = 1 / (1 + win_rate_std * 10) if win_rate_std > 0 else 1
        scores.append(win_rate_score)
        
        # Drawdown stability (lower is better)
        dd_mean = confidence_intervals['max_drawdown']['mean']
        dd_score = max(0, 1 - dd_mean)
        scores.append(dd_score)
        
        return statistics.mean(scores)
    
    def _interpret_significance(self, p_value: float, effect_size: float) -> str:
        """Interpret statistical significance results"""
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        direction = "positive" if effect_size > 0 else "negative"
        
        return f"The performance difference is {significance} ({direction} effect)"
    
    def _interpret_accuracy_test(self, p_value: float, actual: float, claimed: float) -> str:
        """Interpret sentiment accuracy test results"""
        if p_value > 0.05:
            if actual >= claimed * 0.95:
                return "Claimed accuracy is statistically supported"
            else:
                return "Actual accuracy is lower than claimed, but difference not significant"
        else:
            if actual < claimed:
                return "Actual accuracy is significantly lower than claimed"
            else:
                return "Actual accuracy differs significantly from claimed value"


def run_comprehensive_validation():
    """Run comprehensive statistical validation"""
    validator = StatisticalValidation()
    
    print("=" * 60)
    print("COMPREHENSIVE STATISTICAL VALIDATION REPORT")
    print("=" * 60)
    
    # Sample size validation
    print("\n1. SAMPLE SIZE VALIDATION")
    print("-" * 30)
    sample_validation = validator.validate_sample_size()
    print(f"Current sample size: {sample_validation['current_sample_size']}")
    print(f"Required sample size (95% confidence): {sample_validation['required_sample_size']}")
    print(f"Statistical power: {sample_validation['statistical_power']:.2%}")
    print(f"Sample size adequate: {sample_validation['is_sufficient']}")
    
    # Performance significance testing
    print("\n2. PERFORMANCE SIGNIFICANCE TESTING")
    print("-" * 40)
    performance_test = validator.test_performance_significance()
    if 'error' not in performance_test:
        print(f"Actual win rate: {performance_test['actual_win_rate']:.2%}")
        print(f"P-value vs random: {performance_test['p_value']:.4f}")
        print(f"Significant improvement: {performance_test['is_significant']}")
        print(f"95% Confidence interval: [{performance_test['confidence_interval_95'][0]:.2%}, {performance_test['confidence_interval_95'][1]:.2%}]")
        print(f"Interpretation: {performance_test['interpretation']}")
    
    # Sentiment accuracy validation
    print("\n3. SENTIMENT ACCURACY VALIDATION")
    print("-" * 40)
    sentiment_test = validator.validate_sentiment_accuracy()
    if 'error' not in sentiment_test:
        print(f"Claimed accuracy: {sentiment_test['claimed_accuracy']:.2%}")
        print(f"Actual accuracy: {sentiment_test['actual_accuracy']:.2%}")
        print(f"P-value: {sentiment_test['p_value']:.4f}")
        print(f"Claim supported: {sentiment_test['is_claim_supported']}")
        print(f"Interpretation: {sentiment_test['interpretation']}")
    
    # Robust performance metrics
    print("\n4. ROBUST PERFORMANCE METRICS")
    print("-" * 40)
    metrics = validator.calculate_robust_performance_metrics()
    if 'error' not in metrics:
        print(f"Corrected Sharpe ratio: {metrics['sharpe_ratio_corrected']:.3f}")
        print(f"Corrected max drawdown: {metrics['max_drawdown_corrected']:.2%}")
        print(f"Win rate: {metrics['win_rate']:.2%} (95% CI: [{metrics['win_rate_ci_95'][0]:.2%}, {metrics['win_rate_ci_95'][1]:.2%}])")
        print(f"Sample size adequate: {metrics['statistical_validation']['sample_size_adequate']}")
    
    # Monte Carlo validation
    print("\n5. MONTE CARLO ROBUSTNESS TEST")
    print("-" * 40)
    print("Running 1000 bootstrap simulations...")
    mc_results = validator.monte_carlo_validation(1000)
    if 'error' not in mc_results:
        print(f"Robustness score: {mc_results['robustness_score']:.3f}")
        print(f"Win rate 95% CI: [{mc_results['confidence_intervals']['win_rate']['ci_95'][0]:.2%}, {mc_results['confidence_intervals']['win_rate']['ci_95'][1]:.2%}]")
        print(f"Sharpe ratio 95% CI: [{mc_results['confidence_intervals']['sharpe_ratio']['ci_95'][0]:.3f}, {mc_results['confidence_intervals']['sharpe_ratio']['ci_95'][1]:.3f}]")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_comprehensive_validation()
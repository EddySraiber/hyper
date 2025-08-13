"""
Statistical Validation Framework - Validates trading system performance with statistical rigor

Features:
- Minimum sample size validation using Cohen's formula
- Statistical significance testing (z-tests, t-tests)
- Sharpe ratio calculations with proper annualization
- Confidence intervals for performance metrics
- Kelly Criterion validation
- Risk-adjusted return analysis
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from scipy import stats
import math


@dataclass
class StatisticalResults:
    """Results from statistical analysis"""
    sample_size: int
    min_required_sample: int
    is_statistically_significant: bool
    confidence_level: float
    
    # Performance metrics
    win_rate: float
    win_rate_ci: Tuple[float, float]  # Confidence interval
    expected_return: float
    sharpe_ratio: float
    sharpe_ci: Tuple[float, float]
    
    # Risk metrics
    max_drawdown: float
    volatility: float
    var_95: float  # Value at Risk 95%
    
    # Kelly Criterion
    kelly_fraction: float
    kelly_applicable: bool
    
    # Statistical tests
    z_score: float
    p_value: float
    significance_level: float
    
    # Recommendations
    recommendations: List[str]
    warnings: List[str]


class StatisticalValidator:
    """
    Provides statistical validation and performance analysis for trading systems
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__name__)
        
        # Configuration
        self.confidence_level = config.get("confidence_level", 0.95)
        self.significance_level = config.get("significance_level", 0.05)
        self.min_effect_size = config.get("min_effect_size", 0.1)  # 10% difference from random
        self.risk_free_rate = config.get("risk_free_rate", 0.02)  # 2% annual
        self.trading_days_per_year = config.get("trading_days_per_year", 252)
        
        # Thresholds
        self.min_sharpe_threshold = config.get("min_sharpe_threshold", 1.0)
        self.max_drawdown_threshold = config.get("max_drawdown_threshold", 0.20)  # 20%
        self.min_kelly_threshold = config.get("min_kelly_threshold", 0.01)
        
        self.logger.info("Statistical Validator initialized")
    
    def calculate_minimum_sample_size(self, 
                                    expected_win_rate: float = 0.6,
                                    null_win_rate: float = 0.5,
                                    power: float = 0.8) -> int:
        """
        Calculate minimum sample size needed for statistical significance
        using Cohen's formula for proportion testing
        """
        alpha = self.significance_level
        beta = 1 - power
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p = (expected_win_rate + null_win_rate) / 2
        delta = abs(expected_win_rate - null_win_rate)
        
        n = ((z_alpha + z_beta)**2 * p * (1 - p)) / (delta**2)
        
        return int(math.ceil(n))
    
    def validate_trading_performance(self, trades_data: List[Dict[str, Any]]) -> StatisticalResults:
        """
        Comprehensive statistical validation of trading performance
        """
        if not trades_data:
            return self._create_empty_results()
        
        sample_size = len(trades_data)
        min_required = self.calculate_minimum_sample_size()
        
        # Extract trade results
        returns = []
        wins = 0
        losses = 0
        
        for trade in trades_data:
            pnl = float(trade.get('realized_pnl', 0))
            returns.append(pnl)
            
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
        
        # Calculate basic metrics
        win_rate = wins / sample_size if sample_size > 0 else 0
        expected_return = np.mean(returns) if returns else 0
        volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0
        
        # Statistical significance test (comparing win rate to 50%)
        z_score, p_value = self._test_win_rate_significance(wins, sample_size)
        is_significant = p_value < self.significance_level and sample_size >= min_required
        
        # Confidence intervals
        win_rate_ci = self._calculate_proportion_ci(wins, sample_size)
        
        # Sharpe ratio calculation
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sharpe_ci = self._calculate_sharpe_ci(returns)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        var_95 = self._calculate_var(returns, 0.95)
        
        # Kelly Criterion
        kelly_fraction, kelly_applicable = self._calculate_kelly_criterion(returns)
        
        # Generate recommendations and warnings
        recommendations, warnings = self._generate_insights(
            sample_size, min_required, win_rate, sharpe_ratio, max_drawdown, kelly_fraction
        )
        
        return StatisticalResults(
            sample_size=sample_size,
            min_required_sample=min_required,
            is_statistically_significant=is_significant,
            confidence_level=self.confidence_level,
            win_rate=win_rate,
            win_rate_ci=win_rate_ci,
            expected_return=expected_return,
            sharpe_ratio=sharpe_ratio,
            sharpe_ci=sharpe_ci,
            max_drawdown=max_drawdown,
            volatility=volatility,
            var_95=var_95,
            kelly_fraction=kelly_fraction,
            kelly_applicable=kelly_applicable,
            z_score=z_score,
            p_value=p_value,
            significance_level=self.significance_level,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _test_win_rate_significance(self, wins: int, total: int) -> Tuple[float, float]:
        """Test if win rate is significantly different from 50% using z-test"""
        if total == 0:
            return 0.0, 1.0
        
        p_observed = wins / total
        p_null = 0.5
        
        # Standard error under null hypothesis
        se = math.sqrt(p_null * (1 - p_null) / total)
        
        if se == 0:
            return 0.0, 1.0
        
        z_score = (p_observed - p_null) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return z_score, p_value
    
    def _calculate_proportion_ci(self, successes: int, total: int) -> Tuple[float, float]:
        """Calculate confidence interval for proportion using Wilson score"""
        if total == 0:
            return (0.0, 0.0)
        
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        p = successes / total
        n = total
        
        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * math.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualize assuming daily returns
        excess_return = mean_return - (self.risk_free_rate / self.trading_days_per_year)
        sharpe = (excess_return / std_return) * math.sqrt(self.trading_days_per_year)
        
        return sharpe
    
    def _calculate_sharpe_ci(self, returns: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for Sharpe ratio using Jobson-Korkie method"""
        if not returns or len(returns) < 3:
            return (0.0, 0.0)
        
        n = len(returns)
        sharpe = self._calculate_sharpe_ratio(returns)
        
        # Standard error approximation
        se = math.sqrt((1 + sharpe**2 / 2) / n)
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        margin = z * se
        return (sharpe - margin, sharpe + margin)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns series"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        
        return float(np.max(drawdown))
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk at specified confidence level"""
        if not returns:
            return 0.0
        
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_kelly_criterion(self, returns: List[float]) -> Tuple[float, bool]:
        """Calculate Kelly Criterion fraction"""
        if not returns:
            return 0.0, False
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return 0.0, False
        
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        win_prob = len(wins) / len(returns)
        loss_prob = len(losses) / len(returns)
        
        if avg_loss == 0:
            return 0.0, False
        
        # Kelly fraction: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_prob, q = loss_prob
        b = avg_win / avg_loss
        kelly_fraction = (b * win_prob - loss_prob) / b
        
        return kelly_fraction, True
    
    def _generate_insights(self, sample_size: int, min_required: int, win_rate: float,
                         sharpe_ratio: float, max_drawdown: float, kelly_fraction: float) -> Tuple[List[str], List[str]]:
        """Generate actionable recommendations and warnings"""
        recommendations = []
        warnings = []
        
        # Sample size analysis
        if sample_size < min_required:
            warnings.append(f"Insufficient sample size: {sample_size} < {min_required} required")
            recommendations.append(f"Continue trading to reach minimum {min_required} trades for statistical significance")
        
        # Performance analysis
        if win_rate < 0.45:
            warnings.append(f"Low win rate: {win_rate:.1%} suggests system may be worse than random")
            recommendations.append("Review trading logic and consider reducing position sizes")
        elif win_rate > 0.65:
            recommendations.append(f"Strong win rate: {win_rate:.1%} indicates good signal quality")
        
        # Sharpe ratio analysis
        if sharpe_ratio < 0:
            warnings.append(f"Negative Sharpe ratio: {sharpe_ratio:.2f} indicates poor risk-adjusted returns")
            recommendations.append("Consider stopping trading until system improvements are made")
        elif sharpe_ratio < self.min_sharpe_threshold:
            warnings.append(f"Low Sharpe ratio: {sharpe_ratio:.2f} below threshold {self.min_sharpe_threshold}")
            recommendations.append("Improve risk management or reduce trading frequency")
        elif sharpe_ratio > 2.0:
            recommendations.append(f"Excellent Sharpe ratio: {sharpe_ratio:.2f} indicates strong performance")
        
        # Drawdown analysis
        if max_drawdown > self.max_drawdown_threshold:
            warnings.append(f"High drawdown: {max_drawdown:.1%} exceeds threshold {self.max_drawdown_threshold:.1%}")
            recommendations.append("Implement tighter stop-losses or reduce position sizes")
        
        # Kelly Criterion analysis
        if kelly_fraction < 0:
            warnings.append("Negative Kelly fraction suggests system has negative expectancy")
            recommendations.append("Do not increase position sizes - system may be unprofitable")
        elif kelly_fraction > 0.25:
            warnings.append(f"High Kelly fraction: {kelly_fraction:.1%} may be too aggressive")
            recommendations.append("Consider using 25% of Kelly fraction for conservative sizing")
        
        return recommendations, warnings
    
    def _create_empty_results(self) -> StatisticalResults:
        """Create empty results for when no data is available"""
        return StatisticalResults(
            sample_size=0,
            min_required_sample=self.calculate_minimum_sample_size(),
            is_statistically_significant=False,
            confidence_level=self.confidence_level,
            win_rate=0.0,
            win_rate_ci=(0.0, 0.0),
            expected_return=0.0,
            sharpe_ratio=0.0,
            sharpe_ci=(0.0, 0.0),
            max_drawdown=0.0,
            volatility=0.0,
            var_95=0.0,
            kelly_fraction=0.0,
            kelly_applicable=False,
            z_score=0.0,
            p_value=1.0,
            significance_level=self.significance_level,
            recommendations=["No trading data available for analysis"],
            warnings=["System needs to execute trades before statistical validation can be performed"]
        )
    
    def generate_performance_report(self, results: StatisticalResults) -> str:
        """Generate a comprehensive performance report"""
        report = []
        report.append("📊 STATISTICAL PERFORMANCE ANALYSIS")
        report.append("=" * 50)
        report.append(f"Sample Size: {results.sample_size} (Min Required: {results.min_required_sample})")
        report.append(f"Statistical Significance: {'✅ YES' if results.is_statistically_significant else '❌ NO'}")
        report.append(f"Confidence Level: {results.confidence_level:.0%}")
        report.append("")
        
        report.append("🎯 PERFORMANCE METRICS:")
        report.append(f"Win Rate: {results.win_rate:.1%} (CI: {results.win_rate_ci[0]:.1%} - {results.win_rate_ci[1]:.1%})")
        report.append(f"Expected Return: {results.expected_return:.4f}")
        report.append(f"Sharpe Ratio: {results.sharpe_ratio:.2f} (CI: {results.sharpe_ci[0]:.2f} - {results.sharpe_ci[1]:.2f})")
        report.append("")
        
        report.append("⚠️ RISK METRICS:")
        report.append(f"Maximum Drawdown: {results.max_drawdown:.1%}")
        report.append(f"Volatility: {results.volatility:.4f}")
        report.append(f"Value at Risk (95%): {results.var_95:.4f}")
        report.append("")
        
        report.append("💰 POSITION SIZING:")
        report.append(f"Kelly Fraction: {results.kelly_fraction:.1%}")
        report.append(f"Kelly Applicable: {'✅ YES' if results.kelly_applicable else '❌ NO'}")
        report.append("")
        
        if results.recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for rec in results.recommendations:
                report.append(f"• {rec}")
            report.append("")
        
        if results.warnings:
            report.append("⚠️ WARNINGS:")
            for warning in results.warnings:
                report.append(f"• {warning}")
        
        return "\n".join(report)
#!/usr/bin/env python3
"""
Simplified Statistical Validation Framework using Python Standard Library Only
"""

import json
import math
import statistics
from typing import List, Dict, Any


def load_trades(file_path: str = "/home/eddy/Hyper/data/trade_outcomes.json") -> List[Dict[str, Any]]:
    """Load trade data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Trade data file {file_path} not found")
        return []


def validate_sample_size(n_trades: int, desired_confidence: float = 0.95, 
                        margin_of_error: float = 0.05) -> Dict[str, Any]:
    """Calculate required sample size for statistical significance"""
    
    # Z-score for 95% confidence
    z_score = 1.96  # Approximation for 95% confidence
    
    # Required sample size for proportion (using p=0.5 for maximum)
    required_sample_size = int(math.ceil((z_score ** 2 * 0.25) / (margin_of_error ** 2)))
    
    # Actual margin of error with current sample
    if n_trades > 0:
        actual_margin_of_error = math.sqrt((0.25 * z_score ** 2) / n_trades)
    else:
        actual_margin_of_error = float('inf')
    
    return {
        "current_sample_size": n_trades,
        "required_sample_size": required_sample_size,
        "is_sufficient": n_trades >= required_sample_size,
        "statistical_power": min(n_trades / required_sample_size, 1.0),
        "actual_margin_of_error": actual_margin_of_error,
        "interpretation": "Sufficient" if n_trades >= required_sample_size else "Insufficient"
    }


def test_win_rate_significance(wins: int, total: int, 
                              null_hypothesis: float = 0.5) -> Dict[str, Any]:
    """Test if win rate is significantly different from null hypothesis"""
    if total == 0:
        return {"error": "No trades"}
    
    actual_win_rate = wins / total
    
    # Simple z-test for proportions
    expected_wins = null_hypothesis * total
    std_error = math.sqrt(null_hypothesis * (1 - null_hypothesis) * total)
    
    if std_error > 0:
        z_score = (wins - expected_wins) / std_error
        # Approximate p-value (two-tailed)
        p_value = 2 * (1 - abs(z_score) / 2.58)  # Very rough approximation
        p_value = max(0, min(1, p_value))
    else:
        z_score = 0
        p_value = 1.0
    
    # 95% confidence interval for proportion
    margin = 1.96 * math.sqrt(actual_win_rate * (1 - actual_win_rate) / total)
    ci_lower = max(0, actual_win_rate - margin)
    ci_upper = min(1, actual_win_rate + margin)
    
    return {
        "actual_win_rate": actual_win_rate,
        "null_hypothesis_win_rate": null_hypothesis,
        "z_score": z_score,
        "p_value_approx": p_value,
        "is_significant_approx": abs(z_score) > 1.96,
        "confidence_interval_95": [ci_lower, ci_upper],
        "effect_size": actual_win_rate - null_hypothesis
    }


def calculate_corrected_sharpe_ratio(pnl_values: List[float], 
                                   initial_capital: float = 100000,
                                   risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio with proper risk-free rate"""
    if len(pnl_values) < 2:
        return 0.0
    
    # Convert to returns
    returns = [pnl / initial_capital for pnl in pnl_values]
    
    if statistics.stdev(returns) == 0:
        return 0.0
    
    # Daily risk-free rate
    daily_risk_free = risk_free_rate / 252
    
    # Excess returns
    excess_returns = [r - daily_risk_free for r in returns]
    
    mean_excess = statistics.mean(excess_returns)
    std_returns = statistics.stdev(returns)
    
    # Annualized Sharpe ratio
    sharpe_ratio = (mean_excess / std_returns) * math.sqrt(252) if std_returns != 0 else 0
    
    return sharpe_ratio


def calculate_corrected_max_drawdown(pnl_values: List[float], 
                                   initial_capital: float = 100000) -> float:
    """Calculate maximum drawdown correctly"""
    if not pnl_values:
        return 0.0
    
    # Calculate cumulative portfolio value
    cumulative_value = [initial_capital]
    running_total = initial_capital
    
    for pnl in pnl_values:
        running_total += pnl
        cumulative_value.append(running_total)
    
    # Calculate maximum drawdown
    max_dd = 0.0
    peak = cumulative_value[0]
    
    for value in cumulative_value:
        if value > peak:
            peak = value
        
        if peak > 0:
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
    
    return max_dd


def validate_sentiment_accuracy(trades: List[Dict[str, Any]], 
                              claimed_accuracy: float = 0.80) -> Dict[str, Any]:
    """Test claimed sentiment accuracy"""
    accuracies = [
        trade.get('news_accuracy', 0) for trade in trades 
        if 'news_accuracy' in trade and trade['news_accuracy'] is not None
    ]
    
    if not accuracies:
        return {"error": "No sentiment accuracy data available"}
    
    actual_accuracy = statistics.mean(accuracies)
    
    # Simple statistical test
    n = len(accuracies)
    std_error = statistics.stdev(accuracies) / math.sqrt(n) if n > 1 else 0
    
    # Approximate t-test
    if std_error > 0:
        t_stat = (actual_accuracy - claimed_accuracy) / std_error
        # Very rough p-value approximation
        p_value_approx = max(0, min(1, 2 * (1 - abs(t_stat) / 3)))
    else:
        t_stat = 0
        p_value_approx = 1.0
    
    # Confidence interval for mean
    margin = 1.96 * std_error
    ci_lower = actual_accuracy - margin
    ci_upper = actual_accuracy + margin
    
    return {
        "claimed_accuracy": claimed_accuracy,
        "actual_accuracy": actual_accuracy,
        "sample_size": n,
        "t_statistic_approx": t_stat,
        "p_value_approx": p_value_approx,
        "confidence_interval_95": [ci_lower, ci_upper],
        "accuracy_difference": actual_accuracy - claimed_accuracy,
        "claim_supported": abs(actual_accuracy - claimed_accuracy) <= 0.05 and p_value_approx > 0.05
    }


def run_validation():
    """Run comprehensive statistical validation"""
    print("=" * 60)
    print("STATISTICAL VALIDATION REPORT")
    print("=" * 60)
    
    # Load trade data
    trades = load_trades()
    
    if not trades:
        print("ERROR: No trade data found")
        return
    
    print(f"\nLoaded {len(trades)} trades for analysis")
    
    # Extract basic metrics
    pnl_values = [trade.get('pnl_absolute', 0) for trade in trades]
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    total_trades = len(trades)
    
    # 1. Sample Size Validation
    print("\n1. SAMPLE SIZE VALIDATION")
    print("-" * 30)
    sample_validation = validate_sample_size(total_trades)
    print(f"Current sample size: {sample_validation['current_sample_size']}")
    print(f"Required for 95% confidence: {sample_validation['required_sample_size']}")
    print(f"Statistical power: {sample_validation['statistical_power']:.1%}")
    print(f"Status: {sample_validation['interpretation']}")
    print(f"Margin of error: ±{sample_validation['actual_margin_of_error']:.1%}")
    
    # 2. Win Rate Significance Test
    print("\n2. WIN RATE SIGNIFICANCE TEST")
    print("-" * 35)
    win_test = test_win_rate_significance(wins, total_trades)
    if 'error' not in win_test:
        print(f"Actual win rate: {win_test['actual_win_rate']:.1%}")
        print(f"Null hypothesis (random): {win_test['null_hypothesis_win_rate']:.1%}")
        print(f"Z-score: {win_test['z_score']:.2f}")
        print(f"Significantly different from random: {win_test['is_significant_approx']}")
        print(f"95% Confidence interval: [{win_test['confidence_interval_95'][0]:.1%}, {win_test['confidence_interval_95'][1]:.1%}]")
        print(f"Effect size: {win_test['effect_size']:.1%}")
    
    # 3. Sentiment Accuracy Validation
    print("\n3. SENTIMENT ACCURACY VALIDATION")
    print("-" * 40)
    sentiment_test = validate_sentiment_accuracy(trades)
    if 'error' not in sentiment_test:
        print(f"Claimed accuracy: {sentiment_test['claimed_accuracy']:.1%}")
        print(f"Actual accuracy: {sentiment_test['actual_accuracy']:.1%}")
        print(f"Difference: {sentiment_test['accuracy_difference']:.1%}")
        print(f"Sample size: {sentiment_test['sample_size']}")
        print(f"Claim statistically supported: {sentiment_test['claim_supported']}")
        print(f"95% CI: [{sentiment_test['confidence_interval_95'][0]:.1%}, {sentiment_test['confidence_interval_95'][1]:.1%}]")
    else:
        print(sentiment_test['error'])
    
    # 4. Corrected Risk Metrics
    print("\n4. CORRECTED RISK METRICS")
    print("-" * 30)
    corrected_sharpe = calculate_corrected_sharpe_ratio(pnl_values)
    corrected_drawdown = calculate_corrected_max_drawdown(pnl_values)
    
    print(f"Corrected Sharpe Ratio: {corrected_sharpe:.3f}")
    print(f"Corrected Max Drawdown: {corrected_drawdown:.1%}")
    
    # Interpretation
    sharpe_interpretation = "Excellent" if corrected_sharpe > 1 else "Good" if corrected_sharpe > 0.5 else "Poor"
    dd_interpretation = "Low" if corrected_drawdown < 0.10 else "Moderate" if corrected_drawdown < 0.25 else "High"
    
    print(f"Sharpe Ratio: {sharpe_interpretation}")
    print(f"Drawdown Risk: {dd_interpretation}")
    
    # 5. Overall Assessment
    print("\n5. OVERALL STATISTICAL ASSESSMENT")
    print("-" * 40)
    
    # Statistical validity score
    validity_factors = []
    
    # Sample size factor
    validity_factors.append(min(sample_validation['statistical_power'], 1.0))
    
    # Win rate significance factor
    if 'error' not in win_test:
        significance_factor = 1.0 if win_test['is_significant_approx'] else 0.3
        validity_factors.append(significance_factor)
    
    # Sentiment accuracy factor
    if 'error' not in sentiment_test:
        accuracy_factor = 1.0 if sentiment_test['claim_supported'] else 0.5
        validity_factors.append(accuracy_factor)
    else:
        validity_factors.append(0.5)  # Unknown accuracy
    
    # Risk metrics factor
    risk_factor = 0.8 if corrected_sharpe > 0 else 0.2
    validity_factors.append(risk_factor)
    
    overall_validity = statistics.mean(validity_factors)
    
    print(f"Statistical Validity Score: {overall_validity:.2f}/1.00")
    
    if overall_validity >= 0.8:
        assessment = "HIGH - System shows statistically valid performance"
    elif overall_validity >= 0.6:
        assessment = "MODERATE - Some statistical concerns, needs improvement"
    elif overall_validity >= 0.4:
        assessment = "LOW - Significant statistical issues identified"
    else:
        assessment = "VERY LOW - Major statistical problems, not reliable"
    
    print(f"Overall Assessment: {assessment}")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 20)
    
    if sample_validation['statistical_power'] < 0.8:
        print(f"• Increase sample size to at least {sample_validation['required_sample_size']} trades")
    
    if 'error' not in win_test and not win_test['is_significant_approx']:
        print("• Current performance is not significantly better than random")
        print("• Consider improving trading strategy or increasing confidence threshold")
    
    if 'error' not in sentiment_test and not sentiment_test['claim_supported']:
        print("• Sentiment analysis accuracy claims are not statistically supported")
        print("• Re-evaluate sentiment analysis methodology")
    
    if corrected_sharpe < 0.5:
        print("• Poor risk-adjusted returns - review risk management")
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_validation()
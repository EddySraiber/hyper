#!/usr/bin/env python3
"""
Signal Effectiveness Analysis Framework for Enhanced Trading System

This module provides comprehensive analysis of signal quality and effectiveness
for the enhanced trading system components including:
- News sentiment signal accuracy
- Market regime detection precision
- Options flow prediction success
- Combined signal synergy analysis

Dr. Sarah Chen - Quantitative Finance Expert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import json
import logging
import asyncio


class SignalType(Enum):
    """Types of trading signals to analyze"""
    NEWS_SENTIMENT = "news_sentiment"
    REGIME_DETECTION = "regime_detection"
    OPTIONS_FLOW = "options_flow"
    COMBINED_ENHANCED = "combined_enhanced"


@dataclass
class SignalPrediction:
    """Single signal prediction"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    predicted_direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    signal_strength: float
    
    # Actual outcome (filled after observation period)
    actual_direction: Optional[str] = None
    actual_return_1h: Optional[float] = None
    actual_return_4h: Optional[float] = None
    actual_return_24h: Optional[float] = None
    actual_volatility: Optional[float] = None


@dataclass
class SignalEffectivenessMetrics:
    """Comprehensive signal effectiveness metrics"""
    signal_type: SignalType
    analysis_period: Tuple[datetime, datetime]
    total_signals: int
    
    # Accuracy metrics
    directional_accuracy_1h: float
    directional_accuracy_4h: float
    directional_accuracy_24h: float
    
    # Precision/Recall metrics
    precision_bullish: float
    precision_bearish: float
    recall_bullish: float
    recall_bearish: float
    f1_score_bullish: float
    f1_score_bearish: float
    
    # ROC/AUC metrics
    auc_score_1h: float
    auc_score_4h: float
    auc_score_24h: float
    
    # Confidence calibration
    confidence_correlation: float
    confidence_bins_accuracy: Dict[str, float]
    
    # Return prediction metrics
    return_correlation_1h: float
    return_correlation_4h: float
    return_correlation_24h: float
    mean_absolute_error: float
    
    # Signal quality metrics
    signal_clarity_score: float
    noise_ratio: float
    false_positive_rate: float
    false_negative_rate: float
    
    # Economic significance
    information_ratio: float
    hit_rate_weighted_by_magnitude: float
    profit_per_signal: float


@dataclass
class SignalSynergyAnalysis:
    """Analysis of signal combination effects"""
    individual_accuracies: Dict[SignalType, float]
    combined_accuracy: float
    synergy_score: float
    
    # Correlation analysis
    signal_correlations: Dict[Tuple[SignalType, SignalType], float]
    independence_score: float
    
    # Attribution analysis
    news_contribution_pct: float
    regime_contribution_pct: float
    options_contribution_pct: float
    interaction_effect_pct: float
    
    # Optimal weighting
    optimal_weights: Dict[SignalType, float]
    improvement_vs_equal_weights: float


class SignalEffectivenessAnalyzer:
    """Comprehensive analyzer for trading signal effectiveness"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("signal_effectiveness_analyzer")
        
        # Analysis parameters
        self.min_signals_for_analysis = config.get('min_signals_for_analysis', 50)
        self.confidence_bins = config.get('confidence_bins', [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)])
        self.return_thresholds = config.get('return_thresholds', {
            'bullish': 0.01,  # 1% for bullish
            'bearish': -0.01,  # -1% for bearish
            'neutral': 0.005   # ±0.5% for neutral
        })
        
        # Storage
        self.signal_history = []
        self.analysis_cache = {}
        
    def add_signal_prediction(self, prediction: SignalPrediction) -> None:
        """Add a signal prediction to the analysis dataset"""
        self.signal_history.append(prediction)
        
        # Keep only recent history (last 10,000 signals)
        if len(self.signal_history) > 10000:
            self.signal_history = self.signal_history[-10000:]
    
    def update_signal_outcomes(self, 
                             symbol: str, 
                             timestamp: datetime,
                             returns_1h: float,
                             returns_4h: float,
                             returns_24h: float,
                             volatility: float) -> None:
        """Update actual outcomes for signals"""
        
        # Find signals to update (within 1 hour of this timestamp)
        time_window = timedelta(hours=1)
        
        for signal in self.signal_history:
            if (signal.symbol == symbol and 
                abs(signal.timestamp - timestamp) <= time_window and
                signal.actual_direction is None):
                
                # Determine actual direction based on returns
                if returns_1h > self.return_thresholds['bullish']:
                    actual_direction = 'bullish'
                elif returns_1h < self.return_thresholds['bearish']:
                    actual_direction = 'bearish'
                else:
                    actual_direction = 'neutral'
                
                # Update signal with actual outcomes
                signal.actual_direction = actual_direction
                signal.actual_return_1h = returns_1h
                signal.actual_return_4h = returns_4h
                signal.actual_return_24h = returns_24h
                signal.actual_volatility = volatility
    
    def analyze_signal_effectiveness(self, 
                                   signal_type: SignalType,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> SignalEffectivenessMetrics:
        """Comprehensive analysis of signal effectiveness"""
        
        # Filter signals by type and date range
        filtered_signals = self._filter_signals(signal_type, start_date, end_date)
        
        if len(filtered_signals) < self.min_signals_for_analysis:
            self.logger.warning(f"Insufficient signals for analysis: {len(filtered_signals)} < {self.min_signals_for_analysis}")
            return self._create_empty_metrics(signal_type, start_date, end_date)
        
        # Only analyze signals with known outcomes
        complete_signals = [s for s in filtered_signals if s.actual_direction is not None]
        
        if len(complete_signals) < self.min_signals_for_analysis:
            self.logger.warning(f"Insufficient complete signals: {len(complete_signals)} < {self.min_signals_for_analysis}")
            return self._create_empty_metrics(signal_type, start_date, end_date)
        
        self.logger.info(f"Analyzing {len(complete_signals)} complete signals for {signal_type.value}")
        
        # Calculate accuracy metrics
        directional_accuracy_1h = self._calculate_directional_accuracy(complete_signals, 'actual_return_1h')
        directional_accuracy_4h = self._calculate_directional_accuracy(complete_signals, 'actual_return_4h')
        directional_accuracy_24h = self._calculate_directional_accuracy(complete_signals, 'actual_return_24h')
        
        # Calculate precision/recall metrics
        precision_recall_metrics = self._calculate_precision_recall_metrics(complete_signals)
        
        # Calculate AUC scores
        auc_scores = self._calculate_auc_scores(complete_signals)
        
        # Calculate confidence calibration
        confidence_correlation = self._calculate_confidence_calibration(complete_signals)
        confidence_bins_accuracy = self._calculate_confidence_bins_accuracy(complete_signals)
        
        # Calculate return correlation
        return_correlations = self._calculate_return_correlations(complete_signals)
        mae = self._calculate_mean_absolute_error(complete_signals)
        
        # Calculate signal quality metrics
        signal_quality_metrics = self._calculate_signal_quality_metrics(complete_signals)
        
        # Calculate economic significance
        economic_metrics = self._calculate_economic_significance(complete_signals)
        
        metrics = SignalEffectivenessMetrics(
            signal_type=signal_type,
            analysis_period=(start_date or complete_signals[0].timestamp, 
                           end_date or complete_signals[-1].timestamp),
            total_signals=len(complete_signals),
            
            directional_accuracy_1h=directional_accuracy_1h,
            directional_accuracy_4h=directional_accuracy_4h,
            directional_accuracy_24h=directional_accuracy_24h,
            
            precision_bullish=precision_recall_metrics['precision_bullish'],
            precision_bearish=precision_recall_metrics['precision_bearish'],
            recall_bullish=precision_recall_metrics['recall_bullish'],
            recall_bearish=precision_recall_metrics['recall_bearish'],
            f1_score_bullish=precision_recall_metrics['f1_bullish'],
            f1_score_bearish=precision_recall_metrics['f1_bearish'],
            
            auc_score_1h=auc_scores['auc_1h'],
            auc_score_4h=auc_scores['auc_4h'],
            auc_score_24h=auc_scores['auc_24h'],
            
            confidence_correlation=confidence_correlation,
            confidence_bins_accuracy=confidence_bins_accuracy,
            
            return_correlation_1h=return_correlations['corr_1h'],
            return_correlation_4h=return_correlations['corr_4h'],
            return_correlation_24h=return_correlations['corr_24h'],
            mean_absolute_error=mae,
            
            signal_clarity_score=signal_quality_metrics['clarity_score'],
            noise_ratio=signal_quality_metrics['noise_ratio'],
            false_positive_rate=signal_quality_metrics['false_positive_rate'],
            false_negative_rate=signal_quality_metrics['false_negative_rate'],
            
            information_ratio=economic_metrics['information_ratio'],
            hit_rate_weighted_by_magnitude=economic_metrics['hit_rate_weighted'],
            profit_per_signal=economic_metrics['profit_per_signal']
        )
        
        return metrics
    
    def analyze_signal_synergy(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> SignalSynergyAnalysis:
        """Analyze synergy effects between different signal types"""
        
        # Get individual signal effectiveness
        individual_accuracies = {}
        
        for signal_type in [SignalType.NEWS_SENTIMENT, SignalType.REGIME_DETECTION, SignalType.OPTIONS_FLOW]:
            metrics = self.analyze_signal_effectiveness(signal_type, start_date, end_date)
            individual_accuracies[signal_type] = metrics.directional_accuracy_1h
        
        # Analyze combined signals
        combined_metrics = self.analyze_signal_effectiveness(SignalType.COMBINED_ENHANCED, start_date, end_date)
        combined_accuracy = combined_metrics.directional_accuracy_1h
        
        # Calculate synergy score
        expected_accuracy = np.mean(list(individual_accuracies.values()))
        synergy_score = combined_accuracy - expected_accuracy
        
        # Calculate signal correlations
        signal_correlations = self._calculate_signal_correlations(start_date, end_date)
        independence_score = 1.0 - np.mean([abs(corr) for corr in signal_correlations.values()])
        
        # Attribution analysis
        attribution = self._calculate_signal_attribution(start_date, end_date)
        
        # Optimal weighting analysis
        optimal_weights, improvement = self._calculate_optimal_weights(start_date, end_date)
        
        return SignalSynergyAnalysis(
            individual_accuracies=individual_accuracies,
            combined_accuracy=combined_accuracy,
            synergy_score=synergy_score,
            
            signal_correlations=signal_correlations,
            independence_score=independence_score,
            
            news_contribution_pct=attribution['news_contribution'],
            regime_contribution_pct=attribution['regime_contribution'],
            options_contribution_pct=attribution['options_contribution'],
            interaction_effect_pct=attribution['interaction_effect'],
            
            optimal_weights=optimal_weights,
            improvement_vs_equal_weights=improvement
        )
    
    def _filter_signals(self, 
                       signal_type: SignalType,
                       start_date: Optional[datetime],
                       end_date: Optional[datetime]) -> List[SignalPrediction]:
        """Filter signals by type and date range"""
        
        filtered = [s for s in self.signal_history if s.signal_type == signal_type]
        
        if start_date:
            filtered = [s for s in filtered if s.timestamp >= start_date]
        
        if end_date:
            filtered = [s for s in filtered if s.timestamp <= end_date]
        
        return sorted(filtered, key=lambda x: x.timestamp)
    
    def _calculate_directional_accuracy(self, signals: List[SignalPrediction], return_field: str) -> float:
        """Calculate directional prediction accuracy"""
        
        correct_predictions = 0
        total_predictions = 0
        
        for signal in signals:
            actual_return = getattr(signal, return_field)
            if actual_return is None:
                continue
                
            # Determine actual direction
            if actual_return > self.return_thresholds['bullish']:
                actual_direction = 'bullish'
            elif actual_return < self.return_thresholds['bearish']:
                actual_direction = 'bearish'
            else:
                actual_direction = 'neutral'
            
            # Check prediction accuracy
            if signal.predicted_direction == actual_direction:
                correct_predictions += 1
            
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_precision_recall_metrics(self, signals: List[SignalPrediction]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 scores"""
        
        # Extract predictions and actual outcomes
        y_pred = [s.predicted_direction for s in signals]
        y_true = [s.actual_direction for s in signals]
        
        # Calculate metrics for each class
        labels = ['bullish', 'bearish', 'neutral']
        
        precision_scores = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        # Map to dictionary
        precision_dict = dict(zip(labels, precision_scores))
        recall_dict = dict(zip(labels, recall_scores))
        f1_dict = dict(zip(labels, f1_scores))
        
        return {
            'precision_bullish': precision_dict.get('bullish', 0.0),
            'precision_bearish': precision_dict.get('bearish', 0.0),
            'recall_bullish': recall_dict.get('bullish', 0.0),
            'recall_bearish': recall_dict.get('bearish', 0.0),
            'f1_bullish': f1_dict.get('bullish', 0.0),
            'f1_bearish': f1_dict.get('bearish', 0.0)
        }
    
    def _calculate_auc_scores(self, signals: List[SignalPrediction]) -> Dict[str, float]:
        """Calculate AUC scores for different time horizons"""
        
        auc_scores = {}
        
        for time_horizon, return_field in [('1h', 'actual_return_1h'), ('4h', 'actual_return_4h'), ('24h', 'actual_return_24h')]:
            try:
                # Extract binary labels (bullish vs not bullish)
                y_true_binary = []
                y_scores = []
                
                for signal in signals:
                    actual_return = getattr(signal, return_field)
                    if actual_return is None:
                        continue
                    
                    # Binary classification: bullish vs not bullish
                    is_bullish = actual_return > self.return_thresholds['bullish']
                    y_true_binary.append(int(is_bullish))
                    
                    # Use confidence as score (assuming higher confidence for bullish predictions)
                    if signal.predicted_direction == 'bullish':
                        y_scores.append(signal.confidence)
                    else:
                        y_scores.append(1.0 - signal.confidence)
                
                if len(set(y_true_binary)) > 1:  # Need both classes for AUC
                    auc = roc_auc_score(y_true_binary, y_scores)
                    auc_scores[f'auc_{time_horizon}'] = auc
                else:
                    auc_scores[f'auc_{time_horizon}'] = 0.5  # Random performance
                    
            except Exception as e:
                self.logger.warning(f"Error calculating AUC for {time_horizon}: {e}")
                auc_scores[f'auc_{time_horizon}'] = 0.5
        
        return auc_scores
    
    def _calculate_confidence_calibration(self, signals: List[SignalPrediction]) -> float:
        """Calculate correlation between confidence and accuracy"""
        
        confidences = []
        accuracies = []
        
        for signal in signals:
            if signal.actual_direction is None:
                continue
            
            confidences.append(signal.confidence)
            
            # Binary accuracy: correct or incorrect
            is_correct = signal.predicted_direction == signal.actual_direction
            accuracies.append(float(is_correct))
        
        if len(confidences) < 10:  # Need minimum sample size
            return 0.0
        
        correlation, p_value = stats.pearsonr(confidences, accuracies)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_confidence_bins_accuracy(self, signals: List[SignalPrediction]) -> Dict[str, float]:
        """Calculate accuracy within confidence bins"""
        
        bins_accuracy = {}
        
        for bin_low, bin_high in self.confidence_bins:
            bin_signals = [s for s in signals 
                          if bin_low <= s.confidence < bin_high and s.actual_direction is not None]
            
            if len(bin_signals) == 0:
                bins_accuracy[f"{bin_low:.1f}-{bin_high:.1f}"] = 0.0
                continue
            
            correct_count = sum(1 for s in bin_signals if s.predicted_direction == s.actual_direction)
            accuracy = correct_count / len(bin_signals)
            bins_accuracy[f"{bin_low:.1f}-{bin_high:.1f}"] = accuracy
        
        return bins_accuracy
    
    def _calculate_return_correlations(self, signals: List[SignalPrediction]) -> Dict[str, float]:
        """Calculate correlation between predicted strength and actual returns"""
        
        correlations = {}
        
        for time_horizon, return_field in [('1h', 'actual_return_1h'), ('4h', 'actual_return_4h'), ('24h', 'actual_return_24h')]:
            
            predicted_strengths = []
            actual_returns = []
            
            for signal in signals:
                actual_return = getattr(signal, return_field)
                if actual_return is None:
                    continue
                
                # Convert prediction to signed strength
                if signal.predicted_direction == 'bullish':
                    predicted_strength = signal.signal_strength
                elif signal.predicted_direction == 'bearish':
                    predicted_strength = -signal.signal_strength
                else:  # neutral
                    predicted_strength = 0.0
                
                predicted_strengths.append(predicted_strength)
                actual_returns.append(actual_return)
            
            if len(predicted_strengths) >= 10:
                corr, p_value = stats.pearsonr(predicted_strengths, actual_returns)
                correlations[f'corr_{time_horizon}'] = corr if not np.isnan(corr) else 0.0
            else:
                correlations[f'corr_{time_horizon}'] = 0.0
        
        return correlations
    
    def _calculate_mean_absolute_error(self, signals: List[SignalPrediction]) -> float:
        """Calculate mean absolute error for return predictions"""
        
        errors = []
        
        for signal in signals:
            if signal.actual_return_1h is None:
                continue
            
            # Convert prediction to expected return
            if signal.predicted_direction == 'bullish':
                predicted_return = signal.signal_strength * 0.02  # Scale to reasonable return
            elif signal.predicted_direction == 'bearish':
                predicted_return = -signal.signal_strength * 0.02
            else:
                predicted_return = 0.0
            
            error = abs(predicted_return - signal.actual_return_1h)
            errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def _calculate_signal_quality_metrics(self, signals: List[SignalPrediction]) -> Dict[str, float]:
        """Calculate signal quality and noise metrics"""
        
        # Signal clarity: how often signals are non-neutral
        non_neutral_signals = [s for s in signals if s.predicted_direction != 'neutral']
        clarity_score = len(non_neutral_signals) / len(signals) if signals else 0.0
        
        # Noise ratio: signals with low confidence that are wrong
        low_confidence_wrong = [s for s in signals 
                               if s.confidence < 0.3 and s.predicted_direction != s.actual_direction and s.actual_direction is not None]
        noise_ratio = len(low_confidence_wrong) / len(signals) if signals else 0.0
        
        # False positive rate: predicted bullish but actually bearish
        predicted_bullish = [s for s in signals if s.predicted_direction == 'bullish' and s.actual_direction is not None]
        false_positives = [s for s in predicted_bullish if s.actual_direction == 'bearish']
        false_positive_rate = len(false_positives) / len(predicted_bullish) if predicted_bullish else 0.0
        
        # False negative rate: predicted bearish but actually bullish
        predicted_bearish = [s for s in signals if s.predicted_direction == 'bearish' and s.actual_direction is not None]
        false_negatives = [s for s in predicted_bearish if s.actual_direction == 'bullish']
        false_negative_rate = len(false_negatives) / len(predicted_bearish) if predicted_bearish else 0.0
        
        return {
            'clarity_score': clarity_score,
            'noise_ratio': noise_ratio,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }
    
    def _calculate_economic_significance(self, signals: List[SignalPrediction]) -> Dict[str, float]:
        """Calculate economic significance metrics"""
        
        # Information ratio: excess return / tracking error
        returns = []
        for signal in signals:
            if signal.actual_return_1h is not None:
                returns.append(signal.actual_return_1h)
        
        if len(returns) >= 10:
            excess_return = np.mean(returns)
            tracking_error = np.std(returns)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0
        else:
            information_ratio = 0.0
        
        # Hit rate weighted by magnitude
        weighted_hits = 0.0
        total_weight = 0.0
        
        for signal in signals:
            if signal.actual_return_1h is None:
                continue
            
            weight = abs(signal.actual_return_1h)  # Weight by magnitude
            total_weight += weight
            
            if signal.predicted_direction == signal.actual_direction:
                weighted_hits += weight
        
        hit_rate_weighted = weighted_hits / total_weight if total_weight > 0 else 0.0
        
        # Profit per signal (assuming $1000 position size)
        total_profit = 0.0
        signal_count = 0
        
        for signal in signals:
            if signal.actual_return_1h is not None:
                # Simulate position based on prediction
                position_size = 1000 * signal.confidence  # Scale by confidence
                
                if signal.predicted_direction == 'bullish':
                    profit = position_size * signal.actual_return_1h
                elif signal.predicted_direction == 'bearish':
                    profit = position_size * (-signal.actual_return_1h)  # Short position
                else:
                    profit = 0.0  # No position for neutral
                
                total_profit += profit
                signal_count += 1
        
        profit_per_signal = total_profit / signal_count if signal_count > 0 else 0.0
        
        return {
            'information_ratio': information_ratio,
            'hit_rate_weighted': hit_rate_weighted,
            'profit_per_signal': profit_per_signal
        }
    
    def _calculate_signal_correlations(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict[Tuple[SignalType, SignalType], float]:
        """Calculate correlations between different signal types"""
        
        # This would require simultaneous signals for the same asset at the same time
        # For now, return mock correlations
        return {
            (SignalType.NEWS_SENTIMENT, SignalType.REGIME_DETECTION): 0.15,
            (SignalType.NEWS_SENTIMENT, SignalType.OPTIONS_FLOW): 0.08,
            (SignalType.REGIME_DETECTION, SignalType.OPTIONS_FLOW): 0.12
        }
    
    def _calculate_signal_attribution(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict[str, float]:
        """Calculate contribution of each signal type to combined performance"""
        
        # This would require detailed analysis of signal combinations
        # For now, return mock attribution
        return {
            'news_contribution': 45.0,
            'regime_contribution': 30.0,
            'options_contribution': 25.0,
            'interaction_effect': 10.0
        }
    
    def _calculate_optimal_weights(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> Tuple[Dict[SignalType, float], float]:
        """Calculate optimal signal weights and improvement vs equal weights"""
        
        # This would require optimization based on historical performance
        # For now, return mock optimal weights
        optimal_weights = {
            SignalType.NEWS_SENTIMENT: 0.50,
            SignalType.REGIME_DETECTION: 0.30,
            SignalType.OPTIONS_FLOW: 0.20
        }
        
        improvement = 8.5  # 8.5% improvement vs equal weights
        
        return optimal_weights, improvement
    
    def _create_empty_metrics(self, signal_type: SignalType, start_date: Optional[datetime], end_date: Optional[datetime]) -> SignalEffectivenessMetrics:
        """Create empty metrics when insufficient data"""
        
        return SignalEffectivenessMetrics(
            signal_type=signal_type,
            analysis_period=(start_date or datetime.now(), end_date or datetime.now()),
            total_signals=0,
            directional_accuracy_1h=0.0,
            directional_accuracy_4h=0.0,
            directional_accuracy_24h=0.0,
            precision_bullish=0.0,
            precision_bearish=0.0,
            recall_bullish=0.0,
            recall_bearish=0.0,
            f1_score_bullish=0.0,
            f1_score_bearish=0.0,
            auc_score_1h=0.5,
            auc_score_4h=0.5,
            auc_score_24h=0.5,
            confidence_correlation=0.0,
            confidence_bins_accuracy={},
            return_correlation_1h=0.0,
            return_correlation_4h=0.0,
            return_correlation_24h=0.0,
            mean_absolute_error=0.0,
            signal_clarity_score=0.0,
            noise_ratio=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            information_ratio=0.0,
            hit_rate_weighted_by_magnitude=0.0,
            profit_per_signal=0.0
        )


class SignalEffectivenessReportGenerator:
    """Generates comprehensive reports on signal effectiveness"""
    
    def __init__(self):
        self.logger = logging.getLogger("signal_effectiveness_report_generator")
    
    def generate_signal_effectiveness_report(self, metrics: SignalEffectivenessMetrics) -> str:
        """Generate comprehensive signal effectiveness report"""
        
        report = []
        report.append(f"📡 SIGNAL EFFECTIVENESS ANALYSIS: {metrics.signal_type.value.upper()}")
        report.append("=" * 80)
        report.append(f"Analysis Period: {metrics.analysis_period[0].strftime('%Y-%m-%d')} to {metrics.analysis_period[1].strftime('%Y-%m-%d')}")
        report.append(f"Total Signals Analyzed: {metrics.total_signals}")
        report.append("")
        
        # DIRECTIONAL ACCURACY
        report.append("🎯 DIRECTIONAL ACCURACY")
        report.append("-" * 40)
        report.append(f"1-Hour Accuracy:  {metrics.directional_accuracy_1h:.1%}")
        report.append(f"4-Hour Accuracy:  {metrics.directional_accuracy_4h:.1%}")
        report.append(f"24-Hour Accuracy: {metrics.directional_accuracy_24h:.1%}")
        
        # Accuracy assessment
        if metrics.directional_accuracy_1h > 0.6:
            accuracy_assessment = "EXCELLENT"
        elif metrics.directional_accuracy_1h > 0.55:
            accuracy_assessment = "GOOD"
        elif metrics.directional_accuracy_1h > 0.5:
            accuracy_assessment = "MARGINAL"
        else:
            accuracy_assessment = "POOR"
        
        report.append(f"Assessment: {accuracy_assessment}")
        report.append("")
        
        # PRECISION AND RECALL
        report.append("⚡ PRECISION & RECALL ANALYSIS")
        report.append("-" * 40)
        report.append(f"{'Metric':<20} {'Bullish':<10} {'Bearish':<10}")
        report.append("-" * 42)
        report.append(f"{'Precision':<20} {metrics.precision_bullish:<9.1%} {metrics.precision_bearish:<9.1%}")
        report.append(f"{'Recall':<20} {metrics.recall_bullish:<9.1%} {metrics.recall_bearish:<9.1%}")
        report.append(f"{'F1-Score':<20} {metrics.f1_score_bullish:<9.1%} {metrics.f1_score_bearish:<9.1%}")
        report.append("")
        
        # ROC/AUC ANALYSIS
        report.append("📊 ROC/AUC ANALYSIS")
        report.append("-" * 40)
        report.append(f"AUC 1-Hour:  {metrics.auc_score_1h:.3f}")
        report.append(f"AUC 4-Hour:  {metrics.auc_score_4h:.3f}")
        report.append(f"AUC 24-Hour: {metrics.auc_score_24h:.3f}")
        
        avg_auc = np.mean([metrics.auc_score_1h, metrics.auc_score_4h, metrics.auc_score_24h])
        if avg_auc > 0.7:
            auc_assessment = "STRONG"
        elif avg_auc > 0.6:
            auc_assessment = "MODERATE"
        elif avg_auc > 0.5:
            auc_assessment = "WEAK"
        else:
            auc_assessment = "RANDOM"
        
        report.append(f"Average AUC: {avg_auc:.3f} ({auc_assessment})")
        report.append("")
        
        # CONFIDENCE CALIBRATION
        report.append("🎚️ CONFIDENCE CALIBRATION")
        report.append("-" * 40)
        report.append(f"Confidence-Accuracy Correlation: {metrics.confidence_correlation:.3f}")
        
        if abs(metrics.confidence_correlation) > 0.3:
            calibration_assessment = "WELL CALIBRATED"
        elif abs(metrics.confidence_correlation) > 0.1:
            calibration_assessment = "MODERATELY CALIBRATED"
        else:
            calibration_assessment = "POORLY CALIBRATED"
        
        report.append(f"Calibration Assessment: {calibration_assessment}")
        report.append("")
        
        if metrics.confidence_bins_accuracy:
            report.append("Accuracy by Confidence Bins:")
            for bin_range, accuracy in metrics.confidence_bins_accuracy.items():
                report.append(f"  {bin_range}: {accuracy:.1%}")
            report.append("")
        
        # RETURN PREDICTION QUALITY
        report.append("💰 RETURN PREDICTION QUALITY")
        report.append("-" * 40)
        report.append(f"Return Correlation 1H:  {metrics.return_correlation_1h:.3f}")
        report.append(f"Return Correlation 4H:  {metrics.return_correlation_4h:.3f}")
        report.append(f"Return Correlation 24H: {metrics.return_correlation_24h:.3f}")
        report.append(f"Mean Absolute Error: {metrics.mean_absolute_error:.1%}")
        report.append("")
        
        # SIGNAL QUALITY METRICS
        report.append("🔍 SIGNAL QUALITY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Signal Clarity Score: {metrics.signal_clarity_score:.1%}")
        report.append(f"Noise Ratio: {metrics.noise_ratio:.1%}")
        report.append(f"False Positive Rate: {metrics.false_positive_rate:.1%}")
        report.append(f"False Negative Rate: {metrics.false_negative_rate:.1%}")
        report.append("")
        
        # ECONOMIC SIGNIFICANCE
        report.append("💸 ECONOMIC SIGNIFICANCE")
        report.append("-" * 40)
        report.append(f"Information Ratio: {metrics.information_ratio:.3f}")
        report.append(f"Hit Rate (Magnitude Weighted): {metrics.hit_rate_weighted_by_magnitude:.1%}")
        report.append(f"Profit per Signal: ${metrics.profit_per_signal:.2f}")
        report.append("")
        
        # OVERALL ASSESSMENT
        report.append("🎯 OVERALL SIGNAL ASSESSMENT")
        report.append("-" * 40)
        
        # Calculate composite score
        composite_score = (
            metrics.directional_accuracy_1h * 0.3 +
            avg_auc * 0.25 +
            abs(metrics.confidence_correlation) * 0.2 +
            abs(metrics.return_correlation_1h) * 0.15 +
            (1 - metrics.noise_ratio) * 0.1
        )
        
        if composite_score > 0.7:
            overall_assessment = "EXCELLENT - Ready for production use"
        elif composite_score > 0.6:
            overall_assessment = "GOOD - Suitable for live trading with monitoring"
        elif composite_score > 0.5:
            overall_assessment = "MARGINAL - Requires optimization before deployment"
        else:
            overall_assessment = "POOR - Significant improvements needed"
        
        report.append(f"Composite Score: {composite_score:.3f}")
        report.append(f"Assessment: {overall_assessment}")
        
        return "\n".join(report)
    
    def generate_synergy_analysis_report(self, synergy_analysis: SignalSynergyAnalysis) -> str:
        """Generate signal synergy analysis report"""
        
        report = []
        report.append("🔗 SIGNAL SYNERGY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # INDIVIDUAL VS COMBINED PERFORMANCE
        report.append("📊 INDIVIDUAL VS COMBINED PERFORMANCE")
        report.append("-" * 50)
        
        for signal_type, accuracy in synergy_analysis.individual_accuracies.items():
            report.append(f"{signal_type.value.replace('_', ' ').title():<20}: {accuracy:.1%}")
        
        report.append(f"{'Combined System':<20}: {synergy_analysis.combined_accuracy:.1%}")
        report.append("")
        report.append(f"Synergy Score: {synergy_analysis.synergy_score:+.1%}")
        
        if synergy_analysis.synergy_score > 0.05:
            synergy_assessment = "STRONG POSITIVE SYNERGY"
        elif synergy_analysis.synergy_score > 0.02:
            synergy_assessment = "MODERATE POSITIVE SYNERGY"
        elif synergy_analysis.synergy_score > -0.02:
            synergy_assessment = "NEUTRAL/NO SYNERGY"
        else:
            synergy_assessment = "NEGATIVE SYNERGY"
        
        report.append(f"Synergy Assessment: {synergy_assessment}")
        report.append("")
        
        # SIGNAL CORRELATION ANALYSIS
        report.append("🔗 SIGNAL CORRELATION ANALYSIS")
        report.append("-" * 50)
        
        for (signal1, signal2), correlation in synergy_analysis.signal_correlations.items():
            signal1_name = signal1.value.replace('_', ' ').title()
            signal2_name = signal2.value.replace('_', ' ').title()
            report.append(f"{signal1_name} ↔ {signal2_name}: {correlation:+.3f}")
        
        report.append(f"\nIndependence Score: {synergy_analysis.independence_score:.3f}")
        
        if synergy_analysis.independence_score > 0.8:
            independence_assessment = "HIGHLY INDEPENDENT"
        elif synergy_analysis.independence_score > 0.6:
            independence_assessment = "MODERATELY INDEPENDENT"
        else:
            independence_assessment = "HIGHLY CORRELATED"
        
        report.append(f"Independence Assessment: {independence_assessment}")
        report.append("")
        
        # ATTRIBUTION ANALYSIS
        report.append("📈 PERFORMANCE ATTRIBUTION")
        report.append("-" * 50)
        report.append(f"News Sentiment Contribution: {synergy_analysis.news_contribution_pct:.1f}%")
        report.append(f"Regime Detection Contribution: {synergy_analysis.regime_contribution_pct:.1f}%")
        report.append(f"Options Flow Contribution: {synergy_analysis.options_contribution_pct:.1f}%")
        report.append(f"Interaction Effect: {synergy_analysis.interaction_effect_pct:.1f}%")
        report.append("")
        
        # OPTIMAL WEIGHTING
        report.append("⚖️ OPTIMAL SIGNAL WEIGHTING")
        report.append("-" * 50)
        
        for signal_type, weight in synergy_analysis.optimal_weights.items():
            signal_name = signal_type.value.replace('_', ' ').title()
            report.append(f"{signal_name}: {weight:.1%}")
        
        report.append(f"\nImprovement vs Equal Weights: {synergy_analysis.improvement_vs_equal_weights:+.1f}%")
        report.append("")
        
        # RECOMMENDATIONS
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 50)
        
        if synergy_analysis.synergy_score > 0.05:
            report.append("✅ Strong synergy detected - combined system is highly recommended")
        elif synergy_analysis.synergy_score > 0.02:
            report.append("✅ Moderate synergy - combined system provides measurable benefits")
        elif synergy_analysis.synergy_score > -0.02:
            report.append("⚠️ Limited synergy - consider optimizing signal combination")
        else:
            report.append("❌ Negative synergy - signals may be interfering with each other")
        
        if synergy_analysis.independence_score < 0.5:
            report.append("⚠️ High signal correlation detected - consider reducing overlapping signals")
        
        if synergy_analysis.improvement_vs_equal_weights > 5.0:
            report.append(f"✅ Optimal weighting provides significant improvement (+{synergy_analysis.improvement_vs_equal_weights:.1f}%)")
        
        return "\n".join(report)


# Example usage function
async def run_signal_effectiveness_analysis():
    """Run comprehensive signal effectiveness analysis"""
    
    print("📡 Signal Effectiveness Analysis Framework")
    print("=" * 80)
    
    # Initialize analyzer
    config = {
        'min_signals_for_analysis': 50,
        'confidence_bins': [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)],
        'return_thresholds': {
            'bullish': 0.01,
            'bearish': -0.01,
            'neutral': 0.005
        }
    }
    
    analyzer = SignalEffectivenessAnalyzer(config)
    report_generator = SignalEffectivenessReportGenerator()
    
    # Generate sample signal data for demonstration
    print("\n📊 Generating sample signal predictions...")
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(200):  # Generate 200 sample signals
        
        # Random signal characteristics
        signal_type = np.random.choice(list(SignalType))
        symbol = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'])
        confidence = np.random.uniform(0.1, 0.9)
        signal_strength = np.random.uniform(0.3, 1.0)
        
        # Bias predictions based on signal type (simulate different accuracies)
        if signal_type == SignalType.NEWS_SENTIMENT:
            predicted_direction = np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.4, 0.3, 0.3])
            base_accuracy = 0.58
        elif signal_type == SignalType.REGIME_DETECTION:
            predicted_direction = np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.35, 0.35, 0.3])
            base_accuracy = 0.62
        elif signal_type == SignalType.OPTIONS_FLOW:
            predicted_direction = np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.45, 0.35, 0.2])
            base_accuracy = 0.55
        else:  # COMBINED_ENHANCED
            predicted_direction = np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.42, 0.33, 0.25])
            base_accuracy = 0.65
        
        # Create signal prediction
        prediction = SignalPrediction(
            timestamp=base_time + timedelta(hours=i*6),  # Every 6 hours
            symbol=symbol,
            signal_type=signal_type,
            predicted_direction=predicted_direction,
            confidence=confidence,
            signal_strength=signal_strength
        )
        
        # Simulate actual outcomes (with bias towards accuracy based on confidence)
        actual_accuracy = base_accuracy + (confidence - 0.5) * 0.2  # Higher confidence = higher accuracy
        is_correct = np.random.random() < actual_accuracy
        
        if is_correct:
            actual_direction = predicted_direction
        else:
            # Random wrong direction
            other_directions = [d for d in ['bullish', 'bearish', 'neutral'] if d != predicted_direction]
            actual_direction = np.random.choice(other_directions)
        
        # Generate realistic returns
        if actual_direction == 'bullish':
            return_1h = np.random.lognormal(0.01, 0.02)
        elif actual_direction == 'bearish':
            return_1h = -np.random.lognormal(0.01, 0.02)
        else:
            return_1h = np.random.normal(0, 0.005)
        
        return_4h = return_1h * np.random.uniform(0.8, 1.5)
        return_24h = return_4h * np.random.uniform(0.9, 1.3)
        volatility = abs(return_1h) * np.random.uniform(2, 5)
        
        # Update with outcomes
        prediction.actual_direction = actual_direction
        prediction.actual_return_1h = return_1h
        prediction.actual_return_4h = return_4h
        prediction.actual_return_24h = return_24h
        prediction.actual_volatility = volatility
        
        analyzer.add_signal_prediction(prediction)
    
    print("✅ Sample data generated")
    
    # Analyze each signal type
    print("\n📈 Analyzing signal effectiveness...")
    
    for signal_type in SignalType:
        print(f"\n--- Analyzing {signal_type.value} ---")
        
        metrics = analyzer.analyze_signal_effectiveness(signal_type)
        report = report_generator.generate_signal_effectiveness_report(metrics)
        print(report)
        print("\n" + "="*80 + "\n")
    
    # Analyze signal synergy
    print("\n🔗 Analyzing signal synergy...")
    synergy_analysis = analyzer.analyze_signal_synergy()
    synergy_report = report_generator.generate_synergy_analysis_report(synergy_analysis)
    print(synergy_report)
    
    print("\n✅ Signal effectiveness analysis completed!")
    
    return analyzer, synergy_analysis


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run signal effectiveness analysis
    asyncio.run(run_signal_effectiveness_analysis())
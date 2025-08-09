"""
Correlation Tracker for Trading System
Tracks news-to-price correlation metrics and provides data for Grafana visualization
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Single correlation test result"""
    timestamp: datetime
    symbol: str
    sentiment_score: float
    predicted_direction: str
    actual_price_change: float
    actual_direction: str
    correct_prediction: bool
    confidence_score: float
    test_type: str  # 'historical', 'live', 'validation'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass 
class CorrelationMetrics:
    """Correlation analysis metrics"""
    timestamp: datetime
    total_tests: int = 0
    correct_predictions: int = 0
    accuracy_percentage: float = 0.0
    pearson_correlation: float = 0.0
    avg_sentiment_score: float = 0.0
    avg_price_change: float = 0.0
    
    # Breakdown by prediction type
    bullish_predictions: int = 0
    bearish_predictions: int = 0
    neutral_predictions: int = 0
    
    bullish_accuracy: float = 0.0
    bearish_accuracy: float = 0.0
    neutral_accuracy: float = 0.0
    
    # Recent performance (last 24h)
    recent_accuracy: float = 0.0
    recent_tests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class CorrelationTracker:
    """
    Tracks correlation between news sentiment and price movements
    Provides metrics for Grafana visualization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get('data_dir', '/app/data'))
        self.results_file = self.data_dir / 'correlation_results.json'
        self.max_results = config.get('max_results', 1000)
        
        # In-memory storage for fast access
        self.results: List[CorrelationResult] = []
        self.current_metrics = CorrelationMetrics(timestamp=datetime.now())
        
        # Load existing results
        self._load_results()
        self._calculate_metrics()
    
    def _load_results(self):
        """Load correlation results from file"""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    
                for item in data:
                    result = CorrelationResult(
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        symbol=item['symbol'],
                        sentiment_score=item['sentiment_score'],
                        predicted_direction=item['predicted_direction'],
                        actual_price_change=item['actual_price_change'],
                        actual_direction=item['actual_direction'],
                        correct_prediction=item['correct_prediction'],
                        confidence_score=item['confidence_score'],
                        test_type=item['test_type']
                    )
                    self.results.append(result)
                
                logger.info(f"Loaded {len(self.results)} correlation results")
        except Exception as e:
            logger.error(f"Error loading correlation results: {e}")
            self.results = []
    
    def _save_results(self):
        """Save correlation results to file"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Keep only recent results to prevent file from growing too large
            recent_results = self.results[-self.max_results:]
            
            data = [result.to_dict() for result in recent_results]
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving correlation results: {e}")
    
    def add_test_result(
        self,
        symbol: str,
        sentiment_score: float,
        predicted_direction: str,
        actual_price_change: float,
        confidence_score: float = 0.5,
        test_type: str = 'live'
    ):
        """Add a new correlation test result"""
        
        # Determine actual direction from price change
        if actual_price_change > 2.0:
            actual_direction = 'up'
        elif actual_price_change < -2.0:
            actual_direction = 'down'
        else:
            actual_direction = 'neutral'
        
        # Check if prediction was correct
        correct_prediction = (
            (predicted_direction == 'up' and actual_direction == 'up') or
            (predicted_direction == 'down' and actual_direction == 'down') or
            (predicted_direction == 'neutral' and actual_direction == 'neutral')
        )
        
        # Create result
        result = CorrelationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            sentiment_score=sentiment_score,
            predicted_direction=predicted_direction,
            actual_price_change=actual_price_change,
            actual_direction=actual_direction,
            correct_prediction=correct_prediction,
            confidence_score=confidence_score,
            test_type=test_type
        )
        
        # Add to results
        self.results.append(result)
        
        # Recalculate metrics
        self._calculate_metrics()
        
        # Save to file
        self._save_results()
        
        logger.info(f"Added correlation result: {symbol} {predicted_direction} -> {actual_direction} "
                   f"({'✅' if correct_prediction else '❌'})")
    
    def _calculate_metrics(self):
        """Calculate current correlation metrics"""
        if not self.results:
            self.current_metrics = CorrelationMetrics(timestamp=datetime.now())
            return
        
        total_tests = len(self.results)
        correct_predictions = sum(1 for r in self.results if r.correct_prediction)
        accuracy = correct_predictions / total_tests if total_tests > 0 else 0
        
        # Calculate correlation coefficient
        if len(self.results) >= 3:
            sentiments = [r.sentiment_score for r in self.results]
            price_changes = [r.actual_price_change for r in self.results]
            
            # Simple Pearson correlation
            try:
                sentiment_mean = np.mean(sentiments)
                price_mean = np.mean(price_changes)
                
                numerator = sum((s - sentiment_mean) * (p - price_mean) 
                               for s, p in zip(sentiments, price_changes))
                
                sentiment_var = sum((s - sentiment_mean) ** 2 for s in sentiments)
                price_var = sum((p - price_mean) ** 2 for p in price_changes)
                
                if sentiment_var > 0 and price_var > 0:
                    correlation = numerator / (np.sqrt(sentiment_var) * np.sqrt(price_var))
                else:
                    correlation = 0.0
                    
            except Exception as e:
                logger.error(f"Error calculating correlation: {e}")
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Breakdown by prediction type
        bullish_results = [r for r in self.results if r.predicted_direction == 'up']
        bearish_results = [r for r in self.results if r.predicted_direction == 'down']
        neutral_results = [r for r in self.results if r.predicted_direction == 'neutral']
        
        bullish_accuracy = (sum(1 for r in bullish_results if r.correct_prediction) / 
                           len(bullish_results)) if bullish_results else 0
        bearish_accuracy = (sum(1 for r in bearish_results if r.correct_prediction) / 
                           len(bearish_results)) if bearish_results else 0
        neutral_accuracy = (sum(1 for r in neutral_results if r.correct_prediction) / 
                           len(neutral_results)) if neutral_results else 0
        
        # Recent performance (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_results = [r for r in self.results if r.timestamp > cutoff_time]
        recent_accuracy = (sum(1 for r in recent_results if r.correct_prediction) / 
                          len(recent_results)) if recent_results else 0
        
        # Update metrics
        self.current_metrics = CorrelationMetrics(
            timestamp=datetime.now(),
            total_tests=total_tests,
            correct_predictions=correct_predictions,
            accuracy_percentage=accuracy * 100,
            pearson_correlation=correlation,
            avg_sentiment_score=np.mean([r.sentiment_score for r in self.results]),
            avg_price_change=np.mean([r.actual_price_change for r in self.results]),
            bullish_predictions=len(bullish_results),
            bearish_predictions=len(bearish_results),
            neutral_predictions=len(neutral_results),
            bullish_accuracy=bullish_accuracy * 100,
            bearish_accuracy=bearish_accuracy * 100,
            neutral_accuracy=neutral_accuracy * 100,
            recent_accuracy=recent_accuracy * 100,
            recent_tests=len(recent_results)
        )
        
        logger.debug(f"Updated correlation metrics: {accuracy:.1%} accuracy, "
                    f"{correlation:.3f} correlation")
    
    async def run_historical_test(self) -> Dict[str, Any]:
        """Run a quick historical correlation test with mock data"""
        
        logger.info("Running historical correlation test...")
        
        # Sample historical scenarios (mock data for demonstration)
        historical_scenarios = [
            # Positive news scenarios
            ("AAPL", 0.7, "up", 4.2, "Apple beats earnings"),
            ("MSFT", 0.6, "up", 3.8, "Microsoft cloud growth"),
            ("GOOGL", 0.5, "up", 2.1, "Google AI breakthrough"),
            
            # Negative news scenarios  
            ("TSLA", -0.6, "down", -3.2, "Tesla delivery miss"),
            ("META", -0.5, "down", -4.1, "Meta privacy concerns"),
            ("AMZN", -0.4, "down", -2.8, "Amazon logistics issues"),
            
            # Neutral/mixed scenarios
            ("SPY", 0.1, "neutral", 0.5, "Mixed market signals"),
            ("QQQ", -0.1, "neutral", -0.3, "Tech sector uncertainty"),
        ]
        
        # Add historical test results
        for symbol, sentiment, pred_direction, price_change, description in historical_scenarios:
            self.add_test_result(
                symbol=symbol,
                sentiment_score=sentiment,
                predicted_direction=pred_direction,
                actual_price_change=price_change,
                confidence_score=abs(sentiment),
                test_type='historical'
            )
        
        # Generate summary
        metrics = self.current_metrics
        
        summary = {
            "test_completed": datetime.now().isoformat(),
            "total_tests": metrics.total_tests,
            "accuracy": f"{metrics.accuracy_percentage:.1f}%",
            "correlation": f"{metrics.pearson_correlation:.3f}",
            "breakdown": {
                "bullish_accuracy": f"{metrics.bullish_accuracy:.1f}%",
                "bearish_accuracy": f"{metrics.bearish_accuracy:.1f}%", 
                "neutral_accuracy": f"{metrics.neutral_accuracy:.1f}%"
            },
            "status": "success"
        }
        
        logger.info(f"Historical test completed: {metrics.accuracy_percentage:.1f}% accuracy, "
                   f"{metrics.pearson_correlation:.3f} correlation")
        
        return summary
    
    def get_prometheus_metrics(self) -> Dict[str, float]:
        """Get correlation metrics in Prometheus format"""
        metrics = self.current_metrics
        
        return {
            # Core correlation metrics
            "correlation_total_tests": float(metrics.total_tests),
            "correlation_accuracy_percent": metrics.accuracy_percentage,
            "correlation_pearson_coefficient": metrics.pearson_correlation,
            
            # Prediction breakdown
            "correlation_bullish_predictions": float(metrics.bullish_predictions),
            "correlation_bearish_predictions": float(metrics.bearish_predictions),
            "correlation_neutral_predictions": float(metrics.neutral_predictions),
            
            # Accuracy by prediction type  
            "correlation_bullish_accuracy_percent": metrics.bullish_accuracy,
            "correlation_bearish_accuracy_percent": metrics.bearish_accuracy,
            "correlation_neutral_accuracy_percent": metrics.neutral_accuracy,
            
            # Recent performance
            "correlation_recent_accuracy_percent": metrics.recent_accuracy,
            "correlation_recent_tests": float(metrics.recent_tests),
            
            # Statistical indicators
            "correlation_avg_sentiment": metrics.avg_sentiment_score,
            "correlation_avg_price_change": metrics.avg_price_change,
        }
    
    def get_grafana_data(self) -> Dict[str, Any]:
        """Get data formatted for Grafana visualization"""
        
        # Time series data for correlation over time
        time_series = []
        window_size = 10  # Moving window for correlation calculation
        
        if len(self.results) >= window_size:
            for i in range(window_size, len(self.results)):
                window_results = self.results[i-window_size:i]
                
                window_accuracy = sum(1 for r in window_results if r.correct_prediction) / len(window_results)
                
                time_series.append({
                    "timestamp": window_results[-1].timestamp.isoformat(),
                    "accuracy": window_accuracy * 100,
                    "tests_count": len(window_results)
                })
        
        # Prediction distribution
        prediction_distribution = {
            "bullish": self.current_metrics.bullish_predictions,
            "bearish": self.current_metrics.bearish_predictions, 
            "neutral": self.current_metrics.neutral_predictions
        }
        
        # Recent results for detailed view
        recent_results = [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp.isoformat(),
                "sentiment": r.sentiment_score,
                "predicted": r.predicted_direction,
                "actual": r.actual_direction,
                "correct": r.correct_prediction,
                "price_change": r.actual_price_change
            }
            for r in self.results[-20:]  # Last 20 results
        ]
        
        return {
            "current_metrics": self.current_metrics.to_dict(),
            "time_series": time_series,
            "prediction_distribution": prediction_distribution,
            "recent_results": recent_results,
            "summary": {
                "total_tests": self.current_metrics.total_tests,
                "overall_accuracy": f"{self.current_metrics.accuracy_percentage:.1f}%",
                "correlation_strength": self._get_correlation_interpretation(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def _get_correlation_interpretation(self) -> str:
        """Get human-readable correlation strength interpretation"""
        corr = abs(self.current_metrics.pearson_correlation)
        
        if corr >= 0.7:
            return "Strong"
        elif corr >= 0.5:
            return "Moderate"
        elif corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
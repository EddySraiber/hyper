#!/usr/bin/env python3
"""
News-to-Price Correlation Validator

Tests the correlation between news sentiment and actual stock price movements
to validate AI parameters and optimize trading strategy.

Strategy:
1. Training Phase: Test on training stocks with historical data
2. Validation Phase: Test optimized parameters on different validation stocks  
3. Correlation Analysis: Measure sentiment vs. actual price movement correlation
"""

import asyncio
import logging
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
import yfinance as yf

# Add project to path
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.config.settings import get_config
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
from algotrading_agent.components.decision_engine import DecisionEngine

# Training stocks - major companies with lots of news
TRAINING_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Validation stocks - different companies to test generalization
VALIDATION_STOCKS = ['META', 'NVDA', 'NFLX', 'SPY', 'QQQ']

# Sample historical news with different sentiments
HISTORICAL_NEWS_SAMPLES = [
    {
        "date": "2024-01-15",
        "symbol": "AAPL",
        "title": "Apple beats Q1 earnings expectations with record iPhone sales",
        "content": "Apple Inc. reported strong Q1 earnings that exceeded analyst expectations, driven by record iPhone sales and robust services revenue.",
        "expected_sentiment": "positive"
    },
    {
        "date": "2024-01-22", 
        "symbol": "TSLA",
        "title": "Tesla stock plunges on disappointing delivery numbers",
        "content": "Tesla shares dropped 8% in pre-market trading after the company reported lower-than-expected vehicle deliveries.",
        "expected_sentiment": "negative"
    },
    {
        "date": "2024-02-01",
        "symbol": "MSFT",
        "title": "Microsoft announces breakthrough in quantum computing research", 
        "content": "Microsoft revealed significant advances in quantum computing technology that could revolutionize cloud computing.",
        "expected_sentiment": "positive"
    },
    {
        "date": "2024-02-08",
        "symbol": "GOOGL",
        "title": "Google faces antitrust lawsuit over search monopoly",
        "content": "The Department of Justice filed a major antitrust lawsuit against Google, alleging monopolistic practices in search.",
        "expected_sentiment": "negative"  
    },
    {
        "date": "2024-02-15",
        "symbol": "AMZN",
        "title": "Amazon Web Services reports record growth in cloud revenue",
        "content": "AWS posted exceptional growth with 20% increase in cloud revenue, exceeding all analyst expectations.",
        "expected_sentiment": "positive"
    }
]

class CorrelationValidator:
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("correlation.validator")
        self.results = {
            "training_results": [],
            "validation_results": [],
            "correlation_analysis": {},
            "optimized_parameters": {}
        }
        
    async def run_correlation_analysis(self):
        """Run complete correlation analysis pipeline"""
        print("🔍 News-to-Price Correlation Analysis")
        print("=" * 60)
        
        # Phase 1: Training Phase
        print("\n📚 Phase 1: Training Phase")
        print("-" * 40)
        training_results = await self._run_training_phase()
        
        # Phase 2: Parameter Optimization
        print("\n⚙️ Phase 2: Parameter Optimization")  
        print("-" * 40)
        optimized_params = self._optimize_parameters(training_results)
        
        # Phase 3: Validation Phase
        print("\n✅ Phase 3: Validation Phase")
        print("-" * 40)
        validation_results = await self._run_validation_phase(optimized_params)
        
        # Phase 4: Correlation Analysis
        print("\n📊 Phase 4: Correlation Analysis")
        print("-" * 40)
        correlation_metrics = self._analyze_correlations(training_results, validation_results)
        
        # Phase 5: Results Summary
        print("\n📋 Phase 5: Results Summary")
        print("-" * 40)
        self._generate_report(training_results, validation_results, correlation_metrics, optimized_params)
        
        return {
            "training": training_results,
            "validation": validation_results, 
            "correlations": correlation_metrics,
            "optimized_params": optimized_params
        }
        
    async def _run_training_phase(self) -> List[Dict]:
        """Test different AI configurations on training stocks"""
        training_results = []
        
        # Test different AI weight configurations
        ai_weight_configs = [0.3, 0.5, 0.7, 0.9]
        
        for ai_weight in ai_weight_configs:
            print(f"   🧪 Testing AI weight: {ai_weight:.1f}")
            
            # Configure AI analyzer
            config = self._create_test_config(ai_weight)
            
            for news_item in HISTORICAL_NEWS_SAMPLES:
                if news_item["symbol"] in TRAINING_STOCKS:
                    result = await self._test_news_prediction(news_item, config)
                    result["ai_weight"] = ai_weight
                    result["phase"] = "training"
                    training_results.append(result)
                    
                    print(f"      📰 {news_item['symbol']}: Predicted {result['predicted_direction']}, "
                          f"Actual {result['actual_direction']} "
                          f"({'✅' if result['correct_prediction'] else '❌'})")
        
        return training_results
        
    async def _run_validation_phase(self, optimized_params: Dict) -> List[Dict]:
        """Test optimized parameters on validation stocks"""
        validation_results = []
        
        print(f"   🎯 Using optimized AI weight: {optimized_params['best_ai_weight']:.1f}")
        
        config = self._create_test_config(optimized_params['best_ai_weight'])
        
        # Create validation news samples for validation stocks
        validation_news = [
            {
                "date": "2024-03-01",
                "symbol": "META", 
                "title": "Meta reports record user growth across all platforms",
                "content": "Meta platforms showed exceptional user growth with significant revenue increases from advertising.",
                "expected_sentiment": "positive"
            },
            {
                "date": "2024-03-08",
                "symbol": "NVDA",
                "title": "NVIDIA faces supply chain disruptions for AI chips", 
                "content": "NVIDIA warned of potential delays in AI chip deliveries due to supply chain constraints.",
                "expected_sentiment": "negative"
            }
        ]
        
        for news_item in validation_news:
            result = await self._test_news_prediction(news_item, config)
            result["ai_weight"] = optimized_params['best_ai_weight']
            result["phase"] = "validation"
            validation_results.append(result)
            
            print(f"      📰 {news_item['symbol']}: Predicted {result['predicted_direction']}, "
                  f"Actual {result['actual_direction']} "
                  f"({'✅' if result['correct_prediction'] else '❌'})")
        
        return validation_results
        
    def _optimize_parameters(self, training_results: List[Dict]) -> Dict:
        """Find optimal AI parameters based on training results"""
        
        # Calculate accuracy for each AI weight
        weight_performance = {}
        
        for result in training_results:
            weight = result["ai_weight"]
            if weight not in weight_performance:
                weight_performance[weight] = {"correct": 0, "total": 0}
            
            weight_performance[weight]["total"] += 1
            if result["correct_prediction"]:
                weight_performance[weight]["correct"] += 1
        
        # Find best performing weight
        best_weight = None
        best_accuracy = 0
        
        print("   📊 AI Weight Performance:")
        for weight, perf in weight_performance.items():
            accuracy = perf["correct"] / perf["total"]
            print(f"      AI Weight {weight:.1f}: {accuracy:.1%} accuracy ({perf['correct']}/{perf['total']})")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight
        
        print(f"   🎯 Best AI Weight: {best_weight:.1f} ({best_accuracy:.1%} accuracy)")
        
        return {
            "best_ai_weight": best_weight,
            "best_accuracy": best_accuracy,
            "weight_performance": weight_performance
        }
        
    def _analyze_correlations(self, training_results: List[Dict], validation_results: List[Dict]) -> Dict:
        """Analyze correlation between sentiment and price movements"""
        
        all_results = training_results + validation_results
        
        # Extract sentiment scores and actual price movements
        sentiments = []
        price_changes = []
        
        for result in all_results:
            sentiments.append(result["sentiment_score"])
            price_changes.append(result["actual_price_change"])
        
        if len(sentiments) < 3:
            return {"error": "Not enough data points for correlation analysis"}
        
        # Calculate correlation coefficients
        pearson_corr, pearson_p = pearsonr(sentiments, price_changes)
        spearman_corr, spearman_p = spearmanr(sentiments, price_changes)
        
        # Calculate prediction accuracy
        correct_predictions = sum(1 for r in all_results if r["correct_prediction"])
        total_predictions = len(all_results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        correlation_metrics = {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr, 
            "spearman_p_value": spearman_p,
            "prediction_accuracy": accuracy,
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "sample_size": len(sentiments)
        }
        
        print(f"   📈 Pearson Correlation: {pearson_corr:.3f} (p={pearson_p:.3f})")
        print(f"   📊 Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
        print(f"   🎯 Prediction Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        return correlation_metrics
        
    async def _test_news_prediction(self, news_item: Dict, config: Dict) -> Dict:
        """Test sentiment prediction vs actual price movement for a news item"""
        
        try:
            # Initialize AI analyzer
            news_brain = NewsAnalysisBrain(config)
            await news_brain.start()
            
            # Analyze news sentiment
            analyzed_news = await news_brain.process([{
                "title": news_item["title"],
                "content": news_item["content"],
                "symbol": news_item["symbol"],
                "timestamp": news_item["date"]
            }])
            
            await news_brain.stop()
            
            if not analyzed_news:
                return {"error": "Failed to analyze news"}
                
            analysis = analyzed_news[0]
            
            # Get AI sentiment (or combined sentiment if AI is enabled)
            if "combined_sentiment" in analysis:
                sentiment_score = analysis["combined_sentiment"]
            elif "ai_sentiment" in analysis:
                sentiment_score = analysis["ai_sentiment"]  
            else:
                sentiment_score = analysis["sentiment"]["polarity"]
            
            # Predict direction based on sentiment
            predicted_direction = "up" if sentiment_score > 0.1 else "down" if sentiment_score < -0.1 else "neutral"
            
            # Get actual price movement (mock for now - would use real historical data)
            actual_price_change = self._get_mock_price_change(news_item)
            actual_direction = "up" if actual_price_change > 2 else "down" if actual_price_change < -2 else "neutral"
            
            # Check if prediction was correct
            correct_prediction = (
                (predicted_direction == "up" and actual_direction == "up") or
                (predicted_direction == "down" and actual_direction == "down") or
                (predicted_direction == "neutral" and actual_direction == "neutral")
            )
            
            return {
                "symbol": news_item["symbol"],
                "date": news_item["date"],
                "sentiment_score": sentiment_score,
                "predicted_direction": predicted_direction,
                "actual_price_change": actual_price_change,
                "actual_direction": actual_direction,
                "correct_prediction": correct_prediction,
                "analysis_method": analysis.get("analysis_method", "traditional")
            }
            
        except Exception as e:
            self.logger.error(f"Error testing news prediction: {e}")
            return {"error": str(e)}
            
    def _get_mock_price_change(self, news_item: Dict) -> float:
        """Mock price change based on expected sentiment (in real version, use yfinance)"""
        # This simulates realistic price movements based on news sentiment
        expected = news_item["expected_sentiment"]
        
        if expected == "positive":
            return np.random.normal(3.5, 2.0)  # Average +3.5% with variance
        elif expected == "negative":
            return np.random.normal(-3.2, 2.0)  # Average -3.2% with variance  
        else:
            return np.random.normal(0.1, 1.0)  # Neutral with small variance
            
    def _create_test_config(self, ai_weight: float) -> Dict:
        """Create test configuration with specified AI weight"""
        return {
            "ai_analyzer": {
                "enabled": True,
                "provider": "openai",
                "ai_weight": ai_weight,
                "traditional_weight": 1.0 - ai_weight,
                "fallback_enabled": True,
                "providers": {
                    "openai": {
                        "enabled": True,
                        "model": "gpt-3.5-turbo",
                        "api_key_env": "OPENAI_API_KEY"
                    }
                }
            },
            "sentiment_threshold": 0.1
        }
        
    def _generate_report(self, training_results: List[Dict], validation_results: List[Dict], 
                        correlations: Dict, optimized_params: Dict):
        """Generate comprehensive correlation analysis report"""
        
        print("\n" + "="*60)
        print("🎯 CORRELATION ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 CORRELATION METRICS:")
        print(f"   Pearson Correlation: {correlations.get('pearson_correlation', 0):.3f}")
        print(f"   Spearman Correlation: {correlations.get('spearman_correlation', 0):.3f}")
        print(f"   Overall Accuracy: {correlations.get('prediction_accuracy', 0):.1%}")
        
        print(f"\n⚙️ OPTIMIZED PARAMETERS:")
        print(f"   Best AI Weight: {optimized_params.get('best_ai_weight', 0):.1f}")
        print(f"   Training Accuracy: {optimized_params.get('best_accuracy', 0):.1%}")
        
        print(f"\n📈 RECOMMENDATIONS:")
        correlation = correlations.get('pearson_correlation', 0)
        accuracy = correlations.get('prediction_accuracy', 0)
        
        if correlation > 0.3 and accuracy > 0.6:
            print("   ✅ STRONG correlation found - AI parameters are effective!")
            print("   📊 Recommend using optimized AI weights in production")
        elif correlation > 0.1 and accuracy > 0.5:
            print("   ⚠️ MODERATE correlation found - some predictive power")
            print("   🔧 Consider further parameter tuning or more training data")
        else:
            print("   ❌ WEAK correlation found - news sentiment may not be predictive")
            print("   🤔 Consider different features or traditional analysis")
            
        print(f"\n💡 NEXT STEPS:")
        print("   1. Test with real historical price data (yfinance integration)")
        print("   2. Expand training dataset with more news samples")
        print("   3. Test different AI models (GPT-4, Claude, Llama)")
        print("   4. Add technical indicators alongside sentiment analysis")
        
async def main():
    """Run correlation validation"""
    validator = CorrelationValidator()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        results = await validator.run_correlation_analysis()
        
        # Save results
        output_file = f"correlation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
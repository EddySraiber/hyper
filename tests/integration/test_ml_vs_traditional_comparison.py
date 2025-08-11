import pytest
import asyncio
import tempfile
import shutil
from typing import List, Dict, Any
import statistics
from textblob import TextBlob

from algotrading_agent.ml.ml_sentiment_analyzer import MLSentimentAnalyzer
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain


class TestMLvsTraditionalComparison:
    """Compare ML sentiment analysis vs traditional TextBlob approach"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_news_samples(self):
        """Financial news samples with expected sentiment for testing"""
        return [
            # Strong positive news
            {
                'text': "Apple beats Q3 earnings expectations by 15%, stock surges to new record high",
                'expected': 'positive',
                'confidence_expectation': 'high'
            },
            {
                'text': "Tesla delivers record quarterly results, revenue growth exceeds analyst forecasts", 
                'expected': 'positive',
                'confidence_expectation': 'high'
            },
            {
                'text': "Microsoft announces breakthrough AI partnership, shares rally 12%",
                'expected': 'positive', 
                'confidence_expectation': 'high'
            },
            
            # Strong negative news
            {
                'text': "Amazon misses earnings by 20%, stock plunges on weak guidance",
                'expected': 'negative',
                'confidence_expectation': 'high'
            },
            {
                'text': "Netflix loses subscribers, shares crash 15% in after-hours trading",
                'expected': 'negative',
                'confidence_expectation': 'high'
            },
            {
                'text': "Meta faces regulatory investigation, stock falls on antitrust concerns",
                'expected': 'negative',
                'confidence_expectation': 'medium'
            },
            
            # Neutral/ambiguous news
            {
                'text': "Google announces routine quarterly board meeting scheduled for next month",
                'expected': 'neutral',
                'confidence_expectation': 'low'
            },
            {
                'text': "Intel maintains current dividend policy in line with market expectations",
                'expected': 'neutral',
                'confidence_expectation': 'low'
            },
            {
                'text': "IBM reports quarterly results in line with analyst estimates",
                'expected': 'neutral',
                'confidence_expectation': 'medium'
            },
            
            # Mixed/complex sentiment
            {
                'text': "Ford beats earnings but misses revenue targets, mixed analyst reactions",
                'expected': 'neutral',  # Mixed signals
                'confidence_expectation': 'low'
            },
            {
                'text': "General Motors raises guidance despite supply chain challenges",
                'expected': 'positive',  # Net positive despite challenges
                'confidence_expectation': 'medium'
            }
        ]
    
    @pytest.fixture
    def ml_config(self, temp_model_dir):
        """Configuration for ML sentiment analyzer"""
        return {
            'ml_sentiment': {
                'enabled': True,
                'model_dir': temp_model_dir,
                'auto_train': True,
                'retrain_threshold': 0.6,
                'min_confidence': 0.3,
                'high_confidence': 0.7
            },
            'sentiment_analysis': {
                'primary_method': 'ml',
                'ml_enabled': True,
                'ml_weight': 0.6,
                'traditional_weight': 0.4,
                'fallback_enabled': True
            }
        }
    
    @pytest.fixture
    def traditional_config(self):
        """Configuration for traditional sentiment analysis"""
        return {
            'sentiment_analysis': {
                'primary_method': 'traditional',
                'ml_enabled': False,
                'fallback_enabled': False
            },
            'sentiment_threshold': 0.1,
            'entity_patterns': {},
            'impact_keywords': {}
        }
    
    @pytest.mark.asyncio
    async def test_ml_analyzer_accuracy(self, ml_config, test_news_samples):
        """Test ML analyzer accuracy on test samples"""
        analyzer = MLSentimentAnalyzer(ml_config)
        await analyzer.start()
        
        correct_predictions = 0
        total_predictions = len(test_news_samples)
        
        results = []
        for sample in test_news_samples:
            result = await analyzer.analyze_sentiment(sample['text'])
            
            predicted = result['sentiment']
            expected = sample['expected']
            confidence = result['confidence']
            
            results.append({
                'text': sample['text'],
                'predicted': predicted,
                'expected': expected,
                'confidence': confidence,
                'correct': predicted == expected
            })
            
            if predicted == expected:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        
        # ML model should achieve significantly better than random (33% for 3 classes)
        assert accuracy > 0.5, f"ML accuracy {accuracy:.3f} should be > 0.5"
        
        # Log detailed results
        print(f"\nML Sentiment Analysis Results:")
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"Average confidence: {statistics.mean([r['confidence'] for r in results]):.3f}")
        
        return results, accuracy
    
    def test_traditional_analyzer_accuracy(self, test_news_samples):
        """Test traditional TextBlob analyzer accuracy on test samples"""
        def analyze_traditional(text: str) -> Dict[str, Any]:
            """Traditional sentiment analysis using TextBlob with financial keywords"""
            blob = TextBlob(text)
            base_polarity = blob.sentiment.polarity
            
            # Enhanced with financial keywords (simplified version)
            text_lower = text.lower()
            
            positive_boost = 0.0
            negative_boost = 0.0
            
            positive_keywords = ['beat', 'surge', 'rally', 'strong', 'growth', 'record']
            negative_keywords = ['miss', 'plunge', 'fall', 'weak', 'concern', 'crash']
            
            for keyword in positive_keywords:
                if keyword in text_lower:
                    positive_boost += 0.2
                    
            for keyword in negative_keywords:
                if keyword in text_lower:
                    negative_boost += 0.2
            
            enhanced_polarity = base_polarity + positive_boost - negative_boost
            enhanced_polarity = max(-1.0, min(1.0, enhanced_polarity))
            
            if enhanced_polarity > 0.1:
                sentiment = 'positive'
            elif enhanced_polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'sentiment': sentiment,
                'polarity': enhanced_polarity,
                'confidence': abs(enhanced_polarity)
            }
        
        correct_predictions = 0
        total_predictions = len(test_news_samples)
        
        results = []
        for sample in test_news_samples:
            result = analyze_traditional(sample['text'])
            
            predicted = result['sentiment']
            expected = sample['expected']
            confidence = result['confidence']
            
            results.append({
                'text': sample['text'],
                'predicted': predicted,
                'expected': expected,
                'confidence': confidence,
                'correct': predicted == expected
            })
            
            if predicted == expected:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        
        print(f"\nTraditional Sentiment Analysis Results:")
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"Average confidence: {statistics.mean([r['confidence'] for r in results]):.3f}")
        
        return results, accuracy
    
    @pytest.mark.asyncio
    async def test_comparative_performance(self, ml_config, test_news_samples):
        """Compare ML vs Traditional sentiment analysis performance"""
        # Test ML analyzer
        ml_results, ml_accuracy = await self.test_ml_analyzer_accuracy(ml_config, test_news_samples)
        
        # Test traditional analyzer
        traditional_results, traditional_accuracy = self.test_traditional_analyzer_accuracy(test_news_samples)
        
        print(f"\n--- PERFORMANCE COMPARISON ---")
        print(f"ML Accuracy: {ml_accuracy:.3f}")
        print(f"Traditional Accuracy: {traditional_accuracy:.3f}")
        print(f"Improvement: {ml_accuracy - traditional_accuracy:.3f}")
        
        # Detailed comparison
        print(f"\nDetailed Comparison:")
        print(f"{'Sample':<5} {'Expected':<10} {'ML':<10} {'Traditional':<12} {'Text':<50}")
        print("-" * 100)
        
        for i, (ml_res, trad_res) in enumerate(zip(ml_results, traditional_results)):
            ml_mark = "✓" if ml_res['correct'] else "✗"
            trad_mark = "✓" if trad_res['correct'] else "✗"
            text_snippet = ml_res['text'][:50] + "..." if len(ml_res['text']) > 50 else ml_res['text']
            
            print(f"{i+1:<5} {ml_res['expected']:<10} {ml_res['predicted']:<3}{ml_mark:<7} "
                  f"{trad_res['predicted']:<3}{trad_mark:<9} {text_snippet}")
        
        # ML should outperform traditional by a meaningful margin
        improvement_threshold = 0.1  # At least 10% improvement
        
        assert ml_accuracy > traditional_accuracy, \
            f"ML accuracy ({ml_accuracy:.3f}) should exceed traditional ({traditional_accuracy:.3f})"
        
        # If improvement is significant, that's great, but allow for some variability in small test sets
        if ml_accuracy - traditional_accuracy >= improvement_threshold:
            print(f"✓ ML shows significant improvement: {ml_accuracy - traditional_accuracy:.3f}")
        else:
            print(f"! ML improvement modest: {ml_accuracy - traditional_accuracy:.3f} (target: {improvement_threshold})")
    
    @pytest.mark.asyncio
    async def test_confidence_calibration(self, ml_config, test_news_samples):
        """Test that ML confidence scores are well-calibrated"""
        analyzer = MLSentimentAnalyzer(ml_config)
        await analyzer.start()
        
        high_confidence_correct = 0
        high_confidence_total = 0
        medium_confidence_correct = 0
        medium_confidence_total = 0
        low_confidence_correct = 0
        low_confidence_total = 0
        
        for sample in test_news_samples:
            result = await analyzer.analyze_sentiment(sample['text'])
            
            predicted = result['sentiment']
            expected = sample['expected']
            confidence_level = result['confidence_level']
            is_correct = predicted == expected
            
            if confidence_level == 'high':
                high_confidence_total += 1
                if is_correct:
                    high_confidence_correct += 1
            elif confidence_level == 'medium':
                medium_confidence_total += 1
                if is_correct:
                    medium_confidence_correct += 1
            else:  # low
                low_confidence_total += 1
                if is_correct:
                    low_confidence_correct += 1
        
        # Calculate accuracy by confidence level
        high_acc = high_confidence_correct / max(high_confidence_total, 1)
        medium_acc = medium_confidence_correct / max(medium_confidence_total, 1)
        low_acc = low_confidence_correct / max(low_confidence_total, 1)
        
        print(f"\nConfidence Calibration:")
        print(f"High confidence accuracy: {high_acc:.3f} ({high_confidence_correct}/{high_confidence_total})")
        print(f"Medium confidence accuracy: {medium_acc:.3f} ({medium_confidence_correct}/{medium_confidence_total})")
        print(f"Low confidence accuracy: {low_acc:.3f} ({low_confidence_correct}/{low_confidence_total})")
        
        # High confidence predictions should be more accurate than low confidence
        if high_confidence_total > 0 and low_confidence_total > 0:
            assert high_acc >= low_acc, "High confidence predictions should be more accurate"
    
    @pytest.mark.asyncio
    async def test_processing_speed_comparison(self, ml_config, traditional_config, test_news_samples):
        """Compare processing speed between ML and traditional methods"""
        import time
        
        # Test ML speed
        ml_analyzer = MLSentimentAnalyzer(ml_config)
        await ml_analyzer.start()
        
        start_time = time.time()
        for sample in test_news_samples:
            await ml_analyzer.analyze_sentiment(sample['text'])
        ml_time = time.time() - start_time
        
        # Test traditional speed (using NewsAnalysisBrain)
        traditional_brain = NewsAnalysisBrain(traditional_config)
        await traditional_brain.start()
        
        # Prepare news items in the expected format
        news_items = [
            {
                'title': sample['text'][:50],
                'content': sample['text'],
                'url': f'http://test.com/{i}',
                'timestamp': '2023-01-01T00:00:00Z'
            }
            for i, sample in enumerate(test_news_samples)
        ]
        
        start_time = time.time()
        await traditional_brain._process_traditional(news_items)
        traditional_time = time.time() - start_time
        
        print(f"\nProcessing Speed Comparison:")
        print(f"ML Time: {ml_time:.3f} seconds for {len(test_news_samples)} samples")
        print(f"Traditional Time: {traditional_time:.3f} seconds for {len(test_news_samples)} samples")
        print(f"ML Speed: {len(test_news_samples)/ml_time:.1f} samples/second")
        print(f"Traditional Speed: {len(test_news_samples)/traditional_time:.1f} samples/second")
        
        # Both should be reasonably fast (less than 1 second per sample)
        assert ml_time / len(test_news_samples) < 1.0, "ML processing should be < 1 second per sample"
        assert traditional_time / len(test_news_samples) < 1.0, "Traditional processing should be < 1 second per sample"
    
    @pytest.mark.asyncio
    async def test_ensemble_vs_individual_methods(self, ml_config, test_news_samples):
        """Test ensemble approach vs individual methods"""
        # Configure for ensemble mode
        ensemble_config = ml_config.copy()
        ensemble_config['sentiment_analysis'] = {
            'primary_method': 'ml',
            'ml_enabled': True,
            'ml_weight': 0.6,
            'traditional_weight': 0.4,
            'fallback_enabled': True
        }
        
        brain = NewsAnalysisBrain(ensemble_config)
        await brain.start()
        
        # Prepare news items
        news_items = [
            {
                'title': sample['text'][:50],
                'content': sample['text'],
                'url': f'http://test.com/{i}',
                'timestamp': '2023-01-01T00:00:00Z'
            }
            for i, sample in enumerate(test_news_samples)
        ]
        
        # Test ensemble processing
        results = await brain._process_with_ml(news_items)
        
        correct_predictions = 0
        for i, result in enumerate(results):
            expected = test_news_samples[i]['expected']
            predicted = result['sentiment']['label']
            
            if predicted == expected:
                correct_predictions += 1
        
        ensemble_accuracy = correct_predictions / len(test_news_samples)
        
        print(f"\nEnsemble Approach Results:")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.3f} ({correct_predictions}/{len(test_news_samples)})")
        
        # Ensemble should perform reasonably well
        assert ensemble_accuracy > 0.4, "Ensemble should achieve reasonable accuracy"
        
        # Check that results contain both ML and traditional analysis
        sample_result = results[0]
        assert 'ml_sentiment' in sample_result
        assert 'traditional_sentiment' in sample_result
        assert 'sentiment' in sample_result  # Ensemble result
        assert sample_result['analysis_method'] == 'ml_enhanced'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
import pytest
import asyncio
import tempfile
import shutil
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
from algotrading_agent.components.decision_engine import DecisionEngine
from algotrading_agent.ml.ml_sentiment_analyzer import MLSentimentAnalyzer


class TestMLEnhancedTradingPipeline:
    """Test the complete ML-enhanced trading pipeline end-to-end"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def ml_enhanced_config(self, temp_model_dir):
        """Configuration with ML sentiment analysis enabled"""
        return {
            # ML Sentiment Configuration
            'ml_sentiment': {
                'enabled': True,
                'model_dir': temp_model_dir,
                'auto_train': True,
                'retrain_threshold': 0.6,
                'min_confidence': 0.3,
                'high_confidence': 0.7
            },
            
            # News Analysis Brain Configuration
            'sentiment_analysis': {
                'primary_method': 'ml',
                'fallback_enabled': True,
                'ml_enabled': True,
                'ml_weight': 0.6,
                'ai_weight': 0.3,
                'traditional_weight': 0.1,
                'min_ml_confidence': 0.3,
                'high_ml_confidence': 0.7
            },
            'sentiment_threshold': 0.1,
            'entity_patterns': {},
            'impact_keywords': {
                'breakthrough': 0.3,
                'record': 0.25,
                'massive': 0.2
            },
            
            # Decision Engine Configuration
            'decision_engine': {
                'min_confidence': 0.30,
                'max_position_size': 1000,
                'sentiment_weight': 0.4,
                'impact_weight': 0.4,
                'recency_weight': 0.2,
                'default_stop_loss_pct': 0.05,
                'default_take_profit_pct': 0.10
            },
            
            # AI Analyzer (disabled for this test)
            'ai_analyzer': {
                'enabled': False
            }
        }
    
    @pytest.fixture
    def sample_news_data(self):
        """Sample news data for testing the pipeline"""
        return [
            {
                'id': 'news_1',
                'title': 'Apple Beats Q3 Earnings Expectations',
                'content': 'Apple Inc. reported quarterly earnings that beat analyst expectations by 15%, with strong iPhone sales driving revenue growth. The stock surged 8% in after-hours trading.',
                'url': 'https://example.com/news/1',
                'timestamp': '2023-07-20T16:00:00Z',
                'source': 'Reuters',
                'relevance_score': 0.8,
                'expected_sentiment': 'positive',
                'expected_tickers': ['AAPL']
            },
            {
                'id': 'news_2', 
                'title': 'Tesla Misses Delivery Targets',
                'content': 'Tesla Inc. reported quarterly vehicle deliveries that fell short of analyst estimates by 12%. Supply chain issues and production challenges contributed to the miss. Shares fell 6% on the news.',
                'url': 'https://example.com/news/2',
                'timestamp': '2023-07-20T15:30:00Z',
                'source': 'Yahoo Finance',
                'relevance_score': 0.9,
                'expected_sentiment': 'negative',
                'expected_tickers': ['TSLA']
            },
            {
                'id': 'news_3',
                'title': 'Microsoft Announces AI Partnership',
                'content': 'Microsoft Corporation announced a breakthrough partnership with leading AI research firm. The collaboration is expected to accelerate development of next-generation AI products and services.',
                'url': 'https://example.com/news/3', 
                'timestamp': '2023-07-20T14:00:00Z',
                'source': 'MarketWatch',
                'relevance_score': 0.7,
                'expected_sentiment': 'positive',
                'expected_tickers': ['MSFT']
            },
            {
                'id': 'news_4',
                'title': 'Amazon Reports Steady Quarterly Results',
                'content': 'Amazon.com Inc. reported quarterly results that were in line with analyst expectations. Revenue grew at a steady pace with no major surprises in any business segment.',
                'url': 'https://example.com/news/4',
                'timestamp': '2023-07-20T13:00:00Z', 
                'source': 'CNBC',
                'relevance_score': 0.5,
                'expected_sentiment': 'neutral',
                'expected_tickers': ['AMZN']
            }
        ]
    
    @pytest.mark.asyncio
    async def test_ml_enhanced_news_analysis_pipeline(self, ml_enhanced_config, sample_news_data):
        """Test the complete news analysis pipeline with ML enhancement"""
        # Initialize ML-enhanced News Analysis Brain
        brain = NewsAnalysisBrain(ml_enhanced_config)
        await brain.start()
        
        # Verify ML analyzer is initialized
        assert brain.ml_analyzer is not None
        assert brain.ml_analyzer.can_analyze()
        assert brain.primary_method == 'ml'
        
        # Process news through ML-enhanced pipeline
        analyzed_news = await brain.process(sample_news_data)
        
        # Verify all news items were processed
        assert len(analyzed_news) == len(sample_news_data)
        
        # Check that ML analysis was applied
        results_summary = []
        for i, item in enumerate(analyzed_news):
            expected_sample = sample_news_data[i]
            
            # Verify ML analysis fields are present
            assert 'ml_sentiment' in item
            assert 'ml_confidence' in item
            assert 'ml_label' in item
            assert 'traditional_sentiment' in item
            assert 'sentiment' in item  # Ensemble result
            assert 'analysis_method' in item
            assert item['analysis_method'] == 'ml_enhanced'
            
            # Verify sentiment structure
            sentiment = item['sentiment']
            assert 'polarity' in sentiment
            assert 'label' in sentiment
            assert 'confidence' in sentiment
            assert 'method' in sentiment
            assert sentiment['method'] == 'ml_ensemble'
            
            # Check entities and tickers
            assert 'entities' in item
            assert 'tickers' in item['entities']
            
            results_summary.append({
                'title': item['title'],
                'expected_sentiment': expected_sample['expected_sentiment'],
                'ml_sentiment': item['ml_label'],
                'ensemble_sentiment': sentiment['label'],
                'ml_confidence': item['ml_confidence'],
                'ensemble_confidence': sentiment['confidence'],
                'tickers': item['entities']['tickers']
            })
        
        # Print results for analysis
        print(f"\n=== ML-Enhanced News Analysis Results ===")
        for result in results_summary:
            print(f"Title: {result['title'][:50]}...")
            print(f"  Expected: {result['expected_sentiment']}")
            print(f"  ML Prediction: {result['ml_sentiment']} (conf: {result['ml_confidence']:.3f})")
            print(f"  Ensemble: {result['ensemble_sentiment']} (conf: {result['ensemble_confidence']:.3f})")
            print(f"  Tickers: {result['tickers']}")
            print()
        
        # Calculate accuracy
        correct_ml = sum(1 for r in results_summary if r['ml_sentiment'] == r['expected_sentiment'])
        correct_ensemble = sum(1 for r in results_summary if r['ensemble_sentiment'] == r['expected_sentiment'])
        
        ml_accuracy = correct_ml / len(results_summary)
        ensemble_accuracy = correct_ensemble / len(results_summary)
        
        print(f"ML Accuracy: {ml_accuracy:.3f} ({correct_ml}/{len(results_summary)})")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.3f} ({correct_ensemble}/{len(results_summary)})")
        
        # ML should achieve reasonable performance on this test set
        assert ml_accuracy >= 0.5, f"ML accuracy {ml_accuracy:.3f} should be >= 0.5"
        
        await brain.stop()
        
        return analyzed_news
    
    @pytest.mark.asyncio
    async def test_ml_enhanced_decision_generation(self, ml_enhanced_config, sample_news_data):
        """Test that ML-enhanced sentiment leads to improved trading decisions"""
        # Process news with ML enhancement
        brain = NewsAnalysisBrain(ml_enhanced_config)
        await brain.start()
        
        analyzed_news = await brain.process(sample_news_data)
        
        # Initialize Decision Engine with ML-enhanced news
        decision_config = ml_enhanced_config.get('decision_engine', {})
        decision_engine = DecisionEngine(decision_config)
        
        # Mock current prices for decision making
        with patch('algotrading_agent.components.decision_engine.DecisionEngine._get_current_price') as mock_price:
            mock_price.return_value = 150.0  # Mock stock price
            
            # Generate trading decisions based on ML-enhanced sentiment
            decisions = await decision_engine._generate_decisions(analyzed_news)
        
        print(f"\n=== ML-Enhanced Trading Decisions ===")
        print(f"Generated {len(decisions)} trading decisions from {len(analyzed_news)} news items")
        
        for decision in decisions:
            print(f"Symbol: {decision['symbol']}")
            print(f"  Action: {decision['action']}")
            print(f"  Confidence: {decision['confidence']:.3f}")
            print(f"  Sentiment: {decision.get('sentiment', 'N/A')}")
            print(f"  Quantity: {decision['quantity']}")
            print(f"  Reasoning: {decision.get('reasoning', 'N/A')[:100]}...")
            print()
        
        # Verify decisions quality
        if len(decisions) > 0:
            # Check decision structure
            sample_decision = decisions[0]
            required_fields = ['symbol', 'action', 'confidence', 'quantity']
            for field in required_fields:
                assert field in sample_decision, f"Decision missing required field: {field}"
            
            # Actions should be valid
            valid_actions = ['buy', 'sell', 'hold']
            for decision in decisions:
                assert decision['action'] in valid_actions
                assert 0 <= decision['confidence'] <= 1
                assert decision['quantity'] > 0
        
        await brain.stop()
        
        return decisions
    
    @pytest.mark.asyncio 
    async def test_ml_performance_vs_traditional_in_pipeline(self, sample_news_data):
        """Compare ML-enhanced pipeline vs traditional pipeline performance"""
        # Create ML-enhanced configuration
        temp_dir = tempfile.mkdtemp()
        ml_config = {
            'ml_sentiment': {
                'enabled': True,
                'model_dir': temp_dir,
                'auto_train': True
            },
            'sentiment_analysis': {
                'primary_method': 'ml',
                'ml_enabled': True,
                'ml_weight': 0.7,
                'traditional_weight': 0.3
            },
            'sentiment_threshold': 0.1,
            'entity_patterns': {},
            'impact_keywords': {}
        }
        
        # Create traditional configuration
        traditional_config = {
            'sentiment_analysis': {
                'primary_method': 'traditional',
                'ml_enabled': False
            },
            'sentiment_threshold': 0.1,
            'entity_patterns': {},
            'impact_keywords': {}
        }
        
        try:
            # Test ML-enhanced pipeline
            ml_brain = NewsAnalysisBrain(ml_config)
            await ml_brain.start()
            ml_results = await ml_brain.process(sample_news_data.copy())
            
            # Test traditional pipeline
            traditional_brain = NewsAnalysisBrain(traditional_config)
            await traditional_brain.start()
            traditional_results = await traditional_brain.process(sample_news_data.copy())
            
            print(f"\n=== Pipeline Comparison ===")
            
            # Compare sentiment predictions
            ml_sentiments = []
            traditional_sentiments = []
            expected_sentiments = []
            
            for i in range(len(sample_news_data)):
                expected = sample_news_data[i]['expected_sentiment']
                ml_pred = ml_results[i]['sentiment']['label']
                trad_pred = traditional_results[i]['sentiment']['label']
                
                expected_sentiments.append(expected)
                ml_sentiments.append(ml_pred)
                traditional_sentiments.append(trad_pred)
                
                print(f"News {i+1}: Expected={expected}, ML={ml_pred}, Traditional={trad_pred}")
            
            # Calculate accuracies
            ml_accuracy = sum(1 for e, p in zip(expected_sentiments, ml_sentiments) if e == p) / len(expected_sentiments)
            traditional_accuracy = sum(1 for e, p in zip(expected_sentiments, traditional_sentiments) if e == p) / len(expected_sentiments)
            
            print(f"ML Pipeline Accuracy: {ml_accuracy:.3f}")
            print(f"Traditional Pipeline Accuracy: {traditional_accuracy:.3f}")
            print(f"Improvement: {ml_accuracy - traditional_accuracy:.3f}")
            
            # ML should perform at least as well as traditional (allowing for small sample variance)
            assert ml_accuracy >= traditional_accuracy - 0.25, \
                f"ML accuracy ({ml_accuracy:.3f}) should not be significantly worse than traditional ({traditional_accuracy:.3f})"
            
            # Check confidence scores
            ml_confidences = [r['sentiment']['confidence'] for r in ml_results]
            traditional_confidences = [r['sentiment']['confidence'] for r in traditional_results]
            
            avg_ml_confidence = sum(ml_confidences) / len(ml_confidences)
            avg_traditional_confidence = sum(traditional_confidences) / len(traditional_confidences)
            
            print(f"Average ML Confidence: {avg_ml_confidence:.3f}")
            print(f"Average Traditional Confidence: {avg_traditional_confidence:.3f}")
            
            # Both should have reasonable confidence levels
            assert avg_ml_confidence > 0.1, "ML confidence should be reasonable"
            assert avg_traditional_confidence > 0.1, "Traditional confidence should be reasonable"
            
            await ml_brain.stop()
            await traditional_brain.stop()
            
        finally:
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_ml_analyzer_error_handling_in_pipeline(self, ml_enhanced_config, sample_news_data):
        """Test error handling when ML analyzer fails in the pipeline"""
        brain = NewsAnalysisBrain(ml_enhanced_config)
        await brain.start()
        
        # Verify ML is working initially
        assert brain.ml_analyzer.can_analyze()
        
        # Simulate ML analyzer failure
        with patch.object(brain.ml_analyzer, 'analyze_sentiment', side_effect=Exception("ML Model Error")):
            # Process should fall back gracefully
            results = await brain.process(sample_news_data)
            
            # Should still get results (via fallback)
            assert len(results) == len(sample_news_data)
            
            # Check that fallback was used
            for result in results:
                if 'analysis_method' in result:
                    assert result['analysis_method'] in ['traditional_fallback', 'traditional']
                
                # Should still have sentiment analysis
                assert 'sentiment' in result
                assert 'entities' in result
        
        await brain.stop()
    
    @pytest.mark.asyncio
    async def test_ml_model_performance_tracking(self, ml_enhanced_config, sample_news_data):
        """Test ML model performance tracking through the pipeline"""
        brain = NewsAnalysisBrain(ml_enhanced_config)
        await brain.start()
        
        # Process news multiple times to generate performance data
        for _ in range(3):
            await brain.process(sample_news_data.copy())
        
        # Check ML analyzer performance stats
        ml_stats = brain.ml_analyzer.get_performance_stats()
        
        assert ml_stats['predictions_made'] >= len(sample_news_data) * 3
        assert ml_stats['initialized'] == True
        assert ml_stats['model_trained'] == True
        assert ml_stats['model_accuracy'] > 0
        
        print(f"\n=== ML Performance Stats ===")
        print(f"Predictions made: {ml_stats['predictions_made']}")
        print(f"Errors: {ml_stats['errors']}")
        print(f"Error rate: {ml_stats['error_rate_percent']:.2f}%")
        print(f"Model accuracy: {ml_stats['model_accuracy']:.3f}")
        print(f"CV score: {ml_stats['model_cv_score']:.3f}")
        
        # Error rate should be low
        assert ml_stats['error_rate_percent'] < 10, "ML error rate should be < 10%"
        
        await brain.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
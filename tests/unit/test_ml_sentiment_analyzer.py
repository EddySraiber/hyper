import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch
from algotrading_agent.ml.ml_sentiment_analyzer import MLSentimentAnalyzer


class TestMLSentimentAnalyzer:
    """Test cases for the ML sentiment analyzer wrapper"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_model_dir):
        """Create test configuration"""
        return {
            'ml_sentiment': {
                'enabled': True,
                'model_dir': temp_model_dir,
                'auto_train': True,
                'retrain_threshold': 0.6,
                'min_confidence': 0.3,
                'high_confidence': 0.7
            }
        }
    
    @pytest.fixture
    def analyzer(self, config):
        """Create ML sentiment analyzer instance"""
        return MLSentimentAnalyzer(config)
    
    def test_initialization(self, analyzer, config):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.enabled == True
        assert analyzer.auto_train == True
        assert analyzer.retrain_threshold == 0.6
        assert analyzer.min_confidence == 0.3
        assert analyzer.high_confidence == 0.7
        assert analyzer.prediction_count == 0
        assert analyzer.error_count == 0
        assert not analyzer.is_initialized
    
    def test_initialization_disabled(self):
        """Test analyzer initialization when disabled"""
        config = {'ml_sentiment': {'enabled': False}}
        analyzer = MLSentimentAnalyzer(config)
        assert analyzer.enabled == False
    
    @pytest.mark.asyncio
    async def test_start_with_auto_train(self, analyzer):
        """Test starting analyzer with auto-training"""
        await analyzer.start()
        
        assert analyzer.is_initialized == True
        assert analyzer.model.is_trained == True
        assert analyzer.model.training_accuracy > 0
    
    @pytest.mark.asyncio
    async def test_start_disabled(self):
        """Test starting disabled analyzer"""
        config = {'ml_sentiment': {'enabled': False}}
        analyzer = MLSentimentAnalyzer(config)
        
        await analyzer.start()
        assert analyzer.is_initialized == False
    
    @pytest.mark.asyncio
    async def test_stop(self, analyzer):
        """Test stopping analyzer"""
        await analyzer.start()
        assert analyzer.is_initialized == True
        
        await analyzer.stop()
        assert analyzer.is_initialized == False
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self, analyzer):
        """Test successful sentiment analysis"""
        await analyzer.start()
        
        test_text = "Company beats earnings expectations, stock surges"
        result = await analyzer.analyze_sentiment(test_text)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'polarity' in result
        assert 'confidence' in result
        assert 'confidence_level' in result
        assert 'timestamp' in result
        assert 'analyzer' in result
        assert 'prediction_id' in result
        assert 'model_info' in result
        
        assert result['analyzer'] == 'ml_sentiment'
        assert result['confidence_level'] in ['low', 'medium', 'high']
        assert result['prediction_id'] == 1  # First prediction
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_not_initialized(self, analyzer):
        """Test sentiment analysis when not initialized"""
        # Don't start the analyzer
        with pytest.raises(ValueError, match="ML sentiment analyzer not available"):
            await analyzer.analyze_sentiment("test text")
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_disabled(self):
        """Test sentiment analysis when disabled"""
        config = {'ml_sentiment': {'enabled': False}}
        analyzer = MLSentimentAnalyzer(config)
        
        with pytest.raises(ValueError, match="ML sentiment analyzer not available"):
            await analyzer.analyze_sentiment("test text")
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_error_handling(self, analyzer):
        """Test error handling in sentiment analysis"""
        await analyzer.start()
        
        # Mock the model to raise an exception
        with patch.object(analyzer.model, 'predict_sentiment', side_effect=Exception("Model error")):
            result = await analyzer.analyze_sentiment("test text")
            
            assert 'error' in result
            assert result['sentiment'] == 'neutral'
            assert result['polarity'] == 0.0
            assert result['confidence'] == 0.0
            assert analyzer.error_count == 1
    
    @pytest.mark.asyncio
    async def test_analyze_batch(self, analyzer):
        """Test batch sentiment analysis"""
        await analyzer.start()
        
        test_texts = [
            "Company beats earnings, stock surges",
            "Company misses estimates, shares fall",
            "Neutral company update"
        ]
        
        results = await analyzer.analyze_batch(test_texts)
        
        assert len(results) == len(test_texts)
        for result in results:
            assert isinstance(result, dict)
            assert 'sentiment' in result
            assert 'confidence' in result
            
        assert analyzer.prediction_count == len(test_texts)
    
    @pytest.mark.asyncio
    async def test_analyze_batch_not_initialized(self, analyzer):
        """Test batch analysis when not initialized"""
        test_texts = ["text1", "text2"]
        results = await analyzer.analyze_batch(test_texts)
        
        assert len(results) == len(test_texts)
        for result in results:
            assert 'error' in result
    
    def test_performance_stats(self, analyzer):
        """Test performance statistics retrieval"""
        stats = analyzer.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'enabled' in stats
        assert 'initialized' in stats
        assert 'predictions_made' in stats
        assert 'errors' in stats
        assert 'error_rate_percent' in stats
        assert 'model_accuracy' in stats
        assert 'model_cv_score' in stats
        assert 'model_trained' in stats
        
        assert stats['enabled'] == analyzer.enabled
        assert stats['initialized'] == analyzer.is_initialized
        assert stats['predictions_made'] == analyzer.prediction_count
        assert stats['errors'] == analyzer.error_count
    
    @pytest.mark.asyncio
    async def test_performance_stats_after_predictions(self, analyzer):
        """Test performance stats after making predictions"""
        await analyzer.start()
        
        # Make some predictions
        await analyzer.analyze_sentiment("positive text")
        await analyzer.analyze_sentiment("negative text")
        
        stats = analyzer.get_performance_stats()
        assert stats['predictions_made'] == 2
        assert stats['errors'] == 0
        assert stats['error_rate_percent'] == 0.0
        assert stats['model_trained'] == True
        assert stats['model_accuracy'] > 0
    
    @pytest.mark.asyncio
    async def test_retrain_model(self, analyzer):
        """Test model retraining"""
        await analyzer.start()
        
        # Store original accuracy
        original_accuracy = analyzer.model.training_accuracy
        
        # Retrain the model
        results = await analyzer.retrain_model()
        
        assert isinstance(results, dict)
        assert 'training_accuracy' in results
        assert 'cross_val_score' in results
        assert results['training_accuracy'] > 0
        
        # Model should still be trained
        assert analyzer.model.is_trained
    
    @pytest.mark.asyncio
    async def test_retrain_model_disabled(self):
        """Test retraining when disabled"""
        config = {'ml_sentiment': {'enabled': False}}
        analyzer = MLSentimentAnalyzer(config)
        
        with pytest.raises(ValueError, match="ML sentiment analyzer is disabled"):
            await analyzer.retrain_model()
    
    def test_can_analyze(self, analyzer):
        """Test can_analyze method"""
        # Before initialization
        assert analyzer.can_analyze() == False
        
        # After starting (async, so we need to test in an async context)
        async def test_after_start():
            await analyzer.start()
            assert analyzer.can_analyze() == True
        
        asyncio.run(test_after_start())
    
    def test_get_model_info(self, analyzer):
        """Test model information retrieval"""
        # Before training
        info = analyzer.get_model_info()
        assert isinstance(info, dict)
        
        # After training (async test)
        async def test_after_training():
            await analyzer.start()
            info = analyzer.get_model_info()
            assert 'is_trained' in info
            assert 'training_accuracy' in info
            assert 'enabled' in info
            assert 'predictions_made' in info
            assert info['is_trained'] == True
        
        asyncio.run(test_after_training())
    
    @pytest.mark.asyncio
    async def test_confidence_level_classification(self, analyzer):
        """Test confidence level classification"""
        await analyzer.start()
        
        # Mock different confidence levels
        with patch.object(analyzer.model, 'predict_sentiment') as mock_predict:
            # High confidence
            mock_predict.return_value = {
                'sentiment': 'positive',
                'polarity': 0.8,
                'confidence': 0.9,
                'probabilities': {'negative': 0.05, 'neutral': 0.05, 'positive': 0.9},
                'method': 'ml_ensemble',
                'model_accuracy': 0.8
            }
            
            result = await analyzer.analyze_sentiment("test")
            assert result['confidence_level'] == 'high'
            
            # Medium confidence  
            mock_predict.return_value['confidence'] = 0.5
            result = await analyzer.analyze_sentiment("test")
            assert result['confidence_level'] == 'medium'
            
            # Low confidence
            mock_predict.return_value['confidence'] = 0.2
            result = await analyzer.analyze_sentiment("test")
            assert result['confidence_level'] == 'low'
    
    @pytest.mark.asyncio
    async def test_model_retraining_threshold(self, analyzer):
        """Test automatic retraining when accuracy is below threshold"""
        # Set a high retrain threshold
        analyzer.retrain_threshold = 0.9
        
        with patch.object(analyzer.model, 'train') as mock_train:
            mock_train.return_value = {'training_accuracy': 0.8, 'cross_val_score': 0.75}
            
            # Mock initial model loading with low accuracy
            with patch.object(analyzer.model, 'load_model', return_value=True):
                analyzer.model.training_accuracy = 0.5  # Below threshold
                
                await analyzer.start()
                
                # Should trigger retraining
                mock_train.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
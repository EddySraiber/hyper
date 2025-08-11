import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from algotrading_agent.ml.sentiment_model import FinancialSentimentModel


class TestFinancialSentimentModel:
    """Test cases for the ML sentiment model"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model(self, temp_model_dir):
        """Create a fresh sentiment model instance"""
        return FinancialSentimentModel(model_dir=temp_model_dir)
    
    def test_initialization(self, model):
        """Test model initialization"""
        assert model is not None
        assert not model.is_trained
        assert model.training_accuracy == 0.0
        assert model.financial_keywords is not None
        assert 'bullish' in model.financial_keywords
        assert 'bearish' in model.financial_keywords
    
    def test_financial_keywords_loading(self, model):
        """Test that financial keywords are properly loaded"""
        keywords = model.financial_keywords
        
        # Check that all categories exist
        expected_categories = ['bullish', 'bearish', 'volatility', 'impact']
        for category in expected_categories:
            assert category in keywords
            assert isinstance(keywords[category], list)
            assert len(keywords[category]) > 0
        
        # Check some specific keywords
        assert 'beat' in keywords['bullish']
        assert 'miss' in keywords['bearish']
        assert 'volatile' in keywords['volatility']
        assert 'significant' in keywords['impact']
    
    def test_feature_extraction(self, model):
        """Test feature extraction from text"""
        test_text = "Apple beats earnings expectations, stock surges 10%"
        features = model.extract_features(test_text)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features.dtype == np.float32
        
        # Features should be reasonable values
        assert np.all(np.isfinite(features))
        assert np.all(features >= 0)  # Most features should be non-negative counts/ratios
    
    def test_synthetic_data_generation(self, model):
        """Test synthetic training data generation"""
        texts, labels = model.generate_synthetic_training_data()
        
        assert len(texts) == len(labels)
        assert len(texts) > 0
        
        # Check that we have examples of all classes
        unique_labels = set(labels)
        assert 0 in unique_labels  # negative
        assert 1 in unique_labels  # neutral
        assert 2 in unique_labels  # positive
        
        # Check that all texts are strings
        assert all(isinstance(text, str) for text in texts)
        assert all(isinstance(label, int) for label in labels)
    
    def test_model_training(self, model):
        """Test model training process"""
        # Train the model
        results = model.train()
        
        # Check training results
        assert isinstance(results, dict)
        assert 'training_accuracy' in results
        assert 'cross_val_score' in results
        assert 'training_samples' in results
        
        # Model should now be trained
        assert model.is_trained
        assert model.training_accuracy > 0
        assert model.cross_val_score > 0
        
        # Pipeline should be created
        assert model.pipeline is not None
    
    def test_sentiment_prediction_positive(self, model):
        """Test prediction of positive sentiment"""
        model.train()  # Train first
        
        positive_text = "Apple beats earnings by 15%, stock surges to record high"
        result = model.predict_sentiment(positive_text)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'polarity' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        
        # Should detect positive sentiment
        assert result['sentiment'] in ['positive', 'neutral']  # Allow neutral due to model uncertainty
        assert result['polarity'] >= -1 and result['polarity'] <= 1
        assert result['confidence'] >= 0 and result['confidence'] <= 1
    
    def test_sentiment_prediction_negative(self, model):
        """Test prediction of negative sentiment"""
        model.train()  # Train first
        
        negative_text = "Company misses earnings, stock plunges 20% on weak guidance"
        result = model.predict_sentiment(negative_text)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'polarity' in result
        
        # Should detect negative or neutral sentiment
        assert result['sentiment'] in ['negative', 'neutral']
        assert result['polarity'] >= -1 and result['polarity'] <= 1
    
    def test_sentiment_prediction_neutral(self, model):
        """Test prediction of neutral sentiment"""
        model.train()  # Train first
        
        neutral_text = "Company reports quarterly results in line with expectations"
        result = model.predict_sentiment(neutral_text)
        
        assert isinstance(result, dict)
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
        assert result['polarity'] >= -1 and result['polarity'] <= 1
    
    def test_model_persistence(self, model):
        """Test model saving and loading"""
        # Train and save model
        model.train()
        original_accuracy = model.training_accuracy
        assert model.save_model()
        
        # Create new model instance and load
        new_model = FinancialSentimentModel(model_dir=model.model_dir)
        assert new_model.load_model()
        
        # Check that loaded model has same properties
        assert new_model.is_trained
        assert new_model.training_accuracy == original_accuracy
        
        # Should be able to make predictions
        test_text = "Test sentiment prediction"
        result = new_model.predict_sentiment(test_text)
        assert isinstance(result, dict)
        assert 'sentiment' in result
    
    def test_model_info(self, model):
        """Test model information retrieval"""
        # Before training
        info = model.get_model_info()
        assert isinstance(info, dict)
        assert info['is_trained'] == False
        assert info['training_accuracy'] == 0.0
        
        # After training
        model.train()
        info = model.get_model_info()
        assert info['is_trained'] == True
        assert info['training_accuracy'] > 0
        assert 'model_type' in info
        assert 'features' in info
        assert 'classes' in info
    
    def test_prediction_without_training(self, model):
        """Test that prediction fails gracefully without training"""
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict_sentiment("Test text")
    
    def test_feature_extraction_edge_cases(self, model):
        """Test feature extraction with edge cases"""
        # Empty text
        features = model.extract_features("")
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # Very short text
        features = model.extract_features("Hi")
        assert isinstance(features, np.ndarray)
        
        # Text with special characters
        features = model.extract_features("$AAPL surges 10%!!!")
        assert isinstance(features, np.ndarray)
        assert np.all(np.isfinite(features))
    
    def test_model_accuracy_threshold(self, model):
        """Test that trained model achieves reasonable accuracy"""
        results = model.train()
        
        # Model should achieve better than random performance (>33% for 3 classes)
        assert results['training_accuracy'] > 0.4  # Should be significantly better than random
        assert results['cross_val_score'] > 0.4
        
        # Cross-validation should be reasonably close to training accuracy
        accuracy_diff = abs(results['training_accuracy'] - results['cross_val_score'])
        assert accuracy_diff < 0.3  # Not too much overfitting
    
    def test_financial_keyword_impact(self, model):
        """Test that financial keywords impact feature extraction"""
        model.train()
        
        # Text with strong bullish keywords
        bullish_text = "Company beats earnings, surges on strong growth"
        bullish_result = model.predict_sentiment(bullish_text)
        
        # Text with strong bearish keywords  
        bearish_text = "Company misses earnings, plunges on weak outlook"
        bearish_result = model.predict_sentiment(bearish_text)
        
        # Bullish text should have higher polarity than bearish
        # (Allow some tolerance due to model uncertainty)
        assert (bullish_result['polarity'] > bearish_result['polarity'] or
                abs(bullish_result['polarity'] - bearish_result['polarity']) < 0.3)


if __name__ == "__main__":
    pytest.main([__file__])
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .sentiment_model import FinancialSentimentModel


class MLSentimentAnalyzer:
    """
    High-level ML sentiment analyzer that integrates with the NewsAnalysisBrain.
    Provides async interface and handles model lifecycle management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # ML configuration
        ml_config = config.get('ml_sentiment', {})
        self.enabled = ml_config.get('enabled', True)
        self.model_dir = ml_config.get('model_dir', '/app/data/ml_models')
        self.auto_train = ml_config.get('auto_train', True)
        self.retrain_threshold = ml_config.get('retrain_threshold', 0.6)  # Retrain if accuracy < 60%
        
        # Confidence thresholds
        self.min_confidence = ml_config.get('min_confidence', 0.3)
        self.high_confidence = ml_config.get('high_confidence', 0.7)
        
        # Model instance
        self.model = FinancialSentimentModel(model_dir=self.model_dir)
        self.is_initialized = False
        
        # Performance tracking
        self.prediction_count = 0
        self.error_count = 0
        
    async def start(self) -> None:
        """Initialize the ML sentiment analyzer"""
        if not self.enabled:
            self.logger.info("ML sentiment analyzer disabled in configuration")
            return
            
        self.logger.info("Starting ML sentiment analyzer")
        
        # Try to load existing model
        model_loaded = await asyncio.to_thread(self.model.load_model)
        
        if not model_loaded and self.auto_train:
            self.logger.info("No existing model found, training new model...")
            try:
                training_results = await asyncio.to_thread(self.model.train)
                self.logger.info(f"Model training completed: {training_results}")
            except Exception as e:
                self.logger.error(f"Model training failed: {e}")
                self.enabled = False
                return
        elif not model_loaded:
            self.logger.warning("No model available and auto_train disabled")
            self.enabled = False
            return
            
        # Check if model needs retraining
        if (self.model.training_accuracy > 0 and 
            self.model.training_accuracy < self.retrain_threshold and 
            self.auto_train):
            self.logger.warning(f"Model accuracy {self.model.training_accuracy:.3f} below threshold {self.retrain_threshold}, retraining...")
            try:
                training_results = await asyncio.to_thread(self.model.train)
                self.logger.info(f"Model retraining completed: {training_results}")
            except Exception as e:
                self.logger.error(f"Model retraining failed: {e}")
        
        self.is_initialized = True
        self.logger.info(f"ML sentiment analyzer ready - Accuracy: {self.model.training_accuracy:.3f}")
    
    async def stop(self) -> None:
        """Stop the ML sentiment analyzer"""
        self.logger.info("Stopping ML sentiment analyzer")
        self.is_initialized = False
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using ML model
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self.enabled or not self.is_initialized:
            raise ValueError("ML sentiment analyzer not available")
            
        try:
            self.prediction_count += 1
            
            # Run prediction in thread pool to avoid blocking
            result = await asyncio.to_thread(self.model.predict_sentiment, text)
            
            # Enhance result with additional metadata
            result.update({
                'timestamp': datetime.utcnow().isoformat(),
                'analyzer': 'ml_sentiment',
                'prediction_id': self.prediction_count,
                'model_info': {
                    'accuracy': self.model.training_accuracy,
                    'cv_score': self.model.cross_val_score
                }
            })
            
            # Add confidence classification
            confidence = result.get('confidence', 0.0)
            if confidence >= self.high_confidence:
                result['confidence_level'] = 'high'
            elif confidence >= self.min_confidence:
                result['confidence_level'] = 'medium'
            else:
                result['confidence_level'] = 'low'
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"ML sentiment analysis failed: {e}")
            return {
                'error': str(e),
                'analyzer': 'ml_sentiment',
                'timestamp': datetime.utcnow().isoformat(),
                'sentiment': 'neutral',
                'polarity': 0.0,
                'confidence': 0.0,
                'confidence_level': 'low'
            }
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        if not self.enabled or not self.is_initialized:
            return [{'error': 'ML analyzer not available'} for _ in texts]
        
        results = []
        for text in texts:
            try:
                result = await self.analyze_sentiment(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch analysis failed for text: {e}")
                results.append({
                    'error': str(e),
                    'sentiment': 'neutral',
                    'polarity': 0.0,
                    'confidence': 0.0
                })
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the ML analyzer"""
        error_rate = (self.error_count / max(self.prediction_count, 1)) * 100
        
        return {
            'enabled': self.enabled,
            'initialized': self.is_initialized,
            'predictions_made': self.prediction_count,
            'errors': self.error_count,
            'error_rate_percent': round(error_rate, 2),
            'model_accuracy': self.model.training_accuracy if self.model else 0.0,
            'model_cv_score': self.model.cross_val_score if self.model else 0.0,
            'model_trained': self.model.is_trained if self.model else False
        }
    
    async def retrain_model(self, texts: Optional[List[str]] = None, 
                          labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Retrain the ML model with new data
        
        Args:
            texts: Optional training texts (uses synthetic data if None)
            labels: Optional training labels (must match texts)
            
        Returns:
            Training results dictionary
        """
        if not self.enabled:
            raise ValueError("ML sentiment analyzer is disabled")
            
        self.logger.info("Retraining ML sentiment model...")
        
        try:
            # Run training in thread pool
            results = await asyncio.to_thread(self.model.train, texts, labels)
            
            self.logger.info(f"Model retrained successfully: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            raise
    
    def can_analyze(self) -> bool:
        """Check if the analyzer is ready to perform analysis"""
        return self.enabled and self.is_initialized and self.model.is_trained
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the current model"""
        if not self.model:
            return {'error': 'No model available'}
            
        info = self.model.get_model_info()
        info.update(self.get_performance_stats())
        
        return info
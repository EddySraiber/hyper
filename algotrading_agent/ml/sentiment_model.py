import re
import pickle
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import os


class FinancialSentimentModel:
    """
    Advanced ML-based sentiment analyzer for financial news.
    Uses ensemble approach with feature engineering for financial context.
    """
    
    def __init__(self, model_dir: str = "/app/data/ml_models"):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.pipeline = None
        self.is_trained = False
        self.feature_names = []
        
        # Financial keywords for feature engineering
        self.financial_keywords = self._load_financial_keywords()
        
        # Model performance metrics
        self.training_accuracy = 0.0
        self.cross_val_score = 0.0
        
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive financial keywords for feature engineering"""
        return {
            'bullish': [
                'beat', 'beats', 'beating', 'exceeded', 'exceeds', 'outperform',
                'surge', 'surged', 'surging', 'soar', 'soared', 'rally', 'rallied',
                'gain', 'gains', 'gained', 'rise', 'rose', 'rising', 'boost', 'boosted',
                'strong', 'robust', 'solid', 'growth', 'expanding', 'expansion',
                'record', 'breakthrough', 'milestone', 'success', 'successful',
                'profit', 'profitable', 'revenue', 'earnings', 'upgrade', 'upgraded',
                'buy', 'bullish', 'positive', 'optimistic', 'confidence', 'confident',
                'dividend', 'acquisition', 'merger', 'partnership', 'deal', 'agreement'
            ],
            'bearish': [
                'miss', 'missed', 'missing', 'below', 'underperform', 'disappointing',
                'drop', 'dropped', 'fall', 'fell', 'falling', 'decline', 'declined',
                'plunge', 'plunged', 'crash', 'crashed', 'tumble', 'tumbled',
                'loss', 'losses', 'losing', 'lost', 'weak', 'weaken', 'weakness',
                'poor', 'concerning', 'concern', 'worried', 'worry', 'risk', 'risky',
                'trouble', 'troubled', 'problem', 'problems', 'issue', 'issues',
                'downgrade', 'downgraded', 'cut', 'reduce', 'reduced', 'lawsuit',
                'sell', 'bearish', 'negative', 'pessimistic', 'uncertainty', 'uncertain',
                'bankruptcy', 'debt', 'default', 'investigation', 'scandal', 'fraud'
            ],
            'volatility': [
                'volatile', 'volatility', 'swing', 'swings', 'fluctuation', 'fluctuate',
                'uncertain', 'uncertainty', 'speculation', 'rumor', 'rumors', 'alert',
                'warning', 'caution', 'watch', 'monitor', 'tracking', 'pressure'
            ],
            'impact': [
                'massive', 'huge', 'significant', 'major', 'substantial', 'dramatic',
                'unprecedented', 'historic', 'record-breaking', 'extraordinary',
                'breakthrough', 'game-changing', 'disruptive', 'revolutionary'
            ]
        }
    
    def extract_features(self, text: str) -> np.ndarray:
        """Extract comprehensive features from financial text"""
        text_lower = text.lower()
        features = []
        
        # 1. Financial keyword counts (normalized by text length)
        text_length = max(len(text_lower.split()), 1)
        
        for category, keywords in self.financial_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features.append(count / text_length)  # Normalized count
        
        # 2. Sentiment intensity features
        bullish_score = sum(1 for word in self.financial_keywords['bullish'] if word in text_lower)
        bearish_score = sum(1 for word in self.financial_keywords['bearish'] if word in text_lower)
        
        # Net sentiment
        features.append((bullish_score - bearish_score) / text_length)
        
        # Sentiment strength
        features.append((bullish_score + bearish_score) / text_length)
        
        # 3. Text structure features
        features.append(len(text_lower.split()))  # Word count
        features.append(len([s for s in text.split('.') if s.strip()]))  # Sentence count
        features.append(text_lower.count('!'))  # Exclamation marks
        features.append(text_lower.count('?'))  # Question marks
        
        # 4. Financial entity indicators
        ticker_pattern = r'\$[A-Z]{1,5}\b'
        features.append(len(re.findall(ticker_pattern, text)))  # Ticker mentions
        
        # 5. Number and percentage mentions
        number_pattern = r'\b\d+\.?\d*%?\b'
        features.append(len(re.findall(number_pattern, text_lower)))
        
        # 6. Time urgency indicators
        urgency_words = ['breaking', 'urgent', 'immediate', 'now', 'today', 'just', 'alert']
        features.append(sum(1 for word in urgency_words if word in text_lower) / text_length)
        
        return np.array(features, dtype=np.float32)
    
    def create_training_pipeline(self) -> Pipeline:
        """Create the ML pipeline for sentiment classification"""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )),
            ('scaler', StandardScaler(with_mean=False)),  # TF-IDF is sparse
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',  # Handle imbalanced data
                n_jobs=-1
            ))
        ])
        return pipeline
    
    def generate_synthetic_training_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic training data based on financial patterns"""
        self.logger.info("Generating synthetic training data for financial sentiment")
        
        texts = []
        labels = []  # 0: negative, 1: neutral, 2: positive
        
        # Positive examples
        positive_templates = [
            "{company} beats earnings expectations by {percent}%",
            "{company} reports strong Q{quarter} results with revenue growth of {percent}%",
            "{company} announces breakthrough {product} launch",
            "{company} stock surges {percent}% on positive outlook",
            "{company} upgrades guidance, shares rally {percent}%",
            "{company} secures major {deal_type} worth ${amount}M",
            "Analysts upgrade {company} to buy with {percent}% upside target",
            "{company} dividend increase signals strong cash flow",
            "{company} merger creates value, stock jumps {percent}%",
            "Record quarterly profits drive {company} shares higher"
        ]
        
        # Negative examples
        negative_templates = [
            "{company} misses earnings estimates, stock falls {percent}%",
            "{company} reports disappointing Q{quarter} results",
            "{company} faces regulatory investigation over {issue}",
            "{company} stock plunges {percent}% on weak guidance",
            "Analysts downgrade {company} citing {concern}",
            "{company} lawsuit threatens ${amount}M in damages",
            "{company} debt concerns weigh on share price",
            "Supply chain issues hurt {company} profitability",
            "{company} CEO departure creates uncertainty",
            "Market volatility pressures {company} valuation"
        ]
        
        # Neutral examples
        neutral_templates = [
            "{company} maintains steady performance in Q{quarter}",
            "{company} announces routine board meeting",
            "{company} stock trading sideways amid market conditions",
            "{company} reports in-line quarterly results",
            "No significant changes in {company} operations",
            "{company} scheduled to report earnings next week",
            "Market analysts await {company} guidance update",
            "{company} continues standard business operations",
            "Industry trends may impact {company} performance",
            "{company} maintains current dividend policy"
        ]
        
        companies = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA", "JPM", "V", "PG"]
        
        # Generate positive examples (label = 2)
        for template in positive_templates:
            for company in companies:
                text = template.format(
                    company=company,
                    percent=np.random.randint(5, 25),
                    quarter=np.random.randint(1, 4),
                    product=np.random.choice(["AI chip", "software", "service", "platform"]),
                    deal_type=np.random.choice(["contract", "partnership", "acquisition"]),
                    amount=np.random.randint(100, 2000)
                )
                texts.append(text)
                labels.append(2)
        
        # Generate negative examples (label = 0)
        for template in negative_templates:
            for company in companies:
                text = template.format(
                    company=company,
                    percent=np.random.randint(5, 20),
                    quarter=np.random.randint(1, 4),
                    issue=np.random.choice(["privacy", "antitrust", "safety", "compliance"]),
                    concern=np.random.choice(["competition", "regulation", "costs", "demand"]),
                    amount=np.random.randint(50, 500)
                )
                texts.append(text)
                labels.append(0)
        
        # Generate neutral examples (label = 1)
        for template in neutral_templates:
            for company in companies:
                text = template.format(
                    company=company,
                    quarter=np.random.randint(1, 4)
                )
                texts.append(text)
                labels.append(1)
        
        self.logger.info(f"Generated {len(texts)} training examples: "
                        f"{labels.count(2)} positive, {labels.count(1)} neutral, {labels.count(0)} negative")
        
        return texts, labels
    
    def train(self, texts: Optional[List[str]] = None, labels: Optional[List[int]] = None) -> Dict[str, float]:
        """Train the sentiment model"""
        if texts is None or labels is None:
            texts, labels = self.generate_synthetic_training_data()
        
        self.logger.info(f"Training ML sentiment model on {len(texts)} samples")
        
        # Create and train pipeline
        self.pipeline = self.create_training_pipeline()
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = self.pipeline.predict(X_test)
        self.training_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=5, scoring='accuracy')
        self.cross_val_score = cv_scores.mean()
        
        self.is_trained = True
        
        # Log detailed results
        self.logger.info(f"Training completed - Accuracy: {self.training_accuracy:.3f}, "
                        f"CV Score: {self.cross_val_score:.3f}")
        self.logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save the model
        self.save_model()
        
        return {
            'training_accuracy': self.training_accuracy,
            'cross_val_score': self.cross_val_score,
            'training_samples': len(texts)
        }
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Get prediction and probabilities
            prediction = self.pipeline.predict([text])[0]
            probabilities = self.pipeline.predict_proba([text])[0]
            
            # Map prediction to sentiment
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_label = sentiment_map[prediction]
            
            # Calculate confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Calculate polarity score (-1 to 1)
            # Weight by probabilities: negative=-1, neutral=0, positive=1
            polarity = float(probabilities[0] * (-1) + probabilities[1] * 0 + probabilities[2] * 1)
            
            # Extract additional features for context
            features = self.extract_features(text)
            
            return {
                'sentiment': sentiment_label,
                'polarity': polarity,
                'confidence': confidence,
                'probabilities': {
                    'negative': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'positive': float(probabilities[2])
                },
                'method': 'ml_ensemble',
                'model_accuracy': self.training_accuracy
            }
            
        except Exception as e:
            self.logger.error(f"ML sentiment prediction failed: {e}")
            return {
                'sentiment': 'neutral',
                'polarity': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'method': 'ml_ensemble_failed'
            }
    
    def save_model(self) -> bool:
        """Save the trained model to disk"""
        try:
            model_path = self.model_dir / "sentiment_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'is_trained': self.is_trained,
                    'training_accuracy': self.training_accuracy,
                    'cross_val_score': self.cross_val_score,
                    'financial_keywords': self.financial_keywords
                }, f)
            
            self.logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load a pre-trained model from disk"""
        try:
            model_path = self.model_dir / "sentiment_model.pkl"
            if not model_path.exists():
                self.logger.warning(f"No saved model found at {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.is_trained = model_data['is_trained']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.cross_val_score = model_data.get('cross_val_score', 0.0)
            self.financial_keywords = model_data.get('financial_keywords', self.financial_keywords)
            
            self.logger.info(f"Model loaded from {model_path} - Accuracy: {self.training_accuracy:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'cross_val_score': self.cross_val_score,
            'model_type': 'RandomForest + TF-IDF',
            'features': ['financial_keywords', 'tfidf_features', 'structural_features'],
            'classes': ['negative', 'neutral', 'positive']
        }
import re
from typing import List, Dict, Any, Optional
from textblob import TextBlob
from datetime import datetime
from ..core.base import ComponentBase


class NewsAnalysisBrain(ComponentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("news_analysis_brain", config)
        self.sentiment_threshold = config.get("sentiment_threshold", 0.1)
        self.entity_patterns = config.get("entity_patterns", {})
        self.impact_keywords = config.get("impact_keywords", {})
        
    def start(self) -> None:
        self.logger.info("Starting News Analysis Brain")
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping News Analysis Brain")
        self.is_running = False
        
    def process(self, filtered_news: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.is_running or not filtered_news:
            return []
            
        analyzed_items = []
        for item in filtered_news:
            try:
                analysis = self._analyze_item(item)
                item.update(analysis)
                analyzed_items.append(item)
            except Exception as e:
                self.logger.error(f"Error analyzing item: {e}")
                
        self.logger.info(f"Analyzed {len(analyzed_items)} news items")
        return analyzed_items
        
    def _analyze_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        text = f"{item.get('title', '')} {item.get('content', '')}"
        
        analysis = {
            "sentiment": self._analyze_sentiment(text),
            "entities": self._extract_entities(text),
            "events": self._classify_events(text),
            "impact_score": self._calculate_impact(text),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return analysis
        
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > self.sentiment_threshold:
                sentiment_label = "positive"
            elif polarity < -self.sentiment_threshold:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
                
            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "label": sentiment_label,
                "confidence": abs(polarity)
            }
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "neutral",
                "confidence": 0.0
            }
            
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {
            "companies": [],
            "people": [],
            "sectors": [],
            "tickers": []
        }
        
        try:
            # Extract stock tickers (e.g., $AAPL, TSLA)
            ticker_pattern = r'\$?([A-Z]{1,5})\b'
            tickers = re.findall(ticker_pattern, text.upper())
            entities["tickers"] = list(set(tickers))
            
            # Extract entities using predefined patterns
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    entities[entity_type].extend(matches)
                    
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
                
        except Exception as e:
            self.logger.error(f"Entity extraction error: {e}")
            
        return entities
        
    def _classify_events(self, text: str) -> List[str]:
        events = []
        text_lower = text.lower()
        
        event_keywords = {
            "earnings": ["earnings", "quarterly", "eps", "revenue", "profit"],
            "merger": ["merger", "acquisition", "buyout", "takeover"],
            "regulatory": ["fda", "approval", "regulation", "compliance"],
            "partnership": ["partnership", "collaboration", "deal", "agreement"],
            "product_launch": ["launch", "release", "unveil", "introduce"],
            "lawsuit": ["lawsuit", "litigation", "legal", "court"],
            "bankruptcy": ["bankruptcy", "chapter 11", "insolvent"],
            "dividend": ["dividend", "payout", "distribution"]
        }
        
        for event_type, keywords in event_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                events.append(event_type)
                
        return events
        
    def _calculate_impact(self, text: str) -> float:
        impact_score = 0.0
        text_lower = text.lower()
        
        # High impact keywords
        high_impact = ["breakthrough", "record", "massive", "huge", "unprecedented"]
        medium_impact = ["significant", "major", "important", "substantial"]
        low_impact = ["minor", "small", "slight", "modest"]
        
        for keyword in high_impact:
            if keyword in text_lower:
                impact_score += 0.3
                
        for keyword in medium_impact:
            if keyword in text_lower:
                impact_score += 0.2
                
        for keyword in low_impact:
            if keyword in text_lower:
                impact_score += 0.1
                
        # Custom impact keywords from config
        for keyword, weight in self.impact_keywords.items():
            if keyword.lower() in text_lower:
                impact_score += weight
                
        return min(impact_score, 1.0)
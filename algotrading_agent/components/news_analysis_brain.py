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
            base_polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Enhance with financial sentiment keywords
            enhanced_polarity = self._enhance_financial_sentiment(text, base_polarity)
            
            # Classify sentiment
            if enhanced_polarity > self.sentiment_threshold:
                sentiment_label = "positive"
            elif enhanced_polarity < -self.sentiment_threshold:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
                
            return {
                "polarity": enhanced_polarity,
                "subjectivity": subjectivity,
                "label": sentiment_label,
                "confidence": abs(enhanced_polarity)
            }
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "neutral",
                "confidence": 0.0
            }
            
    def _enhance_financial_sentiment(self, text: str, base_polarity: float) -> float:
        """Enhance sentiment with financial-specific keywords"""
        text_lower = text.lower()
        
        # Positive financial indicators
        positive_keywords = {
            'beat': 0.3, 'beats': 0.3, 'beating': 0.3, 'exceeded': 0.4, 'exceeds': 0.4,
            'surge': 0.4, 'surged': 0.4, 'surging': 0.4, 'soar': 0.4, 'soared': 0.4,
            'rally': 0.3, 'rallied': 0.3, 'gain': 0.2, 'gains': 0.2, 'gained': 0.2,
            'up': 0.15, 'rise': 0.2, 'rose': 0.2, 'rising': 0.2, 'boost': 0.3,
            'strong': 0.2, 'robust': 0.3, 'solid': 0.2, 'growth': 0.2,
            'record': 0.4, 'breakthrough': 0.4, 'milestone': 0.3, 'success': 0.2,
            'profit': 0.2, 'revenue': 0.15, 'earnings': 0.1, 'upgrade': 0.3
        }
        
        # Negative financial indicators  
        negative_keywords = {
            'miss': -0.3, 'missed': -0.3, 'missing': -0.3, 'below': -0.2,
            'drop': -0.3, 'dropped': -0.3, 'fall': -0.2, 'fell': -0.2, 'falling': -0.2,
            'plunge': -0.4, 'plunged': -0.4, 'crash': -0.5, 'crashed': -0.5,
            'decline': -0.3, 'declined': -0.3, 'down': -0.15, 'loss': -0.3,
            'weak': -0.2, 'poor': -0.3, 'disappointing': -0.3, 'concern': -0.2,
            'risk': -0.15, 'trouble': -0.3, 'problem': -0.2, 'issue': -0.15,
            'downgrade': -0.3, 'cut': -0.2, 'reduce': -0.2, 'lawsuit': -0.3
        }
        
        # Calculate sentiment boost/penalty
        sentiment_adjustment = 0.0
        
        for word, weight in positive_keywords.items():
            if word in text_lower:
                sentiment_adjustment += weight
                
        for word, weight in negative_keywords.items():
            if word in text_lower:
                sentiment_adjustment += weight  # weight is already negative
                
        # Combine base sentiment with financial keywords (with dampening)
        enhanced_polarity = base_polarity + (sentiment_adjustment * 0.5)
        
        # Keep within bounds
        return max(-1.0, min(1.0, enhanced_polarity))
            
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {
            "companies": [],
            "people": [],
            "sectors": [],
            "tickers": []
        }
        
        try:
            text_lower = text.lower()
            
            # Company name to ticker mapping - most comprehensive
            company_ticker_map = {
                "apple": "AAPL", "apple inc": "AAPL", "microsoft": "MSFT", "microsoft corp": "MSFT",
                "tesla": "TSLA", "tesla inc": "TSLA", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
                "meta": "META", "facebook": "META", "nvidia": "NVDA", "boeing": "BA", "coca-cola": "KO",
                "pepsi": "PEP", "walmart": "WMT", "jpmorgan": "JPM", "bank of america": "BAC",
                "wells fargo": "WFC", "goldman sachs": "GS", "morgan stanley": "MS", "visa": "V",
                "mastercard": "MA", "intel": "INTC", "amd": "AMD", "netflix": "NFLX", "disney": "DIS",
                "salesforce": "CRM", "oracle": "ORCL", "ibm": "IBM", "ge": "GE", "general electric": "GE",
                "ford": "F", "general motors": "GM", "exxon": "XOM", "chevron": "CVX", "pfizer": "PFE",
                "johnson": "JNJ", "merck": "MRK", "abbvie": "ABBV", "bristol myers": "BMY"
            }
            
            # Only extract explicitly mentioned stock tickers with $ prefix
            dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
            entities["tickers"].extend(dollar_tickers)
            
            # Extract company names and map to tickers
            for company_name, ticker in company_ticker_map.items():
                if company_name in text_lower:
                    entities["tickers"].append(ticker)
                    entities["companies"].append(company_name)
            
            # Add specific tickers for economic news only if exact phrases match
            if "jobless claims" in text_lower or "unemployment" in text_lower:
                entities["tickers"].append("SPY")
            if "federal reserve" in text_lower or " fed " in text_lower:
                entities["tickers"].append("SPY")
            if "inflation" in text_lower:
                entities["tickers"].append("SPY")
                    
            # Remove duplicates
            entities["tickers"] = list(set(entities["tickers"]))
            entities["companies"] = list(set(entities["companies"]))
                
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
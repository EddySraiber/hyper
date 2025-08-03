import re
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ..core.base import ComponentBase


class NewsFilter(ComponentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("news_filter", config)
        self.keywords = config.get("keywords", [])
        self.blacklist = config.get("blacklist", [])
        self.source_weights = config.get("source_weights", {})
        self.recency_weight = config.get("recency_weight", 0.5)
        self.min_score = config.get("min_score", 0.3)
        
    def start(self) -> None:
        self.logger.info("Starting News Filter")
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping News Filter")
        self.is_running = False
        
    def process(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.is_running or not news_items:
            return []
            
        filtered_items = []
        for item in news_items:
            try:
                score = self._calculate_score(item)
                if score >= self.min_score:
                    item["filter_score"] = score
                    filtered_items.append(item)
            except Exception as e:
                self.logger.error(f"Error filtering item: {e}")
                
        # Sort by score (highest first)
        filtered_items.sort(key=lambda x: x["filter_score"], reverse=True)
        
        self.logger.info(f"Filtered {len(news_items)} items to {len(filtered_items)}")
        return filtered_items
        
    def _calculate_score(self, item: Dict[str, Any]) -> float:
        score = 0.0
        
        # Keyword relevance
        relevance_score = self._calculate_relevance(item)
        score += relevance_score * 0.4
        
        # Source credibility
        source_score = self._calculate_source_score(item)
        score += source_score * 0.3
        
        # Recency score
        recency_score = self._calculate_recency_score(item)
        score += recency_score * 0.3
        
        # Apply blacklist penalty
        if self._is_blacklisted(item):
            score *= 0.1
            
        return min(score, 1.0)
        
    def _calculate_relevance(self, item: Dict[str, Any]) -> float:
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        if not self.keywords:
            return 0.5  # Neutral score if no keywords defined
            
        matches = 0
        for keyword in self.keywords:
            if isinstance(keyword, dict):
                pattern = keyword.get("pattern", "")
                weight = keyword.get("weight", 1.0)
                if re.search(pattern.lower(), text):
                    matches += weight
            else:
                if keyword.lower() in text:
                    matches += 1
                    
        return min(matches / len(self.keywords), 1.0)
        
    def _calculate_source_score(self, item: Dict[str, Any]) -> float:
        source = item.get("source", "unknown")
        return self.source_weights.get(source, 0.5)
        
    def _calculate_recency_score(self, item: Dict[str, Any]) -> float:
        published = item.get("published")
        if not published:
            return 0.0
            
        if isinstance(published, str):
            try:
                published = datetime.fromisoformat(published.replace('Z', '+00:00'))
            except:
                return 0.0
                
        age_hours = (datetime.utcnow() - published.replace(tzinfo=None)).total_seconds() / 3600
        
        # Score decreases with age (24 hours = 0.5, 48 hours = 0.25, etc.)
        if age_hours <= 1:
            return 1.0
        elif age_hours <= 24:
            return 0.8
        elif age_hours <= 48:
            return 0.5
        else:
            return 0.2
            
    def _is_blacklisted(self, item: Dict[str, Any]) -> bool:
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        return any(blacklisted.lower() in text for blacklisted in self.blacklist)
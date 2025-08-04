from typing import Dict, List, Any, Optional
import re
import math
from datetime import datetime, timedelta
from ..core.base import ComponentBase


class NewsImpactScorer(ComponentBase):
    """
    Advanced news impact scoring system to measure market-moving potential
    
    Scores news based on:
    1. Source Authority (Reuters > Bloomberg > CNBC > etc.)
    2. Market Exposure (company size, sector impact)
    3. Content Intensity (keywords, numbers, superlatives)
    4. Timing & Freshness (market hours, breaking vs stale)
    5. Hype Amplification (buzz words, social media indicators)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("news_impact_scorer", config)
        
        # Source authority rankings (higher = more market impact)
        self.source_authority = {
            # Tier 1: Market Moving Sources
            'Reuters': 1.0,
            'Bloomberg': 1.0,
            'Wall Street Journal': 0.95,
            'Financial Times': 0.95,
            
            # Tier 2: Mainstream Finance
            'CNBC': 0.85,
            'MarketWatch': 0.80,
            'Yahoo Finance': 0.75,
            'Seeking Alpha': 0.70,
            
            # Tier 3: General News
            'CNN Business': 0.65,
            'AP News': 0.60,
            'Business Insider': 0.55,
            
            # Default for unknown sources
            'Unknown': 0.30
        }
        
        # Market exposure factors (company/sector importance)
        self.market_exposure = {
            # Mega Cap (>$1T market cap)
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
            'mega_cap_weight': 1.0,
            
            # Large Cap ($10B-$1T)  
            'large_cap': ['TSLA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'BAC', 'XOM'],
            'large_cap_weight': 0.8,
            
            # Market-moving sectors
            'high_impact_sectors': ['AI', 'artificial intelligence', 'electric vehicle', 'crypto', 
                                   'federal reserve', 'inflation', 'interest rates', 'GDP'],
            'sector_weight': 0.6
        }
        
        # Content intensity keywords (higher weight = more market impact)
        self.intensity_keywords = {
            # Extreme Impact (3.0x multiplier)
            'extreme': {
                'patterns': ['breakthrough', 'revolutionary', 'game-changer', 'historic', 'unprecedented'],
                'weight': 3.0
            },
            
            # High Impact (2.5x multiplier) 
            'high': {
                'patterns': ['beats expectations', 'smashes estimates', 'crushes forecasts', 
                           'misses badly', 'plunges', 'soars', 'surges', 'collapses'],
                'weight': 2.5
            },
            
            # Medium Impact (2.0x multiplier)
            'medium': {
                'patterns': ['beats estimates', 'exceeds expectations', 'disappoints', 
                           'drops sharply', 'rises strongly', 'major deal', 'acquisition'],
                'weight': 2.0
            },
            
            # Quantified Impact (numbers make news more credible)
            'numbers': {
                'patterns': [r'\d+%', r'\$\d+[MB]', r'\d+\.\d+%', 'billion', 'million', 'trillion'],
                'weight': 1.5
            }
        }
        
        # Timing factors
        self.timing_factors = {
            'market_hours': 1.3,      # News during trading hours
            'pre_market': 1.2,        # Pre-market news (7-9:30 AM ET)
            'after_hours': 1.1,       # After hours (4-8 PM ET)
            'weekend': 0.8,           # Weekend news (less immediate impact)
            'breaking_fresh': 1.5,    # Fresh breaking news (<1 hour)
            'recent': 1.2,            # Recent news (<4 hours)
            'stale': 0.6              # Old news (>24 hours)
        }
        
        # Hype amplification indicators
        self.hype_indicators = {
            'viral_words': ['trending', 'viral', 'exploding', 'rocket', 'moon', 'massive surge'],
            'urgency_words': ['urgent', 'breaking', 'alert', 'flash', 'just in'],
            'social_buzz': ['twitter', 'reddit', 'social media', 'viral', 'trending'],
            'hype_weight': 1.3
        }
        
    def start(self) -> None:
        self.logger.info("Starting News Impact Scorer")
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping News Impact Scorer") 
        self.is_running = False
        
    def process(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score news items for market impact potential"""
        if not self.is_running or not news_items:
            return news_items
            
        scored_items = []
        for item in news_items:
            try:
                impact_score = self._calculate_impact_score(item)
                item['impact_score'] = impact_score
                item['impact_grade'] = self._get_impact_grade(impact_score)
                scored_items.append(item)
            except Exception as e:
                self.logger.error(f"Error scoring news item: {e}")
                item['impact_score'] = 0.1  # Default low score
                item['impact_grade'] = 'D'
                scored_items.append(item)
        
        # Sort by impact score (highest first)
        scored_items.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
        
        # Log high-impact news
        high_impact = [item for item in scored_items if item.get('impact_score', 0) > 0.8]
        if high_impact:
            self.logger.warning(f"🔥 HIGH IMPACT NEWS: {len(high_impact)} items with score >0.8")
            for item in high_impact[:3]:
                self.logger.info(f"  📈 {item.get('impact_grade', 'N/A')}: {item.get('title', 'Unknown')[:60]}")
        
        return scored_items
        
    def _calculate_impact_score(self, item: Dict[str, Any]) -> float:
        """Calculate comprehensive impact score (0.0 to 1.0+)"""
        
        # 1. Source Authority (30% weight)
        source_score = self._calculate_source_score(item) * 0.30
        
        # 2. Market Exposure (25% weight)
        exposure_score = self._calculate_exposure_score(item) * 0.25
        
        # 3. Content Intensity (20% weight)
        intensity_score = self._calculate_intensity_score(item) * 0.20
        
        # 4. Timing & Freshness (15% weight)
        timing_score = self._calculate_timing_score(item) * 0.15
        
        # 5. Hype Amplification (10% weight)
        hype_score = self._calculate_hype_score(item) * 0.10
        
        # Combine all factors
        total_score = source_score + exposure_score + intensity_score + timing_score + hype_score
        
        # Cap at reasonable maximum (can exceed 1.0 for exceptional news)
        return min(total_score, 2.0)
    
    def _calculate_source_score(self, item: Dict[str, Any]) -> float:
        """Score based on source credibility and authority"""
        source = item.get('source', 'Unknown')
        
        # Exact match
        if source in self.source_authority:
            return self.source_authority[source]
            
        # Partial match (e.g., "Reuters Business" matches "Reuters")
        for auth_source, weight in self.source_authority.items():
            if auth_source.lower() in source.lower():
                return weight
                
        return self.source_authority['Unknown']
    
    def _calculate_exposure_score(self, item: Dict[str, Any]) -> float:
        """Score based on market exposure and company/sector importance"""
        content = f"{item.get('title', '')} {item.get('content', '')}".upper()
        max_exposure = 0.0
        
        # Check for mega-cap mentions
        for symbol in self.market_exposure['mega_cap']:
            if symbol in content or self._company_name_in_content(symbol, content):
                max_exposure = max(max_exposure, self.market_exposure['mega_cap_weight'])
        
        # Check for large-cap mentions  
        for symbol in self.market_exposure['large_cap']:
            if symbol in content or self._company_name_in_content(symbol, content):
                max_exposure = max(max_exposure, self.market_exposure['large_cap_weight'])
                
        # Check for high-impact sectors
        for sector in self.market_exposure['high_impact_sectors']:
            if sector.upper() in content:
                max_exposure = max(max_exposure, self.market_exposure['sector_weight'])
        
        return max_exposure
    
    def _company_name_in_content(self, symbol: str, content: str) -> bool:
        """Check if company name appears in content (not just ticker)"""
        company_names = {
            'AAPL': 'APPLE',
            'MSFT': 'MICROSOFT', 
            'GOOGL': 'GOOGLE',
            'AMZN': 'AMAZON',
            'TSLA': 'TESLA',
            'META': 'META',
            'NVDA': 'NVIDIA'
        }
        
        if symbol in company_names:
            return company_names[symbol] in content
        return False
        
    def _calculate_intensity_score(self, item: Dict[str, Any]) -> float:
        """Score based on content intensity and impact keywords"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        max_intensity = 0.3  # Base score
        
        for category, config in self.intensity_keywords.items():
            for pattern in config['patterns']:
                if category == 'numbers':
                    # Use regex for number patterns
                    if re.search(pattern, content):
                        max_intensity = max(max_intensity, config['weight'] * 0.3)
                else:
                    # Simple string matching for text patterns
                    if pattern in content:
                        max_intensity = max(max_intensity, config['weight'] * 0.3)
        
        return min(max_intensity, 1.0)
    
    def _calculate_timing_score(self, item: Dict[str, Any]) -> float:
        """Score based on timing and freshness"""
        published_obj = item.get('published') or item.get('timestamp', '')
        if not published_obj:
            return 0.5  # Neutral if no timestamp
            
        try:
            # Handle both datetime objects and string timestamps
            if isinstance(published_obj, datetime):
                published = published_obj
            elif isinstance(published_obj, str):
                # Parse timestamp string
                if 'T' in published_obj:
                    published = datetime.fromisoformat(published_obj.replace('Z', '+00:00'))
                else:
                    published = datetime.fromisoformat(published_obj)
            else:
                return 0.5  # Unknown format
                
            now = datetime.utcnow()
            age_hours = (now - published.replace(tzinfo=None)).total_seconds() / 3600
            
            # Freshness factor
            if age_hours < 1:
                freshness = self.timing_factors['breaking_fresh']
            elif age_hours < 4:
                freshness = self.timing_factors['recent']  
            elif age_hours > 24:
                freshness = self.timing_factors['stale']
            else:
                freshness = 1.0
                
            # Market hours factor (assuming ET timezone)
            hour = published.hour
            if 9 <= hour <= 16:  # Market hours (9:30 AM - 4 PM ET)
                timing = self.timing_factors['market_hours']
            elif 7 <= hour <= 9:  # Pre-market
                timing = self.timing_factors['pre_market']
            elif 16 <= hour <= 20:  # After hours
                timing = self.timing_factors['after_hours']
            else:
                timing = 1.0
                
            return min(freshness * timing, 2.0)
            
        except Exception as e:
            self.logger.warning(f"Error parsing timestamp {published_obj}: {e}")
            return 0.5
    
    def _calculate_hype_score(self, item: Dict[str, Any]) -> float:
        """Score based on hype and viral potential"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        hype_multiplier = 1.0
        
        # Check for viral indicators
        viral_count = sum(1 for word in self.hype_indicators['viral_words'] if word in content)
        urgency_count = sum(1 for word in self.hype_indicators['urgency_words'] if word in content)  
        social_count = sum(1 for word in self.hype_indicators['social_buzz'] if word in content)
        
        total_hype_signals = viral_count + urgency_count + social_count
        
        if total_hype_signals > 0:
            hype_multiplier = min(1.0 + (total_hype_signals * 0.2), self.hype_indicators['hype_weight'])
            
        return hype_multiplier
    
    def _get_impact_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 1.5:
            return 'A+'  # Exceptional market impact
        elif score >= 1.2:
            return 'A'   # High market impact
        elif score >= 1.0:
            return 'B+'  # Above average impact
        elif score >= 0.8:
            return 'B'   # Good market impact
        elif score >= 0.6:
            return 'C'   # Moderate impact
        elif score >= 0.4:
            return 'D'   # Low impact
        else:
            return 'F'   # Minimal impact
            
    def get_impact_summary(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of news impact distribution"""
        if not news_items:
            return {}
            
        scores = [item.get('impact_score', 0) for item in news_items]
        grades = [item.get('impact_grade', 'F') for item in news_items]
        
        return {
            'total_items': len(news_items),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'high_impact_count': len([s for s in scores if s >= 0.8]),
            'grade_distribution': {grade: grades.count(grade) for grade in set(grades)},
            'top_stories': [
                {
                    'title': item.get('title', 'Unknown')[:60],
                    'score': item.get('impact_score', 0),
                    'grade': item.get('impact_grade', 'F')
                }
                for item in sorted(news_items, key=lambda x: x.get('impact_score', 0), reverse=True)[:5]
            ]
        }
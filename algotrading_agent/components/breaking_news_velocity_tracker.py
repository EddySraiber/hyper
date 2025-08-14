"""
Breaking News Velocity Tracker - Real-time hype detection system

Monitors news story spread velocity, social media acceleration, and breaking news
indicators to identify rapid market-moving events that require immediate action.
Designed to detect "hype" patterns for momentum trading opportunities.
"""

import asyncio
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
from ..core.base import ComponentBase


class VelocityLevel(Enum):
    """News velocity levels for trading priority"""
    VIRAL = "viral"           # Extreme velocity - immediate action
    BREAKING = "breaking"     # High velocity - express action
    TRENDING = "trending"     # Medium velocity - fast action
    NORMAL = "normal"         # Standard velocity - normal processing


@dataclass
class NewsVelocitySignal:
    """Breaking news velocity signal for fast trading"""
    story_id: str
    title: str
    content: str
    symbols: List[str]
    
    # Velocity metrics
    velocity_level: VelocityLevel
    velocity_score: float      # 0.0 to 10.0
    spread_rate: float        # mentions per minute
    acceleration: float       # change in spread rate
    
    # Social metrics
    social_mentions: int
    social_velocity: float    # mentions per minute across platforms
    sentiment_velocity: float # sentiment change rate
    
    # Financial relevance
    financial_impact_score: float  # 0.0 to 1.0
    market_relevance: float       # 0.0 to 1.0
    urgency_score: float         # 0.0 to 1.0
    
    # Timing
    first_seen: datetime
    peak_velocity: datetime
    detected_at: datetime
    expires_at: datetime
    
    # Sources
    sources: List[str]
    source_diversity: float   # How many different sources
    
    # Breaking indicators
    breaking_keywords: List[str]
    hype_indicators: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "symbols": self.symbols,
            "velocity_level": self.velocity_level.value,
            "velocity_score": self.velocity_score,
            "spread_rate": self.spread_rate,
            "acceleration": self.acceleration,
            "social_mentions": self.social_mentions,
            "social_velocity": self.social_velocity,
            "sentiment_velocity": self.sentiment_velocity,
            "financial_impact_score": self.financial_impact_score,
            "market_relevance": self.market_relevance,
            "urgency_score": self.urgency_score,
            "first_seen": self.first_seen.isoformat(),
            "peak_velocity": self.peak_velocity.isoformat(),
            "detected_at": self.detected_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "sources": self.sources,
            "source_diversity": self.source_diversity,
            "breaking_keywords": self.breaking_keywords,
            "hype_indicators": self.hype_indicators
        }


@dataclass 
class StoryMention:
    """Individual mention of a story"""
    timestamp: datetime
    source: str
    title: str
    sentiment: float  # -1.0 to 1.0
    social_engagement: int  # likes, shares, etc.


class BreakingNewsVelocityTracker(ComponentBase):
    """
    Real-time breaking news velocity tracking system
    
    Monitors news story spread patterns, social media acceleration, and hype
    indicators to identify rapid market-moving events requiring immediate trading action.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("breaking_news_velocity_tracker", config)
        
        # Configuration
        self.update_interval = config.get("update_interval", 30)  # 30 seconds
        self.velocity_window_minutes = config.get("velocity_window_minutes", 10)
        self.min_velocity_score = config.get("min_velocity_score", 2.0)
        self.story_expiry_hours = config.get("story_expiry_hours", 4)
        
        # Velocity thresholds
        self.velocity_thresholds = {
            VelocityLevel.VIRAL: 8.0,     # Extreme velocity
            VelocityLevel.BREAKING: 5.0,  # High velocity  
            VelocityLevel.TRENDING: 2.5,  # Medium velocity
            VelocityLevel.NORMAL: 1.0     # Standard velocity
        }
        
        # Breaking news indicators
        self.breaking_keywords = [
            # Urgency indicators
            "BREAKING:", "URGENT:", "JUST IN:", "ALERT:", "DEVELOPING:",
            "EXCLUSIVE:", "CONFIRMED:", "OFFICIAL:",
            
            # Market shock words
            "crashes", "plunges", "soars", "surges", "halted", "suspended",
            "emergency", "investigation", "lawsuit", "merger", "acquisition",
            
            # Financial events
            "earnings beat", "earnings miss", "guidance raised", "guidance lowered",
            "FDA approval", "clinical trial", "regulatory action", "recall",
            "bankruptcy", "default", "downgrade", "upgrade",
            
            # Crypto specific
            "hack", "exploit", "fork", "listing", "delisting", "regulation"
        ]
        
        # Hype indicators (social/momentum)
        self.hype_indicators = [
            # Social momentum
            "viral", "trending", "exploding", "moon", "rocket", "diamond hands",
            "to the moon", "squeeze", "gamma squeeze", "short squeeze",
            
            # FOMO indicators
            "don't miss", "last chance", "opportunity", "breakout", "pump",
            "bullish", "bearish", "all-time high", "new high", "record",
            
            # Volume indicators
            "unusual activity", "high volume", "spike", "surge", "momentum"
        ]
        
        # Financial relevance keywords
        self.financial_keywords = [
            # Companies/Markets
            "stock", "shares", "market", "trading", "price", "valuation",
            "earnings", "revenue", "profit", "loss", "dividend", "split",
            
            # Crypto
            "bitcoin", "ethereum", "crypto", "cryptocurrency", "blockchain",
            "defi", "nft", "token", "coin", "mining", "staking",
            
            # Economic
            "fed", "federal reserve", "interest rate", "inflation", "recession",
            "gdp", "unemployment", "jobs", "consumer", "retail"
        ]
        
        # Story tracking
        self.active_stories: Dict[str, NewsVelocitySignal] = {}  # story_id -> signal
        self.story_mentions: Dict[str, List[StoryMention]] = {}  # story_id -> mentions
        self.story_hashes: Dict[str, str] = {}  # content_hash -> story_id
        
        # Performance tracking
        self.velocity_signals_generated = 0
        self.viral_stories_detected = 0
        self.breaking_stories_detected = 0
        self.successful_predictions = 0
        
        # Source tracking
        self.source_reliability: Dict[str, float] = defaultdict(lambda: 1.0)
        self.source_velocity_multipliers: Dict[str, float] = {
            # High-priority sources get velocity boost
            "reuters": 1.5,
            "bloomberg": 1.4,
            "cnbc": 1.3,
            "marketwatch": 1.2,
            "yahoo finance": 1.1,
            "reddit": 2.0,  # High social velocity
            "twitter": 1.8,
            "seekingalpha": 1.1
        }
    
    async def start(self) -> None:
        """Start velocity tracking"""
        self.logger.info("⚡ Starting Breaking News Velocity Tracker")
        self.is_running = True
        
        self.logger.info("✅ Velocity tracker started - monitoring news acceleration")
    
    async def stop(self) -> None:
        """Stop velocity tracking"""
        self.logger.info("🛑 Stopping Breaking News Velocity Tracker")
        self.is_running = False
        
        self._log_performance_summary()
    
    async def process(self, news_articles: List[Dict[str, Any]]) -> List[NewsVelocitySignal]:
        """Process news articles and detect velocity patterns"""
        if not self.is_running or not news_articles:
            return []
        
        current_time = datetime.utcnow()
        velocity_signals = []
        
        try:
            # Process each article for velocity tracking
            for article in news_articles:
                story_mention = self._create_story_mention(article)
                if story_mention:
                    story_id = self._get_or_create_story_id(article)
                    self._add_story_mention(story_id, story_mention)
            
            # Analyze velocity for all tracked stories
            for story_id in list(self.story_mentions.keys()):
                velocity_signal = await self._analyze_story_velocity(story_id)
                if velocity_signal and velocity_signal.velocity_score >= self.min_velocity_score:
                    velocity_signals.append(velocity_signal)
                    self.active_stories[story_id] = velocity_signal
            
            # Clean up old stories
            await self._cleanup_expired_stories()
            
            # Log results
            if velocity_signals:
                self.logger.info(f"⚡ Detected {len(velocity_signals)} velocity signals")
                for signal in velocity_signals:
                    self.logger.info(f"   {signal.velocity_level.value.upper()}: {signal.title[:60]}... "
                                   f"(score: {signal.velocity_score:.1f}, rate: {signal.spread_rate:.1f}/min)")
            
            self.velocity_signals_generated += len(velocity_signals)
            return velocity_signals
            
        except Exception as e:
            self.logger.error(f"Velocity tracking error: {e}")
            return []
    
    def _create_story_mention(self, article: Dict[str, Any]) -> Optional[StoryMention]:
        """Create a story mention from news article"""
        try:
            title = article.get("title", "").strip()
            content = article.get("content", "").strip()
            
            if not title and not content:
                return None
            
            # Parse timestamp
            published = article.get("published")
            if isinstance(published, str):
                timestamp = datetime.fromisoformat(published.replace('Z', '+00:00'))
            elif isinstance(published, datetime):
                timestamp = published
            else:
                timestamp = datetime.utcnow()
            
            # Extract sentiment
            sentiment = article.get("sentiment", 0.0)
            if isinstance(sentiment, str):
                # Convert sentiment labels to scores
                sentiment_map = {"positive": 0.5, "negative": -0.5, "neutral": 0.0}
                sentiment = sentiment_map.get(sentiment.lower(), 0.0)
            
            # Extract social engagement
            social_metrics = article.get("social_metrics", {})
            engagement = (
                social_metrics.get("upvotes", 0) + 
                social_metrics.get("likes", 0) + 
                social_metrics.get("retweets", 0) + 
                social_metrics.get("score", 0)
            )
            
            return StoryMention(
                timestamp=timestamp,
                source=article.get("source", "unknown"),
                title=title,
                sentiment=float(sentiment),
                social_engagement=engagement
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating story mention: {e}")
            return None
    
    def _get_or_create_story_id(self, article: Dict[str, Any]) -> str:
        """Get existing story ID or create new one based on content similarity"""
        title = article.get("title", "").strip().lower()
        content = article.get("content", "").strip().lower()
        
        # Create content hash for similarity matching
        combined_text = f"{title} {content}"[:500]  # Use first 500 chars
        content_hash = hashlib.md5(combined_text.encode()).hexdigest()
        
        # Check for existing similar story
        if content_hash in self.story_hashes:
            return self.story_hashes[content_hash]
        
        # Create new story ID
        story_id = f"story_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{content_hash[:8]}"
        self.story_hashes[content_hash] = story_id
        
        return story_id
    
    def _add_story_mention(self, story_id: str, mention: StoryMention):
        """Add mention to story tracking"""
        if story_id not in self.story_mentions:
            self.story_mentions[story_id] = []
        
        self.story_mentions[story_id].append(mention)
        
        # Keep only recent mentions within velocity window
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.velocity_window_minutes * 2)
        self.story_mentions[story_id] = [
            m for m in self.story_mentions[story_id] 
            if m.timestamp > cutoff_time
        ]
    
    async def _analyze_story_velocity(self, story_id: str) -> Optional[NewsVelocitySignal]:
        """Analyze velocity pattern for a story"""
        mentions = self.story_mentions.get(story_id, [])
        if len(mentions) < 2:
            return None
        
        try:
            current_time = datetime.utcnow()
            
            # Sort mentions by time
            mentions.sort(key=lambda m: m.timestamp)
            
            # Get latest mention for story details
            latest_mention = mentions[-1]
            first_mention = mentions[0]
            
            # Calculate velocity window
            velocity_window = timedelta(minutes=self.velocity_window_minutes)
            recent_mentions = [
                m for m in mentions 
                if current_time - m.timestamp <= velocity_window
            ]
            
            if len(recent_mentions) < 2:
                return None
            
            # Calculate spread rate (mentions per minute)
            time_span_minutes = (recent_mentions[-1].timestamp - recent_mentions[0].timestamp).total_seconds() / 60
            if time_span_minutes <= 0:
                return None
            
            spread_rate = len(recent_mentions) / time_span_minutes
            
            # Calculate acceleration (change in spread rate)
            if len(mentions) >= 4:
                half_point = len(recent_mentions) // 2
                early_mentions = recent_mentions[:half_point]
                late_mentions = recent_mentions[half_point:]
                
                early_rate = len(early_mentions) / (time_span_minutes / 2) if time_span_minutes > 0 else 0
                late_rate = len(late_mentions) / (time_span_minutes / 2) if time_span_minutes > 0 else 0
                acceleration = late_rate - early_rate
            else:
                acceleration = 0.0
            
            # Extract symbols from mentions
            symbols = self._extract_symbols(mentions)
            
            # Calculate social metrics
            social_mentions = len([m for m in recent_mentions if 'reddit' in m.source.lower() or 'twitter' in m.source.lower()])
            total_engagement = sum(m.social_engagement for m in recent_mentions)
            social_velocity = social_mentions / time_span_minutes if time_span_minutes > 0 else 0
            
            # Calculate sentiment velocity
            sentiment_changes = []
            for i in range(1, len(recent_mentions)):
                sentiment_changes.append(abs(recent_mentions[i].sentiment - recent_mentions[i-1].sentiment))
            sentiment_velocity = sum(sentiment_changes) / len(sentiment_changes) if sentiment_changes else 0.0
            
            # Calculate scores
            financial_impact_score = self._calculate_financial_impact(mentions)
            market_relevance = self._calculate_market_relevance(mentions, symbols)
            urgency_score = self._calculate_urgency_score(mentions)
            
            # Calculate source diversity
            unique_sources = len(set(m.source for m in recent_mentions))
            source_diversity = min(unique_sources / 5.0, 1.0)  # Normalize to 0-1
            
            # Apply source velocity multipliers
            source_multiplier = max([
                self.source_velocity_multipliers.get(m.source.lower(), 1.0) 
                for m in recent_mentions
            ])
            
            # Calculate overall velocity score
            base_velocity = spread_rate * source_multiplier
            velocity_score = (
                base_velocity * 0.4 +                    # Base spread rate
                acceleration * 0.2 +                     # Acceleration
                social_velocity * 0.2 +                  # Social acceleration
                sentiment_velocity * 10 * 0.1 +          # Sentiment change
                source_diversity * 3 * 0.1               # Source diversity
            )
            
            # Determine velocity level
            velocity_level = VelocityLevel.NORMAL
            for level in [VelocityLevel.VIRAL, VelocityLevel.BREAKING, VelocityLevel.TRENDING]:
                if velocity_score >= self.velocity_thresholds[level]:
                    velocity_level = level
                    break
            
            # Find breaking keywords and hype indicators
            all_text = ' '.join([m.title for m in mentions]).lower()
            breaking_keywords = [kw for kw in self.breaking_keywords if kw.lower() in all_text]
            hype_indicators = [hi for hi in self.hype_indicators if hi.lower() in all_text]
            
            # Find peak velocity time
            mention_times = [m.timestamp for m in mentions]
            peak_velocity = max(mention_times) if mention_times else current_time
            
            # Create velocity signal
            signal = NewsVelocitySignal(
                story_id=story_id,
                title=latest_mention.title,
                content=' '.join([m.title for m in mentions[:3]]),  # Combine first few titles
                symbols=symbols,
                velocity_level=velocity_level,
                velocity_score=velocity_score,
                spread_rate=spread_rate,
                acceleration=acceleration,
                social_mentions=social_mentions,
                social_velocity=social_velocity,
                sentiment_velocity=sentiment_velocity,
                financial_impact_score=financial_impact_score,
                market_relevance=market_relevance,
                urgency_score=urgency_score,
                first_seen=first_mention.timestamp,
                peak_velocity=peak_velocity,
                detected_at=current_time,
                expires_at=current_time + timedelta(hours=self.story_expiry_hours),
                sources=list(set(m.source for m in mentions)),
                source_diversity=source_diversity,
                breaking_keywords=breaking_keywords,
                hype_indicators=hype_indicators
            )
            
            # Update counters
            if velocity_level == VelocityLevel.VIRAL:
                self.viral_stories_detected += 1
            elif velocity_level == VelocityLevel.BREAKING:
                self.breaking_stories_detected += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing story velocity for {story_id}: {e}")
            return None
    
    def _extract_symbols(self, mentions: List[StoryMention]) -> List[str]:
        """Extract stock symbols from mentions"""
        symbols = set()
        
        for mention in mentions:
            text = f"{mention.title}".upper()
            
            # Find $SYMBOL patterns
            dollar_symbols = re.findall(r'\$([A-Z]{1,5})\b', text)
            symbols.update(dollar_symbols)
            
            # Find standalone symbols (2-5 capital letters)
            standalone_symbols = re.findall(r'\b([A-Z]{2,5})\b', text)
            
            # Filter for likely stock symbols
            for symbol in standalone_symbols:
                if (len(symbol) >= 2 and 
                    symbol not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'USD', 'EUR', 'GBP'] and
                    any(keyword in text.lower() for keyword in ['stock', 'shares', 'trading', 'price', 'earnings', 'analyst', 'upgrade', 'downgrade'])):
                    symbols.add(symbol)
        
        return list(symbols)
    
    def _calculate_financial_impact(self, mentions: List[StoryMention]) -> float:
        """Calculate financial impact score based on content"""
        all_text = ' '.join([m.title for m in mentions]).lower()
        
        impact_score = 0.0
        
        # High-impact keywords
        high_impact_keywords = [
            'earnings', 'guidance', 'merger', 'acquisition', 'bankruptcy', 
            'lawsuit', 'investigation', 'fda approval', 'clinical trial',
            'upgrade', 'downgrade', 'price target', 'analyst'
        ]
        
        for keyword in high_impact_keywords:
            if keyword in all_text:
                impact_score += 0.2
        
        # Breaking news multiplier
        if any(kw in all_text for kw in ['breaking', 'just in', 'urgent']):
            impact_score *= 1.5
        
        return min(impact_score, 1.0)
    
    def _calculate_market_relevance(self, mentions: List[StoryMention], symbols: List[str]) -> float:
        """Calculate market relevance score"""
        all_text = ' '.join([m.title for m in mentions]).lower()
        
        relevance_score = 0.0
        
        # Symbol mentions boost relevance
        relevance_score += min(len(symbols) * 0.2, 0.6)
        
        # Financial keywords
        financial_mentions = sum(1 for kw in self.financial_keywords if kw in all_text)
        relevance_score += min(financial_mentions * 0.1, 0.4)
        
        return min(relevance_score, 1.0)
    
    def _calculate_urgency_score(self, mentions: List[StoryMention]) -> float:
        """Calculate urgency score based on language and timing"""
        all_text = ' '.join([m.title for m in mentions]).lower()
        
        urgency_score = 0.0
        
        # Urgent keywords
        urgent_keywords = ['breaking', 'urgent', 'just in', 'alert', 'immediate', 'emergency']
        for keyword in urgent_keywords:
            if keyword in all_text:
                urgency_score += 0.3
        
        # Time-sensitive words
        time_sensitive = ['now', 'today', 'immediately', 'seconds', 'minutes', 'crashing', 'soaring']
        for keyword in time_sensitive:
            if keyword in all_text:
                urgency_score += 0.2
        
        # Recent mentions boost urgency
        recent_mentions = len([m for m in mentions if (datetime.utcnow() - m.timestamp).total_seconds() < 300])  # 5 minutes
        if recent_mentions >= 3:
            urgency_score += 0.3
        
        return min(urgency_score, 1.0)
    
    async def _cleanup_expired_stories(self):
        """Remove expired stories from tracking"""
        current_time = datetime.utcnow()
        expired_stories = []
        
        # Check active stories
        for story_id, signal in self.active_stories.items():
            if current_time > signal.expires_at:
                expired_stories.append(story_id)
        
        # Check story mentions
        expiry_cutoff = current_time - timedelta(hours=self.story_expiry_hours)
        for story_id in list(self.story_mentions.keys()):
            mentions = self.story_mentions[story_id]
            if not mentions or all(m.timestamp < expiry_cutoff for m in mentions):
                expired_stories.append(story_id)
        
        # Clean up expired stories
        for story_id in set(expired_stories):
            self.active_stories.pop(story_id, None)
            self.story_mentions.pop(story_id, None)
        
        # Clean up story hashes
        active_story_ids = set(self.story_mentions.keys())
        expired_hashes = [h for h, sid in self.story_hashes.items() if sid not in active_story_ids]
        for h in expired_hashes:
            del self.story_hashes[h]
        
        if expired_stories:
            self.logger.debug(f"Cleaned up {len(set(expired_stories))} expired stories")
    
    def get_active_signals(self) -> List[NewsVelocitySignal]:
        """Get all active velocity signals"""
        return list(self.active_stories.values())
    
    def get_viral_signals(self) -> List[NewsVelocitySignal]:
        """Get only viral-level velocity signals"""
        return [s for s in self.active_stories.values() if s.velocity_level == VelocityLevel.VIRAL]
    
    def get_breaking_signals(self) -> List[NewsVelocitySignal]:
        """Get breaking and viral velocity signals"""
        return [s for s in self.active_stories.values() 
                if s.velocity_level in [VelocityLevel.VIRAL, VelocityLevel.BREAKING]]
    
    def update_prediction_success(self, story_id: str, success: bool):
        """Update prediction success tracking"""
        if success:
            self.successful_predictions += 1
    
    def _log_performance_summary(self):
        """Log velocity tracking performance summary"""
        self.logger.info("⚡ BREAKING NEWS VELOCITY TRACKER PERFORMANCE:")
        self.logger.info(f"  Velocity signals generated: {self.velocity_signals_generated}")
        self.logger.info(f"  Viral stories detected: {self.viral_stories_detected}")
        self.logger.info(f"  Breaking stories detected: {self.breaking_stories_detected}")
        self.logger.info(f"  Successful predictions: {self.successful_predictions}")
        self.logger.info(f"  Active stories: {len(self.active_stories)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            "is_running": self.is_running,
            "active_stories": len(self.active_stories),
            "velocity_signals_generated": self.velocity_signals_generated,
            "viral_stories_detected": self.viral_stories_detected,
            "breaking_stories_detected": self.breaking_stories_detected,
            "successful_predictions": self.successful_predictions,
            "current_signals": [s.to_dict() for s in self.active_stories.values()],
            "velocity_levels": {
                level.value: len([s for s in self.active_stories.values() if s.velocity_level == level])
                for level in VelocityLevel
            }
        }
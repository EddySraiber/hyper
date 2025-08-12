"""
Reddit Sentiment Processing for Financial Analysis
Processes Reddit posts and comments to extract financial sentiment and trading signals
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from textblob import TextBlob

from .reddit_client import RedditPost, RedditComment, preprocess_reddit_text, calculate_reddit_influence_score


@dataclass
class RedditSentimentSignal:
    """Sentiment signal extracted from Reddit data"""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float      # 0.0 to 1.0
    quality_score: float   # 0.0 to 1.0 (based on discussion quality)
    influence_score: float # 0.0 to 1.0 (based on engagement/reach)
    post_count: int
    comment_count: int
    timeframe: str         # e.g., "1h", "4h", "24h"
    created_at: str
    top_posts: List[Dict[str, Any]]  # Sample of most influential posts
    subreddit_distribution: Dict[str, int]  # Count by subreddit
    keywords: List[str]    # Financial keywords mentioned
    sentiment_distribution: Dict[str, int]  # bullish, bearish, neutral counts


class RedditSentimentProcessor:
    """
    Processes Reddit data to extract actionable financial sentiment signals
    Focuses on high-quality financial discussions vs meme/hype content
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('reddit_sentiment', {})
        self.logger = logging.getLogger(__name__)
        
        # Sentiment analysis configuration
        self.min_posts_for_signal = self.config.get('min_posts_for_signal', 3)
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.1)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.4)
        self.quality_threshold = self.config.get('quality_threshold', 0.3)
        
        # Time-based analysis windows
        self.analysis_windows = {
            '1h': 1,
            '4h': 4, 
            '12h': 12,
            '24h': 24,
            '3d': 72,     # Reddit discussions evolve over days
            '7d': 168     # Weekly investment thesis discussions
        }
        
        # Subreddit quality weights (higher = more reliable)
        self.subreddit_weights = {
            'SecurityAnalysis': 1.0,      # Highest quality fundamental analysis
            'ValueInvesting': 0.95,       # Value investing community
            'investing': 0.9,             # Broad high-quality community
            'financialindependence': 0.85, # Long-term focused
            'dividends': 0.8,             # Conservative dividend focus
            'StockMarket': 0.75,          # General market discussions
            'stocks': 0.7,                # Popular but varied quality
            'options': 0.65,              # More speculative but sophisticated
        }
        
        # Enhanced financial sentiment keywords with context awareness
        self.sentiment_keywords = {
            'bullish': {
                # Investment thesis terms (higher weight)
                'undervalued': 0.9, 'strong fundamentals': 1.0, 'solid balance sheet': 0.8,
                'growing revenue': 0.8, 'expanding margins': 0.7, 'competitive advantage': 0.9,
                'market leader': 0.7, 'dividend growth': 0.6, 'free cash flow': 0.8,
                
                # Positive performance terms
                'beat estimates': 1.0, 'exceeded expectations': 0.9, 'strong guidance': 0.8,
                'bullish': 0.8, 'buy the dip': 0.6, 'long term hold': 0.7,
                'accumulating': 0.6, 'dollar cost averaging': 0.5,
                
                # Growth and momentum terms
                'breakthrough': 0.8, 'innovation': 0.6, 'market expansion': 0.7,
                'partnership': 0.6, 'acquisition target': 0.8, 'rally': 0.6,
                
                # Technical terms
                'breakout': 0.7, 'support level': 0.5, 'uptrend': 0.6, 'momentum': 0.5
            },
            'bearish': {
                # Fundamental concerns (higher weight)
                'overvalued': -0.9, 'weak fundamentals': -1.0, 'debt concerns': -0.8,
                'declining revenue': -0.8, 'margin compression': -0.7, 'losing market share': -0.9,
                'competitive threats': -0.7, 'regulatory risk': -0.6, 'cash burn': -0.8,
                
                # Negative performance terms
                'missed estimates': -1.0, 'disappointed': -0.8, 'weak guidance': -0.8,
                'bearish': -0.8, 'sell recommendation': -0.9, 'taking profits': -0.4,
                'reducing position': -0.5, 'stop loss triggered': -0.7,
                
                # Market concerns
                'recession': -0.7, 'economic headwinds': -0.6, 'inflation pressure': -0.5,
                'supply chain': -0.4, 'geopolitical risk': -0.5, 'correction': -0.6,
                
                # Technical terms
                'breakdown': -0.7, 'resistance level': -0.5, 'downtrend': -0.6, 'selling pressure': -0.6
            }
        }
        
        # Context modifiers that affect sentiment strength
        self.context_modifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.2, 'significantly': 1.2,
            'somewhat': 0.7, 'slightly': 0.6, 'moderately': 0.8,
            'not': -1.0, "don't": -1.0, 'unlikely': -0.8, 'doubtful': -0.7
        }
        
    async def process_symbol_sentiment(self, symbol: str, posts: List[RedditPost], 
                                     comments: Dict[str, List[RedditComment]] = None,
                                     timeframe: str = "24h") -> Optional[RedditSentimentSignal]:
        """
        Process Reddit posts and comments for a specific symbol to generate sentiment signal
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            posts: List of Reddit posts related to the symbol
            comments: Optional dictionary mapping post IDs to their comments
            timeframe: Analysis timeframe (1h, 4h, 12h, 24h, 3d, 7d)
            
        Returns:
            RedditSentimentSignal or None if insufficient data
        """
        if not posts or len(posts) < self.min_posts_for_signal:
            self.logger.debug(f"Insufficient posts for {symbol}: {len(posts)} < {self.min_posts_for_signal}")
            return None
            
        # Filter posts by timeframe
        filtered_posts = self._filter_posts_by_time(posts, timeframe)
        if len(filtered_posts) < self.min_posts_for_signal:
            return None
            
        comments = comments or {}
        
        # Analyze sentiment for each post
        post_sentiments = []
        total_comments = 0
        
        for post in filtered_posts:
            post_comments = comments.get(post.id, [])
            total_comments += len(post_comments)
            
            sentiment_data = await self._analyze_post_sentiment(post, post_comments, symbol)
            if sentiment_data:
                post_sentiments.append(sentiment_data)
                
        if not post_sentiments:
            return None
            
        # Aggregate sentiment signals
        signal = self._aggregate_sentiment_signals(symbol, post_sentiments, timeframe)
        signal.comment_count = total_comments
        
        # Add sample of most influential posts
        signal.top_posts = self._get_top_influential_posts(post_sentiments, limit=3)
        
        return signal
        
    def _filter_posts_by_time(self, posts: List[RedditPost], timeframe: str) -> List[RedditPost]:
        """Filter posts by specified timeframe"""
        hours_back = self.analysis_windows.get(timeframe, 24)
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        filtered = []
        for post in posts:
            try:
                # Parse post timestamp
                post_time = datetime.fromisoformat(post.created_at.replace('Z', '+00:00'))
                if post_time.replace(tzinfo=None) >= cutoff_time:
                    filtered.append(post)
            except (ValueError, AttributeError):
                # If timestamp parsing fails, include the post
                filtered.append(post)
                
        return filtered
        
    async def _analyze_post_sentiment(self, post: RedditPost, comments: List[RedditComment], 
                                    symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of individual post and its comments"""
        try:
            # Combine post and top comments for analysis
            full_text = f"{post.title} {post.text}"
            
            # Add high-quality comments (score > 5, not too deep)
            quality_comments = [c for c in comments if c.score > 5 and c.depth <= 2][:5]
            comment_text = " ".join([c.text for c in quality_comments])
            
            if comment_text:
                full_text += " " + comment_text
                
            # Preprocess combined text
            clean_text = preprocess_reddit_text(full_text)
            
            # Basic TextBlob sentiment
            blob = TextBlob(clean_text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Enhanced Reddit-specific financial sentiment analysis
            reddit_sentiment = self._analyze_reddit_financial_sentiment(full_text, symbol, post.subreddit)
            
            # Combine sentiments with weighting (favor Reddit-specific analysis)
            combined_sentiment = (textblob_sentiment * 0.3) + (reddit_sentiment * 0.7)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_sentiment_confidence(
                post, comments, textblob_sentiment, reddit_sentiment
            )
            
            # Calculate quality score
            quality_score = self._calculate_discussion_quality(post, comments)
            
            # Calculate influence score
            influence_score = calculate_reddit_influence_score(post, comments)
            
            return {
                'post': post,
                'comments': quality_comments,
                'clean_text': clean_text,
                'textblob_sentiment': textblob_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'combined_sentiment': combined_sentiment,
                'confidence': confidence,
                'quality_score': quality_score,
                'influence_score': influence_score,
                'subreddit_weight': self.subreddit_weights.get(post.subreddit, 0.5),
                'keywords': self._extract_financial_keywords(full_text),
                'discussion_depth': len(comments),
                'avg_comment_score': sum(c.score for c in comments) / max(len(comments), 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing post sentiment: {e}")
            return None
            
    def _analyze_reddit_financial_sentiment(self, text: str, symbol: str, subreddit: str) -> float:
        """Analyze Reddit-specific financial sentiment with context awareness"""
        text_lower = text.lower()
        sentiment_score = 0.0
        keyword_count = 0
        
        # Split text into sentences for context analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_sentiment = 0.0
            sentence_keywords = 0
            
            # Check bullish keywords with context
            for keyword, weight in self.sentiment_keywords['bullish'].items():
                if keyword in sentence_lower:
                    # Apply context modifiers
                    modified_weight = self._apply_context_modifiers(sentence_lower, keyword, weight)
                    sentence_sentiment += modified_weight
                    sentence_keywords += 1
                    
            # Check bearish keywords with context
            for keyword, weight in self.sentiment_keywords['bearish'].items():
                if keyword in sentence_lower:
                    modified_weight = self._apply_context_modifiers(sentence_lower, keyword, weight)
                    sentence_sentiment += modified_weight  # weight is already negative
                    sentence_keywords += 1
            
            # Add sentence sentiment to total
            if sentence_keywords > 0:
                sentiment_score += sentence_sentiment / sentence_keywords
                keyword_count += 1
        
        # Apply subreddit quality weighting
        subreddit_weight = self.subreddit_weights.get(subreddit, 0.5)
        
        # Normalize and apply subreddit weighting
        if keyword_count > 0:
            sentiment_score = (sentiment_score / keyword_count) * subreddit_weight
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment_score))
        
    def _apply_context_modifiers(self, sentence: str, keyword: str, base_weight: float) -> float:
        """Apply context modifiers to adjust keyword sentiment strength"""
        modified_weight = base_weight
        
        # Find the keyword position
        keyword_pos = sentence.find(keyword)
        if keyword_pos == -1:
            return modified_weight
            
        # Look for modifiers in the vicinity of the keyword
        context_window = sentence[max(0, keyword_pos - 50):keyword_pos + len(keyword) + 50]
        
        for modifier, multiplier in self.context_modifiers.items():
            if modifier in context_window:
                modified_weight *= multiplier
                break  # Use first modifier found
                
        return modified_weight
        
    def _calculate_sentiment_confidence(self, post: RedditPost, comments: List[RedditComment],
                                      textblob_score: float, reddit_score: float) -> float:
        """Calculate confidence score for Reddit sentiment analysis"""
        # Base confidence from sentiment strength
        sentiment_strength = max(abs(textblob_score), abs(reddit_score))
        
        # Agreement bonus (both methods agree on direction)
        agreement_bonus = 0.15 if (textblob_score * reddit_score > 0) else 0
        
        # Discussion quality bonus
        discussion_quality = self._calculate_discussion_quality(post, comments)
        quality_bonus = discussion_quality * 0.2
        
        # Subreddit credibility bonus
        subreddit_bonus = self.subreddit_weights.get(post.subreddit, 0.5) * 0.15
        
        # Engagement bonus (Reddit-specific metrics)
        engagement_score = (
            min(post.score / 50.0, 1.0) * 0.4 +          # Normalized upvotes
            post.upvote_ratio * 0.3 +                     # Upvote ratio
            min(post.num_comments / 20.0, 1.0) * 0.3     # Comment engagement
        )
        engagement_bonus = engagement_score * 0.15
        
        # Content depth bonus (longer, more detailed posts)
        content_length = len(post.title) + len(post.text)
        if comments:
            content_length += sum(len(c.text) for c in comments[:5])
        content_bonus = min(content_length / 1000.0, 0.15)
        
        confidence = (
            sentiment_strength + agreement_bonus + quality_bonus + 
            subreddit_bonus + engagement_bonus + content_bonus
        )
        return min(confidence, 1.0)
        
    def _calculate_discussion_quality(self, post: RedditPost, comments: List[RedditComment]) -> float:
        """Calculate quality score for Reddit discussion"""
        # Post quality factors
        post_score_norm = min(post.score / 100.0, 1.0)
        upvote_ratio = post.upvote_ratio
        text_quality = min(len(post.text) / 500.0, 1.0)  # Longer posts often more thoughtful
        
        # Comment quality factors
        comment_quality = 0.0
        if comments:
            avg_comment_score = sum(c.score for c in comments) / len(comments)
            comment_score_norm = min(avg_comment_score / 10.0, 1.0)
            
            # Depth of discussion (replies to replies indicate engagement)
            max_depth = max((c.depth for c in comments), default=0)
            depth_bonus = min(max_depth / 5.0, 0.2)
            
            comment_quality = comment_score_norm + depth_bonus
            
        # Combine quality factors
        quality_score = (
            post_score_norm * 0.3 +
            upvote_ratio * 0.2 +
            text_quality * 0.2 +
            comment_quality * 0.3
        )
        
        return min(quality_score, 1.0)
        
    def _extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial keywords from Reddit text"""
        text_lower = text.lower()
        found_keywords = []
        
        all_keywords = list(self.sentiment_keywords['bullish'].keys()) + \
                      list(self.sentiment_keywords['bearish'].keys())
        
        for keyword in all_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                
        return found_keywords
        
    def _aggregate_sentiment_signals(self, symbol: str, post_sentiments: List[Dict[str, Any]], 
                                   timeframe: str) -> RedditSentimentSignal:
        """Aggregate individual post sentiments into overall signal"""
        if not post_sentiments:
            return None
            
        # Calculate weighted average sentiment (weighted by quality, influence, and subreddit)
        total_weighted_sentiment = 0.0
        total_weights = 0.0
        sentiment_distribution = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        subreddit_distribution = {}
        all_keywords = []
        
        for sentiment_data in post_sentiments:
            sentiment = sentiment_data['combined_sentiment']
            confidence = sentiment_data['confidence']
            quality = sentiment_data['quality_score']
            influence = sentiment_data['influence_score']
            subreddit_weight = sentiment_data['subreddit_weight']
            subreddit = sentiment_data['post'].subreddit
            
            # Weight = confidence * quality * influence * subreddit_credibility
            weight = confidence * quality * max(influence, 0.1) * subreddit_weight
            
            total_weighted_sentiment += sentiment * weight
            total_weights += weight
            
            # Update sentiment distribution
            if sentiment > 0.15:  # Higher threshold for Reddit (more conservative)
                sentiment_distribution['bullish'] += 1
            elif sentiment < -0.15:
                sentiment_distribution['bearish'] += 1
            else:
                sentiment_distribution['neutral'] += 1
                
            # Update subreddit distribution
            subreddit_distribution[subreddit] = subreddit_distribution.get(subreddit, 0) + 1
                
            # Collect keywords
            all_keywords.extend(sentiment_data.get('keywords', []))
            
        # Calculate final scores
        avg_sentiment = total_weighted_sentiment / max(total_weights, 1.0)
        avg_confidence = sum(s['confidence'] for s in post_sentiments) / len(post_sentiments)
        avg_quality = sum(s['quality_score'] for s in post_sentiments) / len(post_sentiments)
        avg_influence = sum(s['influence_score'] for s in post_sentiments) / len(post_sentiments)
        
        return RedditSentimentSignal(
            symbol=symbol,
            sentiment_score=avg_sentiment,
            confidence=avg_confidence,
            quality_score=avg_quality,
            influence_score=avg_influence,
            post_count=len(post_sentiments),
            comment_count=0,  # Will be set by caller
            timeframe=timeframe,
            created_at=datetime.utcnow().isoformat(),
            top_posts=[],  # Will be filled by caller
            subreddit_distribution=subreddit_distribution,
            keywords=list(set(all_keywords)),  # Deduplicate
            sentiment_distribution=sentiment_distribution
        )
        
    def _get_top_influential_posts(self, post_sentiments: List[Dict[str, Any]], 
                                  limit: int = 3) -> List[Dict[str, Any]]:
        """Get top influential posts for the signal"""
        # Sort by combined score (influence * quality * confidence)
        sorted_posts = sorted(
            post_sentiments,
            key=lambda x: x['influence_score'] * x['quality_score'] * x['confidence'],
            reverse=True
        )
        
        top_posts = []
        for sentiment_data in sorted_posts[:limit]:
            post = sentiment_data['post']
            top_posts.append({
                'id': post.id,
                'title': post.title,
                'text': post.text[:300] + '...' if len(post.text) > 300 else post.text,
                'subreddit': post.subreddit,
                'author': post.author,
                'created_at': post.created_at,
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'sentiment': sentiment_data['combined_sentiment'],
                'confidence': sentiment_data['confidence'],
                'quality_score': sentiment_data['quality_score'],
                'influence_score': sentiment_data['influence_score'],
                'keywords': sentiment_data.get('keywords', []),
                'url': f"https://reddit.com{post.permalink}"
            })
            
        return top_posts
        
    def get_signal_quality_score(self, signal: RedditSentimentSignal) -> float:
        """
        Calculate overall quality score for the Reddit sentiment signal
        
        Returns:
            Quality score from 0.0 to 1.0
        """
        if not signal:
            return 0.0
            
        # Components of quality score (Reddit-specific)
        sentiment_strength = abs(signal.sentiment_score)   # 0.0 - 1.0
        confidence_score = signal.confidence               # 0.0 - 1.0  
        discussion_quality = signal.quality_score          # 0.0 - 1.0
        influence_score = signal.influence_score           # 0.0 - 1.0
        
        # Post count bonus (more posts = more reliable, but with diminishing returns)
        count_bonus = min(signal.post_count / 10.0, 0.2)  # Cap at 0.2
        
        # Comment engagement bonus (indicates active discussion)
        comment_bonus = min(signal.comment_count / (signal.post_count * 5.0), 0.15)
        
        # Subreddit diversity bonus (multiple subreddits = broader consensus)
        subreddit_count = len(signal.subreddit_distribution)
        diversity_bonus = min((subreddit_count - 1) * 0.05, 0.15)  # Up to 0.15 for 4+ subreddits
        
        # Sentiment consensus bonus (clear direction is better)
        total_directional = (signal.sentiment_distribution['bullish'] + 
                           signal.sentiment_distribution['bearish'])
        if total_directional > 0:
            consensus_ratio = max(signal.sentiment_distribution.values()) / total_directional
            consensus_bonus = (consensus_ratio - 0.5) * 0.3  # 0 to 0.15 bonus
        else:
            consensus_bonus = 0
        
        # Weighted combination (Reddit emphasizes discussion quality)
        quality_score = (
            sentiment_strength * 0.25 +
            confidence_score * 0.25 +
            discussion_quality * 0.2 +    # Higher weight for Reddit
            influence_score * 0.1 +
            count_bonus +
            comment_bonus +
            diversity_bonus +
            max(0, consensus_bonus)
        )
        
        return min(quality_score, 1.0)
        
    def is_signal_actionable(self, signal: RedditSentimentSignal) -> bool:
        """
        Determine if Reddit signal is strong enough to be actionable for trading
        
        Returns:
            True if signal meets actionability criteria
        """
        if not signal:
            return False
            
        # Minimum criteria (more conservative than Twitter)
        min_confidence = self.confidence_threshold
        min_sentiment = self.sentiment_threshold
        min_posts = self.min_posts_for_signal
        min_quality = self.quality_threshold
        min_overall_quality = 0.5  # Higher bar for Reddit signals
        
        # Check all criteria
        overall_quality = self.get_signal_quality_score(signal)
        
        return (
            signal.confidence >= min_confidence and
            abs(signal.sentiment_score) >= min_sentiment and
            signal.post_count >= min_posts and
            signal.quality_score >= min_quality and
            overall_quality >= min_overall_quality
        )
        
    def get_signal_summary(self, signal: RedditSentimentSignal) -> Dict[str, Any]:
        """Get human-readable summary of Reddit sentiment signal"""
        if not signal:
            return {}
            
        # Determine sentiment direction and strength
        if signal.sentiment_score > 0.3:
            sentiment_desc = "Strongly Bullish"
        elif signal.sentiment_score > 0.1:
            sentiment_desc = "Moderately Bullish"
        elif signal.sentiment_score < -0.3:
            sentiment_desc = "Strongly Bearish"
        elif signal.sentiment_score < -0.1:
            sentiment_desc = "Moderately Bearish"
        else:
            sentiment_desc = "Neutral"
            
        # Top subreddits
        top_subreddits = sorted(signal.subreddit_distribution.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'symbol': signal.symbol,
            'sentiment': sentiment_desc,
            'sentiment_score': round(signal.sentiment_score, 3),
            'confidence': round(signal.confidence, 3),
            'quality_score': round(signal.quality_score, 3),
            'overall_quality': round(self.get_signal_quality_score(signal), 3),
            'is_actionable': self.is_signal_actionable(signal),
            'post_count': signal.post_count,
            'comment_count': signal.comment_count,
            'timeframe': signal.timeframe,
            'top_subreddits': [f"r/{sub} ({count} posts)" for sub, count in top_subreddits],
            'top_keywords': signal.keywords[:8],  # Top 8 keywords
            'sentiment_breakdown': signal.sentiment_distribution
        }
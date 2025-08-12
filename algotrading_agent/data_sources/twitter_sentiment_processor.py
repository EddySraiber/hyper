"""
Twitter Sentiment Processing for Financial Analysis
Processes tweets to extract financial sentiment and trading signals
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from textblob import TextBlob

from .twitter_client import TwitterTweet, preprocess_tweet_text, calculate_tweet_influence_score


@dataclass
class TwitterSentimentSignal:
    """Sentiment signal extracted from Twitter data"""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float      # 0.0 to 1.0
    volume_score: float    # 0.0 to 1.0 (based on tweet volume)
    influence_score: float # 0.0 to 1.0 (based on engagement/reach)
    tweet_count: int
    timeframe: str         # e.g., "1h", "4h", "24h"
    created_at: str
    top_tweets: List[Dict[str, Any]]  # Sample of most influential tweets
    keywords: List[str]    # Financial keywords mentioned
    sentiment_distribution: Dict[str, int]  # bullish, bearish, neutral counts


class TwitterSentimentProcessor:
    """
    Processes Twitter data to extract actionable financial sentiment signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('twitter_sentiment', {})
        self.logger = logging.getLogger(__name__)
        
        # Sentiment analysis configuration
        self.min_tweets_for_signal = self.config.get('min_tweets_for_signal', 5)
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.1)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        
        # Time-based analysis windows
        self.analysis_windows = {
            '1h': 1,
            '4h': 4, 
            '12h': 12,
            '24h': 24
        }
        
        # Financial sentiment keywords with weights
        self.sentiment_keywords = {
            'bullish': {
                'moon': 1.0, 'rocket': 0.9, 'pump': 0.8, 'surge': 0.7, 'rally': 0.7,
                'bullish': 0.9, 'buy': 0.6, 'long': 0.6, 'breakout': 0.8, 'bull': 0.7,
                'strong': 0.5, 'up': 0.4, 'rising': 0.5, 'gain': 0.6, 'profit': 0.7,
                'beat': 0.8, 'exceed': 0.7, 'outperform': 0.6, 'upgrade': 0.7
            },
            'bearish': {
                'crash': -1.0, 'dump': -0.9, 'plunge': -0.8, 'tank': -0.8, 'collapse': -0.9,
                'bearish': -0.9, 'sell': -0.6, 'short': -0.6, 'breakdown': -0.8, 'bear': -0.7,
                'weak': -0.5, 'down': -0.4, 'falling': -0.5, 'loss': -0.6, 'miss': -0.8,
                'disappointing': -0.7, 'concern': -0.5, 'risk': -0.4, 'downgrade': -0.7
            }
        }
        
        # Emoji sentiment mapping
        self.emoji_sentiment = {
            '🚀': 0.8, '🌙': 0.7, '📈': 0.6, '💎': 0.5, '🔥': 0.4, '💪': 0.4,
            '📉': -0.6, '💀': -0.8, '⚠️': -0.4, '😱': -0.7, '🔻': -0.5, '💩': -0.9
        }
        
    async def process_symbol_sentiment(self, symbol: str, tweets: List[TwitterTweet], 
                                     timeframe: str = "24h") -> Optional[TwitterSentimentSignal]:
        """
        Process tweets for a specific symbol to generate sentiment signal
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            tweets: List of tweets related to the symbol
            timeframe: Analysis timeframe (1h, 4h, 12h, 24h)
            
        Returns:
            TwitterSentimentSignal or None if insufficient data
        """
        if not tweets or len(tweets) < self.min_tweets_for_signal:
            self.logger.debug(f"Insufficient tweets for {symbol}: {len(tweets)} < {self.min_tweets_for_signal}")
            return None
            
        # Filter tweets by timeframe
        filtered_tweets = self._filter_tweets_by_time(tweets, timeframe)
        if len(filtered_tweets) < self.min_tweets_for_signal:
            return None
            
        # Analyze sentiment for each tweet
        tweet_sentiments = []
        for tweet in filtered_tweets:
            sentiment_data = await self._analyze_tweet_sentiment(tweet, symbol)
            if sentiment_data:
                tweet_sentiments.append(sentiment_data)
                
        if not tweet_sentiments:
            return None
            
        # Aggregate sentiment signals
        signal = self._aggregate_sentiment_signals(symbol, tweet_sentiments, timeframe)
        
        # Add sample of most influential tweets
        signal.top_tweets = self._get_top_influential_tweets(tweet_sentiments, limit=5)
        
        return signal
        
    def _filter_tweets_by_time(self, tweets: List[TwitterTweet], timeframe: str) -> List[TwitterTweet]:
        """Filter tweets by specified timeframe"""
        hours_back = self.analysis_windows.get(timeframe, 24)
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        filtered = []
        for tweet in tweets:
            try:
                # Parse tweet timestamp
                tweet_time = datetime.fromisoformat(tweet.created_at.replace('Z', '+00:00'))
                if tweet_time.replace(tzinfo=None) >= cutoff_time:
                    filtered.append(tweet)
            except (ValueError, AttributeError):
                # If timestamp parsing fails, include the tweet
                filtered.append(tweet)
                
        return filtered
        
    async def _analyze_tweet_sentiment(self, tweet: TwitterTweet, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of individual tweet"""
        try:
            # Preprocess tweet text
            clean_text = preprocess_tweet_text(tweet.text)
            
            # Basic TextBlob sentiment
            blob = TextBlob(clean_text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Enhanced financial sentiment analysis
            financial_sentiment = self._analyze_financial_sentiment(tweet.text, symbol)
            
            # Combine sentiments with weighting
            combined_sentiment = (textblob_sentiment * 0.4) + (financial_sentiment * 0.6)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_sentiment_confidence(tweet, textblob_sentiment, financial_sentiment)
            
            # Calculate influence score
            influence_score = calculate_tweet_influence_score(tweet)
            
            return {
                'tweet': tweet,
                'clean_text': clean_text,
                'textblob_sentiment': textblob_sentiment,
                'financial_sentiment': financial_sentiment,
                'combined_sentiment': combined_sentiment,
                'confidence': confidence,
                'influence_score': influence_score,
                'engagement_score': self._calculate_engagement_score(tweet),
                'keywords': self._extract_financial_keywords(tweet.text),
                'emojis': self._extract_emoji_sentiment(tweet.text)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing tweet sentiment: {e}")
            return None
            
    def _analyze_financial_sentiment(self, text: str, symbol: str) -> float:
        """Analyze financial-specific sentiment in tweet text"""
        text_lower = text.lower()
        sentiment_score = 0.0
        keyword_count = 0
        
        # Check bullish keywords
        for keyword, weight in self.sentiment_keywords['bullish'].items():
            if keyword in text_lower:
                sentiment_score += weight
                keyword_count += 1
                
        # Check bearish keywords  
        for keyword, weight in self.sentiment_keywords['bearish'].items():
            if keyword in text_lower:
                sentiment_score += weight  # weight is already negative
                keyword_count += 1
                
        # Emoji sentiment
        emoji_score = sum(self.emoji_sentiment.get(char, 0) for char in text)
        
        # Combine and normalize
        if keyword_count > 0:
            sentiment_score = sentiment_score / max(keyword_count, 1)
        
        # Add emoji influence (weighted lower)
        total_sentiment = sentiment_score + (emoji_score * 0.3)
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, total_sentiment))
        
    def _calculate_sentiment_confidence(self, tweet: TwitterTweet, textblob_score: float, 
                                      financial_score: float) -> float:
        """Calculate confidence score for sentiment analysis"""
        # Base confidence from sentiment strength
        sentiment_strength = max(abs(textblob_score), abs(financial_score))
        
        # Agreement bonus (both methods agree on direction)
        agreement_bonus = 0.2 if (textblob_score * financial_score > 0) else 0
        
        # Engagement bonus (higher engagement = more reliable)
        metrics = tweet.public_metrics
        total_engagement = (
            metrics.get('retweet_count', 0) +
            metrics.get('like_count', 0) +
            metrics.get('reply_count', 0)
        )
        engagement_bonus = min(total_engagement / 100.0, 0.3)  # Cap at 0.3
        
        # Text length bonus (longer tweets often more thoughtful)
        text_length_bonus = min(len(tweet.text) / 280.0, 0.2)  # Cap at 0.2
        
        confidence = sentiment_strength + agreement_bonus + engagement_bonus + text_length_bonus
        return min(confidence, 1.0)
        
    def _calculate_engagement_score(self, tweet: TwitterTweet) -> float:
        """Calculate normalized engagement score (0.0 to 1.0)"""
        metrics = tweet.public_metrics
        
        # Weighted engagement calculation
        engagement = (
            metrics.get('retweet_count', 0) * 3 +     # Retweets most valuable
            metrics.get('like_count', 0) +             # Likes are basic engagement
            metrics.get('reply_count', 0) * 2 +        # Replies show discussion
            metrics.get('quote_count', 0) * 2          # Quotes show amplification
        )
        
        # Normalize to 0-1 scale (assuming max reasonable engagement is 10000)
        return min(engagement / 10000.0, 1.0)
        
    def _extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial keywords from tweet text"""
        text_lower = text.lower()
        found_keywords = []
        
        all_keywords = list(self.sentiment_keywords['bullish'].keys()) + \
                      list(self.sentiment_keywords['bearish'].keys())
        
        for keyword in all_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                
        return found_keywords
        
    def _extract_emoji_sentiment(self, text: str) -> Dict[str, float]:
        """Extract emoji sentiment from text"""
        emoji_data = {}
        for char in text:
            if char in self.emoji_sentiment:
                emoji_data[char] = self.emoji_sentiment[char]
        return emoji_data
        
    def _aggregate_sentiment_signals(self, symbol: str, tweet_sentiments: List[Dict[str, Any]], 
                                   timeframe: str) -> TwitterSentimentSignal:
        """Aggregate individual tweet sentiments into overall signal"""
        if not tweet_sentiments:
            return None
            
        # Calculate weighted average sentiment (weighted by influence and confidence)
        total_weighted_sentiment = 0.0
        total_weights = 0.0
        sentiment_distribution = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        all_keywords = []
        
        for sentiment_data in tweet_sentiments:
            sentiment = sentiment_data['combined_sentiment']
            confidence = sentiment_data['confidence']
            influence = sentiment_data['influence_score']
            
            # Weight = confidence * influence
            weight = confidence * max(influence, 0.1)  # Minimum weight of 0.1
            
            total_weighted_sentiment += sentiment * weight
            total_weights += weight
            
            # Update sentiment distribution
            if sentiment > 0.1:
                sentiment_distribution['bullish'] += 1
            elif sentiment < -0.1:
                sentiment_distribution['bearish'] += 1
            else:
                sentiment_distribution['neutral'] += 1
                
            # Collect keywords
            all_keywords.extend(sentiment_data.get('keywords', []))
            
        # Calculate final scores
        avg_sentiment = total_weighted_sentiment / max(total_weights, 1.0)
        avg_confidence = sum(s['confidence'] for s in tweet_sentiments) / len(tweet_sentiments)
        avg_influence = sum(s['influence_score'] for s in tweet_sentiments) / len(tweet_sentiments)
        
        # Volume score based on tweet count relative to symbol's typical activity
        volume_score = self._calculate_volume_score(len(tweet_sentiments), timeframe)
        
        return TwitterSentimentSignal(
            symbol=symbol,
            sentiment_score=avg_sentiment,
            confidence=avg_confidence,
            volume_score=volume_score,
            influence_score=avg_influence,
            tweet_count=len(tweet_sentiments),
            timeframe=timeframe,
            created_at=datetime.utcnow().isoformat(),
            top_tweets=[],  # Will be filled by caller
            keywords=list(set(all_keywords)),  # Deduplicate
            sentiment_distribution=sentiment_distribution
        )
        
    def _calculate_volume_score(self, tweet_count: int, timeframe: str) -> float:
        """Calculate volume score based on tweet count and timeframe"""
        # Expected tweet volumes by timeframe (rough estimates)
        expected_volumes = {
            '1h': 10,
            '4h': 30, 
            '12h': 80,
            '24h': 150
        }
        
        expected = expected_volumes.get(timeframe, 50)
        volume_ratio = tweet_count / expected
        
        # Normalize to 0-1 scale with diminishing returns
        return min(1.0, volume_ratio ** 0.7)
        
    def _get_top_influential_tweets(self, tweet_sentiments: List[Dict[str, Any]], 
                                  limit: int = 5) -> List[Dict[str, Any]]:
        """Get top influential tweets for the signal"""
        # Sort by influence score * confidence
        sorted_tweets = sorted(
            tweet_sentiments,
            key=lambda x: x['influence_score'] * x['confidence'],
            reverse=True
        )
        
        top_tweets = []
        for sentiment_data in sorted_tweets[:limit]:
            tweet = sentiment_data['tweet']
            top_tweets.append({
                'id': tweet.id,
                'text': tweet.text[:200] + '...' if len(tweet.text) > 200 else tweet.text,
                'username': tweet.username,
                'created_at': tweet.created_at,
                'sentiment': sentiment_data['combined_sentiment'],
                'confidence': sentiment_data['confidence'],
                'influence_score': sentiment_data['influence_score'],
                'engagement': tweet.public_metrics,
                'keywords': sentiment_data.get('keywords', [])
            })
            
        return top_tweets
        
    def get_signal_quality_score(self, signal: TwitterSentimentSignal) -> float:
        """
        Calculate overall quality score for the sentiment signal
        
        Returns:
            Quality score from 0.0 to 1.0
        """
        if not signal:
            return 0.0
            
        # Components of quality score
        sentiment_strength = abs(signal.sentiment_score)  # 0.0 - 1.0
        confidence_score = signal.confidence              # 0.0 - 1.0  
        volume_score = signal.volume_score               # 0.0 - 1.0
        influence_score = signal.influence_score         # 0.0 - 1.0
        
        # Tweet count bonus (more tweets = more reliable)
        count_bonus = min(signal.tweet_count / 50.0, 0.3)  # Cap at 0.3
        
        # Sentiment consensus bonus (clear direction is better)
        total_directional = signal.sentiment_distribution['bullish'] + signal.sentiment_distribution['bearish']
        consensus_ratio = max(signal.sentiment_distribution.values()) / max(total_directional, 1)
        consensus_bonus = (consensus_ratio - 0.5) * 0.4  # 0 to 0.2 bonus
        
        # Weighted combination
        quality_score = (
            sentiment_strength * 0.3 +
            confidence_score * 0.25 +
            volume_score * 0.2 +
            influence_score * 0.15 +
            count_bonus * 0.1 +
            max(0, consensus_bonus)  # Only positive consensus bonus
        )
        
        return min(quality_score, 1.0)
        
    def is_signal_actionable(self, signal: TwitterSentimentSignal) -> bool:
        """
        Determine if signal is strong enough to be actionable for trading
        
        Returns:
            True if signal meets actionability criteria
        """
        if not signal:
            return False
            
        # Minimum criteria
        min_confidence = self.confidence_threshold
        min_sentiment = self.sentiment_threshold
        min_tweets = self.min_tweets_for_signal
        min_quality = 0.4
        
        # Check all criteria
        quality_score = self.get_signal_quality_score(signal)
        
        return (
            signal.confidence >= min_confidence and
            abs(signal.sentiment_score) >= min_sentiment and
            signal.tweet_count >= min_tweets and
            quality_score >= min_quality
        )
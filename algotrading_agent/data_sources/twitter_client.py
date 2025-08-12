"""
Twitter Client for Financial Sentiment Data Collection
Uses Twitter API v2 with free tier limitations (1,500 tweets/month)
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp
import logging
from urllib.parse import urlencode


@dataclass
class TwitterTweet:
    """Data structure for Twitter tweet"""
    id: str
    text: str
    author_id: str
    username: str = ""
    created_at: str = ""
    public_metrics: Dict[str, int] = None
    context_annotations: List[Dict] = None
    entities: Dict = None
    
    def __post_init__(self):
        if self.public_metrics is None:
            self.public_metrics = {}
        if self.context_annotations is None:
            self.context_annotations = []
        if self.entities is None:
            self.entities = {}


@dataclass 
class TwitterRateLimit:
    """Rate limit tracking for Twitter API"""
    remaining: int
    reset_time: int
    limit: int
    
    def can_make_request(self) -> bool:
        return self.remaining > 0 or time.time() > self.reset_time
    
    def time_until_reset(self) -> int:
        return max(0, self.reset_time - int(time.time()))


class TwitterClient:
    """
    Twitter API v2 client optimized for financial sentiment collection
    Features:
    - Free tier rate limiting (1,500 tweets/month)
    - Financial keyword search
    - Symbol-specific tweet collection
    - Engagement metrics collection
    - Error handling and retry logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('twitter', {})
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.bearer_token = self.config.get('bearer_token', '')
        self.base_url = "https://api.twitter.com/2"
        
        # Rate limiting (Twitter API v2 free tier)
        self.rate_limits = {
            'tweets/search/recent': TwitterRateLimit(remaining=180, reset_time=0, limit=180),  # Per 15 min
            'users/by': TwitterRateLimit(remaining=300, reset_time=0, limit=300),  # Per 15 min  
        }
        
        # Search configuration
        self.max_results_per_query = min(self.config.get('max_results_per_query', 10), 100)
        self.search_operators = self.config.get('search_operators', {
            'exclude_retweets': True,
            'exclude_replies': True,
            'language': 'en',
            'min_retweets': self.config.get('min_retweets', 2),
            'min_likes': self.config.get('min_likes', 5)
        })
        
        # Financial keywords for enhanced filtering
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'beat', 'miss', 'guidance',
            'acquisition', 'merger', 'deal', 'partnership', 'buyout',
            'bullish', 'bearish', 'rally', 'crash', 'surge', 'plunge',
            'breakthrough', 'approval', 'fda', 'patent', 'lawsuit',
            'dividend', 'split', 'buyback', 'ipo', 'listing'
        ]
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self) -> None:
        """Initialize the Twitter client"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'Authorization': f'Bearer {self.bearer_token}',
                'User-Agent': 'AlgoTradingAgent/1.0.0'
            }
        )
        
        # Validate API credentials
        if await self._validate_credentials():
            self.logger.info("Twitter client initialized successfully")
        else:
            self.logger.error("Twitter API credentials validation failed")
            
    async def stop(self) -> None:
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def _validate_credentials(self) -> bool:
        """Validate Twitter API credentials"""
        try:
            url = f"{self.base_url}/users/me"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return True
                else:
                    self.logger.error(f"Twitter API validation failed: {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Twitter API validation error: {e}")
            return False
            
    async def search_financial_tweets(self, symbols: List[str], hours_back: int = 24) -> List[TwitterTweet]:
        """
        Search for financial tweets related to specified symbols
        
        Args:
            symbols: List of stock symbols to search for (e.g., ['AAPL', 'TSLA'])
            hours_back: How many hours back to search (default: 24)
            
        Returns:
            List of TwitterTweet objects
        """
        if not self.session or not symbols:
            return []
            
        all_tweets = []
        
        for symbol in symbols:
            try:
                tweets = await self._search_symbol_tweets(symbol, hours_back)
                all_tweets.extend(tweets)
                
                # Respect rate limits
                await self._handle_rate_limit('tweets/search/recent')
                
            except Exception as e:
                self.logger.error(f"Error searching tweets for {symbol}: {e}")
                continue
                
        return self._deduplicate_tweets(all_tweets)
        
    async def _search_symbol_tweets(self, symbol: str, hours_back: int) -> List[TwitterTweet]:
        """Search tweets for a specific symbol"""
        # Build search query with financial context
        query_parts = [
            f"${symbol}",  # Cashtag
            f"{symbol}",   # Symbol mention
        ]
        
        # Add financial keywords to improve relevance
        financial_context = " OR ".join([
            f"({symbol} {keyword})" for keyword in self.financial_keywords[:5]  # Limit to avoid query length
        ])
        
        # Combine symbol search with financial context
        query = f"({' OR '.join(query_parts)}) OR ({financial_context})"
        
        # Add search operators
        if self.search_operators.get('exclude_retweets'):
            query += " -is:retweet"
        if self.search_operators.get('exclude_replies'):
            query += " -is:reply"
        if self.search_operators.get('language'):
            query += f" lang:{self.search_operators['language']}"
            
        # Time range
        start_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat() + "Z"
        
        # Build API request
        params = {
            'query': query,
            'max_results': self.max_results_per_query,
            'start_time': start_time,
            'tweet.fields': 'created_at,author_id,public_metrics,context_annotations,entities',
            'user.fields': 'username,verified,public_metrics',
            'expansions': 'author_id'
        }
        
        try:
            url = f"{self.base_url}/tweets/search/recent"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_tweet_response(data)
                elif response.status == 429:
                    # Rate limited
                    self._update_rate_limit('tweets/search/recent', response.headers)
                    self.logger.warning(f"Twitter API rate limited for symbol {symbol}")
                    return []
                else:
                    self.logger.error(f"Twitter API error for {symbol}: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error searching tweets for {symbol}: {e}")
            return []
            
    def _parse_tweet_response(self, data: Dict) -> List[TwitterTweet]:
        """Parse Twitter API response into TwitterTweet objects"""
        tweets = []
        
        # Get tweets and users data
        tweet_data = data.get('data', [])
        includes = data.get('includes', {})
        users_data = {user['id']: user for user in includes.get('users', [])}
        
        for tweet_info in tweet_data:
            try:
                author_id = tweet_info.get('author_id', '')
                user_info = users_data.get(author_id, {})
                
                tweet = TwitterTweet(
                    id=tweet_info['id'],
                    text=tweet_info['text'],
                    author_id=author_id,
                    username=user_info.get('username', ''),
                    created_at=tweet_info.get('created_at', ''),
                    public_metrics=tweet_info.get('public_metrics', {}),
                    context_annotations=tweet_info.get('context_annotations', []),
                    entities=tweet_info.get('entities', {})
                )
                
                # Filter by engagement if specified
                if self._meets_engagement_criteria(tweet):
                    tweets.append(tweet)
                    
            except Exception as e:
                self.logger.error(f"Error parsing tweet: {e}")
                continue
                
        return tweets
        
    def _meets_engagement_criteria(self, tweet: TwitterTweet) -> bool:
        """Check if tweet meets minimum engagement criteria"""
        metrics = tweet.public_metrics
        
        min_retweets = self.search_operators.get('min_retweets', 0)
        min_likes = self.search_operators.get('min_likes', 0)
        
        retweet_count = metrics.get('retweet_count', 0)
        like_count = metrics.get('like_count', 0)
        
        return retweet_count >= min_retweets and like_count >= min_likes
        
    def _deduplicate_tweets(self, tweets: List[TwitterTweet]) -> List[TwitterTweet]:
        """Remove duplicate tweets by ID"""
        seen_ids = set()
        deduplicated = []
        
        for tweet in tweets:
            if tweet.id not in seen_ids:
                seen_ids.add(tweet.id)
                deduplicated.append(tweet)
                
        return deduplicated
        
    async def _handle_rate_limit(self, endpoint: str) -> None:
        """Handle rate limiting for specific endpoint"""
        rate_limit = self.rate_limits.get(endpoint)
        if not rate_limit:
            return
            
        if not rate_limit.can_make_request():
            wait_time = rate_limit.time_until_reset()
            if wait_time > 0:
                self.logger.info(f"Rate limited. Waiting {wait_time} seconds for {endpoint}")
                await asyncio.sleep(min(wait_time + 1, 60))  # Cap wait time to 1 minute
                
    def _update_rate_limit(self, endpoint: str, headers: Dict) -> None:
        """Update rate limit info from response headers"""
        try:
            remaining = int(headers.get('x-rate-limit-remaining', 0))
            reset_time = int(headers.get('x-rate-limit-reset', 0))
            limit = int(headers.get('x-rate-limit-limit', 0))
            
            self.rate_limits[endpoint] = TwitterRateLimit(
                remaining=remaining,
                reset_time=reset_time,
                limit=limit
            )
        except (ValueError, TypeError):
            pass
            
    def extract_financial_context(self, tweet: TwitterTweet) -> Dict[str, Any]:
        """
        Extract financial context and sentiment indicators from tweet
        
        Args:
            tweet: TwitterTweet object
            
        Returns:
            Dictionary containing financial context analysis
        """
        text = tweet.text.lower()
        
        # Extract mentioned symbols (cashtags)
        symbols = re.findall(r'\$([A-Z]{1,5})\b', tweet.text)
        
        # Extract financial keywords
        mentioned_keywords = [kw for kw in self.financial_keywords if kw in text]
        
        # Basic sentiment indicators
        bullish_words = ['bullish', 'bull', 'buy', 'long', 'up', 'moon', 'rocket', 'surge', 'rally']
        bearish_words = ['bearish', 'bear', 'sell', 'short', 'down', 'crash', 'dump', 'plunge', 'drop']
        
        bullish_count = sum(1 for word in bullish_words if word in text)
        bearish_count = sum(1 for word in bearish_words if word in text)
        
        # Engagement score (normalized)
        metrics = tweet.public_metrics
        engagement_score = (
            metrics.get('retweet_count', 0) * 2 +  # Retweets are more valuable
            metrics.get('like_count', 0) +
            metrics.get('reply_count', 0) +
            metrics.get('quote_count', 0) * 1.5
        )
        
        return {
            'symbols': symbols,
            'financial_keywords': mentioned_keywords,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'engagement_score': engagement_score,
            'sentiment_direction': 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral',
            'confidence_score': min(abs(bullish_count - bearish_count) / max(bullish_count + bearish_count, 1), 1.0),
            'created_at': tweet.created_at,
            'username': tweet.username,
            'tweet_id': tweet.id
        }
        
    async def get_trending_financial_topics(self, location_id: int = 1) -> List[Dict[str, Any]]:
        """
        Get trending topics with financial relevance
        Note: This requires Twitter API v1.1 which may not be available in free tier
        """
        self.logger.info("Trending topics feature requires higher Twitter API access")
        return []
        
    def get_rate_limit_status(self) -> Dict[str, Dict]:
        """Get current rate limit status for all endpoints"""
        status = {}
        for endpoint, rate_limit in self.rate_limits.items():
            status[endpoint] = {
                'remaining': rate_limit.remaining,
                'limit': rate_limit.limit,
                'reset_time': rate_limit.reset_time,
                'time_until_reset': rate_limit.time_until_reset(),
                'can_make_request': rate_limit.can_make_request()
            }
        return status


# Utility functions for tweet processing
def preprocess_tweet_text(text: str) -> str:
    """
    Preprocess tweet text for sentiment analysis
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text suitable for analysis
    """
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions (but keep the context)
    text = re.sub(r'@[\w]+', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    # Remove excessive emojis/special chars while keeping financial symbols
    text = re.sub(r'[^\w\s$#.!?-]', ' ', text)
    
    return text.strip()


def calculate_tweet_influence_score(tweet: TwitterTweet, user_metrics: Dict[str, int] = None) -> float:
    """
    Calculate influence score for a tweet based on engagement and user metrics
    
    Args:
        tweet: TwitterTweet object
        user_metrics: Optional user metrics (followers, etc.)
        
    Returns:
        Influence score (0.0 to 1.0)
    """
    if user_metrics is None:
        user_metrics = {}
        
    # Tweet engagement
    metrics = tweet.public_metrics
    engagement = (
        metrics.get('retweet_count', 0) * 3 +
        metrics.get('like_count', 0) +
        metrics.get('reply_count', 0) * 2 +
        metrics.get('quote_count', 0) * 2
    )
    
    # User influence (if available)
    user_followers = user_metrics.get('followers_count', 100)  # Default assumption
    
    # Normalize scores
    engagement_score = min(engagement / 1000.0, 1.0)  # Cap at 1000 engagements = max score
    user_score = min(user_followers / 100000.0, 1.0)  # Cap at 100k followers = max score
    
    # Weighted combination (60% engagement, 40% user influence)
    influence_score = (engagement_score * 0.6) + (user_score * 0.4)
    
    return min(influence_score, 1.0)
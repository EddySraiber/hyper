"""
Reddit Client for Financial Sentiment Data Collection
Uses Reddit's free API with rate limiting (100 requests/minute, 1000/hour)
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
class RedditPost:
    """Data structure for Reddit post"""
    id: str
    title: str
    text: str
    author: str
    subreddit: str
    created_at: str
    score: int
    upvote_ratio: float
    num_comments: int
    url: str
    flair: str = ""
    is_self: bool = True
    permalink: str = ""
    
    def __post_init__(self):
        if not self.permalink and self.url:
            self.permalink = f"/r/{self.subreddit}/comments/{self.id}/"


@dataclass
class RedditComment:
    """Data structure for Reddit comment"""
    id: str
    text: str
    author: str
    post_id: str
    parent_id: str
    created_at: str
    score: int
    depth: int = 0
    is_submitter: bool = False


@dataclass 
class RedditRateLimit:
    """Rate limit tracking for Reddit API"""
    remaining: int
    reset_time: int
    limit: int
    
    def can_make_request(self) -> bool:
        return self.remaining > 0 or time.time() > self.reset_time
    
    def time_until_reset(self) -> int:
        return max(0, self.reset_time - int(time.time()))


class RedditClient:
    """
    Reddit API client optimized for financial sentiment collection
    Features:
    - Free tier rate limiting (100 requests/minute, 1000/hour)
    - Financial subreddit monitoring  
    - Quality post filtering based on karma, upvote ratio
    - Comment analysis for deeper sentiment
    - Error handling and retry logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('reddit', {})
        self.logger = logging.getLogger(__name__)
        
        # API configuration (uses public Reddit JSON API - no authentication needed)
        self.base_url = "https://www.reddit.com"
        self.user_agent = self.config.get('user_agent', 'AlgoTradingAgent/1.0.0 (Financial Sentiment Analysis)')
        
        # Rate limiting (Reddit's informal limits for public API)
        self.rate_limits = {
            'posts': RedditRateLimit(remaining=100, reset_time=0, limit=100),  # Per minute
            'comments': RedditRateLimit(remaining=60, reset_time=0, limit=60),  # Per minute
        }
        
        # Request configuration
        self.max_posts_per_subreddit = self.config.get('max_posts_per_subreddit', 25)
        self.max_comments_per_post = self.config.get('max_comments_per_post', 10)
        self.request_delay = self.config.get('request_delay', 1)  # 1 second between requests
        
        # Quality filters
        self.min_score = self.config.get('min_score', 10)
        self.min_upvote_ratio = self.config.get('min_upvote_ratio', 0.7)
        self.min_comments = self.config.get('min_comments', 5)
        self.max_post_age_hours = self.config.get('max_post_age_hours', 48)
        
        # Financial subreddits (high quality discussion-focused)
        self.financial_subreddits = self.config.get('subreddits', [
            'investing',           # High-quality investment discussions
            'SecurityAnalysis',    # Fundamental analysis community
            'ValueInvesting',      # Long-term value investing focus
            'stocks',             # General stock discussions
            'StockMarket',        # Market analysis and news
            'financialindependence', # Long-term financial planning
            'dividends',          # Dividend investing community
            'options',            # Options trading (more sophisticated than WSB)
        ])
        
        # Financial keywords for enhanced filtering
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'beat', 'miss', 'guidance',
            'acquisition', 'merger', 'deal', 'partnership', 'buyout',
            'bullish', 'bearish', 'rally', 'correction', 'crash', 'surge',
            'breakthrough', 'approval', 'fda', 'patent', 'lawsuit',
            'dividend', 'split', 'buyback', 'ipo', 'listing', 'valuation',
            'pe ratio', 'dcf', 'free cash flow', 'debt', 'balance sheet',
            'quarterly', 'annual', 'forecast', 'target price', 'upgrade', 'downgrade'
        ]
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        
    async def start(self) -> None:
        """Initialize the Reddit client"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
        )
        
        # Test connection
        if await self._test_connection():
            self.logger.info("Reddit client initialized successfully")
        else:
            self.logger.error("Reddit API connection test failed")
            
    async def stop(self) -> None:
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def _test_connection(self) -> bool:
        """Test Reddit API connection"""
        try:
            url = f"{self.base_url}/r/investing/hot.json?limit=1"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return 'data' in data and 'children' in data['data']
                else:
                    self.logger.error(f"Reddit API test failed: {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Reddit API test error: {e}")
            return False
            
    async def get_financial_posts(self, symbols: List[str], hours_back: int = 24) -> List[RedditPost]:
        """
        Get financial posts related to specified symbols from monitored subreddits
        
        Args:
            symbols: List of stock symbols to search for (e.g., ['AAPL', 'TSLA'])
            hours_back: How many hours back to search (default: 24)
            
        Returns:
            List of RedditPost objects
        """
        if not self.session or not symbols:
            return []
            
        all_posts = []
        
        for subreddit in self.financial_subreddits:
            try:
                posts = await self._get_subreddit_posts(subreddit, symbols, hours_back)
                all_posts.extend(posts)
                
                # Respect rate limits
                await self._handle_rate_limit('posts')
                
            except Exception as e:
                self.logger.error(f"Error getting posts from r/{subreddit}: {e}")
                continue
                
        return self._deduplicate_posts(all_posts)
        
    async def _get_subreddit_posts(self, subreddit: str, symbols: List[str], 
                                 hours_back: int) -> List[RedditPost]:
        """Get posts from a specific subreddit"""
        posts = []
        
        # Get hot and new posts for comprehensive coverage
        for sort_type in ['hot', 'new']:
            try:
                subreddit_posts = await self._fetch_subreddit_posts(subreddit, sort_type)
                
                # Filter posts by symbols, quality, and time
                filtered_posts = self._filter_posts(subreddit_posts, symbols, hours_back)
                posts.extend(filtered_posts)
                
                await self._rate_limit_delay()
                
            except Exception as e:
                self.logger.error(f"Error fetching {sort_type} posts from r/{subreddit}: {e}")
                continue
                
        return posts
        
    async def _fetch_subreddit_posts(self, subreddit: str, sort_type: str) -> List[Dict]:
        """Fetch raw posts from Reddit API"""
        url = f"{self.base_url}/r/{subreddit}/{sort_type}.json"
        params = {
            'limit': self.max_posts_per_subreddit,
            'raw_json': 1  # Avoid HTML encoding
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [child['data'] for child in data.get('data', {}).get('children', [])]
            elif response.status == 429:
                # Rate limited
                self._update_rate_limit('posts', response.headers)
                self.logger.warning(f"Reddit API rate limited for r/{subreddit}")
                return []
            else:
                self.logger.error(f"Reddit API error for r/{subreddit}: {response.status}")
                return []
                
    def _filter_posts(self, raw_posts: List[Dict], symbols: List[str], 
                     hours_back: int) -> List[RedditPost]:
        """Filter posts by relevance, quality, and time"""
        filtered_posts = []
        cutoff_time = time.time() - (hours_back * 3600)
        
        for post_data in raw_posts:
            try:
                # Create RedditPost object
                post = self._parse_post_data(post_data)
                if not post:
                    continue
                    
                # Time filter
                post_time = datetime.fromisoformat(post.created_at.replace('Z', '+00:00')).timestamp()
                if post_time < cutoff_time:
                    continue
                    
                # Quality filters
                if not self._meets_quality_criteria(post):
                    continue
                    
                # Symbol relevance filter
                if self._is_symbol_relevant(post, symbols):
                    filtered_posts.append(post)
                    
            except Exception as e:
                self.logger.error(f"Error filtering post: {e}")
                continue
                
        return filtered_posts
        
    def _parse_post_data(self, post_data: Dict) -> Optional[RedditPost]:
        """Parse raw Reddit post data into RedditPost object"""
        try:
            # Skip deleted/removed posts
            if post_data.get('removed_by') or post_data.get('author') == '[deleted]':
                return None
                
            # Extract text content
            text = post_data.get('selftext', '')
            if not text and post_data.get('is_self', True):
                text = post_data.get('title', '')  # Use title if no selftext
                
            created_timestamp = post_data.get('created_utc', 0)
            created_at = datetime.fromtimestamp(created_timestamp).isoformat() + 'Z'
            
            post = RedditPost(
                id=post_data['id'],
                title=post_data.get('title', ''),
                text=text,
                author=post_data.get('author', ''),
                subreddit=post_data.get('subreddit', ''),
                created_at=created_at,
                score=post_data.get('score', 0),
                upvote_ratio=post_data.get('upvote_ratio', 0.5),
                num_comments=post_data.get('num_comments', 0),
                url=post_data.get('url', ''),
                flair=post_data.get('link_flair_text') or '',
                is_self=post_data.get('is_self', True),
                permalink=post_data.get('permalink', '')
            )
            
            return post
            
        except Exception as e:
            self.logger.error(f"Error parsing post data: {e}")
            return None
            
    def _meets_quality_criteria(self, post: RedditPost) -> bool:
        """Check if post meets quality criteria"""
        return (
            post.score >= self.min_score and
            post.upvote_ratio >= self.min_upvote_ratio and
            post.num_comments >= self.min_comments and
            len(post.text) >= 50  # Minimum text length for meaningful content
        )
        
    def _is_symbol_relevant(self, post: RedditPost, symbols: List[str]) -> bool:
        """Check if post is relevant to any of the specified symbols"""
        # Combine title and text for searching
        full_text = f"{post.title} {post.text}".lower()
        
        # Check for direct symbol mentions
        for symbol in symbols:
            symbol_patterns = [
                f"${symbol}",       # Cashtag
                f" {symbol} ",      # Space-bounded symbol
                f"({symbol})",      # Parentheses-bounded
                f"{symbol.lower()}"  # Lowercase mention
            ]
            
            if any(pattern.lower() in full_text for pattern in symbol_patterns):
                return True
                
        # Check for financial keywords to catch broader relevant discussions
        financial_keywords_found = sum(1 for keyword in self.financial_keywords 
                                     if keyword in full_text)
        
        # Require at least 2 financial keywords for non-symbol posts
        return financial_keywords_found >= 2
        
    async def get_post_comments(self, post: RedditPost, max_comments: int = None) -> List[RedditComment]:
        """
        Get comments for a specific post
        
        Args:
            post: RedditPost object
            max_comments: Maximum number of comments to retrieve
            
        Returns:
            List of RedditComment objects
        """
        if not self.session:
            return []
            
        max_comments = max_comments or self.max_comments_per_post
        
        try:
            url = f"{self.base_url}{post.permalink}.json"
            params = {'limit': max_comments, 'sort': 'top'}
            
            await self._rate_limit_delay()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_comments(data, post.id)
                elif response.status == 429:
                    self._update_rate_limit('comments', response.headers)
                    self.logger.warning(f"Rate limited getting comments for post {post.id}")
                    return []
                else:
                    self.logger.error(f"Error getting comments for post {post.id}: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error getting comments for post {post.id}: {e}")
            return []
            
    def _parse_comments(self, data: List[Dict], post_id: str) -> List[RedditComment]:
        """Parse Reddit comments from API response"""
        comments = []
        
        try:
            # Comments are in the second element of the response
            if len(data) < 2 or 'data' not in data[1]:
                return comments
                
            comment_data = data[1]['data']['children']
            
            for comment_item in comment_data:
                comment_info = comment_item.get('data', {})
                
                # Skip deleted/removed comments
                if (comment_info.get('author') == '[deleted]' or 
                    comment_info.get('body') == '[removed]'):
                    continue
                    
                try:
                    created_timestamp = comment_info.get('created_utc', 0)
                    created_at = datetime.fromtimestamp(created_timestamp).isoformat() + 'Z'
                    
                    comment = RedditComment(
                        id=comment_info['id'],
                        text=comment_info.get('body', ''),
                        author=comment_info.get('author', ''),
                        post_id=post_id,
                        parent_id=comment_info.get('parent_id', ''),
                        created_at=created_at,
                        score=comment_info.get('score', 0),
                        depth=comment_info.get('depth', 0),
                        is_submitter=comment_info.get('is_submitter', False)
                    )
                    
                    # Filter by quality (minimum score and length)
                    if comment.score >= 1 and len(comment.text) >= 20:
                        comments.append(comment)
                        
                except Exception as e:
                    self.logger.error(f"Error parsing comment: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error parsing comments: {e}")
            
        return comments
        
    def _deduplicate_posts(self, posts: List[RedditPost]) -> List[RedditPost]:
        """Remove duplicate posts by ID"""
        seen_ids = set()
        deduplicated = []
        
        for post in posts:
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                deduplicated.append(post)
                
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
                await asyncio.sleep(min(wait_time + 1, 120))  # Cap wait time to 2 minutes
                
    async def _rate_limit_delay(self) -> None:
        """Add delay between requests to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            delay = self.request_delay - time_since_last
            await asyncio.sleep(delay)
            
        self.last_request_time = time.time()
        
    def _update_rate_limit(self, endpoint: str, headers: Dict) -> None:
        """Update rate limit info from response headers"""
        # Reddit doesn't always provide rate limit headers in public API
        # Use conservative estimates
        current_time = int(time.time())
        
        if endpoint == 'posts':
            self.rate_limits[endpoint] = RedditRateLimit(
                remaining=max(0, self.rate_limits[endpoint].remaining - 1),
                reset_time=current_time + 60,  # Reset every minute
                limit=100
            )
        elif endpoint == 'comments':
            self.rate_limits[endpoint] = RedditRateLimit(
                remaining=max(0, self.rate_limits[endpoint].remaining - 1),
                reset_time=current_time + 60,
                limit=60
            )
            
    def extract_financial_context(self, post: RedditPost, comments: List[RedditComment] = None) -> Dict[str, Any]:
        """
        Extract financial context and sentiment indicators from post and comments
        
        Args:
            post: RedditPost object
            comments: Optional list of comments
            
        Returns:
            Dictionary containing financial context analysis
        """
        # Combine post content
        full_text = f"{post.title} {post.text}".lower()
        
        if comments:
            # Add high-quality comments to analysis
            comment_text = " ".join([c.text for c in comments[:5] if c.score > 5])
            full_text += " " + comment_text.lower()
            
        # Extract mentioned symbols
        symbols = re.findall(r'\$([A-Z]{2,5})\b', post.title + " " + post.text)
        
        # Extract financial keywords
        mentioned_keywords = [kw for kw in self.financial_keywords if kw in full_text]
        
        # Basic sentiment indicators
        bullish_words = ['bullish', 'bull', 'buy', 'long', 'undervalued', 'opportunity', 
                        'growth', 'strong', 'beat', 'exceed', 'rally', 'moon']
        bearish_words = ['bearish', 'bear', 'sell', 'short', 'overvalued', 'risk',
                        'decline', 'weak', 'miss', 'disappointing', 'crash', 'dump']
        
        bullish_count = sum(1 for word in bullish_words if word in full_text)
        bearish_count = sum(1 for word in bearish_words if word in full_text)
        
        # Quality score based on Reddit metrics
        quality_score = (
            min(post.score / 100.0, 1.0) * 0.4 +           # Normalized score
            post.upvote_ratio * 0.3 +                       # Upvote ratio
            min(post.num_comments / 50.0, 1.0) * 0.2 +     # Comment engagement
            min(len(post.text) / 1000.0, 1.0) * 0.1        # Content length
        )
        
        return {
            'symbols': symbols,
            'financial_keywords': mentioned_keywords,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'quality_score': quality_score,
            'sentiment_direction': ('bullish' if bullish_count > bearish_count 
                                  else 'bearish' if bearish_count > bullish_count 
                                  else 'neutral'),
            'confidence_score': min(abs(bullish_count - bearish_count) / 
                                  max(bullish_count + bearish_count, 1), 1.0),
            'created_at': post.created_at,
            'subreddit': post.subreddit,
            'post_id': post.id,
            'score': post.score,
            'upvote_ratio': post.upvote_ratio,
            'comment_count': post.num_comments
        }
        
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


# Utility functions for Reddit content processing
def preprocess_reddit_text(text: str) -> str:
    """
    Preprocess Reddit text for sentiment analysis
    
    Args:
        text: Raw Reddit text (post or comment)
        
    Returns:
        Cleaned text suitable for analysis
    """
    # Remove Reddit markdown
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove user mentions
    text = re.sub(r'/?u/[\w-]+', '', text)
    text = re.sub(r'/?r/[\w-]+', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    # Remove excessive special characters while keeping financial symbols
    text = re.sub(r'[^\w\s$#.!?%-]', ' ', text)
    
    return text.strip()


def calculate_reddit_influence_score(post: RedditPost, comments: List[RedditComment] = None) -> float:
    """
    Calculate influence score for a Reddit post based on engagement and quality
    
    Args:
        post: RedditPost object
        comments: Optional list of comments
        
    Returns:
        Influence score (0.0 to 1.0)
    """
    # Base score from post metrics
    score_component = min(post.score / 1000.0, 1.0)  # Normalize to 1000 upvotes = max
    ratio_component = post.upvote_ratio  # Already 0.0 to 1.0
    comments_component = min(post.num_comments / 100.0, 1.0)  # Normalize to 100 comments = max
    
    # Text quality component
    text_length = len(post.title) + len(post.text)
    text_quality = min(text_length / 500.0, 1.0)  # Normalize to 500 chars = max
    
    # Comment quality bonus if available
    comment_bonus = 0.0
    if comments:
        avg_comment_score = sum(c.score for c in comments) / len(comments)
        comment_bonus = min(avg_comment_score / 10.0, 0.2)  # Cap at 0.2 bonus
    
    # Weighted combination
    influence_score = (
        score_component * 0.3 +
        ratio_component * 0.25 +
        comments_component * 0.2 +
        text_quality * 0.15 +
        comment_bonus * 0.1
    )
    
    return min(influence_score, 1.0)
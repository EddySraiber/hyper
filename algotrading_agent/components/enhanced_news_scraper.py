"""
Enhanced News Scraper - Multi-source financial news aggregation

Features:
- RSS feeds (existing)  
- Financial APIs (Alpha Vantage, Finnhub)
- Real-time WebSocket streams
- Full article content extraction
- Smart scheduling and error handling
- Deduplication and source reliability scoring
"""

import asyncio
import aiohttp
import feedparser
import hashlib
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse
from dataclasses import dataclass
from ..core.base import ComponentBase


@dataclass
class NewsSource:
    """News source configuration"""
    name: str
    type: str  # rss, api, websocket
    url: str
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    rate_limit: float = 1.0  # seconds between requests
    api_key_env: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    last_request: Optional[datetime] = None
    failure_count: int = 0
    reliability_score: float = 1.0


class EnhancedNewsScraper(ComponentBase):
    """
    Advanced news scraper with multiple data sources and real-time capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("enhanced_news_scraper", config)
        
        # Configuration
        self.update_interval = config.get("update_interval", 60)
        self.max_age_minutes = config.get("max_age_minutes", 30)
        self.max_concurrent = config.get("max_concurrent_requests", 10)
        self.enable_full_content = config.get("enable_full_content", True)
        self.enable_deduplication = config.get("enable_deduplication", True)
        self.retry_failed_sources = config.get("retry_failed_sources", True)
        self.max_failures_before_disable = config.get("max_failures_before_disable", 5)
        
        # Initialize news sources
        self.sources: List[NewsSource] = self._initialize_sources(config)
        
        # Session and state management
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self.seen_articles: Set[str] = set()  # For deduplication
        self.content_cache: Dict[str, Dict[str, Any]] = {}  # URL -> content cache
        
        # Performance tracking
        self.total_articles_scraped = 0
        self.duplicate_articles_filtered = 0
        self.failed_requests = 0
        self.source_performance: Dict[str, Dict[str, Any]] = {}
        
    def _initialize_sources(self, config: Dict[str, Any]) -> List[NewsSource]:
        """Initialize all configured news sources"""
        sources = []
        
        # Legacy RSS sources
        legacy_sources = config.get("sources", [])
        for source_config in legacy_sources:
            source = NewsSource(
                name=source_config["name"],
                type=source_config.get("type", "rss"),
                url=source_config["url"],
                priority=source_config.get("priority", 2)
            )
            sources.append(source)
            
        # Enhanced API sources
        api_sources = config.get("api_sources", [])
        for source_config in api_sources:
            source = NewsSource(
                name=source_config["name"],
                type="api",
                url=source_config["url"],
                enabled=source_config.get("enabled", True),
                priority=source_config.get("priority", 1),
                rate_limit=source_config.get("rate_limit", 1.0),
                api_key_env=source_config.get("api_key_env"),
                headers=source_config.get("headers", {})
            )
            sources.append(source)
            
        # Social media sources
        social_sources = config.get("social_sources", [])
        for source_config in social_sources:
            source = NewsSource(
                name=source_config["name"],
                type="social",
                url=source_config["url"],
                enabled=source_config.get("enabled", True),
                priority=source_config.get("priority", 2),
                rate_limit=source_config.get("rate_limit", 60.0),
                headers=source_config.get("headers", {})
            )
            sources.append(source)
            
        # WebSocket streaming sources  
        ws_sources = config.get("websocket_sources", [])
        for source_config in ws_sources:
            source = NewsSource(
                name=source_config["name"],
                type="websocket",
                url=source_config["url"],
                enabled=source_config.get("enabled", True),
                priority=1,  # WebSocket sources are always high priority
                api_key_env=source_config.get("api_key_env")
            )
            sources.append(source)
            
        self.logger.info(f"Initialized {len(sources)} news sources")
        return sources
    
    async def start(self) -> None:
        """Start the enhanced news scraper with all sources"""
        self.logger.info("🚀 Starting Enhanced News Scraper")
        self.is_running = True
        
        # Initialize HTTP session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=50,  # Connection pool size
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=20   # Socket read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'AlgoTradingAgent/1.0 (Financial News Aggregator)'
            }
        )
        
        # Start WebSocket connections for real-time sources
        await self._start_websocket_sources()
        
        # Initialize source performance tracking
        for source in self.sources:
            self.source_performance[source.name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0.0,
                "last_success": None,
                "articles_contributed": 0
            }
        
        self.logger.info(f"✅ Enhanced News Scraper started with {len(self.sources)} sources")
        
    async def stop(self) -> None:
        """Stop all scraping activities and clean up resources"""
        self.logger.info("🛑 Stopping Enhanced News Scraper")
        self.is_running = False
        
        # Close WebSocket connections
        for name, ws in self.websocket_connections.items():
            try:
                await ws.close()
                self.logger.info(f"Closed WebSocket connection: {name}")
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket {name}: {e}")
        
        self.websocket_connections.clear()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            
        # Log final statistics
        self._log_performance_summary()
        
    async def process(self, data: Any = None) -> List[Dict[str, Any]]:
        """Main processing method - scrape from all enabled sources"""
        if not self.is_running:
            return []
            
        start_time = datetime.utcnow()
        all_news = []
        
        # Prioritize sources and process them
        active_sources = [s for s in self.sources if s.enabled and s.type != "websocket"]
        active_sources.sort(key=lambda x: x.priority)  # Priority 1 = highest
        
        # Process sources with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def scrape_source_safe(source: NewsSource) -> List[Dict[str, Any]]:
            async with semaphore:
                return await self._scrape_source_with_retry(source)
        
        # Gather results from all sources (RSS, API, Social)
        tasks = [scrape_source_safe(source) for source in active_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        for i, result in enumerate(results):
            source = active_sources[i]
            if isinstance(result, Exception):
                self.logger.error(f"Exception scraping {source.name}: {result}")
                self._update_source_performance(source.name, success=False)
            elif isinstance(result, list):
                all_news.extend(result)
                self._update_source_performance(source.name, success=True, articles=len(result))
            
        # Post-processing
        if self.enable_deduplication:
            all_news = self._deduplicate_articles(all_news)
            
        # Add processing metadata
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.logger.info(
            f"📰 Scraped {len(all_news)} articles from {len(active_sources)} sources "
            f"in {processing_time:.2f}s"
        )
        
        self.total_articles_scraped += len(all_news)
        return all_news
    
    async def _start_websocket_sources(self) -> None:
        """Initialize WebSocket connections for real-time news"""
        ws_sources = [s for s in self.sources if s.type == "websocket" and s.enabled]
        
        for source in ws_sources:
            try:
                ws = await self._connect_websocket(source)
                if ws:
                    self.websocket_connections[source.name] = ws
                    # Start background task to handle WebSocket messages
                    asyncio.create_task(self._handle_websocket_messages(source, ws))
                    self.logger.info(f"📡 WebSocket connected: {source.name}")
            except Exception as e:
                self.logger.error(f"Failed to connect WebSocket {source.name}: {e}")
                source.failure_count += 1
    
    async def _connect_websocket(self, source: NewsSource) -> Optional[aiohttp.ClientWebSocketResponse]:
        """Connect to a WebSocket news source"""
        try:
            headers = {}
            if source.api_key_env:
                import os
                api_key = os.getenv(source.api_key_env)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
            
            ws = await self.session.ws_connect(source.url, headers=headers)
            return ws
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed for {source.name}: {e}")
            return None
    
    async def _handle_websocket_messages(self, source: NewsSource, ws: aiohttp.ClientWebSocketResponse):
        """Handle incoming WebSocket messages"""
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        article = self._parse_websocket_message(source, data)
                        if article:
                            # Process real-time article immediately
                            await self._process_realtime_article(article)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON from WebSocket {source.name}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error {source.name}: {ws.exception()}")
                    break
                    
        except Exception as e:
            self.logger.error(f"WebSocket handler error {source.name}: {e}")
        finally:
            self.logger.warning(f"WebSocket connection closed: {source.name}")
            # Try to reconnect
            if self.is_running:
                await asyncio.sleep(5)
                await self._reconnect_websocket(source)
    
    async def _reconnect_websocket(self, source: NewsSource):
        """Attempt to reconnect a failed WebSocket"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ws = await self._connect_websocket(source)
                if ws:
                    self.websocket_connections[source.name] = ws
                    asyncio.create_task(self._handle_websocket_messages(source, ws))
                    self.logger.info(f"🔄 WebSocket reconnected: {source.name}")
                    return
            except Exception as e:
                self.logger.warning(f"Reconnect attempt {attempt+1} failed for {source.name}: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        self.logger.error(f"❌ Failed to reconnect WebSocket after {max_retries} attempts: {source.name}")
        source.enabled = False
    
    def _parse_websocket_message(self, source: NewsSource, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse WebSocket message into standardized article format"""
        try:
            # This would be customized based on the WebSocket source format
            # Example for a generic financial news WebSocket
            article = {
                "title": data.get("title", ""),
                "content": data.get("content", data.get("summary", "")),
                "url": data.get("url", ""),
                "published": self._parse_timestamp(data.get("timestamp")),
                "source": source.name,
                "real_time": True,
                "priority": "breaking" if data.get("breaking") else "normal",
                "raw_data": data
            }
            
            return article if article["title"] else None
            
        except Exception as e:
            self.logger.error(f"Error parsing WebSocket message from {source.name}: {e}")
            return None
    
    async def _process_realtime_article(self, article: Dict[str, Any]):
        """Process real-time article immediately (bypass normal pipeline)"""
        if self.enable_deduplication:
            article_hash = self._generate_content_hash(article)
            if article_hash in self.seen_articles:
                return
            self.seen_articles.add(article_hash)
        
        # Emit real-time article for immediate processing
        # This could trigger an event or callback to the main application
        self.logger.info(f"⚡ Real-time article: {article['title'][:80]}...")
        
        # Add to cache for the next regular processing cycle
        if hasattr(self, '_realtime_buffer'):
            self._realtime_buffer.append(article)
        else:
            self._realtime_buffer = [article]
    
    async def _scrape_source_with_retry(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Scrape a source with retry logic and rate limiting"""
        
        # Rate limiting
        if source.last_request:
            elapsed = (datetime.utcnow() - source.last_request).total_seconds()
            if elapsed < source.rate_limit:
                await asyncio.sleep(source.rate_limit - elapsed)
        
        source.last_request = datetime.utcnow()
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if source.type == "rss":
                    return await self._scrape_rss_enhanced(source)
                elif source.type == "api":
                    return await self._scrape_api_enhanced(source)
                elif source.type == "social":
                    return await self._scrape_social_enhanced(source)
                else:
                    self.logger.warning(f"Unknown source type: {source.type}")
                    return []
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1} failed for {source.name}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    source.failure_count += 1
                    if source.failure_count >= self.max_failures_before_disable:
                        source.enabled = False
                        self.logger.error(f"❌ Disabled {source.name} after {source.failure_count} failures")
                    return []
        
        return []
    
    async def _scrape_rss_enhanced(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Enhanced RSS scraping with full content extraction"""
        try:
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                if feed.bozo and feed.bozo_exception:
                    self.logger.warning(f"RSS feed parsing warning for {source.name}: {feed.bozo_exception}")
                
                cutoff_time = datetime.utcnow() - timedelta(minutes=self.max_age_minutes)
                articles = []
                
                for entry in feed.entries:
                    try:
                        published_date = self._parse_date(entry.get("published", ""))
                        if published_date < cutoff_time:
                            continue
                        
                        article = {
                            "title": entry.get("title", ""),
                            "content": entry.get("summary", ""),
                            "url": entry.get("link", ""),
                            "published": published_date,
                            "source": source.name,
                            "age_minutes": (datetime.utcnow() - published_date).total_seconds() / 60,
                            "raw_data": dict(entry)
                        }
                        
                        # Extract full content if enabled and URL is available
                        if self.enable_full_content and article["url"]:
                            full_content = await self._extract_full_content(article["url"])
                            if full_content:
                                article["full_content"] = full_content
                        
                        articles.append(article)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing RSS entry from {source.name}: {e}")
                        continue
                
                source.failure_count = 0  # Reset failure count on success
                return articles
                
        except Exception as e:
            self.logger.error(f"RSS scraping error for {source.name}: {e}")
            raise
    
    async def _scrape_api_enhanced(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Enhanced API scraping for financial news services"""
        try:
            headers = source.headers.copy() if source.headers else {}
            
            # Add API key if configured
            if source.api_key_env:
                import os
                api_key = os.getenv(source.api_key_env)
                if api_key:
                    if "alpha" in source.name.lower():
                        # Alpha Vantage format
                        url = f"{source.url}&apikey={api_key}"
                    elif "finnhub" in source.name.lower():
                        # Finnhub format
                        headers["X-Finnhub-Token"] = api_key
                        url = source.url
                    else:
                        # Generic bearer token
                        headers["Authorization"] = f"Bearer {api_key}"
                        url = source.url
                else:
                    self.logger.warning(f"API key not found for {source.name}")
                    return []
            else:
                url = source.url
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                
                data = await response.json()
                articles = self._parse_api_response(source, data)
                
                source.failure_count = 0  # Reset failure count on success
                return articles
                
        except Exception as e:
            self.logger.error(f"API scraping error for {source.name}: {e}")
            raise
    
    def _parse_api_response(self, source: NewsSource, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse API response based on source type"""
        articles = []
        
        try:
            if "alpha" in source.name.lower():
                # Alpha Vantage News API format
                feed_data = data.get("feed", [])
                for item in feed_data:
                    published_date = self._parse_timestamp(item.get("time_published"))
                    article = {
                        "title": item.get("title", ""),
                        "content": item.get("summary", ""),
                        "url": item.get("url", ""),
                        "published": published_date,
                        "source": source.name,
                        "sentiment": item.get("overall_sentiment_label", "neutral"),
                        "relevance_score": float(item.get("relevance_score", 0.0)),
                        "tickers": item.get("ticker_sentiment", []),
                        "raw_data": item
                    }
                    articles.append(article)
                    
            elif "finnhub" in source.name.lower():
                # Finnhub News API format
                news_data = data.get("news", [])
                for item in news_data:
                    published_date = self._parse_timestamp(item.get("datetime"))
                    article = {
                        "title": item.get("headline", ""),
                        "content": item.get("summary", ""),
                        "url": item.get("url", ""),
                        "published": published_date,
                        "source": source.name,
                        "category": item.get("category", ""),
                        "image": item.get("image", ""),
                        "raw_data": item
                    }
                    articles.append(article)
            
            elif "newsapi" in source.name.lower():
                # News API format
                articles_data = data.get("articles", [])
                for item in articles_data:
                    published_date = self._parse_date(item.get("publishedAt", ""))
                    article = {
                        "title": item.get("title", ""),
                        "content": item.get("description", ""),
                        "url": item.get("url", ""),
                        "published": published_date,
                        "source": source.name,
                        "author": item.get("author", ""),
                        "image": item.get("urlToImage", ""),
                        "raw_data": item
                    }
                    articles.append(article)
                    
            elif "polygon" in source.name.lower():
                # Polygon.io News API format
                results = data.get("results", [])
                for item in results:
                    published_date = self._parse_date(item.get("published_utc", ""))
                    article = {
                        "title": item.get("title", ""),
                        "content": item.get("description", ""),
                        "url": item.get("article_url", ""),
                        "published": published_date,
                        "source": source.name,
                        "author": item.get("author", ""),
                        "tickers": item.get("tickers", []),
                        "raw_data": item
                    }
                    articles.append(article)
                    
            elif "fred" in source.name.lower():
                # Federal Reserve Economic Data API format
                releases = data.get("releases", [])
                for item in releases:
                    published_date = self._parse_date(item.get("realtime_start", ""))
                    article = {
                        "title": f"Economic Release: {item.get('name', '')}",
                        "content": f"FRED Release ID: {item.get('id', '')}, Press Release: {item.get('press_release', 'Available')}",
                        "url": item.get("link", ""),
                        "published": published_date,
                        "source": source.name,
                        "category": "economic_data",
                        "fred_id": item.get("id", ""),
                        "raw_data": item
                    }
                    articles.append(article)
            else:
                # Generic API format
                self.logger.warning(f"Unknown API format for {source.name}")
                
        except Exception as e:
            self.logger.error(f"Error parsing API response from {source.name}: {e}")
            
        return articles
    
    async def _scrape_social_enhanced(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Enhanced social media scraping for Reddit and Twitter"""
        try:
            if "reddit" in source.name.lower():
                return await self._scrape_reddit(source)
            elif "twitter" in source.name.lower():
                return await self._scrape_twitter(source)
            else:
                self.logger.warning(f"Unknown social source type: {source.name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Social scraping error for {source.name}: {e}")
            raise
    
    async def _scrape_reddit(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Scrape Reddit posts for financial sentiment"""
        try:
            headers = source.headers.copy() if source.headers else {}
            headers.update({
                'Accept': 'application/json',
                'User-Agent': headers.get('User-Agent', 'AlgoTradingAgent/1.0')
            })
            
            async with self.session.get(source.url, headers=headers) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                
                data = await response.json()
                posts_data = data.get("data", {}).get("children", [])
                articles = []
                
                cutoff_time = datetime.utcnow() - timedelta(minutes=self.max_age_minutes)
                
                for post_data in posts_data:
                    post = post_data.get("data", {})
                    
                    # Skip if not relevant or too old
                    created_utc = post.get("created_utc", 0)
                    created_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.utcnow()
                    
                    if created_date < cutoff_time:
                        continue
                    
                    # Extract financial relevance
                    title = post.get("title", "")
                    content = post.get("selftext", "")
                    
                    # Basic filter for financial relevance
                    financial_keywords = [
                        'stock', 'stocks', 'trading', 'buy', 'sell', 'call', 'put', 
                        'earnings', 'dividend', 'market', 'bull', 'bear', 'DD',
                        'YOLO', 'diamond hands', 'paper hands', 'to the moon'
                    ]
                    
                    combined_text = f"{title} {content}".lower()
                    if not any(keyword in combined_text for keyword in financial_keywords):
                        continue
                    
                    # Check for stock ticker mentions ($SYMBOL or SYMBOL pattern)
                    import re
                    tickers = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', combined_text.upper())
                    ticker_mentions = [t[0] if t[0] else t[1] for t in tickers if t[0] or t[1]]
                    
                    article = {
                        "title": title,
                        "content": content[:500] + "..." if len(content) > 500 else content,
                        "url": f"https://reddit.com{post.get('permalink', '')}",
                        "published": created_date,
                        "source": source.name,
                        "upvotes": post.get("ups", 0),
                        "downvotes": post.get("downs", 0),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "subreddit": post.get("subreddit", ""),
                        "author": post.get("author", ""),
                        "tickers": ticker_mentions,
                        "social_metrics": {
                            "engagement_score": post.get("score", 0) + post.get("num_comments", 0),
                            "upvote_ratio": post.get("upvote_ratio", 0),
                            "controversiality": post.get("controversiality", 0)
                        },
                        "raw_data": post
                    }
                    
                    # Add basic sentiment using VADER (if available)
                    try:
                        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                        analyzer = SentimentIntensityAnalyzer()
                        sentiment = analyzer.polarity_scores(combined_text)
                        article["sentiment"] = sentiment["compound"]  # Overall sentiment score
                        article["sentiment_detail"] = sentiment
                    except ImportError:
                        article["sentiment"] = 0.0  # Neutral if VADER not available
                    
                    articles.append(article)
                
                source.failure_count = 0  # Reset failure count on success
                return articles[:25]  # Limit to 25 most recent relevant posts
                
        except Exception as e:
            self.logger.error(f"Reddit scraping error for {source.name}: {e}")
            raise
    
    async def _scrape_twitter(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Scrape Twitter for financial sentiment (requires API key)"""
        try:
            # Twitter API v2 implementation (simplified)
            headers = source.headers.copy() if source.headers else {}
            
            # Add API key if configured
            if source.api_key_env:
                import os
                api_key = os.getenv(source.api_key_env)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                else:
                    self.logger.warning(f"Twitter API key not found for {source.name}")
                    return []
            else:
                self.logger.warning(f"No API key configured for {source.name}")
                return []
            
            async with self.session.get(source.url, headers=headers) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                
                data = await response.json()
                tweets = data.get("data", [])
                articles = []
                
                cutoff_time = datetime.utcnow() - timedelta(minutes=self.max_age_minutes)
                
                for tweet in tweets:
                    created_date = self._parse_date(tweet.get("created_at", ""))
                    
                    if created_date < cutoff_time:
                        continue
                    
                    text = tweet.get("text", "")
                    
                    # Extract financial relevance and tickers
                    import re
                    tickers = re.findall(r'\$([A-Z]{1,5})\b', text.upper())
                    
                    if not tickers:  # Skip tweets without stock mentions
                        continue
                    
                    # Get public metrics if available
                    public_metrics = tweet.get("public_metrics", {})
                    
                    article = {
                        "title": text[:100] + "..." if len(text) > 100 else text,
                        "content": text,
                        "url": f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                        "published": created_date,
                        "source": source.name,
                        "author": tweet.get("author_id", ""),
                        "tickers": tickers,
                        "social_metrics": {
                            "retweets": public_metrics.get("retweet_count", 0),
                            "likes": public_metrics.get("like_count", 0),
                            "replies": public_metrics.get("reply_count", 0),
                            "quotes": public_metrics.get("quote_count", 0)
                        },
                        "raw_data": tweet
                    }
                    
                    # Add basic sentiment using VADER (if available)
                    try:
                        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                        analyzer = SentimentIntensityAnalyzer()
                        sentiment = analyzer.polarity_scores(text)
                        article["sentiment"] = sentiment["compound"]
                        article["sentiment_detail"] = sentiment
                    except ImportError:
                        article["sentiment"] = 0.0
                    
                    articles.append(article)
                
                source.failure_count = 0  # Reset failure count on success
                return articles
                
        except Exception as e:
            self.logger.error(f"Twitter scraping error for {source.name}: {e}")
            raise
    
    async def _extract_full_content(self, url: str) -> Optional[str]:
        """Extract full article content from URL"""
        if url in self.content_cache:
            return self.content_cache[url].get("content")
        
        try:
            # Only extract from trusted domains to avoid scraping issues
            domain = urlparse(url).netloc.lower()
            trusted_domains = {
                'reuters.com', 'bloomberg.com', 'cnbc.com', 'marketwatch.com',
                'finance.yahoo.com', 'seekingalpha.com', 'benzinga.com'
            }
            
            if not any(trusted in domain for trusted in trusted_domains):
                return None
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html_content = await response.text()
                
                # Basic content extraction (could be enhanced with readability libraries)
                # For now, we'll cache the HTML and extract basic text
                self.content_cache[url] = {
                    "content": self._extract_text_from_html(html_content),
                    "cached_at": datetime.utcnow()
                }
                
                return self.content_cache[url]["content"]
                
        except Exception as e:
            self.logger.warning(f"Failed to extract content from {url}: {e}")
            return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """Basic HTML text extraction"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit length to avoid excessive content
            return text[:5000] if len(text) > 5000 else text
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available for HTML parsing")
            return ""
        except Exception as e:
            self.logger.warning(f"HTML parsing error: {e}")
            return ""
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content similarity"""
        if not articles:
            return articles
        
        unique_articles = []
        initial_count = len(articles)
        
        for article in articles:
            content_hash = self._generate_content_hash(article)
            
            if content_hash not in self.seen_articles:
                self.seen_articles.add(content_hash)
                unique_articles.append(article)
            else:
                self.duplicate_articles_filtered += 1
        
        # Clean old hashes to prevent memory growth
        if len(self.seen_articles) > 10000:
            # Keep only the most recent 5000 hashes
            self.seen_articles = set(list(self.seen_articles)[-5000:])
        
        duplicates_removed = initial_count - len(unique_articles)
        if duplicates_removed > 0:
            self.logger.info(f"🔍 Removed {duplicates_removed} duplicate articles")
        
        return unique_articles
    
    def _generate_content_hash(self, article: Dict[str, Any]) -> str:
        """Generate hash for article deduplication"""
        # Use title + first 200 chars of content for similarity detection
        title = article.get("title", "").strip().lower()
        content = article.get("content", "").strip().lower()[:200]
        combined = f"{title}|{content}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        if not date_str:
            return datetime.utcnow()
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                # Try RFC 2822 format (common in RSS)
                import email.utils
                timestamp = email.utils.parsedate_tz(date_str)
                if timestamp:
                    return datetime.fromtimestamp(email.utils.mktime_tz(timestamp))
            except:
                pass
            
        # Fallback to current time
        return datetime.utcnow()
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp from API responses"""
        if isinstance(timestamp, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            return self._parse_date(timestamp)
        else:
            return datetime.utcnow()
    
    def _update_source_performance(self, source_name: str, success: bool, articles: int = 0):
        """Update performance tracking for a source"""
        if source_name not in self.source_performance:
            self.source_performance[source_name] = {
                "requests": 0, "successes": 0, "failures": 0,
                "avg_response_time": 0.0, "last_success": None, "articles_contributed": 0
            }
        
        perf = self.source_performance[source_name]
        perf["requests"] += 1
        
        if success:
            perf["successes"] += 1
            perf["articles_contributed"] += articles
            perf["last_success"] = datetime.utcnow()
        else:
            perf["failures"] += 1
    
    def _log_performance_summary(self):
        """Log performance summary for all sources"""
        self.logger.info("📊 NEWS SCRAPER PERFORMANCE SUMMARY:")
        self.logger.info(f"  Total articles scraped: {self.total_articles_scraped}")
        self.logger.info(f"  Duplicates filtered: {self.duplicate_articles_filtered}")
        self.logger.info(f"  Failed requests: {self.failed_requests}")
        
        for name, perf in self.source_performance.items():
            success_rate = (perf["successes"] / perf["requests"] * 100) if perf["requests"] > 0 else 0
            self.logger.info(f"  {name}: {perf['successes']}/{perf['requests']} success ({success_rate:.1f}%), {perf['articles_contributed']} articles")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return {
            "total_articles_scraped": self.total_articles_scraped,
            "duplicate_articles_filtered": self.duplicate_articles_filtered,
            "failed_requests": self.failed_requests,
            "active_sources": len([s for s in self.sources if s.enabled]),
            "total_sources": len(self.sources),
            "websocket_connections": len(self.websocket_connections),
            "cache_size": len(self.content_cache),
            "source_performance": self.source_performance,
            "sources_status": [
                {
                    "name": s.name,
                    "type": s.type,
                    "enabled": s.enabled,
                    "priority": s.priority,
                    "failure_count": s.failure_count,
                    "reliability_score": s.reliability_score
                }
                for s in self.sources
            ]
        }
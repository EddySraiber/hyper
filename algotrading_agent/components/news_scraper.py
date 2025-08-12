import asyncio
import aiohttp
import feedparser
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ..core.base import ComponentBase


class NewsScraper(ComponentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("news_scraper", config)
        self.sources = config.get("sources", [])
        self.update_interval = config.get("update_interval", 300)  # 5 minutes
        self.max_age_minutes = config.get("max_age_minutes", 30)  # Default 30 minutes freshness
        self.session = None
        
    async def start(self) -> None:
        self.logger.info("Starting News Scraper")
        self.is_running = True
        self.session = aiohttp.ClientSession()
        
    async def stop(self) -> None:
        self.logger.info("Stopping News Scraper")
        self.is_running = False
        if self.session:
            await self.session.close()
            
    async def process(self, data: Any = None) -> List[Dict[str, Any]]:
        if not self.is_running:
            return []
            
        news_items = []
        for source in self.sources:
            try:
                items = await self._scrape_source(source)
                news_items.extend(items)
                self.logger.info(f"Scraped {len(items)} items from {source['name']}")
            except Exception as e:
                self.logger.error(f"Error scraping {source['name']}: {e}")
                
        return news_items
        
    async def _scrape_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_type = source.get("type", "rss")
        
        if source_type == "rss":
            return await self._scrape_rss(source)
        elif source_type == "api":
            return await self._scrape_api(source)
        else:
            self.logger.warning(f"Unknown source type: {source_type}")
            return []
            
    async def _scrape_rss(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            async with self.session.get(source["url"]) as response:
                content = await response.text()
                feed = feedparser.parse(content)
                
                # Calculate freshness cutoff time
                cutoff_time = datetime.utcnow() - timedelta(minutes=self.max_age_minutes)
                
                items = []
                total_entries = len(feed.entries)
                fresh_count = 0
                
                for entry in feed.entries:
                    published_date = self._parse_date(entry.get("published"))
                    
                    # Skip news older than max_age_minutes
                    if published_date < cutoff_time:
                        continue
                        
                    fresh_count += 1
                    item = {
                        "title": entry.get("title", ""),
                        "content": entry.get("summary", ""),
                        "url": entry.get("link", ""),
                        "published": published_date,
                        "source": source["name"],
                        "age_minutes": (datetime.utcnow() - published_date).total_seconds() / 60,
                        "raw_data": entry
                    }
                    items.append(item)
                
                if total_entries > 0:
                    self.logger.info(f"Filtered {source['name']}: {fresh_count}/{total_entries} items fresh (within {self.max_age_minutes} min)")
                    
                return items
        except Exception as e:
            self.logger.error(f"RSS scraping error for {source['name']}: {e}")
            return []
            
    async def _scrape_api(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Placeholder for API-based news sources
        self.logger.info(f"API scraping not yet implemented for {source['name']}")
        return []
        
    def _parse_date(self, date_str: str) -> datetime:
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return datetime.utcnow()
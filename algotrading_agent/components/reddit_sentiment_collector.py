"""
Reddit Sentiment Collector Component
Collects and processes Reddit sentiment data for financial symbols following the system architecture patterns
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ..core.base import ComponentBase
from ..data_sources.reddit_client import RedditClient
from ..data_sources.reddit_sentiment_processor import RedditSentimentProcessor, RedditSentimentSignal


class RedditSentimentCollector(ComponentBase):
    """
    Component for collecting and processing Reddit sentiment data
    Focuses on high-quality financial discussions from reputable subreddits
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("reddit_sentiment_collector", config)
        
        # Component configuration
        reddit_config = config.get('reddit', {})
        self.enabled = reddit_config.get('enabled', True)  # Enabled by default (free API)
        self.update_interval = reddit_config.get('update_interval', 600)  # 10 minutes default (conservative)
        
        # Symbol tracking
        self.tracked_symbols = reddit_config.get('tracked_symbols', [
            'AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT', 'NVDA', 'META', 'SPY',
            'QQQ', 'VTI', 'GME', 'AMC', 'PLTR', 'AMD', 'NFLX', 'ORCL'
        ])
        self.max_symbols_per_cycle = reddit_config.get('max_symbols_per_cycle', 4)  # Conservative for free API
        
        # Analysis configuration  
        self.timeframes = reddit_config.get('timeframes', ['4h', '12h', '24h', '3d'])
        self.min_signal_quality = reddit_config.get('min_signal_quality', 0.5)
        self.signal_cache_hours = reddit_config.get('signal_cache_hours', 48)  # Longer cache for Reddit
        self.collect_comments = reddit_config.get('collect_comments', True)
        self.max_comments_per_post = reddit_config.get('max_comments_per_post', 8)
        
        # Components
        self.reddit_client: Optional[RedditClient] = None
        self.sentiment_processor: Optional[RedditSentimentProcessor] = None
        
        # Internal state
        self.current_signals: Dict[str, List[RedditSentimentSignal]] = {}  # symbol -> signals
        self.signal_history: List[Dict[str, Any]] = []
        self.last_update: Optional[datetime] = None
        self.symbol_rotation_index = 0  # For round-robin symbol processing
        
        # Performance tracking
        self.stats = {
            'total_posts_processed': 0,
            'total_comments_processed': 0,
            'signals_generated': 0,
            'actionable_signals': 0,
            'api_calls_made': 0,
            'rate_limit_hits': 0,
            'subreddit_coverage': {},  # Track which subreddits we're getting data from
            'last_reset_time': datetime.utcnow()
        }
        
    async def start(self) -> None:
        """Initialize the Reddit sentiment collector"""
        if not self.enabled:
            self.logger.info("Reddit sentiment collection is disabled")
            return
            
        self.logger.info("Starting Reddit Sentiment Collector")
        
        try:
            # Initialize Reddit client
            self.reddit_client = RedditClient(self.config)
            await self.reddit_client.start()
            
            # Initialize sentiment processor
            self.sentiment_processor = RedditSentimentProcessor(self.config)
            
            # Load persisted state
            await self._load_state()
            
            # Start background collection task
            asyncio.create_task(self._collection_loop())
            
            self.is_running = True
            self.logger.info("Reddit sentiment collector started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Reddit sentiment collector: {e}")
            self.is_running = False
            
    async def stop(self) -> None:
        """Stop the Reddit sentiment collector"""
        self.logger.info("Stopping Reddit Sentiment Collector")
        
        # Save current state
        await self._save_state()
        
        # Stop Reddit client
        if self.reddit_client:
            await self.reddit_client.stop()
            
        self.is_running = False
        
    async def process(self, data: Any) -> Any:
        """Process method required by ComponentBase - not used in this implementation"""
        return None
        
    async def _collection_loop(self) -> None:
        """Main collection loop - runs continuously"""
        while self.is_running:
            try:
                await self._collect_and_process()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Reddit collection loop: {e}")
                await asyncio.sleep(min(self.update_interval, 120))  # Wait at least 2 minutes on error
                
    async def _collect_and_process(self) -> None:
        """Collect Reddit posts and process sentiment for tracked symbols"""
        if not self.reddit_client or not self.sentiment_processor:
            return
            
        # Select symbols to process this cycle (round-robin to respect rate limits)
        symbols_to_process = self._get_symbols_for_cycle()
        
        self.logger.info(f"Processing Reddit sentiment for symbols: {symbols_to_process}")
        
        for symbol in symbols_to_process:
            try:
                await self._process_symbol_sentiment(symbol)
                
                # Delay between symbols to be respectful to Reddit's API
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Error processing Reddit sentiment for {symbol}: {e}")
                continue
                
        # Update timestamp and save state
        self.last_update = datetime.utcnow()
        await self._save_state()
        
        # Log collection summary
        self._log_collection_summary()
        
    def _get_symbols_for_cycle(self) -> List[str]:
        """Get symbols to process in this cycle using round-robin"""
        if len(self.tracked_symbols) <= self.max_symbols_per_cycle:
            return self.tracked_symbols
            
        # Round-robin selection
        end_index = self.symbol_rotation_index + self.max_symbols_per_cycle
        
        if end_index <= len(self.tracked_symbols):
            selected = self.tracked_symbols[self.symbol_rotation_index:end_index]
        else:
            # Wrap around
            remaining = end_index - len(self.tracked_symbols)
            selected = self.tracked_symbols[self.symbol_rotation_index:] + self.tracked_symbols[:remaining]
            
        # Update index for next cycle
        self.symbol_rotation_index = (self.symbol_rotation_index + self.max_symbols_per_cycle) % len(self.tracked_symbols)
        
        return selected
        
    async def _process_symbol_sentiment(self, symbol: str) -> None:
        """Process sentiment for a specific symbol"""
        try:
            # Collect posts for the symbol
            posts = await self.reddit_client.get_financial_posts([symbol], hours_back=72)  # Longer lookback for Reddit
            
            if not posts:
                self.logger.debug(f"No Reddit posts found for {symbol}")
                return
                
            self.stats['total_posts_processed'] += len(posts)
            self.stats['api_calls_made'] += 1
            
            # Track subreddit coverage
            for post in posts:
                subreddit = post.subreddit
                self.stats['subreddit_coverage'][subreddit] = self.stats['subreddit_coverage'].get(subreddit, 0) + 1
            
            # Optionally collect comments for high-quality posts
            post_comments = {}
            if self.collect_comments:
                high_quality_posts = [p for p in posts if p.score >= 20 and p.num_comments >= 10][:5]
                
                for post in high_quality_posts:
                    try:
                        comments = await self.reddit_client.get_post_comments(post, self.max_comments_per_post)
                        if comments:
                            post_comments[post.id] = comments
                            self.stats['total_comments_processed'] += len(comments)
                            
                        # Delay between comment requests
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        self.logger.error(f"Error collecting comments for post {post.id}: {e}")
                        continue
            
            # Process sentiment for different timeframes
            symbol_signals = []
            
            for timeframe in self.timeframes:
                signal = await self.sentiment_processor.process_symbol_sentiment(
                    symbol, posts, post_comments, timeframe
                )
                
                if signal:
                    signal_quality = self.sentiment_processor.get_signal_quality_score(signal)
                    
                    if signal_quality >= self.min_signal_quality:
                        symbol_signals.append(signal)
                        self.stats['signals_generated'] += 1
                        
                        if self.sentiment_processor.is_signal_actionable(signal):
                            self.stats['actionable_signals'] += 1
                            
            # Store signals for this symbol
            if symbol_signals:
                self.current_signals[symbol] = symbol_signals
                
                # Add to history
                self._add_to_signal_history(symbol, symbol_signals)
                
                self.logger.info(f"Generated {len(symbol_signals)} Reddit sentiment signals for {symbol}")
                
                # Log signal details for debugging
                for signal in symbol_signals:
                    summary = self.sentiment_processor.get_signal_summary(signal)
                    self.logger.debug(f"Reddit signal for {symbol} ({signal.timeframe}): {summary}")
            else:
                self.logger.debug(f"No quality Reddit signals generated for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error processing Reddit sentiment for {symbol}: {e}")
            
    def _add_to_signal_history(self, symbol: str, signals: List[RedditSentimentSignal]) -> None:
        """Add signals to historical record"""
        timestamp = datetime.utcnow().isoformat()
        
        for signal in signals:
            history_entry = {
                'timestamp': timestamp,
                'symbol': symbol,
                'timeframe': signal.timeframe,
                'sentiment_score': signal.sentiment_score,
                'confidence': signal.confidence,
                'quality_score': signal.quality_score,
                'influence_score': signal.influence_score,
                'post_count': signal.post_count,
                'comment_count': signal.comment_count,
                'overall_quality': self.sentiment_processor.get_signal_quality_score(signal),
                'actionable': self.sentiment_processor.is_signal_actionable(signal),
                'sentiment_distribution': signal.sentiment_distribution,
                'subreddit_distribution': dict(list(signal.subreddit_distribution.items())[:5]),  # Top 5
                'keywords': signal.keywords[:10]  # Limit keywords for storage
            }
            
            self.signal_history.append(history_entry)
            
        # Limit history size
        max_history = 800  # Smaller than Twitter due to less frequent updates
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
            
    def _log_collection_summary(self) -> None:
        """Log summary of collection cycle"""
        current_signals_count = sum(len(signals) for signals in self.current_signals.values())
        subreddit_count = len(self.stats['subreddit_coverage'])
        
        self.logger.info(
            f"Reddit collection summary: "
            f"{self.stats['total_posts_processed']} posts, "
            f"{self.stats['total_comments_processed']} comments processed, "
            f"{current_signals_count} current signals, "
            f"{self.stats['actionable_signals']} actionable signals, "
            f"{subreddit_count} subreddits active"
        )
        
    async def get_symbol_sentiment(self, symbol: str, timeframe: str = "24h") -> Optional[RedditSentimentSignal]:
        """
        Get current sentiment signal for a specific symbol and timeframe
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Timeframe for signal (4h, 12h, 24h, 3d)
            
        Returns:
            RedditSentimentSignal or None if not available
        """
        symbol_signals = self.current_signals.get(symbol, [])
        
        for signal in symbol_signals:
            if signal.timeframe == timeframe:
                return signal
                
        return None
        
    async def get_all_current_signals(self) -> Dict[str, List[RedditSentimentSignal]]:
        """Get all current sentiment signals"""
        return self.current_signals.copy()
        
    async def get_actionable_signals(self) -> List[RedditSentimentSignal]:
        """Get all current signals that are actionable for trading"""
        actionable = []
        
        for symbol_signals in self.current_signals.values():
            for signal in symbol_signals:
                if self.sentiment_processor.is_signal_actionable(signal):
                    actionable.append(signal)
                    
        return actionable
        
    async def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get summary of current Reddit sentiment landscape"""
        summary = {
            'total_symbols': len(self.current_signals),
            'total_signals': sum(len(signals) for signals in self.current_signals.values()),
            'actionable_signals': len(await self.get_actionable_signals()),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'stats': self.stats.copy(),
            'rate_limit_status': {}
        }
        
        # Add rate limit status if available
        if self.reddit_client:
            summary['rate_limit_status'] = self.reddit_client.get_rate_limit_status()
            
        # Add symbol breakdown
        symbol_breakdown = {}
        for symbol, signals in self.current_signals.items():
            actionable_count = sum(1 for s in signals if self.sentiment_processor.is_signal_actionable(s))
            avg_sentiment = sum(s.sentiment_score for s in signals) / len(signals) if signals else 0
            avg_quality = sum(s.quality_score for s in signals) / len(signals) if signals else 0
            total_posts = sum(s.post_count for s in signals)
            total_comments = sum(s.comment_count for s in signals)
            
            # Get subreddit distribution for this symbol
            subreddit_dist = {}
            for signal in signals:
                for sub, count in signal.subreddit_distribution.items():
                    subreddit_dist[sub] = subreddit_dist.get(sub, 0) + count
                    
            symbol_breakdown[symbol] = {
                'signal_count': len(signals),
                'actionable_count': actionable_count,
                'avg_sentiment': round(avg_sentiment, 3),
                'avg_quality': round(avg_quality, 3),
                'total_posts': total_posts,
                'total_comments': total_comments,
                'timeframes': [s.timeframe for s in signals],
                'top_subreddits': dict(sorted(subreddit_dist.items(), key=lambda x: x[1], reverse=True)[:3])
            }
            
        summary['symbol_breakdown'] = symbol_breakdown
        
        # Add overall subreddit activity
        top_subreddits = dict(sorted(self.stats['subreddit_coverage'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10])
        summary['top_subreddits'] = top_subreddits
        
        return summary
        
    async def get_trending_discussions(self) -> List[Dict[str, Any]]:
        """Get trending financial discussions across all tracked symbols"""
        trending = []
        
        for symbol, signals in self.current_signals.items():
            for signal in signals:
                if signal.post_count >= 5 and signal.quality_score >= 0.6:
                    trending.extend([
                        {
                            'symbol': symbol,
                            'timeframe': signal.timeframe,
                            'sentiment': signal.sentiment_score,
                            'quality': signal.quality_score,
                            'post_count': signal.post_count,
                            'comment_count': signal.comment_count,
                            'top_post': post
                        }
                        for post in signal.top_posts[:1]  # Just the top post
                    ])
                    
        # Sort by combined quality and activity score
        trending.sort(key=lambda x: (x['quality'] * x['post_count']), reverse=True)
        
        return trending[:20]  # Top 20 trending discussions
        
    def get_status(self) -> Dict[str, Any]:
        """Get component status following system patterns"""
        base_status = super().get_status()
        
        # Add Reddit-specific status
        base_status.update({
            'enabled': self.enabled,
            'tracked_symbols': len(self.tracked_symbols),
            'current_signals': sum(len(signals) for signals in self.current_signals.values()),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'stats': self.stats.copy(),
            'active_subreddits': len(self.stats['subreddit_coverage']),
            'collection_health': self._get_collection_health()
        })
        
        return base_status
        
    def _get_collection_health(self) -> Dict[str, Any]:
        """Get collection health metrics"""
        # Time since last update
        if self.last_update:
            minutes_since_update = (datetime.utcnow() - self.last_update).total_seconds() / 60
        else:
            minutes_since_update = float('inf')
            
        # Signal generation rate
        total_symbols = len(self.tracked_symbols)
        signals_per_symbol = (sum(len(signals) for signals in self.current_signals.values()) / 
                             max(total_symbols, 1))
        
        return {
            'minutes_since_last_update': round(minutes_since_update, 1),
            'is_healthy': minutes_since_update < (self.update_interval / 60) * 2,  # Within 2x interval
            'signals_per_symbol': round(signals_per_symbol, 1),
            'actionable_rate': (self.stats['actionable_signals'] / 
                              max(self.stats['signals_generated'], 1)),
            'subreddit_diversity': len(self.stats['subreddit_coverage'])
        }
        
    async def _load_state(self) -> None:
        """Load persisted component state"""
        try:
            state_data = await self.load_memory()
            if state_data:
                self.current_signals = {}  # Don't persist signals (they get stale faster on Reddit)
                self.signal_history = state_data.get('signal_history', [])[:400]  # Limit on load
                self.stats = state_data.get('stats', self.stats)
                self.symbol_rotation_index = state_data.get('symbol_rotation_index', 0)
                
                self.logger.info(f"Loaded Reddit sentiment state with {len(self.signal_history)} historical entries")
                
        except Exception as e:
            self.logger.error(f"Error loading Reddit sentiment state: {e}")
            
    async def _save_state(self) -> None:
        """Save current component state"""
        try:
            state_data = {
                'signal_history': self.signal_history[-400:],  # Keep recent history
                'stats': self.stats,
                'symbol_rotation_index': self.symbol_rotation_index,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
            await self.save_memory(state_data)
            
        except Exception as e:
            self.logger.error(f"Error saving Reddit sentiment state: {e}")
            
    async def load_memory(self) -> Dict[str, Any]:
        """Load memory from persistent storage"""
        try:
            memory_file = f"/app/data/{self.name}_memory.json"
            with open(memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
            return {}
            
    async def save_memory(self, data: Dict[str, Any]) -> None:
        """Save memory to persistent storage"""
        try:
            memory_file = f"/app/data/{self.name}_memory.json"
            with open(memory_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)  # Convert datetime to string
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
            
    async def add_symbol(self, symbol: str) -> bool:
        """
        Add a new symbol to tracking list
        
        Args:
            symbol: Stock symbol to add
            
        Returns:
            True if added successfully
        """
        if symbol not in self.tracked_symbols:
            self.tracked_symbols.append(symbol.upper())
            self.logger.info(f"Added {symbol} to Reddit sentiment tracking")
            return True
        return False
        
    async def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from tracking list
        
        Args:
            symbol: Stock symbol to remove
            
        Returns:
            True if removed successfully
        """
        if symbol in self.tracked_symbols:
            self.tracked_symbols.remove(symbol)
            # Also remove current signals
            self.current_signals.pop(symbol, None)
            self.logger.info(f"Removed {symbol} from Reddit sentiment tracking")
            return True
        return False
        
    def reset_stats(self) -> None:
        """Reset collection statistics"""
        self.stats = {
            'total_posts_processed': 0,
            'total_comments_processed': 0,
            'signals_generated': 0,
            'actionable_signals': 0,
            'api_calls_made': 0,
            'rate_limit_hits': 0,
            'subreddit_coverage': {},
            'last_reset_time': datetime.utcnow()
        }
        
    async def force_update(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Force an immediate update for specified symbols (or all if None)
        Useful for testing and manual triggers
        
        Args:
            symbols: List of symbols to update, or None for all tracked symbols
            
        Returns:
            Update results summary
        """
        if symbols is None:
            symbols = self.tracked_symbols
        else:
            symbols = [s for s in symbols if s in self.tracked_symbols]
            
        if not symbols:
            return {'error': 'No valid symbols specified'}
            
        self.logger.info(f"Forcing Reddit sentiment update for symbols: {symbols}")
        
        results = {
            'symbols_processed': [],
            'signals_generated': 0,
            'errors': []
        }
        
        for symbol in symbols:
            try:
                await self._process_symbol_sentiment(symbol)
                results['symbols_processed'].append(symbol)
                
                if symbol in self.current_signals:
                    results['signals_generated'] += len(self.current_signals[symbol])
                    
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        self.last_update = datetime.utcnow()
        await self._save_state()
        
        return results
"""
Twitter Sentiment Collector Component
Collects and processes Twitter sentiment data for financial symbols following the system architecture patterns
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ..core.base import ComponentBase
from ..data_sources.twitter_client import TwitterClient
from ..data_sources.twitter_sentiment_processor import TwitterSentimentProcessor, TwitterSentimentSignal


class TwitterSentimentCollector(ComponentBase):
    """
    Component for collecting and processing Twitter sentiment data
    Integrates with the existing NewsAnalysisBrain for multi-source sentiment fusion
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("twitter_sentiment_collector", config)
        
        # Component configuration
        twitter_config = config.get('twitter', {})
        self.enabled = twitter_config.get('enabled', False)
        self.update_interval = twitter_config.get('update_interval', 300)  # 5 minutes default
        
        # Symbol tracking
        self.tracked_symbols = twitter_config.get('tracked_symbols', [
            'AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT', 'NVDA', 'META', 'SPY'
        ])
        self.max_symbols_per_cycle = twitter_config.get('max_symbols_per_cycle', 5)
        
        # Analysis configuration
        self.timeframes = twitter_config.get('timeframes', ['1h', '4h', '12h'])
        self.min_signal_quality = twitter_config.get('min_signal_quality', 0.4)
        self.signal_cache_hours = twitter_config.get('signal_cache_hours', 24)
        
        # Components
        self.twitter_client: Optional[TwitterClient] = None
        self.sentiment_processor: Optional[TwitterSentimentProcessor] = None
        
        # Internal state
        self.current_signals: Dict[str, List[TwitterSentimentSignal]] = {}  # symbol -> signals
        self.signal_history: List[Dict[str, Any]] = []
        self.last_update: Optional[datetime] = None
        self.symbol_rotation_index = 0  # For round-robin symbol processing
        
        # Performance tracking
        self.stats = {
            'total_tweets_processed': 0,
            'signals_generated': 0,
            'actionable_signals': 0,
            'api_calls_made': 0,
            'rate_limit_hits': 0,
            'last_reset_time': datetime.utcnow()
        }
        
    async def start(self) -> None:
        """Initialize the Twitter sentiment collector"""
        if not self.enabled:
            self.logger.info("Twitter sentiment collection is disabled")
            return
            
        self.logger.info("Starting Twitter Sentiment Collector")
        
        try:
            # Initialize Twitter client
            self.twitter_client = TwitterClient(self.config)
            await self.twitter_client.start()
            
            # Initialize sentiment processor
            self.sentiment_processor = TwitterSentimentProcessor(self.config)
            
            # Load persisted state
            await self._load_state()
            
            # Start background collection task
            asyncio.create_task(self._collection_loop())
            
            self.is_running = True
            self.logger.info("Twitter sentiment collector started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Twitter sentiment collector: {e}")
            self.is_running = False
            
    async def stop(self) -> None:
        """Stop the Twitter sentiment collector"""
        self.logger.info("Stopping Twitter Sentiment Collector")
        
        # Save current state
        await self._save_state()
        
        # Stop Twitter client
        if self.twitter_client:
            await self.twitter_client.stop()
            
        self.is_running = False
        
    async def _collection_loop(self) -> None:
        """Main collection loop - runs continuously"""
        while self.is_running:
            try:
                await self._collect_and_process()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(min(self.update_interval, 60))  # Wait at least 1 minute on error
                
    async def _collect_and_process(self) -> None:
        """Collect tweets and process sentiment for tracked symbols"""
        if not self.twitter_client or not self.sentiment_processor:
            return
            
        # Select symbols to process this cycle (round-robin to respect rate limits)
        symbols_to_process = self._get_symbols_for_cycle()
        
        self.logger.info(f"Processing Twitter sentiment for symbols: {symbols_to_process}")
        
        for symbol in symbols_to_process:
            try:
                await self._process_symbol_sentiment(symbol)
                
                # Small delay between symbols to be nice to the API
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error processing sentiment for {symbol}: {e}")
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
            # Collect tweets for the symbol
            tweets = await self.twitter_client.search_financial_tweets([symbol], hours_back=24)
            
            if not tweets:
                self.logger.debug(f"No tweets found for {symbol}")
                return
                
            self.stats['total_tweets_processed'] += len(tweets)
            self.stats['api_calls_made'] += 1
            
            # Process sentiment for different timeframes
            symbol_signals = []
            
            for timeframe in self.timeframes:
                signal = await self.sentiment_processor.process_symbol_sentiment(
                    symbol, tweets, timeframe
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
                
                self.logger.info(f"Generated {len(symbol_signals)} sentiment signals for {symbol}")
            else:
                self.logger.debug(f"No quality signals generated for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error processing sentiment for {symbol}: {e}")
            
    def _add_to_signal_history(self, symbol: str, signals: List[TwitterSentimentSignal]) -> None:
        """Add signals to historical record"""
        timestamp = datetime.utcnow().isoformat()
        
        for signal in signals:
            history_entry = {
                'timestamp': timestamp,
                'symbol': symbol,
                'timeframe': signal.timeframe,
                'sentiment_score': signal.sentiment_score,
                'confidence': signal.confidence,
                'volume_score': signal.volume_score,
                'influence_score': signal.influence_score,
                'tweet_count': signal.tweet_count,
                'quality_score': self.sentiment_processor.get_signal_quality_score(signal),
                'actionable': self.sentiment_processor.is_signal_actionable(signal),
                'sentiment_distribution': signal.sentiment_distribution,
                'keywords': signal.keywords[:10]  # Limit keywords for storage
            }
            
            self.signal_history.append(history_entry)
            
        # Limit history size
        max_history = 1000
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
            
    def _log_collection_summary(self) -> None:
        """Log summary of collection cycle"""
        current_signals_count = sum(len(signals) for signals in self.current_signals.values())
        
        self.logger.info(
            f"Twitter collection summary: "
            f"{self.stats['total_tweets_processed']} tweets processed, "
            f"{current_signals_count} current signals, "
            f"{self.stats['actionable_signals']} actionable signals"
        )
        
    async def get_symbol_sentiment(self, symbol: str, timeframe: str = "4h") -> Optional[TwitterSentimentSignal]:
        """
        Get current sentiment signal for a specific symbol and timeframe
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Timeframe for signal (1h, 4h, 12h)
            
        Returns:
            TwitterSentimentSignal or None if not available
        """
        symbol_signals = self.current_signals.get(symbol, [])
        
        for signal in symbol_signals:
            if signal.timeframe == timeframe:
                return signal
                
        return None
        
    async def get_all_current_signals(self) -> Dict[str, List[TwitterSentimentSignal]]:
        """Get all current sentiment signals"""
        return self.current_signals.copy()
        
    async def get_actionable_signals(self) -> List[TwitterSentimentSignal]:
        """Get all current signals that are actionable for trading"""
        actionable = []
        
        for symbol_signals in self.current_signals.values():
            for signal in symbol_signals:
                if self.sentiment_processor.is_signal_actionable(signal):
                    actionable.append(signal)
                    
        return actionable
        
    async def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get summary of current sentiment landscape"""
        summary = {
            'total_symbols': len(self.current_signals),
            'total_signals': sum(len(signals) for signals in self.current_signals.values()),
            'actionable_signals': len(await self.get_actionable_signals()),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'stats': self.stats.copy(),
            'rate_limit_status': {}
        }
        
        # Add rate limit status if available
        if self.twitter_client:
            summary['rate_limit_status'] = self.twitter_client.get_rate_limit_status()
            
        # Add symbol breakdown
        symbol_breakdown = {}
        for symbol, signals in self.current_signals.items():
            actionable_count = sum(1 for s in signals if self.sentiment_processor.is_signal_actionable(s))
            avg_sentiment = sum(s.sentiment_score for s in signals) / len(signals) if signals else 0
            
            symbol_breakdown[symbol] = {
                'signal_count': len(signals),
                'actionable_count': actionable_count,
                'avg_sentiment': round(avg_sentiment, 3),
                'timeframes': [s.timeframe for s in signals]
            }
            
        summary['symbol_breakdown'] = symbol_breakdown
        
        return summary
        
    def get_status(self) -> Dict[str, Any]:
        """Get component status following system patterns"""
        base_status = super().get_status()
        
        # Add Twitter-specific status
        base_status.update({
            'enabled': self.enabled,
            'tracked_symbols': len(self.tracked_symbols),
            'current_signals': sum(len(signals) for signals in self.current_signals.values()),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'stats': self.stats.copy()
        })
        
        return base_status
        
    async def _load_state(self) -> None:
        """Load persisted component state"""
        try:
            state_data = await self.load_memory()
            if state_data:
                self.current_signals = {}  # Don't persist signals (they get stale)
                self.signal_history = state_data.get('signal_history', [])[:500]  # Limit on load
                self.stats = state_data.get('stats', self.stats)
                self.symbol_rotation_index = state_data.get('symbol_rotation_index', 0)
                
                self.logger.info(f"Loaded Twitter sentiment state with {len(self.signal_history)} historical entries")
                
        except Exception as e:
            self.logger.error(f"Error loading Twitter sentiment state: {e}")
            
    async def _save_state(self) -> None:
        """Save current component state"""
        try:
            state_data = {
                'signal_history': self.signal_history[-500:],  # Keep recent history
                'stats': self.stats,
                'symbol_rotation_index': self.symbol_rotation_index,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
            await self.save_memory(state_data)
            
        except Exception as e:
            self.logger.error(f"Error saving Twitter sentiment state: {e}")
            
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
            self.logger.info(f"Added {symbol} to Twitter sentiment tracking")
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
            self.logger.info(f"Removed {symbol} from Twitter sentiment tracking")
            return True
        return False
        
    def reset_stats(self) -> None:
        """Reset collection statistics"""
        self.stats = {
            'total_tweets_processed': 0,
            'signals_generated': 0,
            'actionable_signals': 0,
            'api_calls_made': 0,
            'rate_limit_hits': 0,
            'last_reset_time': datetime.utcnow()
        }
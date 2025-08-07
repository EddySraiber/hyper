from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from ..core.base import PersistentComponent
import asyncio
import logging


class TrailingStop:
    """Represents a trailing stop-loss order for a position"""
    
    def __init__(self, symbol: str, side: str, quantity: int, entry_price: float,
                 initial_stop_price: float, trailing_amount: float, trailing_percent: Optional[float] = None):
        self.symbol = symbol
        self.side = side  # "long" or "short"
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_stop_price = initial_stop_price
        self.trailing_amount = trailing_amount  # Dollar amount to trail by
        self.trailing_percent = trailing_percent  # Percentage to trail by (alternative to amount)
        
        # Tracking variables
        self.highest_price = entry_price if side == "long" else entry_price  # For long positions
        self.lowest_price = entry_price if side == "short" else entry_price  # For short positions
        self.last_updated = datetime.utcnow()
        self.created_at = datetime.utcnow()
        self.times_adjusted = 0
        self.total_protection_gained = 0.0  # How much extra protection we've gained
        
        # Noise filtering
        self.min_move_threshold = 0.01  # Minimum price move to consider (1 cent default)
        self.consolidation_period = 300  # 5 minutes in seconds
        self.last_significant_move = datetime.utcnow()
        
    def should_update_stop(self, current_price: float, noise_filter: bool = True) -> Tuple[bool, float, str]:
        """
        Determine if stop should be updated based on current price
        Returns: (should_update, new_stop_price, reason)
        """
        now = datetime.utcnow()
        
        if self.side == "long":
            return self._should_update_long_stop(current_price, now, noise_filter)
        else:
            return self._should_update_short_stop(current_price, now, noise_filter)
    
    def _should_update_long_stop(self, current_price: float, now: datetime, noise_filter: bool) -> Tuple[bool, float, str]:
        """Handle trailing stop logic for long positions"""
        
        # Check if price moved up significantly from our highest tracked price
        if current_price > self.highest_price:
            price_move = current_price - self.highest_price
            
            # Noise filtering - ignore tiny moves unless enough time has passed
            if noise_filter and price_move < self.min_move_threshold:
                time_since_last_move = (now - self.last_significant_move).total_seconds()
                if time_since_last_move < self.consolidation_period:
                    return False, self.current_stop_price, "price_move_too_small"
            
            # Update highest price and calculate new stop
            self.highest_price = current_price
            self.last_significant_move = now
            
            if self.trailing_percent:
                # Percentage-based trailing
                new_stop = current_price * (1 - self.trailing_percent / 100)
            else:
                # Dollar-based trailing
                new_stop = current_price - self.trailing_amount
            
            # Only move stop up, never down (for long positions)
            if new_stop > self.current_stop_price:
                protection_gained = new_stop - self.current_stop_price
                return True, new_stop, f"trailing_up_${protection_gained:.2f}_protection"
        
        return False, self.current_stop_price, "no_update_needed"
    
    def _should_update_short_stop(self, current_price: float, now: datetime, noise_filter: bool) -> Tuple[bool, float, str]:
        """Handle trailing stop logic for short positions"""
        
        # Check if price moved down significantly from our lowest tracked price
        if current_price < self.lowest_price:
            price_move = self.lowest_price - current_price
            
            # Noise filtering - ignore tiny moves unless enough time has passed
            if noise_filter and price_move < self.min_move_threshold:
                time_since_last_move = (now - self.last_significant_move).total_seconds()
                if time_since_last_move < self.consolidation_period:
                    return False, self.current_stop_price, "price_move_too_small"
            
            # Update lowest price and calculate new stop
            self.lowest_price = current_price
            self.last_significant_move = now
            
            if self.trailing_percent:
                # Percentage-based trailing
                new_stop = current_price * (1 + self.trailing_percent / 100)
            else:
                # Dollar-based trailing
                new_stop = current_price + self.trailing_amount
            
            # Only move stop down, never up (for short positions)
            if new_stop < self.current_stop_price:
                protection_gained = self.current_stop_price - new_stop
                return True, new_stop, f"trailing_down_${protection_gained:.2f}_protection"
        
        return False, self.current_stop_price, "no_update_needed"
    
    def update_stop_price(self, new_stop_price: float, reason: str):
        """Update the stop price and tracking metrics"""
        old_stop = self.current_stop_price
        self.current_stop_price = new_stop_price
        self.last_updated = datetime.utcnow()
        self.times_adjusted += 1
        
        if self.side == "long":
            self.total_protection_gained += (new_stop_price - old_stop)
        else:
            self.total_protection_gained += (old_stop - new_stop_price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_stop_price": self.current_stop_price,
            "trailing_amount": self.trailing_amount,
            "trailing_percent": self.trailing_percent,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "last_updated": self.last_updated.isoformat(),
            "created_at": self.created_at.isoformat(),
            "times_adjusted": self.times_adjusted,
            "total_protection_gained": self.total_protection_gained,
            "min_move_threshold": self.min_move_threshold,
            "consolidation_period": self.consolidation_period
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrailingStop':
        """Create TrailingStop from dictionary"""
        trailing_stop = cls(
            symbol=data["symbol"],
            side=data["side"],
            quantity=data["quantity"],
            entry_price=data["entry_price"],
            initial_stop_price=data["current_stop_price"],
            trailing_amount=data["trailing_amount"],
            trailing_percent=data.get("trailing_percent")
        )
        
        # Restore state
        trailing_stop.highest_price = data.get("highest_price", data["entry_price"])
        trailing_stop.lowest_price = data.get("lowest_price", data["entry_price"])
        trailing_stop.last_updated = datetime.fromisoformat(data["last_updated"])
        trailing_stop.created_at = datetime.fromisoformat(data["created_at"])
        trailing_stop.times_adjusted = data.get("times_adjusted", 0)
        trailing_stop.total_protection_gained = data.get("total_protection_gained", 0.0)
        trailing_stop.min_move_threshold = data.get("min_move_threshold", 0.01)
        trailing_stop.consolidation_period = data.get("consolidation_period", 300)
        
        return trailing_stop


class TrailingStopManager(PersistentComponent):
    """
    Manages trailing stops for all positions to protect profits while letting winners run
    
    Key Features:
    - Automatic trailing stop adjustment as prices move favorably
    - Noise filtering to prevent premature exits on small moves
    - Separate logic for long and short positions
    - Configurable trailing amounts (dollar or percentage)
    - Profit protection thresholds
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("trailing_stop_manager", config)
        
        # Configuration parameters
        self.enable_trailing_stops = config.get("enable_trailing_stops", True)
        self.default_trailing_percent = config.get("default_trailing_percent", 3.0)  # 3% default
        self.default_trailing_amount = config.get("default_trailing_amount", 0.50)   # $0.50 default
        self.min_profit_threshold = config.get("min_profit_threshold", 0.02)        # 2% min profit before trailing
        self.max_daily_updates_per_symbol = config.get("max_daily_updates_per_symbol", 10)  # Prevent over-trading
        
        # Noise filtering settings
        self.noise_filter_enabled = config.get("noise_filter_enabled", True)
        self.min_move_threshold = config.get("min_move_threshold", 0.01)            # 1 cent minimum move
        self.consolidation_period_minutes = config.get("consolidation_period_minutes", 5)  # 5 minute consolidation
        
        # Performance settings
        self.update_interval = config.get("update_interval", 60)  # Check every 60 seconds
        self.max_concurrent_updates = config.get("max_concurrent_updates", 5)  # Limit API calls
        
        # Dependencies (will be injected)
        self.alpaca_client = None
        
        # State tracking
        self.trailing_stops: Dict[str, TrailingStop] = {}  # symbol -> TrailingStop
        self.daily_update_counts: Dict[str, int] = {}      # symbol -> count of updates today
        self.last_reset_date = datetime.utcnow().date()
        
    def start(self) -> None:
        self.logger.info("Starting Trailing Stop Manager")
        self._load_memory()
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping Trailing Stop Manager")
        self._save_memory()
        self.is_running = False
    
    def _load_memory(self) -> None:
        """Load trailing stops from persistent memory"""
        try:
            memory = self.load_memory()
            if memory:
                # Restore trailing stops
                stops_data = memory.get("trailing_stops", {})
                for symbol, stop_data in stops_data.items():
                    self.trailing_stops[symbol] = TrailingStop.from_dict(stop_data)
                
                # Restore daily counts
                self.daily_update_counts = memory.get("daily_update_counts", {})
                
                # Check if we need to reset daily counts
                last_date_str = memory.get("last_reset_date")
                if last_date_str:
                    last_date = datetime.fromisoformat(last_date_str).date()
                    if last_date != datetime.utcnow().date():
                        self.daily_update_counts = {}  # Reset for new day
                
                self.logger.info(f"Loaded {len(self.trailing_stops)} trailing stops from memory")
                
        except Exception as e:
            self.logger.error(f"Error loading trailing stop memory: {e}")
    
    def _save_memory(self) -> None:
        """Save trailing stops to persistent memory"""
        try:
            memory = {
                "trailing_stops": {symbol: stop.to_dict() for symbol, stop in self.trailing_stops.items()},
                "daily_update_counts": self.daily_update_counts,
                "last_reset_date": datetime.utcnow().isoformat()
            }
            self.save_memory(memory)
            self.logger.debug(f"Saved {len(self.trailing_stops)} trailing stops to memory")
        except Exception as e:
            self.logger.error(f"Error saving trailing stop memory: {e}")
    
    async def add_trailing_stop(self, symbol: str, side: str, quantity: int, entry_price: float,
                              initial_stop_price: float, trailing_amount: Optional[float] = None,
                              trailing_percent: Optional[float] = None) -> bool:
        """
        Add a new trailing stop for a position
        
        Args:
            symbol: Stock symbol
            side: "long" or "short"
            quantity: Number of shares
            entry_price: Original entry price
            initial_stop_price: Initial stop-loss price
            trailing_amount: Dollar amount to trail by (optional)
            trailing_percent: Percentage to trail by (optional, alternative to amount)
        
        Returns:
            True if successfully added, False otherwise
        """
        if not self.enable_trailing_stops:
            return False
        
        try:
            # Use defaults if not specified
            if not trailing_amount and not trailing_percent:
                trailing_percent = self.default_trailing_percent
            
            # Create trailing stop
            trailing_stop = TrailingStop(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                initial_stop_price=initial_stop_price,
                trailing_amount=trailing_amount or self.default_trailing_amount,
                trailing_percent=trailing_percent
            )
            
            # Configure noise filtering
            trailing_stop.min_move_threshold = self.min_move_threshold
            trailing_stop.consolidation_period = self.consolidation_period_minutes * 60
            
            self.trailing_stops[symbol] = trailing_stop
            self._save_memory()
            
            self.logger.info(f"🎯 Added trailing stop for {symbol} ({side}): "
                           f"entry=${entry_price:.2f}, stop=${initial_stop_price:.2f}, "
                           f"trailing={trailing_percent or trailing_amount}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding trailing stop for {symbol}: {e}")
            return False
    
    async def update_trailing_stops(self) -> Dict[str, Any]:
        """
        Update all trailing stops based on current prices
        
        Returns:
            Dictionary with update results and statistics
        """
        if not self.is_running or not self.trailing_stops or not self.alpaca_client:
            return {"updated": 0, "errors": 0, "total": 0}
        
        # Reset daily counts if new day
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_update_counts = {}
            self.last_reset_date = current_date
        
        results = {
            "updated": 0,
            "errors": 0,
            "total": len(self.trailing_stops),
            "updates": [],
            "skipped": []
        }
        
        # Process trailing stops in batches to avoid overwhelming the API
        symbols_to_process = list(self.trailing_stops.keys())
        
        for i in range(0, len(symbols_to_process), self.max_concurrent_updates):
            batch = symbols_to_process[i:i + self.max_concurrent_updates]
            await self._process_batch(batch, results)
            
            # Small delay between batches
            if i + self.max_concurrent_updates < len(symbols_to_process):
                await asyncio.sleep(1)
        
        # Save updated state
        if results["updated"] > 0:
            self._save_memory()
            self.logger.info(f"🔄 Trailing stops update: {results['updated']} updated, "
                           f"{len(results['skipped'])} skipped, {results['errors']} errors")
        
        return results
    
    async def _process_batch(self, symbols: List[str], results: Dict[str, Any]):
        """Process a batch of symbols for trailing stop updates"""
        
        # Get current prices for all symbols in batch
        price_tasks = []
        for symbol in symbols:
            if symbol in self.trailing_stops:
                price_tasks.append(self._get_current_price_safe(symbol))
        
        if not price_tasks:
            return
        
        # Fetch prices concurrently
        prices = await asyncio.gather(*price_tasks, return_exceptions=True)
        
        # Process each symbol
        for symbol, current_price in zip(symbols, prices):
            if symbol not in self.trailing_stops:
                continue
                
            trailing_stop = self.trailing_stops[symbol]
            
            try:
                # Handle price fetch errors
                if isinstance(current_price, Exception) or current_price is None:
                    results["errors"] += 1
                    results["skipped"].append(f"{symbol}: price_fetch_failed")
                    continue
                
                # Check daily update limit
                daily_count = self.daily_update_counts.get(symbol, 0)
                if daily_count >= self.max_daily_updates_per_symbol:
                    results["skipped"].append(f"{symbol}: daily_limit_reached")
                    continue
                
                # Check if position should be updated
                should_update, new_stop_price, reason = trailing_stop.should_update_stop(
                    current_price, self.noise_filter_enabled
                )
                
                if should_update:
                    # Check if we've reached minimum profit threshold
                    if self._meets_profit_threshold(trailing_stop, current_price):
                        # Update the trailing stop
                        success = await self._update_stop_order(symbol, new_stop_price)
                        
                        if success:
                            old_stop = trailing_stop.current_stop_price
                            trailing_stop.update_stop_price(new_stop_price, reason)
                            
                            # Track daily updates
                            self.daily_update_counts[symbol] = daily_count + 1
                            
                            results["updated"] += 1
                            results["updates"].append({
                                "symbol": symbol,
                                "old_stop": old_stop,
                                "new_stop": new_stop_price,
                                "current_price": current_price,
                                "reason": reason,
                                "protection_gained": trailing_stop.total_protection_gained
                            })
                            
                            self.logger.info(f"📈 {symbol} trailing stop updated: "
                                           f"${old_stop:.2f} → ${new_stop_price:.2f} "
                                           f"(current: ${current_price:.2f}) - {reason}")
                        else:
                            results["errors"] += 1
                    else:
                        results["skipped"].append(f"{symbol}: below_profit_threshold")
                else:
                    results["skipped"].append(f"{symbol}: {reason}")
                    
            except Exception as e:
                results["errors"] += 1
                self.logger.error(f"Error processing trailing stop for {symbol}: {e}")
    
    async def _get_current_price_safe(self, symbol: str) -> Optional[float]:
        """Safely get current price with error handling"""
        try:
            return await self.alpaca_client.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _meets_profit_threshold(self, trailing_stop: TrailingStop, current_price: float) -> bool:
        """Check if position meets minimum profit threshold before trailing"""
        if trailing_stop.side == "long":
            profit_pct = (current_price - trailing_stop.entry_price) / trailing_stop.entry_price
        else:
            profit_pct = (trailing_stop.entry_price - current_price) / trailing_stop.entry_price
        
        return profit_pct >= self.min_profit_threshold
    
    async def _update_stop_order(self, symbol: str, new_stop_price: float) -> bool:
        """Update the actual stop order with the broker"""
        try:
            # This would integrate with the broker's API to update the stop order
            # For now, we'll implement a placeholder that logs the action
            self.logger.info(f"🔄 Would update {symbol} stop order to ${new_stop_price:.2f}")
            
            # TODO: Implement actual broker integration
            # success = await self.alpaca_client.update_stop_order(symbol, new_stop_price)
            # return success
            
            return True  # Placeholder - always return success for now
            
        except Exception as e:
            self.logger.error(f"Error updating stop order for {symbol}: {e}")
            return False
    
    def remove_trailing_stop(self, symbol: str) -> bool:
        """Remove trailing stop for a symbol (when position is closed)"""
        try:
            if symbol in self.trailing_stops:
                del self.trailing_stops[symbol]
                if symbol in self.daily_update_counts:
                    del self.daily_update_counts[symbol]
                self._save_memory()
                self.logger.info(f"🗑️ Removed trailing stop for {symbol}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing trailing stop for {symbol}: {e}")
            return False
    
    def get_trailing_stop_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get status of trailing stops"""
        if symbol and symbol in self.trailing_stops:
            stop = self.trailing_stops[symbol]
            return {
                symbol: {
                    "current_stop": stop.current_stop_price,
                    "entry_price": stop.entry_price,
                    "side": stop.side,
                    "times_adjusted": stop.times_adjusted,
                    "protection_gained": stop.total_protection_gained,
                    "last_updated": stop.last_updated.isoformat(),
                    "daily_updates": self.daily_update_counts.get(symbol, 0)
                }
            }
        else:
            return {
                sym: {
                    "current_stop": stop.current_stop_price,
                    "entry_price": stop.entry_price,
                    "side": stop.side,
                    "times_adjusted": stop.times_adjusted,
                    "protection_gained": stop.total_protection_gained,
                    "last_updated": stop.last_updated.isoformat(),
                    "daily_updates": self.daily_update_counts.get(sym, 0)
                }
                for sym, stop in self.trailing_stops.items()
            }
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method called by the system"""
        if not self.is_running:
            return {"status": "not_running"}
        
        return await self.update_trailing_stops()
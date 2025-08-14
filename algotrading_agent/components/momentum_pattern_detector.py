"""
Momentum Pattern Detector - High-speed pattern recognition for fast trading

Detects rapid market movements, volatility breakouts, and momentum patterns
for sub-minute trading execution. Designed to capture "Monday losses" style
opportunities with automated pattern classification.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from ..core.base import ComponentBase
from ..trading.alpaca_client import AlpacaClient


class PatternType(Enum):
    """Types of momentum patterns that trigger fast trades"""
    FLASH_CRASH = "flash_crash"           # >5% drop in <5 minutes
    FLASH_SURGE = "flash_surge"           # >5% rise in <5 minutes  
    EARNINGS_SURPRISE = "earnings_surprise"  # Beat/miss estimates
    REGULATORY_SHOCK = "regulatory_shock"    # FDA approval, lawsuit
    VOLUME_BREAKOUT = "volume_breakout"      # Unusual volume spike
    MOMENTUM_CONTINUATION = "momentum_continuation"  # Trend acceleration
    REVERSAL_PATTERN = "reversal_pattern"    # Sharp direction change
    NEWS_VELOCITY_SPIKE = "news_velocity_spike"  # Rapid news coverage


@dataclass
class PatternSignal:
    """Detected momentum pattern signal"""
    symbol: str
    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    speed_target: int  # seconds for execution
    volatility: float  # expected volatility
    direction: str     # "bullish", "bearish", "neutral"
    
    # Pattern specifics
    trigger_price: float
    current_price: float
    price_change_pct: float
    volume_ratio: float  # vs average volume
    
    # Timing
    detected_at: datetime
    expires_at: datetime
    
    # Additional context
    triggers: List[str]  # What triggered this pattern
    risk_level: str     # "low", "medium", "high", "extreme"
    expected_duration_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "confidence": self.confidence,
            "speed_target": self.speed_target,
            "volatility": self.volatility,
            "direction": self.direction,
            "trigger_price": self.trigger_price,
            "current_price": self.current_price,
            "price_change_pct": self.price_change_pct,
            "volume_ratio": self.volume_ratio,
            "detected_at": self.detected_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "triggers": self.triggers,
            "risk_level": self.risk_level,
            "expected_duration_minutes": self.expected_duration_minutes
        }


class MomentumPatternDetector(ComponentBase):
    """
    High-frequency momentum pattern detection for fast trading
    
    Analyzes price movements, volume patterns, and news velocity to identify
    rapid trading opportunities that require sub-minute execution.
    """
    
    def __init__(self, config: Dict[str, Any], alpaca_client: AlpacaClient):
        super().__init__("momentum_pattern_detector", config)
        self.alpaca_client = alpaca_client
        
        # Configuration
        self.scan_interval = config.get("scan_interval", 10)  # 10 seconds
        self.lookback_minutes = config.get("lookback_minutes", 15)
        self.min_confidence = config.get("min_confidence", 0.6)
        self.max_concurrent_patterns = config.get("max_concurrent_patterns", 50)
        
        # Pattern thresholds
        self.volatility_thresholds = {
            PatternType.FLASH_CRASH: 0.05,      # 5% minimum
            PatternType.FLASH_SURGE: 0.05,      # 5% minimum
            PatternType.EARNINGS_SURPRISE: 0.03, # 3% minimum
            PatternType.REGULATORY_SHOCK: 0.08,  # 8% minimum
            PatternType.VOLUME_BREAKOUT: 0.02,   # 2% minimum with volume
            PatternType.MOMENTUM_CONTINUATION: 0.015, # 1.5% with trend
            PatternType.REVERSAL_PATTERN: 0.04,  # 4% minimum
            PatternType.NEWS_VELOCITY_SPIKE: 0.025  # 2.5% with news
        }
        
        # Speed targets (seconds for execution)
        self.speed_targets = {
            PatternType.FLASH_CRASH: 5,          # Lightning speed
            PatternType.FLASH_SURGE: 5,          # Lightning speed
            PatternType.EARNINGS_SURPRISE: 15,   # Express speed
            PatternType.REGULATORY_SHOCK: 20,    # Fast speed
            PatternType.VOLUME_BREAKOUT: 30,     # Fast speed
            PatternType.MOMENTUM_CONTINUATION: 45, # Standard+ speed
            PatternType.REVERSAL_PATTERN: 25,    # Fast speed
            PatternType.NEWS_VELOCITY_SPIKE: 15  # Express speed
        }
        
        # Tracking
        self.active_patterns: Dict[str, PatternSignal] = {}  # symbol -> pattern
        self.price_history: Dict[str, List[Tuple[datetime, float, float]]] = {}  # symbol -> [(time, price, volume)]
        self.pattern_performance: Dict[PatternType, Dict[str, Any]] = {}
        
        # Watchlist for patterns
        self.monitored_symbols: Set[str] = set()
        self.last_scan_time: Optional[datetime] = None
        
        # Statistics
        self.patterns_detected = 0
        self.patterns_expired = 0
        self.successful_predictions = 0
        
        # Initialize pattern performance tracking
        for pattern_type in PatternType:
            self.pattern_performance[pattern_type] = {
                "detected": 0,
                "successful": 0,
                "failed": 0,
                "avg_confidence": 0.0,
                "avg_execution_time": 0.0,
                "avg_profit": 0.0
            }
    
    async def start(self) -> None:
        """Start momentum pattern detection"""
        self.logger.info("🎯 Starting Momentum Pattern Detector")
        self.is_running = True
        
        # Initialize watchlist from trending symbols
        await self._initialize_watchlist()
        
        self.logger.info(f"✅ Pattern detector started - monitoring {len(self.monitored_symbols)} symbols")
        
    async def stop(self) -> None:
        """Stop pattern detection"""
        self.logger.info("🛑 Stopping Momentum Pattern Detector")
        self.is_running = False
        
        # Log final performance
        self._log_performance_summary()
    
    async def process(self, data: Any = None) -> List[PatternSignal]:
        """Main processing - detect momentum patterns"""
        if not self.is_running:
            return []
        
        self.last_scan_time = datetime.utcnow()
        detected_patterns = []
        
        try:
            # Update price data for monitored symbols
            await self._update_price_data()
            
            # Scan for patterns in parallel
            pattern_tasks = []
            for symbol in self.monitored_symbols:
                pattern_tasks.append(self._detect_patterns_for_symbol(symbol))
            
            # Gather results
            results = await asyncio.gather(*pattern_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                symbol = list(self.monitored_symbols)[i]
                if isinstance(result, Exception):
                    self.logger.warning(f"Pattern detection error for {symbol}: {result}")
                elif isinstance(result, list):
                    detected_patterns.extend(result)
            
            # Clean up expired patterns
            await self._cleanup_expired_patterns()
            
            # Log results
            if detected_patterns:
                self.logger.info(f"🎯 Detected {len(detected_patterns)} momentum patterns")
                for pattern in detected_patterns:
                    self.logger.info(f"   {pattern.symbol}: {pattern.pattern_type.value} "
                                   f"({pattern.confidence:.2f} confidence, {pattern.speed_target}s target)")
            
            self.patterns_detected += len(detected_patterns)
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            return []
    
    async def _initialize_watchlist(self):
        """Initialize watchlist with high-volume symbols"""
        try:
            # Get trending symbols (could be from news, volume, etc.)
            default_watchlist = [
                # High-volume stocks
                "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "NVDA", "META", "NFLX", "AMD", "INTC", "CRM", "UBER", "LYFT",
                
                # High-volatility stocks
                "GME", "AMC", "BBBY", "PLTR", "COIN", "HOOD", "RIVN", "LCID",
                
                # Crypto pairs
                "BTCUSD", "ETHUSD", "DOGEUSD", "SOLUSD", "AVAXUSD"
            ]
            
            # Add symbols that have recent news activity
            news_symbols = getattr(self, 'recent_news_symbols', [])
            
            self.monitored_symbols = set(default_watchlist + news_symbols)
            self.logger.info(f"Initialized watchlist with {len(self.monitored_symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error initializing watchlist: {e}")
            # Fallback to basic list
            self.monitored_symbols = {"SPY", "QQQ", "AAPL", "TSLA", "BTCUSD"}
    
    async def _update_price_data(self):
        """Update price and volume data for all monitored symbols"""
        try:
            # Get current prices and volumes
            current_time = datetime.utcnow()
            
            # For each symbol, get recent price/volume data
            for symbol in list(self.monitored_symbols):
                try:
                    # Get current price
                    current_price = await self.alpaca_client.get_current_price(symbol)
                    if not current_price:
                        continue
                    
                    # Get recent volume (simplified - in real implementation would use bars)
                    volume = 1000000  # Placeholder - would get from market data
                    
                    # Update price history
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    self.price_history[symbol].append((current_time, current_price, volume))
                    
                    # Keep only recent data
                    cutoff_time = current_time - timedelta(minutes=self.lookback_minutes)
                    self.price_history[symbol] = [
                        (time, price, vol) for time, price, vol in self.price_history[symbol]
                        if time > cutoff_time
                    ]
                    
                except Exception as e:
                    self.logger.debug(f"Error updating price for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error updating price data: {e}")
    
    async def _detect_patterns_for_symbol(self, symbol: str) -> List[PatternSignal]:
        """Detect momentum patterns for a specific symbol"""
        patterns = []
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return patterns
        
        price_data = self.price_history[symbol]
        current_time = datetime.utcnow()
        
        try:
            # Calculate price changes and volume ratios
            current_price = price_data[-1][1]
            previous_prices = [p[1] for p in price_data[:-1]]
            current_volume = price_data[-1][2]
            previous_volumes = [p[2] for p in price_data[:-1]]
            
            if not previous_prices:
                return patterns
            
            # Calculate key metrics
            price_5min_ago = previous_prices[-5] if len(previous_prices) >= 5 else previous_prices[0]
            price_change_pct = (current_price - price_5min_ago) / price_5min_ago
            
            avg_volume = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Flash Crash Detection
            if price_change_pct <= -self.volatility_thresholds[PatternType.FLASH_CRASH]:
                pattern = self._create_pattern_signal(
                    symbol, PatternType.FLASH_CRASH, current_price, price_5min_ago,
                    price_change_pct, volume_ratio, 
                    triggers=["rapid_price_decline", f"{price_change_pct*100:.1f}%_drop"],
                    direction="bearish"
                )
                if pattern:
                    patterns.append(pattern)
            
            # Flash Surge Detection
            elif price_change_pct >= self.volatility_thresholds[PatternType.FLASH_SURGE]:
                pattern = self._create_pattern_signal(
                    symbol, PatternType.FLASH_SURGE, current_price, price_5min_ago,
                    price_change_pct, volume_ratio,
                    triggers=["rapid_price_surge", f"{price_change_pct*100:.1f}%_rise"],
                    direction="bullish"
                )
                if pattern:
                    patterns.append(pattern)
            
            # Volume Breakout Detection
            if (volume_ratio >= 3.0 and 
                abs(price_change_pct) >= self.volatility_thresholds[PatternType.VOLUME_BREAKOUT]):
                
                direction = "bullish" if price_change_pct > 0 else "bearish"
                pattern = self._create_pattern_signal(
                    symbol, PatternType.VOLUME_BREAKOUT, current_price, price_5min_ago,
                    price_change_pct, volume_ratio,
                    triggers=["unusual_volume", f"{volume_ratio:.1f}x_normal"],
                    direction=direction
                )
                if pattern:
                    patterns.append(pattern)
            
            # Momentum Continuation Detection
            if len(previous_prices) >= 10:
                recent_trend = self._calculate_trend(price_data[-10:])
                if (abs(recent_trend) > 0.02 and  # 2% trend
                    abs(price_change_pct) >= self.volatility_thresholds[PatternType.MOMENTUM_CONTINUATION] and
                    (recent_trend > 0) == (price_change_pct > 0)):  # Same direction
                    
                    direction = "bullish" if recent_trend > 0 else "bearish"
                    pattern = self._create_pattern_signal(
                        symbol, PatternType.MOMENTUM_CONTINUATION, current_price, price_5min_ago,
                        price_change_pct, volume_ratio,
                        triggers=["momentum_continuation", f"{recent_trend*100:.1f}%_trend"],
                        direction=direction
                    )
                    if pattern:
                        patterns.append(pattern)
            
            # Reversal Pattern Detection
            if len(previous_prices) >= 10:
                recent_trend = self._calculate_trend(price_data[-10:-3])
                current_trend = self._calculate_trend(price_data[-3:])
                
                # Detect trend reversal
                if (abs(recent_trend) > 0.02 and abs(current_trend) > 0.015 and
                    recent_trend * current_trend < 0):  # Opposite directions
                    
                    direction = "bullish" if current_trend > 0 else "bearish"
                    pattern = self._create_pattern_signal(
                        symbol, PatternType.REVERSAL_PATTERN, current_price, price_5min_ago,
                        price_change_pct, volume_ratio,
                        triggers=["trend_reversal", "momentum_shift"],
                        direction=direction
                    )
                    if pattern:
                        patterns.append(pattern)
            
        except Exception as e:
            self.logger.warning(f"Pattern detection error for {symbol}: {e}")
        
        return patterns
    
    def _create_pattern_signal(self, symbol: str, pattern_type: PatternType, 
                              current_price: float, trigger_price: float,
                              price_change_pct: float, volume_ratio: float,
                              triggers: List[str], direction: str) -> Optional[PatternSignal]:
        """Create a pattern signal with all required parameters"""
        
        try:
            # Calculate confidence based on strength of signals
            base_confidence = min(abs(price_change_pct) / self.volatility_thresholds[pattern_type], 1.0)
            volume_boost = min(volume_ratio / 2.0, 0.3)  # Up to 30% boost from volume
            confidence = min(base_confidence + volume_boost, 1.0)
            
            if confidence < self.min_confidence:
                return None
            
            # Calculate risk level
            volatility = abs(price_change_pct)
            if volatility >= 0.10:
                risk_level = "extreme"
            elif volatility >= 0.05:
                risk_level = "high"
            elif volatility >= 0.02:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Set expiration time based on pattern type
            if pattern_type in [PatternType.FLASH_CRASH, PatternType.FLASH_SURGE]:
                expires_in_minutes = 5  # Very short-lived
            elif pattern_type in [PatternType.EARNINGS_SURPRISE, PatternType.NEWS_VELOCITY_SPIKE]:
                expires_in_minutes = 30  # Medium duration
            else:
                expires_in_minutes = 15  # Standard duration
            
            current_time = datetime.utcnow()
            
            pattern = PatternSignal(
                symbol=symbol,
                pattern_type=pattern_type,
                confidence=confidence,
                speed_target=self.speed_targets[pattern_type],
                volatility=volatility,
                direction=direction,
                trigger_price=trigger_price,
                current_price=current_price,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                detected_at=current_time,
                expires_at=current_time + timedelta(minutes=expires_in_minutes),
                triggers=triggers,
                risk_level=risk_level,
                expected_duration_minutes=expires_in_minutes
            )
            
            # Update active patterns (remove existing if any)
            self.active_patterns[symbol] = pattern
            
            # Update performance tracking
            self.pattern_performance[pattern_type]["detected"] += 1
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error creating pattern signal for {symbol}: {e}")
            return None
    
    def _calculate_trend(self, price_data: List[Tuple[datetime, float, float]]) -> float:
        """Calculate price trend over given data points"""
        if len(price_data) < 2:
            return 0.0
        
        prices = [p[1] for p in price_data]
        start_price = prices[0]
        end_price = prices[-1]
        
        return (end_price - start_price) / start_price
    
    async def _cleanup_expired_patterns(self):
        """Remove expired patterns from tracking"""
        current_time = datetime.utcnow()
        expired_symbols = []
        
        for symbol, pattern in self.active_patterns.items():
            if current_time > pattern.expires_at:
                expired_symbols.append(symbol)
                self.patterns_expired += 1
        
        for symbol in expired_symbols:
            del self.active_patterns[symbol]
        
        if expired_symbols:
            self.logger.debug(f"Cleaned up {len(expired_symbols)} expired patterns")
    
    def get_active_patterns(self) -> List[PatternSignal]:
        """Get all currently active patterns"""
        return list(self.active_patterns.values())
    
    def get_patterns_for_symbol(self, symbol: str) -> Optional[PatternSignal]:
        """Get active pattern for specific symbol"""
        return self.active_patterns.get(symbol)
    
    def add_news_symbols(self, symbols: List[str]):
        """Add symbols that have recent news activity"""
        initial_count = len(self.monitored_symbols)
        self.monitored_symbols.update(symbols)
        new_count = len(self.monitored_symbols) - initial_count
        
        if new_count > 0:
            self.logger.info(f"Added {new_count} news-driven symbols to watchlist")
        
        # Store for pattern detection
        self.recent_news_symbols = symbols
    
    def update_pattern_performance(self, symbol: str, success: bool, profit: float = 0.0):
        """Update pattern performance tracking"""
        if symbol in self.active_patterns:
            pattern = self.active_patterns[symbol]
            perf = self.pattern_performance[pattern.pattern_type]
            
            if success:
                perf["successful"] += 1
                perf["avg_profit"] = (perf["avg_profit"] + profit) / 2
                self.successful_predictions += 1
            else:
                perf["failed"] += 1
            
            # Update average confidence
            total_detected = perf["detected"]
            if total_detected > 0:
                perf["avg_confidence"] = (perf["avg_confidence"] * (total_detected - 1) + pattern.confidence) / total_detected
    
    def _log_performance_summary(self):
        """Log pattern detection performance summary"""
        self.logger.info("🎯 MOMENTUM PATTERN DETECTOR PERFORMANCE:")
        self.logger.info(f"  Total patterns detected: {self.patterns_detected}")
        self.logger.info(f"  Patterns expired: {self.patterns_expired}")
        self.logger.info(f"  Successful predictions: {self.successful_predictions}")
        self.logger.info(f"  Active patterns: {len(self.active_patterns)}")
        
        for pattern_type, perf in self.pattern_performance.items():
            if perf["detected"] > 0:
                success_rate = (perf["successful"] / perf["detected"] * 100) if perf["detected"] > 0 else 0
                self.logger.info(f"  {pattern_type.value}: {perf['detected']} detected, "
                               f"{success_rate:.1f}% success rate, avg confidence: {perf['avg_confidence']:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            "is_running": self.is_running,
            "monitored_symbols": len(self.monitored_symbols),
            "active_patterns": len(self.active_patterns),
            "patterns_detected": self.patterns_detected,
            "patterns_expired": self.patterns_expired,
            "successful_predictions": self.successful_predictions,
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "pattern_performance": {
                pt.value: perf for pt, perf in self.pattern_performance.items()
            },
            "current_patterns": [p.to_dict() for p in self.active_patterns.values()]
        }
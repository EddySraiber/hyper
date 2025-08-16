#!/usr/bin/env python3
"""
Market Regime Detector - Adaptive Strategy Selection
Detects market regimes to optimize trading strategy performance

Key Regimes:
- Bull/Bear/Sideways markets
- High/Low volatility environments  
- Risk-on/Risk-off sentiment
- Sector rotation patterns
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from algotrading_agent.core.base import ComponentBase


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TRENDING = "bull_trending"          # Strong upward momentum
    BEAR_TRENDING = "bear_trending"          # Strong downward momentum
    SIDEWAYS_RANGING = "sideways_ranging"    # Choppy, no clear direction
    HIGH_VOLATILITY = "high_volatility"      # Elevated volatility environment
    LOW_VOLATILITY = "low_volatility"        # Suppressed volatility environment
    RISK_ON = "risk_on"                      # Risk appetite high
    RISK_OFF = "risk_off"                    # Flight to safety


@dataclass
class RegimeSignal:
    """Market regime detection signal"""
    regime: MarketRegime
    confidence: float
    duration_days: int
    strength: float
    supporting_indicators: List[str]
    timestamp: datetime


class MarketRegimeDetector(ComponentBase):
    """
    Advanced market regime detection for adaptive trading strategies
    
    Uses multiple market indicators to detect regime changes:
    - Price momentum and trend strength
    - Volatility clustering and mean reversion
    - Market breadth and sector rotation
    - Sentiment indicators and risk appetite
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("market_regime_detector", config)
        self.config = config
        self.logger = logging.getLogger(f"algotrading.{self.__class__.__name__.lower()}")
        
        # Configuration parameters
        self.lookback_periods = config.get("lookback_periods", {
            "short": 10,    # 10 days for short-term signals
            "medium": 30,   # 30 days for medium-term regime
            "long": 90      # 90 days for long-term context
        })
        
        self.volatility_thresholds = config.get("volatility_thresholds", {
            "high": 0.25,   # 25% annual volatility threshold
            "low": 0.10     # 10% annual volatility threshold
        })
        
        self.trend_thresholds = config.get("trend_thresholds", {
            "bull": 0.05,   # 5% price appreciation over medium term
            "bear": -0.05,  # 5% price depreciation over medium term
            "sideways": 0.02  # +/- 2% for sideways classification
        })
        
        self.regime_confidence_threshold = config.get("regime_confidence_threshold", 0.60)
        self.regime_persistence_days = config.get("regime_persistence_days", 5)
        
        # Market data storage
        self.price_history = []  # Store recent price data
        self.volume_history = []  # Store volume data
        self.volatility_history = []  # Store volatility measures
        
        # Current regime state
        self.current_regime = None
        self.regime_start_date = None
        self.regime_confidence = 0.0
        self.regime_history = []
        
        # Initialize component
        self.is_running = False
        
    def start(self) -> None:
        """Start the market regime detector"""
        self.logger.info("Starting Market Regime Detector")
        self.is_running = True
        
    def stop(self) -> None:
        """Stop the market regime detector"""
        self.logger.info("Stopping Market Regime Detector")
        self.is_running = False
        
    async def detect_regime(self, market_data: Dict[str, Any]) -> Optional[RegimeSignal]:
        """
        Main regime detection method
        
        Args:
            market_data: Current market data including prices, volumes, etc.
            
        Returns:
            RegimeSignal if regime detected, None otherwise
        """
        if not self.is_running:
            return None
            
        try:
            # Update market data history
            self._update_market_data(market_data)
            
            # Only analyze if we have sufficient data
            if len(self.price_history) < self.lookback_periods["short"]:
                self.logger.debug("Insufficient data for regime detection")
                return None
            
            # Detect different regime aspects
            trend_regime = self._detect_trend_regime()
            volatility_regime = self._detect_volatility_regime()
            sentiment_regime = self._detect_sentiment_regime(market_data)
            
            # Combine regime signals
            primary_regime = self._combine_regime_signals(trend_regime, volatility_regime, sentiment_regime)
            
            if primary_regime:
                # Update current regime
                self._update_current_regime(primary_regime)
                
                self.logger.info(f"🎯 REGIME DETECTED: {primary_regime.regime.value} "
                               f"(confidence: {primary_regime.confidence:.2f}, "
                               f"duration: {primary_regime.duration_days} days)")
                
                return primary_regime
                
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            
        return None
    
    def _update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update market data history"""
        # Extract key market indicators
        current_price = market_data.get("current_price", 100.0)
        current_volume = market_data.get("volume", 1000000)
        
        # Update histories (keep only necessary lookback)
        max_lookback = max(self.lookback_periods.values())
        
        self.price_history.append(current_price)
        if len(self.price_history) > max_lookback:
            self.price_history.pop(0)
            
        self.volume_history.append(current_volume)
        if len(self.volume_history) > max_lookback:
            self.volume_history.pop(0)
            
        # Calculate volatility
        if len(self.price_history) >= 5:  # Need minimum data for volatility
            recent_prices = self.price_history[-10:]  # Last 10 days
            returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i-1] > 0:  # Avoid log(0)
                    ret = (recent_prices[i] / recent_prices[i-1]) - 1
                    returns.append(ret)
            
            if returns:
                current_volatility = statistics.stdev(returns) * (252 ** 0.5)  # Annualized
                self.volatility_history.append(current_volatility)
            
            if len(self.volatility_history) > max_lookback:
                self.volatility_history.pop(0)
    
    def _detect_trend_regime(self) -> Optional[RegimeSignal]:
        """Detect trend-based market regime"""
        if len(self.price_history) < self.lookback_periods["medium"]:
            return None
            
        # Calculate trend metrics
        short_prices = self.price_history[-self.lookback_periods["short"]:]
        medium_prices = self.price_history[-self.lookback_periods["medium"]:]
        
        # Short-term and medium-term returns
        short_return = (short_prices[-1] - short_prices[0]) / short_prices[0]
        medium_return = (medium_prices[-1] - medium_prices[0]) / medium_prices[0]
        
        # Trend consistency (what % of recent days were positive)
        short_daily_returns = []
        for i in range(1, len(short_prices)):
            if short_prices[i-1] != 0:
                daily_return = (short_prices[i] - short_prices[i-1]) / short_prices[i-1]
                short_daily_returns.append(daily_return)
        
        positive_days_pct = sum(1 for ret in short_daily_returns if ret > 0) / len(short_daily_returns) if short_daily_returns else 0
        
        # Determine regime
        supporting_indicators = []
        
        if medium_return > self.trend_thresholds["bull"] and positive_days_pct > 0.6:
            regime = MarketRegime.BULL_TRENDING
            confidence = min(0.9, medium_return * 5 + positive_days_pct * 0.5)
            supporting_indicators = ["medium_term_uptrend", "positive_day_majority"]
            
        elif medium_return < self.trend_thresholds["bear"] and positive_days_pct < 0.4:
            regime = MarketRegime.BEAR_TRENDING
            confidence = min(0.9, abs(medium_return) * 5 + (1 - positive_days_pct) * 0.5)
            supporting_indicators = ["medium_term_downtrend", "negative_day_majority"]
            
        elif abs(medium_return) < self.trend_thresholds["sideways"]:
            regime = MarketRegime.SIDEWAYS_RANGING
            confidence = min(0.8, 1 - abs(medium_return) * 10)
            supporting_indicators = ["low_directional_movement"]
            
        else:
            return None
        
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            duration_days=self.lookback_periods["medium"],
            strength=abs(medium_return),
            supporting_indicators=supporting_indicators,
            timestamp=datetime.now()
        )
    
    def _detect_volatility_regime(self) -> Optional[RegimeSignal]:
        """Detect volatility-based market regime"""
        if len(self.volatility_history) < self.lookback_periods["short"]:
            return None
            
        # Current volatility vs historical
        current_vol = self.volatility_history[-1]
        recent_vols = self.volatility_history[-self.lookback_periods["medium"]:]
        avg_vol = statistics.mean(recent_vols) if recent_vols else current_vol
        
        # Volatility regime detection
        if current_vol > self.volatility_thresholds["high"]:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(0.9, current_vol / self.volatility_thresholds["high"] - 1)
            supporting_indicators = ["elevated_volatility"]
            
        elif current_vol < self.volatility_thresholds["low"]:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(0.9, 1 - current_vol / self.volatility_thresholds["low"])
            supporting_indicators = ["suppressed_volatility"]
            
        else:
            return None
        
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            duration_days=len(self.volatility_history),
            strength=current_vol,
            supporting_indicators=supporting_indicators,
            timestamp=datetime.now()
        )
    
    def _detect_sentiment_regime(self, market_data: Dict[str, Any]) -> Optional[RegimeSignal]:
        """Detect sentiment-based market regime"""
        # Extract sentiment indicators from market data
        news_sentiment = market_data.get("average_sentiment", 0.0)
        market_breadth = market_data.get("market_breadth", 0.5)  # % of stocks advancing
        
        # Risk-on/Risk-off detection
        if news_sentiment > 0.3 and market_breadth > 0.6:
            regime = MarketRegime.RISK_ON
            confidence = min(0.8, news_sentiment + market_breadth - 0.5)
            supporting_indicators = ["positive_sentiment", "broad_market_strength"]
            
        elif news_sentiment < -0.3 and market_breadth < 0.4:
            regime = MarketRegime.RISK_OFF
            confidence = min(0.8, abs(news_sentiment) + (0.5 - market_breadth))
            supporting_indicators = ["negative_sentiment", "market_weakness"]
            
        else:
            return None
        
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            duration_days=1,  # Sentiment can change daily
            strength=abs(news_sentiment),
            supporting_indicators=supporting_indicators,
            timestamp=datetime.now()
        )
    
    def _combine_regime_signals(self, *signals: Optional[RegimeSignal]) -> Optional[RegimeSignal]:
        """Combine multiple regime signals into primary regime"""
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return None
        
        # Find highest confidence signal
        primary_signal = max(valid_signals, key=lambda s: s.confidence)
        
        # Only return if confidence exceeds threshold
        if primary_signal.confidence >= self.regime_confidence_threshold:
            return primary_signal
        
        return None
    
    def _update_current_regime(self, new_regime: RegimeSignal) -> None:
        """Update current regime state"""
        regime_changed = (self.current_regime is None or 
                         self.current_regime.regime != new_regime.regime)
        
        if regime_changed:
            self.logger.info(f"🔄 REGIME CHANGE: {self.current_regime.regime.value if self.current_regime else 'None'} "
                           f"→ {new_regime.regime.value}")
            self.regime_start_date = datetime.now()
            
        self.current_regime = new_regime
        self.regime_confidence = new_regime.confidence
        
        # Add to history
        self.regime_history.append({
            'regime': new_regime.regime.value,
            'confidence': new_regime.confidence,
            'timestamp': new_regime.timestamp.isoformat(),
            'duration_days': new_regime.duration_days
        })
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
    
    def get_current_regime(self) -> Optional[RegimeSignal]:
        """Get current market regime"""
        return self.current_regime
    
    def get_regime_adjustments(self) -> Dict[str, Any]:
        """
        Get trading parameter adjustments based on current regime
        
        Returns:
            Dictionary of parameter adjustments for different strategies
        """
        if not self.current_regime:
            return {}
        
        regime = self.current_regime.regime
        confidence = self.current_regime.confidence
        
        adjustments = {
            'position_size_multiplier': 1.0,
            'confidence_threshold_adjustment': 0.0,
            'strategy_preference': 'momentum',  # or 'mean_reversion'
            'risk_multiplier': 1.0,
            'holding_period_adjustment': 1.0
        }
        
        # Regime-specific adjustments
        if regime == MarketRegime.BULL_TRENDING:
            adjustments.update({
                'position_size_multiplier': 1.2,  # Increase position sizes
                'confidence_threshold_adjustment': -0.05,  # Lower confidence threshold
                'strategy_preference': 'momentum',
                'risk_multiplier': 1.1,
                'holding_period_adjustment': 1.2  # Hold positions longer
            })
            
        elif regime == MarketRegime.BEAR_TRENDING:
            adjustments.update({
                'position_size_multiplier': 0.8,  # Reduce position sizes
                'confidence_threshold_adjustment': 0.05,  # Raise confidence threshold
                'strategy_preference': 'momentum',
                'risk_multiplier': 0.8,
                'holding_period_adjustment': 0.8  # Shorter holding periods
            })
            
        elif regime == MarketRegime.SIDEWAYS_RANGING:
            adjustments.update({
                'position_size_multiplier': 0.9,
                'strategy_preference': 'mean_reversion',  # Better for ranging markets
                'risk_multiplier': 0.9,
                'holding_period_adjustment': 0.7  # Quick in and out
            })
            
        elif regime == MarketRegime.HIGH_VOLATILITY:
            adjustments.update({
                'position_size_multiplier': 0.7,  # Much smaller positions
                'confidence_threshold_adjustment': 0.10,  # Much higher confidence needed
                'risk_multiplier': 0.6,  # Very conservative
                'holding_period_adjustment': 0.5  # Very short holding periods
            })
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            adjustments.update({
                'position_size_multiplier': 1.3,  # Can take larger positions
                'confidence_threshold_adjustment': -0.05,
                'risk_multiplier': 1.2,
                'holding_period_adjustment': 1.5  # Can hold longer
            })
        
        # Scale adjustments by confidence
        for key in ['position_size_multiplier', 'risk_multiplier']:
            base_value = adjustments[key]
            # Scale deviation from 1.0 by confidence
            deviation = (base_value - 1.0) * confidence
            adjustments[key] = 1.0 + deviation
        
        adjustments['confidence_threshold_adjustment'] *= confidence
        
        return adjustments
    
    async def process(self, market_data: Dict[str, Any]) -> Optional[RegimeSignal]:
        """
        Process method required by ComponentBase
        
        Args:
            market_data: Market data for regime detection
            
        Returns:
            Detected regime signal or None
        """
        return await self.detect_regime(market_data)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'is_running': self.is_running,
            'current_regime': self.current_regime.regime.value if self.current_regime else None,
            'regime_confidence': self.regime_confidence,
            'regime_duration': (datetime.now() - self.regime_start_date).days if self.regime_start_date else 0,
            'data_points': len(self.price_history),
            'recent_regimes': self.regime_history[-5:] if self.regime_history else []
        }
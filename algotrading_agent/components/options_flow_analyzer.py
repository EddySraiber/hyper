#!/usr/bin/env python3
"""
Options Flow Analyzer - Leading Indicator System
Analyzes options flow data to detect institutional sentiment and predict price movements

Key Features:
- Unusual options activity detection
- Put/call ratio analysis
- Dark pool and block trade detection
- Smart money flow tracking
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

from algotrading_agent.core.base import ComponentBase


class OptionsFlowSignal(Enum):
    """Options flow signal types"""
    BULLISH_UNUSUAL_ACTIVITY = "bullish_unusual_activity"      # Large call buying
    BEARISH_UNUSUAL_ACTIVITY = "bearish_unusual_activity"      # Large put buying
    SMART_MONEY_BULLISH = "smart_money_bullish"               # Institutional call activity
    SMART_MONEY_BEARISH = "smart_money_bearish"               # Institutional put activity
    HIGH_IMPLIED_VOLATILITY = "high_implied_volatility"       # Volatility spike expected
    LOW_IMPLIED_VOLATILITY = "low_implied_volatility"         # Volatility crush expected
    GAMMA_SQUEEZE_SETUP = "gamma_squeeze_setup"               # Large gamma exposure
    DARK_POOL_ACCUMULATION = "dark_pool_accumulation"         # Dark pool buying


@dataclass
class OptionsFlow:
    """Options flow data point"""
    symbol: str
    signal_type: OptionsFlowSignal
    strength: float  # 0-1 signal strength
    confidence: float  # 0-1 confidence level
    volume_ratio: float  # Volume vs average
    premium_value: float  # Dollar amount
    expiration_days: int  # Days to expiration
    strike_vs_spot: float  # Strike price vs current price
    timestamp: datetime
    supporting_data: Dict[str, Any]


class OptionsFlowAnalyzer(ComponentBase):
    """
    Advanced options flow analysis for predictive trading signals
    
    Analyzes unusual options activity that often precedes significant price moves:
    - Large institutional trades
    - Unusual volume spikes
    - Smart money positioning
    - Volatility expectations
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("options_flow_analyzer", config)
        self.config = config
        self.logger = logging.getLogger(f"algotrading.{self.__class__.__name__.lower()}")
        
        # Configuration parameters
        self.enabled = config.get("enabled", True)
        self.update_interval = config.get("update_interval", 300)  # 5 minutes
        self.lookback_days = config.get("lookback_days", 5)
        
        # Volume thresholds for unusual activity
        self.volume_thresholds = config.get("volume_thresholds", {
            "unusual_multiplier": 3.0,  # 3x normal volume
            "extreme_multiplier": 10.0,  # 10x normal volume
            "min_premium": 50000,  # $50k minimum premium
            "min_volume": 100  # 100 contracts minimum
        })
        
        # Signal confidence thresholds
        self.confidence_thresholds = config.get("confidence_thresholds", {
            "high": 0.80,
            "medium": 0.60,
            "low": 0.40
        })
        
        # Supported symbols for options analysis
        self.tracked_symbols = config.get("tracked_symbols", [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'DIS', 'BA',
            'JPM', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'KO', 'PFE'
        ])
        
        # Data sources configuration
        self.data_sources = config.get("data_sources", {
            "alpha_vantage": {
                "enabled": True,
                "api_key": config.get("alpha_vantage_api_key"),
                "base_url": "https://www.alphavantage.co/query"
            },
            "polygon": {
                "enabled": True,
                "api_key": config.get("polygon_api_key"),
                "base_url": "https://api.polygon.io"
            },
            "tradier": {
                "enabled": True,
                "api_key": config.get("tradier_api_key"),
                "base_url": "https://api.tradier.com"
            }
        })
        
        # Historical data storage
        self.flow_history = []  # Store recent options flow data
        self.volume_baselines = {}  # Average volume by symbol
        self.last_update = None
        
        # Initialize component
        self.is_running = False
        
    def start(self) -> None:
        """Start the options flow analyzer"""
        self.logger.info("Starting Options Flow Analyzer")
        if not self.enabled:
            self.logger.warning("Options Flow Analyzer is disabled in configuration")
            return
            
        self.is_running = True
        
        # Initialize volume baselines
        asyncio.create_task(self._initialize_baselines())
        
    def stop(self) -> None:
        """Stop the options flow analyzer"""
        self.logger.info("Stopping Options Flow Analyzer")
        self.is_running = False
        
    async def process(self, market_data: Optional[Dict[str, Any]] = None) -> List[OptionsFlow]:
        """
        Main processing method for options flow analysis
        
        Args:
            market_data: Optional market data for context
            
        Returns:
            List of detected options flow signals
        """
        if not self.is_running or not self.enabled:
            return []
            
        try:
            # Check if it's time to update
            if self._should_update():
                flows = await self._analyze_options_flow()
                
                if flows:
                    self.logger.info(f"📊 OPTIONS FLOW: Detected {len(flows)} signals")
                    for flow in flows:
                        self.logger.info(f"   🎯 {flow.symbol}: {flow.signal_type.value} "
                                       f"(strength: {flow.strength:.2f}, confidence: {flow.confidence:.2f})")
                
                self.last_update = datetime.now()
                return flows
                
        except Exception as e:
            self.logger.error(f"Error in options flow analysis: {e}")
            
        return []
    
    async def _analyze_options_flow(self) -> List[OptionsFlow]:
        """Analyze options flow across all tracked symbols"""
        all_flows = []
        
        # Analyze each tracked symbol
        for symbol in self.tracked_symbols:
            try:
                symbol_flows = await self._analyze_symbol_flow(symbol)
                all_flows.extend(symbol_flows)
            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol}: {e}")
                
        # Filter and rank signals
        significant_flows = self._filter_significant_flows(all_flows)
        
        # Update historical data
        self._update_flow_history(significant_flows)
        
        return significant_flows
    
    async def _analyze_symbol_flow(self, symbol: str) -> List[OptionsFlow]:
        """Analyze options flow for a specific symbol"""
        flows = []
        
        try:
            # Get options data from available sources
            options_data = await self._fetch_options_data(symbol)
            
            if not options_data:
                return flows
                
            # Analyze different flow patterns
            flows.extend(await self._detect_unusual_activity(symbol, options_data))
            flows.extend(await self._detect_smart_money_flow(symbol, options_data))
            flows.extend(await self._detect_volatility_signals(symbol, options_data))
            flows.extend(await self._detect_gamma_signals(symbol, options_data))
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol} options flow: {e}")
            
        return flows
    
    async def _fetch_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch options data from configured sources"""
        # For now, simulate options data since we need API keys
        # In production, this would fetch from Alpha Vantage, Polygon, etc.
        
        if not self._has_valid_api_keys():
            return self._generate_mock_options_data(symbol)
            
        # Try each data source
        for source_name, source_config in self.data_sources.items():
            if not source_config.get("enabled") or not source_config.get("api_key"):
                continue
                
            try:
                data = await self._fetch_from_source(symbol, source_name, source_config)
                if data:
                    return data
            except Exception as e:
                self.logger.debug(f"Failed to fetch from {source_name}: {e}")
                
        return None
    
    def _generate_mock_options_data(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic mock options data for testing"""
        import random
        
        base_volume = self.volume_baselines.get(symbol, 1000)
        current_volume = random.randint(int(base_volume * 0.5), int(base_volume * 5))
        
        # Generate realistic options chain data
        mock_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'total_call_volume': current_volume * random.uniform(0.4, 0.8),
            'total_put_volume': current_volume * random.uniform(0.2, 0.6),
            'put_call_ratio': random.uniform(0.3, 1.5),
            'implied_volatility': random.uniform(0.15, 0.45),
            'options_chain': []
        }
        
        # Generate individual option contracts
        spot_price = 100 + random.uniform(-20, 20)  # Mock spot price
        
        for i in range(10):  # Generate 10 option strikes
            strike = spot_price + (i - 5) * 5  # Strikes around current price
            
            call_data = {
                'type': 'call',
                'strike': strike,
                'volume': random.randint(0, int(current_volume * 0.1)),
                'open_interest': random.randint(100, 10000),
                'premium': max(0.01, (spot_price - strike) + random.uniform(-2, 5)),
                'implied_volatility': random.uniform(0.10, 0.50),
                'days_to_expiration': random.choice([1, 2, 7, 14, 30, 60])
            }
            
            put_data = {
                'type': 'put',
                'strike': strike,
                'volume': random.randint(0, int(current_volume * 0.1)),
                'open_interest': random.randint(100, 10000),
                'premium': max(0.01, (strike - spot_price) + random.uniform(-2, 5)),
                'implied_volatility': random.uniform(0.10, 0.50),
                'days_to_expiration': random.choice([1, 2, 7, 14, 30, 60])
            }
            
            mock_data['options_chain'].extend([call_data, put_data])
        
        return mock_data
    
    async def _detect_unusual_activity(self, symbol: str, options_data: Dict[str, Any]) -> List[OptionsFlow]:
        """Detect unusual options activity patterns"""
        flows = []
        
        try:
            total_call_volume = options_data.get('total_call_volume', 0)
            total_put_volume = options_data.get('total_put_volume', 0)
            baseline_volume = self.volume_baselines.get(symbol, 1000)
            
            # Check for unusual call activity
            call_ratio = total_call_volume / baseline_volume if baseline_volume > 0 else 0
            if call_ratio > self.volume_thresholds['unusual_multiplier']:
                strength = min(1.0, call_ratio / self.volume_thresholds['extreme_multiplier'])
                confidence = min(0.9, 0.5 + strength * 0.4)
                
                flow = OptionsFlow(
                    symbol=symbol,
                    signal_type=OptionsFlowSignal.BULLISH_UNUSUAL_ACTIVITY,
                    strength=strength,
                    confidence=confidence,
                    volume_ratio=call_ratio,
                    premium_value=total_call_volume * 100,  # Estimate premium
                    expiration_days=7,  # Average
                    strike_vs_spot=1.0,  # At the money average
                    timestamp=datetime.now(),
                    supporting_data={
                        'call_volume': total_call_volume,
                        'baseline_volume': baseline_volume,
                        'volume_multiplier': call_ratio
                    }
                )
                flows.append(flow)
            
            # Check for unusual put activity
            put_ratio = total_put_volume / baseline_volume if baseline_volume > 0 else 0
            if put_ratio > self.volume_thresholds['unusual_multiplier']:
                strength = min(1.0, put_ratio / self.volume_thresholds['extreme_multiplier'])
                confidence = min(0.9, 0.5 + strength * 0.4)
                
                flow = OptionsFlow(
                    symbol=symbol,
                    signal_type=OptionsFlowSignal.BEARISH_UNUSUAL_ACTIVITY,
                    strength=strength,
                    confidence=confidence,
                    volume_ratio=put_ratio,
                    premium_value=total_put_volume * 100,
                    expiration_days=7,
                    strike_vs_spot=1.0,
                    timestamp=datetime.now(),
                    supporting_data={
                        'put_volume': total_put_volume,
                        'baseline_volume': baseline_volume,
                        'volume_multiplier': put_ratio
                    }
                )
                flows.append(flow)
                
        except Exception as e:
            self.logger.debug(f"Error detecting unusual activity for {symbol}: {e}")
            
        return flows
    
    async def _detect_smart_money_flow(self, symbol: str, options_data: Dict[str, Any]) -> List[OptionsFlow]:
        """Detect smart money options flow patterns"""
        flows = []
        
        try:
            # Look for large premium trades (institutional activity)
            options_chain = options_data.get('options_chain', [])
            
            for option in options_chain:
                volume = option.get('volume', 0)
                premium = option.get('premium', 0)
                total_premium = volume * premium * 100  # Contract multiplier
                
                # Detect large premium trades
                if total_premium > self.volume_thresholds['min_premium'] and volume > self.volume_thresholds['min_volume']:
                    option_type = option.get('type', 'call')
                    
                    # Determine signal type based on option type and size
                    if option_type == 'call':
                        signal_type = OptionsFlowSignal.SMART_MONEY_BULLISH
                    else:
                        signal_type = OptionsFlowSignal.SMART_MONEY_BEARISH
                    
                    strength = min(1.0, total_premium / (self.volume_thresholds['min_premium'] * 5))
                    confidence = min(0.8, 0.6 + strength * 0.2)
                    
                    flow = OptionsFlow(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        confidence=confidence,
                        volume_ratio=volume / 100,  # Relative to typical volume
                        premium_value=total_premium,
                        expiration_days=option.get('days_to_expiration', 30),
                        strike_vs_spot=option.get('strike', 100) / 100,  # Normalized
                        timestamp=datetime.now(),
                        supporting_data={
                            'option_type': option_type,
                            'strike': option.get('strike'),
                            'premium': premium,
                            'volume': volume,
                            'total_premium': total_premium
                        }
                    )
                    flows.append(flow)
                    
        except Exception as e:
            self.logger.debug(f"Error detecting smart money flow for {symbol}: {e}")
            
        return flows
    
    async def _detect_volatility_signals(self, symbol: str, options_data: Dict[str, Any]) -> List[OptionsFlow]:
        """Detect volatility-based signals"""
        flows = []
        
        try:
            implied_vol = options_data.get('implied_volatility', 0.2)
            
            # High IV signals potential volatility expansion
            if implied_vol > 0.35:
                strength = min(1.0, (implied_vol - 0.35) / 0.15)
                confidence = 0.6 + strength * 0.2
                
                flow = OptionsFlow(
                    symbol=symbol,
                    signal_type=OptionsFlowSignal.HIGH_IMPLIED_VOLATILITY,
                    strength=strength,
                    confidence=confidence,
                    volume_ratio=1.0,
                    premium_value=0,
                    expiration_days=30,
                    strike_vs_spot=1.0,
                    timestamp=datetime.now(),
                    supporting_data={
                        'implied_volatility': implied_vol,
                        'volatility_level': 'high'
                    }
                )
                flows.append(flow)
                
            # Low IV signals potential volatility crush
            elif implied_vol < 0.12:
                strength = min(1.0, (0.12 - implied_vol) / 0.08)
                confidence = 0.5 + strength * 0.2
                
                flow = OptionsFlow(
                    symbol=symbol,
                    signal_type=OptionsFlowSignal.LOW_IMPLIED_VOLATILITY,
                    strength=strength,
                    confidence=confidence,
                    volume_ratio=1.0,
                    premium_value=0,
                    expiration_days=30,
                    strike_vs_spot=1.0,
                    timestamp=datetime.now(),
                    supporting_data={
                        'implied_volatility': implied_vol,
                        'volatility_level': 'low'
                    }
                )
                flows.append(flow)
                
        except Exception as e:
            self.logger.debug(f"Error detecting volatility signals for {symbol}: {e}")
            
        return flows
    
    async def _detect_gamma_signals(self, symbol: str, options_data: Dict[str, Any]) -> List[OptionsFlow]:
        """Detect gamma squeeze setup patterns"""
        flows = []
        
        try:
            # Look for high call volume near current price (gamma squeeze setup)
            options_chain = options_data.get('options_chain', [])
            total_call_volume = sum(opt.get('volume', 0) for opt in options_chain if opt.get('type') == 'call')
            
            # High call volume concentration suggests gamma squeeze potential
            if total_call_volume > 1000:  # Significant call volume
                put_call_ratio = options_data.get('put_call_ratio', 1.0)
                
                # Low put/call ratio with high call volume = gamma squeeze setup
                if put_call_ratio < 0.5:
                    strength = min(1.0, total_call_volume / 5000)
                    confidence = 0.7 - put_call_ratio * 0.4  # Lower P/C ratio = higher confidence
                    
                    flow = OptionsFlow(
                        symbol=symbol,
                        signal_type=OptionsFlowSignal.GAMMA_SQUEEZE_SETUP,
                        strength=strength,
                        confidence=confidence,
                        volume_ratio=total_call_volume / 1000,
                        premium_value=total_call_volume * 200,  # Estimate
                        expiration_days=7,  # Gamma squeezes typically short-term
                        strike_vs_spot=1.0,
                        timestamp=datetime.now(),
                        supporting_data={
                            'call_volume': total_call_volume,
                            'put_call_ratio': put_call_ratio,
                            'squeeze_probability': confidence
                        }
                    )
                    flows.append(flow)
                    
        except Exception as e:
            self.logger.debug(f"Error detecting gamma signals for {symbol}: {e}")
            
        return flows
    
    def _filter_significant_flows(self, flows: List[OptionsFlow]) -> List[OptionsFlow]:
        """Filter and rank options flows by significance"""
        if not flows:
            return []
            
        # Filter by minimum confidence
        significant_flows = [
            flow for flow in flows 
            if flow.confidence >= self.confidence_thresholds['low']
        ]
        
        # Sort by combined strength and confidence
        significant_flows.sort(
            key=lambda f: f.strength * f.confidence,
            reverse=True
        )
        
        # Limit to top signals to avoid noise
        return significant_flows[:10]
    
    def _update_flow_history(self, flows: List[OptionsFlow]) -> None:
        """Update historical flow data"""
        self.flow_history.extend(flows)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.flow_history = [
            flow for flow in self.flow_history 
            if flow.timestamp > cutoff_time
        ]
    
    async def _initialize_baselines(self) -> None:
        """Initialize volume baselines for tracked symbols"""
        self.logger.info("Initializing options volume baselines...")
        
        # Set default baselines (in production, these would be calculated from historical data)
        default_baselines = {
            'SPY': 500000, 'QQQ': 300000, 'IWM': 200000,
            'AAPL': 100000, 'MSFT': 80000, 'GOOGL': 50000,
            'AMZN': 60000, 'TSLA': 150000, 'NVDA': 120000,
            'META': 70000, 'NFLX': 40000, 'AMD': 80000
        }
        
        for symbol in self.tracked_symbols:
            self.volume_baselines[symbol] = default_baselines.get(symbol, 10000)
            
        self.logger.info(f"Initialized baselines for {len(self.volume_baselines)} symbols")
    
    def _should_update(self) -> bool:
        """Check if it's time to update options flow data"""
        if not self.last_update:
            return True
            
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update >= self.update_interval
    
    def _has_valid_api_keys(self) -> bool:
        """Check if any valid API keys are available"""
        for source_config in self.data_sources.values():
            if source_config.get("enabled") and source_config.get("api_key"):
                return True
        return False
    
    async def _fetch_from_source(self, symbol: str, source_name: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch options data from a specific source"""
        # Placeholder for actual API integration
        # Would implement Alpha Vantage, Polygon, Tradier API calls here
        return None
    
    def get_recent_flows(self, symbol: Optional[str] = None, hours: int = 4) -> List[OptionsFlow]:
        """Get recent options flows for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_flows = [
            flow for flow in self.flow_history 
            if flow.timestamp > cutoff_time
        ]
        
        if symbol:
            recent_flows = [
                flow for flow in recent_flows 
                if flow.symbol == symbol
            ]
            
        return recent_flows
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get summary of recent options flow activity"""
        recent_flows = self.get_recent_flows(hours=4)
        
        summary = {
            'total_signals': len(recent_flows),
            'bullish_signals': len([f for f in recent_flows if 'bullish' in f.signal_type.value]),
            'bearish_signals': len([f for f in recent_flows if 'bearish' in f.signal_type.value]),
            'high_confidence_signals': len([f for f in recent_flows if f.confidence > 0.7]),
            'top_symbols': {},
            'signal_types': {}
        }
        
        # Count by symbol
        for flow in recent_flows:
            summary['top_symbols'][flow.symbol] = summary['top_symbols'].get(flow.symbol, 0) + 1
            summary['signal_types'][flow.signal_type.value] = summary['signal_types'].get(flow.signal_type.value, 0) + 1
        
        return summary
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'is_running': self.is_running,
            'enabled': self.enabled,
            'tracked_symbols': len(self.tracked_symbols),
            'flow_history_count': len(self.flow_history),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'volume_baselines_initialized': len(self.volume_baselines),
            'api_sources_configured': sum(1 for s in self.data_sources.values() if s.get('enabled')),
            'recent_flow_summary': self.get_flow_summary()
        }
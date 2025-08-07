from typing import Dict, List, Any, Optional
import re
import math
from datetime import datetime, timedelta
from ..core.base import ComponentBase


class NewsImpactScorer(ComponentBase):
    """
    Advanced news impact scoring system to measure market-moving potential
    
    Scores news based on:
    1. Source Authority (Reuters > Bloomberg > CNBC > etc.)
    2. Market Exposure (company size, sector impact)
    3. Content Intensity (keywords, numbers, superlatives)
    4. Timing & Freshness (market hours, breaking vs stale)
    5. Hype Amplification (buzz words, social media indicators)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("news_impact_scorer", config)
        
        # Source authority rankings (higher = more market impact)
        self.source_authority = {
            # Tier 1: Market Moving Sources
            'Reuters': 1.0,
            'Bloomberg': 1.0,
            'Wall Street Journal': 0.95,
            'Financial Times': 0.95,
            
            # Tier 2: Mainstream Finance
            'CNBC': 0.85,
            'MarketWatch': 0.80,
            'Yahoo Finance': 0.75,
            'Seeking Alpha': 0.70,
            
            # Tier 3: General News
            'CNN Business': 0.65,
            'AP News': 0.60,
            'Business Insider': 0.55,
            
            # Default for unknown sources
            'Unknown': 0.30
        }
        
        # Market exposure factors (company/sector importance)
        self.market_exposure = {
            # Mega Cap (>$1T market cap)
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
            'mega_cap_weight': 1.0,
            
            # Large Cap ($10B-$1T)  
            'large_cap': ['TSLA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'BAC', 'XOM'],
            'large_cap_weight': 0.8,
            
            # Market-moving sectors
            'high_impact_sectors': ['AI', 'artificial intelligence', 'electric vehicle', 'crypto', 
                                   'federal reserve', 'inflation', 'interest rates', 'GDP'],
            'sector_weight': 0.6
        }
        
        # Content intensity keywords (higher weight = more market impact)
        self.intensity_keywords = {
            # Extreme Impact (3.0x multiplier)
            'extreme': {
                'patterns': ['breakthrough', 'revolutionary', 'game-changer', 'historic', 'unprecedented'],
                'weight': 3.0
            },
            
            # High Impact (2.5x multiplier) 
            'high': {
                'patterns': ['beats expectations', 'smashes estimates', 'crushes forecasts', 
                           'misses badly', 'plunges', 'soars', 'surges', 'collapses'],
                'weight': 2.5
            },
            
            # Medium Impact (2.0x multiplier)
            'medium': {
                'patterns': ['beats estimates', 'exceeds expectations', 'disappoints', 
                           'drops sharply', 'rises strongly', 'major deal', 'acquisition'],
                'weight': 2.0
            },
            
            # Quantified Impact (numbers make news more credible)
            'numbers': {
                'patterns': [r'\d+%', r'\$\d+[MB]', r'\d+\.\d+%', 'billion', 'million', 'trillion'],
                'weight': 1.5
            }
        }
        
        # Timing factors
        self.timing_factors = {
            'market_hours': 1.3,      # News during trading hours
            'pre_market': 1.2,        # Pre-market news (7-9:30 AM ET)
            'after_hours': 1.1,       # After hours (4-8 PM ET)
            'weekend': 0.8,           # Weekend news (less immediate impact)
            'breaking_fresh': 1.5,    # Fresh breaking news (<1 hour)
            'recent': 1.2,            # Recent news (<4 hours)
            'stale': 0.6              # Old news (>24 hours)
        }
        
        # Enhanced Temporal Dynamics Parameters
        self.temporal_dynamics = {
            # Hype duration tracking (how long sentiment momentum lasts)
            'hype_duration_windows': {
                'flash': {'max_hours': 1, 'decay_factor': 0.9},      # Flash news - quick impact
                'short': {'max_hours': 4, 'decay_factor': 0.7},      # Short-term momentum
                'medium': {'max_hours': 24, 'decay_factor': 0.5},    # Daily cycle
                'long': {'max_hours': 72, 'decay_factor': 0.3}       # Multi-day impact
            },
            
            # Peak impact detection windows (when maximum price impact occurs)
            'peak_detection_windows': {
                'immediate': {'hours': 0.25, 'weight': 1.0},        # 15 minutes
                'short_term': {'hours': 1, 'weight': 0.9},          # 1 hour
                'medium_term': {'hours': 4, 'weight': 0.7},         # 4 hours  
                'daily': {'hours': 24, 'weight': 0.5},              # 1 day
                'extended': {'hours': 72, 'weight': 0.3}            # 3 days
            },
            
            # Decay pattern analysis (how fast hype influence fades)
            'decay_patterns': {
                'exponential': {'formula': 'exp(-t/tau)', 'tau': 2.0},    # Fast exponential decay
                'linear': {'formula': 'max(0, 1-t/max_t)', 'max_t': 12.0}, # Linear decay
                'step': {'formula': 'step_function', 'thresholds': [1, 4, 12, 24]} # Step decay
            }
        }
        
        # Enhanced Hype amplification indicators
        self.hype_indicators = {
            'viral_words': ['trending', 'viral', 'exploding', 'rocket', 'moon', 'massive surge'],
            'urgency_words': ['urgent', 'breaking', 'alert', 'flash', 'just in'],
            'social_buzz': ['twitter', 'reddit', 'social media', 'viral', 'trending'],
            'hype_weight': 1.3
        }
        
        # Strength Correlation Parameters
        self.strength_correlation = {
            # Sentiment magnitude vs price velocity mapping
            'magnitude_velocity': {
                'weak': {'sentiment_range': [0.1, 0.3], 'expected_velocity': 0.5},      # Slow moves
                'moderate': {'sentiment_range': [0.3, 0.6], 'expected_velocity': 1.0},  # Normal moves
                'strong': {'sentiment_range': [0.6, 0.8], 'expected_velocity': 2.0},    # Fast moves
                'extreme': {'sentiment_range': [0.8, 1.0], 'expected_velocity': 3.0}    # Very fast moves
            },
            
            # Volume correlation indicators (high hype → trading volume spikes)
            'volume_correlation': {
                'multipliers': {
                    'low_hype': 1.0,      # Normal volume
                    'medium_hype': 2.5,   # 2.5x volume spike
                    'high_hype': 5.0,     # 5x volume spike  
                    'viral_hype': 10.0    # 10x volume spike
                },
                'hype_thresholds': [0.3, 0.6, 0.8]  # Thresholds for hype levels
            },
            
            # Volatility pattern indicators (strong hype → higher price swings)
            'volatility_patterns': {
                'baseline_volatility': 0.02,  # 2% baseline daily volatility
                'hype_multipliers': {
                    'weak': 1.2,    # 20% increase in volatility
                    'moderate': 1.8,  # 80% increase
                    'strong': 2.5,    # 150% increase
                    'extreme': 4.0    # 300% increase
                },
                'momentum_persistence': {
                    'short': 2,   # 2 hours of elevated volatility
                    'medium': 8,  # 8 hours
                    'long': 24    # 24 hours
                }
            }
        }
        
        # Market Context Parameters
        self.market_context = {
            # Sector momentum differences (tech vs energy vs finance hype response)
            'sector_momentum': {
                'technology': {
                    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA'],
                    'hype_multiplier': 1.4,  # Tech responds strongly to hype
                    'keywords': ['AI', 'artificial intelligence', 'cloud', 'software', 'tech'],
                    'volatility_factor': 1.6
                },
                'energy': {
                    'symbols': ['XOM', 'CVX', 'COP', 'SLB', 'MPC'],
                    'hype_multiplier': 0.8,  # Energy less sensitive to hype
                    'keywords': ['oil', 'gas', 'energy', 'drilling', 'refining'],
                    'volatility_factor': 1.2
                },
                'finance': {
                    'symbols': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
                    'hype_multiplier': 1.1,  # Moderate hype sensitivity
                    'keywords': ['bank', 'finance', 'loan', 'credit', 'fed'],
                    'volatility_factor': 1.3
                },
                'healthcare': {
                    'symbols': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
                    'hype_multiplier': 1.2,
                    'keywords': ['drug', 'medicine', 'healthcare', 'FDA', 'clinical'],
                    'volatility_factor': 1.4
                }
            },
            
            # Market regime detection (bull vs bear market hype effectiveness)
            'market_regime': {
                'bull_market': {
                    'positive_hype_multiplier': 1.3,  # Positive news amplified in bull market
                    'negative_hype_multiplier': 0.7,  # Negative news dampened
                    'risk_appetite': 'high'
                },
                'bear_market': {
                    'positive_hype_multiplier': 0.8,  # Positive news dampened in bear market
                    'negative_hype_multiplier': 1.4,  # Negative news amplified
                    'risk_appetite': 'low'
                },
                'neutral_market': {
                    'positive_hype_multiplier': 1.0,  # Normal response
                    'negative_hype_multiplier': 1.0,
                    'risk_appetite': 'moderate'
                }
            },
            
            # Time-of-day impact (morning vs afternoon news effectiveness)
            'time_of_day_impact': {
                'market_open': {'hours': [9, 10], 'multiplier': 1.5},    # 9-10 AM high impact
                'morning_session': {'hours': [10, 12], 'multiplier': 1.3}, # 10-12 PM strong
                'lunch_time': {'hours': [12, 14], 'multiplier': 0.8},     # 12-2 PM weaker
                'afternoon_session': {'hours': [14, 16], 'multiplier': 1.1}, # 2-4 PM moderate
                'market_close': {'hours': [16, 17], 'multiplier': 1.4},   # 4-5 PM high impact
                'after_hours': {'hours': [17, 20], 'multiplier': 1.2}     # 5-8 PM moderate
            }
        }
        
    def start(self) -> None:
        self.logger.info("Starting News Impact Scorer")
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping News Impact Scorer") 
        self.is_running = False
        
    def process(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score news items for market impact potential"""
        if not self.is_running or not news_items:
            return news_items
            
        scored_items = []
        for item in news_items:
            try:
                impact_score = self._calculate_impact_score(item)
                item['impact_score'] = impact_score
                item['impact_grade'] = self._get_impact_grade(impact_score)
                scored_items.append(item)
            except Exception as e:
                self.logger.error(f"Error scoring news item: {e}")
                item['impact_score'] = 0.1  # Default low score
                item['impact_grade'] = 'D'
                scored_items.append(item)
        
        # Sort by impact score (highest first)
        scored_items.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
        
        # Log high-impact news
        high_impact = [item for item in scored_items if item.get('impact_score', 0) > 0.8]
        if high_impact:
            self.logger.warning(f"🔥 HIGH IMPACT NEWS: {len(high_impact)} items with score >0.8")
            for item in high_impact[:3]:
                self.logger.info(f"  📈 {item.get('impact_grade', 'N/A')}: {item.get('title', 'Unknown')[:60]}")
        
        return scored_items
        
    def _calculate_impact_score(self, item: Dict[str, Any]) -> float:
        """Calculate comprehensive impact score (0.0 to 1.0+)"""
        
        # 1. Source Authority (30% weight)
        source_score = self._calculate_source_score(item) * 0.30
        
        # 2. Market Exposure (25% weight)
        exposure_score = self._calculate_exposure_score(item) * 0.25
        
        # 3. Content Intensity (20% weight)
        intensity_score = self._calculate_intensity_score(item) * 0.20
        
        # 4. Timing & Freshness (15% weight)
        timing_score = self._calculate_timing_score(item) * 0.15
        
        # 5. Hype Amplification (10% weight)
        hype_score = self._calculate_hype_score(item) * 0.10
        
        # Combine all factors
        total_score = source_score + exposure_score + intensity_score + timing_score + hype_score
        
        # Cap at reasonable maximum (can exceed 1.0 for exceptional news)
        return min(total_score, 2.0)
    
    def _calculate_source_score(self, item: Dict[str, Any]) -> float:
        """Score based on source credibility and authority"""
        source = item.get('source', 'Unknown')
        
        # Exact match
        if source in self.source_authority:
            return self.source_authority[source]
            
        # Partial match (e.g., "Reuters Business" matches "Reuters")
        for auth_source, weight in self.source_authority.items():
            if auth_source.lower() in source.lower():
                return weight
                
        return self.source_authority['Unknown']
    
    def _calculate_exposure_score(self, item: Dict[str, Any]) -> float:
        """Score based on market exposure and company/sector importance"""
        content = f"{item.get('title', '')} {item.get('content', '')}".upper()
        max_exposure = 0.0
        
        # Check for mega-cap mentions
        for symbol in self.market_exposure['mega_cap']:
            if symbol in content or self._company_name_in_content(symbol, content):
                max_exposure = max(max_exposure, self.market_exposure['mega_cap_weight'])
        
        # Check for large-cap mentions  
        for symbol in self.market_exposure['large_cap']:
            if symbol in content or self._company_name_in_content(symbol, content):
                max_exposure = max(max_exposure, self.market_exposure['large_cap_weight'])
                
        # Check for high-impact sectors
        for sector in self.market_exposure['high_impact_sectors']:
            if sector.upper() in content:
                max_exposure = max(max_exposure, self.market_exposure['sector_weight'])
        
        return max_exposure
    
    def _company_name_in_content(self, symbol: str, content: str) -> bool:
        """Check if company name appears in content (not just ticker)"""
        company_names = {
            'AAPL': 'APPLE',
            'MSFT': 'MICROSOFT', 
            'GOOGL': 'GOOGLE',
            'AMZN': 'AMAZON',
            'TSLA': 'TESLA',
            'META': 'META',
            'NVDA': 'NVIDIA'
        }
        
        if symbol in company_names:
            return company_names[symbol] in content
        return False
        
    def _calculate_intensity_score(self, item: Dict[str, Any]) -> float:
        """Score based on content intensity and impact keywords"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        max_intensity = 0.3  # Base score
        
        for category, config in self.intensity_keywords.items():
            for pattern in config['patterns']:
                if category == 'numbers':
                    # Use regex for number patterns
                    if re.search(pattern, content):
                        max_intensity = max(max_intensity, config['weight'] * 0.3)
                else:
                    # Simple string matching for text patterns
                    if pattern in content:
                        max_intensity = max(max_intensity, config['weight'] * 0.3)
        
        return min(max_intensity, 1.0)
    
    def _calculate_timing_score(self, item: Dict[str, Any]) -> float:
        """Score based on timing and freshness"""
        published_obj = item.get('published') or item.get('timestamp', '')
        if not published_obj:
            return 0.5  # Neutral if no timestamp
            
        try:
            # Handle both datetime objects and string timestamps
            if isinstance(published_obj, datetime):
                published = published_obj
            elif isinstance(published_obj, str):
                # Parse timestamp string
                if 'T' in published_obj:
                    published = datetime.fromisoformat(published_obj.replace('Z', '+00:00'))
                else:
                    published = datetime.fromisoformat(published_obj)
            else:
                return 0.5  # Unknown format
                
            now = datetime.utcnow()
            age_hours = (now - published.replace(tzinfo=None)).total_seconds() / 3600
            
            # Freshness factor
            if age_hours < 1:
                freshness = self.timing_factors['breaking_fresh']
            elif age_hours < 4:
                freshness = self.timing_factors['recent']  
            elif age_hours > 24:
                freshness = self.timing_factors['stale']
            else:
                freshness = 1.0
                
            # Market hours factor (assuming ET timezone)
            hour = published.hour
            if 9 <= hour <= 16:  # Market hours (9:30 AM - 4 PM ET)
                timing = self.timing_factors['market_hours']
            elif 7 <= hour <= 9:  # Pre-market
                timing = self.timing_factors['pre_market']
            elif 16 <= hour <= 20:  # After hours
                timing = self.timing_factors['after_hours']
            else:
                timing = 1.0
                
            return min(freshness * timing, 2.0)
            
        except Exception as e:
            self.logger.warning(f"Error parsing timestamp {published_obj}: {e}")
            return 0.5
    
    def _calculate_hype_score(self, item: Dict[str, Any]) -> float:
        """Score based on hype and viral potential"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        hype_multiplier = 1.0
        
        # Check for viral indicators
        viral_count = sum(1 for word in self.hype_indicators['viral_words'] if word in content)
        urgency_count = sum(1 for word in self.hype_indicators['urgency_words'] if word in content)  
        social_count = sum(1 for word in self.hype_indicators['social_buzz'] if word in content)
        
        total_hype_signals = viral_count + urgency_count + social_count
        
        if total_hype_signals > 0:
            hype_multiplier = min(1.0 + (total_hype_signals * 0.2), self.hype_indicators['hype_weight'])
            
        return hype_multiplier
    
    def _get_impact_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 1.5:
            return 'A+'  # Exceptional market impact
        elif score >= 1.2:
            return 'A'   # High market impact
        elif score >= 1.0:
            return 'B+'  # Above average impact
        elif score >= 0.8:
            return 'B'   # Good market impact
        elif score >= 0.6:
            return 'C'   # Moderate impact
        elif score >= 0.4:
            return 'D'   # Low impact
        else:
            return 'F'   # Minimal impact
            
    def calculate_temporal_dynamics(self, item: Dict[str, Any], sentiment_strength: float = 0.0) -> Dict[str, Any]:
        """Calculate temporal dynamics including hype duration, peak detection, and decay patterns"""
        published_obj = item.get('published') or item.get('timestamp', '')
        current_time = datetime.utcnow()
        
        if not published_obj:
            return self._get_default_temporal_dynamics()
            
        try:
            # Parse timestamp
            if isinstance(published_obj, datetime):
                published = published_obj
            elif isinstance(published_obj, str):
                if 'T' in published_obj:
                    published = datetime.fromisoformat(published_obj.replace('Z', '+00:00'))
                else:
                    published = datetime.fromisoformat(published_obj)
            else:
                return self._get_default_temporal_dynamics()
            
            age_hours = (current_time - published.replace(tzinfo=None)).total_seconds() / 3600
            
            # Determine hype duration window
            hype_window = self._determine_hype_window(item, sentiment_strength)
            
            # Calculate peak detection timing
            peak_detection = self._calculate_peak_detection_window(age_hours, sentiment_strength)
            
            # Calculate decay pattern
            decay_info = self._calculate_decay_pattern(age_hours, sentiment_strength)
            
            return {
                'age_hours': age_hours,
                'hype_window': hype_window,
                'peak_detection': peak_detection,
                'decay_info': decay_info,
                'temporal_multiplier': self._get_temporal_multiplier(age_hours, hype_window, decay_info)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating temporal dynamics: {e}")
            return self._get_default_temporal_dynamics()
    
    def _determine_hype_window(self, item: Dict[str, Any], sentiment_strength: float) -> Dict[str, Any]:
        """Determine appropriate hype duration window based on content analysis"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        # Check for flash indicators (breaking news, urgent alerts)
        flash_indicators = ['breaking', 'urgent', 'flash', 'alert', 'just in', 'developing']
        if any(indicator in content for indicator in flash_indicators):
            return {'type': 'flash', **self.temporal_dynamics['hype_duration_windows']['flash']}
        
        # Check for high-impact keywords indicating longer momentum
        long_indicators = ['acquisition', 'merger', 'breakthrough', 'revolutionary', 'partnership']
        if any(indicator in content for indicator in long_indicators) or sentiment_strength > 0.7:
            return {'type': 'long', **self.temporal_dynamics['hype_duration_windows']['long']}
        
        # Medium-term for moderate sentiment
        if sentiment_strength > 0.4:
            return {'type': 'medium', **self.temporal_dynamics['hype_duration_windows']['medium']}
        
        # Default to short-term
        return {'type': 'short', **self.temporal_dynamics['hype_duration_windows']['short']}
    
    def _calculate_peak_detection_window(self, age_hours: float, sentiment_strength: float) -> Dict[str, Any]:
        """Calculate which peak detection window applies and expected impact timing"""
        for window_name, window_config in self.temporal_dynamics['peak_detection_windows'].items():
            if age_hours <= window_config['hours']:
                # Adjust weight based on sentiment strength
                adjusted_weight = window_config['weight'] * (1 + sentiment_strength * 0.3)
                return {
                    'window': window_name,
                    'base_weight': window_config['weight'],
                    'adjusted_weight': min(adjusted_weight, 1.5),  # Cap at 1.5x
                    'hours_threshold': window_config['hours']
                }
        
        # If past all windows, use extended with reduced weight
        return {
            'window': 'expired',
            'base_weight': 0.1,
            'adjusted_weight': 0.1,
            'hours_threshold': 72
        }
    
    def _calculate_decay_pattern(self, age_hours: float, sentiment_strength: float) -> Dict[str, Any]:
        """Calculate decay pattern and current decay multiplier"""
        # Choose decay pattern based on sentiment strength
        if sentiment_strength > 0.8:
            pattern = 'linear'  # Strong sentiment decays more slowly
        elif sentiment_strength > 0.4:
            pattern = 'exponential'  # Moderate sentiment has exponential decay
        else:
            pattern = 'step'  # Weak sentiment has step decay
        
        decay_config = self.temporal_dynamics['decay_patterns'][pattern]
        
        if pattern == 'exponential':
            tau = decay_config['tau']
            decay_multiplier = math.exp(-age_hours / tau)
        elif pattern == 'linear':
            max_t = decay_config['max_t']
            decay_multiplier = max(0, 1 - age_hours / max_t)
        else:  # step
            thresholds = decay_config['thresholds']
            if age_hours <= thresholds[0]:
                decay_multiplier = 1.0
            elif age_hours <= thresholds[1]:
                decay_multiplier = 0.8
            elif age_hours <= thresholds[2]:
                decay_multiplier = 0.5
            elif age_hours <= thresholds[3]:
                decay_multiplier = 0.2
            else:
                decay_multiplier = 0.1
        
        return {
            'pattern': pattern,
            'decay_multiplier': max(decay_multiplier, 0.05),  # Minimum 5% retention
            'config': decay_config
        }
    
    def _get_temporal_multiplier(self, age_hours: float, hype_window: Dict, decay_info: Dict) -> float:
        """Calculate overall temporal multiplier combining all factors"""
        # Base multiplier from hype window
        base_multiplier = hype_window.get('decay_factor', 0.5)
        
        # Apply decay
        decay_multiplier = decay_info['decay_multiplier']
        
        # Combine with age-based adjustment
        if age_hours < 1:
            age_boost = 1.2  # Recent news boost
        elif age_hours < 4:
            age_boost = 1.0  # Normal
        else:
            age_boost = 0.8  # Older news penalty
        
        return base_multiplier * decay_multiplier * age_boost
    
    def _get_default_temporal_dynamics(self) -> Dict[str, Any]:
        """Return default temporal dynamics when calculation fails"""
        return {
            'age_hours': 24,  # Assume 24 hours old
            'hype_window': {'type': 'medium', **self.temporal_dynamics['hype_duration_windows']['medium']},
            'peak_detection': {'window': 'extended', 'base_weight': 0.3, 'adjusted_weight': 0.3},
            'decay_info': {'pattern': 'exponential', 'decay_multiplier': 0.5},
            'temporal_multiplier': 0.5
        }

    def calculate_strength_correlation(self, sentiment_score: float, hype_score: float, market_symbols: List[str] = None) -> Dict[str, Any]:
        """Calculate strength correlation metrics between sentiment and expected market impact"""
        
        # 1. Sentiment magnitude to velocity mapping
        velocity_info = self._get_sentiment_velocity_mapping(sentiment_score)
        
        # 2. Volume correlation based on hype
        volume_info = self._calculate_volume_correlation(hype_score, sentiment_score)
        
        # 3. Volatility pattern prediction
        volatility_info = self._calculate_volatility_patterns(sentiment_score, hype_score)
        
        # 4. Symbol-specific adjustments
        symbol_adjustments = self._calculate_symbol_adjustments(market_symbols, sentiment_score) if market_symbols else {}
        
        return {
            'velocity_mapping': velocity_info,
            'volume_correlation': volume_info, 
            'volatility_patterns': volatility_info,
            'symbol_adjustments': symbol_adjustments,
            'overall_strength_score': self._calculate_overall_strength_score(velocity_info, volume_info, volatility_info)
        }
    
    def _get_sentiment_velocity_mapping(self, sentiment_score: float) -> Dict[str, Any]:
        """Map sentiment magnitude to expected price velocity"""
        abs_sentiment = abs(sentiment_score)
        
        for strength_level, config in self.strength_correlation['magnitude_velocity'].items():
            min_range, max_range = config['sentiment_range']
            if min_range <= abs_sentiment <= max_range:
                return {
                    'strength_level': strength_level,
                    'expected_velocity': config['expected_velocity'],
                    'sentiment_magnitude': abs_sentiment,
                    'direction': 'positive' if sentiment_score > 0 else 'negative'
                }
        
        # Default to weak if outside ranges
        return {
            'strength_level': 'weak',
            'expected_velocity': 0.5,
            'sentiment_magnitude': abs_sentiment,
            'direction': 'positive' if sentiment_score > 0 else 'negative'
        }
    
    def _calculate_volume_correlation(self, hype_score: float, sentiment_score: float) -> Dict[str, Any]:
        """Calculate expected volume correlation based on hype levels"""
        combined_hype = (hype_score + abs(sentiment_score)) / 2
        thresholds = self.strength_correlation['volume_correlation']['hype_thresholds']
        multipliers = self.strength_correlation['volume_correlation']['multipliers']
        
        if combined_hype >= thresholds[2]:  # >= 0.8
            hype_level = 'viral_hype'
        elif combined_hype >= thresholds[1]:  # >= 0.6
            hype_level = 'high_hype'
        elif combined_hype >= thresholds[0]:  # >= 0.3
            hype_level = 'medium_hype'
        else:
            hype_level = 'low_hype'
        
        return {
            'hype_level': hype_level,
            'volume_multiplier': multipliers[hype_level],
            'combined_hype_score': combined_hype
        }
    
    def _calculate_volatility_patterns(self, sentiment_score: float, hype_score: float) -> Dict[str, Any]:
        """Calculate expected volatility patterns"""
        abs_sentiment = abs(sentiment_score)
        baseline_vol = self.strength_correlation['volatility_patterns']['baseline_volatility']
        
        # Determine volatility level
        if abs_sentiment >= 0.8 or hype_score >= 0.8:
            vol_level = 'extreme'
        elif abs_sentiment >= 0.6 or hype_score >= 0.6:
            vol_level = 'strong'
        elif abs_sentiment >= 0.3 or hype_score >= 0.4:
            vol_level = 'moderate'
        else:
            vol_level = 'weak'
        
        multiplier = self.strength_correlation['volatility_patterns']['hype_multipliers'][vol_level]
        expected_volatility = baseline_vol * multiplier
        
        # Determine momentum persistence
        if vol_level in ['extreme', 'strong']:
            persistence = 'long'
        elif vol_level == 'moderate':
            persistence = 'medium'
        else:
            persistence = 'short'
        
        persistence_hours = self.strength_correlation['volatility_patterns']['momentum_persistence'][persistence]
        
        return {
            'volatility_level': vol_level,
            'expected_volatility': expected_volatility,
            'volatility_multiplier': multiplier,
            'persistence': persistence,
            'persistence_hours': persistence_hours
        }
    
    def _calculate_symbol_adjustments(self, symbols: List[str], sentiment_score: float) -> Dict[str, Any]:
        """Calculate symbol-specific adjustments based on sector momentum"""
        if not symbols:
            return {}
        
        adjustments = {}
        for symbol in symbols:
            sector_info = self._get_symbol_sector(symbol)
            if sector_info:
                sector_config = self.market_context['sector_momentum'][sector_info['sector']]
                
                # Calculate adjusted impact
                base_multiplier = sector_config['hype_multiplier']
                volatility_factor = sector_config['volatility_factor']
                
                # Stronger sentiment gets more sector-specific amplification
                adjusted_multiplier = base_multiplier * (1 + abs(sentiment_score) * 0.3)
                
                adjustments[symbol] = {
                    'sector': sector_info['sector'],
                    'base_multiplier': base_multiplier,
                    'adjusted_multiplier': adjusted_multiplier,
                    'volatility_factor': volatility_factor
                }
        
        return adjustments
    
    def _get_symbol_sector(self, symbol: str) -> Optional[Dict[str, str]]:
        """Get sector information for a symbol"""
        for sector, config in self.market_context['sector_momentum'].items():
            if symbol in config['symbols']:
                return {'sector': sector}
        return None
    
    def _calculate_overall_strength_score(self, velocity_info: Dict, volume_info: Dict, volatility_info: Dict) -> float:
        """Calculate overall strength correlation score"""
        velocity_score = velocity_info['expected_velocity'] / 3.0  # Normalize to 0-1
        volume_score = min(volume_info['volume_multiplier'] / 10.0, 1.0)  # Normalize to 0-1
        volatility_score = min(volatility_info['volatility_multiplier'] / 4.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        overall_score = (velocity_score * 0.4 + volume_score * 0.3 + volatility_score * 0.3)
        
        return min(overall_score, 1.0)

    def calculate_market_context(self, item: Dict[str, Any], symbols: List[str] = None) -> Dict[str, Any]:
        """Calculate market context including sector momentum, market regime, and time-of-day impact"""
        
        # 1. Sector momentum analysis
        sector_analysis = self._analyze_sector_momentum(item, symbols)
        
        # 2. Market regime detection (simplified - would need more data for full implementation)
        market_regime = self._detect_market_regime(item)
        
        # 3. Time-of-day impact
        time_impact = self._calculate_time_of_day_impact(item)
        
        # 4. Combined market context score
        context_score = self._calculate_context_score(sector_analysis, market_regime, time_impact)
        
        return {
            'sector_analysis': sector_analysis,
            'market_regime': market_regime,
            'time_impact': time_impact,
            'context_score': context_score,
            'context_multiplier': self._get_context_multiplier(context_score)
        }
    
    def _analyze_sector_momentum(self, item: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Analyze sector-specific momentum and impact"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        detected_sectors = []
        sector_scores = {}
        
        # Check content for sector keywords
        for sector, config in self.market_context['sector_momentum'].items():
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in content)
            symbol_matches = sum(1 for symbol in config['symbols'] if symbols and symbol in symbols)
            
            if keyword_matches > 0 or symbol_matches > 0:
                detected_sectors.append(sector)
                sector_scores[sector] = {
                    'keyword_matches': keyword_matches,
                    'symbol_matches': symbol_matches,
                    'hype_multiplier': config['hype_multiplier'],
                    'volatility_factor': config['volatility_factor']
                }
        
        return {
            'detected_sectors': detected_sectors,
            'sector_scores': sector_scores,
            'dominant_sector': max(sector_scores.keys(), key=lambda s: sector_scores[s]['keyword_matches'] + sector_scores[s]['symbol_matches']) if sector_scores else None
        }
    
    def _detect_market_regime(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market regime (simplified implementation)"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        # Bull market indicators
        bull_indicators = ['rally', 'surge', 'bull market', 'all-time high', 'record high', 'strong growth']
        bull_score = sum(1 for indicator in bull_indicators if indicator in content)
        
        # Bear market indicators  
        bear_indicators = ['crash', 'plunge', 'bear market', 'recession', 'downturn', 'selloff']
        bear_score = sum(1 for indicator in bear_indicators if indicator in content)
        
        # Determine regime (simplified)
        if bull_score > bear_score and bull_score > 0:
            regime = 'bull_market'
        elif bear_score > bull_score and bear_score > 0:
            regime = 'bear_market'
        else:
            regime = 'neutral_market'
        
        regime_config = self.market_context['market_regime'][regime]
        
        return {
            'regime': regime,
            'bull_indicators': bull_score,
            'bear_indicators': bear_score,
            'positive_hype_multiplier': regime_config['positive_hype_multiplier'],
            'negative_hype_multiplier': regime_config['negative_hype_multiplier'],
            'risk_appetite': regime_config['risk_appetite']
        }
    
    def _calculate_time_of_day_impact(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate time-of-day impact multiplier"""
        published_obj = item.get('published') or item.get('timestamp', '')
        
        if not published_obj:
            return {'time_period': 'unknown', 'multiplier': 1.0}
        
        try:
            if isinstance(published_obj, datetime):
                hour = published_obj.hour
            elif isinstance(published_obj, str):
                if 'T' in published_obj:
                    dt = datetime.fromisoformat(published_obj.replace('Z', '+00:00'))
                    hour = dt.hour
                else:
                    dt = datetime.fromisoformat(published_obj)
                    hour = dt.hour
            else:
                return {'time_period': 'unknown', 'multiplier': 1.0}
            
            # Find matching time period
            for period, config in self.market_context['time_of_day_impact'].items():
                if config['hours'][0] <= hour < config['hours'][1]:
                    return {
                        'time_period': period,
                        'hour': hour,
                        'multiplier': config['multiplier']
                    }
            
            # Default to neutral if no match
            return {'time_period': 'neutral', 'hour': hour, 'multiplier': 1.0}
            
        except Exception as e:
            self.logger.warning(f"Error calculating time impact: {e}")
            return {'time_period': 'unknown', 'multiplier': 1.0}
    
    def _calculate_context_score(self, sector_analysis: Dict, market_regime: Dict, time_impact: Dict) -> float:
        """Calculate combined market context score"""
        # Sector score (0-1)
        sector_score = 0.0
        if sector_analysis['dominant_sector']:
            dominant_config = self.market_context['sector_momentum'][sector_analysis['dominant_sector']]
            sector_score = min(dominant_config['hype_multiplier'], 1.0)
        
        # Market regime score (0-1)
        regime_score = (market_regime['positive_hype_multiplier'] + market_regime['negative_hype_multiplier']) / 2
        regime_score = min(regime_score, 1.0)
        
        # Time impact score (0-1)
        time_score = min(time_impact['multiplier'], 1.0)
        
        # Weighted combination
        context_score = (sector_score * 0.4 + regime_score * 0.3 + time_score * 0.3)
        
        return context_score
    
    def _get_context_multiplier(self, context_score: float) -> float:
        """Convert context score to multiplier for impact scoring"""
        # Scale context score to useful multiplier range (0.7 to 1.5)
        return 0.7 + (context_score * 0.8)

    def get_impact_summary(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of news impact distribution"""
        if not news_items:
            return {}
            
        scores = [item.get('impact_score', 0) for item in news_items]
        grades = [item.get('impact_grade', 'F') for item in news_items]
        
        return {
            'total_items': len(news_items),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'high_impact_count': len([s for s in scores if s >= 0.8]),
            'grade_distribution': {grade: grades.count(grade) for grade in set(grades)},
            'top_stories': [
                {
                    'title': item.get('title', 'Unknown')[:60],
                    'score': item.get('impact_score', 0),
                    'grade': item.get('impact_grade', 'F')
                }
                for item in sorted(news_items, key=lambda x: x.get('impact_score', 0), reverse=True)[:5]
            ]
        }
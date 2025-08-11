from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from ..core.base import ComponentBase


class TradingPair:
    def __init__(self, symbol: str, action: str, entry_price: float, 
                 exit_price: Optional[float] = None, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None, quantity: int = 0,
                 entry_time: Optional[datetime] = None, exit_time: Optional[datetime] = None):
        self.symbol = symbol
        self.action = action  # "buy" or "sell" (for short)
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.quantity = quantity
        self.entry_time = entry_time or datetime.utcnow()
        self.exit_time = exit_time
        self.confidence = 0.0
        self.reasoning = ""
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class DecisionEngine(ComponentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("decision_engine", config)
        self.min_confidence = config.get("min_confidence", 0.6)
        self.max_position_size = config.get("max_position_size", 1000)
        self.sentiment_weight = config.get("sentiment_weight", 0.4)
        self.impact_weight = config.get("impact_weight", 0.3)
        self.recency_weight = config.get("recency_weight", 0.3)
        self.default_stop_loss_pct = config.get("default_stop_loss_pct", 0.05)
        self.default_take_profit_pct = config.get("default_take_profit_pct", 0.10)
        self.alpaca_client = None  # Will be injected by main app
        self.news_impact_scorer = None  # Will be injected by main app for enhanced analysis
        
        # Enhanced parameters for temporal and context analysis
        self.temporal_weight = config.get("temporal_weight", 0.2)  # Weight for temporal dynamics
        self.strength_correlation_weight = config.get("strength_correlation_weight", 0.15)  # Weight for strength correlation  
        self.market_context_weight = config.get("market_context_weight", 0.15)  # Weight for market context
        self.enable_enhanced_analysis = config.get("enable_enhanced_analysis", True)  # Toggle for enhanced features
        
    def start(self) -> None:
        self.logger.info("Starting Decision Engine")
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping Decision Engine")
        self.is_running = False
        
    async def process(self, analyzed_news: List[Dict[str, Any]], 
                      market_data: Optional[Dict[str, Any]] = None) -> List[TradingPair]:
        if not self.is_running or not analyzed_news:
            return []
            
        # Check if market is open before generating any trading decisions
        if self.alpaca_client and not await self.alpaca_client.is_market_open():
            self.logger.info("Market is closed - skipping trading decision generation (rest mode)")
            return []
            
        trading_pairs = []
        
        # Group news by symbols
        symbol_news = self._group_by_symbols(analyzed_news)
        
        for symbol, news_items in symbol_news.items():
            try:
                decision = await self._make_decision(symbol, news_items, market_data)
                if decision:
                    trading_pairs.append(decision)
            except Exception as e:
                self.logger.error(f"Error making decision for {symbol}: {e}")
                
        self.logger.info(f"Generated {len(trading_pairs)} trading decisions")
        return trading_pairs
        
    def _group_by_symbols(self, analyzed_news: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        symbol_groups = {}
        
        for item in analyzed_news:
            entities = item.get("entities", {})
            tickers = entities.get("tickers", [])
            
            if not tickers:
                # Try to extract from title or content
                tickers = self._extract_tickers_fallback(item)
                
            for ticker in tickers:
                if ticker not in symbol_groups:
                    symbol_groups[ticker] = []
                symbol_groups[ticker].append(item)
                
        return symbol_groups
        
    def _extract_tickers_fallback(self, item: Dict[str, Any]) -> List[str]:
        # Disabled fallback extraction to avoid false positives
        # The news analysis brain should handle all ticker extraction
        return []
        
    async def _make_decision(self, symbol: str, news_items: List[Dict[str, Any]], 
                            market_data: Optional[Dict[str, Any]] = None) -> Optional[TradingPair]:
        
        # Calculate aggregate signals
        signal_strength = self._calculate_signal_strength(news_items)
        confidence = self._calculate_confidence(news_items, signal_strength)
        
        if confidence < self.min_confidence:
            return None
            
        # Determine action (buy/sell)
        action = "buy" if signal_strength > 0 else "sell"
        
        # Get current market price
        current_price = await self._get_current_price(symbol, market_data)
        if not current_price:
            return None
            
        # Check for momentum and adjust targets dynamically
        adjusted_targets = self._adjust_targets_for_momentum(current_price, signal_strength, news_items)
        take_profit_pct = adjusted_targets.get("take_profit_pct", self.default_take_profit_pct)
        stop_loss_pct = adjusted_targets.get("stop_loss_pct", self.default_stop_loss_pct)
            
        # Calculate position size
        quantity = self._calculate_position_size(confidence, current_price)
        
        # Set stop loss and take profit with dynamic adjustment
        if action == "buy":
            # Buy: stop loss below, take profit above
            stop_loss = round(current_price * (1 - stop_loss_pct), 2)
            take_profit = round(current_price * (1 + take_profit_pct), 2)
        else:  # sell (short)
            # Sell: stop loss above, take profit below
            stop_loss = round(current_price * (1 + stop_loss_pct), 2)
            take_profit = round(current_price * (1 - take_profit_pct), 2)
            
        # Ensure minimum price difference for Alpaca (0.01)
        if action == "buy":
            if stop_loss >= current_price - 0.01:
                stop_loss = round(current_price - 0.01, 2)
            if take_profit <= current_price + 0.01:
                take_profit = round(current_price + 0.01, 2)
        else:  # sell
            if stop_loss <= current_price + 0.01:
                stop_loss = round(current_price + 0.01, 2)
            if take_profit >= current_price - 0.01:
                take_profit = round(current_price - 0.01, 2)
            
        # Create trading pair
        pair = TradingPair(
            symbol=symbol,
            action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity
        )
        
        pair.confidence = confidence
        pair.reasoning = self._generate_reasoning(news_items, signal_strength)
        
        # MANDATORY BRACKET ORDER VALIDATION
        validation_result = self._validate_bracket_order(pair)
        if not validation_result["valid"]:
            self.logger.warning(f"Bracket order validation failed for {symbol}: {validation_result['errors']}")
            return None
        
        return pair
        
    def _calculate_signal_strength(self, news_items: List[Dict[str, Any]]) -> float:
        total_signal = 0.0
        total_weight = 0.0
        
        # Extract symbols for enhanced analysis
        symbols = self._extract_symbols_from_news(news_items)
        
        for item in news_items:
            sentiment = item.get("sentiment", {})
            impact_score = item.get("impact_score", 0.0)
            filter_score = item.get("filter_score", 0.0)
            
            # Calculate weighted signal
            sentiment_score = sentiment.get("polarity", 0.0)
            
            # Base weight calculation (traditional approach)
            base_weight = (
                impact_score * self.impact_weight +
                filter_score * self.recency_weight +
                sentiment.get("confidence", 0.0) * self.sentiment_weight
            )
            
            # Enhanced analysis if enabled and scorer available
            enhanced_weight = 0.0
            if self.enable_enhanced_analysis and self.news_impact_scorer:
                enhanced_weight = self._calculate_enhanced_weight(item, sentiment_score, symbols)
            
            # Combine traditional and enhanced weights
            total_item_weight = base_weight + enhanced_weight
            
            total_signal += sentiment_score * total_item_weight
            total_weight += total_item_weight
            
        return total_signal / max(total_weight, 0.001)
    
    def _extract_symbols_from_news(self, news_items: List[Dict[str, Any]]) -> List[str]:
        """Extract all symbols mentioned in the news items"""
        symbols = []
        for item in news_items:
            entities = item.get("entities", {})
            tickers = entities.get("tickers", [])
            symbols.extend(tickers)
        return list(set(symbols))  # Remove duplicates
    
    def _calculate_enhanced_weight(self, item: Dict[str, Any], sentiment_score: float, symbols: List[str]) -> float:
        """Calculate enhanced weight using temporal dynamics, strength correlation, and market context"""
        try:
            enhanced_weight = 0.0
            
            # 1. Temporal dynamics analysis
            temporal_dynamics = self.news_impact_scorer.calculate_temporal_dynamics(
                item, abs(sentiment_score)
            )
            temporal_multiplier = temporal_dynamics.get('temporal_multiplier', 1.0)
            enhanced_weight += temporal_multiplier * self.temporal_weight
            
            # 2. Strength correlation analysis
            hype_score = item.get('hype_score', 0.0) or self._estimate_hype_score(item)
            strength_correlation = self.news_impact_scorer.calculate_strength_correlation(
                sentiment_score, hype_score, symbols
            )
            strength_score = strength_correlation.get('overall_strength_score', 0.5)
            enhanced_weight += strength_score * self.strength_correlation_weight
            
            # 3. Market context analysis
            market_context = self.news_impact_scorer.calculate_market_context(item, symbols)
            context_multiplier = market_context.get('context_multiplier', 1.0)
            enhanced_weight += context_multiplier * self.market_context_weight
            
            return enhanced_weight
            
        except Exception as e:
            self.logger.warning(f"Error in enhanced weight calculation: {e}")
            return 0.0  # Fallback to no enhanced weight
    
    def _estimate_hype_score(self, item: Dict[str, Any]) -> float:
        """Estimate hype score from content if not already calculated"""
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        # Look for hype indicators
        hype_words = ['breaking', 'urgent', 'massive', 'huge', 'explosive', 'surge', 'plunge', 
                     'record', 'unprecedented', 'shocking', 'dramatic']
        
        hype_count = sum(1 for word in hype_words if word in content)
        return min(hype_count * 0.2, 1.0)  # Cap at 1.0
        
    def _calculate_confidence(self, news_items: List[Dict[str, Any]], 
                            signal_strength: float) -> float:
        if not news_items:
            return 0.0
            
        # Base confidence on signal strength
        confidence = abs(signal_strength)
        
        # Boost confidence with multiple confirming sources
        unique_sources = len(set(item.get("source", "") for item in news_items))
        confidence *= min(1.0 + (unique_sources - 1) * 0.1, 1.5)
        
        # Consider impact scores - give minimum boost of 0.8 instead of 0.5
        avg_impact = sum(item.get("impact_score", 0.0) for item in news_items) / len(news_items)
        confidence *= (0.8 + avg_impact)
        
        # Enhanced confidence boost from temporal dynamics and market context
        if self.enable_enhanced_analysis and self.news_impact_scorer:
            confidence = self._apply_enhanced_confidence_boost(confidence, news_items)
        
        return min(confidence, 1.0)
    
    def _apply_enhanced_confidence_boost(self, base_confidence: float, news_items: List[Dict[str, Any]]) -> float:
        """Apply enhanced confidence boost based on temporal and market context analysis"""
        try:
            total_temporal_boost = 0.0
            total_context_boost = 0.0
            symbols = self._extract_symbols_from_news(news_items)
            
            for item in news_items:
                sentiment_score = item.get("sentiment", {}).get("polarity", 0.0)
                
                # 1. Temporal dynamics boost
                temporal_dynamics = self.news_impact_scorer.calculate_temporal_dynamics(
                    item, abs(sentiment_score)
                )
                
                # Fresh breaking news gets significant confidence boost
                if temporal_dynamics.get('hype_window', {}).get('type') == 'flash':
                    total_temporal_boost += 0.15  # 15% boost for flash news
                elif temporal_dynamics.get('age_hours', 24) < 1:
                    total_temporal_boost += 0.10  # 10% boost for very recent news
                
                # 2. Market context boost
                market_context = self.news_impact_scorer.calculate_market_context(item, symbols)
                
                # Strong sector momentum gets confidence boost
                sector_analysis = market_context.get('sector_analysis', {})
                if sector_analysis.get('dominant_sector'):
                    total_context_boost += 0.08  # 8% boost for sector-specific news
                
                # Market regime alignment boost
                market_regime = market_context.get('market_regime', {})
                if market_regime.get('regime') in ['bull_market', 'bear_market']:
                    # Regime-aligned sentiment gets boost
                    if ((sentiment_score > 0 and market_regime.get('regime') == 'bull_market') or
                        (sentiment_score < 0 and market_regime.get('regime') == 'bear_market')):
                        total_context_boost += 0.05  # 5% boost for regime alignment
            
            # Apply boosts (average across all news items)
            avg_temporal_boost = total_temporal_boost / len(news_items)
            avg_context_boost = total_context_boost / len(news_items)
            
            # Apply boosts multiplicatively but cap the total enhancement
            enhanced_confidence = base_confidence * (1 + avg_temporal_boost + avg_context_boost)
            return min(enhanced_confidence, base_confidence * 1.3)  # Max 30% boost
            
        except Exception as e:
            self.logger.warning(f"Error in enhanced confidence calculation: {e}")
            return base_confidence
        
    async def _get_current_price(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> Optional[float]:
        # Try to get real-time price from Alpaca if available
        if self.alpaca_client:
            try:
                price = await self.alpaca_client.get_current_price(symbol)
                if price:
                    return price
            except Exception as e:
                self.logger.warning(f"Failed to get real-time price for {symbol}: {e}")
        
        # Check if market data provided
        if market_data and symbol in market_data:
            return market_data[symbol].get("price")
        
        # Fallback to mock prices (for testing when markets closed)
        mock_prices = {
            'SPY': 533.94, 'QQQ': 418.0, 'AAPL': 202.44, 'TSLA': 304.50, 'AMZN': 213.75,
            'MSFT': 430.0, 'GOOGL': 163.0, 'META': 565.0, 'NVDA': 135.0, 'GM': 52.42,
            'F': 11.0, 'GE': 269.45, 'JPM': 234.0, 'BAC': 44.0, 'XOM': 116.0, 'BA': 227.0
        }
        
        return mock_prices.get(symbol, 100.0)  # Default to $100 if unknown
        
    def _calculate_position_size(self, confidence: float, price: float) -> int:
        # Simple position sizing based on confidence
        max_value = self.max_position_size
        position_value = max_value * confidence
        return max(1, int(position_value / price))
        
    def _generate_reasoning(self, news_items: List[Dict[str, Any]], 
                          signal_strength: float) -> str:
        num_items = len(news_items)
        avg_sentiment = sum(item.get("sentiment", {}).get("polarity", 0.0) for item in news_items) / num_items
        
        direction = "bullish" if signal_strength > 0 else "bearish"
        
        events = []
        for item in news_items:
            events.extend(item.get("events", []))
        unique_events = list(set(events))
        
        reasoning = f"{direction.capitalize()} signal from {num_items} news items. "
        reasoning += f"Average sentiment: {avg_sentiment:.2f}. "
        
        if unique_events:
            reasoning += f"Key events: {', '.join(unique_events[:3])}."
            
        return reasoning
        
    def _adjust_targets_for_momentum(self, current_price: float, signal_strength: float, 
                                   news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Dynamically adjust take-profit and stop-loss based on momentum, breaking news, and enhanced analysis"""
        
        # Default values
        take_profit_pct = self.default_take_profit_pct
        stop_loss_pct = self.default_stop_loss_pct
        
        # Traditional checks
        has_breaking_news = any(item.get("priority") == "breaking" for item in news_items)
        strong_signal = abs(signal_strength) > 0.7
        high_confidence = len(news_items) > 3
        
        # Enhanced analysis adjustments
        enhanced_adjustments = self._get_enhanced_target_adjustments(news_items, signal_strength)
        
        if has_breaking_news:
            # Breaking news: Increase targets to capture momentum
            if signal_strength > 0:  # Bullish
                take_profit_pct *= 1.5  # 10% -> 15% take profit
                stop_loss_pct *= 0.8    # 5% -> 4% stop loss (tighter)
                self.logger.info(f"📈 MOMENTUM BOOST: Breaking news detected - "
                               f"targeting {take_profit_pct*100:.1f}% profit")
            else:  # Bearish
                take_profit_pct *= 1.3  # More conservative on short side
                stop_loss_pct *= 0.9
                self.logger.info(f"📉 MOMENTUM BOOST: Breaking bearish news - "
                               f"targeting {take_profit_pct*100:.1f}% profit")
                
        elif strong_signal and high_confidence:
            # Strong signal with multiple sources: Moderate boost
            take_profit_pct *= 1.2  # 10% -> 12% take profit
            self.logger.info(f"💪 STRONG SIGNAL: High confidence trade - "
                           f"targeting {take_profit_pct*100:.1f}% profit")
        
        # Apply enhanced adjustments
        if enhanced_adjustments:
            take_profit_multiplier = enhanced_adjustments.get('take_profit_multiplier', 1.0)
            stop_loss_multiplier = enhanced_adjustments.get('stop_loss_multiplier', 1.0)
            volatility_adjustment = enhanced_adjustments.get('volatility_adjustment', 1.0)
            
            take_profit_pct *= take_profit_multiplier
            stop_loss_pct *= stop_loss_multiplier * volatility_adjustment
            
            if enhanced_adjustments.get('reasoning'):
                self.logger.info(f"🎯 ENHANCED TARGETING: {enhanced_adjustments['reasoning']}")
            
        # Safety limits - don't get too greedy
        take_profit_pct = min(take_profit_pct, 0.25)  # Max 25% target
        stop_loss_pct = max(stop_loss_pct, 0.02)      # Min 2% stop loss
        
        return {
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct
        }
    
    def _get_enhanced_target_adjustments(self, news_items: List[Dict[str, Any]], signal_strength: float) -> Optional[Dict[str, Any]]:
        """Get enhanced target adjustments based on temporal dynamics and market context"""
        if not (self.enable_enhanced_analysis and self.news_impact_scorer and news_items):
            return None
            
        try:
            symbols = self._extract_symbols_from_news(news_items)
            total_volatility_factor = 0.0
            total_temporal_factor = 0.0
            reasoning_parts = []
            
            for item in news_items:
                sentiment_score = item.get("sentiment", {}).get("polarity", 0.0)
                hype_score = item.get('hype_score', 0.0) or self._estimate_hype_score(item)
                
                # 1. Strength correlation analysis for volatility expectations
                strength_correlation = self.news_impact_scorer.calculate_strength_correlation(
                    sentiment_score, hype_score, symbols
                )
                
                volatility_info = strength_correlation.get('volatility_patterns', {})
                volatility_level = volatility_info.get('volatility_level', 'weak')
                volatility_multiplier = volatility_info.get('volatility_multiplier', 1.0)
                total_volatility_factor += volatility_multiplier
                
                # 2. Temporal dynamics for momentum persistence
                temporal_dynamics = self.news_impact_scorer.calculate_temporal_dynamics(
                    item, abs(sentiment_score)
                )
                
                hype_window = temporal_dynamics.get('hype_window', {})
                hype_type = hype_window.get('type', 'short')
                temporal_multiplier = temporal_dynamics.get('temporal_multiplier', 1.0)
                total_temporal_factor += temporal_multiplier
                
                # Build reasoning
                if volatility_level in ['strong', 'extreme']:
                    reasoning_parts.append(f"{volatility_level} volatility expected")
                if hype_type in ['flash', 'long']:
                    reasoning_parts.append(f"{hype_type}-term momentum")
            
            # Calculate average factors
            avg_volatility_factor = total_volatility_factor / len(news_items)
            avg_temporal_factor = total_temporal_factor / len(news_items)
            
            # Convert to target adjustments
            take_profit_multiplier = 1.0
            stop_loss_multiplier = 1.0
            
            # High volatility = wider targets to capture moves
            if avg_volatility_factor > 2.0:  # Strong/extreme volatility
                take_profit_multiplier = 1.3  # 30% wider profit targets
                stop_loss_multiplier = 1.2    # 20% wider stop loss
                reasoning_parts.append("wide targets for volatility")
            elif avg_volatility_factor > 1.5:  # Moderate volatility  
                take_profit_multiplier = 1.15  # 15% wider profit targets
                stop_loss_multiplier = 1.1     # 10% wider stop loss
                
            # Strong temporal momentum = tighter stops, wider profits
            if avg_temporal_factor > 1.0:
                take_profit_multiplier *= 1.1  # Extra profit capture
                stop_loss_multiplier *= 0.9    # Tighter stops for momentum
                reasoning_parts.append("momentum-optimized")
            
            return {
                'take_profit_multiplier': min(take_profit_multiplier, 1.5),  # Max 50% boost
                'stop_loss_multiplier': max(stop_loss_multiplier, 0.8),      # Max 20% tighter
                'volatility_adjustment': min(avg_volatility_factor / 2.0, 1.3),  # Volatility-based adjustment
                'reasoning': ', '.join(reasoning_parts) if reasoning_parts else None
            }
            
        except Exception as e:
            self.logger.warning(f"Error in enhanced target adjustments: {e}")
            return None
    
    def _validate_bracket_order(self, pair: TradingPair) -> Dict[str, Any]:
        """
        Mandatory bracket order validation to ensure all trades have proper risk management.
        This is a critical safety mechanism that prevents any trade without stop-loss/take-profit.
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Critical validation: Stop-loss must be present
        if pair.stop_loss is None:
            validation_result["valid"] = False
            validation_result["errors"].append("CRITICAL: Stop-loss is required for all trades")
        
        # Critical validation: Take-profit must be present
        if pair.take_profit is None:
            validation_result["valid"] = False
            validation_result["errors"].append("CRITICAL: Take-profit is required for all trades")
            
        # Validate stop-loss placement (directionally correct)
        if pair.stop_loss is not None:
            if pair.action == "buy" and pair.stop_loss >= pair.entry_price:
                validation_result["valid"] = False
                validation_result["errors"].append(f"CRITICAL: Buy order stop-loss must be below entry price ({pair.stop_loss:.2f} >= {pair.entry_price:.2f})")
            elif pair.action == "sell" and pair.stop_loss <= pair.entry_price:
                validation_result["valid"] = False
                validation_result["errors"].append(f"CRITICAL: Sell order stop-loss must be above entry price ({pair.stop_loss:.2f} <= {pair.entry_price:.2f})")
                
        # Validate take-profit placement (directionally correct)  
        if pair.take_profit is not None:
            if pair.action == "buy" and pair.take_profit <= pair.entry_price:
                validation_result["valid"] = False
                validation_result["errors"].append(f"CRITICAL: Buy order take-profit must be above entry price ({pair.take_profit:.2f} <= {pair.entry_price:.2f})")
            elif pair.action == "sell" and pair.take_profit >= pair.entry_price:
                validation_result["valid"] = False
                validation_result["errors"].append(f"CRITICAL: Sell order take-profit must be below entry price ({pair.take_profit:.2f} >= {pair.entry_price:.2f})")
        
        # Validate minimum price differences (Alpaca requirements)
        min_diff = 0.01
        if pair.stop_loss is not None and abs(pair.stop_loss - pair.entry_price) < min_diff:
            validation_result["valid"] = False
            validation_result["errors"].append(f"CRITICAL: Stop-loss too close to entry price (minimum {min_diff})")
            
        if pair.take_profit is not None and abs(pair.take_profit - pair.entry_price) < min_diff:
            validation_result["valid"] = False
            validation_result["errors"].append(f"CRITICAL: Take-profit too close to entry price (minimum {min_diff})")
        
        # Risk/reward ratio validation (warning level)
        if pair.stop_loss is not None and pair.take_profit is not None:
            if pair.action == "buy":
                risk = pair.entry_price - pair.stop_loss
                reward = pair.take_profit - pair.entry_price
            else:  # sell
                risk = pair.stop_loss - pair.entry_price
                reward = pair.entry_price - pair.take_profit
                
            if risk > 0 and reward > 0:
                risk_reward_ratio = reward / risk
                if risk_reward_ratio < 0.5:  # Risk more than 2x reward
                    validation_result["warnings"].append(f"Poor risk/reward ratio: {risk_reward_ratio:.2f} (high risk)")
                elif risk_reward_ratio > 3.0:  # Extremely favorable
                    validation_result["warnings"].append(f"Unusually high risk/reward ratio: {risk_reward_ratio:.2f} (verify targets)")
        
        # Log validation results
        if validation_result["errors"]:
            self.logger.error(f"Bracket validation FAILED for {pair.symbol}: {'; '.join(validation_result['errors'])}")
        elif validation_result["warnings"]:
            self.logger.warning(f"Bracket validation warnings for {pair.symbol}: {'; '.join(validation_result['warnings'])}")
        else:
            self.logger.info(f"✅ Bracket validation PASSED for {pair.symbol} - Entry: ${pair.entry_price:.2f}, SL: ${pair.stop_loss:.2f}, TP: ${pair.take_profit:.2f}")
            
        return validation_result
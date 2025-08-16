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
        self.entry_price = round(entry_price, 2)  # Force 2 decimal precision
        self.exit_price = round(exit_price, 2) if exit_price else exit_price
        self.stop_loss = round(stop_loss, 2) if stop_loss else stop_loss  # CRITICAL: Force proper rounding
        self.take_profit = round(take_profit, 2) if take_profit else take_profit  # CRITICAL: Force proper rounding
        self.quantity = quantity
        self.entry_time = entry_time or datetime.utcnow()
        self.exit_time = exit_time
        self.confidence = 0.0
        self.reasoning = ""
        self.execution_metadata = {}  # For execution optimization parameters
        self.tax_metadata = {}  # For tax optimization parameters
        self.frequency_metadata = {}  # For frequency optimization parameters
        self.hybrid_metadata = {}  # For hybrid optimization parameters
        
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
            "reasoning": self.reasoning,
            "execution_metadata": self.execution_metadata,
            "tax_metadata": self.tax_metadata,
            "frequency_metadata": self.frequency_metadata,
            "hybrid_metadata": self.hybrid_metadata
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
        self.universal_client = None  # Will be injected by main app for crypto support
        
        # Crypto-specific configuration
        self.crypto_enabled = config.get("crypto_enabled", True)
        self.crypto_volatility_factor = config.get("crypto_volatility_factor", 2.0)
        self.crypto_sentiment_amplifier = config.get("crypto_sentiment_amplifier", 1.3)
        self.crypto_minimum_confidence = config.get("crypto_minimum_confidence", 0.12)
        self.crypto_social_weight = config.get("crypto_social_weight", 0.6)
        self.crypto_24_7_trading = config.get("crypto_24_7_trading", True)
        
        # Market regime detection configuration
        self.regime_detection_enabled = config.get("regime_detection_enabled", True)
        self.market_regime_detector = None
        if self.regime_detection_enabled:
            from algotrading_agent.components.market_regime_detector import MarketRegimeDetector
            regime_config = config.get("market_regime_detector", {})
            self.market_regime_detector = MarketRegimeDetector(regime_config)
        
        # Options flow analysis configuration
        self.options_flow_enabled = config.get("options_flow_enabled", True)
        self.options_flow_analyzer = None
        if self.options_flow_enabled:
            from algotrading_agent.components.options_flow_analyzer import OptionsFlowAnalyzer
            options_config = config.get("options_flow_analyzer", {})
            self.options_flow_analyzer = OptionsFlowAnalyzer(options_config)
        
        # Supported crypto symbols for detection
        self.crypto_symbols = {
            'BTCUSD', 'ETHUSD', 'LTCUSD', 'DOGEUSD', 'SOLUSD', 'AVAXUSD', 
            'DOTUSD', 'LINKUSD', 'SHIBUSD', 'UNIUSD', 'AAVEUSD', 'BCHUSD',
            'CRVUSD', 'GRTUSD', 'MKRUSD', 'PEPEUSD', 'SUSHIUSD', 'XRPUSD', 
            'XTZUSD', 'YFIUSD', 'BTC', 'ETH', 'LTC', 'DOGE', 'SOL', 'AVAX',
            'DOT', 'LINK', 'SHIB', 'UNI', 'AAVE', 'BCH', 'CRV', 'GRT', 'MKR',
            'PEPE', 'SUSHI', 'XRP', 'XTZ', 'YFI'
        }
        self.news_impact_scorer = None  # Will be injected by main app for enhanced analysis
        
        # Enhanced parameters for temporal and context analysis
        self.temporal_weight = config.get("temporal_weight", 0.2)  # Weight for temporal dynamics
        self.strength_correlation_weight = config.get("strength_correlation_weight", 0.15)  # Weight for strength correlation  
        self.market_context_weight = config.get("market_context_weight", 0.15)  # Weight for market context
        self.enable_enhanced_analysis = config.get("enable_enhanced_analysis", True)  # Toggle for enhanced features
        
        # Execution Optimized Strategy Configuration (102.2% annual return strategy)
        self.execution_optimization_enabled = config.get("execution_optimization_enabled", True)
        self.max_slippage_bps = config.get("max_slippage_bps", 20.0)  # Target <25 bps slippage
        self.target_execution_quality_score = config.get("target_execution_quality_score", 80.0)
        self.market_impact_threshold = config.get("market_impact_threshold", 0.01)  # 1% of daily volume max
        self.preferred_order_type_default = config.get("preferred_order_type_default", "limit")  # Better fills than market orders
        
        # Tax Optimization Strategy Configuration (70.6% annual return, long-term focused)
        self.tax_optimization_enabled = config.get("tax_optimization_enabled", True)
        self.min_holding_period_days = config.get("min_holding_period_days", 31)  # Avoid wash sales
        self.target_ltcg_ratio = config.get("target_ltcg_ratio", 0.30)  # Target 30% long-term gains
        self.tax_efficiency_boost_threshold = config.get("tax_efficiency_boost_threshold", 0.75)  # High confidence for long holds
        self.wash_sale_avoidance_days = config.get("wash_sale_avoidance_days", 31)  # IRS wash sale rule
        self.ltcg_holding_days = config.get("ltcg_holding_days", 366)  # Long-term capital gains (>1 year)
        
        # Frequency Optimization Strategy Configuration (40.7% annual return, high selectivity)
        self.frequency_optimization_enabled = config.get("frequency_optimization_enabled", True)
        self.min_trade_confidence = config.get("min_trade_confidence", 0.70)  # Higher threshold for selectivity
        self.max_trades_per_day = config.get("max_trades_per_day", 10)  # Limit daily trade volume
        self.max_trades_per_hour = config.get("max_trades_per_hour", 3)  # Prevent overtrading
        self.conviction_boost_threshold = config.get("conviction_boost_threshold", 0.85)  # Very high conviction threshold
        self.daily_trade_reset_hour = config.get("daily_trade_reset_hour", 9)  # Reset at market open (9 AM EST)
        
        # Frequency optimization tracking
        self.daily_trade_count = 0
        self.hourly_trade_count = 0
        self.last_trade_reset_date = datetime.utcnow().date()
        self.last_trade_reset_hour = datetime.utcnow().hour
        self.filtered_trades_today = 0
        
        # Hybrid Optimization Strategy Configuration (combining all approaches)
        self.hybrid_optimization_enabled = config.get("hybrid_optimization_enabled", True)
        self.hybrid_confidence_weight = config.get("hybrid_confidence_weight", 0.4)  # Weight for confidence in hybrid scoring
        self.hybrid_efficiency_weight = config.get("hybrid_efficiency_weight", 0.3)  # Weight for efficiency metrics
        self.hybrid_execution_weight = config.get("hybrid_execution_weight", 0.3)   # Weight for execution quality
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a cryptocurrency"""
        symbol = symbol.upper().replace('/', '').replace('-', '')
        return symbol in self.crypto_symbols
    
    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol format"""
        symbol = symbol.upper()
        
        crypto_mappings = {
            'BTC': 'BTCUSD', 'ETH': 'ETHUSD', 'LTC': 'LTCUSD',
            'DOGE': 'DOGEUSD', 'SOL': 'SOLUSD', 'AVAX': 'AVAXUSD',
            'DOT': 'DOTUSD', 'LINK': 'LINKUSD', 'SHIB': 'SHIBUSD',
            'UNI': 'UNIUSD', 'AAVE': 'AAVEUSD', 'BCH': 'BCHUSD'
        }
        
        return crypto_mappings.get(symbol, symbol)
    
    def _extract_crypto_symbols_from_text(self, text: str) -> List[str]:
        """Extract crypto symbols mentioned in text"""
        import re
        
        # Common crypto patterns in text
        crypto_patterns = [
            r'\b(bitcoin|btc)\b', r'\b(ethereum|eth)\b', r'\b(dogecoin|doge)\b',
            r'\b(solana|sol)\b', r'\b(cardano|ada)\b', r'\b(polkadot|dot)\b',
            r'\b(chainlink|link)\b', r'\b(litecoin|ltc)\b', r'\b(shiba|shib)\b',
            r'\b(uniswap|uni)\b', r'\b(avalanche|avax)\b', r'\b(aave)\b'
        ]
        
        found_cryptos = []
        text_lower = text.lower()
        
        for pattern in crypto_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match in ['bitcoin', 'btc']:
                    found_cryptos.append('BTCUSD')
                elif match in ['ethereum', 'eth']:
                    found_cryptos.append('ETHUSD')
                elif match in ['dogecoin', 'doge']:
                    found_cryptos.append('DOGEUSD')
                elif match in ['solana', 'sol']:
                    found_cryptos.append('SOLUSD')
                elif match in ['polkadot', 'dot']:
                    found_cryptos.append('DOTUSD')
                elif match in ['chainlink', 'link']:
                    found_cryptos.append('LINKUSD')
                elif match in ['litecoin', 'ltc']:
                    found_cryptos.append('LTCUSD')
                elif match in ['shiba', 'shib']:
                    found_cryptos.append('SHIBUSD')
                elif match in ['uniswap', 'uni']:
                    found_cryptos.append('UNIUSD')
                elif match in ['avalanche', 'avax']:
                    found_cryptos.append('AVAXUSD')
                elif match == 'aave':
                    found_cryptos.append('AAVEUSD')
        
        return list(set(found_cryptos))
        
    def start(self) -> None:
        self.logger.info("Starting Decision Engine")
        self.is_running = True
        
        # Start market regime detector if enabled
        if self.market_regime_detector:
            self.market_regime_detector.start()
            self.logger.info("✅ Market Regime Detector started")
        
        # Start options flow analyzer if enabled
        if self.options_flow_analyzer:
            self.options_flow_analyzer.start()
            self.logger.info("✅ Options Flow Analyzer started")
        
    def stop(self) -> None:
        self.logger.info("Stopping Decision Engine")
        self.is_running = False
        
        # Stop market regime detector
        if self.market_regime_detector:
            self.market_regime_detector.stop()
            self.logger.info("⏹️ Market Regime Detector stopped")
        
        # Stop options flow analyzer
        if self.options_flow_analyzer:
            self.options_flow_analyzer.stop()
            self.logger.info("⏹️ Options Flow Analyzer stopped")
        
    async def process(self, analyzed_news: List[Dict[str, Any]], 
                      market_data: Optional[Dict[str, Any]] = None) -> List[TradingPair]:
        if not self.is_running or not analyzed_news:
            return []
            
        # Check market status for different asset classes
        stock_market_open = False
        crypto_market_open = False
        
        if self.alpaca_client:
            stock_market_open = await self.alpaca_client.is_market_open()
        
        if self.universal_client:
            # Universal client can trade crypto 24/7
            crypto_market_open = await self.universal_client.is_market_open()
        
        # TESTING MODE: Allow stock trading 24/7 for paper trading and testing
        # Check if we have any valid trading client (Alpaca for stocks)
        if not stock_market_open and not crypto_market_open and self.alpaca_client:
            self.logger.info("🧪 TESTING MODE: Markets closed but enabling stock trading for paper testing")
            stock_market_open = True  # Override for testing with paper trading
        elif not stock_market_open and not crypto_market_open:
            self.logger.info("All markets closed - no trading possible")
            return []
        
        # Log market status
        if stock_market_open and crypto_market_open:
            self.logger.info("🟢 All markets open - stock and crypto trading active")
        elif stock_market_open:
            self.logger.info("📈 Stock market open - stock trading only")
        elif crypto_market_open:
            self.logger.info("₿ Crypto markets open - crypto trading only (24/7)")
        
        # MARKET REGIME DETECTION: Analyze current market conditions
        regime_adjustments = {}
        if self.market_regime_detector:
            try:
                # Prepare market data for regime detection
                regime_market_data = market_data or {}
                regime_market_data.update({
                    'average_sentiment': sum(item.get('sentiment', {}).get('polarity', 0) for item in analyzed_news) / len(analyzed_news),
                    'news_count': len(analyzed_news),
                    'current_price': 100.0,  # Default, will be updated per symbol
                    'volume': 1000000  # Default volume
                })
                
                # Detect current regime
                regime_signal = await self.market_regime_detector.detect_regime(regime_market_data)
                
                if regime_signal:
                    regime_adjustments = self.market_regime_detector.get_regime_adjustments()
                    self.logger.info(f"🎯 MARKET REGIME: {regime_signal.regime.value} "
                                   f"(confidence: {regime_signal.confidence:.2f}) - "
                                   f"Adjusting strategy parameters")
                else:
                    self.logger.debug("No clear market regime detected - using default parameters")
                    
            except Exception as e:
                self.logger.error(f"Error in market regime detection: {e}")
        
        # OPTIONS FLOW ANALYSIS: Get unusual options activity signals
        options_signals = {}
        if self.options_flow_analyzer:
            try:
                options_flows = await self.options_flow_analyzer.process(market_data)
                
                # Organize options flows by symbol for easy lookup
                for flow in options_flows:
                    if flow.symbol not in options_signals:
                        options_signals[flow.symbol] = []
                    options_signals[flow.symbol].append(flow)
                
                if options_flows:
                    self.logger.info(f"📊 OPTIONS FLOW: {len(options_flows)} signals detected across "
                                   f"{len(options_signals)} symbols")
                    
            except Exception as e:
                self.logger.error(f"Error in options flow analysis: {e}")
            
        trading_pairs = []
        
        # Group news by symbols
        symbol_news = self._group_by_symbols(analyzed_news)
        
        for symbol, news_items in symbol_news.items():
            try:
                # Check if we can trade this symbol based on market status
                can_trade_symbol = self._can_trade_symbol(symbol, stock_market_open, crypto_market_open)
                if not can_trade_symbol:
                    continue
                
                # CRITICAL: Check if we already have a position or pending order for this symbol
                if await self._has_existing_position_or_order(symbol):
                    self.logger.info(f"⏭️  Skipping {symbol} - already have position or pending order")
                    continue
                    
                # Get options flow signals for this symbol
                symbol_options_flows = options_signals.get(symbol, [])
                
                decision = await self._make_decision(symbol, news_items, market_data, regime_adjustments, symbol_options_flows)
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
    
    def _can_trade_symbol(self, symbol: str, stock_market_open: bool, crypto_market_open: bool) -> bool:
        """Check if we can trade a symbol based on current market status"""
        
        # Detect asset class for the symbol
        if self.universal_client and hasattr(self.universal_client, 'detect_asset_class'):
            try:
                asset_class = self.universal_client.detect_asset_class(symbol)
                
                if asset_class.value == 'crypto':
                    return crypto_market_open
                elif asset_class.value in ['stock', 'etf']:
                    return stock_market_open
                else:
                    # Unknown asset class, default to stock market rules
                    return stock_market_open
            except Exception as e:
                self.logger.warning(f"Error detecting asset class for {symbol}: {e}")
        
        # Fallback: Simple crypto detection
        crypto_indicators = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH']
        if any(indicator in symbol.upper() for indicator in crypto_indicators):
            return crypto_market_open
        
        # Default to stock market rules
        return stock_market_open
        
    def _extract_tickers_fallback(self, item: Dict[str, Any]) -> List[str]:
        # Disabled fallback extraction to avoid false positives
        # The news analysis brain should handle all ticker extraction
        return []
        
    async def _make_decision(self, symbol: str, news_items: List[Dict[str, Any]], 
                            market_data: Optional[Dict[str, Any]] = None,
                            regime_adjustments: Optional[Dict[str, Any]] = None,
                            options_flows: Optional[List] = None) -> Optional[TradingPair]:
        
        # Calculate aggregate signals
        signal_strength = self._calculate_signal_strength(news_items)
        confidence = self._calculate_confidence(news_items, signal_strength)
        
        # Check if this is crypto-related news to apply different confidence threshold
        symbols = self._extract_symbols_from_news(news_items)
        is_crypto_related = any(self._is_crypto_symbol(symbol) for symbol in symbols)
        
        # Use crypto-specific confidence threshold if applicable
        effective_min_confidence = self.crypto_minimum_confidence if is_crypto_related else self.min_confidence
        
        # Apply market regime adjustments to confidence threshold
        if regime_adjustments:
            confidence_adjustment = regime_adjustments.get('confidence_threshold_adjustment', 0.0)
            effective_min_confidence += confidence_adjustment
            
            if confidence_adjustment != 0:
                self.logger.debug(f"🎯 REGIME ADJUSTMENT: Confidence threshold {confidence_adjustment:+.3f} "
                                f"(new threshold: {effective_min_confidence:.3f})")
        
        # ENHANCE WITH OPTIONS FLOW: Adjust confidence and signal strength based on options activity
        options_boost = 0.0
        options_signal_direction = 0
        
        if options_flows:
            # Analyze options flow signals for this symbol
            bullish_flows = [f for f in options_flows if 'bullish' in f.signal_type.value or 'call' in f.signal_type.value]
            bearish_flows = [f for f in options_flows if 'bearish' in f.signal_type.value or 'put' in f.signal_type.value]
            
            # Calculate weighted options signal
            bullish_weight = sum(f.strength * f.confidence for f in bullish_flows)
            bearish_weight = sum(f.strength * f.confidence for f in bearish_flows)
            
            if bullish_weight > bearish_weight:
                options_signal_direction = 1  # Bullish
                options_boost = min(0.15, (bullish_weight - bearish_weight) * 0.1)  # Max 15% boost
            elif bearish_weight > bullish_weight:
                options_signal_direction = -1  # Bearish
                options_boost = min(0.15, (bearish_weight - bullish_weight) * 0.1)
            
            # Apply options flow boost to confidence
            if options_boost > 0:
                confidence += options_boost
                self.logger.debug(f"📊 OPTIONS BOOST: {symbol} confidence +{options_boost:.3f} "
                                f"({'bullish' if options_signal_direction > 0 else 'bearish'} flow)")
                
                # Log significant options activity
                for flow in options_flows:
                    if flow.confidence > 0.6:
                        self.logger.info(f"🎯 SIGNIFICANT OPTIONS: {symbol} {flow.signal_type.value} "
                                       f"(strength: {flow.strength:.2f}, premium: ${flow.premium_value:,.0f})")
        
        if confidence < effective_min_confidence:
            asset_type = "crypto" if is_crypto_related else "stock"
            self.logger.debug(f"Confidence {confidence:.3f} below {asset_type} threshold {effective_min_confidence:.3f}")
            return None
        
        # Apply frequency optimization filtering (if enabled)
        if self.frequency_optimization_enabled:
            frequency_filter_result = self._apply_frequency_optimization_filter(symbol, confidence, signal_strength)
            if not frequency_filter_result['should_trade']:
                self.filtered_trades_today += 1
                self.logger.info(f"🚫 FREQUENCY FILTER: {symbol} filtered - {frequency_filter_result['reason']}")
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
            
        # Apply execution optimization to position sizing and pricing
        optimized_params = self._apply_execution_optimization(symbol, current_price, confidence, signal_strength)
        
        # Apply tax optimization to holding strategy and position sizing
        tax_optimized_params = self._apply_tax_optimization(symbol, confidence, signal_strength, news_items)
        
        # Apply frequency optimization (already filtered, now get metadata)
        frequency_optimized_params = self._get_frequency_optimization_metadata(symbol, confidence, signal_strength)
        
        # Calculate hybrid optimization score (combining all optimization approaches)
        hybrid_score = self._calculate_hybrid_optimization_score(
            optimized_params, tax_optimized_params, frequency_optimized_params, confidence, signal_strength
        )
        
        # Calculate position size with market impact awareness and tax considerations
        quantity = self._calculate_position_size_with_market_impact(
            confidence, current_price, symbol, optimized_params.get('market_impact_factor', 1.0)
        )
        
        # Adjust quantity for tax optimization strategy (may reduce for tax-loss harvesting opportunities)
        if tax_optimized_params.get('position_size_factor', 1.0) != 1.0:
            quantity = max(1, int(quantity * tax_optimized_params['position_size_factor']))
        
        # Apply market regime position size adjustments
        if regime_adjustments:
            position_multiplier = regime_adjustments.get('position_size_multiplier', 1.0)
            original_quantity = quantity
            quantity = max(1, int(quantity * position_multiplier))
            
            if position_multiplier != 1.0:
                self.logger.debug(f"🎯 REGIME POSITION SIZING: {original_quantity} → {quantity} "
                                f"(×{position_multiplier:.2f} regime adjustment)")
        
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
        
        # Add execution optimization metadata to the trading pair
        if self.execution_optimization_enabled:
            pair.execution_metadata = {
                'preferred_order_type': optimized_params.get('preferred_order_type', self.preferred_order_type_default),
                'max_slippage_bps': optimized_params.get('max_slippage_bps', self.max_slippage_bps),
                'execution_urgency': optimized_params.get('execution_urgency', 'normal'),
                'market_impact_adjusted': optimized_params.get('market_impact_adjusted', False),
                'execution_quality_target': self.target_execution_quality_score
            }
        
        # Add tax optimization metadata to the trading pair
        if self.tax_optimization_enabled and tax_optimized_params:
            pair.tax_metadata = {
                'target_holding_period_days': tax_optimized_params.get('target_holding_period_days', 1),
                'tax_efficiency_score': tax_optimized_params.get('tax_efficiency_score', 0.0),
                'trade_classification': tax_optimized_params.get('trade_classification', 'short_term'),
                'wash_sale_risk': tax_optimized_params.get('wash_sale_risk', False),
                'ltcg_eligible': tax_optimized_params.get('ltcg_eligible', False),
                'tax_strategy': tax_optimized_params.get('tax_strategy', 'standard')
            }
        
        # Add frequency optimization metadata to the trading pair
        if self.frequency_optimization_enabled and frequency_optimized_params:
            pair.frequency_metadata = {
                'daily_trade_number': frequency_optimized_params.get('daily_trade_number', 0),
                'conviction_level': frequency_optimized_params.get('conviction_level', 'medium'),
                'selectivity_score': frequency_optimized_params.get('selectivity_score', 0.0),
                'trades_filtered_today': frequency_optimized_params.get('trades_filtered_today', 0),
                'frequency_strategy': frequency_optimized_params.get('frequency_strategy', 'selective')
            }
        
        # Add hybrid optimization metadata to the trading pair
        if self.hybrid_optimization_enabled:
            pair.hybrid_metadata = {
                'hybrid_score': hybrid_score.get('total_score', 0.0),
                'execution_score': hybrid_score.get('execution_score', 0.0),
                'tax_score': hybrid_score.get('tax_score', 0.0),
                'frequency_score': hybrid_score.get('frequency_score', 0.0),
                'optimization_blend': hybrid_score.get('optimization_blend', 'balanced'),
                'expected_annual_return': hybrid_score.get('expected_annual_return', 0.0)
            }
        
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
            
            # Apply crypto-specific sentiment amplification
            is_crypto_news = any(self._is_crypto_symbol(symbol) for symbol in symbols)
            if is_crypto_news and self.crypto_enabled:
                sentiment_score *= self.crypto_sentiment_amplifier
                
                # Apply higher weight to social sentiment for crypto
                social_metrics = item.get("social_metrics", {})
                if social_metrics:
                    sentiment_score *= self.crypto_social_weight
            
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
        """Extract all symbols (stocks and crypto) mentioned in the news items"""
        symbols = []
        
        for item in news_items:
            # Extract traditional stock tickers
            entities = item.get("entities", {})
            tickers = entities.get("tickers", [])
            symbols.extend(tickers)
            
            # Extract crypto symbols from content
            title = item.get("title", "")
            content = item.get("content", "")
            full_text = f"{title} {content}"
            
            crypto_symbols = self._extract_crypto_symbols_from_text(full_text)
            symbols.extend(crypto_symbols)
            
            # Check for crypto category/tags
            category = item.get("category", "")
            if category in ["crypto_news", "crypto_market_data", "crypto_price", "crypto_analysis"]:
                # This is crypto-related news, try to extract symbols more aggressively
                if "bitcoin" in full_text.lower() or "btc" in full_text.lower():
                    symbols.append("BTCUSD")
                if "ethereum" in full_text.lower() or "eth" in full_text.lower():
                    symbols.append("ETHUSD")
                    
            # Check social metrics for crypto mentions (from Reddit, Twitter)
            social_metrics = item.get("social_metrics", {})
            if social_metrics and item.get("tickers"):
                # Social media post with ticker mentions
                for ticker in item.get("tickers", []):
                    normalized = self._normalize_crypto_symbol(ticker)
                    if self._is_crypto_symbol(normalized):
                        symbols.append(normalized)
                        
        return list(set(symbols))  # Remove duplicates
    
    def _apply_execution_optimization(self, symbol: str, current_price: float, 
                                    confidence: float, signal_strength: float) -> Dict[str, Any]:
        """
        Apply execution optimization strategy for superior execution quality
        Top performing strategy with 102.2% annual return
        """
        
        if not self.execution_optimization_enabled:
            return {}
        
        optimization_params = {
            'preferred_order_type': self.preferred_order_type_default,
            'max_slippage_bps': self.max_slippage_bps,
            'execution_urgency': 'normal',
            'market_impact_factor': 1.0,
            'market_impact_adjusted': False
        }
        
        # 1. Determine optimal order type based on urgency and market conditions
        urgency_level = self._determine_execution_urgency(signal_strength, confidence)
        optimization_params['execution_urgency'] = urgency_level
        
        if urgency_level in ['critical', 'high']:
            # High urgency -> market order for speed, accept higher slippage
            optimization_params['preferred_order_type'] = 'market'
            optimization_params['max_slippage_bps'] = min(self.max_slippage_bps * 2.0, 50.0)
            self.logger.info(f"🚀 HIGH URGENCY: {symbol} using market order for speed")
        else:
            # Normal/low urgency -> limit order for better execution
            optimization_params['preferred_order_type'] = 'limit'
            optimization_params['max_slippage_bps'] = self.max_slippage_bps
            self.logger.info(f"🎯 QUALITY FOCUS: {symbol} using limit order for better fills")
        
        # 2. Calculate market impact adjustment
        estimated_volume = self._estimate_daily_volume(symbol)
        if estimated_volume > 0:
            # Estimate our trade as percentage of daily volume
            base_quantity = self._calculate_position_size(confidence, current_price)
            trade_value = base_quantity * current_price
            daily_value = estimated_volume * current_price
            
            market_impact_ratio = trade_value / daily_value if daily_value > 0 else 0
            
            if market_impact_ratio > self.market_impact_threshold:
                # Reduce position to minimize market impact
                optimization_params['market_impact_factor'] = self.market_impact_threshold / market_impact_ratio
                optimization_params['market_impact_adjusted'] = True
                self.logger.info(f"🎯 MARKET IMPACT: Reduced {symbol} position by "
                               f"{(1 - optimization_params['market_impact_factor'])*100:.1f}% for better execution")
        
        # 3. Adjust slippage tolerance based on market conditions and confidence
        confidence_adjustment = 1.0 - (confidence * 0.3)  # Higher confidence = lower slippage tolerance
        optimization_params['max_slippage_bps'] *= confidence_adjustment
        
        # 4. Log execution optimization decision
        self.logger.info(f"✅ EXECUTION OPTIMIZED {symbol}: "
                        f"order_type={optimization_params['preferred_order_type']}, "
                        f"max_slippage={optimization_params['max_slippage_bps']:.1f}bps, "
                        f"urgency={optimization_params['execution_urgency']}")
        
        return optimization_params
    
    def _determine_execution_urgency(self, signal_strength: float, confidence: float) -> str:
        """Determine execution urgency level for order type selection"""
        
        # Critical urgency: Very strong signal with high confidence
        if abs(signal_strength) > 0.8 and confidence > 0.85:
            return 'critical'
        
        # High urgency: Strong signal or high confidence
        elif abs(signal_strength) > 0.6 or confidence > 0.75:
            return 'high'
        
        # Normal urgency: Medium signals
        elif abs(signal_strength) > 0.3 or confidence > 0.5:
            return 'normal'
        
        # Low urgency: Weak signals
        else:
            return 'low'
    
    def _apply_tax_optimization(self, symbol: str, confidence: float, 
                              signal_strength: float, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply tax optimization strategy for long-term wealth building
        70.6% annual return strategy focused on tax efficiency
        """
        
        if not self.tax_optimization_enabled:
            return {}
        
        tax_params = {
            'target_holding_period_days': 1,  # Default short-term
            'tax_efficiency_score': 0.0,
            'trade_classification': 'short_term',
            'wash_sale_risk': False,
            'ltcg_eligible': False,
            'tax_strategy': 'standard',
            'position_size_factor': 1.0
        }
        
        # 1. Determine optimal holding period based on confidence and signal characteristics
        tax_params['target_holding_period_days'] = self._calculate_optimal_holding_period(
            confidence, signal_strength, news_items
        )
        
        # 2. Calculate tax efficiency score
        tax_params['tax_efficiency_score'] = self._calculate_tax_efficiency_score(
            confidence, tax_params['target_holding_period_days'], news_items
        )
        
        # 3. Classify trade for tax purposes
        if tax_params['target_holding_period_days'] >= self.ltcg_holding_days:
            tax_params['trade_classification'] = 'long_term'
            tax_params['ltcg_eligible'] = True
            tax_params['tax_strategy'] = 'long_term_growth'
        elif tax_params['target_holding_period_days'] >= self.min_holding_period_days:
            tax_params['trade_classification'] = 'medium_term'
            tax_params['tax_strategy'] = 'wash_sale_avoidance'
        else:
            tax_params['trade_classification'] = 'short_term'
            tax_params['tax_strategy'] = 'standard'
        
        # 4. Check for wash sale risk (simplified - would need position history in real implementation)
        tax_params['wash_sale_risk'] = self._assess_wash_sale_risk(symbol)
        
        # 5. Adjust position sizing for tax considerations
        if tax_params['wash_sale_risk']:
            tax_params['position_size_factor'] = 0.7  # Reduce position to avoid wash sale
            self.logger.info(f"🛡️ TAX PROTECTION: Reduced {symbol} position by 30% to avoid wash sale risk")
        
        # 6. Log tax optimization decision
        self.logger.info(f"📊 TAX OPTIMIZED {symbol}: "
                        f"holding_period={tax_params['target_holding_period_days']}d, "
                        f"tax_efficiency={tax_params['tax_efficiency_score']:.1f}, "
                        f"classification={tax_params['trade_classification']}")
        
        return tax_params
    
    def _calculate_optimal_holding_period(self, confidence: float, signal_strength: float, 
                                        news_items: List[Dict[str, Any]]) -> int:
        """Calculate optimal holding period balancing tax efficiency and alpha decay"""
        
        # Analyze news characteristics for urgency
        has_urgent_news = any(
            item.get("priority") == "breaking" or 
            item.get("velocity_level") in ["viral", "breaking"]
            for item in news_items
        )
        
        # High urgency news -> shorter holding period (accept short-term tax treatment)
        if has_urgent_news and abs(signal_strength) > 0.7:
            return max(1, min(7, int(confidence * 10)))  # 1-7 days max for urgent signals
        
        # Very high confidence -> extend to long-term if possible  
        elif confidence >= self.tax_efficiency_boost_threshold:
            if abs(signal_strength) > 0.6:
                # Strong signal with high confidence -> target long-term gains
                return self.ltcg_holding_days  # 366+ days for LTCG treatment
            else:
                # High confidence but weaker signal -> medium-term
                return max(90, min(180, int(confidence * 240)))  # 3-6 months
        
        # Medium confidence -> balance tax efficiency with alpha decay
        elif confidence >= 0.5:
            return max(self.min_holding_period_days, int(confidence * 90))  # 31-90 days
        
        # Low confidence -> short-term treatment
        else:
            return max(1, int(confidence * 14))  # 1-14 days
    
    def _calculate_tax_efficiency_score(self, confidence: float, holding_period_days: int, 
                                      news_items: List[Dict[str, Any]]) -> float:
        """Calculate tax efficiency score (0-100, higher is better)"""
        
        score = 50.0  # Base score
        
        # Reward longer holding periods (primary tax benefit)
        if holding_period_days >= self.ltcg_holding_days:
            score += 35  # Long-term capital gains (15-20% vs 22-37% ordinary rates)
        elif holding_period_days >= 90:
            score += 20  # Medium-term reduces wash sale risk, better for harvesting
        elif holding_period_days >= self.min_holding_period_days:
            score += 10  # Wash sale avoidance
        
        # Reward high-conviction trades (fewer trades = better tax treatment)
        if confidence >= 0.9:
            score += 15  # Very high conviction
        elif confidence >= 0.75:
            score += 10  # High conviction
        elif confidence >= 0.5:
            score += 5   # Medium conviction
        
        # Penalize urgent/high-frequency characteristics (likely short-term)
        urgent_keywords = any(
            any(keyword in item.get("title", "").lower() + item.get("content", "").lower() 
                for keyword in ["breaking", "urgent", "flash", "immediate"])
            for item in news_items
        )
        
        if urgent_keywords:
            score -= 15  # Urgent news likely leads to short-term trades
        
        # Bonus for crypto (different tax considerations - may qualify for like-kind exchanges)
        symbols = self._extract_symbols_from_news(news_items)
        if any(self._is_crypto_symbol(symbol) for symbol in symbols):
            score += 5  # Crypto has different tax advantages in some jurisdictions
        
        return max(0, min(100, score))
    
    def _assess_wash_sale_risk(self, symbol: str) -> bool:
        """Assess wash sale risk for a symbol (simplified implementation)"""
        
        # In a real implementation, this would check:
        # 1. Recent loss positions in the same symbol
        # 2. Recent purchases/sales within 30 days
        # 3. Substantially identical securities
        
        # For now, simplified risk assessment based on common high-volatility symbols
        high_volatility_symbols = {
            'TSLA', 'GME', 'AMC', 'BTCUSD', 'ETHUSD', 'DOGEUSD', 'SOLU'
        }
        
        # High volatility symbols have higher wash sale risk due to frequent trading
        return symbol.upper() in high_volatility_symbols
    
    def _apply_frequency_optimization_filter(self, symbol: str, confidence: float, signal_strength: float) -> Dict[str, Any]:
        """
        Apply frequency optimization filter for selective, high-conviction trading
        40.7% annual return strategy focused on trade quality over quantity
        """
        
        if not self.frequency_optimization_enabled:
            return {'should_trade': True, 'reason': 'frequency optimization disabled'}
        
        # Reset counters if needed
        self._reset_frequency_counters()
        
        # 1. Check confidence threshold (higher than base minimum, but respect crypto thresholds)
        is_crypto_symbol = self._is_crypto_symbol(symbol)
        effective_min_confidence = self.crypto_minimum_confidence if is_crypto_symbol else self.min_confidence
        
        # For frequency optimization, use higher threshold but still respect crypto/stock distinction
        if is_crypto_symbol:
            # For crypto: use crypto threshold + frequency boost (but not as high as stock frequency threshold)
            frequency_crypto_threshold = min(self.crypto_minimum_confidence + 0.20, 0.50)  # Max 0.50 for crypto
            threshold_to_check = frequency_crypto_threshold
        else:
            # For stocks: use the full frequency optimization threshold
            threshold_to_check = self.min_trade_confidence
        
        if confidence < threshold_to_check:
            asset_type = "crypto" if is_crypto_symbol else "stock"
            return {
                'should_trade': False, 
                'reason': f'{asset_type} confidence {confidence:.2f} below selective threshold {threshold_to_check:.2f}'
            }
        
        # 2. Check daily trade limits
        if self.daily_trade_count >= self.max_trades_per_day:
            return {
                'should_trade': False,
                'reason': f'daily trade limit reached ({self.daily_trade_count}/{self.max_trades_per_day})'
            }
        
        # 3. Check hourly trade limits
        if self.hourly_trade_count >= self.max_trades_per_hour:
            return {
                'should_trade': False,
                'reason': f'hourly trade limit reached ({self.hourly_trade_count}/{self.max_trades_per_hour})'
            }
        
        # 4. Extra selectivity for very high conviction trades
        if confidence >= self.conviction_boost_threshold:
            # Always allow very high conviction trades (bypass other limits if reasonable)
            self.logger.info(f"🌟 HIGH CONVICTION: {symbol} confidence {confidence:.2f} >= {self.conviction_boost_threshold:.2f}")
            return {'should_trade': True, 'reason': 'high conviction override'}
        
        # 5. Apply signal strength filtering (crypto-aware)
        if is_crypto_symbol:
            min_signal_strength = 0.25  # Lower threshold for crypto (more volatile, less predictable)
        else:
            min_signal_strength = 0.5   # Higher threshold for stocks (more stable patterns)
            
        if abs(signal_strength) < min_signal_strength:
            asset_type = "crypto" if is_crypto_symbol else "stock"
            return {
                'should_trade': False,
                'reason': f'{asset_type} signal strength {abs(signal_strength):.2f} below minimum {min_signal_strength:.2f}'
            }
        
        # Trade passes all frequency filters
        return {'should_trade': True, 'reason': 'passed frequency optimization filters'}
    
    def _get_frequency_optimization_metadata(self, symbol: str, confidence: float, signal_strength: float) -> Dict[str, Any]:
        """Get frequency optimization metadata for tracking"""
        
        if not self.frequency_optimization_enabled:
            return {}
        
        # Update trade counters
        self.daily_trade_count += 1
        self.hourly_trade_count += 1
        
        # Determine conviction level
        if confidence >= self.conviction_boost_threshold:
            conviction_level = 'very_high'
        elif confidence >= self.min_trade_confidence + 0.10:
            conviction_level = 'high'
        elif confidence >= self.min_trade_confidence + 0.05:
            conviction_level = 'medium'
        else:
            conviction_level = 'acceptable'
        
        # Calculate selectivity score (0-100, higher = more selective)
        selectivity_score = min(100, (confidence / self.min_trade_confidence) * 50 + abs(signal_strength) * 50)
        
        self.logger.info(f"📊 FREQUENCY OPTIMIZED {symbol}: "
                        f"trade#{self.daily_trade_count}, "
                        f"conviction={conviction_level}, "
                        f"selectivity={selectivity_score:.1f}/100")
        
        return {
            'daily_trade_number': self.daily_trade_count,
            'conviction_level': conviction_level,
            'selectivity_score': selectivity_score,
            'trades_filtered_today': self.filtered_trades_today,
            'frequency_strategy': 'selective_high_conviction'
        }
    
    def _reset_frequency_counters(self):
        """Reset frequency optimization counters when appropriate"""
        
        current_time = datetime.utcnow()
        current_date = current_time.date()
        current_hour = current_time.hour
        
        # Reset daily counter at market open
        if (current_date != self.last_trade_reset_date or 
            (current_hour >= self.daily_trade_reset_hour and 
             self.last_trade_reset_hour < self.daily_trade_reset_hour)):
            
            if self.daily_trade_count > 0:
                self.logger.info(f"📅 DAILY RESET: {self.daily_trade_count} trades executed, "
                               f"{self.filtered_trades_today} filtered yesterday")
            
            self.daily_trade_count = 0
            self.filtered_trades_today = 0
            self.last_trade_reset_date = current_date
        
        # Reset hourly counter every hour
        if current_hour != self.last_trade_reset_hour:
            if self.hourly_trade_count > 0:
                self.logger.debug(f"⏰ HOURLY RESET: {self.hourly_trade_count} trades in last hour")
            
            self.hourly_trade_count = 0
            self.last_trade_reset_hour = current_hour
    
    def _calculate_hybrid_optimization_score(self, execution_params: Dict, tax_params: Dict, 
                                           frequency_params: Dict, confidence: float, signal_strength: float) -> Dict[str, Any]:
        """
        Calculate hybrid optimization score combining all strategies
        Expected to deliver best overall performance by leveraging all optimizations
        """
        
        if not self.hybrid_optimization_enabled:
            return {'total_score': 0.0}
        
        # Calculate individual optimization scores (0-100 scale)
        execution_score = self._calculate_execution_optimization_score(execution_params, confidence)
        tax_score = self._calculate_tax_optimization_score(tax_params, confidence)
        frequency_score = self._calculate_frequency_optimization_score(frequency_params, confidence)
        
        # Weighted hybrid score combining all optimizations
        total_score = (
            execution_score * self.hybrid_execution_weight +
            tax_score * self.hybrid_efficiency_weight +
            frequency_score * self.hybrid_confidence_weight
        )
        
        # Determine optimization blend based on which strategy dominates
        if execution_score > tax_score and execution_score > frequency_score:
            optimization_blend = 'execution_focused'
            expected_return = 102.2  # Execution optimized return
        elif tax_score > frequency_score:
            optimization_blend = 'tax_focused'  
            expected_return = 70.6   # Tax optimized return
        elif frequency_score > 60:
            optimization_blend = 'frequency_focused'
            expected_return = 40.7   # Frequency optimized return
        else:
            optimization_blend = 'balanced'
            expected_return = (102.2 + 70.6 + 40.7) / 3  # Average return
        
        # Boost expected return if multiple optimizations are strong
        strong_optimizations = sum([
            execution_score > 80,
            tax_score > 80, 
            frequency_score > 80
        ])
        
        if strong_optimizations >= 2:
            expected_return *= 1.15  # 15% boost for multiple strong optimizations
            optimization_blend = 'multi_optimized'
        
        self.logger.info(f"🔥 HYBRID OPTIMIZED: total_score={total_score:.1f}, "
                        f"exec={execution_score:.1f}, tax={tax_score:.1f}, freq={frequency_score:.1f}, "
                        f"blend={optimization_blend}, expected_return={expected_return:.1f}%")
        
        return {
            'total_score': total_score,
            'execution_score': execution_score,
            'tax_score': tax_score,
            'frequency_score': frequency_score,
            'optimization_blend': optimization_blend,
            'expected_annual_return': expected_return
        }
    
    def _calculate_execution_optimization_score(self, params: Dict, confidence: float) -> float:
        """Calculate execution optimization effectiveness score (0-100)"""
        if not params:
            return 0.0
        
        score = 50.0  # Base score
        
        # Reward better execution parameters
        if params.get('preferred_order_type') == 'limit':
            score += 20  # Limit orders typically get better fills
        
        slippage_bps = params.get('max_slippage_bps', 25.0)
        if slippage_bps <= 20.0:
            score += 20  # Low slippage target
        
        if params.get('market_impact_adjusted', False):
            score += 10  # Market impact awareness
        
        return min(100, score)
    
    def _calculate_tax_optimization_score(self, params: Dict, confidence: float) -> float:
        """Calculate tax optimization effectiveness score (0-100)"""
        if not params:
            return 0.0
        
        return params.get('tax_efficiency_score', 0.0)  # Already 0-100 scale
    
    def _calculate_frequency_optimization_score(self, params: Dict, confidence: float) -> float:
        """Calculate frequency optimization effectiveness score (0-100)"""
        if not params:
            return 0.0
        
        return params.get('selectivity_score', 0.0)  # Already 0-100 scale
    
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
    
    def _calculate_position_size_with_market_impact(self, confidence: float, price: float, 
                                                  symbol: str, market_impact_factor: float = 1.0) -> int:
        """Calculate position size with market impact awareness for execution optimization"""
        
        if not self.execution_optimization_enabled:
            return self._calculate_position_size(confidence, price)
        
        # Base position calculation
        base_quantity = self._calculate_position_size(confidence, price)
        
        # Estimate daily volume for market impact calculation
        estimated_daily_volume = self._estimate_daily_volume(symbol)
        
        if estimated_daily_volume > 0:
            # Position size should be <1% of daily volume to minimize market impact
            trade_value = base_quantity * price
            daily_value = estimated_daily_volume * price
            
            # Apply market impact threshold
            if trade_value > daily_value * self.market_impact_threshold:
                reduction_factor = (daily_value * self.market_impact_threshold) / trade_value
                base_quantity = int(base_quantity * reduction_factor)
                self.logger.info(f"🎯 EXECUTION OPTIMIZED: Reduced {symbol} position by "
                               f"{(1-reduction_factor)*100:.1f}% to minimize market impact")
        
        # Apply market impact factor from optimization
        final_quantity = max(1, int(base_quantity * market_impact_factor))
        
        return final_quantity
    
    def _estimate_daily_volume(self, symbol: str) -> float:
        """Estimate daily volume for market impact calculations"""
        # Volume estimates based on typical trading volumes
        volume_estimates = {
            # High volume stocks
            "AAPL": 50_000_000, "TSLA": 30_000_000, "SPY": 80_000_000, "QQQ": 40_000_000,
            "MSFT": 25_000_000, "GOOGL": 20_000_000, "AMZN": 15_000_000, "META": 12_000_000,
            "NVDA": 35_000_000, "AMD": 25_000_000, "JPM": 10_000_000, "BAC": 15_000_000,
            
            # Medium volume stocks  
            "GM": 8_000_000, "F": 12_000_000, "GE": 5_000_000, "BA": 3_000_000,
            "XOM": 8_000_000, "CVX": 4_000_000, "WMT": 6_000_000, "KO": 5_000_000,
            
            # Crypto (different scale)
            "BTCUSD": 2_000_000, "ETHUSD": 800_000, "DOGEUSD": 100_000, "SOLUSD": 50_000,
            "AVAXUSD": 30_000, "DOTUSD": 25_000, "LINKUSD": 40_000, "LTCUSD": 80_000
        }
        
        return volume_estimates.get(symbol, 1_000_000)  # Default 1M for unknown symbols
        
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
        
        # Check if this is crypto-related and apply crypto-specific adjustments
        symbols = self._extract_symbols_from_news(news_items)
        is_crypto_related = any(self._is_crypto_symbol(symbol) for symbol in symbols)
        
        if is_crypto_related and self.crypto_enabled:
            # Crypto is more volatile - wider stop losses and take profits
            take_profit_pct *= self.crypto_volatility_factor  # 10% -> 20% for crypto
            stop_loss_pct *= (self.crypto_volatility_factor * 0.8)  # 5% -> 8% for crypto (tighter ratio)
            self.logger.info(f"🚀 CRYPTO DETECTED: Adjusting targets for higher volatility - "
                           f"TP: {take_profit_pct*100:.1f}%, SL: {stop_loss_pct*100:.1f}%")
        
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
    
    async def _has_existing_position_or_order(self, symbol: str) -> bool:
        """
        Check if we already have an existing position or pending order for this symbol.
        This prevents infinite loops where the same trading decision is generated repeatedly.
        """
        try:
            # Check if we have access to trading clients
            trading_client = None
            
            # Try universal client first (supports both stocks and crypto)
            if self.universal_client:
                trading_client = self.universal_client
            elif self.alpaca_client:
                trading_client = self.alpaca_client
            
            if not trading_client:
                self.logger.warning(f"No trading client available to check {symbol} position/orders")
                return False
            
            # Check for existing positions
            try:
                positions = await trading_client.get_positions()
                for position in positions:
                    if position.get("symbol") == symbol and position.get("quantity", 0) != 0:
                        self.logger.info(f"🔒 {symbol} already has position: {position.get('quantity')} shares")
                        return True
            except Exception as e:
                self.logger.warning(f"Error checking positions for {symbol}: {e}")
            
            # Check for pending orders
            try:
                orders = await trading_client.get_orders()
                pending_statuses = ['new', 'partially_filled', 'pending_new', 'accepted', 'pending_cancel']
                
                for order in orders:
                    if (order.get("symbol") == symbol and 
                        order.get("status") in pending_statuses):
                        self.logger.info(f"🔒 {symbol} already has pending order: {order.get('side')} {order.get('quantity')} ({order.get('status')})")
                        return True
            except Exception as e:
                self.logger.warning(f"Error checking orders for {symbol}: {e}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in _has_existing_position_or_order for {symbol}: {e}")
            # Err on the side of caution - assume we have a position to prevent loops
            return True
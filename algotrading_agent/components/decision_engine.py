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
            
        # Calculate position size
        quantity = self._calculate_position_size(confidence, current_price)
        
        # Set stop loss and take profit with proper rounding for Alpaca requirements
        if action == "buy":
            # Buy: stop loss below, take profit above
            stop_loss = round(current_price * (1 - self.default_stop_loss_pct), 2)
            take_profit = round(current_price * (1 + self.default_take_profit_pct), 2)
        else:  # sell (short)
            # Sell: stop loss above, take profit below
            stop_loss = round(current_price * (1 + self.default_stop_loss_pct), 2)
            take_profit = round(current_price * (1 - self.default_take_profit_pct), 2)
            
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
        
        return pair
        
    def _calculate_signal_strength(self, news_items: List[Dict[str, Any]]) -> float:
        total_signal = 0.0
        total_weight = 0.0
        
        for item in news_items:
            sentiment = item.get("sentiment", {})
            impact_score = item.get("impact_score", 0.0)
            filter_score = item.get("filter_score", 0.0)
            
            # Calculate weighted signal
            sentiment_score = sentiment.get("polarity", 0.0)
            
            weight = (
                impact_score * self.impact_weight +
                filter_score * self.recency_weight +
                sentiment.get("confidence", 0.0) * self.sentiment_weight
            )
            
            total_signal += sentiment_score * weight
            total_weight += weight
            
        return total_signal / max(total_weight, 0.001)
        
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
        
        return min(confidence, 1.0)
        
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
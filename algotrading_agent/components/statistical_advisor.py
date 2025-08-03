import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from ..core.base import PersistentComponent
from .decision_engine import TradingPair


class TradeRecord:
    def __init__(self, pair: TradingPair, outcome: Dict[str, Any]):
        self.symbol = pair.symbol
        self.action = pair.action
        self.entry_price = pair.entry_price
        self.exit_price = outcome.get("exit_price")
        self.quantity = pair.quantity
        self.entry_time = pair.entry_time
        self.exit_time = outcome.get("exit_time")
        self.pnl = outcome.get("pnl", 0.0)
        self.confidence = pair.confidence
        self.reasoning = pair.reasoning
        self.market_conditions = outcome.get("market_conditions", {})
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "market_conditions": self.market_conditions,
            "duration_hours": self._calculate_duration_hours()
        }
        
    def _calculate_duration_hours(self) -> Optional[float]:
        if self.entry_time and self.exit_time:
            if isinstance(self.exit_time, str):
                exit_time = datetime.fromisoformat(self.exit_time.replace('Z', '+00:00'))
            else:
                exit_time = self.exit_time
            return (exit_time - self.entry_time).total_seconds() / 3600
        return None


class StatisticalAdvisor(PersistentComponent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("statistical_advisor", config)
        self.min_trades_for_analysis = config.get("min_trades_for_analysis", 10)
        self.lookback_days = config.get("lookback_days", 30)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        
        # Load historical trades
        self.trade_history = self._load_trade_history()
        
    def start(self) -> None:
        self.logger.info("Starting Statistical Advisor")
        self.is_running = True
        
    def stop(self) -> None:
        self.logger.info("Stopping Statistical Advisor")
        self.is_running = False
        
    def process(self, trading_pairs: List[TradingPair]) -> List[TradingPair]:
        if not self.is_running:
            return trading_pairs
            
        # Enhance trading pairs with statistical insights
        enhanced_pairs = []
        
        for pair in trading_pairs:
            try:
                insights = self._generate_insights(pair)
                pair.confidence = self._adjust_confidence(pair, insights)
                enhanced_pairs.append(pair)
            except Exception as e:
                self.logger.error(f"Error generating insights for {pair.symbol}: {e}")
                enhanced_pairs.append(pair)
                
        return enhanced_pairs
        
    def record_trade_outcome(self, pair: TradingPair, outcome: Dict[str, Any]) -> None:
        """Record the outcome of a completed trade"""
        trade_record = TradeRecord(pair, outcome)
        self.trade_history.append(trade_record.to_dict())
        
        # Keep only recent trades
        cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days * 2)
        self.trade_history = [
            trade for trade in self.trade_history
            if trade.get("entry_time") and 
            datetime.fromisoformat(trade["entry_time"].replace('Z', '+00:00')) > cutoff_date
        ]
        
        self._save_trade_history()
        self.logger.info(f"Recorded trade outcome for {pair.symbol}")
        
    def _load_trade_history(self) -> List[Dict[str, Any]]:
        return self.get_memory("trade_history", [])
        
    def _save_trade_history(self) -> None:
        self.update_memory("trade_history", self.trade_history)
        
    def _generate_insights(self, pair: TradingPair) -> Dict[str, Any]:
        insights = {
            "symbol_performance": self._analyze_symbol_performance(pair.symbol),
            "action_performance": self._analyze_action_performance(pair.action),
            "confidence_correlation": self._analyze_confidence_correlation(pair.confidence),
            "market_conditions": self._analyze_market_conditions(),
            "timing_analysis": self._analyze_timing_patterns()
        }
        
        return insights
        
    def _analyze_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        symbol_trades = [t for t in self.trade_history if t["symbol"] == symbol]
        
        if len(symbol_trades) < 3:
            return {"insufficient_data": True}
            
        pnls = [t["pnl"] for t in symbol_trades if t["pnl"] is not None]
        
        if not pnls:
            return {"insufficient_data": True}
            
        return {
            "total_trades": len(symbol_trades),
            "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
            "avg_pnl": np.mean(pnls),
            "total_pnl": sum(pnls),
            "volatility": np.std(pnls) if len(pnls) > 1 else 0.0,
            "max_win": max(pnls),
            "max_loss": min(pnls)
        }
        
    def _analyze_action_performance(self, action: str) -> Dict[str, Any]:
        action_trades = [t for t in self.trade_history if t["action"] == action]
        
        if len(action_trades) < 5:
            return {"insufficient_data": True}
            
        pnls = [t["pnl"] for t in action_trades if t["pnl"] is not None]
        
        if not pnls:
            return {"insufficient_data": True}
            
        return {
            "total_trades": len(action_trades),
            "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
            "avg_pnl": np.mean(pnls),
            "sharpe_ratio": self._calculate_sharpe_ratio(pnls)
        }
        
    def _analyze_confidence_correlation(self, confidence: float) -> Dict[str, Any]:
        # Analyze performance at similar confidence levels
        confidence_range = 0.1
        similar_trades = [
            t for t in self.trade_history 
            if abs(t.get("confidence", 0) - confidence) <= confidence_range
            and t.get("pnl") is not None
        ]
        
        if len(similar_trades) < 3:
            return {"insufficient_data": True}
            
        pnls = [t["pnl"] for t in similar_trades]
        
        return {
            "trades_at_confidence": len(similar_trades),
            "avg_pnl": np.mean(pnls),
            "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
            "confidence_reliability": self._calculate_confidence_reliability(similar_trades)
        }
        
    def _analyze_market_conditions(self) -> Dict[str, Any]:
        # Simplified market condition analysis
        recent_trades = [
            t for t in self.trade_history
            if t.get("entry_time") and 
            datetime.fromisoformat(t["entry_time"].replace('Z', '+00:00')) > 
            datetime.utcnow() - timedelta(days=7)
        ]
        
        if not recent_trades:
            return {"trend": "unknown"}
            
        recent_pnls = [t["pnl"] for t in recent_trades if t["pnl"] is not None]
        
        if not recent_pnls:
            return {"trend": "unknown"}
            
        avg_recent_pnl = np.mean(recent_pnls)
        
        return {
            "recent_performance": avg_recent_pnl,
            "market_trend": "bullish" if avg_recent_pnl > 0 else "bearish",
            "volatility": np.std(recent_pnls) if len(recent_pnls) > 1 else 0.0
        }
        
    def _analyze_timing_patterns(self) -> Dict[str, Any]:
        # Analyze performance by hour of day, day of week
        hourly_performance = {}
        
        for trade in self.trade_history:
            if trade.get("entry_time") and trade.get("pnl") is not None:
                try:
                    entry_time = datetime.fromisoformat(trade["entry_time"].replace('Z', '+00:00'))
                    hour = entry_time.hour
                    
                    if hour not in hourly_performance:
                        hourly_performance[hour] = []
                    hourly_performance[hour].append(trade["pnl"])
                except:
                    continue
                    
        best_hours = []
        for hour, pnls in hourly_performance.items():
            if len(pnls) >= 3:
                avg_pnl = np.mean(pnls)
                if avg_pnl > 0:
                    best_hours.append((hour, avg_pnl))
                    
        best_hours.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "best_trading_hours": [h[0] for h in best_hours[:3]],
            "hourly_analysis": hourly_performance
        }
        
    def _adjust_confidence(self, pair: TradingPair, insights: Dict[str, Any]) -> float:
        adjusted_confidence = pair.confidence
        
        # Adjust based on symbol performance
        symbol_perf = insights.get("symbol_performance", {})
        if not symbol_perf.get("insufficient_data"):
            win_rate = symbol_perf.get("win_rate", 0.5)
            if win_rate > 0.6:
                adjusted_confidence *= 1.1
            elif win_rate < 0.4:
                adjusted_confidence *= 0.9
                
        # Adjust based on action performance
        action_perf = insights.get("action_performance", {})
        if not action_perf.get("insufficient_data"):
            action_win_rate = action_perf.get("win_rate", 0.5)
            if action_win_rate > 0.6:
                adjusted_confidence *= 1.05
            elif action_win_rate < 0.4:
                adjusted_confidence *= 0.95
                
        # Adjust based on market conditions
        market_conditions = insights.get("market_conditions", {})
        market_trend = market_conditions.get("market_trend")
        if market_trend == "bearish" and pair.action == "buy":
            adjusted_confidence *= 0.9
        elif market_trend == "bullish" and pair.action == "sell":
            adjusted_confidence *= 0.9
            
        return min(max(adjusted_confidence, 0.1), 1.0)
        
    def _calculate_sharpe_ratio(self, pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
            
        mean_return = np.mean(pnls)
        std_return = np.std(pnls)
        
        if std_return == 0:
            return 0.0
            
        return mean_return / std_return
        
    def _calculate_confidence_reliability(self, trades: List[Dict[str, Any]]) -> float:
        # Calculate how well confidence predicts actual outcomes
        correct_predictions = 0
        
        for trade in trades:
            confidence = trade.get("confidence", 0.5)
            pnl = trade.get("pnl", 0)
            
            # High confidence should correspond to positive PnL
            if (confidence > 0.7 and pnl > 0) or (confidence < 0.3 and pnl < 0):
                correct_predictions += 1
                
        return correct_predictions / len(trades) if trades else 0.0
        
    def get_performance_report(self) -> Dict[str, Any]:
        if len(self.trade_history) < self.min_trades_for_analysis:
            return {"insufficient_data": True, "total_trades": len(self.trade_history)}
            
        pnls = [t["pnl"] for t in self.trade_history if t["pnl"] is not None]
        
        if not pnls:
            return {"insufficient_data": True}
            
        return {
            "total_trades": len(self.trade_history),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
            "sharpe_ratio": self._calculate_sharpe_ratio(pnls),
            "max_win": max(pnls),
            "max_loss": min(pnls),
            "volatility": np.std(pnls),
            "profit_factor": self._calculate_profit_factor(pnls)
        }
        
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1
        
        return total_wins / total_losses if total_losses > 0 else float('inf')
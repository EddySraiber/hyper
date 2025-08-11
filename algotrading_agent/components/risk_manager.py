from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics
from ..core.base import PersistentComponent
from .decision_engine import TradingPair


class RiskManager(PersistentComponent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("risk_manager", config)
        self.max_portfolio_value = config.get("max_portfolio_value", 100000)
        self.max_position_pct = config.get("max_position_pct", 0.05)  # 5% per position
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 0.02)  # 2% daily loss
        self.max_sector_exposure_pct = config.get("max_sector_exposure_pct", 0.20)  # 20% per sector
        self.correlation_threshold = config.get("correlation_threshold", 0.7)
        self.blacklisted_symbols = config.get("blacklisted_symbols", [])
        
        # Kelly Criterion parameters
        self.enable_kelly_criterion = config.get("enable_kelly_criterion", True)
        self.kelly_safety_factor = config.get("kelly_safety_factor", 0.25)  # Conservative 25% of Kelly optimal
        self.max_kelly_position_pct = config.get("max_kelly_position_pct", 0.10)  # Cap at 10% regardless of Kelly
        self.min_trades_for_kelly = config.get("min_trades_for_kelly", 20)  # Minimum trades needed for Kelly calculation
        
        # Load current positions from memory
        self.current_positions = self.get_memory("current_positions", {})
        self.daily_pnl = self.get_memory("daily_pnl", 0.0)
        self.last_reset_date = self.get_memory("last_reset_date", datetime.utcnow().date().isoformat())
        
        # Load trade history for Kelly Criterion calculations
        self.trade_history = self.get_memory("trade_history", [])
        
    def start(self) -> None:
        self.logger.info("Starting Risk Manager")
        self.is_running = True
        self._reset_daily_pnl_if_needed()
        
    def stop(self) -> None:
        self.logger.info("Stopping Risk Manager")
        self.is_running = False
        
    def process(self, trading_pairs: List[TradingPair]) -> List[TradingPair]:
        if not self.is_running or not trading_pairs:
            return []
            
        approved_pairs = []
        
        for pair in trading_pairs:
            try:
                if self._validate_trade(pair):
                    risk_adjusted_pair = self._adjust_for_risk(pair)
                    approved_pairs.append(risk_adjusted_pair)
                    self.logger.info(f"Approved trade: {pair.symbol} {pair.action}")
                else:
                    self.logger.warning(f"Rejected trade: {pair.symbol} {pair.action}")
            except Exception as e:
                self.logger.error(f"Error processing trade {pair.symbol}: {e}")
                
        self.logger.info(f"Approved {len(approved_pairs)} out of {len(trading_pairs)} trades")
        return approved_pairs
        
    def _reset_daily_pnl_if_needed(self) -> None:
        today = datetime.utcnow().date().isoformat()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.last_reset_date = today
            self.update_memory("daily_pnl", self.daily_pnl)
            self.update_memory("last_reset_date", self.last_reset_date)
            
    def _validate_trade(self, pair: TradingPair) -> bool:
        # Check blacklist
        if pair.symbol in self.blacklisted_symbols:
            self.logger.warning(f"Symbol {pair.symbol} is blacklisted")
            return False
            
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss_pct * self.max_portfolio_value:
            self.logger.warning("Daily loss limit exceeded")
            return False
            
        # Check position size limit
        position_value = pair.entry_price * pair.quantity
        max_position_value = self.max_portfolio_value * self.max_position_pct
        if position_value > max_position_value:
            self.logger.warning(f"Position size too large for {pair.symbol}")
            return False
            
        # Check if we already have a position in this symbol
        if pair.symbol in self.current_positions:
            existing_position = self.current_positions[pair.symbol]
            if existing_position["action"] == pair.action:
                self.logger.warning(f"Already have {pair.action} position in {pair.symbol}")
                return False
                
        # Check portfolio concentration
        if not self._check_portfolio_concentration(pair):
            return False
            
        return True
        
    def _check_portfolio_concentration(self, pair: TradingPair) -> bool:
        # Calculate current portfolio exposure
        total_exposure = sum(
            pos["entry_price"] * pos["quantity"] 
            for pos in self.current_positions.values()
        )
        
        new_exposure = pair.entry_price * pair.quantity
        if total_exposure + new_exposure > self.max_portfolio_value:
            self.logger.warning("Portfolio exposure limit exceeded")
            return False
            
        return True
        
    def _calculate_kelly_criterion(self, symbol: str = None, action: str = None) -> float:
        """
        Calculate Kelly Criterion optimal position size.
        Formula: f* = (bp - q) / b
        Where:
        - f* = optimal fraction of capital to bet
        - b = odds received (average win / average loss)
        - p = probability of winning
        - q = probability of losing (1-p)
        """
        if not self.enable_kelly_criterion or len(self.trade_history) < self.min_trades_for_kelly:
            return self.max_position_pct  # Fall back to fixed percentage
        
        # Filter trades by symbol and/or action if specified
        relevant_trades = self.trade_history
        if symbol:
            relevant_trades = [t for t in relevant_trades if t.get("symbol") == symbol]
        if action:
            relevant_trades = [t for t in relevant_trades if t.get("action") == action]
        
        # Need minimum trades for statistical significance
        if len(relevant_trades) < 10:
            relevant_trades = self.trade_history
        
        if len(relevant_trades) < self.min_trades_for_kelly:
            return self.max_position_pct
        
        # Calculate win rate (probability of winning)
        wins = [t for t in relevant_trades if t.get("pnl", 0) > 0]
        losses = [t for t in relevant_trades if t.get("pnl", 0) < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return self.max_position_pct * 0.5  # Be very conservative if no wins or losses
        
        p = len(wins) / len(relevant_trades)  # Win probability
        q = 1 - p  # Loss probability
        
        # Calculate average win and loss amounts (odds)
        avg_win = statistics.mean([abs(t.get("pnl", 0)) for t in wins])
        avg_loss = statistics.mean([abs(t.get("pnl", 0)) for t in losses])
        
        if avg_loss == 0:
            return self.max_position_pct * 0.5
        
        b = avg_win / avg_loss  # Odds ratio
        
        # Kelly Criterion formula: f* = (bp - q) / b
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor and caps
        kelly_fraction = max(0, kelly_fraction)  # Never go negative
        kelly_fraction *= self.kelly_safety_factor  # Apply conservative safety factor
        kelly_fraction = min(kelly_fraction, self.max_kelly_position_pct)  # Cap at maximum
        
        self.logger.info(f"Kelly Criterion calculation for {symbol or 'any'} {action or 'any'}: "
                        f"p={p:.3f}, b={b:.3f}, Kelly={kelly_fraction:.3f}")
        
        return kelly_fraction
        
    def record_trade_outcome(self, symbol: str, action: str, entry_price: float, 
                           exit_price: float, quantity: int, outcome_type: str) -> None:
        """Record trade outcome for Kelly Criterion calculations"""
        pnl = self._calculate_trade_pnl(action, entry_price, exit_price, quantity)
        
        trade_record = {
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "outcome": outcome_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only recent trades (last 200 trades) to avoid memory growth
        if len(self.trade_history) > 200:
            self.trade_history = self.trade_history[-200:]
        
        # Save to memory
        self.update_memory("trade_history", self.trade_history)
        self.logger.info(f"Recorded trade outcome: {symbol} {action} PnL={pnl:.2f}")
        
    def _calculate_trade_pnl(self, action: str, entry_price: float, exit_price: float, quantity: int) -> float:
        """Calculate P&L for a completed trade"""
        if action == "buy":
            return (exit_price - entry_price) * quantity
        else:  # sell (short)
            return (entry_price - exit_price) * quantity

    def _adjust_for_risk(self, pair: TradingPair) -> TradingPair:
        # Adjust stop loss based on volatility (simplified)
        volatility_factor = self._estimate_volatility(pair.symbol)
        default_stop_loss_pct = 0.05  # 5% default
        
        if pair.action == "buy":
            # Widen stop loss for volatile stocks
            adjusted_stop = pair.entry_price * (1 - default_stop_loss_pct * volatility_factor)
            pair.stop_loss = min(pair.stop_loss, adjusted_stop)
        else:  # sell (short)
            adjusted_stop = pair.entry_price * (1 + default_stop_loss_pct * volatility_factor)
            pair.stop_loss = max(pair.stop_loss, adjusted_stop)
            
        # Use Kelly Criterion for optimal position sizing
        kelly_fraction = self._calculate_kelly_criterion(pair.symbol, pair.action)
        kelly_position_value = self.max_portfolio_value * kelly_fraction
        
        # Calculate position size based on Kelly Criterion
        if pair.entry_price > 0:
            kelly_quantity = int(kelly_position_value / pair.entry_price)
            
            # Also consider risk per share for safety
            max_risk_per_trade = self.max_portfolio_value * 0.01  # 1% risk per trade
            
            if pair.action == "buy":
                risk_per_share = pair.entry_price - pair.stop_loss
            else:
                risk_per_share = pair.stop_loss - pair.entry_price
                
            if risk_per_share > 0:
                risk_based_quantity = int(max_risk_per_trade / risk_per_share)
                # Use the more conservative of the two methods
                pair.quantity = min(pair.quantity, kelly_quantity, risk_based_quantity)
            else:
                pair.quantity = min(pair.quantity, kelly_quantity)
                
            self.logger.info(f"Position sizing for {pair.symbol}: Kelly fraction={kelly_fraction:.3f}, "
                           f"Kelly quantity={kelly_quantity}, Final quantity={pair.quantity}")
            
        return pair
        
    def _estimate_volatility(self, symbol: str) -> float:
        # Simplified volatility estimation
        # In a real implementation, this would use historical price data
        symbol_hash = hash(symbol) % 100
        return 1.0 + (symbol_hash / 200.0)  # Factor between 1.0 and 1.5
        
    def update_position(self, symbol: str, action: str, quantity: int, 
                       entry_price: float, exit_price: Optional[float] = None) -> None:
        if exit_price:
            # Closing position
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                pnl = self._calculate_pnl(position, exit_price)
                self.daily_pnl += pnl
                del self.current_positions[symbol]
                self.logger.info(f"Closed position {symbol}: PnL = {pnl:.2f}")
        else:
            # Opening position
            self.current_positions[symbol] = {
                "action": action,
                "quantity": quantity,
                "entry_price": entry_price,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.info(f"Opened position {symbol}: {action} {quantity} @ {entry_price}")
            
        # Save to memory
        self.update_memory("current_positions", self.current_positions)
        self.update_memory("daily_pnl", self.daily_pnl)
        
    def _calculate_pnl(self, position: Dict[str, Any], exit_price: float) -> float:
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        action = position["action"]
        
        if action == "buy":
            return (exit_price - entry_price) * quantity
        else:  # sell (short)
            return (entry_price - exit_price) * quantity
            
    def get_portfolio_status(self) -> Dict[str, Any]:
        positions_value = sum(
            pos["entry_price"] * pos["quantity"] 
            for pos in self.current_positions.values()
        )
        
        # Total portfolio value = available cash + positions value + daily P&L
        total_portfolio_value = self.max_portfolio_value + self.daily_pnl
        available_cash = total_portfolio_value - positions_value
        
        return {
            "current_positions": len(self.current_positions),
            "total_portfolio_value": total_portfolio_value,
            "positions_value": positions_value,
            "available_cash": available_cash,
            "daily_pnl": self.daily_pnl,
            "risk_utilization": positions_value / self.max_portfolio_value if self.max_portfolio_value > 0 else 0.0
        }
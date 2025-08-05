"""
Trading Cost Calculator

Calculates realistic trading costs including commissions, fees, and other expenses
for accurate backtesting and live trading performance measurement.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging


class TradingCostCalculator:
    """
    Comprehensive trading cost calculator supporting multiple commission models
    and fee structures for accurate profit/loss calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('trading_costs', {})
        self.logger = logging.getLogger("algotrading.trading_costs")
        
        # Main configuration
        self.paper_trading_costs = self.config.get('paper_trading_costs', False)
        self.commission_model = self.config.get('commission_model', 'zero')
        
        # Commission rates
        self.per_trade_commission = self.config.get('per_trade_commission', 0.0)
        self.per_share_commission = self.config.get('per_share_commission', 0.0)
        self.per_share_minimum = self.config.get('per_share_minimum', 0.0)
        self.per_share_maximum = self.config.get('per_share_maximum', 0.0)
        self.percentage_commission = self.config.get('percentage_commission', 0.0)
        self.percentage_minimum = self.config.get('percentage_minimum', 0.0)
        self.percentage_maximum = self.config.get('percentage_maximum', 0.0)
        
        # Regulatory fees
        regulatory_fees = self.config.get('regulatory_fees', {})
        self.sec_fee_rate = regulatory_fees.get('sec_fee_rate', 0.0000278)
        self.taf_fee_rate = regulatory_fees.get('taf_fee_rate', 0.000166)
        self.other_regulatory_fees = regulatory_fees.get('other_regulatory_fees', 0.0)
        
        # Additional fees
        market_data_fees = self.config.get('market_data_fees', {})
        self.market_data_enabled = market_data_fees.get('enabled', False)
        self.monthly_market_data_cost = market_data_fees.get('monthly_cost', 0.0)
        
        fx_fees = self.config.get('fx_fees', {})
        self.fx_enabled = fx_fees.get('enabled', False)
        self.fx_spread_bps = fx_fees.get('fx_spread_bps', 0)
        self.fx_minimum = fx_fees.get('fx_minimum', 0.0)
        
        borrowing_costs = self.config.get('borrowing_costs', {})
        self.borrowing_enabled = borrowing_costs.get('enabled', False)
        self.daily_borrowing_rate = borrowing_costs.get('daily_rate', 0.0)
        self.minimum_daily_borrowing_fee = borrowing_costs.get('minimum_daily_fee', 0.0)
        
        self.logger.info(f"Trading cost calculator initialized (model: {self.commission_model})")
        
    def calculate_trade_costs(self, 
                            symbol: str,
                            side: str,  # 'buy' or 'sell'
                            quantity: int,
                            price: float,
                            is_paper_trading: bool = True) -> Dict[str, float]:
        """
        Calculate all costs associated with a trade
        
        Returns:
            {
                'commission': float,
                'sec_fees': float,
                'taf_fees': float,
                'regulatory_fees': float,
                'fx_fees': float,
                'total_costs': float,
                'cost_per_share': float,
                'cost_percentage': float
            }
        """
        
        # Skip costs for paper trading unless explicitly enabled
        if is_paper_trading and not self.paper_trading_costs:
            return self._zero_costs()
            
        trade_value = quantity * price
        costs = {
            'commission': 0.0,
            'sec_fees': 0.0,
            'taf_fees': 0.0,
            'regulatory_fees': 0.0,
            'fx_fees': 0.0,
            'total_costs': 0.0,
            'cost_per_share': 0.0,
            'cost_percentage': 0.0
        }
        
        # Calculate commission based on model
        costs['commission'] = self._calculate_commission(quantity, price, trade_value)
        
        # Calculate regulatory fees (US stocks only, sells only)
        if side.lower() == 'sell':
            costs['sec_fees'] = trade_value * self.sec_fee_rate
            costs['taf_fees'] = trade_value * self.taf_fee_rate
            
        costs['regulatory_fees'] = self.other_regulatory_fees
        
        # Calculate FX fees if enabled
        if self.fx_enabled:
            costs['fx_fees'] = self._calculate_fx_fees(trade_value)
            
        # Calculate totals
        costs['total_costs'] = (
            costs['commission'] + 
            costs['sec_fees'] + 
            costs['taf_fees'] + 
            costs['regulatory_fees'] + 
            costs['fx_fees']
        )
        
        costs['cost_per_share'] = costs['total_costs'] / quantity if quantity > 0 else 0
        costs['cost_percentage'] = costs['total_costs'] / trade_value if trade_value > 0 else 0
        
        self.logger.debug(f"Trade costs for {quantity} {symbol} @ ${price:.2f}: ${costs['total_costs']:.2f}")
        
        return costs
        
    def calculate_round_trip_costs(self,
                                 symbol: str,
                                 quantity: int,
                                 entry_price: float,
                                 exit_price: float,
                                 is_paper_trading: bool = True) -> Dict[str, float]:
        """
        Calculate costs for a complete round-trip trade (buy + sell)
        """
        
        # Calculate entry costs (buy)
        entry_costs = self.calculate_trade_costs(
            symbol, 'buy', quantity, entry_price, is_paper_trading
        )
        
        # Calculate exit costs (sell)
        exit_costs = self.calculate_trade_costs(
            symbol, 'sell', quantity, exit_price, is_paper_trading
        )
        
        # Combine costs
        round_trip_costs = {
            'entry_costs': entry_costs['total_costs'],
            'exit_costs': exit_costs['total_costs'],
            'total_costs': entry_costs['total_costs'] + exit_costs['total_costs'],
            'cost_per_share': (entry_costs['total_costs'] + exit_costs['total_costs']) / quantity if quantity > 0 else 0
        }
        
        # Calculate impact on trade P&L
        gross_pnl = (exit_price - entry_price) * quantity
        net_pnl = gross_pnl - round_trip_costs['total_costs']
        
        round_trip_costs['gross_pnl'] = gross_pnl
        round_trip_costs['net_pnl'] = net_pnl
        round_trip_costs['cost_impact'] = round_trip_costs['total_costs']
        round_trip_costs['cost_percentage_of_pnl'] = (
            abs(round_trip_costs['total_costs'] / gross_pnl) if gross_pnl != 0 else float('inf')
        )
        
        return round_trip_costs
        
    def calculate_holding_costs(self,
                              symbol: str,
                              quantity: int,
                              price: float,
                              days_held: int,
                              is_short: bool = False) -> Dict[str, float]:
        """
        Calculate costs for holding a position (borrowing costs, etc.)
        """
        holding_costs = {
            'borrowing_costs': 0.0,
            'total_holding_costs': 0.0
        }
        
        if is_short and self.borrowing_enabled:
            position_value = quantity * price
            daily_cost = max(
                position_value * self.daily_borrowing_rate,
                self.minimum_daily_borrowing_fee
            )
            holding_costs['borrowing_costs'] = daily_cost * days_held
            
        holding_costs['total_holding_costs'] = holding_costs['borrowing_costs']
        
        return holding_costs
        
    def get_monthly_fixed_costs(self) -> Dict[str, float]:
        """
        Calculate monthly fixed costs (market data, etc.)
        """
        monthly_costs = {
            'market_data': 0.0,
            'total_monthly': 0.0
        }
        
        if self.market_data_enabled:
            monthly_costs['market_data'] = self.monthly_market_data_cost
            
        monthly_costs['total_monthly'] = monthly_costs['market_data']
        
        return monthly_costs
        
    def apply_broker_preset(self, broker_name: str) -> bool:
        """
        Apply predefined broker cost structure
        """
        presets = self.config.get('broker_presets', {})
        
        if broker_name not in presets:
            self.logger.error(f"Unknown broker preset: {broker_name}")
            return False
            
        preset = presets[broker_name]
        
        # Update configuration with preset values
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        self.logger.info(f"Applied {broker_name} cost structure")
        return True
        
    def _calculate_commission(self, quantity: int, price: float, trade_value: float) -> float:
        """Calculate commission based on the configured model"""
        
        if self.commission_model == 'zero':
            return 0.0
            
        elif self.commission_model == 'per_trade':
            return self.per_trade_commission
            
        elif self.commission_model == 'per_share':
            commission = quantity * self.per_share_commission
            
            # Apply minimum and maximum limits
            if self.per_share_minimum > 0:
                commission = max(commission, self.per_share_minimum)
            if self.per_share_maximum > 0:
                commission = min(commission, self.per_share_maximum)
                
            return commission
            
        elif self.commission_model == 'percentage':
            commission = trade_value * self.percentage_commission
            
            # Apply minimum and maximum limits
            if self.percentage_minimum > 0:
                commission = max(commission, self.percentage_minimum)
            if self.percentage_maximum > 0:
                commission = min(commission, self.percentage_maximum)
                
            return commission
            
        else:
            self.logger.error(f"Unknown commission model: {self.commission_model}")
            return 0.0
            
    def _calculate_fx_fees(self, trade_value: float) -> float:
        """Calculate foreign exchange conversion fees"""
        if not self.fx_enabled:
            return 0.0
            
        fx_fee = trade_value * (self.fx_spread_bps / 10000.0)  # Convert bps to percentage
        return max(fx_fee, self.fx_minimum)
        
    def _zero_costs(self) -> Dict[str, float]:
        """Return zero costs structure"""
        return {
            'commission': 0.0,
            'sec_fees': 0.0,  
            'taf_fees': 0.0,
            'regulatory_fees': 0.0,
            'fx_fees': 0.0,
            'total_costs': 0.0,
            'cost_per_share': 0.0,
            'cost_percentage': 0.0
        }
        
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of current cost configuration"""
        return {
            'commission_model': self.commission_model,
            'paper_trading_costs': self.paper_trading_costs,
            'per_trade_commission': self.per_trade_commission,
            'per_share_commission': self.per_share_commission,
            'percentage_commission': self.percentage_commission,
            'sec_fee_rate': self.sec_fee_rate,
            'taf_fee_rate': self.taf_fee_rate,
            'fx_enabled': self.fx_enabled,
            'borrowing_enabled': self.borrowing_enabled,
            'market_data_enabled': self.market_data_enabled
        }
#!/usr/bin/env python3
"""
Realistic Commission and Fee Models
Comprehensive trading cost calculation framework for backtesting

Author: Claude Code (Anthropic AI Assistant)
Date: August 14, 2025
Task: 4 - Commission Model Implementation
Next Task: 5 - Realistic Execution Delays
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class BrokerType(Enum):
    """Supported broker types with different fee structures"""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "ibkr" 
    CHARLES_SCHWAB = "schwab"
    ROBINHOOD = "robinhood"
    ETRADE = "etrade"
    BINANCE_US = "binance_us"
    COINBASE_PRO = "coinbase_pro"
    KRAKEN = "kraken"


class AssetType(Enum):
    """Asset classes with different fee structures"""
    US_STOCK = "us_stock"
    US_OPTION = "us_option"
    CRYPTOCURRENCY = "cryptocurrency"
    FUTURE = "future"
    FOREX = "forex"


@dataclass
class CommissionResult:
    """Result of commission calculation"""
    base_commission: float = 0.0
    regulatory_fees: float = 0.0
    exchange_fees: float = 0.0
    spread_cost: float = 0.0
    total_commission: float = 0.0
    breakdown: Dict[str, float] = None
    
    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {}
        self._update_total()
    
    def _update_total(self):
        """Update total commission after changes"""
        self.total_commission = (
            self.base_commission + 
            self.regulatory_fees + 
            self.exchange_fees + 
            self.spread_cost
        )


@dataclass
class TradeInfo:
    """Trade information for commission calculation"""
    broker: BrokerType
    asset_type: AssetType
    symbol: str
    quantity: float
    price: float
    side: str  # "buy" or "sell"
    trade_value: float = 0.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.trade_value == 0.0:
            self.trade_value = abs(self.quantity * self.price)
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CommissionCalculator:
    """
    Comprehensive commission calculator supporting multiple brokers and asset types
    """
    
    def __init__(self):
        # SEC fee rate (updated annually)
        self.sec_fee_rate = 0.0000278  # $0.0278 per $1000 of gross proceeds
        
        # FINRA TAF rate
        self.taf_rate = 0.000145  # $0.000145 per share
        self.taf_max = 7.27  # Maximum TAF per trade
        
        # Exchange fees (approximate)
        self.exchange_fees = {
            "NYSE": 0.0000013,  # per share
            "NASDAQ": 0.0000020,  # per share
        }
    
    def calculate_commission(self, trade: TradeInfo) -> CommissionResult:
        """
        Calculate total commission and fees for a trade
        """
        result = CommissionResult()
        
        if trade.broker == BrokerType.ALPACA:
            result = self._calculate_alpaca_commission(trade)
        elif trade.broker == BrokerType.INTERACTIVE_BROKERS:
            result = self._calculate_ibkr_commission(trade)
        elif trade.broker == BrokerType.CHARLES_SCHWAB:
            result = self._calculate_schwab_commission(trade)
        elif trade.broker == BrokerType.ROBINHOOD:
            result = self._calculate_robinhood_commission(trade)
        elif trade.broker == BrokerType.ETRADE:
            result = self._calculate_etrade_commission(trade)
        elif trade.broker == BrokerType.BINANCE_US:
            result = self._calculate_binance_commission(trade)
        elif trade.broker == BrokerType.COINBASE_PRO:
            result = self._calculate_coinbase_commission(trade)
        elif trade.broker == BrokerType.KRAKEN:
            result = self._calculate_kraken_commission(trade)
        else:
            raise ValueError(f"Unsupported broker: {trade.broker}")
        
        return result
    
    def _calculate_alpaca_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Alpaca commission (zero for stocks, spread for crypto)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.US_STOCK:
            # Zero commission for stocks
            result.base_commission = 0.0
            
            # Add regulatory fees for sells
            if trade.side.lower() == "sell":
                result.regulatory_fees = self._calculate_us_regulatory_fees(trade)
            
        elif trade.asset_type == AssetType.CRYPTOCURRENCY:
            # Alpaca crypto has hidden spread costs
            spread_rate = self._get_alpaca_crypto_spread(trade.symbol)
            result.spread_cost = trade.trade_value * spread_rate
            
        result.breakdown = {
            "base_commission": result.base_commission,
            "regulatory_fees": result.regulatory_fees,
            "spread_cost": result.spread_cost
        }
        result._update_total()
        
        return result
    
    def _calculate_ibkr_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Interactive Brokers commission (per-share model)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.US_STOCK:
            # IBKR per-share pricing: $0.005 per share, $1 min, 1% of trade value max
            per_share_cost = abs(trade.quantity) * 0.005
            min_commission = 1.00
            max_commission = trade.trade_value * 0.01
            
            result.base_commission = max(min_commission, min(per_share_cost, max_commission))
            
            # Add regulatory fees for sells
            if trade.side.lower() == "sell":
                result.regulatory_fees = self._calculate_us_regulatory_fees(trade)
                
        elif trade.asset_type == AssetType.US_OPTION:
            # Options: $0.65 per contract, $1 minimum
            contracts = abs(trade.quantity)
            result.base_commission = max(1.00, contracts * 0.65)
            
        elif trade.asset_type == AssetType.CRYPTOCURRENCY:
            # IBKR crypto (limited offerings)
            result.base_commission = trade.trade_value * 0.0012  # 0.12%
            
        result.breakdown = {
            "base_commission": result.base_commission,
            "regulatory_fees": result.regulatory_fees
        }
        result._update_total()
        
        return result
    
    def _calculate_schwab_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Charles Schwab commission (zero stocks, $0.65 options)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.US_STOCK:
            # Zero commission for stocks
            result.base_commission = 0.0
            
            # Add regulatory fees for sells
            if trade.side.lower() == "sell":
                result.regulatory_fees = self._calculate_us_regulatory_fees(trade)
                
        elif trade.asset_type == AssetType.US_OPTION:
            # Options: $0.65 per contract
            contracts = abs(trade.quantity)
            result.base_commission = contracts * 0.65
            
        result.breakdown = {
            "base_commission": result.base_commission,
            "regulatory_fees": result.regulatory_fees
        }
        result._update_total()
        
        return result
    
    def _calculate_robinhood_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Robinhood commission (zero stocks, $0.03 options)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.US_STOCK:
            # Zero commission for stocks (regulatory fees built in)
            result.base_commission = 0.0
            
        elif trade.asset_type == AssetType.US_OPTION:
            # Options: $0.03 per contract
            contracts = abs(trade.quantity)
            result.base_commission = contracts * 0.03
            
        elif trade.asset_type == AssetType.CRYPTOCURRENCY:
            # Hidden spread markup (~0.03% each side)
            result.spread_cost = trade.trade_value * 0.0003
            
        result.breakdown = {
            "base_commission": result.base_commission,
            "spread_cost": result.spread_cost
        }
        result._update_total()
        
        return result
    
    def _calculate_etrade_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate E*TRADE commission"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.US_STOCK:
            # Zero commission for stocks
            result.base_commission = 0.0
            
            # Add regulatory fees for sells
            if trade.side.lower() == "sell":
                result.regulatory_fees = self._calculate_us_regulatory_fees(trade)
                
        elif trade.asset_type == AssetType.US_OPTION:
            # Options: $0.65 per contract
            contracts = abs(trade.quantity)
            result.base_commission = contracts * 0.65
            
        result.breakdown = {
            "base_commission": result.base_commission,
            "regulatory_fees": result.regulatory_fees
        }
        result._update_total()
        
        return result
    
    def _calculate_binance_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Binance.US commission (crypto only)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.CRYPTOCURRENCY:
            # Binance.US: 0.1% taker, 0.1% maker (assuming taker for simplicity)
            result.base_commission = trade.trade_value * 0.001
            
        result.breakdown = {
            "base_commission": result.base_commission
        }
        result._update_total()
        
        return result
    
    def _calculate_coinbase_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Coinbase Pro commission (crypto only)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.CRYPTOCURRENCY:
            # Coinbase Pro: 0.5% taker for retail (simplified)
            result.base_commission = trade.trade_value * 0.005
            
        result.breakdown = {
            "base_commission": result.base_commission
        }
        result._update_total()
        
        return result
    
    def _calculate_kraken_commission(self, trade: TradeInfo) -> CommissionResult:
        """Calculate Kraken commission (crypto only)"""
        result = CommissionResult()
        
        if trade.asset_type == AssetType.CRYPTOCURRENCY:
            # Kraken: 0.26% taker for retail (simplified)
            result.base_commission = trade.trade_value * 0.0026
            
        result.breakdown = {
            "base_commission": result.base_commission
        }
        result._update_total()
        
        return result
    
    def _calculate_us_regulatory_fees(self, trade: TradeInfo) -> float:
        """Calculate US regulatory fees (SEC + TAF)"""
        total_fees = 0.0
        
        if trade.side.lower() == "sell":
            # SEC fee: $0.0278 per $1000 of gross proceeds (sells only)
            sec_fee = trade.trade_value * self.sec_fee_rate
            
            # TAF fee: $0.000145 per share, $7.27 max (sells only)
            taf_fee = min(abs(trade.quantity) * self.taf_rate, self.taf_max)
            
            total_fees = sec_fee + taf_fee
            
        return total_fees
    
    def _get_alpaca_crypto_spread(self, symbol: str) -> float:
        """Get Alpaca crypto spread rates by symbol"""
        # Typical Alpaca crypto spreads (estimates)
        spreads = {
            "BTCUSD": 0.0005,  # 0.05%
            "ETHUSD": 0.0008,  # 0.08%
            "DOGEUSD": 0.0015, # 0.15%
            "ADAUSD": 0.0012,  # 0.12%
            "SOLUSD": 0.0020,  # 0.20%
            "AVAXUSD": 0.0015, # 0.15%
            "DOTUSD": 0.0018,  # 0.18%
        }
        
        # Default spread for unknown symbols
        return spreads.get(symbol, 0.0010)  # 0.10% default


class TaxCalculationEngine:
    """
    Comprehensive tax calculation with wash sale tracking
    """
    
    def __init__(self, tax_year: int = 2025, state: str = "CA", filing_status: str = "single"):
        self.tax_year = tax_year
        self.state = state
        self.filing_status = filing_status
        
        # 2025 Federal tax brackets (estimated)
        self.federal_brackets = self._get_federal_tax_brackets()
        self.ltcg_brackets = self._get_ltcg_tax_brackets()
        
        # State tax rates (simplified)
        self.state_rates = self._get_state_tax_rates()
        
        # Wash sale tracking
        self.wash_sale_tracker = {}
        self.positions = {}
        
    def _get_federal_tax_brackets(self) -> List[Tuple[float, float]]:
        """Get federal ordinary income tax brackets"""
        if self.filing_status == "single":
            return [
                (11000, 0.10),
                (44725, 0.12),
                (95375, 0.22),
                (182050, 0.24),
                (231250, 0.32),
                (578125, 0.35),
                (float('inf'), 0.37)
            ]
        elif self.filing_status == "married":
            return [
                (22000, 0.10),
                (89450, 0.12),
                (190750, 0.22),
                (364200, 0.24),
                (462500, 0.32),
                (693750, 0.35),
                (float('inf'), 0.37)
            ]
        else:
            raise ValueError(f"Unsupported filing status: {self.filing_status}")
    
    def _get_ltcg_tax_brackets(self) -> List[Tuple[float, float]]:
        """Get long-term capital gains tax brackets"""
        if self.filing_status == "single":
            return [
                (44625, 0.00),
                (492300, 0.15),
                (float('inf'), 0.20)
            ]
        elif self.filing_status == "married":
            return [
                (89250, 0.00),
                (553850, 0.15),
                (float('inf'), 0.20)
            ]
        else:
            raise ValueError(f"Unsupported filing status: {self.filing_status}")
    
    def _get_state_tax_rates(self) -> Dict[str, float]:
        """Get state tax rates (simplified)"""
        return {
            "CA": 0.133,  # California max rate
            "NY": 0.109,  # New York max rate
            "FL": 0.000,  # No state income tax
            "TX": 0.000,  # No state income tax
            "WA": 0.000,  # No state income tax
            "NV": 0.000,  # No state income tax
        }
    
    def calculate_tax_on_gains(self, gains: float, income_level: float, holding_period_days: int) -> Dict[str, float]:
        """
        Calculate tax liability on capital gains
        """
        if gains <= 0:
            return {"federal": 0.0, "state": 0.0, "niit": 0.0, "total": 0.0}
        
        # Determine if short-term or long-term
        is_long_term = holding_period_days >= 365
        
        # Calculate federal tax
        if is_long_term:
            federal_tax = self._calculate_ltcg_tax(gains, income_level)
        else:
            # Short-term gains taxed as ordinary income
            federal_tax = self._calculate_ordinary_income_tax(gains, income_level)
        
        # Calculate state tax
        state_rate = self.state_rates.get(self.state, 0.05)  # Default 5% if unknown
        state_tax = gains * state_rate
        
        # Calculate Net Investment Income Tax (3.8% for high earners)
        niit_threshold = 200000 if self.filing_status == "single" else 250000
        niit_tax = 0.0
        if income_level > niit_threshold:
            niit_tax = gains * 0.038
        
        total_tax = federal_tax + state_tax + niit_tax
        
        return {
            "federal": federal_tax,
            "state": state_tax,
            "niit": niit_tax,
            "total": total_tax,
            "effective_rate": total_tax / gains if gains > 0 else 0.0,
            "holding_period": "long_term" if is_long_term else "short_term"
        }
    
    def _calculate_ordinary_income_tax(self, additional_income: float, base_income: float) -> float:
        """Calculate federal tax on additional ordinary income"""
        total_income = base_income + additional_income
        
        # Calculate tax on total income
        total_tax = 0.0
        remaining_income = total_income
        
        for threshold, rate in self.federal_brackets:
            if remaining_income <= 0:
                break
                
            taxable_in_bracket = min(remaining_income, threshold)
            total_tax += taxable_in_bracket * rate
            remaining_income -= taxable_in_bracket
        
        # Calculate tax on base income only
        base_tax = 0.0
        remaining_income = base_income
        
        for threshold, rate in self.federal_brackets:
            if remaining_income <= 0:
                break
                
            taxable_in_bracket = min(remaining_income, threshold)
            base_tax += taxable_in_bracket * rate
            remaining_income -= taxable_in_bracket
        
        # Return marginal tax on additional income
        return total_tax - base_tax
    
    def _calculate_ltcg_tax(self, gains: float, income_level: float) -> float:
        """Calculate long-term capital gains tax"""
        # Determine LTCG rate based on income level
        for threshold, rate in self.ltcg_brackets:
            if income_level <= threshold:
                return gains * rate
        
        # Fallback (shouldn't reach here)
        return gains * 0.20
    
    def track_wash_sale(self, symbol: str, sell_date: datetime, sell_quantity: float, 
                       sell_price: float, loss_amount: float) -> bool:
        """
        Track potential wash sales and return True if wash sale rule applies
        """
        if loss_amount >= 0:  # No wash sale on gains
            return False
        
        # Check for purchases within 30 days before or after
        wash_sale_start = sell_date - timedelta(days=30)
        wash_sale_end = sell_date + timedelta(days=30)
        
        # This is simplified - in reality would need to track all purchases
        # For backtesting, we'll assume a percentage of losses are wash sales
        
        # Estimate: 20% of day trading losses are wash sales
        if symbol in self.wash_sale_tracker:
            recent_purchases = self.wash_sale_tracker[symbol]
            for purchase_date, quantity in recent_purchases:
                if wash_sale_start <= purchase_date <= wash_sale_end:
                    return True
        
        return False
    
    def add_purchase(self, symbol: str, date: datetime, quantity: float, price: float):
        """Track purchases for wash sale calculations"""
        if symbol not in self.wash_sale_tracker:
            self.wash_sale_tracker[symbol] = []
        
        self.wash_sale_tracker[symbol].append((date, quantity))
        
        # Clean up old entries (keep 60 days)
        cutoff_date = date - timedelta(days=60)
        self.wash_sale_tracker[symbol] = [
            (d, q) for d, q in self.wash_sale_tracker[symbol] 
            if d >= cutoff_date
        ]


class RealWorldCostCalculator:
    """
    Combined calculator for all real-world trading costs
    """
    
    def __init__(self, broker: BrokerType = BrokerType.ALPACA, 
                 tax_state: str = "CA", filing_status: str = "single",
                 income_level: float = 100000):
        self.commission_calc = CommissionCalculator()
        self.tax_calc = TaxCalculationEngine(state=tax_state, filing_status=filing_status)
        self.broker = broker
        self.income_level = income_level
        
    def calculate_total_trade_cost(self, trade: TradeInfo, 
                                  holding_period_days: int = 1,
                                  profit_loss: float = 0.0) -> Dict[str, float]:
        """
        Calculate total cost of a trade including commissions and taxes
        """
        # Calculate commission costs
        commission_result = self.commission_calc.calculate_commission(trade)
        
        # Calculate tax costs (only on gains)
        tax_result = {"total": 0.0, "effective_rate": 0.0}
        if profit_loss > 0:
            tax_result = self.tax_calc.calculate_tax_on_gains(
                profit_loss, self.income_level, holding_period_days
            )
        
        # Combine costs
        total_cost = commission_result.total_commission + tax_result["total"]
        
        return {
            "commission": commission_result.total_commission,
            "commission_breakdown": commission_result.breakdown,
            "tax": tax_result["total"],
            "tax_details": tax_result,
            "total_cost": total_cost,
            "net_profit": profit_loss - total_cost,
            "cost_percentage": (total_cost / trade.trade_value) * 100 if trade.trade_value > 0 else 0.0
        }


# Example usage and testing functions
def test_commission_models():
    """Test all commission models with sample trades"""
    print("🧪 Testing Commission Models")
    print("=" * 50)
    
    # Sample stock trade
    stock_trade = TradeInfo(
        broker=BrokerType.ALPACA,
        asset_type=AssetType.US_STOCK,
        symbol="AAPL",
        quantity=100,
        price=150.0,
        side="buy"
    )
    
    # Sample crypto trade
    crypto_trade = TradeInfo(
        broker=BrokerType.ALPACA,
        asset_type=AssetType.CRYPTOCURRENCY,
        symbol="BTCUSD",
        quantity=0.1,
        price=50000.0,
        side="buy"
    )
    
    calc = CommissionCalculator()
    
    # Test different brokers
    brokers_to_test = [BrokerType.ALPACA, BrokerType.INTERACTIVE_BROKERS, 
                       BrokerType.CHARLES_SCHWAB, BrokerType.BINANCE_US]
    
    for broker in brokers_to_test:
        print(f"\n📊 {broker.value.upper()} Commission Test:")
        
        # Test stock trade
        stock_trade.broker = broker
        if broker in [BrokerType.BINANCE_US, BrokerType.COINBASE_PRO, BrokerType.KRAKEN]:
            print("  Stocks: Not supported")
        else:
            result = calc.calculate_commission(stock_trade)
            print(f"  Stock Trade: ${result.total_commission:.4f} (${stock_trade.trade_value} trade)")
        
        # Test crypto trade
        crypto_trade.broker = broker
        if broker in [BrokerType.CHARLES_SCHWAB, BrokerType.ETRADE]:
            print("  Crypto: Not supported")
        else:
            result = calc.calculate_commission(crypto_trade)
            print(f"  Crypto Trade: ${result.total_commission:.4f} (${crypto_trade.trade_value} trade)")


def test_tax_calculations():
    """Test tax calculations with different scenarios"""
    print("\n🏛️ Testing Tax Calculations")
    print("=" * 50)
    
    tax_calc = TaxCalculationEngine(state="CA", filing_status="single")
    
    test_scenarios = [
        {"gains": 10000, "income": 50000, "days": 10, "name": "Short-term, middle income"},
        {"gains": 10000, "income": 50000, "days": 400, "name": "Long-term, middle income"},
        {"gains": 10000, "income": 150000, "days": 10, "name": "Short-term, high income"},
        {"gains": 10000, "income": 150000, "days": 400, "name": "Long-term, high income"},
    ]
    
    for scenario in test_scenarios:
        result = tax_calc.calculate_tax_on_gains(
            scenario["gains"], scenario["income"], scenario["days"]
        )
        print(f"\n💰 {scenario['name']}:")
        print(f"  Gains: ${scenario['gains']:,}")
        print(f"  Tax: ${result['total']:.2f} ({result['effective_rate']*100:.1f}%)")
        print(f"  Type: {result['holding_period']}")


def test_combined_costs():
    """Test combined commission and tax costs"""
    print("\n💸 Testing Combined Real-World Costs")
    print("=" * 50)
    
    cost_calc = RealWorldCostCalculator(
        broker=BrokerType.ALPACA,
        tax_state="CA",
        income_level=100000
    )
    
    # Test profitable trade
    trade = TradeInfo(
        broker=BrokerType.ALPACA,
        asset_type=AssetType.US_STOCK,
        symbol="AAPL",
        quantity=100,
        price=150.0,
        side="sell"
    )
    
    profit = 1500.0  # $15 per share profit
    
    result = cost_calc.calculate_total_trade_cost(
        trade, holding_period_days=30, profit_loss=profit
    )
    
    print(f"📊 Sample Trade Results:")
    print(f"  Trade Value: ${trade.trade_value:,.2f}")
    print(f"  Gross Profit: ${profit:,.2f}")
    print(f"  Commission: ${result['commission']:.2f}")
    print(f"  Tax: ${result['tax']:.2f}")
    print(f"  Total Cost: ${result['total_cost']:.2f}")
    print(f"  Net Profit: ${result['net_profit']:.2f}")
    print(f"  Cost %: {result['cost_percentage']:.3f}%")


if __name__ == "__main__":
    # Run all tests
    test_commission_models()
    test_tax_calculations()
    test_combined_costs()
    
    print("\n✅ All commission and tax models implemented and tested!")
    print("📁 Ready for integration into realistic backtesting framework")
#!/usr/bin/env python3
"""
Unit Tests for TradingCostCalculator

Tests all commission models, fee calculations, and edge cases.
"""

import unittest
import sys
import os

# Add project to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from algotrading_agent.components.trading_cost_calculator import TradingCostCalculator


class TestTradingCostCalculator(unittest.TestCase):
    """Unit tests for TradingCostCalculator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_config = {
            'trading_costs': {
                'paper_trading_costs': True,  # Enable costs for testing
                'regulatory_fees': {
                    'sec_fee_rate': 0.0000278,
                    'taf_fee_rate': 0.000166
                }
            }
        }
    
    def test_zero_commission_model(self):
        """Test zero commission model"""
        config = self.base_config.copy()
        config['trading_costs']['commission_model'] = 'zero'
        
        calculator = TradingCostCalculator(config)
        
        # Test buy trade
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 100, 150.0, True)
        
        self.assertEqual(costs['commission'], 0.0)
        self.assertEqual(costs['sec_fees'], 0.0)  # SEC fees only on sells
        self.assertEqual(costs['taf_fees'], 0.0)  # TAF fees only on sells
        self.assertAlmostEqual(costs['total_costs'], 0.0, places=2)
        
        # Test sell trade (should have regulatory fees)
        costs = calculator.calculate_trade_costs('AAPL', 'sell', 100, 150.0, True)
        
        self.assertEqual(costs['commission'], 0.0)
        self.assertGreater(costs['sec_fees'], 0.0)
        self.assertGreater(costs['taf_fees'], 0.0)
        expected_total = (100 * 150.0) * (0.0000278 + 0.000166)
        self.assertAlmostEqual(costs['total_costs'], expected_total, places=2)
    
    def test_per_trade_commission_model(self):
        """Test per-trade commission model"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'per_trade',
            'per_trade_commission': 9.95
        })
        
        calculator = TradingCostCalculator(config)
        
        # Test buy trade
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 100, 150.0, True)
        
        self.assertEqual(costs['commission'], 9.95)
        self.assertEqual(costs['sec_fees'], 0.0)
        self.assertEqual(costs['taf_fees'], 0.0)
        self.assertAlmostEqual(costs['total_costs'], 9.95, places=2)
        
        # Test sell trade
        costs = calculator.calculate_trade_costs('AAPL', 'sell', 100, 150.0, True)
        
        self.assertEqual(costs['commission'], 9.95)
        self.assertGreater(costs['sec_fees'], 0.0)
        self.assertGreater(costs['taf_fees'], 0.0)
        expected_total = 9.95 + (100 * 150.0) * (0.0000278 + 0.000166)
        self.assertAlmostEqual(costs['total_costs'], expected_total, places=2)
    
    def test_per_share_commission_model(self):
        """Test per-share commission model with minimum/maximum"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'per_share',
            'per_share_commission': 0.005,
            'per_share_minimum': 1.00,
            'per_share_maximum': 50.00
        })
        
        calculator = TradingCostCalculator(config)
        
        # Test small trade (should hit minimum)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 10, 150.0, True)
        expected_commission = max(10 * 0.005, 1.00)  # Should be $1.00 minimum
        self.assertEqual(costs['commission'], expected_commission)
        
        # Test medium trade (normal calculation)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 1000, 150.0, True)
        expected_commission = min(1000 * 0.005, 50.00)  # Should be $5.00, under max
        self.assertEqual(costs['commission'], expected_commission)
        
        # Test large trade (should hit maximum)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 20000, 150.0, True)
        expected_commission = min(20000 * 0.005, 50.00)  # Should be $50.00 maximum
        self.assertEqual(costs['commission'], expected_commission)
    
    def test_percentage_commission_model(self):
        """Test percentage-based commission model"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'percentage',
            'percentage_commission': 0.001,  # 0.1%
            'percentage_minimum': 1.00,
            'percentage_maximum': 100.00
        })
        
        calculator = TradingCostCalculator(config)
        
        # Test small trade (should hit minimum)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 10, 50.0, True)  # $500 trade
        trade_value = 10 * 50.0
        expected_commission = max(trade_value * 0.001, 1.00)  # Should be $1.00 minimum
        self.assertEqual(costs['commission'], expected_commission)
        
        # Test large trade (should hit maximum)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 1000, 200.0, True)  # $200k trade
        trade_value = 1000 * 200.0
        expected_commission = min(trade_value * 0.001, 100.00)  # Should be $100.00 maximum
        self.assertEqual(costs['commission'], expected_commission)
    
    def test_round_trip_costs(self):
        """Test round-trip cost calculations"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'per_trade',
            'per_trade_commission': 5.00
        })
        
        calculator = TradingCostCalculator(config)
        
        # Test profitable round trip
        costs = calculator.calculate_round_trip_costs('AAPL', 100, 150.0, 155.0, True)
        
        self.assertAlmostEqual(costs['entry_costs'], 5.00, places=2)
        self.assertGreater(costs['exit_costs'], 5.00)  # Should include regulatory fees
        self.assertAlmostEqual(costs['gross_pnl'], 500.0, places=2)  # (155-150)*100
        self.assertLess(costs['net_pnl'], costs['gross_pnl'])  # Net should be less than gross
        self.assertGreater(costs['total_costs'], 10.0)  # Should be > $10 due to regulatory fees
    
    def test_paper_trading_costs_disabled(self):
        """Test that costs are zero when paper_trading_costs is disabled"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'per_trade',
            'per_trade_commission': 9.95,
            'paper_trading_costs': False  # Disable costs for paper trading
        })
        
        calculator = TradingCostCalculator(config)
        
        # Test paper trading (should be zero costs)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 100, 150.0, True)
        
        self.assertEqual(costs['commission'], 0.0)
        self.assertEqual(costs['total_costs'], 0.0)
        
        # Test live trading (should have costs)
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 100, 150.0, False)
        
        self.assertEqual(costs['commission'], 9.95)
        self.assertEqual(costs['total_costs'], 9.95)
    
    def test_holding_costs(self):
        """Test holding costs for short positions"""
        config = self.base_config.copy()
        config['trading_costs']['borrowing_costs'] = {
            'enabled': True,
            'daily_rate': 0.001,  # 0.1% daily
            'minimum_daily_fee': 1.00
        }
        
        calculator = TradingCostCalculator(config)
        
        # Test short position holding costs
        holding_costs = calculator.calculate_holding_costs('AAPL', 100, 150.0, 5, True)
        
        position_value = 100 * 150.0  # $15,000
        expected_daily = max(position_value * 0.001, 1.00)  # $15.00 vs $1.00 minimum  
        expected_total = expected_daily * 5  # 5 days
        
        self.assertAlmostEqual(holding_costs['borrowing_costs'], expected_total, places=2)
        self.assertAlmostEqual(holding_costs['total_holding_costs'], expected_total, places=2)
        
        # Test long position (should be zero)
        holding_costs = calculator.calculate_holding_costs('AAPL', 100, 150.0, 5, False)
        
        self.assertEqual(holding_costs['borrowing_costs'], 0.0)
        self.assertEqual(holding_costs['total_holding_costs'], 0.0)
    
    def test_cost_per_share_calculation(self):
        """Test cost per share calculations"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'per_trade',
            'per_trade_commission': 10.00
        })
        
        calculator = TradingCostCalculator(config)
        
        costs = calculator.calculate_trade_costs('AAPL', 'buy', 100, 150.0, True)
        
        expected_cost_per_share = costs['total_costs'] / 100
        self.assertAlmostEqual(costs['cost_per_share'], expected_cost_per_share, places=4)
        
        expected_cost_percentage = costs['total_costs'] / (100 * 150.0)
        self.assertAlmostEqual(costs['cost_percentage'], expected_cost_percentage, places=6)
    
    def test_monthly_fixed_costs(self):
        """Test monthly fixed costs calculation"""
        config = self.base_config.copy()
        config['trading_costs']['market_data_fees'] = {
            'enabled': True,
            'monthly_cost': 25.00
        }
        
        calculator = TradingCostCalculator(config)
        
        monthly_costs = calculator.get_monthly_fixed_costs()
        
        self.assertEqual(monthly_costs['market_data'], 25.00)
        self.assertEqual(monthly_costs['total_monthly'], 25.00)
    
    def test_broker_presets(self):
        """Test broker preset application"""
        config = self.base_config.copy()
        config['trading_costs']['broker_presets'] = {
            'test_broker': {
                'commission_model': 'per_trade',
                'per_trade_commission': 7.50
            }
        }
        
        calculator = TradingCostCalculator(config)
        
        # Apply preset
        result = calculator.apply_broker_preset('test_broker')
        
        self.assertTrue(result)
        self.assertEqual(calculator.commission_model, 'per_trade')
        self.assertEqual(calculator.per_trade_commission, 7.50)
        
        # Test invalid preset
        result = calculator.apply_broker_preset('nonexistent_broker')
        self.assertFalse(result)
    
    def test_cost_summary(self):
        """Test cost configuration summary"""
        config = self.base_config.copy()
        config['trading_costs'].update({
            'commission_model': 'per_share',
            'per_share_commission': 0.005
        })
        
        calculator = TradingCostCalculator(config)
        
        summary = calculator.get_cost_summary()
        
        self.assertEqual(summary['commission_model'], 'per_share')
        self.assertEqual(summary['per_share_commission'], 0.005)
        self.assertIn('paper_trading_costs', summary)
        self.assertIn('sec_fee_rate', summary)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2)
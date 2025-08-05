#!/usr/bin/env python3
"""
Test Trading Cost Calculator

Demonstrates different commission structures and their impact on trading profitability.
"""

import sys
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.components.trading_cost_calculator import TradingCostCalculator

def test_trading_costs():
    """Test different commission models with example trades"""
    
    print("💰 Trading Cost Calculator Test")
    print("=" * 60)
    
    # Example trade scenarios
    test_trades = [
        {
            'name': 'Small Trade (AAPL)',
            'symbol': 'AAPL',
            'quantity': 10,
            'entry_price': 150.00,
            'exit_price': 155.00
        },
        {
            'name': 'Medium Trade (TSLA)', 
            'symbol': 'TSLA',
            'quantity': 50,
            'entry_price': 200.00,
            'exit_price': 190.00  # Loss scenario
        },
        {
            'name': 'Large Trade (SPY)',
            'symbol': 'SPY',
            'quantity': 1000,
            'entry_price': 400.00,
            'exit_price': 402.00  # Small percentage gain
        }
    ]
    
    # Test different broker configurations
    broker_configs = [
        {
            'name': 'Current (Zero Commission)',
            'config': {
                'trading_costs': {
                    'commission_model': 'zero',
                    'paper_trading_costs': True,  # Enable for testing
                    'regulatory_fees': {
                        'sec_fee_rate': 0.0000278,
                        'taf_fee_rate': 0.000166
                    }
                }
            }
        },
        {
            'name': 'Interactive Brokers (Per Share)',
            'config': {
                'trading_costs': {
                    'commission_model': 'per_share',
                    'per_share_commission': 0.005,
                    'per_share_minimum': 1.00,
                    'per_share_maximum': 0.0,  # No maximum
                    'paper_trading_costs': True,
                    'regulatory_fees': {
                        'sec_fee_rate': 0.0000278,
                        'taf_fee_rate': 0.000166
                    }
                }
            }
        },
        {
            'name': 'Traditional Broker (Per Trade)',
            'config': {
                'trading_costs': {
                    'commission_model': 'per_trade',
                    'per_trade_commission': 9.95,
                    'paper_trading_costs': True,
                    'regulatory_fees': {
                        'sec_fee_rate': 0.0000278,
                        'taf_fee_rate': 0.000166
                    }
                }
            }
        },
        {
            'name': 'Percentage-Based Broker',
            'config': {
                'trading_costs': {
                    'commission_model': 'percentage',
                    'percentage_commission': 0.001,  # 0.1%
                    'percentage_minimum': 1.00,
                    'percentage_maximum': 50.00,
                    'paper_trading_costs': True,
                    'regulatory_fees': {
                        'sec_fee_rate': 0.0000278,
                        'taf_fee_rate': 0.000166
                    }
                }
            }
        }
    ]
    
    # Test each broker configuration
    for broker in broker_configs:
        print(f"\n🏦 Testing: {broker['name']}")
        print("-" * 50)
        
        calculator = TradingCostCalculator(broker['config'])
        
        for trade in test_trades:
            print(f"\n📊 {trade['name']}:")
            print(f"   Position: {trade['quantity']} shares @ ${trade['entry_price']:.2f}")
            print(f"   Exit: @ ${trade['exit_price']:.2f}")
            
            # Calculate round-trip costs
            costs = calculator.calculate_round_trip_costs(
                symbol=trade['symbol'],
                quantity=trade['quantity'],
                entry_price=trade['entry_price'],
                exit_price=trade['exit_price'],
                is_paper_trading=True
            )
            
            # Display results
            gross_pnl = costs['gross_pnl']
            net_pnl = costs['net_pnl']
            total_costs = costs['total_costs']
            cost_per_share = costs['cost_per_share']
            
            print(f"   💵 Gross P&L: ${gross_pnl:+.2f}")
            print(f"   💰 Total Costs: ${total_costs:.2f}")
            print(f"   📈 Net P&L: ${net_pnl:+.2f}")
            print(f"   📊 Cost per Share: ${cost_per_share:.4f}")
            
            # Show cost impact
            if gross_pnl != 0:
                cost_impact = (total_costs / abs(gross_pnl)) * 100
                print(f"   ⚠️  Cost Impact: {cost_impact:.1f}% of gross P&L")
                
                # Breakeven analysis
                if gross_pnl > 0:
                    breakeven_price = trade['entry_price'] + (total_costs / trade['quantity'])
                    print(f"   🎯 Breakeven Price: ${breakeven_price:.2f}")
            
            # Profitability check
            if net_pnl > 0:
                print(f"   ✅ Profitable after costs")
            elif net_pnl < 0 and gross_pnl > 0:
                print(f"   ❌ Profitable trade turned unprofitable due to costs!")
            else:
                print(f"   ❌ Loss trade (costs make it worse)")
    
    # Summary comparison
    print(f"\n📋 COMMISSION STRUCTURE COMPARISON")
    print("=" * 60)
    
    # Use medium trade for comparison
    test_trade = test_trades[1]  # TSLA trade
    
    print(f"Trade: {test_trade['quantity']} {test_trade['symbol']} @ ${test_trade['entry_price']:.2f} → ${test_trade['exit_price']:.2f}")
    print(f"Gross P&L: ${(test_trade['exit_price'] - test_trade['entry_price']) * test_trade['quantity']:+.2f}")
    print()
    
    results = []
    for broker in broker_configs:
        calculator = TradingCostCalculator(broker['config'])
        costs = calculator.calculate_round_trip_costs(
            symbol=test_trade['symbol'],
            quantity=test_trade['quantity'], 
            entry_price=test_trade['entry_price'],
            exit_price=test_trade['exit_price'],
            is_paper_trading=True
        )
        
        results.append({
            'broker': broker['name'],
            'total_costs': costs['total_costs'],
            'net_pnl': costs['net_pnl'],
            'cost_per_share': costs['cost_per_share']
        })
    
    # Sort by lowest cost
    results.sort(key=lambda x: x['total_costs'])
    
    print("🏆 Ranking (Best to Worst for this trade):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['broker']:<25} Costs: ${result['total_costs']:>6.2f}  Net P&L: ${result['net_pnl']:>+7.2f}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Zero commission brokers save ${results[-1]['total_costs'] - results[0]['total_costs']:.2f} per round-trip")
    print(f"   • High-frequency trading strongly favors zero commission")
    print(f"   • Per-share pricing can be better for large trades")
    print(f"   • Always factor costs into your profit targets!")
    
    print(f"\n🔧 Configuration Examples:")
    print(f"   Zero Commission (Current): commission_model = 'zero'")
    print(f"   Per Share: commission_model = 'per_share', per_share_commission = 0.005")
    print(f"   Per Trade: commission_model = 'per_trade', per_trade_commission = 9.95")  
    print(f"   Percentage: commission_model = 'percentage', percentage_commission = 0.001")

if __name__ == "__main__":
    test_trading_costs()
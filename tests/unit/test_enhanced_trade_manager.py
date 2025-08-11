#!/usr/bin/env python3
"""
Test Enhanced Trade Manager bracket order creation
Verify that new trades create complete bracket orders with both stop-loss and take-profit
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.enhanced_trade_manager import EnhancedTradeManager
from algotrading_agent.components.decision_engine import TradingPair

async def test_enhanced_trade_manager():
    print('🧪 TESTING: Enhanced Trade Manager Bracket Order Creation')
    print('=' * 70)
    
    try:
        config = get_config()
        alpaca_client = AlpacaClient(config.get_alpaca_config())
        
        # Initialize Enhanced Trade Manager
        trade_manager = EnhancedTradeManager(config.get_component_config('trade_manager'))
        trade_manager.alpaca_client = alpaca_client
        trade_manager.start()
        
        # Create a test trading pair
        current_price = await alpaca_client.get_current_price("SPY")
        if not current_price:
            print("❌ Could not get current price for SPY - using mock price")
            current_price = 450.0
            
        print(f"📈 SPY Current Price: ${current_price:.2f}")
        
        test_pair = TradingPair(
            symbol="SPY",
            action="buy",
            entry_price=current_price,
            quantity=1,
            stop_loss=round(current_price * 0.95, 2),  # 5% below
            take_profit=round(current_price * 1.05, 2)   # 5% above
        )
        
        # Set additional attributes after initialization
        test_pair.confidence = 0.85
        test_pair.reasoning = "Test trade to verify Enhanced Trade Manager bracket order creation"
        
        print(f"🎯 Test Trade Details:")
        print(f"   Symbol: {test_pair.symbol}")
        print(f"   Action: {test_pair.action.upper()}")
        print(f"   Quantity: {test_pair.quantity}")
        print(f"   Entry Price: ${test_pair.entry_price:.2f}")
        print(f"   Stop-Loss: ${test_pair.stop_loss:.2f}")
        print(f"   Take-Profit: ${test_pair.take_profit:.2f}")
        print(f"   Confidence: {test_pair.confidence:.2f}")
        
        # Test bracket order validation first
        print(f"\n🔍 Validating trading pair...")
        validation = await alpaca_client.validate_trading_pair(test_pair)
        
        if not validation["valid"]:
            print(f"❌ Validation failed: {validation['errors']}")
            return
        else:
            print(f"✅ Validation passed")
            if validation["warnings"]:
                print(f"⚠️  Warnings: {validation['warnings']}")
        
        # Check if markets are open
        market_open = await alpaca_client.is_market_open()
        print(f"📊 Market Status: {'OPEN' if market_open else 'CLOSED'}")
        
        if not market_open:
            print(f"⏸️  Markets are closed - this is a DRY RUN test")
            print(f"   In live trading, this would create a complete bracket order")
            print(f"   ✅ Enhanced Trade Manager is properly configured")
            return
        
        # Execute test trade (CAREFUL - this creates a real trade!)
        print(f"\n⚠️  CAUTION: About to create REAL trade in paper account")
        print(f"   This will test complete bracket order creation")
        
        # Ask for confirmation (in a real scenario)
        confirm = input("Continue with test trade? (y/N): ").lower().strip()
        if confirm != 'y':
            print(f"❌ Test cancelled by user")
            return
        
        print(f"\n🚀 Executing test trade via Enhanced Trade Manager...")
        
        # Execute through Enhanced Trade Manager
        result = await trade_manager.execute_trade(test_pair)
        
        if result["success"]:
            print(f"✅ SUCCESS: Trade executed via Enhanced Trade Manager")
            print(f"   Message: {result['message']}")
            print(f"   Data: {result['data']}")
            
            # Get the bracket status to verify complete protection
            if trade_manager.bracket_manager:
                bracket_status = trade_manager.bracket_manager.get_bracket_status()
                print(f"\n🛡️  Bracket Protection Status:")
                print(f"   Total Brackets: {bracket_status['total_brackets']}")
                print(f"   Protected Positions: {bracket_status['protected_positions']}")
                print(f"   Unprotected Positions: {bracket_status['unprotected_positions']}")
                
                if bracket_status["unprotected_positions"] == 0:
                    print(f"   ✅ ALL POSITIONS ARE PROPERLY PROTECTED!")
                else:
                    print(f"   ❌ WARNING: {bracket_status['unprotected_positions']} positions lack protection")
            
            # Verify with Alpaca directly
            print(f"\n🔍 Verifying complete bracket order with Alpaca...")
            await asyncio.sleep(2)  # Give time for orders to process
            
            orders = await alpaca_client.get_orders()
            spy_orders = [o for o in orders if o.get('symbol') == 'SPY' and o.get('status') in ['new', 'accepted', 'filled']]
            
            print(f"   Found {len(spy_orders)} recent SPY orders:")
            stop_orders = [o for o in spy_orders if 'stop' in o.get('order_type', '').lower()]
            limit_orders = [o for o in spy_orders if o.get('order_type') == 'limit']
            
            print(f"   Stop-Loss Orders: {len(stop_orders)}")
            print(f"   Take-Profit Orders: {len(limit_orders)}")
            
            if len(stop_orders) > 0 and len(limit_orders) > 0:
                print(f"   ✅ COMPLETE BRACKET ORDER CREATED!")
                print(f"   🛡️  Stop-Loss: {stop_orders[-1].get('order_id')}")
                print(f"   🎯 Take-Profit: {limit_orders[-1].get('order_id')}")
            else:
                print(f"   ⚠️  Incomplete bracket order detected")
                
        else:
            print(f"❌ FAILED: Trade execution failed")
            print(f"   Error: {result['message']}")
        
        # Cleanup
        trade_manager.stop()
        
        print(f"\n🎉 ENHANCED TRADE MANAGER TEST COMPLETE")
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_trade_manager())
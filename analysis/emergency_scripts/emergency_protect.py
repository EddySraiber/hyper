#!/usr/bin/env python3
"""
Emergency Position Protection Script
Immediately protects all unprotected positions
"""

import asyncio
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def fix_unprotected_positions():
    print("🚨 EMERGENCY PROTECTION FIX - Starting...")
    
    try:
        # Initialize components
        config = get_config()
        alpaca_client = AlpacaClient(config.get_alpaca_config())
        
        print("📡 Connected to Alpaca API")
        
        # Get all positions
        positions = await alpaca_client.get_positions()
        print(f"📊 Found {len(positions)} positions")
        
        if not positions:
            print("✅ No positions to protect")
            return
        
        protection_results = []
        
        for position in positions:
            symbol = position["symbol"]
            quantity = position["quantity"]
            market_value = position["market_value"]
            
            print(f"\n🔍 Checking {symbol}: {quantity} shares, ${market_value:.2f}")
            
            # Calculate current price
            if quantity != 0:
                current_price = abs(market_value / quantity)
            else:
                current_price = 0
            
            # Calculate protective prices
            if quantity > 0:  # Long position
                stop_loss = round(current_price * 0.95, 2)  # 5% below
                take_profit = round(current_price * 1.10, 2)  # 10% above
            else:  # Short position
                stop_loss = round(current_price * 1.05, 2)  # 5% above
                take_profit = round(current_price * 0.90, 2)  # 10% below
            
            print(f"   Current: ${current_price:.2f}")
            print(f"   Stop-Loss: ${stop_loss:.2f}")
            print(f"   Take-Profit: ${take_profit:.2f}")
            
            stop_success = False
            tp_success = False
            
            # Create stop-loss
            print(f"🛡️  Creating stop-loss for {symbol}...")
            try:
                stop_result = await alpaca_client.update_stop_loss(symbol, stop_loss)
                if stop_result["success"]:
                    print(f"   ✅ Stop-loss created: Order {stop_result['new_order_id']}")
                    stop_success = True
                else:
                    print(f"   ❌ Stop-loss failed: {stop_result.get('error')}")
            except Exception as e:
                print(f"   ❌ Stop-loss exception: {e}")
            
            # Create take-profit
            print(f"🎯 Creating take-profit for {symbol}...")
            try:
                tp_result = await alpaca_client.update_take_profit(symbol, take_profit)
                if tp_result["success"]:
                    print(f"   ✅ Take-profit created: Order {tp_result['new_order_id']}")
                    tp_success = True
                else:
                    print(f"   ❌ Take-profit failed: {tp_result.get('error')}")
            except Exception as e:
                print(f"   ❌ Take-profit exception: {e}")
            
            protection_results.append({
                "symbol": symbol,
                "stop_loss_created": stop_success,
                "take_profit_created": tp_success
            })
        
        # Summary
        print(f"\n📊 PROTECTION SUMMARY:")
        total_positions = len(protection_results)
        fully_protected = sum(1 for r in protection_results if r["stop_loss_created"] and r["take_profit_created"])
        partially_protected = sum(1 for r in protection_results if r["stop_loss_created"] or r["take_profit_created"])
        
        print(f"   Total positions: {total_positions}")
        print(f"   Fully protected: {fully_protected}")
        print(f"   Partially protected: {partially_protected - fully_protected}")
        print(f"   Still unprotected: {total_positions - partially_protected}")
        
        if fully_protected == total_positions:
            print("✅ ALL POSITIONS ARE NOW PROTECTED!")
        elif partially_protected > 0:
            print("⚠️  Some positions have partial protection")
        else:
            print("🚨 NO POSITIONS COULD BE PROTECTED - MANUAL INTERVENTION REQUIRED")
        
    except Exception as e:
        print(f"❌ EMERGENCY PROTECTION FIX FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(fix_unprotected_positions())
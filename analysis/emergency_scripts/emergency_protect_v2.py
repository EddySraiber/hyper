#!/usr/bin/env python3
"""
Emergency Position Protection Script V2
Creates new protective orders for unprotected positions
"""

import asyncio
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

async def create_protective_orders():
    print("🚨 EMERGENCY PROTECTION V2 - Creating New Protective Orders...")
    
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
            
            print(f"\n🔍 Protecting {symbol}: {quantity} shares, ${market_value:.2f}")
            
            # Calculate current price
            if quantity != 0:
                current_price = abs(market_value / quantity)
            else:
                current_price = 0
                continue
            
            # Calculate protective prices
            if quantity > 0:  # Long position
                stop_loss_price = round(current_price * 0.95, 2)  # 5% below
                take_profit_price = round(current_price * 1.10, 2)  # 10% above
                stop_side = OrderSide.SELL
                tp_side = OrderSide.SELL
            else:  # Short position
                stop_loss_price = round(current_price * 1.05, 2)  # 5% above
                take_profit_price = round(current_price * 0.90, 2)  # 10% below
                stop_side = OrderSide.BUY
                tp_side = OrderSide.BUY
            
            position_qty = abs(quantity)
            
            print(f"   Current: ${current_price:.2f}")
            print(f"   Stop-Loss: ${stop_loss_price:.2f}")
            print(f"   Take-Profit: ${take_profit_price:.2f}")
            
            stop_success = False
            tp_success = False
            
            # Create stop-loss order
            print(f"🛡️  Creating stop-loss order for {symbol}...")
            try:
                stop_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=position_qty,
                    side=stop_side,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_loss_price
                )
                
                stop_order = alpaca_client.trading_client.submit_order(stop_request)
                print(f"   ✅ Stop-loss created: Order {stop_order.id}")
                stop_success = True
                
            except Exception as e:
                print(f"   ❌ Stop-loss exception: {e}")
            
            # Create take-profit order
            print(f"🎯 Creating take-profit order for {symbol}...")
            try:
                tp_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=position_qty,
                    side=tp_side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=take_profit_price
                )
                
                tp_order = alpaca_client.trading_client.submit_order(tp_request)
                print(f"   ✅ Take-profit created: Order {tp_order.id}")
                tp_success = True
                
            except Exception as e:
                print(f"   ❌ Take-profit exception: {e}")
            
            protection_results.append({
                "symbol": symbol,
                "stop_loss_created": stop_success,
                "take_profit_created": tp_success
            })
            
            # Small delay between positions
            await asyncio.sleep(1)
        
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
            print("🚨 PROTECTION CREATION FAILED - CHECK BROKER REQUIREMENTS")
            
        print(f"\n🔍 Verification - Checking final status...")
        
        # Verify protection was created
        final_positions = await alpaca_client.get_positions()
        all_orders = await alpaca_client.get_orders()
        active_orders = [o for o in all_orders if o["status"] in ["new", "accepted", "pending_new"]]
        
        print(f"📊 Final verification: {len(final_positions)} positions, {len(active_orders)} active orders")
        
        for pos in final_positions:
            symbol_orders = [o for o in active_orders if o["symbol"] == pos["symbol"]]
            stop_orders = [o for o in symbol_orders if "stop" in o.get("order_type", "").lower()]
            limit_orders = [o for o in symbol_orders if o.get("order_type") == "limit"]
            
            protection_status = "✅ PROTECTED" if (len(stop_orders) > 0 and len(limit_orders) > 0) else "🚨 UNPROTECTED"
            print(f"   {pos['symbol']}: {len(symbol_orders)} orders ({len(stop_orders)} stop, {len(limit_orders)} limit) - {protection_status}")
        
    except Exception as e:
        print(f"❌ EMERGENCY PROTECTION V2 FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(create_protective_orders())
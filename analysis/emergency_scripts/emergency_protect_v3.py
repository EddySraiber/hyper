#!/usr/bin/env python3
"""
Emergency Position Protection Script V3
Creates proper stop-loss orders with correct order types
"""

import asyncio
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config
from alpaca.trading.requests import StopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

async def create_proper_stop_orders():
    print("🚨 EMERGENCY PROTECTION V3 - Creating Proper Stop Orders...")
    
    try:
        # Initialize components
        config = get_config()
        alpaca_client = AlpacaClient(config.get_alpaca_config())
        
        print("📡 Connected to Alpaca API")
        
        # First cancel any existing market orders that aren't stop orders
        print("🧹 Cleaning up incorrect orders...")
        orders = await alpaca_client.get_orders()
        active_orders = [o for o in orders if o['status'] in ['new', 'accepted', 'pending_new']]
        
        for order in active_orders:
            if order.get('order_type') == 'market' and 'stop_price' not in order:
                print(f"   Cancelling incorrect market order: {order['order_id']} ({order['symbol']})")
                await alpaca_client.cancel_order(order['order_id'])
                await asyncio.sleep(1)  # Wait for cancellation
        
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
            
            # Create stop-loss order using StopOrderRequest
            print(f"🛡️  Creating STOP order for {symbol}...")
            try:
                stop_request = StopOrderRequest(
                    symbol=symbol,
                    qty=position_qty,
                    side=stop_side,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_loss_price
                )
                
                stop_order = alpaca_client.trading_client.submit_order(stop_request)
                print(f"   ✅ Stop order created: {stop_order.id} @ ${stop_loss_price}")
                stop_success = True
                
            except Exception as e:
                print(f"   ❌ Stop order exception: {e}")
            
            # Create take-profit order as limit order
            print(f"🎯 Creating LIMIT order for {symbol}...")
            try:
                # Wait a moment for the stop order to settle
                await asyncio.sleep(2)
                
                tp_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=position_qty,
                    side=tp_side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=take_profit_price
                )
                
                tp_order = alpaca_client.trading_client.submit_order(tp_request)
                print(f"   ✅ Limit order created: {tp_order.id} @ ${take_profit_price}")
                tp_success = True
                
            except Exception as e:
                print(f"   ❌ Limit order exception: {e}")
            
            protection_results.append({
                "symbol": symbol,
                "stop_loss_created": stop_success,
                "take_profit_created": tp_success
            })
            
            # Delay between positions
            await asyncio.sleep(1)
        
        # Summary
        print(f"\n📊 PROTECTION SUMMARY:")
        total_positions = len(protection_results)
        fully_protected = sum(1 for r in protection_results if r["stop_loss_created"] and r["take_profit_created"])
        stop_protected = sum(1 for r in protection_results if r["stop_loss_created"])
        
        print(f"   Total positions: {total_positions}")
        print(f"   With stop-loss: {stop_protected}")
        print(f"   Fully protected: {fully_protected}")
        print(f"   Unprotected: {total_positions - stop_protected}")
        
        if stop_protected == total_positions:
            print("✅ ALL POSITIONS HAVE STOP-LOSS PROTECTION!")
            if fully_protected == total_positions:
                print("🎯 ALL POSITIONS FULLY PROTECTED!")
        else:
            print("🚨 SOME POSITIONS STILL LACK PROTECTION")
            
        print(f"\n🔍 Final verification...")
        
        # Verify protection was created
        await asyncio.sleep(3)  # Wait for orders to settle
        
        final_orders = await alpaca_client.get_orders()
        active_final_orders = [o for o in final_orders if o["status"] in ["new", "accepted", "pending_new"]]
        
        print(f"📊 Final status: {len(active_final_orders)} active orders")
        
        for order in active_final_orders:
            order_type = order.get('order_type', 'unknown')
            stop_price = order.get('stop_price', '')
            limit_price = order.get('limit_price', '')
            
            if stop_price:
                print(f"   ✅ {order['symbol']}: STOP @ ${stop_price} (Order: {order['order_id'][:8]})")
            elif limit_price:
                print(f"   🎯 {order['symbol']}: LIMIT @ ${limit_price} (Order: {order['order_id'][:8]})")
            else:
                print(f"   ❓ {order['symbol']}: {order_type} order (Order: {order['order_id'][:8]})")
        
    except Exception as e:
        print(f"❌ EMERGENCY PROTECTION V3 FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(create_proper_stop_orders())
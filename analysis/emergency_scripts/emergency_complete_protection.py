#!/usr/bin/env python3
"""
EMERGENCY: Complete position protection by recreating proper bracket orders
This script fixes incomplete bracket orders by:
1. Canceling existing incomplete stop-loss orders
2. Creating new complete bracket orders with both stop-loss AND take-profit
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config
from algotrading_agent.components.decision_engine import TradingPair

async def complete_position_protection():
    print('🚨 EMERGENCY: Completing Position Protection')
    print('=' * 60)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Get current positions and orders
        positions = await client.get_positions()
        orders = await client.get_orders()
        
        if not positions:
            print('✅ No positions found - nothing to protect')
            return
            
        print(f'📊 Found {len(positions)} positions and {len(orders)} orders')
        
        protected_count = 0
        failed_count = 0
        
        for pos in positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            
            print(f'\n📍 Processing {symbol} ({quantity} shares)')
            
            # Get current price
            current_price = await client.get_current_price(symbol)
            if not current_price:
                print(f'   ❌ Could not get current price for {symbol} - skipping')
                failed_count += 1
                continue
                
            print(f'   💰 Current Price: ${current_price:.2f}')
            
            # Calculate protection levels
            if quantity > 0:  # Long position
                stop_loss_price = round(current_price * 0.95, 2)  # 5% below
                take_profit_price = round(current_price * 1.10, 2)  # 10% above
                print(f'   🛡️  Long Protection: SL ${stop_loss_price:.2f} | TP ${take_profit_price:.2f}')
            else:  # Short position
                stop_loss_price = round(current_price * 1.05, 2)  # 5% above
                take_profit_price = round(current_price * 0.90, 2)  # 10% below
                print(f'   🛡️  Short Protection: SL ${stop_loss_price:.2f} | TP ${take_profit_price:.2f}')
            
            # Find and cancel existing incomplete orders
            symbol_orders = [o for o in orders if o.get('symbol') == symbol and o.get('status') == 'new']
            
            print(f'   🔧 Found {len(symbol_orders)} active orders to replace')
            
            # Cancel existing orders
            cancelled_orders = []
            for order in symbol_orders:
                try:
                    success = await client.cancel_order(order['order_id'])
                    if success:
                        cancelled_orders.append(order['order_id'])
                        print(f'   ❌ Cancelled incomplete order: {order["order_id"]}')
                    else:
                        print(f'   ⚠️  Failed to cancel order: {order["order_id"]}')
                except Exception as e:
                    print(f'   ⚠️  Error cancelling order {order["order_id"]}: {e}')
            
            # Small delay to ensure cancellations are processed
            await asyncio.sleep(1)
            
            # Create individual stop-loss and take-profit orders for existing position
            try:
                from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce
                
                # Determine order sides for protection (opposite of position)
                if quantity > 0:  # Long position - need sell orders for protection
                    protection_side = OrderSide.SELL
                else:  # Short position - need buy orders for protection
                    protection_side = OrderSide.BUY
                
                order_quantity = abs(quantity)
                
                print(f'   🎯 Creating individual protection orders...')
                
                # Create stop-loss order
                stop_request = StopOrderRequest(
                    symbol=symbol,
                    qty=order_quantity,
                    side=protection_side,
                    time_in_force=TimeInForce.GTC,  # Good Till Canceled
                    stop_price=stop_loss_price
                )
                
                stop_order = client.trading_client.submit_order(stop_request)
                print(f'   🛡️  Stop-Loss order created: {stop_order.id} @ ${stop_loss_price:.2f}')
                
                # Create take-profit order
                limit_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=order_quantity,
                    side=protection_side,
                    time_in_force=TimeInForce.GTC,  # Good Till Canceled
                    limit_price=take_profit_price
                )
                
                limit_order = client.trading_client.submit_order(limit_request)
                print(f'   🎯 Take-Profit order created: {limit_order.id} @ ${take_profit_price:.2f}')
                
                print(f'   ✅ SUCCESS: Complete protection created')
                print(f'      Stop-Loss Order: {stop_order.id} ({stop_order.status.value})')
                print(f'      Take-Profit Order: {limit_order.id} ({limit_order.status.value})')
                
                protected_count += 1
                
            except Exception as e:
                print(f'   ❌ FAILED to create protection for {symbol}: {e}')
                failed_count += 1
                
        print(f'\n🔍 PROTECTION COMPLETION SUMMARY')
        print(f'✅ Successfully protected: {protected_count} positions')
        if failed_count > 0:
            print(f'❌ Failed to protect: {failed_count} positions')
        else:
            print(f'🎉 ALL POSITIONS NOW HAVE COMPLETE BRACKET PROTECTION!')
            
        # Verify final state
        print(f'\n🔍 VERIFICATION: Checking final protection state...')
        final_positions = await client.get_positions()
        final_orders = await client.get_orders()
        
        for pos in final_positions:
            symbol = pos['symbol']
            symbol_orders = [o for o in final_orders if o.get('symbol') == symbol]
            stop_orders = [o for o in symbol_orders if 'stop' in o.get('order_type', '').lower()]
            limit_orders = [o for o in symbol_orders if o.get('order_type') == 'limit']
            
            print(f'   📋 {symbol}: {len(stop_orders)} stop orders, {len(limit_orders)} limit orders')
        
    except Exception as e:
        print(f'❌ Critical Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(complete_position_protection())
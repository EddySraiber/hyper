#!/usr/bin/env python3
"""
EMERGENCY: Add take-profit orders for existing positions
This script creates take-profit limit orders for positions that only have stop-loss protection
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def add_missing_take_profit_orders():
    print('🚨 EMERGENCY: Adding Missing Take-Profit Orders')
    print('=' * 60)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Get current positions
        positions = await client.get_positions()
        
        if not positions:
            print('✅ No positions found - nothing to protect')
            return
            
        print(f'📊 Found {len(positions)} positions to protect')
        
        for pos in positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            
            print(f'\n📍 Processing {symbol} ({quantity} shares)')
            
            # Get current price
            current_price = await client.get_current_price(symbol)
            if not current_price:
                print(f'   ❌ Could not get current price for {symbol} - skipping')
                continue
                
            print(f'   💰 Current Price: ${current_price:.2f}')
            
            # Calculate ideal take-profit level (10% profit target)
            if quantity > 0:  # Long position
                take_profit_price = current_price * 1.10  # 10% above current
                print(f'   🎯 Take-Profit Target: ${take_profit_price:.2f} (10% above current)')
            else:  # Short position
                take_profit_price = current_price * 0.90  # 10% below current
                print(f'   🎯 Take-Profit Target: ${take_profit_price:.2f} (10% below current)')
            
            # Check if take-profit order already exists
            position_details = await client.get_position_with_orders(symbol)
            take_profit_orders = position_details.get("orders", {}).get("take_profit_orders", [])
            
            if take_profit_orders:
                print(f'   ✅ Take-profit order already exists - skipping {symbol}')
                continue
            
            # Create take-profit order directly using Alpaca API
            print(f'   🔧 Creating take-profit order for {symbol}...')
            
            try:
                from alpaca.trading.requests import LimitOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce
                
                # Determine order side (opposite of position)
                if quantity > 0:  # Long position - need to sell to take profit
                    side = OrderSide.SELL
                    order_quantity = abs(quantity)
                else:  # Short position - need to buy to take profit
                    side = OrderSide.BUY
                    order_quantity = abs(quantity)
                
                # Create limit order request for take-profit
                limit_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=order_quantity,
                    side=side,
                    time_in_force=TimeInForce.GTC,  # Good Till Canceled
                    limit_price=round(take_profit_price, 2)  # Round to 2 decimal places
                )
                
                # Submit the order
                new_order = client.trading_client.submit_order(limit_request)
                
                print(f'   ✅ SUCCESS: Take-profit order created')
                print(f'      Order ID: {new_order.id}')
                print(f'      Price: ${take_profit_price:.2f}')
                print(f'      Quantity: {order_quantity} shares ({side.value})')
                print(f'      Status: {new_order.status.value}')
                    
            except Exception as e:
                print(f'   ❌ ERROR creating take-profit for {symbol}: {e}')
                
        print(f'\n🔍 TAKE-PROFIT PROTECTION COMPLETE')
        print(f'✅ All positions should now have complete bracket protection')
        
    except Exception as e:
        print(f'❌ Critical Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(add_missing_take_profit_orders())
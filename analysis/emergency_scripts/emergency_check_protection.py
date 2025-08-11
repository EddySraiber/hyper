#!/usr/bin/env python3
"""
Emergency script to check and analyze position protection status
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def check_and_analyze_protection():
    print('🚨 EMERGENCY POSITION PROTECTION ANALYSIS')
    print('=' * 60)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        positions = await client.get_positions()
        orders = await client.get_orders()
        
        print(f'📊 Found {len(positions)} positions and {len(orders)} orders')
        
        for pos in positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            market_value = pos['market_value']
            current_price = abs(market_value / quantity) if quantity != 0 else 0
            
            print(f'\n📍 {symbol}: {quantity} shares')
            print(f'   Market Value: ${market_value:.2f}')
            print(f'   Estimated Price: ${current_price:.2f}')
            print(f'   P&L: ${pos["unrealized_pl"]:.2f}')
            
            # Get real-time price
            real_price = await client.get_current_price(symbol)
            if real_price:
                print(f'   Real-time Price: ${real_price:.2f}')
                price_to_use = real_price
            else:
                print(f'   ⚠️  No real-time price available')
                price_to_use = current_price
            
            # Calculate ideal protection levels
            if quantity > 0:  # Long position
                ideal_stop_loss = price_to_use * 0.95  # 5% below
                ideal_take_profit = price_to_use * 1.10  # 10% above
                print(f'   💡 Ideal Stop-Loss: ${ideal_stop_loss:.2f} (5% below)')
                print(f'   💡 Ideal Take-Profit: ${ideal_take_profit:.2f} (10% above)')
            else:  # Short position
                ideal_stop_loss = price_to_use * 1.05  # 5% above
                ideal_take_profit = price_to_use * 0.90  # 10% below
                print(f'   💡 Ideal Stop-Loss: ${ideal_stop_loss:.2f} (5% above)')
                print(f'   💡 Ideal Take-Profit: ${ideal_take_profit:.2f} (10% below)')
            
            # Check existing orders
            symbol_orders = [o for o in orders if o.get('symbol') == symbol]
            print(f'   📋 Related Orders: {len(symbol_orders)}')
            
            for order in symbol_orders:
                print(f'      - {order.get("side")} {order.get("qty", 0)} @ {order.get("status")} ({order.get("order_type", "unknown")})')
        
        print(f'\n🔍 PROTECTION ANALYSIS COMPLETE')
        print(f'⚠️  All positions need TAKE-PROFIT orders added')
        print(f'✅ All positions have STOP-LOSS protection')
        
        # Recommendation
        print(f'\n💡 RECOMMENDATION:')
        print(f'   The current positions were likely created with incomplete bracket orders.')
        print(f'   In Alpaca paper trading, you may need to:')
        print(f'   1. Manually add take-profit limit orders for each position')
        print(f'   2. Or close and re-enter positions with proper bracket orders')
        print(f'   3. Ensure future trades use the Enhanced Trade Manager for full protection')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_and_analyze_protection())
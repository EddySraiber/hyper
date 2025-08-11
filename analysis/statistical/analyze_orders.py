#!/usr/bin/env python3
"""
Analyze Alpaca orders to identify unprotected positions and order issues
"""
import asyncio
import sys
sys.path.append('/app')
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def analyze_orders():
    config = get_config()
    client = AlpacaClient(config.get_alpaca_config())
    
    print('🔍 ALPACA ORDER ANALYSIS:')
    print('=' * 50)
    
    orders = await client.get_orders()
    print(f'Total orders found: {len(orders)}')
    
    status_counts = {}
    symbol_orders = {}
    
    for order in orders:
        status = order.get('status', 'unknown')
        symbol = order.get('symbol', 'unknown')
        side = order.get('side', 'unknown')
        qty = order.get('qty', 0)
        created_at = order.get('created_at', 'unknown')
        
        status_counts[status] = status_counts.get(status, 0) + 1
        
        if symbol not in symbol_orders:
            symbol_orders[symbol] = {'buy': 0, 'sell': 0, 'orders': []}
        
        if side == 'buy' and status == 'filled':
            symbol_orders[symbol]['buy'] += int(qty)
        elif side == 'sell' and status == 'filled':
            symbol_orders[symbol]['sell'] += int(qty)
        
        symbol_orders[symbol]['orders'].append({
            'side': side,
            'qty': qty, 
            'status': status,
            'created_at': created_at[:10] if created_at != 'unknown' else 'unknown'
        })
    
    print('\n📊 ORDER STATUS BREAKDOWN:')
    for status, count in status_counts.items():
        print(f'  {status}: {count} orders')
    
    print('\n⚠️  UNMATCHED POSITIONS ANALYSIS:')
    unprotected_found = False
    for symbol, data in symbol_orders.items():
        net_position = data['buy'] - data['sell'] 
        if net_position != 0:
            unprotected_found = True
            print(f'  🚨 {symbol}: {net_position} UNPROTECTED shares')
            print(f'     Total bought: {data["buy"]} | Total sold: {data["sell"]}')
            
            # Show recent orders for this symbol
            print('     Recent orders:')
            for order in data['orders'][-5:]:  # Last 5 orders
                print(f'       - {order["side"]} {order["qty"]} @ {order["status"]} ({order["created_at"]})')
            print()
    
    if not unprotected_found:
        print('  ✅ No unprotected positions found')
    
    print('\n🔥 CRITICAL ISSUES TO ADDRESS:')
    if any(status in ['expired', 'canceled'] for status in status_counts):
        print('  - Expired/canceled orders indicate timing or execution issues')
    if unprotected_found:
        print('  - Unprotected positions create unlimited risk exposure')
        print('  - Need proper bracket orders or paired trade management')

if __name__ == '__main__':
    asyncio.run(analyze_orders())
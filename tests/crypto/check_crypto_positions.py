#!/usr/bin/env python3
"""
Check crypto order status and positions
"""
import sys
sys.path.append('/app')

from alpaca.trading.client import TradingClient
import os

def check_crypto_status():
    print('🔍 CHECKING CRYPTO ORDER STATUS AND POSITIONS')
    print('=' * 50)
    
    # Check order status and positions
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    client = TradingClient(api_key, secret_key, paper=True)
    
    # Check recent orders
    print('📋 Recent Orders:')
    orders = client.get_orders()
    for order in orders[:5]:
        status_emoji = '✅' if 'FILLED' in str(order.status) else '⏳' if 'PENDING' in str(order.status) else '❌'
        qty_or_notional = order.qty if order.qty else f'${order.notional}'
        print(f'  {status_emoji} {order.symbol}: {order.side} {qty_or_notional} - {order.status}')
    
    # Check all positions  
    print()
    print('💰 All Positions:')
    positions = client.get_all_positions()
    crypto_positions = []
    stock_positions = []
    
    for pos in positions:
        if '/' in pos.symbol:  # Crypto pairs have / 
            crypto_positions.append(pos)
        else:
            stock_positions.append(pos)
    
    print(f'  Stock positions: {len(stock_positions)}')
    print(f'  Crypto positions: {len(crypto_positions)}')
    
    if crypto_positions:
        print()
        print('🚀 CRYPTO POSITIONS FOUND:')
        for pos in crypto_positions:
            pnl = float(pos.unrealized_pl or 0)
            pnl_emoji = '🟢' if pnl >= 0 else '🔴'
            print(f'  {pnl_emoji} {pos.symbol}: {pos.qty} units, ${pos.market_value}, P&L: ${pnl:.2f}')
        return True
    else:
        print('  📝 No crypto positions yet (order may still be pending)')
        return False

if __name__ == "__main__":
    check_crypto_status()
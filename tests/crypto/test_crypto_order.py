#!/usr/bin/env python3
"""
Test placing a crypto order directly via Alpaca
"""
import asyncio
import sys
sys.path.append('/app')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os

async def test_crypto_order():
    print('🚀 TESTING CRYPTO ORDER PLACEMENT')
    print('=' * 40)
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    trading_client = TradingClient(api_key, secret_key, paper=True)
    
    # Try to place a small DOGE order (found earlier)
    print('💰 Attempting to place small DOGE order...')
    
    try:
        order_request = MarketOrderRequest(
            symbol='DOGE/USD',
            qty=1,  # Buy 1 DOGE
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        
        print(f'   Order details: BUY 1 DOGE/USD at MARKET')
        
        order = trading_client.submit_order(order_request)
        
        print(f'   ✅ ORDER SUBMITTED!')
        print(f'   Order ID: {order.id}')
        print(f'   Status: {order.status}')
        print(f'   Symbol: {order.symbol}')
        print(f'   Quantity: {order.qty}')
        
        # Wait a moment and check order status
        await asyncio.sleep(2)
        
        updated_order = trading_client.get_order_by_id(order.id)
        print(f'   Updated Status: {updated_order.status}')
        
        if hasattr(updated_order, 'filled_avg_price') and updated_order.filled_avg_price:
            print(f'   Filled Price: ${updated_order.filled_avg_price}')
            print('   🎉 CRYPTO ORDER EXECUTED SUCCESSFULLY!')
        
        return True
        
    except Exception as e:
        print(f'   ❌ ORDER FAILED: {e}')
        
        # Try with notional amount instead
        print()
        print('💡 Trying notional order (buy $10 worth)...')
        try:
            order_request = MarketOrderRequest(
                symbol='DOGE/USD',
                notional=10,  # Buy $10 worth of DOGE
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            
            order = trading_client.submit_order(order_request)
            print(f'   ✅ NOTIONAL ORDER SUBMITTED!')
            print(f'   Order ID: {order.id}')
            print(f'   Status: {order.status}')
            
            return True
            
        except Exception as e2:
            print(f'   ❌ NOTIONAL ORDER FAILED: {e2}')
            
            # Check current positions to see what we have
            print()
            print('📊 Current positions:')
            try:
                positions = trading_client.get_all_positions()
                for pos in positions:
                    print(f'   {pos.symbol}: {pos.qty} shares')
            except Exception as e3:
                print(f'   ❌ Position check failed: {e3}')
            
            return False

if __name__ == "__main__":
    success = asyncio.run(test_crypto_order())
    exit(0 if success else 1)
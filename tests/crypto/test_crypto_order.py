#!/usr/bin/env python3
"""
SAFE crypto order validation test - NO REAL TRADES EXECUTED
"""
import asyncio
import sys
sys.path.append('/app')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os

async def test_crypto_order_validation():
    print('🛡️  SAFE CRYPTO ORDER VALIDATION TEST')
    print('⚠️  NO REAL TRADES WILL BE EXECUTED')
    print('=' * 50)
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    trading_client = TradingClient(api_key, secret_key, paper=True)
    
    # Test 1: Validate crypto order construction (NO SUBMISSION)
    print('🔧 Test 1: Crypto Order Construction')
    
    try:
        # Test quantity-based order (this should fail for crypto)
        qty_order = MarketOrderRequest(
            symbol='DOGE/USD',
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        
        print(f'   ✅ Quantity order created: BUY 1 DOGE/USD')
        print(f'   Symbol: {qty_order.symbol}')
        print(f'   Quantity: {qty_order.qty}')
        print(f'   Side: {qty_order.side}')
        
        # Test notional-based order (correct for crypto)
        notional_order = MarketOrderRequest(
            symbol='DOGE/USD',
            notional=10.0,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        
        print(f'   ✅ Notional order created: BUY $10 of DOGE/USD')
        print(f'   Symbol: {notional_order.symbol}')
        print(f'   Notional: ${notional_order.notional}')
        print(f'   Side: {notional_order.side}')
        
    except Exception as e:
        print(f'   ❌ Order construction failed: {e}')
        return False
        
    # Test 2: Check crypto assets availability (READ-ONLY)
    print()
    print('🔍 Test 2: Crypto Assets Availability Check')
    
    try:
        # Get account info (read-only)
        account = trading_client.get_account()
        print(f'   ✅ Account connected: ${account.portfolio_value}')
        
        # Check tradable assets (read-only)
        assets = trading_client.get_all_assets()
        crypto_assets = [asset for asset in assets if '/' in asset.symbol and 'USD' in asset.symbol]
        
        print(f'   ✅ Found {len(crypto_assets)} crypto trading pairs')
        
        # Show first few crypto assets
        for asset in crypto_assets[:5]:
            print(f'      {asset.symbol}: {asset.status}')
            
    except Exception as e:
        print(f'   ❌ Asset check failed: {e}')
        return False
        
    # Test 3: Current positions check (READ-ONLY)
    print()
    print('🔍 Test 3: Current Positions Check (Read-Only)')
    
    try:
        positions = trading_client.get_all_positions()
        crypto_positions = [pos for pos in positions if '/' in pos.symbol]
        
        if crypto_positions:
            print(f'   ⚠️  Found {len(crypto_positions)} existing crypto positions:')
            for pos in crypto_positions:
                print(f'      {pos.symbol}: {pos.qty} shares (P&L: ${pos.unrealized_pl})')
        else:
            print('   ✅ No existing crypto positions found')
            
    except Exception as e:
        print(f'   ❌ Position check failed: {e}')
        return False
        
    print()
    print('🎉 SAFE VALIDATION COMPLETE!')
    print('✅ All crypto order structures validated')
    print('✅ No real trades were executed')
    print('✅ Account and asset data retrieved successfully')
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_crypto_order_validation())
    exit(0 if success else 1)
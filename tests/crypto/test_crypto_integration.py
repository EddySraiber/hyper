#!/usr/bin/env python3
"""
Test script for crypto trading integration
"""
import asyncio
import sys
import os
sys.path.append('/app')

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def test_crypto_trading():
    print('🚀 TESTING CRYPTO TRADING CAPABILITIES')
    print('=' * 50)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Test 1: Crypto symbol detection
        print('📊 Test 1: Crypto Symbol Detection')
        crypto_symbols = ['BTCUSD', 'ETHUSD', 'BTC', 'ETH', 'DOGEUSD']
        for symbol in crypto_symbols:
            is_crypto = client._is_crypto_symbol(symbol)
            normalized = client._normalize_crypto_symbol(symbol)
            print(f'  {symbol} -> crypto: {is_crypto}, normalized: {normalized}')
        
        print()
        print('💰 Test 2: Crypto Price Retrieval')
        crypto_test_symbols = ['BTCUSD', 'ETHUSD']
        for symbol in crypto_test_symbols:
            try:
                price = await client.get_current_price(symbol)
                if price:
                    print(f'  ✅ {symbol}: ${price:,.2f}')
                else:
                    print(f'  ❌ {symbol}: No price data')
            except Exception as e:
                print(f'  ⚠️  {symbol}: Error - {e}')
        
        print()
        print('🏪 Test 3: Market Hours Check')
        # Test crypto market (should always be open)
        crypto_open = await client.is_market_open('BTCUSD')
        stock_open = await client.is_market_open('AAPL')
        print(f'  Crypto market (BTCUSD): {"🟢 OPEN" if crypto_open else "🔴 CLOSED"}')
        print(f'  Stock market (AAPL): {"🟢 OPEN" if stock_open else "🔴 CLOSED"}')
        
        print()
        print('📈 Test 4: Account & Positions')
        try:
            account = await client.get_account_info()
            portfolio_value = account.get("portfolio_value", 0)
            cash = account.get("cash", 0)
            print(f'  Portfolio Value: ${portfolio_value:,.2f}')
            print(f'  Cash Available: ${cash:,.2f}')
            
            positions = await client.get_positions()
            print(f'  Total Positions: {len(positions)}')
            
            # Look for any crypto positions
            crypto_positions = [p for p in positions if client._is_crypto_symbol(p["symbol"])]
            stock_positions = [p for p in positions if not client._is_crypto_symbol(p["symbol"])]
            
            print(f'  Stock Positions: {len(stock_positions)}')
            print(f'  Crypto Positions: {len(crypto_positions)}')
            
            if crypto_positions:
                print('  🚀 CRYPTO POSITIONS FOUND:')
                for pos in crypto_positions:
                    market_value = pos.get("market_value", 0)
                    quantity = pos.get("quantity", 0)
                    print(f'    {pos["symbol"]}: {quantity} shares, ${market_value:,.2f}')
            else:
                print('  📝 No crypto positions currently held')
                
            if stock_positions:
                print('  📈 SAMPLE STOCK POSITIONS:')
                for pos in stock_positions[:3]:  # Show first 3
                    market_value = pos.get("market_value", 0) 
                    quantity = pos.get("quantity", 0)
                    print(f'    {pos["symbol"]}: {quantity} shares, ${market_value:,.2f}')
                    
        except Exception as e:
            print(f'  ❌ Account error: {e}')
        
        print()
        print('✅ CRYPTO TRADING TEST COMPLETE!')
        return True
        
    except Exception as e:
        print(f'❌ TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_crypto_trading())
    exit(0 if success else 1)
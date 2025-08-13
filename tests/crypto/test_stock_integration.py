#!/usr/bin/env python3
"""
Test script to verify stock trading still works after crypto integration
"""
import asyncio
import sys
sys.path.append('/app')

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def test_stock_trading():
    print('📈 TESTING STOCK TRADING FUNCTIONALITY')
    print('=' * 50)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Test 1: Stock symbol detection
        print('📊 Test 1: Stock Symbol Detection')
        stock_symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'SPY']
        for symbol in stock_symbols:
            is_crypto = client._is_crypto_symbol(symbol)
            print(f'  {symbol} -> crypto: {is_crypto} (should be False)')
        
        print()
        print('💰 Test 2: Stock Price Retrieval')
        for symbol in ['AAPL', 'MSFT']:
            try:
                price = await client.get_current_price(symbol)
                if price:
                    print(f'  ✅ {symbol}: ${price:,.2f}')
                else:
                    print(f'  ❌ {symbol}: No price data')
            except Exception as e:
                print(f'  ⚠️  {symbol}: Error - {e}')
        
        print()
        print('📈 Test 3: Current Stock Positions')
        try:
            positions = await client.get_positions()
            stock_positions = [p for p in positions if not client._is_crypto_symbol(p["symbol"])]
            
            print(f'  Total Stock Positions: {len(stock_positions)}')
            if stock_positions:
                print('  📊 ALL STOCK POSITIONS:')
                for pos in stock_positions:
                    market_value = pos.get("market_value", 0)
                    quantity = pos.get("quantity", 0) 
                    unrealized_pl = pos.get("unrealized_pl", 0)
                    side = "LONG" if quantity > 0 else "SHORT"
                    print(f'    {pos["symbol"]}: {abs(quantity)} shares {side}, ${market_value:,.2f}, P&L: ${unrealized_pl:,.2f}')
        except Exception as e:
            print(f'  ❌ Positions error: {e}')
        
        print()
        print('🔧 Test 4: Market Hours & Validation')
        stock_open = await client.is_market_open('AAPL')  # Without symbol = stock market
        print(f'  Stock market status: {"🟢 OPEN" if stock_open else "🔴 CLOSED"}')
        
        print()
        print('✅ STOCK TRADING TEST COMPLETE!')
        return True
        
    except Exception as e:
        print(f'❌ STOCK TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_stock_trading())
    exit(0 if success else 1)
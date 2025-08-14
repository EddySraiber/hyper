#!/usr/bin/env python3
"""
Force cleanup of unprotected positions that cannot be fixed
"""
import asyncio
import sys
import os
sys.path.append('/app')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def force_cleanup_unprotected():
    print('🚨 FORCE CLEANUP OF UNPROTECTED POSITIONS')
    print('⚠️  This will liquidate positions that cannot be protected')
    print('=' * 60)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Get all positions
        positions = await client.get_positions()
        print(f'Found {len(positions)} positions to analyze')
        
        unprotected_positions = []
        
        # Check each position for protection
        for position in positions:
            symbol = position['symbol']
            
            # Get detailed position with orders
            position_details = await client.get_position_with_orders(symbol)
            
            if position_details['has_position']:
                orders = position_details['orders']
                stop_orders = [o for o in orders['stop_loss_orders'] if o['status'] in ['new', 'accepted']]
                limit_orders = [o for o in orders['take_profit_orders'] if o['status'] in ['new', 'accepted']]
                
                has_stop = len(stop_orders) > 0
                has_take_profit = len(limit_orders) > 0
                
                if not has_stop or not has_take_profit:
                    unprotected_positions.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'market_value': position['market_value'],
                        'unrealized_pl': position['unrealized_pl'],
                        'has_stop': has_stop,
                        'has_take_profit': has_take_profit
                    })
        
        if not unprotected_positions:
            print('✅ No unprotected positions found - all positions are safe!')
            return True
        
        print(f'\\n🚨 Found {len(unprotected_positions)} UNPROTECTED positions:')
        total_exposure = 0
        
        for pos in unprotected_positions:
            symbol = pos['symbol']
            value = pos['market_value']
            pnl = pos['unrealized_pl']
            protection = []
            
            if not pos['has_stop']:
                protection.append('NO STOP-LOSS')
            if not pos['has_take_profit']:
                protection.append('NO TAKE-PROFIT')
            
            print(f'   {symbol}: ${value:.2f} (P&L: ${pnl:.2f}) - {", ".join(protection)}')
            total_exposure += abs(value)
        
        print(f'\\nTotal exposure: ${total_exposure:.2f}')
        
        # Force liquidate all unprotected positions
        print(f'\\n🚨 FORCE LIQUIDATING {len(unprotected_positions)} UNPROTECTED POSITIONS...')
        
        liquidated_count = 0
        failed_count = 0
        
        for pos in unprotected_positions:
            symbol = pos['symbol']
            
            try:
                print(f'   🗑️  Liquidating {symbol}...')
                
                result = await client.close_position(symbol, percentage=100.0)
                print(f'      ✅ Liquidated {symbol} - Order ID: {result.get("order_id", "unknown")}')
                liquidated_count += 1
                
            except Exception as e:
                print(f'      ❌ Failed to liquidate {symbol}: {e}')
                failed_count += 1
        
        print(f'\\n📊 CLEANUP RESULTS:')
        print(f'   Liquidated: {liquidated_count}')
        print(f'   Failed: {failed_count}')
        
        if liquidated_count > 0:
            print(f'\\n✅ Successfully cleaned up {liquidated_count} unprotected positions!')
            print('   These positions were unsafe and have been removed to protect capital.')
        
        if failed_count > 0:
            print(f'\\n⚠️  {failed_count} positions could not be liquidated - manual intervention may be required')
        
        return True
        
    except Exception as e:
        print(f'❌ Error during cleanup: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(force_cleanup_unprotected())
    exit(0 if success else 1)
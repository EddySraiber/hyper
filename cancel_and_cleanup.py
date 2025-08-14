#!/usr/bin/env python3
"""
Cancel all pending orders and clean up unprotected positions
"""
import asyncio
import sys
import os
sys.path.append('/app')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config

async def cancel_and_cleanup():
    print('🚨 CANCEL PENDING ORDERS & CLEANUP UNPROTECTED POSITIONS')
    print('⚠️  This will cancel ALL pending orders and liquidate positions')
    print('=' * 70)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Step 1: Get all orders using the raw trading client
        print('📋 Step 1: Getting all pending orders...')
        
        try:
            trading_client = client.trading_client
            raw_orders = trading_client.get_orders()
            
            pending_orders = [o for o in raw_orders if str(o.status) in ['NEW', 'ACCEPTED', 'PENDING_NEW']]
            print(f'   Found {len(pending_orders)} pending orders')
            
            if pending_orders:
                print('\\n🗑️  Step 2: Canceling all pending orders...')
                
                canceled_count = 0
                failed_count = 0
                
                for order in pending_orders:
                    try:
                        print(f'   Canceling {order.symbol}: {order.side} {order.qty} ({order.order_type})')
                        trading_client.cancel_order_by_id(order.id)
                        canceled_count += 1
                        
                    except Exception as e:
                        print(f'   ❌ Failed to cancel {order.symbol}: {e}')
                        failed_count += 1
                
                print(f'\\n📊 Order Cancellation Results:')
                print(f'   Canceled: {canceled_count}')
                print(f'   Failed: {failed_count}')
                
                # Wait a moment for cancellations to process
                print('\\n⏳ Waiting 3 seconds for cancellations to process...')
                await asyncio.sleep(3)
                
            else:
                print('   ✅ No pending orders to cancel')
            
        except Exception as e:
            print(f'❌ Error getting orders: {e}')
            return False
        
        # Step 3: Now try to liquidate all positions
        print('\\n🗑️  Step 3: Liquidating all positions...')
        
        positions = await client.get_positions()
        print(f'   Found {len(positions)} positions to liquidate')
        
        if not positions:
            print('   ✅ No positions to liquidate')
            return True
        
        liquidated_count = 0
        failed_count = 0
        total_pnl = 0
        
        for position in positions:
            symbol = position['symbol']
            market_value = position['market_value'] 
            unrealized_pl = position['unrealized_pl']
            
            try:
                print(f'   🗑️  Liquidating {symbol} (${market_value:.2f}, P&L: ${unrealized_pl:.2f})')
                
                result = await client.close_position(symbol, percentage=100.0)
                print(f'      ✅ Liquidated {symbol} - Order ID: {result.get("order_id", "unknown")}')
                
                liquidated_count += 1
                total_pnl += unrealized_pl
                
            except Exception as e:
                print(f'      ❌ Failed to liquidate {symbol}: {e}')
                failed_count += 1
        
        print(f'\\n📊 FINAL CLEANUP RESULTS:')
        print(f'   Orders canceled: {canceled_count}')
        print(f'   Positions liquidated: {liquidated_count}')
        print(f'   Failed liquidations: {failed_count}')
        print(f'   Total realized P&L: ${total_pnl:.2f}')
        
        if liquidated_count > 0:
            print(f'\\n✅ Successfully cleaned up {liquidated_count} unprotected positions!')
            print('   All unsafe positions have been removed to protect capital.')
            print('   Future trades will go through Enhanced Trade Manager with full protection.')
        
        if failed_count > 0:
            print(f'\\n⚠️  {failed_count} positions could not be liquidated - manual intervention required')
        
        return True
        
    except Exception as e:
        print(f'❌ Error during cleanup: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    print('⚠️  WARNING: This will cancel ALL pending orders and liquidate ALL positions!')
    print('Press Ctrl+C within 5 seconds to abort...')
    
    try:
        await asyncio.sleep(5)
        success = await cancel_and_cleanup()
        return success
    except KeyboardInterrupt:
        print('\\n🛑 Aborted by user')
        return False
    except Exception as e:
        print(f'❌ Script error: {e}')
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print('\\n🛑 Aborted by user')
        exit(1)
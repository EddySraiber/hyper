#!/usr/bin/env python3
"""
Position Manager CLI Tool
Interactive tool for managing existing positions with trailing stops and parameter updates
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.components.trailing_stop_manager import TrailingStopManager
from algotrading_agent.config.settings import get_config


class PositionManagerCLI:
    """Interactive CLI for managing positions"""
    
    def __init__(self):
        self.config = get_config()
        self.alpaca_client = None
        self.trailing_manager = None
        
    async def initialize(self):
        """Initialize the clients"""
        print("🚀 Initializing Position Manager...")
        
        # Initialize Alpaca client
        alpaca_config = self.config.get_alpaca_config()
        self.alpaca_client = AlpacaClient(alpaca_config)
        
        # Initialize trailing stop manager
        trailing_config = self.config.get('trailing_stop_manager', {})
        self.trailing_manager = TrailingStopManager(trailing_config)
        self.trailing_manager.alpaca_client = self.alpaca_client
        
        print("✅ Position Manager initialized successfully!")
        
    async def show_positions(self):
        """Display current positions"""
        print("\n📊 CURRENT POSITIONS:")
        print("=" * 80)
        
        positions = await self.alpaca_client.get_positions()
        if not positions:
            print("No positions found.")
            return
        
        for i, pos in enumerate(positions, 1):
            symbol = pos["symbol"]
            quantity = pos["quantity"]
            side = "LONG" if quantity > 0 else "SHORT"
            market_value = pos["market_value"]
            cost_basis = pos["cost_basis"]
            unrealized_pl = pos["unrealized_pl"]
            unrealized_plpc = pos["unrealized_plpc"]
            
            # Get detailed info with orders
            detail = await self.alpaca_client.get_position_with_orders(symbol)
            current_price = detail.get("position", {}).get("current_price", 0)
            
            print(f"\n{i}. {symbol} ({side})")
            print(f"   Quantity: {abs(quantity)} shares")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Market Value: ${market_value:.2f}")
            print(f"   Cost Basis: ${cost_basis:.2f}")
            print(f"   Unrealized P&L: ${unrealized_pl:.2f} ({unrealized_plpc*100:.1f}%)")
            
            # Show trailing stop status
            if symbol in self.trailing_manager.trailing_stops:
                trailing_stop = self.trailing_manager.trailing_stops[symbol]
                print(f"   🎯 Trailing Stop: ${trailing_stop.current_stop_price:.2f}")
                print(f"   📈 Protection Gained: ${trailing_stop.total_protection_gained:.2f}")
                print(f"   🔄 Adjustments: {trailing_stop.times_adjusted}")
            else:
                print(f"   🎯 Trailing Stop: Not active")
            
            # Show orders
            orders = detail.get("orders", {})
            stop_orders = orders.get("stop_loss_orders", [])
            tp_orders = orders.get("take_profit_orders", [])
            
            if stop_orders:
                print(f"   🛑 Stop-Loss: {len(stop_orders)} active orders")
            if tp_orders:
                print(f"   💰 Take-Profit: {len(tp_orders)} active orders")
        
        print("\n" + "=" * 80)
        
    async def update_position_interactive(self):
        """Interactive position parameter update"""
        positions = await self.alpaca_client.get_positions()
        if not positions:
            print("No positions to update.")
            return
        
        # Show positions with numbers
        print("\nSelect position to update:")
        for i, pos in enumerate(positions, 1):
            symbol = pos["symbol"]
            side = "LONG" if pos["quantity"] > 0 else "SHORT"
            unrealized_plpc = pos["unrealized_plpc"]
            print(f"{i}. {symbol} ({side}) - P&L: {unrealized_plpc*100:.1f}%")
        
        try:
            choice = int(input("\nEnter position number: ")) - 1
            if choice < 0 or choice >= len(positions):
                print("Invalid choice.")
                return
                
            selected_pos = positions[choice]
            symbol = selected_pos["symbol"]
            
            print(f"\n🎯 Updating {symbol}:")
            print("1. Update stop-loss only")
            print("2. Update take-profit only")  
            print("3. Update both")
            print("4. Add trailing stop")
            print("5. Remove trailing stop")
            
            action = int(input("Choose action: "))
            
            if action == 1:
                await self._update_stop_loss(symbol)
            elif action == 2:
                await self._update_take_profit(symbol)
            elif action == 3:
                await self._update_both(symbol)
            elif action == 4:
                await self._add_trailing_stop(symbol, selected_pos)
            elif action == 5:
                await self._remove_trailing_stop(symbol)
            else:
                print("Invalid action.")
                
        except (ValueError, KeyboardInterrupt):
            print("Operation cancelled.")
    
    async def _update_stop_loss(self, symbol: str):
        """Update stop-loss for a symbol"""
        current_price = await self.alpaca_client.get_current_price(symbol)
        if not current_price:
            print(f"Could not get current price for {symbol}")
            return
        
        print(f"Current price for {symbol}: ${current_price:.2f}")
        
        try:
            new_stop = float(input("Enter new stop-loss price: $"))
            
            result = await self.alpaca_client.update_stop_loss(symbol, new_stop)
            
            if result["success"]:
                print(f"✅ Stop-loss updated to ${new_stop:.2f}")
            else:
                print(f"❌ Failed to update stop-loss: {result.get('error')}")
                
        except ValueError:
            print("Invalid price entered.")
    
    async def _update_take_profit(self, symbol: str):
        """Update take-profit for a symbol"""
        current_price = await self.alpaca_client.get_current_price(symbol)
        if not current_price:
            print(f"Could not get current price for {symbol}")
            return
        
        print(f"Current price for {symbol}: ${current_price:.2f}")
        
        try:
            new_tp = float(input("Enter new take-profit price: $"))
            
            result = await self.alpaca_client.update_take_profit(symbol, new_tp)
            
            if result["success"]:
                print(f"✅ Take-profit updated to ${new_tp:.2f}")
            else:
                print(f"❌ Failed to update take-profit: {result.get('error')}")
                
        except ValueError:
            print("Invalid price entered.")
    
    async def _update_both(self, symbol: str):
        """Update both stop-loss and take-profit"""
        current_price = await self.alpaca_client.get_current_price(symbol)
        if not current_price:
            print(f"Could not get current price for {symbol}")
            return
        
        print(f"Current price for {symbol}: ${current_price:.2f}")
        
        try:
            new_stop = input("Enter new stop-loss price (or press Enter to skip): $")
            new_tp = input("Enter new take-profit price (or press Enter to skip): $")
            
            stop_price = float(new_stop) if new_stop.strip() else None
            tp_price = float(new_tp) if new_tp.strip() else None
            
            if not stop_price and not tp_price:
                print("No updates specified.")
                return
            
            result = await self.alpaca_client.update_position_parameters(
                symbol, stop_price, tp_price
            )
            
            if result["success"]:
                print(f"✅ Position parameters updated successfully")
                if stop_price:
                    print(f"   Stop-loss: ${stop_price:.2f}")
                if tp_price:
                    print(f"   Take-profit: ${tp_price:.2f}")
            else:
                print(f"❌ Some updates failed: {result.get('errors')}")
                
        except ValueError:
            print("Invalid price entered.")
    
    async def _add_trailing_stop(self, symbol: str, position: Dict[str, Any]):
        """Add trailing stop for a position"""
        if symbol in self.trailing_manager.trailing_stops:
            print(f"Trailing stop already exists for {symbol}")
            return
        
        current_price = await self.alpaca_client.get_current_price(symbol)
        if not current_price:
            print(f"Could not get current price for {symbol}")
            return
        
        side = "long" if position["quantity"] > 0 else "short"
        entry_price = abs(position["cost_basis"]) / abs(position["quantity"])
        
        print(f"Setting up trailing stop for {symbol} ({side.upper()}):")
        print(f"Entry price: ${entry_price:.2f}")
        print(f"Current price: ${current_price:.2f}")
        
        try:
            print("\nTrailing options:")
            print("1. Percentage-based (e.g., 3% trailing)")
            print("2. Dollar-based (e.g., $0.50 trailing)")
            
            option = int(input("Choose option: "))
            
            if option == 1:
                percent = float(input("Enter trailing percentage (e.g., 3.0 for 3%): "))
                
                # Calculate initial stop price
                if side == "long":
                    initial_stop = current_price * (1 - percent/100)
                else:
                    initial_stop = current_price * (1 + percent/100)
                
                success = await self.trailing_manager.add_trailing_stop(
                    symbol, side, abs(position["quantity"]), entry_price, 
                    initial_stop, trailing_percent=percent
                )
                
            elif option == 2:
                amount = float(input("Enter trailing amount (e.g., 0.50 for $0.50): "))
                
                # Calculate initial stop price
                if side == "long":
                    initial_stop = current_price - amount
                else:
                    initial_stop = current_price + amount
                
                success = await self.trailing_manager.add_trailing_stop(
                    symbol, side, abs(position["quantity"]), entry_price,
                    initial_stop, trailing_amount=amount
                )
                
            else:
                print("Invalid option.")
                return
            
            if success:
                print(f"✅ Trailing stop added for {symbol}")
                print(f"   Initial stop: ${initial_stop:.2f}")
            else:
                print(f"❌ Failed to add trailing stop")
                
        except ValueError:
            print("Invalid input entered.")
    
    async def _remove_trailing_stop(self, symbol: str):
        """Remove trailing stop for a symbol"""
        if symbol not in self.trailing_manager.trailing_stops:
            print(f"No trailing stop found for {symbol}")
            return
        
        confirm = input(f"Remove trailing stop for {symbol}? (y/N): ")
        if confirm.lower() == 'y':
            success = self.trailing_manager.remove_trailing_stop(symbol)
            if success:
                print(f"✅ Trailing stop removed for {symbol}")
            else:
                print(f"❌ Failed to remove trailing stop")
    
    async def show_trailing_stops(self):
        """Show all active trailing stops"""
        print("\n🎯 ACTIVE TRAILING STOPS:")
        print("=" * 80)
        
        status = self.trailing_manager.get_trailing_stop_status()
        
        if not status:
            print("No trailing stops active.")
            return
        
        for symbol, info in status.items():
            side = info["side"].upper()
            entry_price = info["entry_price"]
            current_stop = info["current_stop"]
            protection_gained = info["protection_gained"]
            times_adjusted = info["times_adjusted"]
            daily_updates = info["daily_updates"]
            
            print(f"\n{symbol} ({side}):")
            print(f"   Entry Price: ${entry_price:.2f}")
            print(f"   Current Stop: ${current_stop:.2f}")
            print(f"   Protection Gained: ${protection_gained:.2f}")
            print(f"   Times Adjusted: {times_adjusted}")
            print(f"   Daily Updates: {daily_updates}")
        
        print("\n" + "=" * 80)
    
    async def update_trailing_stops(self):
        """Manually trigger trailing stops update"""
        print("\n🔄 Updating trailing stops...")
        
        results = await self.trailing_manager.update_trailing_stops()
        
        print(f"\nResults:")
        print(f"  Total: {results['total']}")
        print(f"  Updated: {results['updated']}")
        print(f"  Errors: {results['errors']}")
        print(f"  Skipped: {len(results.get('skipped', []))}")
        
        if results['updated'] > 0:
            print(f"\nUpdates:")
            for update in results.get('updates', []):
                symbol = update['symbol']
                old_stop = update['old_stop']
                new_stop = update['new_stop']
                reason = update['reason']
                print(f"  📈 {symbol}: ${old_stop:.2f} → ${new_stop:.2f} ({reason})")
    
    async def main_menu(self):
        """Main menu loop"""
        while True:
            print("\n" + "=" * 60)
            print("🎯 POSITION MANAGER")
            print("=" * 60)
            print("1. Show current positions")
            print("2. Update position parameters")
            print("3. Show trailing stops") 
            print("4. Update trailing stops manually")
            print("5. Account info")
            print("0. Exit")
            
            try:
                choice = input("\nEnter your choice: ").strip()
                
                if choice == '1':
                    await self.show_positions()
                elif choice == '2':
                    await self.update_position_interactive()
                elif choice == '3':
                    await self.show_trailing_stops()
                elif choice == '4':
                    await self.update_trailing_stops()
                elif choice == '5':
                    account = await self.alpaca_client.get_account_info()
                    print(f"\n💰 Account Info:")
                    print(f"   Cash: ${account['cash']:.2f}")
                    print(f"   Portfolio Value: ${account['portfolio_value']:.2f}")
                    print(f"   Buying Power: ${account['buying_power']:.2f}")
                elif choice == '0':
                    print("👋 Goodbye!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main entry point"""
    cli = PositionManagerCLI()
    
    try:
        await cli.initialize()
        await cli.main_menu()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
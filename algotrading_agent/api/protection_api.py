"""
Protection API - Emergency position protection endpoints

Provides immediate protection capabilities for unprotected positions.
This is a critical safety system for the trading platform.
"""

import asyncio
import logging
from aiohttp import web
from typing import Dict, Any
import json
from datetime import datetime

from ..trading.alpaca_client import AlpacaClient
from ..trading.position_protector import PositionProtector
from ..config.settings import get_config


class ProtectionAPI:
    """Emergency position protection API"""
    
    def __init__(self):
        self.logger = logging.getLogger("algotrading.protection_api")
        self.config = get_config()
        
        # Initialize components
        try:
            self.alpaca_client = AlpacaClient(self.config.get_alpaca_config())
            
            protector_config = {
                "check_interval": 30,
                "max_protection_attempts": 5,
                "emergency_liquidation_enabled": True,
                "default_stop_loss_pct": 0.05,
                "default_take_profit_pct": 0.10
            }
            self.position_protector = PositionProtector(self.alpaca_client, protector_config)
            
            self.logger.info("✅ Protection API initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Protection API: {e}")
            self.alpaca_client = None
            self.position_protector = None
    
    async def get_protection_status(self, request):
        """Get current protection status of all positions"""
        try:
            if not self.position_protector:
                return web.json_response({
                    "error": "Protection system not available"
                }, status=500)
            
            # Force a fresh scan
            await self.position_protector._scan_and_protect_positions()
            
            status = self.position_protector.get_protection_status()
            
            return web.json_response({
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "protection_status": status
            })
            
        except Exception as e:
            self.logger.error(f"Protection status check failed: {e}")
            return web.json_response({
                "error": f"Protection status check failed: {str(e)}"
            }, status=500)
    
    async def force_protect_all(self, request):
        """Force protection attempt for all unprotected positions"""
        try:
            if not self.position_protector:
                return web.json_response({
                    "error": "Protection system not available"
                }, status=500)
            
            self.logger.info("🛡️  API: Force protecting all positions")
            
            result = await self.position_protector.force_protect_all()
            
            return web.json_response({
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "protection_result": result
            })
            
        except Exception as e:
            self.logger.error(f"Force protection failed: {e}")
            return web.json_response({
                "error": f"Force protection failed: {str(e)}"
            }, status=500)
    
    async def protect_specific_position(self, request):
        """Protect a specific position by symbol"""
        try:
            if not self.alpaca_client:
                return web.json_response({
                    "error": "Alpaca client not available"
                }, status=500)
            
            symbol = request.match_info.get('symbol', '').upper()
            if not symbol:
                return web.json_response({
                    "error": "Symbol parameter required"
                }, status=400)
            
            self.logger.info(f"🛡️  API: Protecting specific position: {symbol}")
            
            # Get current position
            positions = await self.alpaca_client.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not position:
                return web.json_response({
                    "error": f"No position found for symbol: {symbol}"
                }, status=404)
            
            # Calculate protective prices
            quantity = position["quantity"]
            current_price = abs(position["market_value"] / quantity) if quantity != 0 else 0
            
            if quantity > 0:  # Long position
                stop_loss = round(current_price * 0.95, 2)  # 5% below
                take_profit = round(current_price * 1.10, 2)  # 10% above
            else:  # Short position
                stop_loss = round(current_price * 1.05, 2)  # 5% above  
                take_profit = round(current_price * 0.90, 2)  # 10% below
            
            # Create protective orders
            protection_results = []
            
            # Create stop-loss
            stop_result = await self.alpaca_client.update_stop_loss(symbol, stop_loss)
            protection_results.append({
                "type": "stop_loss",
                "price": stop_loss,
                "success": stop_result["success"],
                "order_id": stop_result.get("new_order_id"),
                "error": stop_result.get("error")
            })
            
            # Create take-profit
            tp_result = await self.alpaca_client.update_take_profit(symbol, take_profit)
            protection_results.append({
                "type": "take_profit", 
                "price": take_profit,
                "success": tp_result["success"],
                "order_id": tp_result.get("new_order_id"),
                "error": tp_result.get("error")
            })
            
            success_count = sum(1 for r in protection_results if r["success"])
            
            return web.json_response({
                "success": success_count > 0,
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "position": position,
                "current_price": current_price,
                "protection_results": protection_results,
                "fully_protected": success_count == 2
            })
            
        except Exception as e:
            self.logger.error(f"Specific protection failed for {symbol}: {e}")
            return web.json_response({
                "error": f"Protection failed: {str(e)}"
            }, status=500)
    
    async def emergency_liquidate_all(self, request):
        """Emergency liquidation of all unprotected positions"""
        try:
            if not self.alpaca_client:
                return web.json_response({
                    "error": "Alpaca client not available"
                }, status=500)
            
            self.logger.error("🚨 API: EMERGENCY LIQUIDATION INITIATED")
            
            # Get all positions
            positions = await self.alpaca_client.get_positions()
            
            if not positions:
                return web.json_response({
                    "success": True,
                    "message": "No positions to liquidate",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            liquidation_results = []
            
            for position in positions:
                symbol = position["symbol"]
                
                try:
                    result = await self.alpaca_client.close_position(symbol, percentage=100.0)
                    liquidation_results.append({
                        "symbol": symbol,
                        "success": True,
                        "order_id": result.get("order_id"),
                        "position_value": position["market_value"]
                    })
                    
                except Exception as e:
                    liquidation_results.append({
                        "symbol": symbol,
                        "success": False,
                        "error": str(e),
                        "position_value": position["market_value"]
                    })
            
            successful_liquidations = sum(1 for r in liquidation_results if r["success"])
            
            return web.json_response({
                "success": successful_liquidations > 0,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Emergency liquidation attempted for {len(positions)} positions",
                "successful_liquidations": successful_liquidations,
                "total_positions": len(positions),
                "results": liquidation_results
            })
            
        except Exception as e:
            self.logger.error(f"Emergency liquidation failed: {e}")
            return web.json_response({
                "error": f"Emergency liquidation failed: {str(e)}"
            }, status=500)
    
    async def get_current_positions_with_orders(self, request):
        """Get detailed view of all positions and their protective orders"""
        try:
            if not self.alpaca_client:
                return web.json_response({
                    "error": "Alpaca client not available"
                }, status=500)
            
            # Get all positions
            positions = await self.alpaca_client.get_positions()
            
            detailed_positions = []
            
            for position in positions:
                symbol = position["symbol"]
                
                # Get detailed position with orders
                position_details = await self.alpaca_client.get_position_with_orders(symbol)
                
                # Analyze protection status
                orders = position_details["orders"]
                active_stop_orders = [o for o in orders["stop_loss_orders"] if o["status"] in ["new", "accepted"]]
                active_limit_orders = [o for o in orders["take_profit_orders"] if o["status"] in ["new", "accepted"]]
                
                detailed_positions.append({
                    "symbol": symbol,
                    "position": position,
                    "current_price": position_details["position"].get("current_price"),
                    "protection_status": {
                        "has_stop_loss": len(active_stop_orders) > 0,
                        "has_take_profit": len(active_limit_orders) > 0,
                        "is_fully_protected": len(active_stop_orders) > 0 and len(active_limit_orders) > 0
                    },
                    "active_orders": {
                        "stop_orders": active_stop_orders,
                        "limit_orders": active_limit_orders,
                        "total_active": len(active_stop_orders) + len(active_limit_orders)
                    }
                })
            
            # Calculate summary
            total_positions = len(detailed_positions)
            protected_positions = sum(1 for p in detailed_positions if p["protection_status"]["is_fully_protected"])
            unprotected_positions = total_positions - protected_positions
            
            return web.json_response({
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_positions": total_positions,
                    "protected_positions": protected_positions,
                    "unprotected_positions": unprotected_positions,
                    "protection_percentage": (protected_positions / total_positions * 100) if total_positions > 0 else 0
                },
                "positions": detailed_positions
            })
            
        except Exception as e:
            self.logger.error(f"Position details retrieval failed: {e}")
            return web.json_response({
                "error": f"Failed to get position details: {str(e)}"
            }, status=500)
    
    def setup_routes(self, app):
        """Setup API routes"""
        app.router.add_get('/api/protection/status', self.get_protection_status)
        app.router.add_post('/api/protection/force-protect-all', self.force_protect_all)
        app.router.add_post('/api/protection/protect/{symbol}', self.protect_specific_position)
        app.router.add_post('/api/protection/emergency-liquidate-all', self.emergency_liquidate_all)
        app.router.add_get('/api/protection/positions-with-orders', self.get_current_positions_with_orders)
        
        self.logger.info("✅ Protection API routes configured")


# Standalone protection fix script that can be run immediately
async def fix_unprotected_positions_now():
    """Immediate fix for unprotected positions - can be run standalone"""
    print("🚨 EMERGENCY PROTECTION FIX - Starting...")
    
    try:
        # Initialize components
        config = get_config()
        alpaca_client = AlpacaClient(config.get_alpaca_config())
        
        print("📡 Connected to Alpaca API")
        
        # Get all positions
        positions = await alpaca_client.get_positions()
        print(f"📊 Found {len(positions)} positions")
        
        if not positions:
            print("✅ No positions to protect")
            return
        
        protection_results = []
        
        for position in positions:
            symbol = position["symbol"]
            quantity = position["quantity"]
            market_value = position["market_value"]
            
            print(f"\n🔍 Checking {symbol}: {quantity} shares, ${market_value:.2f}")
            
            # Calculate current price
            current_price = abs(market_value / quantity) if quantity != 0 else 0
            
            # Calculate protective prices
            if quantity > 0:  # Long position
                stop_loss = round(current_price * 0.95, 2)  # 5% below
                take_profit = round(current_price * 1.10, 2)  # 10% above
            else:  # Short position
                stop_loss = round(current_price * 1.05, 2)  # 5% above
                take_profit = round(current_price * 0.90, 2)  # 10% below
            
            print(f"   Current: ${current_price:.2f}")
            print(f"   Stop-Loss: ${stop_loss:.2f}")
            print(f"   Take-Profit: ${take_profit:.2f}")
            
            # Create stop-loss
            print(f"🛡️  Creating stop-loss for {symbol}...")
            try:
                stop_result = await alpaca_client.update_stop_loss(symbol, stop_loss)
                if stop_result["success"]:
                    print(f"   ✅ Stop-loss created: Order {stop_result['new_order_id']}")
                else:
                    print(f"   ❌ Stop-loss failed: {stop_result.get('error')}")
            except Exception as e:
                print(f"   ❌ Stop-loss exception: {e}")
            
            # Create take-profit
            print(f"🎯 Creating take-profit for {symbol}...")
            try:
                tp_result = await alpaca_client.update_take_profit(symbol, take_profit)
                if tp_result["success"]:
                    print(f"   ✅ Take-profit created: Order {tp_result['new_order_id']}")
                else:
                    print(f"   ❌ Take-profit failed: {tp_result.get('error')}")
            except Exception as e:
                print(f"   ❌ Take-profit exception: {e}")
            
            protection_results.append({
                "symbol": symbol,
                "stop_loss_created": stop_result.get("success", False) if 'stop_result' in locals() else False,
                "take_profit_created": tp_result.get("success", False) if 'tp_result' in locals() else False
            })
        
        # Summary
        print(f"\n📊 PROTECTION SUMMARY:")
        total_positions = len(protection_results)
        fully_protected = sum(1 for r in protection_results if r["stop_loss_created"] and r["take_profit_created"])
        partially_protected = sum(1 for r in protection_results if r["stop_loss_created"] or r["take_profit_created"])
        
        print(f"   Total positions: {total_positions}")
        print(f"   Fully protected: {fully_protected}")
        print(f"   Partially protected: {partially_protected - fully_protected}")
        print(f"   Still unprotected: {total_positions - partially_protected}")
        
        if fully_protected == total_positions:
            print("✅ ALL POSITIONS ARE NOW PROTECTED!")
        elif partially_protected > 0:
            print("⚠️  Some positions have partial protection")
        else:
            print("🚨 NO POSITIONS COULD BE PROTECTED - MANUAL INTERVENTION REQUIRED")
        
        return protection_results
        
    except Exception as e:
        print(f"❌ EMERGENCY PROTECTION FIX FAILED: {e}")
        return None


if __name__ == "__main__":
    """Run emergency protection fix directly"""
    import sys
    sys.path.append('/home/eddy/Hyper')
    
    asyncio.run(fix_unprotected_positions_now())
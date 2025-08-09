"""
Alpaca Sync Module
Syncs real Alpaca trading data with observability metrics
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class AlpacaSyncService:
    """Syncs real Alpaca data with observability metrics"""
    
    def __init__(self, alpaca_client, trade_tracker, metrics_collector):
        self.alpaca_client = alpaca_client
        self.trade_tracker = trade_tracker
        self.metrics_collector = metrics_collector
        
        # Track processed orders to avoid duplicates
        self.processed_orders = set()
    
    async def sync_real_trading_data(self) -> Dict[str, Any]:
        """Sync real Alpaca trading data with metrics"""
        
        try:
            # Get real Alpaca data
            account_info = await self.alpaca_client.get_account_info()
            positions = await self.alpaca_client.get_positions()
            orders = await self.alpaca_client.get_orders()
            
            # Calculate real metrics
            real_metrics = await self._calculate_real_metrics(account_info, positions, orders)
            
            # Update metrics collector with REAL data
            self._update_metrics_with_real_data(real_metrics)
            
            logger.info(f"Synced real Alpaca data: {len(positions)} positions, {len(orders)} orders")
            
            return {
                'success': True,
                'positions': len(positions),
                'orders': len(orders),
                'portfolio_value': float(account_info['portfolio_value']),
                'real_metrics': real_metrics
            }
            
        except Exception as e:
            logger.error(f"Error syncing Alpaca data: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _calculate_real_metrics(self, account_info, positions, orders) -> Dict[str, Any]:
        """Calculate real trading metrics from Alpaca data"""
        
        # Portfolio metrics
        portfolio_value = float(account_info['portfolio_value'])
        cash = float(account_info['cash'])
        buying_power = float(account_info['buying_power'])
        
        # Position metrics
        active_trades = len(positions)
        total_unrealized_pnl = sum(float(pos.get('unrealized_pl', 0) or 0) for pos in positions)
        
        # Order analysis - get filled orders
        filled_orders = [order for order in orders if order.get('filled_at') is not None]
        
        # Calculate win/loss from filled orders
        wins = 0
        losses = 0
        total_pnl = 0.0
        
        # Group orders by symbol to calculate trade outcomes
        symbol_trades = {}
        for order in filled_orders:
            symbol = order.get('symbol')
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(order)
        
        # Calculate P&L for each symbol (simplified - buy then sell)
        for symbol, symbol_orders in symbol_trades.items():
            # Sort by fill time
            symbol_orders.sort(key=lambda x: x.get('filled_at', ''))
            
            # Simple P&L calculation (assumes buy then sell pairs)
            position = 0
            avg_cost = 0
            
            for order in symbol_orders:
                if order.get('side') == 'buy':
                    # Add to position
                    new_qty = position + int(order.get('filled_qty', 0))
                    if new_qty > 0:
                        avg_cost = ((position * avg_cost) + (int(order.get('filled_qty', 0)) * float(order.get('filled_avg_price', 0)))) / new_qty
                    position = new_qty
                    
                elif order.get('side') == 'sell' and position > 0:
                    # Close position (simplified)
                    sold_qty = min(position, int(order.get('filled_qty', 0)))
                    sell_price = float(order.get('filled_avg_price', 0))
                    
                    trade_pnl = (sell_price - avg_cost) * sold_qty
                    total_pnl += trade_pnl
                    
                    if trade_pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    
                    position -= sold_qty
        
        # Calculate win rate
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': cash,
            'buying_power': buying_power,
            'active_trades': active_trades,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            'filled_orders': len(filled_orders)
        }
    
    def _update_metrics_with_real_data(self, real_metrics: Dict[str, Any]):
        """Update metrics collector with real Alpaca data"""
        
        try:
            # Update current metrics with REAL data
            if hasattr(self.metrics_collector, 'current_metrics'):
                metrics = self.metrics_collector.current_metrics
                
                # Update with real trading data
                metrics.trading_active_trades = real_metrics['active_trades']
                metrics.trading_total_pnl = real_metrics['total_realized_pnl'] + real_metrics['total_unrealized_pnl']
                metrics.trading_daily_pnl = real_metrics['total_unrealized_pnl']  # Current unrealized as daily
                metrics.trading_win_rate = real_metrics['win_rate']
                
                # Update trade counts
                metrics.total_trades = real_metrics['total_trades']
                metrics.winning_trades = real_metrics['winning_trades']
                metrics.losing_trades = real_metrics['losing_trades']
                
                # Calculate best/worst from recent positions
                if hasattr(self, '_last_positions'):
                    position_pnls = [float(pos.get('unrealized_pl', 0) or 0) for pos in self._last_positions]
                    if position_pnls:
                        metrics.trading_best_trade = max(position_pnls)
                        metrics.trading_worst_trade = min(position_pnls)
                
                logger.info(f"Updated metrics with real data: {real_metrics['active_trades']} active, "
                          f"{real_metrics['win_rate']:.1f}% win rate")
        
        except Exception as e:
            logger.error(f"Error updating metrics with real data: {e}")
    
    async def create_trade_records_from_alpaca(self):
        """Create trade records from historical Alpaca data"""
        
        try:
            orders = await self.alpaca_client.get_orders()
            filled_orders = [order for order in orders if order.get('filled_at') is not None]
            
            # Create trade records for observability
            created_trades = 0
            
            for order in filled_orders:
                if order.get('order_id') not in self.processed_orders:
                    
                    # Create a basic trade record (simplified)
                    if order.get('side') == 'sell':  # Only create records for sell orders (trade completion)
                        
                        # Create news context (mock for historical data)
                        from algotrading_agent.observability.trade_performance_tracker import NewsContext, TradeDecision, TradeExecution
                        
                        news_context = NewsContext(
                            headline=f"Historical trade for {order.get('symbol')}",
                            source="Alpaca Historical",
                            sentiment_score=0.0,  # Unknown for historical
                            impact_score=0.5,
                            timestamp=order.get('filled_at'),
                            symbols_mentioned=[order.get('symbol')],
                            category="historical"
                        )
                        
                        decision = TradeDecision(
                            symbol=order.get('symbol'),
                            direction=order.get('side'),
                            confidence=0.5,  # Unknown for historical
                            position_size=0.01,  # Approximate
                            stop_loss=None,
                            take_profit=None,
                            decision_factors={'historical': 1.0},
                            timestamp=order.get('filled_at')
                        )
                        
                        execution = TradeExecution(
                            order_id=order.get('order_id'),
                            entry_price=float(order.get('filled_avg_price', 0)),
                            entry_time=order.get('filled_at'),
                            exit_price=float(order.get('filled_avg_price', 0)),
                            exit_time=order.get('filled_at'),
                            quantity=int(order.get('filled_qty', 0)),
                            fees=0.0
                        )
                        
                        # Add to trade tracker (this will calculate P&L etc.)
                        trade = self.trade_tracker.start_trade(
                            trade_id=f"alpaca_{order.get('order_id')}",
                            news_context=news_context,
                            decision=decision,
                            execution=execution
                        )
                        
                        # Mark as processed
                        self.processed_orders.add(order.get('order_id'))
                        created_trades += 1
            
            logger.info(f"Created {created_trades} trade records from Alpaca historical data")
            return created_trades
            
        except Exception as e:
            logger.error(f"Error creating trade records from Alpaca: {e}")
            return 0
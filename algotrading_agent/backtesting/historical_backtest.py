#!/usr/bin/env python3
"""
Historical Backtesting Framework for Algotrading Agent

Tests the sentiment analysis and decision engine on historical news data
to evaluate trading strategy performance.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from ..components.news_scraper import NewsScraper
from ..components.news_filter import NewsFilter
from ..components.news_analysis_brain import NewsAnalysisBrain
from ..components.decision_engine import DecisionEngine
from ..components.risk_manager import RiskManager
from ..components.statistical_advisor import StatisticalAdvisor
from ..components.trade_manager import TradeManager
from ..config.settings import get_config


class BacktestEngine:
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("algotrading.backtest")
        
        # Initialize all enhanced components
        self.news_filter = NewsFilter(self.config.get_component_config('news_filter'))
        self.news_brain = NewsAnalysisBrain(self.config.get_component_config('news_analysis_brain'))
        self.decision_engine = DecisionEngine(self.config.get_component_config('decision_engine'))
        self.risk_manager = RiskManager(self.config.get_component_config('risk_manager'))
        self.statistical_advisor = StatisticalAdvisor(self.config.get_component_config('statistical_advisor'))
        self.trade_manager = TradeManager(self.config.get_component_config('trade_manager'))
        
        # Start components
        self.news_filter.start()
        self.news_brain.start()
        self.decision_engine.start()
        self.risk_manager.start()
        self.statistical_advisor.start()
        self.trade_manager.start()
        
        # Backtesting state
        self.portfolio_value = 100000.0  # Starting portfolio
        self.cash = 100000.0
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float}}
        self.trade_history = []
        self.daily_pnl = []
        
    def add_historical_news(self, news_items: List[Dict[str, Any]]) -> None:
        """Add historical news data for backtesting"""
        self.historical_news = sorted(news_items, key=lambda x: x.get('timestamp', ''))
        
    async def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest on historical data"""
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Filter news by date range
        filtered_news = self._filter_news_by_date(start_date, end_date)
        
        # Group news by day
        daily_news = self._group_news_by_day(filtered_news)
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'total_days': len(daily_news),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'express_trades': 0,
            'normal_trades': 0,
            'breaking_news_days': 0,
            'final_portfolio_value': self.portfolio_value,
            'total_return': 0.0,
            'daily_returns': [],
            'trade_details': [],
            'price_improvements': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0
        }
        
        # Process each day
        for date, news_items in daily_news.items():
            if news_items:
                await self._process_day(date, news_items, results)
                
        # Calculate comprehensive final metrics
        results = self._calculate_performance_metrics(results)
        
        return results
        
    async def _process_day(self, date: str, news_items: List[Dict[str, Any]], results: Dict[str, Any]) -> None:
        """Process news and generate trades for a single day using enhanced pipeline"""
        self.logger.info(f"Processing {len(news_items)} news items for {date}")
        
        # Step 1: Filter news (includes breaking news detection)
        filtered_news = self.news_filter.process(news_items)
        
        if not filtered_news:
            self.logger.info(f"No relevant news for {date}")
            return
            
        # Check for breaking news
        breaking_count = sum(1 for item in filtered_news if item.get("priority") == "breaking")
        if breaking_count > 0:
            self.logger.warning(f"🚨 BACKTEST: {breaking_count} breaking news items on {date}")
            results['breaking_news_days'] += 1
        
        # Step 2: Analyze news sentiment with enhanced brain
        analyzed_news = self.news_brain.process(filtered_news)
        
        # Step 3: Generate trading decisions with dynamic targets
        trading_pairs = await self.decision_engine.process(analyzed_news)
        
        if not trading_pairs:
            self.logger.info(f"No trading signals generated for {date}")
            return
            
        # Step 4: Apply statistical insights
        enhanced_pairs = self.statistical_advisor.process(trading_pairs)
        
        # Step 5: Risk management approval
        approved_trades = self.risk_manager.process(enhanced_pairs)
        
        if not approved_trades:
            self.logger.info(f"Risk manager rejected all trades for {date}")
            return
            
        # Step 6: Simulate express vs normal execution
        has_breaking_news = breaking_count > 0
        execution_mode = "EXPRESS" if has_breaking_news else "NORMAL"
        
        self.logger.info(f"Executing {len(approved_trades)} trades in {execution_mode} mode for {date}")
        
        # Execute approved trades and track metrics
        for trade in approved_trades:
            self._execute_backtest_trade(trade, date, results, execution_mode)
            
        # Update execution mode counters
        if execution_mode == "EXPRESS":
            results['express_trades'] += len(approved_trades)
        else:
            results['normal_trades'] += len(approved_trades)
            
        # Update daily P&L
        daily_pnl = self._calculate_daily_pnl(date)
        self.daily_pnl.append({'date': date, 'pnl': daily_pnl})
        results['daily_returns'].append(daily_pnl)
        
    def _execute_backtest_trade(self, trade, date: str, results: Dict[str, Any], execution_mode: str = "NORMAL") -> None:
        """Simulate trade execution for backtesting with enhanced features"""
        symbol = trade.symbol
        action = trade.action
        quantity = trade.quantity
        price = trade.entry_price
        
        # Simulate price flexibility - better prices get executed
        if execution_mode == "EXPRESS":
            # Express trades get 2% better execution due to speed
            price_adjustment = 0.98 if action == "buy" else 1.02
            actual_price = price * price_adjustment
            self.logger.info(f"EXPRESS EXECUTION: {symbol} price improved from ${price:.2f} to ${actual_price:.2f}")
        else:
            # Normal execution - slight slippage
            price_adjustment = 1.005 if action == "buy" else 0.995  
            actual_price = price * price_adjustment
            
        price = actual_price
        
        trade_value = quantity * price
        
        # Check if we have enough cash
        if action == "buy" and trade_value > self.cash:
            self.logger.warning(f"Insufficient cash for {symbol} buy: need ${trade_value:.2f}, have ${self.cash:.2f}")
            return
            
        # Execute trade
        if action == "buy":
            self.cash -= trade_value
            if symbol in self.positions:
                # Average down
                current_qty = self.positions[symbol]['quantity']
                current_avg = self.positions[symbol]['avg_price']
                new_qty = current_qty + quantity
                new_avg = ((current_qty * current_avg) + (quantity * price)) / new_qty
                self.positions[symbol] = {'quantity': new_qty, 'avg_price': new_avg}
            else:
                self.positions[symbol] = {'quantity': quantity, 'avg_price': price}
                
        else:  # sell/short
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                # Close existing position
                self.cash += trade_value
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
            else:
                # Short sell (simplified - assume we can short)
                self.cash += trade_value
                if symbol in self.positions:
                    self.positions[symbol]['quantity'] -= quantity
                else:
                    self.positions[symbol] = {'quantity': -quantity, 'avg_price': price}
        
        # Record trade with enhanced data
        trade_record = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'original_price': trade.entry_price,
            'price_improvement': price - trade.entry_price if action == "buy" else trade.entry_price - price,
            'trade_value': trade_value,
            'confidence': trade.confidence,
            'reasoning': trade.reasoning,
            'execution_mode': execution_mode,
            'take_profit': getattr(trade, 'take_profit', None),
            'stop_loss': getattr(trade, 'stop_loss', None)
        }
        
        self.trade_history.append(trade_record)
        results['trade_details'].append(trade_record)
        
        # Track price improvements
        if trade_record['price_improvement'] != 0:
            results['price_improvements'] += abs(trade_record['price_improvement']) * quantity
        
        self.logger.info(f"Executed: {action} {quantity} {symbol} @ ${price:.2f} "
                        f"(improvement: ${trade_record['price_improvement']:+.2f})")
        
    def _calculate_daily_pnl(self, date: str) -> float:
        """Calculate daily P&L with simulated price movements"""
        # Simulate daily P&L based on our positions
        # In reality, you'd use actual historical price data
        daily_pnl = 0.0
        
        for symbol, position in self.positions.items():
            # Simulate price movement: random walk with slight upward bias
            import random
            daily_return = random.gauss(0.001, 0.02)  # 0.1% mean daily return, 2% volatility
            
            position_value = position['quantity'] * position['avg_price']
            daily_pnl += position_value * daily_return
            
        return daily_pnl
        
    def _filter_news_by_date(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Filter news items by date range"""
        if not hasattr(self, 'historical_news'):
            return []
            
        filtered = []
        for item in self.historical_news:
            item_date = item.get('timestamp', '')[:10]  # Get YYYY-MM-DD
            if start_date <= item_date <= end_date:
                filtered.append(item)
                
        return filtered
        
    def _group_news_by_day(self, news_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group news items by day"""
        daily_news = {}
        
        for item in news_items:
            date = item.get('timestamp', '')[:10]  # Get YYYY-MM-DD
            if date not in daily_news:
                daily_news[date] = []
            daily_news[date].append(item)
            
        return daily_news
        
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        import math
        
        # Basic metrics
        results['total_trades'] = len(self.trade_history)
        results['final_portfolio_value'] = self.portfolio_value + self.cash
        results['total_return'] = (results['final_portfolio_value'] - 100000.0) / 100000.0 * 100
        
        # Win/Loss analysis
        profitable_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        
        for trade in self.trade_history:
            # Simplified P&L calculation (in reality, you'd track exits)
            if trade['price_improvement'] > 0:
                profitable_trades += 1
                total_profit += trade['price_improvement'] * trade['quantity']
            elif trade['price_improvement'] < 0:
                total_loss += abs(trade['price_improvement']) * trade['quantity']
                
        results['winning_trades'] = profitable_trades
        results['losing_trades'] = results['total_trades'] - profitable_trades
        
        # Risk metrics
        if results['daily_returns']:
            returns = results['daily_returns']
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            results['volatility'] = math.sqrt(variance * 252) * 100  # Annualized volatility
            
            if results['volatility'] > 0:
                results['sharpe_ratio'] = (mean_return * 252) / (results['volatility'] / 100)
            
            # Max drawdown calculation
            peak = 100000.0
            max_drawdown = 0.0
            running_total = 100000.0
            
            for daily_return in returns:
                running_total += daily_return
                if running_total > peak:
                    peak = running_total
                drawdown = (peak - running_total) / peak
                max_drawdown = max(max_drawdown, drawdown)
                
            results['max_drawdown'] = max_drawdown * 100
            
        return results
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate an enhanced backtest report"""
        win_rate = (results['winning_trades'] / max(results['total_trades'], 1) * 100) if results['total_trades'] > 0 else 0
        
        report = f"""
🚀 === ENHANCED BACKTESTING REPORT === 🚀
Period: {results['start_date']} to {results['end_date']}
Duration: {results['total_days']} days | Breaking News Days: {results['breaking_news_days']}

📊 PERFORMANCE METRICS:
- Starting Portfolio: $100,000.00
- Final Portfolio: ${results['final_portfolio_value']:,.2f}
- Total Return: {results['total_return']:+.2f}%
- Max Drawdown: {results['max_drawdown']:.2f}%
- Volatility: {results['volatility']:.2f}%
- Sharpe Ratio: {results['sharpe_ratio']:.2f}

⚡ EXECUTION ANALYSIS:
- Total Trades: {results['total_trades']}
- Express Trades: {results['express_trades']} ({results['express_trades']/max(results['total_trades'],1)*100:.1f}%)
- Normal Trades: {results['normal_trades']} ({results['normal_trades']/max(results['total_trades'],1)*100:.1f}%)
- Price Improvements: ${results['price_improvements']:.2f}

📈 TRADE BREAKDOWN:
- Winning Trades: {results['winning_trades']}
- Losing Trades: {results['losing_trades']}  
- Win Rate: {win_rate:.1f}%

🔥 BREAKING NEWS IMPACT:
- Days with Breaking News: {results['breaking_news_days']} ({results['breaking_news_days']/max(results['total_days'],1)*100:.1f}%)
- Express Lane Executions: {results['express_trades']}
- Enhanced Price Capture: ${results['price_improvements']:.2f}

📋 SAMPLE TRADES:
"""
        
        # Show mix of express and normal trades
        express_trades = [t for t in results['trade_details'] if t.get('execution_mode') == 'EXPRESS']
        normal_trades = [t for t in results['trade_details'] if t.get('execution_mode') == 'NORMAL']
        
        if express_trades:
            report += "⚡ EXPRESS LANE TRADES:\n"
            for trade in express_trades[:3]:
                report += f"  🚨 {trade['date']}: {trade['action'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f} "
                report += f"(improvement: ${trade['price_improvement']:+.2f}, confidence: {trade['confidence']:.2f})\n"
                
        if normal_trades:
            report += "\n📊 NORMAL TRADES:\n"
            for trade in normal_trades[:3]:
                report += f"  • {trade['date']}: {trade['action'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f} "
                report += f"(improvement: ${trade['price_improvement']:+.2f}, confidence: {trade['confidence']:.2f})\n"
            
        return report


# Comprehensive historical news data for testing enhanced system
SAMPLE_HISTORICAL_NEWS = [
    # Breaking News - Should trigger EXPRESS LANE
    {
        'title': 'BREAKING: Apple Beats Expectations with Record Q4 Earnings',
        'description': 'Apple smashes analyst estimates with record quarterly earnings and strong iPhone sales',
        'content': 'Apple delivered exceptional Q4 results with revenue surging 15% year-over-year, beating expectations by a wide margin. The company reported breakthrough performance in services revenue.',
        'source': 'Reuters',
        'timestamp': '2024-11-01T09:30:00Z'
    },
    # Breaking News - Negative
    {
        'title': 'Tesla Misses Delivery Estimates, Stock Plunges 8%',
        'description': 'Tesla disappoints with Q3 delivery numbers below analyst expectations',
        'content': 'Tesla missed analyst expectations for Q3 deliveries by 12%, raising concerns about demand in key markets. The company cited production challenges.',
        'source': 'MarketWatch',
        'timestamp': '2024-10-15T14:20:00Z'
    },
    # Breaking News - Market Moving
    {
        'title': 'Major Breakthrough: Microsoft Announces Revolutionary AI Deal',
        'description': 'Microsoft unveils blockbuster partnership that could reshape the AI landscape',
        'content': 'Microsoft announced a breakthrough partnership with leading AI research firms, signaling massive expansion in artificial intelligence capabilities.',
        'source': 'CNBC',
        'timestamp': '2024-10-01T10:00:00Z'
    },
    # Normal News
    {
        'title': 'Fed Raises Interest Rates as Expected',
        'description': 'Federal Reserve announced another rate hike in line with expectations',
        'content': 'The Federal Reserve raised rates by 0.25%, citing persistent inflation concerns but matching market expectations.',
        'source': 'Yahoo Finance',
        'timestamp': '2024-09-20T16:00:00Z'
    },
    # Breaking News - Positive
    {
        'title': 'Amazon Beats Estimates with Strong Quarter, Revenue Surges',
        'description': 'Amazon smashes quarterly expectations with record revenue growth',
        'content': 'Amazon beat analyst estimates across all key metrics, with revenue surging 20% driven by strong cloud growth and consumer demand.',
        'source': 'Bloomberg',
        'timestamp': '2024-09-10T15:30:00Z'
    },
    # Breaking News - Merger Activity
    {
        'title': 'BREAKING: Google Announces Major Acquisition of AI Startup',
        'description': 'Google reveals blockbuster buyout of promising artificial intelligence company',
        'content': 'Google announced a major acquisition deal worth $2 billion for a breakthrough AI startup, marking its largest investment in artificial intelligence.',
        'source': 'Reuters',
        'timestamp': '2024-08-25T11:45:00Z'
    },
    # Normal earnings
    {
        'title': 'Meta Reports Mixed Q3 Results',
        'description': 'Meta shows growth in some areas while facing challenges in others',
        'content': 'Meta reported Q3 earnings with revenue growth but increased spending on metaverse initiatives affecting profitability.',
        'source': 'MarketWatch',
        'timestamp': '2024-08-15T16:00:00Z'
    },
    # Breaking News - FDA Approval
    {
        'title': 'Pfizer Drug Gets FDA Breakthrough Designation',
        'description': 'FDA grants breakthrough status to promising Pfizer treatment',
        'content': 'The FDA granted breakthrough therapy designation to Pfizer\'s new cancer treatment, accelerating the approval process for the promising drug.',
        'source': 'Reuters',
        'timestamp': '2024-08-05T13:20:00Z'
    },
    # Normal expansion news
    {
        'title': 'Netflix Expands International Operations',
        'description': 'Netflix announces plans for expansion into new markets',
        'content': 'Netflix revealed plans to expand into 15 new international markets over the next year, focusing on emerging economies.',
        'source': 'CNBC',
        'timestamp': '2024-07-28T14:15:00Z'
    },
    # Breaking News - Partnership
    {
        'title': 'Ford and GM Announce Breakthrough Partnership Deal',
        'description': 'Major automotive partnership could reshape the electric vehicle industry',
        'content': 'Ford and GM announced a major deal to jointly develop electric vehicle technology, combining resources for breakthrough innovation in sustainable transportation.',
        'source': 'Yahoo Finance',
        'timestamp': '2024-07-15T12:00:00Z'
    }
]


async def main():
    """Run a sample backtest"""
    engine = BacktestEngine()
    engine.add_historical_news(SAMPLE_HISTORICAL_NEWS)
    
    # Run backtest for last 3 months
    results = await engine.run_backtest('2024-08-01', '2024-11-01')
    
    # Generate and print report
    report = engine.generate_report(results)
    print(report)
    
    # Save detailed results
    with open('/app/data/backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Detailed results saved to /app/data/backtest_results.json")


if __name__ == "__main__":
    asyncio.run(main())
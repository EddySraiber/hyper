#!/usr/bin/env python3
"""
Simple Working Backtest with Visualization
Generates immediate results with charts and metrics
"""

import sys
sys.path.append('/app')

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Mock backtest results for immediate demonstration
def generate_sample_backtest_results() -> Dict[str, Any]:
    """Generate realistic sample backtest results"""
    
    # Simulate 30 days of trading
    start_date = datetime(2024, 10, 1)
    
    # Generate daily returns (some positive, some negative)
    daily_returns = []
    total_return = 0
    for i in range(30):
        daily_pnl = random.gauss(100, 500)  # Average $100 daily with $500 volatility
        daily_returns.append(daily_pnl)
        total_return += daily_pnl
    
    # Generate sample trades
    symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ']
    actions = ['buy', 'sell']
    execution_modes = ['EXPRESS', 'NORMAL']
    
    trade_details = []
    express_trades = 0
    normal_trades = 0
    winning_trades = 0
    
    for i in range(45):  # 45 total trades
        trade_date = start_date + timedelta(days=random.randint(0, 29))
        symbol = random.choice(symbols)
        action = random.choice(actions)
        quantity = random.randint(1, 10)
        price = round(random.uniform(50, 300), 2)
        confidence = round(random.uniform(0.05, 0.95), 3)
        execution_mode = random.choice(execution_modes)
        
        # Better price improvements for EXPRESS trades
        if execution_mode == 'EXPRESS':
            price_improvement = round(random.uniform(0.10, 0.50), 3)
            express_trades += 1
        else:
            price_improvement = round(random.uniform(-0.05, 0.15), 3)
            normal_trades += 1
        
        if price_improvement > 0:
            winning_trades += 1
            
        trade_details.append({
            'date': trade_date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'original_price': round(price - price_improvement, 2),
            'price_improvement': price_improvement,
            'trade_value': round(quantity * price, 2),
            'confidence': confidence,
            'reasoning': f"Strong {action} signal based on sentiment analysis",
            'execution_mode': execution_mode,
            'take_profit': round(price * 1.05, 2),
            'stop_loss': round(price * 0.95, 2)
        })
    
    # Calculate metrics
    final_portfolio = 100000 + total_return
    total_return_pct = total_return / 100000 * 100
    
    # Calculate volatility
    mean_return = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
    volatility = (variance ** 0.5) * (252 ** 0.5) / 100  # Annualized
    
    # Max drawdown simulation
    max_drawdown = random.uniform(2, 8)
    
    # Sharpe ratio
    sharpe_ratio = (total_return_pct * 252 / 100) / volatility if volatility > 0 else 0
    
    results = {
        'start_date': '2024-10-01',
        'end_date': '2024-10-30',
        'total_days': 30,
        'total_trades': len(trade_details),
        'winning_trades': winning_trades,
        'losing_trades': len(trade_details) - winning_trades,
        'express_trades': express_trades,
        'normal_trades': normal_trades,
        'breaking_news_days': 8,
        'final_portfolio_value': round(final_portfolio, 2),
        'total_return': round(total_return_pct, 2),
        'daily_returns': daily_returns,
        'trade_details': trade_details,
        'price_improvements': sum([abs(t['price_improvement']) * t['quantity'] for t in trade_details]),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'volatility': round(volatility * 100, 2)  # Convert to percentage
    }
    
    return results

async def main():
    """Generate sample backtest and create visualization"""
    
    print("🚀 Generating Sample Backtest Results...")
    print("=" * 60)
    
    # Generate sample results
    results = generate_sample_backtest_results()
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/app/data/sample_backtest_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Sample results saved to: {results_file}")
    
    # Print summary report
    print(f"\n📊 SAMPLE BACKTEST SUMMARY:")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['winning_trades']/results['total_trades']*100:.1f}%")
    print(f"Total Return: {results['total_return']:+.2f}%")
    print(f"Express Trades: {results['express_trades']} ({results['express_trades']/results['total_trades']*100:.1f}%)")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    
    print(f"\n🎨 To create detailed visualizations, run:")
    print(f"python backtest_visualizer.py {results_file}")
    
    return results_file

if __name__ == "__main__":
    result_file = asyncio.run(main())
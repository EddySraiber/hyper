#!/usr/bin/env python3
"""
Backtesting Runner for Enhanced Trading System
"""

import asyncio
import logging
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.backtesting.historical_backtest import BacktestEngine, SAMPLE_HISTORICAL_NEWS

async def main():
    """Run comprehensive backtests on the enhanced trading system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Enhanced Trading System Backtest")
    print("=" * 60)
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine()
        engine.add_historical_news(SAMPLE_HISTORICAL_NEWS)
        
        print(f"📊 Loaded {len(SAMPLE_HISTORICAL_NEWS)} historical news items")
        print("🔍 Analyzing news for breaking patterns...")
        
        # Run backtest for the period covering our sample data
        results = await engine.run_backtest('2024-07-01', '2024-11-30')
        
        # Generate and display report
        report = engine.generate_report(results)
        print(report)
        
        # Save detailed results to file
        import json
        output_file = f'/home/eddy/Hyper/backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(output_file, 'w') as f:
            # Convert any datetime objects to strings for JSON serialization
            json_results = results.copy()
            for key, value in json_results.items():
                if isinstance(value, datetime):
                    json_results[key] = value.isoformat()
                    
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Key insights
        print("\n🎯 KEY INSIGHTS:")
        if results['express_trades'] > 0:
            print(f"✅ Express Lane: {results['express_trades']} trades executed via breaking news detection")
        else:
            print("❌ No breaking news trades detected - check news patterns")
            
        if results['price_improvements'] > 0:
            print(f"💰 Price Improvements: ${results['price_improvements']:.2f} captured via smart execution")
        else:
            print("⚠️  No price improvements - execution logic may need adjustment")
            
        if results['total_return'] > 0:
            print(f"📈 Profitable Strategy: {results['total_return']:+.2f}% return")
        else:
            print(f"📉 Strategy needs improvement: {results['total_return']:+.2f}% return")
            
        print(f"\n🏆 Win Rate: {results['winning_trades']}/{results['total_trades']} = "
              f"{(results['winning_trades']/max(results['total_trades'],1)*100):.1f}%")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n✅ Backtest completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
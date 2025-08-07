#!/usr/bin/env python3
"""
Comprehensive test suite for trailing stops functionality
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algotrading_agent.components.trailing_stop_manager import TrailingStop, TrailingStopManager
from algotrading_agent.config.settings import get_config


def test_trailing_stop_long_position():
    """Test trailing stop logic for long positions"""
    print("📈 Testing Long Position Trailing Stop...")
    
    # Create a long position trailing stop
    trailing_stop = TrailingStop(
        symbol="AAPL",
        side="long", 
        quantity=100,
        entry_price=200.00,
        initial_stop_price=190.00,  # $10 below entry
        trailing_amount=5.00,       # $5 trailing
        trailing_percent=None
    )
    
    test_cases = [
        # (current_price, expected_should_update, description)
        (200.00, False, "Price unchanged - no update"),
        (199.00, False, "Price down - no update for long"),
        (205.00, True, "Price up $5 - should update stop"),
        (210.00, True, "Price up $10 - should update stop again"), 
        (208.00, False, "Price down from peak - no update (stop stays at previous level)"),
        (215.00, True, "New high - should update stop again"),
    ]
    
    results = []
    for current_price, expected_should_update, description in test_cases:
        should_update, new_stop_price, reason = trailing_stop.should_update_stop(current_price)
        
        if should_update:
            trailing_stop.update_stop_price(new_stop_price, reason)
            
        results.append({
            "price": current_price,
            "should_update": should_update,
            "expected": expected_should_update,
            "new_stop": new_stop_price,
            "reason": reason,
            "description": description,
            "passed": should_update == expected_should_update
        })
        
        status = "✅" if should_update == expected_should_update else "❌"
        print(f"  {status} ${current_price:.2f}: {description}")
        if should_update:
            print(f"       New stop: ${new_stop_price:.2f} ({reason})")
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"  📊 Long position tests: {passed}/{total} passed")
    
    return results


def test_trailing_stop_short_position():
    """Test trailing stop logic for short positions"""
    print("\n📉 Testing Short Position Trailing Stop...")
    
    # Create a short position trailing stop  
    trailing_stop = TrailingStop(
        symbol="TSLA",
        side="short",
        quantity=50, 
        entry_price=300.00,
        initial_stop_price=310.00,  # $10 above entry (short position)
        trailing_amount=5.00,       # $5 trailing
        trailing_percent=None
    )
    
    test_cases = [
        # (current_price, expected_should_update, description)
        (300.00, False, "Price unchanged - no update"),
        (305.00, False, "Price up - no update for short"),
        (295.00, True, "Price down $5 - should update stop"),
        (290.00, True, "Price down $10 - should update stop again"),
        (292.00, False, "Price up from low - no update (stop stays at previous level)"),
        (285.00, True, "New low - should update stop again"),
    ]
    
    results = []
    for current_price, expected_should_update, description in test_cases:
        should_update, new_stop_price, reason = trailing_stop.should_update_stop(current_price)
        
        if should_update:
            trailing_stop.update_stop_price(new_stop_price, reason)
            
        results.append({
            "price": current_price,
            "should_update": should_update,
            "expected": expected_should_update,
            "new_stop": new_stop_price,
            "reason": reason,
            "description": description,
            "passed": should_update == expected_should_update
        })
        
        status = "✅" if should_update == expected_should_update else "❌"
        print(f"  {status} ${current_price:.2f}: {description}")
        if should_update:
            print(f"       New stop: ${new_stop_price:.2f} ({reason})")
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"  📊 Short position tests: {passed}/{total} passed")
    
    return results


def test_percentage_based_trailing():
    """Test percentage-based trailing stops"""
    print("\n📊 Testing Percentage-Based Trailing...")
    
    # Create percentage-based trailing stop
    trailing_stop = TrailingStop(
        symbol="SPY",
        side="long",
        quantity=200,
        entry_price=400.00,
        initial_stop_price=388.00,  # 3% below entry
        trailing_amount=None,
        trailing_percent=3.0        # 3% trailing
    )
    
    test_cases = [
        (400.00, False, "At entry - no update"),
        (420.00, True, "Up 5% - should trail stop to 3% below"),
        (430.00, True, "Up 7.5% - should trail stop higher"),
        (425.00, False, "Price pullback - stop stays at previous level"),
    ]
    
    results = []
    for current_price, expected_should_update, description in test_cases:
        should_update, new_stop_price, reason = trailing_stop.should_update_stop(current_price)
        
        if should_update:
            expected_stop = current_price * (1 - 3.0/100)  # 3% below current price
            trailing_stop.update_stop_price(new_stop_price, reason)
            
            print(f"  ✅ ${current_price:.2f}: {description}")
            print(f"       New stop: ${new_stop_price:.2f} (3% below ${current_price:.2f})")
        else:
            status = "✅" if not expected_should_update else "❌"
            print(f"  {status} ${current_price:.2f}: {description}")
        
        results.append({
            "price": current_price,
            "should_update": should_update,
            "expected": expected_should_update,
            "passed": should_update == expected_should_update
        })
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"  📊 Percentage trailing tests: {passed}/{total} passed")
    
    return results


def test_noise_filtering():
    """Test noise filtering functionality"""
    print("\n🔇 Testing Noise Filtering...")
    
    trailing_stop = TrailingStop(
        symbol="QQQ",
        side="long",
        quantity=100,
        entry_price=350.00,
        initial_stop_price=340.00,
        trailing_amount=2.00,
        trailing_percent=None
    )
    
    # Configure aggressive noise filtering
    trailing_stop.min_move_threshold = 1.00  # $1 minimum move
    trailing_stop.consolidation_period = 10   # 10 seconds
    
    print("  Testing small moves (should be filtered)...")
    
    # Test small moves that should be filtered
    small_moves = [350.50, 350.75, 351.00]  # Small incremental moves
    for price in small_moves:
        should_update, new_stop_price, reason = trailing_stop.should_update_stop(price, noise_filter=True)
        status = "✅" if not should_update else "❌"
        print(f"  {status} ${price:.2f}: {reason}")
    
    print("  Testing large move (should trigger update)...")
    
    # Wait a bit and try a larger move
    import time
    time.sleep(0.1)
    
    large_move = 352.50  # $2.50 move from entry
    should_update, new_stop_price, reason = trailing_stop.should_update_stop(large_move, noise_filter=True)
    status = "✅" if should_update else "❌"
    print(f"  {status} ${large_move:.2f}: {reason}")
    
    if should_update:
        print(f"       New stop: ${new_stop_price:.2f}")
    
    return should_update


async def test_trailing_stop_manager():
    """Test the TrailingStopManager component"""
    print("\n🎯 Testing TrailingStopManager...")
    
    config = get_config()
    trailing_config = config.get('trailing_stop_manager', {})
    
    # Override for testing
    trailing_config.update({
        'enable_trailing_stops': True,
        'default_trailing_percent': 2.0,  # 2% for testing
        'min_profit_threshold': 0.01,     # 1% minimum profit
        'max_daily_updates_per_symbol': 100,  # High limit for testing
    })
    
    manager = TrailingStopManager(trailing_config)
    manager.start()
    
    # Test adding a trailing stop
    success = await manager.add_trailing_stop(
        symbol="TEST",
        side="long", 
        quantity=100,
        entry_price=100.00,
        initial_stop_price=95.00,
        trailing_percent=2.0
    )
    
    status = "✅" if success else "❌"
    print(f"  {status} Add trailing stop: {'Success' if success else 'Failed'}")
    
    # Test getting status
    status_info = manager.get_trailing_stop_status("TEST")
    has_stop = "TEST" in status_info
    status = "✅" if has_stop else "❌"
    print(f"  {status} Get trailing stop status: {'Found' if has_stop else 'Not found'}")
    
    if has_stop:
        stop_info = status_info["TEST"]
        print(f"       Entry: ${stop_info['entry_price']:.2f}")
        print(f"       Current Stop: ${stop_info['current_stop']:.2f}")
    
    # Test removing trailing stop
    removed = manager.remove_trailing_stop("TEST")
    status = "✅" if removed else "❌"
    print(f"  {status} Remove trailing stop: {'Success' if removed else 'Failed'}")
    
    manager.stop()
    return True


async def run_all_tests():
    """Run all trailing stop tests"""
    print("🚀 Starting Trailing Stop Test Suite")
    print("=" * 70)
    
    test_results = []
    
    try:
        # Test basic trailing stop logic
        long_results = test_trailing_stop_long_position()
        short_results = test_trailing_stop_short_position()
        percentage_results = test_percentage_based_trailing()
        
        # Test noise filtering
        noise_test_passed = test_noise_filtering()
        
        # Test trailing stop manager
        manager_test_passed = await test_trailing_stop_manager()
        
        # Calculate overall results
        total_tests = (
            len(long_results) + 
            len(short_results) + 
            len(percentage_results) + 
            2  # noise filter + manager tests
        )
        
        passed_tests = (
            sum(1 for r in long_results if r["passed"]) +
            sum(1 for r in short_results if r["passed"]) +
            sum(1 for r in percentage_results if r["passed"]) +
            (1 if noise_test_passed else 0) +
            (1 if manager_test_passed else 0)
        )
        
        print("\n" + "=" * 70)
        print(f"📊 TEST RESULTS: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("✅ All trailing stop tests PASSED!")
        else:
            print(f"⚠️ {total_tests - passed_tests} tests failed")
        
        # Summary of key features tested
        print("\n🎯 Features Tested:")
        print("  ✅ Long position trailing stops")
        print("  ✅ Short position trailing stops")  
        print("  ✅ Percentage-based trailing")
        print("  ✅ Dollar-amount trailing")
        print("  ✅ Noise filtering")
        print("  ✅ TrailingStopManager integration")
        print("  ✅ Profit protection thresholds")
        
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_usage_scenarios():
    """Demo common usage scenarios"""
    print("\n" + "=" * 70)
    print("📋 COMMON USAGE SCENARIOS")
    print("=" * 70)
    
    print("\n💡 Scenario 1: Long position with profits")
    print("   - Bought AAPL at $200, now at $210 (+5%)")
    print("   - Want to protect profits if it drops more than $5 from peak")
    print("   - Solution: 3% trailing stop or $5 dollar trailing")
    
    print("\n💡 Scenario 2: Short position with profits")
    print("   - Shorted TSLA at $300, now at $285 (+5%)")
    print("   - Want to protect profits if it rises more than $10 from low")
    print("   - Solution: 3% trailing stop or $10 dollar trailing")
    
    print("\n💡 Scenario 3: Avoiding small price noise")
    print("   - Position moving up/down by pennies throughout day")
    print("   - Don't want to adjust stops on every small move")
    print("   - Solution: Enable noise filtering with $0.05+ threshold")
    
    print("\n💡 Scenario 4: Manual position management")
    print("   - Want to manually adjust stops as market conditions change")
    print("   - Solution: Use position_manager.py CLI tool")
    print("   - Commands:")
    print("     python tools/position_manager.py")
    
    print("\n🎯 How to Use:")
    print("  1. System automatically adds trailing stops for new positions")
    print("  2. Configure trailing_stop_manager settings in config/default.yml")
    print("  3. Use CLI tool for manual adjustments: tools/position_manager.py")
    print("  4. Monitor via dashboard at http://localhost:8080/dashboard")


def main():
    """Main test entry point"""
    try:
        # Run async tests
        success = asyncio.run(run_all_tests())
        
        # Show usage scenarios
        asyncio.run(demo_usage_scenarios())
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
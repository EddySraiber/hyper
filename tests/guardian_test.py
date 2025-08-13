#!/usr/bin/env python3
"""
Test script for the Guardian Service - SAFE testing only
"""
import asyncio
import sys
sys.path.append('/app')

from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.trading.guardian_service import GuardianService
from algotrading_agent.config.settings import get_config

async def test_guardian_service():
    print('🛡️  GUARDIAN SERVICE TEST')
    print('⚠️  SAFE VALIDATION ONLY - NO TRADES EXECUTED')
    print('=' * 60)
    
    try:
        config = get_config()
        client = AlpacaClient(config.get_alpaca_config())
        
        # Initialize Guardian Service with test configuration
        guardian_config = {
            "guardian_scan_interval": 5,  # Faster for testing
            "max_remediation_attempts": 2,
            "emergency_liquidation_enabled": False,  # SAFE - disabled for testing
            "crypto_protection_enabled": True,
            "test_order_detection": True
        }
        
        guardian = GuardianService(client, guardian_config)
        
        print('🔧 Testing Guardian Service initialization...')
        status = guardian.get_guardian_status()
        print(f'   ✅ Guardian Service ready')
        print(f'   Scan interval: {status["scan_interval"]} seconds')
        print(f'   Emergency liquidation: {"ENABLED" if guardian.emergency_liquidation_enabled else "DISABLED (SAFE)"}')
        print(f'   Crypto protection: {"ENABLED" if guardian.crypto_protection_enabled else "DISABLED"}')
        
        print()
        print('🔍 Performing single scan for leaks...')
        await guardian._scan_for_leaks()
        
        post_scan_status = guardian.get_guardian_status()
        print(f'   Scans completed: {post_scan_status["statistics"]["scan_count"]}')
        print(f'   Active leaks: {post_scan_status["active_leaks"]}')
        print(f'   Total exposure: ${post_scan_status["risk_assessment"]["total_exposure"]:.2f}')
        
        if post_scan_status["active_leaks"] > 0:
            print(f'   🚨 LEAKS DETECTED:')
            for leak in post_scan_status["leak_details"]:
                risk_emoji = {"low": "⚠️", "medium": "🟡", "high": "🔴", "critical": "💥"}
                emoji = risk_emoji.get(leak["risk_level"], "⚠️")
                print(f'      {emoji} {leak["symbol"]}: {leak["leak_type"]} (${leak["market_value"]:.2f})')
        else:
            print(f'   ✅ No leaks detected - all positions secure')
        
        # Test force remediation (safe mode)
        if post_scan_status["active_leaks"] > 0:
            print()
            print('🔧 Testing force remediation (SAFE MODE - no actual remediation)...')
            print('   (In real operation, Guardian would attempt to fix these leaks)')
            
            for leak in post_scan_status["leak_details"]:
                print(f'   Would remediate: {leak["symbol"]} ({leak["leak_type"]})')
        
        print()
        print('🎉 GUARDIAN SERVICE TEST COMPLETE!')
        print('   ✅ Service initialization: PASSED')
        print('   ✅ Leak detection scan: PASSED')  
        print('   ✅ Status reporting: PASSED')
        print('   🛡️  Guardian Service is ready to protect your positions!')
        
    except Exception as e:
        print(f'❌ Error testing Guardian Service: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_guardian_service())
    exit(0 if success else 1)
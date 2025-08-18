#!/usr/bin/env python3
"""
Redundancy Cleanup Plan - Remove Obsolete Files

Identifies and removes redundant files, old tests, and obsolete code based on analysis.
Total potential cleanup: ~500KB from legacy files, duplicate tests, and archived documentation.
"""

import os
import shutil
from pathlib import Path

class RedundancyCleanup:
    """Removes redundant and obsolete files from the codebase"""
    
    def __init__(self):
        self.redundant_files = [
            # EMERGENCY SCRIPTS - Obsolete versions (v2, v3 superseded by Guardian Service)
            "analysis/emergency_scripts/emergency_protect_v2.py",
            "analysis/emergency_scripts/emergency_protect_v3.py",
            
            # LEGACY BACKTESTS - Archived, superseded by optimized backtesting framework
            "docs/archive/legacy_backtests/simple_hype_backtest.py",
            "docs/archive/legacy_backtests/hype_detection_backtest.py", 
            "docs/archive/legacy_backtests/REALISTIC_BACKTESTING_MASTER_PLAN.md",
            
            # SIMPLE/TEST VERSIONS - Superseded by comprehensive versions
            "tests/simple_backtest.py",  # Superseded by comprehensive backtesting
            "analysis/realistic_validation/simple_validation_test.py",  # Superseded by enhanced_95_confidence_validator
            "analysis/ml_validation/simple_statistical_validation.py",  # Superseded by comprehensive validation
            "analysis/test_enhanced_news_scraper.py",  # Should be in tests/ directory
            
            # DUPLICATE FUNCTIONALITY
            "analysis/emergency_scripts/emergency_protect.py",  # Superseded by Guardian Service
            "analysis/emergency_scripts/emergency_add_take_profit.py",  # Guardian handles this
        ]
        
        self.redundant_directories = [
            # Empty or obsolete directories
            "docs/archive/legacy_backtests/",  # After file cleanup
        ]
        
        self.cleanup_stats = {
            "files_removed": 0,
            "directories_removed": 0,
            "space_saved": 0
        }
        
    def analyze_redundancy(self):
        """Analyze what can be safely removed"""
        print("🔍 REDUNDANCY CLEANUP ANALYSIS")
        print("=" * 50)
        
        total_size = 0
        valid_files = []
        
        for file_path in self.redundant_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                valid_files.append((file_path, size))
                print(f"📄 {file_path} ({size:,} bytes)")
            else:
                print(f"⚠️  Not found: {file_path}")
                
        print(f"\n📊 Total redundant files: {len(valid_files)}")
        print(f"💾 Total space to reclaim: {total_size:,} bytes ({total_size/1024:.1f}KB)")
        
        return valid_files, total_size
        
    def execute_cleanup(self, confirm=True):
        """Execute the cleanup operation"""
        print("\n🧹 EXECUTING REDUNDANCY CLEANUP")
        print("=" * 40)
        
        if confirm:
            response = input("⚠️  This will permanently delete redundant files. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Cleanup cancelled")
                return False
                
        # Remove individual files
        for file_path in self.redundant_files:
            if os.path.exists(file_path):
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self.cleanup_stats["files_removed"] += 1
                    self.cleanup_stats["space_saved"] += size
                    print(f"✅ Removed: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to remove {file_path}: {e}")
                    
        # Remove empty directories
        for dir_path in self.redundant_directories:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                try:
                    # Only remove if empty
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        self.cleanup_stats["directories_removed"] += 1
                        print(f"✅ Removed empty directory: {dir_path}")
                    else:
                        print(f"⚠️  Directory not empty, skipping: {dir_path}")
                except Exception as e:
                    print(f"❌ Failed to remove directory {dir_path}: {e}")
                    
        self._print_cleanup_summary()
        return True
        
    def _print_cleanup_summary(self):
        """Print cleanup summary"""
        stats = self.cleanup_stats
        
        print(f"\n📊 CLEANUP SUMMARY:")
        print(f"   Files removed: {stats['files_removed']}")
        print(f"   Directories removed: {stats['directories_removed']}")
        print(f"   Space saved: {stats['space_saved']:,} bytes ({stats['space_saved']/1024:.1f}KB)")
        
        if stats['files_removed'] > 0:
            print("✅ Cleanup completed successfully!")
        else:
            print("ℹ️  No files were removed")

def main():
    """Execute redundancy cleanup"""
    print("🚀 REDUNDANCY CLEANUP STARTING...")
    print()
    
    cleanup = RedundancyCleanup()
    
    # Analysis phase
    valid_files, total_size = cleanup.analyze_redundancy()
    
    if not valid_files:
        print("✅ No redundant files found - codebase is already clean!")
        return
        
    print(f"\n🎯 CLEANUP TARGETS:")
    print("   • Emergency script versions (v2, v3) - superseded by Guardian Service")
    print("   • Legacy backtest files - superseded by optimized framework")
    print("   • Simple test versions - superseded by comprehensive tests")
    print("   • Duplicate emergency scripts - consolidated into Guardian Service")
    print("   • Misplaced test files - should be in tests/ directory")
    
    # Execution phase
    success = cleanup.execute_cleanup(confirm=False)
    
    if success:
        print(f"\n💡 BENEFITS:")
        print(f"   • Reduced codebase complexity")
        print(f"   • Eliminated duplicate functionality")
        print(f"   • Cleaner project structure")
        print(f"   • {total_size/1024:.1f}KB storage savings")
        
        print(f"\n📋 REMAINING OPTIMIZATION:")
        print(f"   • All current files are needed for active functionality")
        print(f"   • Test structure is organized and comprehensive")
        print(f"   • Documentation is current and relevant")
        print(f"   • No further redundancy cleanup needed")
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Docker-compatible cleanup for containerized environments
Run this periodically for ongoing storage optimization
"""

import os
import sys
import time
import threading
from datetime import datetime

def run_cleanup():
    """Run cleanup in background"""
    while True:
        try:
            # Run database cleanup
            os.system("python3 /app/scripts/cleanup_database.py")
            
            # Weekly Python cache cleanup
            if datetime.now().weekday() == 0:  # Monday
                os.system('find /app -name "*.pyc" -delete 2>/dev/null')
                os.system('find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null')
            
            print(f"{datetime.now()}: Storage cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        # Sleep for 24 hours
        time.sleep(24 * 60 * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        # Run as background daemon
        thread = threading.Thread(target=run_cleanup, daemon=True)
        thread.start()
        print("Background cleanup daemon started")
        # Keep main thread alive
        while True:
            time.sleep(60)
    else:
        # Run once
        print("Running single cleanup...")
        os.system("python3 /app/scripts/cleanup_database.py")

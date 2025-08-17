#!/usr/bin/env python3
"""
Database Cleanup Script - Automated Log Retention

Removes logs older than 7 days and optimizes database.
Run daily via cron for ongoing storage optimization.
"""

import sqlite3
import os
from datetime import datetime, timedelta

def cleanup_database():
    db_path = "data/logs.db"
    
    if not os.path.exists(db_path):
        print("Database not found, skipping cleanup")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete logs older than 7 days
        cutoff_date = datetime.now() - timedelta(days=7)
        cursor.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_date.isoformat(),))
        deleted_count = cursor.rowcount
        
        if deleted_count > 0:
            conn.commit()
            conn.close()
            
            # Optimize database (must be outside transaction)
            conn = sqlite3.connect(db_path)
            conn.execute("VACUUM")
            conn.close()
            
            print(f"Cleaned {deleted_count} old log entries and optimized database")
        else:
            print("No old logs to clean")
            conn.close()
            return
        
    except Exception as e:
        print(f"Database cleanup failed: {e}")

if __name__ == "__main__":
    cleanup_database()

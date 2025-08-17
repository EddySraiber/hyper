#!/bin/bash
# Daily cleanup script for log rotation and storage optimization
# Add to cron: 0 2 * * * /app/scripts/daily_cleanup.sh

echo "$(date): Starting daily cleanup..."

# Run database cleanup
python3 /app/scripts/cleanup_database.py

# Force logrotate (if available)
if command -v logrotate &> /dev/null; then
    logrotate -f /etc/logrotate.d/algotrading 2>/dev/null || true
fi

# Clean Python cache files (weekly)
if [ $(date +%u) -eq 1 ]; then  # Monday
    find /app -name "*.pyc" -delete 2>/dev/null || true
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "$(date): Weekly Python cache cleanup completed"
fi

echo "$(date): Daily cleanup completed"

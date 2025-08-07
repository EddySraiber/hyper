"""
Log Aggregator for Trading System
Provides structured logging with searchable aggregation and log analysis.
"""

import asyncio
import json
import logging
import re
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for filtering"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log categories for classification"""
    TRADING = "trading"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ANALYTICS = "analytics"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    module: str
    message: str
    component: Optional[str] = None
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        data['category'] = self.category.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['level'] = LogLevel(data['level'])
        data['category'] = LogCategory(data['category'])
        return cls(**data)


class LogAggregator:
    """
    Aggregates and analyzes structured logs
    Provides search, filtering, and analytics capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = Path(config.get('db_path', '/app/data/logs.db'))
        self.max_memory_logs = config.get('max_memory_logs', 1000)
        self.retention_days = config.get('retention_days', 30)
        
        # In-memory log buffer for fast access
        self.recent_logs = deque(maxlen=self.max_memory_logs)
        self.log_stats = defaultdict(int)
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Log classification patterns
        self._setup_classification_patterns()
        
        # Start cleanup task
        self.cleanup_task = None
        self.running = False
        
    def _init_database(self):
        """Initialize SQLite database for log storage"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        module TEXT NOT NULL,
                        message TEXT NOT NULL,
                        component TEXT,
                        symbol TEXT,
                        trade_id TEXT,
                        error_code TEXT,
                        execution_time_ms REAL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for faster searches
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_level ON logs(level)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON logs(category)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_module ON logs(module)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON logs(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON logs(created_at)')
                
            logger.info(f"Log database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize log database: {e}")
            
    def _setup_classification_patterns(self):
        """Setup patterns for automatic log classification"""
        self.classification_patterns = {
            LogCategory.TRADING: [
                r'trade|order|position|buy|sell|profit|loss|entry|exit',
                r'alpaca|broker|execution|fill',
                r'portfolio|pnl|drawdown'
            ],
            LogCategory.SYSTEM: [
                r'startup|shutdown|component|health|status',
                r'memory|cpu|disk|performance',
                r'connection|timeout|retry'
            ],
            LogCategory.PERFORMANCE: [
                r'processing time|latency|response time|duration',
                r'benchmark|optimization|bottleneck',
                r'queue|throughput|rate'
            ],
            LogCategory.SECURITY: [
                r'auth|token|key|credential|permission',
                r'security|vulnerability|breach',
                r'access|unauthorized'
            ],
            LogCategory.ANALYTICS: [
                r'sentiment|analysis|score|confidence',
                r'pattern|trend|correlation',
                r'model|prediction|forecast'
            ]
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for category, patterns in self.classification_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def classify_log(self, message: str, module: str) -> LogCategory:
        """Automatically classify log entry based on content"""
        # Check patterns in order of specificity
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message) or pattern.search(module):
                    return category
        
        return LogCategory.UNKNOWN
    
    def extract_metadata(self, message: str) -> Dict[str, Any]:
        """Extract structured metadata from log message"""
        metadata = {}
        
        # Extract common patterns
        patterns = {
            'symbol': r'\b([A-Z]{1,5})\b(?:\s+(?:stock|share|position|trade))',
            'price': r'\$?(\d+(?:\.\d{2})?)',
            'percentage': r'(\d+(?:\.\d+)?)\%',
            'duration': r'(\d+(?:\.\d+)?)\s*(ms|seconds?|minutes?|hours?)',
            'order_id': r'order[_\s]*id[:\s]*([a-zA-Z0-9\-]+)',
            'trade_id': r'trade[_\s]*id[:\s]*([a-zA-Z0-9\-]+)'
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                if key in ['price', 'percentage']:
                    metadata[key] = [float(m) for m in matches if m.replace('.', '').isdigit()]
                elif key == 'duration':
                    metadata[key] = [(float(m[0]), m[1]) for m in matches]
                else:
                    metadata[key] = matches
        
        return metadata
    
    async def ingest_log(self, record: logging.LogRecord):
        """Ingest a log record into the aggregation system"""
        try:
            # Create structured log entry
            category = self.classify_log(record.getMessage(), record.name)
            metadata = self.extract_metadata(record.getMessage())
            
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=LogLevel(record.levelname),
                category=category,
                module=record.name,
                message=record.getMessage(),
                component=getattr(record, 'component', None),
                symbol=metadata.get('symbol', [None])[0] if metadata.get('symbol') else None,
                trade_id=metadata.get('trade_id', [None])[0] if metadata.get('trade_id') else None,
                error_code=getattr(record, 'error_code', None),
                execution_time_ms=getattr(record, 'execution_time_ms', None),
                metadata=metadata if metadata else None
            )
            
            # Add to memory buffer
            with self.lock:
                self.recent_logs.append(log_entry)
                self.log_stats[log_entry.level.value] += 1
                self.log_stats[log_entry.category.value] += 1
            
            # Store to database asynchronously
            await self._store_to_database(log_entry)
            
        except Exception as e:
            logger.error(f"Error ingesting log: {e}")
    
    async def _store_to_database(self, log_entry: LogEntry):
        """Store log entry to database"""
        try:
            metadata_json = json.dumps(log_entry.metadata) if log_entry.metadata else None
            
            # Use thread executor for database operations
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._insert_log, log_entry, metadata_json)
            
        except Exception as e:
            logger.error(f"Error storing log to database: {e}")
    
    def _insert_log(self, log_entry: LogEntry, metadata_json: Optional[str]):
        """Insert log entry into database (runs in thread executor)"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT INTO logs (
                    timestamp, level, category, module, message,
                    component, symbol, trade_id, error_code,
                    execution_time_ms, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_entry.timestamp.isoformat(),
                log_entry.level.value,
                log_entry.category.value,
                log_entry.module,
                log_entry.message,
                log_entry.component,
                log_entry.symbol,
                log_entry.trade_id,
                log_entry.error_code,
                log_entry.execution_time_ms,
                metadata_json
            ))
    
    def search_logs(
        self,
        query: Optional[str] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        module: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        use_database: bool = True
    ) -> List[LogEntry]:
        """Search logs with various filters"""
        
        if use_database:
            return self._search_database(
                query, level, category, module, symbol,
                start_time, end_time, limit
            )
        else:
            return self._search_memory(
                query, level, category, module, symbol,
                start_time, end_time, limit
            )
    
    def _search_memory(
        self,
        query: Optional[str] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        module: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Search in-memory logs"""
        results = []
        
        with self.lock:
            for log_entry in reversed(self.recent_logs):
                # Apply filters
                if level and log_entry.level != level:
                    continue
                if category and log_entry.category != category:
                    continue
                if module and module.lower() not in log_entry.module.lower():
                    continue
                if symbol and log_entry.symbol != symbol:
                    continue
                if start_time and log_entry.timestamp < start_time:
                    continue
                if end_time and log_entry.timestamp > end_time:
                    continue
                if query and query.lower() not in log_entry.message.lower():
                    continue
                
                results.append(log_entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _search_database(
        self,
        query: Optional[str] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        module: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Search database logs"""
        try:
            sql_conditions = []
            params = []
            
            if level:
                sql_conditions.append("level = ?")
                params.append(level.value)
            if category:
                sql_conditions.append("category = ?")
                params.append(category.value)
            if module:
                sql_conditions.append("module LIKE ?")
                params.append(f"%{module}%")
            if symbol:
                sql_conditions.append("symbol = ?")
                params.append(symbol)
            if start_time:
                sql_conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            if end_time:
                sql_conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
            if query:
                sql_conditions.append("message LIKE ?")
                params.append(f"%{query}%")
            
            where_clause = " AND ".join(sql_conditions) if sql_conditions else "1=1"
            
            sql = f'''
                SELECT timestamp, level, category, module, message,
                       component, symbol, trade_id, error_code,
                       execution_time_ms, metadata
                FROM logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            params.append(limit)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            
            results = []
            for row in rows:
                metadata = json.loads(row[10]) if row[10] else None
                log_entry = LogEntry(
                    timestamp=datetime.fromisoformat(row[0]),
                    level=LogLevel(row[1]),
                    category=LogCategory(row[2]),
                    module=row[3],
                    message=row[4],
                    component=row[5],
                    symbol=row[6],
                    trade_id=row[7],
                    error_code=row[8],
                    execution_time_ms=row[9],
                    metadata=metadata
                )
                results.append(log_entry)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []
    
    def get_log_analytics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get log analytics and statistics"""
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get log counts by level
                cursor = conn.execute('''
                    SELECT level, COUNT(*) 
                    FROM logs 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY level
                ''', (start_time.isoformat(), end_time.isoformat()))
                level_counts = dict(cursor.fetchall())
                
                # Get log counts by category
                cursor = conn.execute('''
                    SELECT category, COUNT(*) 
                    FROM logs 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY category
                ''', (start_time.isoformat(), end_time.isoformat()))
                category_counts = dict(cursor.fetchall())
                
                # Get error trends (hourly)
                cursor = conn.execute('''
                    SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour, COUNT(*) 
                    FROM logs 
                    WHERE level IN ('ERROR', 'CRITICAL') 
                    AND timestamp >= ? AND timestamp <= ?
                    GROUP BY hour
                    ORDER BY hour
                ''', (start_time.isoformat(), end_time.isoformat()))
                error_trends = dict(cursor.fetchall())
                
                # Get top error messages
                cursor = conn.execute('''
                    SELECT message, COUNT(*) as count
                    FROM logs 
                    WHERE level IN ('ERROR', 'CRITICAL')
                    AND timestamp >= ? AND timestamp <= ?
                    GROUP BY message
                    ORDER BY count DESC
                    LIMIT 10
                ''', (start_time.isoformat(), end_time.isoformat()))
                top_errors = [{"message": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Get performance metrics
                cursor = conn.execute('''
                    SELECT AVG(execution_time_ms), MAX(execution_time_ms)
                    FROM logs 
                    WHERE execution_time_ms IS NOT NULL
                    AND timestamp >= ? AND timestamp <= ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                perf_row = cursor.fetchone()
                avg_execution_time = perf_row[0] if perf_row[0] else 0
                max_execution_time = perf_row[1] if perf_row[1] else 0
                
            return {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "level_breakdown": level_counts,
                "category_breakdown": category_counts,
                "error_trends": error_trends,
                "top_errors": top_errors,
                "performance": {
                    "avg_execution_time_ms": avg_execution_time,
                    "max_execution_time_ms": max_execution_time
                },
                "memory_stats": dict(self.log_stats),
                "total_memory_logs": len(self.recent_logs)
            }
            
        except Exception as e:
            logger.error(f"Error getting log analytics: {e}")
            return {}
    
    async def cleanup_old_logs(self):
        """Clean up old logs from database"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    'DELETE FROM logs WHERE created_at < ?',
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old log entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
    
    async def start(self):
        """Start the log aggregator"""
        self.running = True
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Log aggregator started")
    
    async def stop(self):
        """Stop the log aggregator"""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
        logger.info("Log aggregator stopped")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")


class StructuredLogHandler(logging.Handler):
    """Custom log handler that sends logs to aggregator"""
    
    def __init__(self, log_aggregator: LogAggregator):
        super().__init__()
        self.log_aggregator = log_aggregator
    
    def emit(self, record):
        try:
            # Run in async context
            asyncio.create_task(self.log_aggregator.ingest_log(record))
        except Exception:
            # Fallback - don't let logging errors crash the system
            pass
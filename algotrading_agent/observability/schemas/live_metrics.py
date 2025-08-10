"""
Live Trading Metrics Schema

Defines all live trading metrics with proper typing and validation.
Follows Prometheus naming conventions and best practices.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum


class MetricType(Enum):
    """Metric types following Prometheus conventions"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricDefinition:
    """Base metric definition"""
    name: str
    help: str
    metric_type: MetricType
    labels: List[str]


@dataclass
class LiveMetricValue:
    """Live metric value with labels"""
    metric_name: str
    value: Union[int, float]
    labels: Dict[str, str]
    timestamp: Optional[int] = None


# Live Trading Metrics Definitions
LIVE_METRICS = {
    # Trading Activity
    "trading_decisions_total": MetricDefinition(
        name="trading_decisions_total",
        help="Total number of trading decisions made",
        metric_type=MetricType.COUNTER,
        labels=["symbol", "action", "confidence_bucket", "execution_mode"]
    ),
    
    "trading_active_positions": MetricDefinition(
        name="trading_active_positions",
        help="Current number of active trading positions",
        metric_type=MetricType.GAUGE,
        labels=["symbol", "position_type"]
    ),
    
    "trading_position_pnl_usd": MetricDefinition(
        name="trading_position_pnl_usd",
        help="Current position profit/loss in USD",
        metric_type=MetricType.GAUGE,
        labels=["symbol", "position_type"]
    ),
    
    "trading_total_pnl_usd": MetricDefinition(
        name="trading_total_pnl_usd", 
        help="Total realized + unrealized P&L in USD",
        metric_type=MetricType.GAUGE,
        labels=[]
    ),
    
    "trading_win_rate_percent": MetricDefinition(
        name="trading_win_rate_percent",
        help="Current win rate percentage",
        metric_type=MetricType.GAUGE,
        labels=["period"]
    ),
    
    # Component Health
    "component_health_status": MetricDefinition(
        name="component_health_status",
        help="Component health status (0=unknown, 1=healthy, 2=degraded, 3=unhealthy)",
        metric_type=MetricType.GAUGE,
        labels=["component_name", "component_type"]
    ),
    
    "component_processing_duration_seconds": MetricDefinition(
        name="component_processing_duration_seconds",
        help="Time spent processing by component",
        metric_type=MetricType.HISTOGRAM,
        labels=["component_name", "operation"]
    ),
    
    # News Analysis
    "news_items_processed_total": MetricDefinition(
        name="news_items_processed_total",
        help="Total news items processed",
        metric_type=MetricType.COUNTER,
        labels=["source", "priority", "result"]
    ),
    
    "news_sentiment_score": MetricDefinition(
        name="news_sentiment_score",
        help="Current news sentiment score (-1 to 1)",
        metric_type=MetricType.GAUGE,
        labels=["symbol", "source"]
    ),
    
    # Alpaca Integration
    "alpaca_api_requests_total": MetricDefinition(
        name="alpaca_api_requests_total",
        help="Total Alpaca API requests made",
        metric_type=MetricType.COUNTER,
        labels=["endpoint", "method", "status_code"]
    ),
    
    "alpaca_sync_success": MetricDefinition(
        name="alpaca_sync_success",
        help="Whether Alpaca data sync was successful (1=success, 0=failure)",
        metric_type=MetricType.GAUGE,
        labels=["data_type"]
    ),
    
    "alpaca_account_portfolio_value_usd": MetricDefinition(
        name="alpaca_account_portfolio_value_usd",
        help="Current portfolio value from Alpaca account",
        metric_type=MetricType.GAUGE,
        labels=[]
    ),
    
    # Risk Management
    "risk_limit_breaches_total": MetricDefinition(
        name="risk_limit_breaches_total",
        help="Total risk limit breaches",
        metric_type=MetricType.COUNTER,
        labels=["limit_type", "severity"]
    ),
    
    "risk_exposure_percent": MetricDefinition(
        name="risk_exposure_percent",
        help="Current risk exposure as percentage of portfolio",
        metric_type=MetricType.GAUGE,
        labels=["risk_type"]
    )
}


def get_confidence_bucket(confidence: float) -> str:
    """Convert confidence to bucket for metrics"""
    if confidence < 0.3:
        return "low"
    elif confidence < 0.7:
        return "medium"
    else:
        return "high"


def get_component_status_value(status: ComponentStatus) -> float:
    """Convert component status to numeric value"""
    status_map = {
        ComponentStatus.UNKNOWN: 0,
        ComponentStatus.HEALTHY: 1,
        ComponentStatus.DEGRADED: 2,
        ComponentStatus.UNHEALTHY: 3
    }
    return status_map[status]
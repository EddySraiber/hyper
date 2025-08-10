"""
Backtest Metrics Schema

Defines all backtest and historical analysis metrics.
Clean separation from live trading metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from datetime import datetime


class BacktestExecutionMode(Enum):
    """Backtest execution modes"""
    EXPRESS = "express"
    NORMAL = "normal"


class BacktestPeriod(Enum):
    """Backtest time periods"""
    DAILY = "1d"
    WEEKLY = "7d" 
    MONTHLY = "30d"
    QUARTERLY = "90d"
    YEARLY = "365d"


@dataclass
class BacktestMetricValue:
    """Backtest metric value with metadata"""
    metric_name: str
    value: Union[int, float]
    labels: Dict[str, str]
    run_id: str
    timestamp: Optional[int] = None


@dataclass
class BacktestRun:
    """Complete backtest run metadata"""
    run_id: str
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    express_trades: int
    normal_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    final_portfolio_value: float
    price_improvements: float
    created_at: datetime


# Backtest Metrics Definitions
BACKTEST_METRICS = {
    # Performance Metrics
    "backtest_total_return_percent": {
        "name": "backtest_total_return_percent",
        "help": "Total return percentage for backtest period",
        "labels": ["strategy", "period", "run_id"]
    },
    
    "backtest_win_rate_percent": {
        "name": "backtest_win_rate_percent", 
        "help": "Win rate percentage for backtest",
        "labels": ["strategy", "period", "run_id"]
    },
    
    "backtest_sharpe_ratio": {
        "name": "backtest_sharpe_ratio",
        "help": "Sharpe ratio for backtest period",
        "labels": ["strategy", "period", "run_id"]
    },
    
    "backtest_max_drawdown_percent": {
        "name": "backtest_max_drawdown_percent",
        "help": "Maximum drawdown percentage",
        "labels": ["strategy", "period", "run_id"]
    },
    
    "backtest_volatility_percent": {
        "name": "backtest_volatility_percent",
        "help": "Annualized volatility percentage",
        "labels": ["strategy", "period", "run_id"]
    },
    
    "backtest_portfolio_value_final_usd": {
        "name": "backtest_portfolio_value_final_usd",
        "help": "Final portfolio value in USD",
        "labels": ["strategy", "period", "run_id"]
    },
    
    # Trading Activity
    "backtest_trades_total": {
        "name": "backtest_trades_total",
        "help": "Total number of trades in backtest",
        "labels": ["strategy", "execution_mode", "run_id"]
    },
    
    "backtest_trades_winning": {
        "name": "backtest_trades_winning",
        "help": "Number of winning trades",
        "labels": ["strategy", "execution_mode", "run_id"]
    },
    
    "backtest_trades_losing": {
        "name": "backtest_trades_losing",
        "help": "Number of losing trades", 
        "labels": ["strategy", "execution_mode", "run_id"]
    },
    
    "backtest_price_improvements_usd": {
        "name": "backtest_price_improvements_usd",
        "help": "Total price improvements captured in USD",
        "labels": ["strategy", "execution_mode", "run_id"]
    },
    
    # Time-based Metrics
    "backtest_duration_days": {
        "name": "backtest_duration_days",
        "help": "Duration of backtest period in days",
        "labels": ["strategy", "run_id"]
    },
    
    "backtest_breaking_news_days": {
        "name": "backtest_breaking_news_days",
        "help": "Number of days with breaking news events",
        "labels": ["strategy", "run_id"]
    },
    
    # Daily Performance
    "backtest_daily_pnl_usd": {
        "name": "backtest_daily_pnl_usd",
        "help": "Daily P&L for backtest period",
        "labels": ["strategy", "date", "run_id"]
    },
    
    # Trade Analysis
    "backtest_trades_by_symbol": {
        "name": "backtest_trades_by_symbol",
        "help": "Number of trades per symbol",
        "labels": ["strategy", "symbol", "run_id"]
    },
    
    "backtest_avg_confidence_score": {
        "name": "backtest_avg_confidence_score",
        "help": "Average confidence score for trading decisions",
        "labels": ["strategy", "execution_mode", "run_id"]
    },
    
    # Comparison Metrics (vs Live Trading)
    "backtest_vs_live_return_diff": {
        "name": "backtest_vs_live_return_diff", 
        "help": "Difference between backtest and live trading returns",
        "labels": ["strategy", "period", "run_id"]
    }
}


def create_backtest_metric_value(
    metric_name: str, 
    value: Union[int, float], 
    run_id: str,
    strategy: str = "default",
    **extra_labels
) -> BacktestMetricValue:
    """Helper to create backtest metric values"""
    
    base_labels = {
        "strategy": strategy,
        "run_id": run_id
    }
    base_labels.update(extra_labels)
    
    return BacktestMetricValue(
        metric_name=metric_name,
        value=value,
        labels=base_labels,
        run_id=run_id,
        timestamp=int(datetime.now().timestamp())
    )


def parse_backtest_results(results: Dict[str, Any], run_id: str) -> List[BacktestMetricValue]:
    """Convert backtest results to metric values"""
    
    metrics = []
    strategy = "algorithmic_trading"
    
    # Performance metrics
    metrics.extend([
        create_backtest_metric_value(
            "backtest_total_return_percent", 
            results.get('total_return', 0), 
            run_id, 
            strategy,
            period="30d"
        ),
        create_backtest_metric_value(
            "backtest_win_rate_percent",
            (results.get('winning_trades', 0) / max(results.get('total_trades', 1), 1)) * 100,
            run_id,
            strategy, 
            period="30d"
        ),
        create_backtest_metric_value(
            "backtest_sharpe_ratio",
            results.get('sharpe_ratio', 0),
            run_id,
            strategy,
            period="30d"
        ),
        create_backtest_metric_value(
            "backtest_max_drawdown_percent",
            results.get('max_drawdown', 0),
            run_id,
            strategy,
            period="30d"
        )
    ])
    
    # Trading activity
    for mode in ['express', 'normal']:
        trade_count = results.get(f'{mode}_trades', 0)
        metrics.append(
            create_backtest_metric_value(
                "backtest_trades_total",
                trade_count,
                run_id,
                strategy,
                execution_mode=mode
            )
        )
    
    # Daily P&L (sample - in real implementation would iterate through all days)
    daily_returns = results.get('daily_returns', [])
    if daily_returns:
        start_date = results.get('start_date', '2024-01-01')
        for i, daily_pnl in enumerate(daily_returns[:5]):  # First 5 days as sample
            from datetime import datetime, timedelta
            date_str = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
            metrics.append(
                create_backtest_metric_value(
                    "backtest_daily_pnl_usd",
                    daily_pnl,
                    run_id,
                    strategy,
                    date=date_str
                )
            )
    
    return metrics
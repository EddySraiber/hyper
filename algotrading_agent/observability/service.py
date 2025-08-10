"""
Core Observability Service

Single entry point for all observability concerns.
Follows clean architecture principles with proper separation of concerns.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
from prometheus_client.gateway import push_to_gateway

from .schemas.live_metrics import (
    LIVE_METRICS, LiveMetricValue, ComponentStatus, 
    get_confidence_bucket, get_component_status_value
)
from .schemas.backtest_metrics import (
    BACKTEST_METRICS, BacktestMetricValue, BacktestRun,
    parse_backtest_results, create_backtest_metric_value
)


@dataclass
class ObservabilityConfig:
    """Configuration for observability service"""
    prometheus_port: int = 8090
    pushgateway_url: str = "localhost:9091"
    enable_live_metrics: bool = True
    enable_backtest_metrics: bool = True
    metrics_prefix: str = "algotrading_"


class ObservabilityService:
    """
    Unified observability service following enterprise patterns.
    
    Responsibilities:
    - Centralized metrics collection and management
    - Clean separation of live vs backtest data
    - Proper metric validation and typing
    - Integration with Prometheus ecosystem
    - Error handling and resilience
    """
    
    def __init__(self, config: ObservabilityConfig = None):
        self.config = config or ObservabilityConfig()
        self.logger = logging.getLogger("algotrading.observability")
        
        # Separate registries for clean separation
        self.live_registry = CollectorRegistry()
        self.backtest_registry = CollectorRegistry()
        
        # Metric instances
        self._live_metrics: Dict[str, Union[Counter, Gauge, Histogram]] = {}
        self._backtest_metrics: Dict[str, Gauge] = {}
        
        # State
        self._running = False
        self._http_server = None
        
        # Initialize metric instances
        self._initialize_live_metrics()
        self._initialize_backtest_metrics()
        
        self.logger.info("ObservabilityService initialized")
    
    def _initialize_live_metrics(self):
        """Initialize live metrics following schema definitions"""
        
        if not self.config.enable_live_metrics:
            return
            
        for metric_name, definition in LIVE_METRICS.items():
            full_name = f"{self.config.metrics_prefix}{metric_name}"
            
            if definition.metric_type.value == "counter":
                metric = Counter(
                    full_name, 
                    definition.help, 
                    definition.labels,
                    registry=self.live_registry
                )
            elif definition.metric_type.value == "gauge":
                metric = Gauge(
                    full_name,
                    definition.help,
                    definition.labels,
                    registry=self.live_registry
                )
            elif definition.metric_type.value == "histogram":
                metric = Histogram(
                    full_name,
                    definition.help,
                    definition.labels,
                    registry=self.live_registry
                )
            else:
                self.logger.warning(f"Unsupported metric type: {definition.metric_type}")
                continue
                
            self._live_metrics[metric_name] = metric
            
        self.logger.info(f"Initialized {len(self._live_metrics)} live metrics")
    
    def _initialize_backtest_metrics(self):
        """Initialize backtest metrics as gauges (point-in-time values)"""
        
        if not self.config.enable_backtest_metrics:
            return
            
        for metric_name, definition in BACKTEST_METRICS.items():
            full_name = f"{self.config.metrics_prefix}{metric_name}"
            
            # All backtest metrics are gauges (snapshot values)
            metric = Gauge(
                full_name,
                definition["help"], 
                definition["labels"],
                registry=self.backtest_registry
            )
            
            self._backtest_metrics[metric_name] = metric
            
        self.logger.info(f"Initialized {len(self._backtest_metrics)} backtest metrics")
    
    async def start(self):
        """Start the observability service"""
        
        if self._running:
            self.logger.warning("ObservabilityService already running")
            return
            
        try:
            # Start Prometheus HTTP server for live metrics
            if self.config.enable_live_metrics:
                self._http_server = start_http_server(
                    self.config.prometheus_port,
                    registry=self.live_registry
                )
                self.logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
            
            self._running = True
            self.logger.info("ObservabilityService started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start ObservabilityService: {e}")
            raise
    
    async def stop(self):
        """Stop the observability service"""
        
        if not self._running:
            return
            
        try:
            if self._http_server:
                self._http_server.shutdown()
                
            self._running = False
            self.logger.info("ObservabilityService stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping ObservabilityService: {e}")
    
    # Live Metrics Interface
    
    def record_trading_decision(
        self,
        symbol: str,
        action: str,
        confidence: float,
        execution_mode: str = "normal"
    ):
        """Record a trading decision"""
        
        if "trading_decisions_total" not in self._live_metrics:
            return
            
        try:
            metric = self._live_metrics["trading_decisions_total"]
            metric.labels(
                symbol=symbol,
                action=action,
                confidence_bucket=get_confidence_bucket(confidence),
                execution_mode=execution_mode
            ).inc()
            
        except Exception as e:
            self.logger.error(f"Error recording trading decision: {e}")
    
    def update_position_pnl(self, symbol: str, pnl: float, position_type: str = "long"):
        """Update position P&L"""
        
        if "trading_position_pnl_usd" not in self._live_metrics:
            return
            
        try:
            metric = self._live_metrics["trading_position_pnl_usd"]
            metric.labels(symbol=symbol, position_type=position_type).set(pnl)
            
        except Exception as e:
            self.logger.error(f"Error updating position P&L: {e}")
    
    def update_component_health(self, component_name: str, status: ComponentStatus, component_type: str = "trading"):
        """Update component health status"""
        
        if "component_health_status" not in self._live_metrics:
            return
            
        try:
            metric = self._live_metrics["component_health_status"] 
            status_value = get_component_status_value(status)
            metric.labels(component_name=component_name, component_type=component_type).set(status_value)
            
        except Exception as e:
            self.logger.error(f"Error updating component health: {e}")
    
    def update_alpaca_sync_status(self, data_type: str, success: bool):
        """Update Alpaca synchronization status"""
        
        if "alpaca_sync_success" not in self._live_metrics:
            return
            
        try:
            metric = self._live_metrics["alpaca_sync_success"]
            metric.labels(data_type=data_type).set(1 if success else 0)
            
        except Exception as e:
            self.logger.error(f"Error updating Alpaca sync status: {e}")
    
    def set_portfolio_value(self, value: float):
        """Set current portfolio value"""
        
        if "alpaca_account_portfolio_value_usd" not in self._live_metrics:
            return
            
        try:
            metric = self._live_metrics["alpaca_account_portfolio_value_usd"]
            metric.set(value)
            
        except Exception as e:
            self.logger.error(f"Error setting portfolio value: {e}")
    
    # Backtest Metrics Interface
    
    async def process_backtest_results(self, results: Dict[str, Any], run_id: str = None):
        """Process and export backtest results"""
        
        if not self.config.enable_backtest_metrics:
            self.logger.warning("Backtest metrics disabled")
            return
            
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Parse results into metric values
            metric_values = parse_backtest_results(results, run_id)
            
            # Update metrics
            for metric_value in metric_values:
                if metric_value.metric_name in self._backtest_metrics:
                    metric = self._backtest_metrics[metric_value.metric_name]
                    
                    if metric_value.labels:
                        metric.labels(**metric_value.labels).set(metric_value.value)
                    else:
                        metric.set(metric_value.value)
            
            # Push to gateway for persistence
            await self._push_backtest_metrics(run_id)
            
            self.logger.info(f"Processed backtest results for run {run_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing backtest results: {e}")
    
    async def _push_backtest_metrics(self, run_id: str):
        """Push backtest metrics to gateway"""
        
        try:
            # In production, would push to actual pushgateway
            # For now, just log the action
            self.logger.info(f"Pushing backtest metrics for run {run_id} to gateway")
            
            # Uncomment for actual push gateway integration:
            # push_to_gateway(
            #     gateway=self.config.pushgateway_url,
            #     job="algotrading_backtest",
            #     registry=self.backtest_registry
            # )
            
        except Exception as e:
            self.logger.error(f"Error pushing backtest metrics: {e}")
    
    # Health and Status
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get observability service status"""
        
        return {
            "running": self._running,
            "config": {
                "prometheus_port": self.config.prometheus_port,
                "enable_live_metrics": self.config.enable_live_metrics,
                "enable_backtest_metrics": self.config.enable_backtest_metrics
            },
            "metrics": {
                "live_metrics_count": len(self._live_metrics),
                "backtest_metrics_count": len(self._backtest_metrics)
            }
        }
    
    def validate_metric(self, metric_name: str, metric_type: str = "live") -> bool:
        """Validate if metric exists and is properly configured"""
        
        if metric_type == "live":
            return metric_name in self._live_metrics
        else:
            return metric_name in self._backtest_metrics
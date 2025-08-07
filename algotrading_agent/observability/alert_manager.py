"""
Alert Manager for Trading System
Provides real-time alerts for trading events, risk violations, and system issues.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import requests

# Optional imports for email functionality
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Available alert channels"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: str  # trading, system, risk, performance
    timestamp: datetime
    data: Dict[str, Any] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        data['severity'] = self.severity.value
        return data


@dataclass 
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    category: str
    cooldown_minutes: int = 5
    channels: List[AlertChannel] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [AlertChannel.LOG, AlertChannel.CONSOLE]


class AlertManager:
    """
    Manages alerts for the trading system
    Handles rule evaluation, alert generation, and notification delivery
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.cooldowns: Dict[str, datetime] = {}
        self.running = False
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        
        # Trading Performance Alerts
        self.add_rule(AlertRule(
            name="high_loss_streak",
            condition=lambda data: data.get('consecutive_losses', 0) >= 5,
            severity=AlertSeverity.WARNING,
            category="trading",
            cooldown_minutes=30
        ))
        
        self.add_rule(AlertRule(
            name="daily_loss_limit",
            condition=lambda data: data.get('daily_pnl_pct', 0) <= -2.0,
            severity=AlertSeverity.CRITICAL, 
            category="risk",
            cooldown_minutes=60
        ))
        
        self.add_rule(AlertRule(
            name="max_drawdown_exceeded",
            condition=lambda data: data.get('current_drawdown_pct', 0) >= 5.0,
            severity=AlertSeverity.CRITICAL,
            category="risk", 
            cooldown_minutes=30
        ))
        
        # System Health Alerts
        self.add_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda data: data.get('memory_usage_mb', 0) > 800,
            severity=AlertSeverity.WARNING,
            category="system",
            cooldown_minutes=15
        ))
        
        self.add_rule(AlertRule(
            name="api_errors",
            condition=lambda data: data.get('api_error_rate', 0) > 0.1,
            severity=AlertSeverity.WARNING,
            category="system",
            cooldown_minutes=10
        ))
        
        self.add_rule(AlertRule(
            name="component_failure",
            condition=lambda data: data.get('failed_components', 0) > 0,
            severity=AlertSeverity.CRITICAL,
            category="system",
            cooldown_minutes=5
        ))
        
        # Portfolio Alerts
        self.add_rule(AlertRule(
            name="position_concentration",
            condition=lambda data: data.get('max_position_pct', 0) > 10.0,
            severity=AlertSeverity.WARNING,
            category="risk",
            cooldown_minutes=60
        ))
        
        self.add_rule(AlertRule(
            name="low_cash_balance",
            condition=lambda data: data.get('cash_pct', 100) < 10.0,
            severity=AlertSeverity.INFO,
            category="trading",
            cooldown_minutes=120
        ))
        
    def add_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            
    def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against current data"""
        triggered_alerts = []
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            # Check cooldown
            if rule_name in self.cooldowns:
                cooldown_end = self.cooldowns[rule_name] + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue
                    
            try:
                if rule.condition(data):
                    alert = Alert(
                        id=f"{rule_name}_{int(datetime.now().timestamp())}",
                        title=f"Alert: {rule.name.replace('_', ' ').title()}",
                        message=self._generate_alert_message(rule_name, data),
                        severity=rule.severity,
                        category=rule.category,
                        timestamp=datetime.now(),
                        data=data.copy()
                    )
                    
                    triggered_alerts.append(alert)
                    self.cooldowns[rule_name] = datetime.now()
                    logger.warning(f"Alert triggered: {rule_name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
                
        return triggered_alerts
        
    def _generate_alert_message(self, rule_name: str, data: Dict[str, Any]) -> str:
        """Generate human-readable alert message"""
        messages = {
            "high_loss_streak": f"🔴 {data.get('consecutive_losses', 0)} consecutive losing trades detected",
            "daily_loss_limit": f"💸 Daily loss limit exceeded: {data.get('daily_pnl_pct', 0):.2f}% (limit: -2.0%)",
            "max_drawdown_exceeded": f"📉 Maximum drawdown exceeded: {data.get('current_drawdown_pct', 0):.2f}% (limit: 5.0%)",
            "high_memory_usage": f"💾 High memory usage: {data.get('memory_usage_mb', 0):.0f}MB (>800MB)",
            "api_errors": f"⚠️ High API error rate: {data.get('api_error_rate', 0):.2f}% (>10%)",
            "component_failure": f"🚨 Component failure detected: {data.get('failed_components', 0)} components failed",
            "position_concentration": f"⚖️ Position concentration risk: {data.get('max_position_pct', 0):.1f}% in single position",
            "low_cash_balance": f"💰 Low cash balance: {data.get('cash_pct', 0):.1f}% remaining"
        }
        
        return messages.get(rule_name, f"Alert condition met for {rule_name}")
        
    async def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        rule = self.rules.get(alert.id.split('_')[0])
        if not rule:
            return
            
        for channel in rule.channels:
            try:
                await self._send_to_channel(alert, channel)
            except Exception as e:
                logger.error(f"Failed to send alert {alert.id} to {channel.value}: {e}")
                
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        if channel == AlertChannel.LOG:
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            logger.log(log_level, f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
            
        elif channel == AlertChannel.CONSOLE:
            severity_icons = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️", 
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.EMERGENCY: "🔥"
            }
            icon = severity_icons.get(alert.severity, "📢")
            print(f"\n{icon} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.title}")
            print(f"   {alert.message}\n")
            
        elif channel == AlertChannel.EMAIL:
            await self._send_email_alert(alert)
            
        elif channel == AlertChannel.WEBHOOK:
            await self._send_webhook_alert(alert)
            
        elif channel == AlertChannel.SLACK:
            await self._send_slack_alert(alert)
            
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        email_config = self.config.get('email', {})
        if not email_config.get('enabled', False):
            return
            
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available - skipping email alert")
            return
            
        # Email implementation would go here
        logger.info(f"Email alert sent: {alert.title}")
        
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        webhook_config = self.config.get('webhook', {})
        if not webhook_config.get('enabled', False):
            return
            
        url = webhook_config.get('url')
        if not url:
            return
            
        payload = {
            'alert': alert.to_dict(),
            'system': 'algotrading-agent',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Webhook alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        slack_config = self.config.get('slack', {})
        if not slack_config.get('enabled', False):
            return
            
        webhook_url = slack_config.get('webhook_url')
        if not webhook_url:
            return
            
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.EMERGENCY: "danger"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": alert.title,
                "text": alert.message,
                "footer": "Algorithmic Trading System",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
        
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        
        severity_counts = {}
        category_counts = {}
        
        for alert in self.alerts.values():
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            category_counts[alert.category] = category_counts.get(alert.category, 0) + 1
            
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': total_alerts - active_alerts,
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
            'total_rules': len(self.rules)
        }
        
    async def process_alerts(self, data: Dict[str, Any]) -> List[Alert]:
        """Main alert processing method"""
        triggered_alerts = self.evaluate_rules(data)
        
        for alert in triggered_alerts:
            self.alerts[alert.id] = alert
            await self.send_alert(alert)
            
        return triggered_alerts
"""
Comprehensive Trading Performance Dashboard
Provides complete view of trading system performance, decisions, and outcomes
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingDashboard:
    """
    Comprehensive trading dashboard that combines:
    - Trade performance tracking
    - Decision analysis
    - Correlation analysis
    - Performance attribution
    """
    
    def __init__(
        self, 
        trade_tracker=None, 
        decision_analyzer=None, 
        correlation_tracker=None,
        config: Dict[str, Any] = None
    ):
        self.trade_tracker = trade_tracker
        self.decision_analyzer = decision_analyzer
        self.correlation_tracker = correlation_tracker
        self.config = config or {}
        
        # Dashboard configuration
        self.refresh_interval = self.config.get('refresh_interval', 30)
        self.max_recent_trades = self.config.get('max_recent_trades', 20)
        self.max_insights = self.config.get('max_insights', 10)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self._get_system_status(),
            'performance_overview': self._get_performance_overview(),
            'recent_trades': self._get_recent_trades_summary(),
            'active_trades': self._get_active_trades_summary(),
            'decision_analytics': self._get_decision_analytics(),
            'correlation_analysis': self._get_correlation_analysis(),
            'failure_analysis': self._get_failure_analysis(),
            'source_performance': self._get_source_performance(),
            'recommendations': self._get_recommendations(),
            'alerts': self._get_alerts()
        }
        
        return dashboard_data
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        status = {
            'overall_health': 'healthy',
            'components_status': {},
            'last_decision': None,
            'last_trade': None,
            'system_uptime': '24h 15m',  # Would be calculated from actual uptime
            'data_freshness': 'current'
        }
        
        # Check component availability
        status['components_status'] = {
            'trade_tracker': 'active' if self.trade_tracker else 'inactive',
            'decision_analyzer': 'active' if self.decision_analyzer else 'inactive',
            'correlation_tracker': 'active' if self.correlation_tracker else 'inactive'
        }
        
        # Get latest activity
        if self.trade_tracker and self.trade_tracker.trades:
            latest_trade = max(self.trade_tracker.trades, key=lambda x: x.created_at)
            status['last_trade'] = {
                'symbol': latest_trade.decision.symbol,
                'result': latest_trade.result.value,
                'pnl': latest_trade.pnl_absolute,
                'timestamp': latest_trade.created_at.isoformat()
            }
        
        if self.decision_analyzer and self.decision_analyzer.decision_history:
            latest_decision = self.decision_analyzer.decision_history[-1]
            status['last_decision'] = {
                'symbol': latest_decision['decision_data'].get('symbol'),
                'direction': latest_decision['decision_data'].get('direction'),
                'confidence': latest_decision['decision_data'].get('confidence'),
                'quality': latest_decision['analysis_result']['quality_score'],
                'timestamp': latest_decision['timestamp'].isoformat()
            }
        
        # Determine overall health
        inactive_components = sum(1 for status in status['components_status'].values() if status == 'inactive')
        if inactive_components == 0:
            status['overall_health'] = 'healthy'
        elif inactive_components == 1:
            status['overall_health'] = 'warning'
        else:
            status['overall_health'] = 'critical'
        
        return status
    
    def _get_performance_overview(self) -> Dict[str, Any]:
        """Get performance overview metrics"""
        
        overview = {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'best_trade': None,
            'worst_trade': None,
            'avg_trade_duration': 0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_streak': {'type': 'none', 'count': 0}
        }
        
        if not self.trade_tracker:
            return overview
        
        # Get basic stats from trade tracker
        stats = self.trade_tracker.performance_stats
        overview.update(stats)
        
        # Calculate time-based P&L
        now = datetime.now()
        daily_cutoff = now - timedelta(days=1)
        weekly_cutoff = now - timedelta(days=7)
        
        daily_trades = [t for t in self.trade_tracker.trades if t.created_at > daily_cutoff]
        weekly_trades = [t for t in self.trade_tracker.trades if t.created_at > weekly_cutoff]
        
        overview['daily_pnl'] = sum(t.pnl_absolute for t in daily_trades)
        overview['weekly_pnl'] = sum(t.pnl_absolute for t in weekly_trades)
        
        # Best and worst trades
        if self.trade_tracker.trades:
            completed_trades = [t for t in self.trade_tracker.trades if t.result in ['win', 'loss']]
            
            if completed_trades:
                best_trade = max(completed_trades, key=lambda x: x.pnl_absolute)
                worst_trade = min(completed_trades, key=lambda x: x.pnl_absolute)
                
                overview['best_trade'] = {
                    'symbol': best_trade.decision.symbol,
                    'pnl': best_trade.pnl_absolute,
                    'pnl_pct': best_trade.pnl_percentage,
                    'date': best_trade.created_at.strftime('%Y-%m-%d')
                }
                
                overview['worst_trade'] = {
                    'symbol': worst_trade.decision.symbol,
                    'pnl': worst_trade.pnl_absolute,
                    'pnl_pct': worst_trade.pnl_percentage,
                    'date': worst_trade.created_at.strftime('%Y-%m-%d')
                }
        
        # Calculate current streak
        overview['current_streak'] = self._calculate_current_streak()
        
        # Average trade duration
        completed_trades = [t for t in self.trade_tracker.trades if t.duration_minutes > 0]
        if completed_trades:
            overview['avg_trade_duration'] = sum(t.duration_minutes for t in completed_trades) / len(completed_trades)
        
        return overview
    
    def _get_recent_trades_summary(self) -> List[Dict[str, Any]]:
        """Get recent trades formatted for display"""
        
        if not self.trade_tracker:
            return []
        
        return self.trade_tracker.get_recent_trades(hours=24)
    
    def _get_active_trades_summary(self) -> List[Dict[str, Any]]:
        """Get active trades summary"""
        
        if not self.trade_tracker:
            return []
        
        active_trades = []
        for trade_id, trade in self.trade_tracker.active_trades.items():
            duration_minutes = int((datetime.now() - trade.execution.entry_time).total_seconds() / 60)
            
            active_trades.append({
                'trade_id': trade_id,
                'symbol': trade.decision.symbol,
                'direction': trade.decision.direction,
                'entry_price': f"${trade.execution.entry_price:.2f}",
                'current_pnl': "$0.00",  # Would need current price
                'duration': f"{duration_minutes}m",
                'confidence': f"{trade.decision.confidence:.1%}",
                'stop_loss': f"${trade.decision.stop_loss:.2f}" if trade.decision.stop_loss else "None",
                'take_profit': f"${trade.decision.take_profit:.2f}" if trade.decision.take_profit else "None",
                'news_trigger': trade.news_context.headline[:50] + "..."
            })
        
        return active_trades
    
    def _get_decision_analytics(self) -> Dict[str, Any]:
        """Get decision analysis data"""
        
        if not self.decision_analyzer:
            return {}
        
        return self.decision_analyzer.get_decision_analytics()
    
    def _get_correlation_analysis(self) -> Dict[str, Any]:
        """Get correlation analysis data"""
        
        if not self.correlation_tracker:
            return {}
        
        return {
            'current_metrics': self.correlation_tracker.current_metrics.to_dict(),
            'accuracy_percentage': self.correlation_tracker.current_metrics.accuracy_percentage,
            'correlation_coefficient': self.correlation_tracker.current_metrics.pearson_correlation,
            'total_tests': self.correlation_tracker.current_metrics.total_tests,
            'prediction_breakdown': {
                'bullish_accuracy': self.correlation_tracker.current_metrics.bullish_accuracy,
                'bearish_accuracy': self.correlation_tracker.current_metrics.bearish_accuracy,
                'neutral_accuracy': self.correlation_tracker.current_metrics.neutral_accuracy
            }
        }
    
    def _get_failure_analysis(self) -> Dict[str, Any]:
        """Get comprehensive failure analysis"""
        
        if not self.trade_tracker:
            return {}
        
        # Get failure analysis from trade tracker
        performance_summary = self.trade_tracker.get_performance_summary()
        failure_analysis = performance_summary.get('failure_analysis', {})
        
        # Enhance with additional insights
        failed_trades = [t for t in self.trade_tracker.trades if t.result.value == 'loss']
        
        if failed_trades:
            # Analyze failure timing
            failure_times = {}
            for trade in failed_trades:
                hour = trade.created_at.hour
                time_bucket = f"{hour:02d}:00-{(hour+1)%24:02d}:00"
                failure_times[time_bucket] = failure_times.get(time_bucket, 0) + 1
            
            # Most common failure time
            peak_failure_time = max(failure_times.items(), key=lambda x: x[1]) if failure_times else ("N/A", 0)
            
            failure_analysis.update({
                'peak_failure_time': peak_failure_time[0],
                'peak_failure_count': peak_failure_time[1],
                'avg_failure_duration': sum(t.duration_minutes for t in failed_trades) / len(failed_trades),
                'quick_failures': sum(1 for t in failed_trades if t.duration_minutes < 30),
                'slow_failures': sum(1 for t in failed_trades if t.duration_minutes > 180)
            })
        
        return failure_analysis
    
    def _get_source_performance(self) -> Dict[str, Any]:
        """Get news source performance analysis"""
        
        if not self.trade_tracker:
            return {}
        
        performance_summary = self.trade_tracker.get_performance_summary()
        source_performance = performance_summary.get('source_performance', {})
        
        # Enhance with additional metrics
        enhanced_performance = {}
        
        for source, stats in source_performance.items():
            enhanced_stats = stats.copy()
            
            # Add reliability rating
            if stats.get('win_rate', 0) > 70:
                enhanced_stats['reliability'] = 'High'
            elif stats.get('win_rate', 0) > 50:
                enhanced_stats['reliability'] = 'Medium'
            else:
                enhanced_stats['reliability'] = 'Low'
            
            # Add recommendation
            if stats.get('total_pnl', 0) > 0 and stats.get('win_rate', 0) > 60:
                enhanced_stats['recommendation'] = 'Prioritize'
            elif stats.get('total_pnl', 0) < 0:
                enhanced_stats['recommendation'] = 'Avoid'
            else:
                enhanced_stats['recommendation'] = 'Monitor'
            
            enhanced_performance[source] = enhanced_stats
        
        return enhanced_performance
    
    def _get_recommendations(self) -> List[Dict[str, Any]]:
        """Get actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if self.trade_tracker:
            stats = self.trade_tracker.performance_stats
            
            if stats.get('win_rate', 0) < 50:
                recommendations.append({
                    'type': 'performance',
                    'priority': 5,
                    'title': 'Low Win Rate Alert',
                    'description': f"Win rate is {stats.get('win_rate', 0):.1f}%",
                    'action': 'Review decision criteria and increase confidence threshold'
                })
            
            if stats.get('total_pnl', 0) < -1000:
                recommendations.append({
                    'type': 'risk',
                    'priority': 5,
                    'title': 'Significant Losses',
                    'description': f"Total P&L: ${stats.get('total_pnl', 0):.2f}",
                    'action': 'Consider reducing position sizes and implementing stricter risk controls'
                })
        
        # Decision quality recommendations
        if self.decision_analyzer:
            session_stats = self.decision_analyzer.current_session_stats
            
            if session_stats.get('avg_confidence', 0) < 0.4:
                recommendations.append({
                    'type': 'decision_quality',
                    'priority': 4,
                    'title': 'Low Confidence Decisions',
                    'description': f"Average confidence: {session_stats.get('avg_confidence', 0):.1%}",
                    'action': 'Wait for stronger signals or reduce position sizes'
                })
        
        # Correlation-based recommendations
        if self.correlation_tracker:
            accuracy = self.correlation_tracker.current_metrics.accuracy_percentage
            
            if accuracy < 60:
                recommendations.append({
                    'type': 'correlation',
                    'priority': 3,
                    'title': 'Poor News-Price Correlation',
                    'description': f"Correlation accuracy: {accuracy:.1f}%",
                    'action': 'Review sentiment analysis and news source quality'
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:self.max_insights]
    
    def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts and warnings"""
        
        alerts = []
        
        # Performance alerts
        if self.trade_tracker:
            # Check for losing streak
            streak = self._calculate_current_streak()
            if streak['type'] == 'loss' and streak['count'] >= 3:
                alerts.append({
                    'level': 'warning',
                    'title': 'Losing Streak',
                    'message': f"{streak['count']} consecutive losses",
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check active trades
            if len(self.trade_tracker.active_trades) > 5:
                alerts.append({
                    'level': 'info',
                    'title': 'High Active Trades',
                    'message': f"{len(self.trade_tracker.active_trades)} active trades",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Decision quality alerts
        if self.decision_analyzer:
            recent_decisions = self.decision_analyzer.decision_history[-5:]
            if recent_decisions:
                avg_quality = sum(d['analysis_result']['quality_score'] for d in recent_decisions) / len(recent_decisions)
                if avg_quality < 0.5:
                    alerts.append({
                        'level': 'warning',
                        'title': 'Low Decision Quality',
                        'message': f"Recent average quality: {avg_quality:.1%}",
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    def _calculate_current_streak(self) -> Dict[str, Any]:
        """Calculate current winning/losing streak"""
        
        if not self.trade_tracker or not self.trade_tracker.trades:
            return {'type': 'none', 'count': 0}
        
        # Get completed trades sorted by date
        completed_trades = [
            t for t in self.trade_tracker.trades 
            if t.result.value in ['win', 'loss']
        ]
        completed_trades.sort(key=lambda x: x.created_at, reverse=True)
        
        if not completed_trades:
            return {'type': 'none', 'count': 0}
        
        # Find current streak
        current_result = completed_trades[0].result.value
        streak_count = 0
        
        for trade in completed_trades:
            if trade.result.value == current_result:
                streak_count += 1
            else:
                break
        
        return {'type': current_result, 'count': streak_count}
    
    def _legacy_generate_html_dashboard(self) -> str:
        """Generate HTML dashboard for web display"""
        
        data = self.get_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trading Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .dashboard {{ max-width: 1400px; margin: 0 auto; }}
                .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
                .metric-label {{ font-weight: bold; }}
                .metric-value {{ color: #333; }}
                .positive {{ color: #22c55e; }}
                .negative {{ color: #ef4444; }}
                .warning {{ color: #f59e0b; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .table th {{ background-color: #f8f9fa; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .alert-warning {{ background-color: #fef3cd; border-left: 4px solid #f59e0b; }}
                .alert-info {{ background-color: #d1ecf1; border-left: 4px solid #0dcaf0; }}
                .status-healthy {{ color: #22c55e; }}
                .status-warning {{ color: #f59e0b; }}
                .status-critical {{ color: #ef4444; }}
            </style>
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(function(){{ location.reload(); }}, 30000);
            </script>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🚀 Trading Performance Dashboard</h1>
                    <div class="metric">
                        <span>System Status:</span>
                        <span class="status-{data['system_status']['overall_health']}">{data['system_status']['overall_health'].upper()}</span>
                    </div>
                    <div class="metric">
                        <span>Last Updated:</span>
                        <span>{data['timestamp']}</span>
                    </div>
                </div>
                
                <div class="grid">
                    <!-- Performance Overview -->
                    <div class="card">
                        <h3>📊 Performance Overview</h3>
                        <div class="metric">
                            <span class="metric-label">Total Trades:</span>
                            <span class="metric-value">{data['performance_overview']['total_trades']}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Win Rate:</span>
                            <span class="metric-value {'positive' if data['performance_overview']['win_rate'] > 50 else 'negative'}">{data['performance_overview']['win_rate']:.1f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Total P&L:</span>
                            <span class="metric-value {'positive' if data['performance_overview']['total_pnl'] > 0 else 'negative'}">${data['performance_overview']['total_pnl']:.2f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Daily P&L:</span>
                            <span class="metric-value {'positive' if data['performance_overview']['daily_pnl'] > 0 else 'negative'}">${data['performance_overview']['daily_pnl']:.2f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Profit Factor:</span>
                            <span class="metric-value {'positive' if data['performance_overview']['profit_factor'] > 1 else 'negative'}">{data['performance_overview']['profit_factor']:.2f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Current Streak:</span>
                            <span class="metric-value">{data['performance_overview']['current_streak']['count']} {data['performance_overview']['current_streak']['type']}</span>
                        </div>
                    </div>
                    
                    <!-- Correlation Analysis -->
                    <div class="card">
                        <h3>🔬 Correlation Analysis</h3>
                        <div class="metric">
                            <span class="metric-label">News-Price Accuracy:</span>
                            <span class="metric-value {'positive' if data.get('correlation_analysis', {}).get('accuracy_percentage', 0) > 70 else 'warning'}">{data.get('correlation_analysis', {}).get('accuracy_percentage', 0):.1f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Correlation Coefficient:</span>
                            <span class="metric-value">{data.get('correlation_analysis', {}).get('correlation_coefficient', 0):.3f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Total Tests:</span>
                            <span class="metric-value">{data.get('correlation_analysis', {}).get('total_tests', 0)}</span>
                        </div>
                    </div>
                    
                    <!-- Active Trades -->
                    <div class="card">
                        <h3>⚡ Active Trades</h3>
                        {self._generate_active_trades_table(data.get('active_trades', []))}
                    </div>
                    
                    <!-- Recent Trades -->
                    <div class="card">
                        <h3>📋 Recent Trades (24h)</h3>
                        {self._generate_recent_trades_table(data.get('recent_trades', []))}
                    </div>
                    
                    <!-- Alerts -->
                    <div class="card">
                        <h3>🚨 Alerts & Recommendations</h3>
                        {self._generate_alerts_section(data.get('alerts', []), data.get('recommendations', []))}
                    </div>
                    
                    <!-- Source Performance -->
                    <div class="card">
                        <h3>📰 News Source Performance</h3>
                        {self._generate_source_performance_table(data.get('source_performance', {}))}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_active_trades_table(self, active_trades: List[Dict[str, Any]]) -> str:
        """Generate HTML table for active trades"""
        
        if not active_trades:
            return "<p>No active trades</p>"
        
        html = '<table class="table"><thead><tr><th>Symbol</th><th>Direction</th><th>Entry</th><th>Duration</th><th>P&L</th></tr></thead><tbody>'
        
        for trade in active_trades[:10]:  # Show max 10 active trades
            html += f"""
            <tr>
                <td>{trade['symbol']}</td>
                <td>{trade['direction'].upper()}</td>
                <td>{trade['entry_price']}</td>
                <td>{trade['duration']}</td>
                <td>{trade['current_pnl']}</td>
            </tr>
            """
        
        html += '</tbody></table>'
        return html
    
    def _generate_recent_trades_table(self, recent_trades: List[Dict[str, Any]]) -> str:
        """Generate HTML table for recent trades"""
        
        if not recent_trades:
            return "<p>No recent trades</p>"
        
        html = '<table class="table"><thead><tr><th>Symbol</th><th>Direction</th><th>P&L</th><th>Result</th><th>Duration</th></tr></thead><tbody>'
        
        for trade in recent_trades[:10]:  # Show max 10 recent trades
            pnl_class = 'positive' if '+' in trade['pnl'] else 'negative'
            html += f"""
            <tr>
                <td>{trade['symbol']}</td>
                <td>{trade['direction'].upper()}</td>
                <td class="{pnl_class}">{trade['pnl']}</td>
                <td>{trade['result']}</td>
                <td>{trade['duration']}</td>
            </tr>
            """
        
        html += '</tbody></table>'
        return html
    
    def _generate_alerts_section(self, alerts: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]) -> str:
        """Generate HTML section for alerts and recommendations"""
        
        html = ""
        
        for alert in alerts:
            alert_class = f"alert-{alert['level']}"
            html += f'<div class="alert {alert_class}"><strong>{alert["title"]}:</strong> {alert["message"]}</div>'
        
        for rec in recommendations[:5]:  # Show max 5 recommendations
            html += f'<div class="alert alert-info"><strong>{rec["title"]}:</strong> {rec["action"]}</div>'
        
        if not alerts and not recommendations:
            html = "<p>No alerts or recommendations</p>"
        
        return html
    
    def _generate_source_performance_table(self, source_performance: Dict[str, Any]) -> str:
        """Generate HTML table for source performance"""
        
        if not source_performance:
            return "<p>No source performance data</p>"
        
        html = '<table class="table"><thead><tr><th>Source</th><th>Win Rate</th><th>Total P&L</th><th>Reliability</th></tr></thead><tbody>'
        
        for source, stats in list(source_performance.items())[:10]:  # Show max 10 sources
            win_rate = stats.get('win_rate', 0)
            win_rate_class = 'positive' if win_rate > 60 else 'warning' if win_rate > 40 else 'negative'
            
            total_pnl = stats.get('total_pnl', 0)
            pnl_class = 'positive' if total_pnl > 0 else 'negative'
            
            html += f"""
            <tr>
                <td>{source}</td>
                <td class="{win_rate_class}">{win_rate:.1f}%</td>
                <td class="{pnl_class}">${total_pnl:.2f}</td>
                <td>{stats.get('reliability', 'Unknown')}</td>
            </tr>
            """
        
        html += '</tbody></table>'
        return html
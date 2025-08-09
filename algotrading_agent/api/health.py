from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import logging
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, agent_ref=None, **kwargs):
        self.agent = agent_ref
        super().__init__(*args, **kwargs)
        
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/health':
            self._serve_health()
        elif path == '/' or path == '/dashboard':
            self._serve_modern_dashboard()
        elif path == '/metrics':
            self._serve_prometheus_metrics()
        elif path == '/api/metrics':
            self._serve_metrics_json()
        elif path == '/api/trading-summary':
            self._serve_trading_summary()
        elif path == '/api/portfolio':
            self._serve_portfolio()
        elif path == '/api/positions':
            self._serve_positions()
        elif path == '/api/trailing-stops':
            self._serve_trailing_stops()
        elif path == '/api/performance':
            self._serve_performance()
        elif path == '/api/system-health':
            self._serve_system_health()
        elif path == '/api/alerts':
            self._serve_alerts()
        elif path == '/api/alerts/active':
            self._serve_active_alerts()
        elif path == '/api/alerts/stats':
            self._serve_alert_stats()
        elif path == '/api/logs/search':
            self._serve_log_search(parsed_path)
        elif path == '/api/logs/analytics':
            self._serve_log_analytics()
        elif path == '/api/logs/recent':
            self._serve_recent_logs()
        elif path == '/api/correlation/test':
            self._serve_correlation_test()
        elif path == '/api/correlation/results':
            self._serve_correlation_results()
        elif path == '/api/trading/dashboard':
            self._serve_trading_dashboard()
        elif path == '/api/trading/performance':
            self._serve_trading_performance()
        elif path == '/api/trading/decisions':
            self._serve_trading_decisions()
        else:
            self._send_404()
            
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path.startswith('/api/config/'):
            self._handle_config_update(path)
        else:
            self._send_404()
            
    def _serve_health(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "algotrading-agent"
        }
        
        self.wfile.write(json.dumps(health_data).encode())
        
    def _serve_modern_dashboard(self):
        """Serve a modern dashboard with real-time trading metrics"""
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Algorithmic Trading System - Live Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0c1445 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .status-indicators {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 25px;
            font-size: 0.9rem;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4ade80;
            box-shadow: 0 0 10px #4ade80;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(15px);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border-color: rgba(79, 172, 254, 0.5);
        }
        .metric-card h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #4facfe;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .metric-label {
            font-size: 0.85rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        .metric-change {
            font-size: 0.9rem;
            padding: 4px 8px;
            border-radius: 12px;
            display: inline-block;
        }
        .positive { background: rgba(74, 222, 128, 0.2); color: #4ade80; }
        .negative { background: rgba(248, 113, 113, 0.2); color: #f87171; }
        .neutral { background: rgba(156, 163, 175, 0.2); color: #9ca3af; }
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            border: none;
            color: white;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            font-size: 1.5rem;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
            transition: all 0.3s ease;
        }
        .refresh-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 40px rgba(79, 172, 254, 0.6);
        }
        .last-update {
            text-align: center;
            opacity: 0.6;
            font-size: 0.85rem;
            margin-top: 20px;
        }
        @media (max-width: 768px) {
            .metrics-grid { grid-template-columns: 1fr; }
            .status-indicators { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Algorithmic Trading System</h1>
        <div class="status-indicators">
            <div class="status-item">
                <div class="status-dot"></div>
                System Online
            </div>
            <div class="status-item" id="market-status">
                <div class="status-dot"></div>
                <span id="market-text">Market Status</span>
            </div>
            <div class="status-item" id="trades-status">
                <div class="status-dot"></div>
                <span id="trades-text">Trading Active</span>
            </div>
        </div>
    </div>

    <div class="metrics-grid" id="metrics-container">
        <!-- Metrics will be populated by JavaScript -->
    </div>

    <div class="last-update" id="last-update">
        Loading metrics...
    </div>

    <button class="refresh-btn" onclick="loadMetrics()" title="Refresh Data">
        ⟳
    </button>

    <script>
        async function loadMetrics() {
            try {
                const response = await fetch('/api/trading-summary');
                const data = await response.json();
                displayMetrics(data);
                document.getElementById('last-update').textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
            } catch (error) {
                console.error('Error loading metrics:', error);
                document.getElementById('last-update').textContent = `Error loading data: ${error.message}`;
            }
        }

        function displayMetrics(data) {
            const container = document.getElementById('metrics-container');
            container.innerHTML = '';

            // Performance Metrics
            if (data.performance) {
                const perfCard = createMetricCard('📈 Trading Performance', [
                    { label: 'Total Trades', value: data.performance.total_trades, type: 'number' },
                    { label: 'Win Rate', value: data.performance.win_rate, type: 'percentage' },
                    { label: 'Total P&L', value: data.performance.total_pnl, type: 'currency' },
                    { label: 'Profit Factor', value: data.performance.profit_factor, type: 'number' }
                ]);
                container.appendChild(perfCard);
            }

            // Portfolio Metrics
            if (data.portfolio) {
                const portfolioCard = createMetricCard('💼 Portfolio Status', [
                    { label: 'Portfolio Value', value: data.portfolio.value, type: 'currency' },
                    { label: 'Available Cash', value: data.portfolio.cash, type: 'currency' },
                    { label: 'Active Positions', value: data.portfolio.positions, type: 'number' },
                    { label: 'Long/Short', value: `${data.portfolio.long_positions}/${data.portfolio.short_positions}`, type: 'text' }
                ]);
                container.appendChild(portfolioCard);
            }

            // Risk Metrics
            if (data.risk) {
                const riskCard = createMetricCard('⚠️ Risk Management', [
                    { label: 'Current Drawdown', value: data.risk.current_drawdown, type: 'percentage' },
                    { label: 'Max Drawdown', value: data.risk.max_drawdown, type: 'percentage' },
                    { label: 'Risk Utilization', value: data.risk.risk_utilization, type: 'percentage' }
                ]);
                container.appendChild(riskCard);
            }

            // System Health
            if (data.system) {
                const systemCard = createMetricCard('🖥️ System Health', [
                    { label: 'Uptime', value: data.system.uptime, type: 'text' },
                    { label: 'Memory Usage', value: data.system.memory_usage, type: 'text' },
                    { label: 'API Response', value: data.system.api_response_time, type: 'text' },
                    { label: 'Error Rate', value: data.system.error_rate, type: 'percentage' }
                ]);
                container.appendChild(systemCard);
            }

            // Enhanced Features
            if (data.enhanced_features) {
                const enhancedCard = createMetricCard('🚀 Enhanced Features', [
                    { label: 'Trailing Stops Active', value: data.enhanced_features.trailing_stops_active, type: 'number' },
                    { label: 'Enhanced Signals', value: data.enhanced_features.enhanced_signals, type: 'number' },
                    { label: 'AI Success Rate', value: data.enhanced_features.ai_success_rate, type: 'percentage' }
                ]);
                container.appendChild(enhancedCard);
            }
        }

        function createMetricCard(title, metrics) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            
            let content = `<h3>${title}</h3>`;
            metrics.forEach(metric => {
                const valueClass = getValueClass(metric.value, metric.type);
                content += `
                    <div style="margin-bottom: 15px;">
                        <div class="metric-label">${metric.label}</div>
                        <div class="metric-value ${valueClass}">${formatValue(metric.value, metric.type)}</div>
                    </div>
                `;
            });
            
            card.innerHTML = content;
            return card;
        }

        function formatValue(value, type) {
            if (typeof value === 'string') return value;
            
            switch (type) {
                case 'currency':
                    return value.replace(/[$,]/g, '') ? value : '$0.00';
                case 'percentage':
                    return value.includes('%') ? value : value + '%';
                case 'number':
                    return value.toString();
                default:
                    return value;
            }
        }

        function getValueClass(value, type) {
            if (type === 'percentage' || type === 'currency') {
                const numValue = parseFloat(value.toString().replace(/[%$,]/g, ''));
                return numValue > 0 ? 'positive' : numValue < 0 ? 'negative' : 'neutral';
            }
            return '';
        }

        // Auto-refresh every 30 seconds
        setInterval(loadMetrics, 30000);

        // Initial load
        loadMetrics();
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(dashboard_html.encode())
            
    def _serve_portfolio(self):
        try:
            # Default portfolio state
            portfolio_data = {
                "total_portfolio_value": 100000,
                "positions_value": 0,
                "available_cash": 100000,
                "daily_pnl": 0.0,
                "risk_utilization": 0.0,
                "current_positions": 0
            }
            
            if self.agent and hasattr(self.agent, 'risk_manager'):
                portfolio_data = self.agent.risk_manager.get_portfolio_status()
                
            self._send_json(portfolio_data)
        except Exception as e:
            self._send_error(500, f"Error getting portfolio: {str(e)}")
            
    def _serve_news(self):
        try:
            # Get recent news from agent if available
            news_data = []
            if self.agent and hasattr(self.agent, '_last_news'):
                news_data = getattr(self.agent, '_last_news', [])
                
            self._send_json(news_data[:20])  # Last 20 items
        except Exception as e:
            self._send_error(500, f"Error getting news: {str(e)}")
            
    def _serve_decisions(self):
        try:
            # Get recent decisions from agent if available
            decisions_data = []
            if self.agent and hasattr(self.agent, '_last_decisions'):
                decisions_data = getattr(self.agent, '_last_decisions', [])
                
            # Convert TradingPair objects to dict if needed
            decisions_json = []
            for decision in decisions_data:
                if hasattr(decision, 'to_dict'):
                    decisions_json.append(decision.to_dict())
                else:
                    decisions_json.append(decision)
                    
            self._send_json(decisions_json[:10])  # Last 10 decisions
        except Exception as e:
            self._send_error(500, f"Error getting decisions: {str(e)}")
            
    def _serve_prometheus_metrics(self):
        """Serve Prometheus-compatible metrics"""
        try:
            metrics_text = "# HELP trading_system_metrics Trading system metrics\n"
            metrics_text += "# TYPE trading_system_up gauge\n"
            metrics_text += "trading_system_up 1\n"
            
            # Get metrics from metrics collector if available
            if self.agent and hasattr(self.agent, 'metrics_collector'):
                prometheus_data = self.agent.metrics_collector.get_prometheus_metrics()
                if prometheus_data:
                    metrics_text = prometheus_data
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(metrics_text.encode())
        except Exception as e:
            self._send_error(500, f"Error getting metrics: {str(e)}")
    
    def _serve_metrics_json(self):
        """Serve metrics in JSON format"""
        try:
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": "online",
                "metrics": {}
            }
            
            if self.agent and hasattr(self.agent, 'metrics_collector'):
                current_metrics = self.agent.metrics_collector.get_current_metrics()
                metrics_data["metrics"] = current_metrics.to_prometheus_metrics()
            
            self._send_json(metrics_data)
        except Exception as e:
            self._send_error(500, f"Error getting metrics JSON: {str(e)}")
    
    def _serve_trading_summary(self):
        """Serve comprehensive trading summary for dashboard"""
        try:
            summary_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "performance": {
                    "total_trades": 0,
                    "win_rate": "0.0%",
                    "total_pnl": "$0.00",
                    "profit_factor": "0.00"
                },
                "portfolio": {
                    "value": "$100,000.00",
                    "cash": "$100,000.00", 
                    "positions": 0,
                    "long_positions": 0,
                    "short_positions": 0
                },
                "risk": {
                    "current_drawdown": "0.0%",
                    "max_drawdown": "0.0%",
                    "risk_utilization": "0.0%"
                },
                "system": {
                    "uptime": "0.0h",
                    "memory_usage": "0.0MB",
                    "api_response_time": "0ms",
                    "error_rate": "0.00%"
                },
                "enhanced_features": {
                    "trailing_stops_active": 0,
                    "enhanced_signals": 0,
                    "ai_success_rate": "0.0%"
                }
            }
            
            # Get real data from metrics collector if available
            if self.agent and hasattr(self.agent, 'metrics_collector'):
                trading_summary = self.agent.metrics_collector.get_trading_summary()
                summary_data.update(trading_summary)
            
            self._send_json(summary_data)
        except Exception as e:
            self._send_error(500, f"Error getting trading summary: {str(e)}")
    
    def _serve_positions(self):
        """Serve current positions with detailed information"""
        try:
            positions_data = []
            
            if self.agent and hasattr(self.agent, 'alpaca_client'):
                # Get detailed positions from Alpaca client
                import asyncio
                try:
                    positions = asyncio.run(self.agent.alpaca_client.get_positions())
                    for pos in positions:
                        symbol = pos["symbol"]
                        
                        # Get detailed position info if available
                        try:
                            detailed_pos = asyncio.run(
                                self.agent.alpaca_client.get_position_with_orders(symbol)
                            )
                            positions_data.append(detailed_pos)
                        except:
                            positions_data.append(pos)
                            
                except Exception as e:
                    self.logger.warning(f"Error getting positions: {e}")
            
            self._send_json(positions_data)
        except Exception as e:
            self._send_error(500, f"Error getting positions: {str(e)}")
    
    def _serve_trailing_stops(self):
        """Serve trailing stops status"""
        try:
            trailing_data = {}
            
            if self.agent and hasattr(self.agent, 'trailing_stop_manager'):
                trailing_data = self.agent.trailing_stop_manager.get_trailing_stop_status()
            
            self._send_json(trailing_data)
        except Exception as e:
            self._send_error(500, f"Error getting trailing stops: {str(e)}")
    
    def _serve_performance(self):
        """Serve performance metrics over time"""
        try:
            performance_data = {
                "daily_pnl": [],
                "win_rate_history": [],
                "drawdown_history": [],
                "trade_count_history": []
            }
            
            if self.agent and hasattr(self.agent, 'metrics_collector'):
                # Get last 24 hours of metrics
                history = self.agent.metrics_collector.get_metrics_history(24)
                
                for metrics in history:
                    timestamp = metrics.timestamp.isoformat()
                    performance_data["daily_pnl"].append({
                        "timestamp": timestamp,
                        "value": metrics.total_pnl
                    })
                    performance_data["win_rate_history"].append({
                        "timestamp": timestamp,
                        "value": metrics.win_rate()
                    })
                    performance_data["drawdown_history"].append({
                        "timestamp": timestamp,
                        "value": metrics.current_drawdown * 100
                    })
                    performance_data["trade_count_history"].append({
                        "timestamp": timestamp,
                        "value": metrics.total_trades
                    })
            
            self._send_json(performance_data)
        except Exception as e:
            self._send_error(500, f"Error getting performance data: {str(e)}")
    
    def _serve_system_health(self):
        """Serve detailed system health information"""
        try:
            health_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "components": {},
                "alerts": []
            }
            
            if self.agent:
                # Check component health
                components = [
                    'news_scraper', 'news_filter', 'news_analysis_brain', 
                    'decision_engine', 'risk_manager', 'metrics_collector',
                    'trailing_stop_manager'
                ]
                
                for component_name in components:
                    if hasattr(self.agent, component_name):
                        component = getattr(self.agent, component_name)
                        if hasattr(component, 'is_running'):
                            status = "running" if component.is_running else "stopped"
                            health_data["components"][component_name] = {
                                "status": status,
                                "last_updated": getattr(component, 'last_updated', datetime.utcnow()).isoformat()
                            }
            
            self._send_json(health_data)
        except Exception as e:
            self._send_error(500, f"Error getting system health: {str(e)}")
    
    def _serve_alerts(self):
        """Serve all alerts"""
        try:
            alerts_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "alerts": []
            }
            
            if self.agent and hasattr(self.agent, 'alert_manager'):
                alerts_list = list(self.agent.alert_manager.alerts.values())
                alerts_data["alerts"] = [alert.to_dict() for alert in alerts_list]
                alerts_data["total_count"] = len(alerts_list)
            
            self._send_json(alerts_data)
        except Exception as e:
            self._send_error(500, f"Error getting alerts: {str(e)}")
    
    def _serve_active_alerts(self):
        """Serve active (unresolved) alerts"""
        try:
            alerts_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_alerts": []
            }
            
            if self.agent and hasattr(self.agent, 'alert_manager'):
                active_alerts = self.agent.alert_manager.get_active_alerts()
                alerts_data["active_alerts"] = [alert.to_dict() for alert in active_alerts]
                alerts_data["active_count"] = len(active_alerts)
            
            self._send_json(alerts_data)
        except Exception as e:
            self._send_error(500, f"Error getting active alerts: {str(e)}")
    
    def _serve_alert_stats(self):
        """Serve alert statistics"""
        try:
            stats_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "stats": {}
            }
            
            if self.agent and hasattr(self.agent, 'alert_manager'):
                stats_data["stats"] = self.agent.alert_manager.get_alert_stats()
            
            self._send_json(stats_data)
        except Exception as e:
            self._send_error(500, f"Error getting alert stats: {str(e)}")
    
    def _serve_log_search(self, parsed_path):
        """Serve log search results"""
        try:
            query_params = parse_qs(parsed_path.query)
            
            # Extract search parameters
            query = query_params.get('q', [None])[0]
            level = query_params.get('level', [None])[0]
            category = query_params.get('category', [None])[0]
            module = query_params.get('module', [None])[0]
            symbol = query_params.get('symbol', [None])[0]
            limit = int(query_params.get('limit', ['100'])[0])
            
            # Time range
            start_time = None
            end_time = None
            if 'start_time' in query_params:
                start_time = datetime.fromisoformat(query_params['start_time'][0])
            if 'end_time' in query_params:
                end_time = datetime.fromisoformat(query_params['end_time'][0])
            
            search_results = []
            if self.agent and hasattr(self.agent, 'log_aggregator'):
                from algotrading_agent.observability.log_aggregator import LogLevel, LogCategory
                
                # Convert string parameters to enums
                level_enum = LogLevel(level.upper()) if level else None
                category_enum = LogCategory(category.lower()) if category else None
                
                results = self.agent.log_aggregator.search_logs(
                    query=query,
                    level=level_enum,
                    category=category_enum,
                    module=module,
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                search_results = [log.to_dict() for log in results]
            
            response_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "filters": {
                    "level": level,
                    "category": category,
                    "module": module,
                    "symbol": symbol,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None
                },
                "results": search_results,
                "count": len(search_results)
            }
            
            self._send_json(response_data)
        except Exception as e:
            self._send_error(500, f"Error searching logs: {str(e)}")
    
    def _serve_log_analytics(self):
        """Serve log analytics"""
        try:
            analytics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "analytics": {}
            }
            
            if self.agent and hasattr(self.agent, 'log_aggregator'):
                analytics_data["analytics"] = self.agent.log_aggregator.get_log_analytics()
            
            self._send_json(analytics_data)
        except Exception as e:
            self._send_error(500, f"Error getting log analytics: {str(e)}")
    
    def _serve_recent_logs(self):
        """Serve recent logs from memory"""
        try:
            recent_logs_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "logs": []
            }
            
            if self.agent and hasattr(self.agent, 'log_aggregator'):
                with self.agent.log_aggregator.lock:
                    recent_logs_data["logs"] = [
                        log.to_dict() for log in list(self.agent.log_aggregator.recent_logs)[-50:]
                    ]
                    recent_logs_data["count"] = len(recent_logs_data["logs"])
            
            self._send_json(recent_logs_data)
        except Exception as e:
            self._send_error(500, f"Error getting recent logs: {str(e)}")
    
    def _serve_correlation_test(self):
        """Run historical correlation test"""
        try:
            if self.agent and hasattr(self.agent, 'correlation_tracker'):
                # Run async test in thread
                import asyncio
                import threading
                
                result = {"status": "starting", "message": "Correlation test initiated"}
                
                def run_test():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        test_result = loop.run_until_complete(
                            self.agent.correlation_tracker.run_historical_test()
                        )
                        loop.close()
                        
                        # Store result (in production, you'd use a proper job queue)
                        if not hasattr(self.agent, '_correlation_test_results'):
                            self.agent._correlation_test_results = []
                        self.agent._correlation_test_results.append(test_result)
                        
                    except Exception as e:
                        print(f"Correlation test error: {e}")
                
                # Start test in background
                test_thread = threading.Thread(target=run_test)
                test_thread.start()
                
                result["message"] = "Historical correlation test started - check /api/correlation/results"
            else:
                result = {"status": "error", "message": "Correlation tracker not available"}
            
            self._send_json(result)
        except Exception as e:
            self._send_error(500, f"Error running correlation test: {str(e)}")
    
    def _serve_correlation_results(self):
        """Serve correlation test results and current metrics"""
        try:
            results_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "current_metrics": {},
                "grafana_data": {},
                "recent_tests": []
            }
            
            if self.agent and hasattr(self.agent, 'correlation_tracker'):
                # Current correlation metrics
                results_data["current_metrics"] = self.agent.correlation_tracker.current_metrics.to_dict()
                
                # Grafana visualization data
                results_data["grafana_data"] = self.agent.correlation_tracker.get_grafana_data()
                
                # Recent test results (if any)
                if hasattr(self.agent, '_correlation_test_results'):
                    results_data["recent_tests"] = self.agent._correlation_test_results[-5:]  # Last 5 tests
            
            self._send_json(results_data)
        except Exception as e:
            self._send_error(500, f"Error getting correlation results: {str(e)}")
    
    def _serve_trading_dashboard(self):
        """Serve comprehensive trading dashboard data"""
        try:
            if self.agent and hasattr(self.agent, 'trading_dashboard'):
                dashboard_data = self.agent.trading_dashboard.get_dashboard_data()
                self._send_json(dashboard_data)
            else:
                self._send_json({
                    "error": "Trading dashboard not available",
                    "message": "Dashboard components not initialized"
                })
        except Exception as e:
            self._send_error(500, f"Error getting trading dashboard: {str(e)}")
    
    def _serve_trading_performance(self):
        """Serve trading performance summary"""
        try:
            if self.agent and hasattr(self.agent, 'trade_performance_tracker'):
                performance_data = self.agent.trade_performance_tracker.get_performance_summary()
                recent_trades = self.agent.trade_performance_tracker.get_recent_trades(24)
                
                response_data = {
                    "performance_summary": performance_data,
                    "recent_trades": recent_trades,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                self._send_json(response_data)
            else:
                self._send_json({
                    "error": "Trade performance tracker not available",
                    "performance_summary": {},
                    "recent_trades": []
                })
        except Exception as e:
            self._send_error(500, f"Error getting trading performance: {str(e)}")
    
    def _serve_trading_decisions(self):
        """Serve trading decision analytics"""
        try:
            if self.agent and hasattr(self.agent, 'decision_analyzer'):
                decision_data = self.agent.decision_analyzer.get_decision_analytics()
                
                response_data = {
                    "decision_analytics": decision_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                self._send_json(response_data)
            else:
                self._send_json({
                    "error": "Decision analyzer not available",
                    "decision_analytics": {}
                })
        except Exception as e:
            self._send_error(500, f"Error getting trading decisions: {str(e)}")
            
    def _handle_config_update(self, path):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                config_type = path.split('/')[-1]
                value = data.get('value')
                
                # Update configuration based on type
                if self.agent and hasattr(self.agent, 'config'):
                    if config_type == 'min_confidence':
                        self.agent.config.update_config('decision_engine.min_confidence', value)
                    elif config_type == 'stop_loss':
                        self.agent.config.update_config('decision_engine.default_stop_loss_pct', value)
                    elif config_type == 'take_profit':
                        self.agent.config.update_config('decision_engine.default_take_profit_pct', value)
                
                self._send_json({"success": True, "message": f"{config_type} updated to {value}"})
            else:
                self._send_error(400, "No data provided")
        except Exception as e:
            self._send_error(500, f"Error updating config: {str(e)}")
            
    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())
        
    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_data = {"error": message, "timestamp": datetime.utcnow().isoformat()}
        self.wfile.write(json.dumps(error_data).encode())
        
    def _send_404(self):
        self.send_response(404)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'404 Not Found')
        
    def log_message(self, format, *args):
        # Suppress default HTTP server logs
        pass


class HealthServer:
    def __init__(self, port=8080, agent_ref=None):
        self.port = port
        self.agent = agent_ref
        self.server = None
        self.thread = None
        self.logger = logging.getLogger("algotrading.health")
        
    def start(self):
        try:
            # Create handler class with agent reference
            def handler_factory(*args, **kwargs):
                return DashboardHandler(*args, agent_ref=self.agent, **kwargs)
            
            self.server = HTTPServer(('0.0.0.0', self.port), handler_factory)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.logger.info(f"Dashboard server started on port {self.port}")
            self.logger.info(f"Dashboard available at: http://localhost:{self.port}/dashboard")
        except Exception as e:
            self.logger.error(f"Failed to start health server: {e}")
            
    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.logger.info("Dashboard server stopped")
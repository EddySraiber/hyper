from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import logging
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs


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
            self._serve_dashboard()
        elif path == '/api/portfolio':
            self._serve_portfolio()
        elif path == '/api/news':
            self._serve_news()
        elif path == '/api/decisions':
            self._serve_decisions()
        elif path == '/api/logs':
            self._serve_logs()
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
        
    def _serve_dashboard(self):
        try:
            dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
            with open(dashboard_path, 'r') as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(content.encode())
        except Exception as e:
            self._send_error(500, f"Error serving dashboard: {str(e)}")
            
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
            
    def _serve_logs(self):
        try:
            # Get recent logs from agent if available
            logs_data = []
            if self.agent and hasattr(self.agent, '_recent_logs'):
                logs_data = getattr(self.agent, '_recent_logs', [])
                
            self._send_json(logs_data[-50:])  # Last 50 logs
        except Exception as e:
            self._send_error(500, f"Error getting logs: {str(e)}")
            
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
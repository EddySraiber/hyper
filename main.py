#!/usr/bin/env python3
"""
Algotrading Agent - Main Entry Point
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import List, Dict, Any

from algotrading_agent.config.settings import get_config
from algotrading_agent.components.news_scraper import NewsScraper
from algotrading_agent.components.news_filter import NewsFilter
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
from algotrading_agent.components.news_impact_scorer import NewsImpactScorer
from algotrading_agent.components.decision_engine import DecisionEngine
from algotrading_agent.components.risk_manager import RiskManager
from algotrading_agent.components.statistical_advisor import StatisticalAdvisor
from algotrading_agent.components.trade_manager import TradeManager
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.api.health import HealthServer
from algotrading_agent.observability.service import ObservabilityService, ObservabilityConfig
from algotrading_agent.observability.schemas.live_metrics import ComponentStatus
from algotrading_agent.observability.alert_manager import AlertManager
from algotrading_agent.observability.log_aggregator import LogAggregator, StructuredLogHandler
from algotrading_agent.observability.alpaca_sync import AlpacaSyncService


class DashboardLogHandler(logging.Handler):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.utcfromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.name
            }
            
            # Add to recent logs (keep last 100)
            self.agent._recent_logs.append(log_entry)
            if len(self.agent._recent_logs) > 100:
                self.agent._recent_logs.pop(0)
        except:
            pass  # Don't let logging errors crash the system


class AlgotradingAgent:
    def __init__(self):
        self.config = get_config()
        self.logger = self._setup_logging()
        self.running = False
        
        # Initialize components
        self.news_scraper = NewsScraper(self.config.get_component_config('news_scraper'))
        self.news_filter = NewsFilter(self.config.get_component_config('news_filter'))
        self.news_brain = NewsAnalysisBrain(self.config.get_component_config('news_analysis_brain'))
        self.news_impact_scorer = NewsImpactScorer(self.config.get_component_config('news_impact_scorer'))
        self.decision_engine = DecisionEngine(self.config.get_component_config('decision_engine'))
        self.risk_manager = RiskManager(self.config.get_component_config('risk_manager'))
        self.statistical_advisor = StatisticalAdvisor(self.config.get_component_config('statistical_advisor'))
        self.trade_manager = TradeManager(self.config.get_component_config('trade_manager'))
        
        # Initialize trading client
        try:
            self.alpaca_client = AlpacaClient(self.config.get_alpaca_config())
            # Inject Alpaca client into components that need it
            self.decision_engine.alpaca_client = self.alpaca_client
            self.trade_manager.alpaca_client = self.alpaca_client
            # Inject decision engine reference for failure feedback
            self.trade_manager.decision_engine = self.decision_engine
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            self.logger.info("Running in simulation mode without real trading")
            self.alpaca_client = None
            
        # Initialize alert manager
        alert_config = self.config.get_component_config('observability.alerts')
        self.alert_manager = AlertManager(alert_config)
        
        # Initialize log aggregator
        log_agg_config = self.config.get('observability.log_aggregator', {})
        self.log_aggregator = LogAggregator(log_agg_config)
        
        # Initialize observability components (legacy - some still needed)
        from algotrading_agent.observability.correlation_tracker import CorrelationTracker
        # from algotrading_agent.observability.trade_performance_tracker import TradePerformanceTracker  # DELETED
        # from algotrading_agent.observability.decision_analyzer import DecisionAnalyzer  # DELETED
        
        # Note: ObservabilityService is initialized above, replacing MetricsCollector
        
        # Initialize correlation tracker
        correlation_config = self.config.get('observability.correlation_tracker', {'data_dir': '/app/data'})
        self.correlation_tracker = CorrelationTracker(correlation_config)
        
        # DELETED: trade_performance_tracker, decision_analyzer, trading_dashboard
        # These are now replaced by ObservabilityService
        
        # Initialize Alpaca sync service for real data
        self.alpaca_sync = None  # Will be initialized after alpaca_client is ready
        
        # Initialize observability service (enterprise-grade)
        obs_config = ObservabilityConfig(
            prometheus_port=8090,
            enable_live_metrics=True,
            enable_backtest_metrics=True
        )
        self.observability = ObservabilityService(obs_config)
        
        # Initialize health server with dashboard
        self.health_server = HealthServer(port=8080, agent_ref=self)
        
        # Data tracking for dashboard
        self._last_news = []
        self._last_decisions = []
        self._recent_logs = []
        self._active_alerts = []
        
        # Market status tracking for state transition logging
        self._last_market_status = None  # Track previous market status for transition logging
            
        self.logger.info("Algotrading Agent initialized")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging_config = self.config.get_component_config('logging')
        log_level = getattr(logging, logging_config.get('level', 'INFO'))
        
        # Create formatter
        formatter = logging.Formatter(
            logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Dashboard log handler
        dashboard_handler = DashboardLogHandler(self)
        dashboard_handler.setFormatter(formatter)
        logger.addHandler(dashboard_handler)
        
        # Structured log handler (will be added after log aggregator is initialized)
        self._structured_handler = None
        
        # File handler (if configured)
        log_file = logging_config.get('file')
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
                
        return logging.getLogger("algotrading.main")
        
    async def start(self):
        """Start the algotrading agent"""
        self.logger.info("Starting Algotrading Agent...")
        self.running = True
        
        try:
            # Start observability service first
            await self.observability.start()
            
            # Start health server
            self.health_server.start()
            
            # Start log aggregator and add structured logging
            await self.log_aggregator.start()
            if not self._structured_handler:
                self._structured_handler = StructuredLogHandler(self.log_aggregator)
                logging.getLogger().addHandler(self._structured_handler)
            
            # ObservabilityService is started above, no need for separate metrics collector
            
            # Start all components
            await self.news_scraper.start()
            self.news_filter.start()
            await self.news_brain.start()
            self.news_impact_scorer.start()
            self.decision_engine.start()
            self.risk_manager.start()
            self.statistical_advisor.start()
            self.trade_manager.start()
            
            # Print account info if trading is enabled
            if self.alpaca_client:
                try:
                    account_info = await self.alpaca_client.get_account_info()
                    self.logger.info(f"Connected to Alpaca - Portfolio Value: ${account_info['portfolio_value']:,.2f}")
                    
                    # Initialize Alpaca sync service for real data
                    self.alpaca_sync = AlpacaSyncService(
                        alpaca_client=self.alpaca_client,
                        observability_service=self.observability
                    )
                    
                    # Do initial sync of real Alpaca data
                    sync_result = await self.alpaca_sync.sync_real_trading_data()
                    if sync_result['success']:
                        self.logger.info(f"✅ Synced real Alpaca data: {sync_result['positions']} positions, "
                                        f"${sync_result['portfolio_value']:,.2f} portfolio value")
                        
                        # Create trade records from historical Alpaca data  
                        created_trades = await self.alpaca_sync.create_trade_records_from_alpaca()
                        if created_trades > 0:
                            self.logger.info(f"Created {created_trades} trade records from Alpaca history")
                    else:
                        self.logger.warning(f"Failed to sync Alpaca data: {sync_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Alpaca connection failed: {e}")
                    self.logger.info("Continuing in SIMULATION MODE - no real trading")
                    self.alpaca_client = None  # Fall back to simulation
                    self.alpaca_sync = None
                
            self.logger.info("All components started successfully")
            
            # Main processing loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Error during startup: {e}")
            raise
            
    async def stop(self):
        """Stop the algotrading agent"""
        self.logger.info("Stopping Algotrading Agent...")
        self.running = False
        
        # Stop all components
        await self.news_scraper.stop()
        self.news_filter.stop()
        await self.news_brain.stop()
        self.news_impact_scorer.stop()
        self.decision_engine.stop()
        self.risk_manager.stop()
        self.statistical_advisor.stop()
        self.trade_manager.stop()
        
        # Stop health server
        self.health_server.stop()
        
        # Stop log aggregator
        await self.log_aggregator.stop()
        if self._structured_handler:
            logging.getLogger().removeHandler(self._structured_handler)
        
        # Stop observability service
        await self.observability.stop()
        
        self.logger.info("Algotrading Agent stopped")
        
    async def _main_loop(self):
        """Main processing loop"""
        self.logger.info("Starting main processing loop")
        
        while self.running:
            try:
                start_time = datetime.utcnow()
                
                # Check market hours - enter rest mode if markets closed
                market_open = False
                if self.alpaca_client:
                    try:
                        market_open = await self.alpaca_client.is_market_open()
                    except Exception as e:
                        self.logger.warning(f"Could not check market status: {e}")
                        market_open = False
                
                # Log market status transitions (only when status changes)
                if self._last_market_status != market_open:
                    if not market_open:
                        self.logger.info("🛌 ENTERING REST MODE: Markets closed - switching to news analysis only")
                    else:
                        self.logger.info("📈 ENTERING ACTIVE MODE: Markets opened - full trading pipeline activated")
                    self._last_market_status = market_open
                
                # Log current mode (less frequent, every 10th iteration in same state)
                elif hasattr(self, '_status_log_counter'):
                    self._status_log_counter += 1
                    if self._status_log_counter >= 10:
                        status_msg = "REST MODE: Markets closed" if not market_open else "ACTIVE MODE: Markets open"
                        self.logger.info(f"🔄 {status_msg} (continuing...)")
                        self._status_log_counter = 0
                else:
                    self._status_log_counter = 0
                
                # Step 1: Scrape news (always continue - keeps system warm)
                self.logger.info("Scraping news...")
                raw_news = await self.news_scraper.process()
                self.logger.info(f"Scraped {len(raw_news)} news items")
                
                if not raw_news:
                    await asyncio.sleep(60)  # Wait before next iteration
                    continue
                    
                # Step 2: Filter news
                self.logger.info("Filtering news...")
                filtered_news = self.news_filter.process(raw_news)
                self.logger.info(f"Filtered to {len(filtered_news)} relevant items")
                
                if not filtered_news:
                    await asyncio.sleep(60)
                    continue
                    
                # Step 3: Analyze news
                self.logger.info("Analyzing news...")
                analyzed_news = await self.news_brain.process(filtered_news)
                self.logger.info(f"Analyzed {len(analyzed_news)} news items")
                
                # Step 3.5: Score news impact potential
                self.logger.info("Scoring news impact...")
                scored_news = self.news_impact_scorer.process(analyzed_news)
                self.logger.info(f"Scored {len(scored_news)} news items for market impact")
                
                # Log high-impact news summary
                impact_summary = self.news_impact_scorer.get_impact_summary(scored_news)
                if impact_summary.get('high_impact_count', 0) > 0:
                    self.logger.warning(f"🔥 HIGH IMPACT: {impact_summary['high_impact_count']} news items with grade A/B")
                    for story in impact_summary.get('top_stories', [])[:2]:
                        self.logger.info(f"  📈 {story['grade']}: {story['title']} (score: {story['score']:.2f})")
                
                # Store news for dashboard (use scored news)
                self._last_news = scored_news[-20:] if scored_news else []
                
                # Skip trading pipeline if markets are closed (rest mode)
                if not market_open:
                    self.logger.info("⏸️  Skipping trading pipeline - markets closed")
                    # Clear previous decisions in rest mode
                    self._last_decisions = []
                else:
                    # Step 4: Make trading decisions using impact-scored news
                    self.logger.info("Making trading decisions...")
                    trading_pairs = await self.decision_engine.process(scored_news)
                    self.logger.info(f"Generated {len(trading_pairs)} trading decisions")
                    
                    if not trading_pairs:
                        await asyncio.sleep(60)
                        continue
                        
                    # Step 5: Apply statistical insights
                    self.logger.info("Applying statistical insights...")
                    enhanced_pairs = self.statistical_advisor.process(trading_pairs)
                    
                    # Step 6: Risk management
                    self.logger.info("Applying risk management...")
                    approved_pairs = self.risk_manager.process(enhanced_pairs)
                    self.logger.info(f"Risk manager approved {len(approved_pairs)} trades")
                    
                    # Store decisions for dashboard
                    self._last_decisions = approved_pairs[-10:] if approved_pairs else []
                    
                    # Step 7: Execute trades - use express lane for breaking news
                    if approved_pairs:
                        # Check if we have breaking news or high-impact news in our scored data
                        has_breaking_news = any(item.get("priority") == "breaking" for item in scored_news)
                        has_high_impact = any(item.get("impact_score", 0) >= 1.2 for item in scored_news)
                        
                        if has_breaking_news or has_high_impact:
                            if has_breaking_news:
                                self.logger.warning("🚀 EXPRESS LANE: Breaking news detected - executing trades immediately!")
                            else:
                                self.logger.warning("🔥 EXPRESS LANE: High-impact news detected (score ≥1.2) - executing trades immediately!")
                            await self._execute_express_trades(approved_pairs, scored_news)
                        else:
                            # Normal queue processing
                            await self._queue_trades(approved_pairs)
                        
                # Step 8: Process trade failure feedback (always check, even in rest mode)
                await self._process_failure_feedback()
                
                # Step 9: Sync real Alpaca data with observability metrics
                if self.alpaca_sync:
                    try:
                        sync_result = await self.alpaca_sync.sync_real_trading_data()
                        if not sync_result['success']:
                            self.logger.warning(f"Alpaca data sync failed: {sync_result.get('error', 'Unknown')}")
                    except Exception as e:
                        self.logger.error(f"Error syncing Alpaca data: {e}")
                
                # Step 10: Evaluate alerts
                await self._evaluate_alerts()
                    
                # Log processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                self.logger.info(f"Processing cycle completed in {processing_time:.2f} seconds")
                
                # Wait before next iteration - shorter delay if we're in active trading mode
                base_interval = self.config.get('news_scraper.update_interval', 300)
                
                # Speed up processing when markets are open and active
                if market_open and self.trade_manager and len(self.trade_manager.active_trades) < 5:
                    # Faster processing when we have capacity for new trades
                    processing_interval = max(5, base_interval // 2)  # At least 5 seconds, or half the base
                    self.logger.debug(f"Active trading mode: processing every {processing_interval}s")
                else:
                    processing_interval = base_interval
                    
                await asyncio.sleep(processing_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _queue_trades(self, trading_pairs: List):
        """Add approved trading pairs to the trade queue for monitoring"""
        self.logger.info(f"Queueing {len(trading_pairs)} trades for execution and monitoring...")
        
        queued_count = 0
        for pair in trading_pairs:
            # Add to trade queue
            if self.trade_manager.add_trade(pair):
                queued_count += 1
                
                # If we have Alpaca client, immediately execute the entry order
                if self.alpaca_client:
                    try:
                        await self._execute_entry_order(pair)
                    except Exception as e:
                        self.logger.error(f"Failed to execute entry order for {pair.symbol}: {e}")
                else:
                    # Simulation mode - just log the trade
                    self._log_simulated_trade(pair)
            else:
                self.logger.warning(f"Could not queue trade for {pair.symbol} - queue may be full")
                
        self.logger.info(f"Successfully queued {queued_count}/{len(trading_pairs)} trades")
        
    async def _execute_entry_order(self, pair):
        """Execute the entry order for a trading pair and link it to the trade queue"""
        try:
            # Validate the trade first
            validation = await self.alpaca_client.validate_trading_pair(pair)
            
            if not validation["valid"]:
                self.logger.warning(f"Trade validation failed for {pair.symbol}: {validation['errors']}")
                return
                
            if validation["warnings"]:
                self.logger.warning(f"Trade warnings for {pair.symbol}: {validation['warnings']}")
                
            # Execute the entry order (bracket order with stop-loss and take-profit)
            price_flexibility = self.trade_manager.price_flexibility_pct
            result = await self.alpaca_client.execute_trading_pair(pair, price_flexibility)
            
            # Find the corresponding trade in the queue and update with order ID
            for trade in self.trade_manager.active_trades.values():
                if (trade.symbol == pair.symbol and 
                    trade.action == pair.action and 
                    trade.entry_target_price == pair.entry_price):
                    trade.entry_order_id = result["order_id"]
                    break
                    
            self.logger.info(f"Entry order submitted: {result}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute entry order for {pair.symbol}: {e}")
            
    def _log_simulated_trade(self, pair):
        """Log a single simulated trade"""
        self.logger.info(
            f"SIMULATED: {pair.action.upper()} {pair.quantity} shares of {pair.symbol} "
            f"@ ${pair.entry_price:.2f} | SL: ${pair.stop_loss:.2f} | TP: ${pair.take_profit:.2f} "
            f"| Confidence: {pair.confidence:.2f} | Reason: {pair.reasoning[:100]}..."
        )
        
    async def _execute_express_trades(self, trading_pairs: List, analyzed_news: List[Dict[str, Any]]):
        """Execute high-priority trades immediately without normal queue delays"""
        self.logger.warning(f"🚀 EXPRESS EXECUTION: Processing {len(trading_pairs)} urgent trades")
        
        # Log breaking news that triggered express mode
        breaking_news = [item for item in analyzed_news if item.get("priority") == "breaking"]
        for news in breaking_news[:3]:  # Log first 3 breaking news items
            self.logger.warning(f"📰 BREAKING: {news.get('title', 'Unknown')[:100]}...")
        
        for pair in trading_pairs:
            # Add to queue first (for monitoring)
            if self.trade_manager.add_trade(pair):
                # Execute immediately - no validation delays for breaking news
                if self.alpaca_client:
                    try:
                        self.logger.info(f"⚡ URGENT EXECUTION: {pair.symbol} {pair.action}")
                        
                        # Use higher price flexibility for breaking news (2% instead of 1%)
                        express_flexibility = self.trade_manager.price_flexibility_pct * 2
                        result = await self.alpaca_client.execute_trading_pair(pair, express_flexibility)
                        
                        # Update trade queue with order ID
                        for trade in self.trade_manager.active_trades.values():
                            if (trade.symbol == pair.symbol and 
                                trade.action == pair.action and 
                                trade.entry_target_price == pair.entry_price):
                                trade.entry_order_id = result["order_id"]
                                break
                                
                        self.logger.warning(f"✅ EXPRESS EXECUTED: {result}")
                        
                    except Exception as e:
                        self.logger.error(f"💥 EXPRESS EXECUTION FAILED: {pair.symbol} - {e}")
                else:
                    # Simulation mode
                    self.logger.warning(f"🎯 EXPRESS SIMULATED: {pair.symbol} {pair.action} @ ${pair.entry_price:.2f}")
            else:
                self.logger.error(f"Could not queue urgent trade for {pair.symbol} - queue may be full")
                
        self.logger.warning(f"🏁 EXPRESS LANE COMPLETE: Processed {len(trading_pairs)} urgent trades")
        
    async def _process_failure_feedback(self):
        """Process trade failure feedback and log for future improvements"""
        if not self.trade_manager:
            return
            
        failures = self.trade_manager.get_failure_feedback()
        if not failures:
            return
            
        self.logger.info(f"Processing {len(failures)} trade failure reports")
        
        for failure in failures:
            symbol = failure["symbol"]
            reason = failure["failure_reason"]
            retry_count = failure["retry_count"]
            
            # Log detailed failure information
            self.logger.warning(
                f"Trade failure analysis: {symbol} {failure['action']} "
                f"failed ({reason}) - Retry {retry_count}/{self.trade_manager.max_retries}"
            )
            
            # Future enhancement: Use failures to adjust decision engine parameters
            # For now, just log the patterns we see
            if "price" in reason.lower():
                self.logger.info(f"Price-related failure detected for {symbol} - consider adjusting price flexibility")
            elif "timeout" in reason.lower():
                self.logger.info(f"Timeout failure detected for {symbol} - market may be volatile")
            elif "rejected" in reason.lower():
                self.logger.info(f"Order rejection for {symbol} - may indicate insufficient funds or invalid parameters")
                
        # Update failure statistics (could be used by statistical advisor)
        daily_failures = len([f for f in failures 
                            if datetime.fromisoformat(f["failure_time"]).date() == datetime.utcnow().date()])
        
        if daily_failures > 5:  # Configurable threshold
            self.logger.warning(f"High failure rate detected: {daily_failures} failures today - "
                              "consider adjusting trading parameters")

    async def _evaluate_alerts(self):
        """Evaluate alert conditions and send notifications"""
        try:
            # Get current metrics from ObservabilityService
            metrics = await self.observability.get_current_metrics()
            
            # Convert metrics to dict for alert evaluation
            alert_data = metrics.to_dict() if hasattr(metrics, 'to_dict') else {}
            
            # Add additional context
            alert_data.update({
                'failed_components': sum(1 for status in self.get_status()['components'].values() 
                                       if not status.get('is_running', False)),
                'consecutive_losses': self._get_consecutive_losses(),
                'daily_pnl_pct': self._get_daily_pnl_percent(),
                'cash_pct': self._get_cash_percentage(),
                'max_position_pct': self._get_max_position_percentage()
            })
            
            # Evaluate alert rules and send notifications
            triggered_alerts = await self.alert_manager.process_alerts(alert_data)
            
            # Update active alerts list for dashboard
            self._active_alerts = self.alert_manager.get_active_alerts()
            
            # Update observability service with correlation data
            if hasattr(self, 'correlation_tracker'):
                correlation_metrics = self.correlation_tracker.get_prometheus_metrics()
                # Note: ObservabilityService will handle correlation data differently
            
            # Generate session insights from decision analyzer
            if hasattr(self, 'decision_analyzer'):
                session_insights = self.decision_analyzer.generate_session_insights()
                if session_insights:
                    self.logger.info(f"Generated {len(session_insights)} decision insights")
                    for insight in session_insights:
                        if insight.priority >= 4:
                            self.logger.warning(f"High priority insight: {insight.title} - {insight.recommendation}")
            
            if triggered_alerts:
                self.logger.info(f"Alert evaluation completed: {len(triggered_alerts)} alerts triggered")
            
        except Exception as e:
            self.logger.error(f"Error evaluating alerts: {e}")
    
    def _get_consecutive_losses(self) -> int:
        """Get count of consecutive losing trades"""
        if not hasattr(self, 'trade_manager') or not self.trade_manager:
            return 0
        # Implementation would check trade history
        return 0  # Placeholder
        
    def _get_daily_pnl_percent(self) -> float:
        """Get today's P&L as percentage"""
        if not hasattr(self, 'risk_manager') or not self.risk_manager:
            return 0.0
        portfolio_status = self.risk_manager.get_portfolio_status()
        return portfolio_status.get('daily_pnl_pct', 0.0)
        
    def _get_cash_percentage(self) -> float:
        """Get cash as percentage of portfolio"""
        if not hasattr(self, 'risk_manager') or not self.risk_manager:
            return 100.0
        portfolio_status = self.risk_manager.get_portfolio_status()
        return portfolio_status.get('cash_percentage', 100.0)
        
    def _get_max_position_percentage(self) -> float:
        """Get largest single position as percentage of portfolio"""
        if not hasattr(self, 'risk_manager') or not self.risk_manager:
            return 0.0
        portfolio_status = self.risk_manager.get_portfolio_status()
        return portfolio_status.get('max_position_pct', 0.0)
                
    def _log_simulated_trades(self, trading_pairs: List):
        """Log simulated trades when not connected to broker"""
        self.logger.info("=== SIMULATED TRADES ===")
        
        for pair in trading_pairs:
            self.logger.info(
                f"SIMULATED: {pair.action.upper()} {pair.quantity} shares of {pair.symbol} "
                f"@ ${pair.entry_price:.2f} | SL: ${pair.stop_loss:.2f} | TP: ${pair.take_profit:.2f} "
                f"| Confidence: {pair.confidence:.2f} | Reason: {pair.reasoning[:100]}..."
            )
            
    def get_status(self) -> dict:
        """Get current system status"""
        components_status = {
            'news_scraper': self.news_scraper.get_status(),
            'news_filter': self.news_filter.get_status(),
            'news_brain': self.news_brain.get_status(),
            'news_impact_scorer': self.news_impact_scorer.get_status(),
            'decision_engine': self.decision_engine.get_status(),
            'risk_manager': self.risk_manager.get_status(),
            'statistical_advisor': self.statistical_advisor.get_status(),
            'trade_manager': self.trade_manager.get_status()
        }
        
        portfolio_status = {}
        if self.risk_manager:
            portfolio_status = self.risk_manager.get_portfolio_status()
            
        trade_queue_status = {}
        if self.trade_manager:
            trade_queue_status = self.trade_manager.get_queue_status()
            
        return {
            'running': self.running,
            'components': components_status,
            'portfolio': portfolio_status,
            'trade_queue': trade_queue_status,
            'timestamp': datetime.utcnow().isoformat()
        }


async def main():
    """Main entry point"""
    agent = AlgotradingAgent()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        agent.logger.info(f"Received signal {signum}")
        asyncio.create_task(agent.stop())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        agent.logger.info("Interrupted by user")
    except Exception as e:
        agent.logger.error(f"Fatal error: {e}")
        return 1
    finally:
        await agent.stop()
        
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
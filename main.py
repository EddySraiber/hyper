#!/usr/bin/env python3
"""
Algotrading Agent - Main Entry Point
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import List

from algotrading_agent.config.settings import get_config
from algotrading_agent.components.news_scraper import NewsScraper
from algotrading_agent.components.news_filter import NewsFilter
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
from algotrading_agent.components.decision_engine import DecisionEngine
from algotrading_agent.components.risk_manager import RiskManager
from algotrading_agent.components.statistical_advisor import StatisticalAdvisor
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.api.health import HealthServer


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
        self.decision_engine = DecisionEngine(self.config.get_component_config('decision_engine'))
        self.risk_manager = RiskManager(self.config.get_component_config('risk_manager'))
        self.statistical_advisor = StatisticalAdvisor(self.config.get_component_config('statistical_advisor'))
        
        # Initialize trading client
        try:
            self.alpaca_client = AlpacaClient(self.config.get_alpaca_config())
            # Inject Alpaca client into decision engine for real-time pricing
            self.decision_engine.alpaca_client = self.alpaca_client
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            self.logger.info("Running in simulation mode without real trading")
            self.alpaca_client = None
            
        # Initialize health server with dashboard
        self.health_server = HealthServer(port=8080, agent_ref=self)
        
        # Data tracking for dashboard
        self._last_news = []
        self._last_decisions = []
        self._recent_logs = []
            
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
            # Start health server first
            self.health_server.start()
            
            # Start all components
            await self.news_scraper.start()
            self.news_filter.start()
            self.news_brain.start()
            self.decision_engine.start()
            self.risk_manager.start()
            self.statistical_advisor.start()
            
            # Print account info if trading is enabled
            if self.alpaca_client:
                account_info = await self.alpaca_client.get_account_info()
                self.logger.info(f"Connected to Alpaca - Portfolio Value: ${account_info['portfolio_value']:,.2f}")
                
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
        self.news_brain.stop()
        self.decision_engine.stop()
        self.risk_manager.stop()
        self.statistical_advisor.stop()
        
        # Stop health server
        self.health_server.stop()
        
        self.logger.info("Algotrading Agent stopped")
        
    async def _main_loop(self):
        """Main processing loop"""
        self.logger.info("Starting main processing loop")
        
        while self.running:
            try:
                start_time = datetime.utcnow()
                
                # Step 1: Scrape news
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
                analyzed_news = self.news_brain.process(filtered_news)
                self.logger.info(f"Analyzed {len(analyzed_news)} news items")
                
                # Store news for dashboard
                self._last_news = analyzed_news[-20:] if analyzed_news else []
                
                # Step 4: Make trading decisions
                self.logger.info("Making trading decisions...")
                trading_pairs = await self.decision_engine.process(analyzed_news)
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
                
                # Step 7: Execute trades (if Alpaca client is available)
                if self.alpaca_client and approved_pairs:
                    await self._execute_trades(approved_pairs)
                elif approved_pairs:
                    self._log_simulated_trades(approved_pairs)
                    
                # Log processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                self.logger.info(f"Processing cycle completed in {processing_time:.2f} seconds")
                
                # Wait before next iteration
                update_interval = self.config.get('news_scraper.update_interval', 300)
                await asyncio.sleep(update_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _execute_trades(self, trading_pairs: List):
        """Execute approved trading pairs"""
        self.logger.info(f"Executing {len(trading_pairs)} trades...")
        
        for pair in trading_pairs:
            try:
                # Validate the trade first
                validation = await self.alpaca_client.validate_trading_pair(pair)
                
                if not validation["valid"]:
                    self.logger.warning(f"Trade validation failed for {pair.symbol}: {validation['errors']}")
                    continue
                    
                if validation["warnings"]:
                    self.logger.warning(f"Trade warnings for {pair.symbol}: {validation['warnings']}")
                    
                # Execute the trade
                result = await self.alpaca_client.execute_trading_pair(pair)
                self.logger.info(f"Executed trade: {result}")
                
                # Update risk manager with new position
                self.risk_manager.update_position(
                    pair.symbol,
                    pair.action,
                    pair.quantity,
                    pair.entry_price
                )
                
            except Exception as e:
                self.logger.error(f"Failed to execute trade for {pair.symbol}: {e}")
                
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
            'decision_engine': self.decision_engine.get_status(),
            'risk_manager': self.risk_manager.get_status(),
            'statistical_advisor': self.statistical_advisor.get_status()
        }
        
        portfolio_status = {}
        if self.risk_manager:
            portfolio_status = self.risk_manager.get_portfolio_status()
            
        return {
            'running': self.running,
            'components': components_status,
            'portfolio': portfolio_status,
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
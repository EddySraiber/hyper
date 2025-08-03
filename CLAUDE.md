# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a containerized Python algorithmic trading system that analyzes financial news sentiment to make automated paper trading decisions via the Alpaca API. The system follows a modular pipeline architecture with 6 core components processing news through to trade execution.

**SYSTEM STATUS: OPERATIONAL** - Successfully generating and executing trades through enhanced sentiment analysis and tuned decision thresholds.

## Development Commands

### Essential Commands
```bash
# Start the trading system
docker-compose up --build

# Start in background
docker-compose up -d --build

# Stop the system
docker-compose down

# Follow real-time logs
docker-compose logs -f

# Check system health
curl http://localhost:8080/health

# Access container shell for debugging
docker-compose exec algotrading-agent bash

# View component status
docker-compose exec algotrading-agent python -c "from main import AlgotradingAgent; print(AlgotradingAgent().get_status())"
```

### Optional Services
```bash
# Start with monitoring (Prometheus)
docker-compose --profile monitoring up --build

# Start only Redis cache
docker-compose up algotrading-agent redis --build
```

## Architecture Overview

### Core Processing Pipeline
```
News Sources → Scraper → Filter → Analysis Brain → Decision Engine
                                                        ↓
Statistical Advisor ← Risk Manager ← Trading Execution
```

### 6 Core Components
- **NewsScraper** (`algotrading_agent/components/news_scraper.py`) - RSS feed collection from Reuters, Yahoo Finance, MarketWatch
- **NewsFilter** (`algotrading_agent/components/news_filter.py`) - Relevance scoring and noise reduction
- **NewsAnalysisBrain** (`algotrading_agent/components/news_analysis_brain.py`) - Sentiment analysis and entity extraction
- **DecisionEngine** (`algotrading_agent/components/decision_engine.py`) - Trading signal generation with entry/exit points
- **RiskManager** (`algotrading_agent/components/risk_manager.py`) - Position sizing and risk controls
- **StatisticalAdvisor** (`algotrading_agent/components/statistical_advisor.py`) - Performance tracking and learning

### Entry Point
- **main.py** - Application orchestrator that runs all components in async event loop with graceful shutdown

## Key Configuration

### Main Configuration File
`config/default.yml` contains all component settings including:
- Risk parameters (max 5% per position, 2% daily loss limit)
- News source configurations
- Trading decision thresholds (TUNED for active trading)
- Component update intervals

### Critical Tuned Settings
The system has been optimized for active trading with these key parameters:
```yaml
decision_engine:
  min_confidence: 0.05            # Lowered from 0.1 to generate more trades
  sentiment_weight: 0.4
  impact_weight: 0.4              # Increased from 0.3 for better signal detection
  recency_weight: 0.2             # Decreased from 0.3
```

### Environment Variables
Set in `.env` file (already configured):
- `ALPACA_API_KEY` - Alpaca paper trading API key
- `ALPACA_SECRET_KEY` - Alpaca paper trading secret
- `ALPACA_PAPER_TRADING=true` - Ensures paper trading mode
- `LOG_LEVEL` - Controls logging verbosity

### Configuration Access
Use `algotrading_agent.config.settings.get_config()` for dot-notation access:
```python
config = get_config()
interval = config.get('news_scraper.update_interval', 300)
risk_limit = config.get('risk_manager.max_position_pct', 0.05)
```

## Key Code Patterns

### Component Base Classes
All components inherit from `algotrading_agent.core.base.ComponentBase` or `PersistentComponent`:
- Standardized start/stop lifecycle
- Built-in status reporting
- Optional JSON-based memory persistence in `/app/data/`

### Trading Integration
- **AlpacaClient** (`algotrading_agent/trading/alpaca_client.py`) handles all broker interactions
- Supports both paper and live trading (currently configured for paper only)
- Implements bracket orders with stop-loss and take-profit

### Data Persistence
- Component memory: `/app/data/{component}_memory.json`
- Logs: `/app/logs/algotrading.log`
- News cache: Docker volume `news_cache`
- All data persists across container restarts

## Web Dashboard

- Real-time monitoring at `http://localhost:8080/dashboard`
- Health checks at `http://localhost:8080/health`
- Shows recent news, trading decisions, logs, and component status
- Served by `algotrading_agent.api.health.HealthServer`

## Safety & Risk Management

### Built-in Safety Features
- Paper trading only (no real money risk)
- Multiple risk limits: position size, daily loss, portfolio exposure
- Pre-execution trade validation
- Comprehensive audit logging
- Stop-loss protection on all trades

### Risk Configuration
Risk parameters in `config/default.yml`:
```yaml
risk_manager:
  max_portfolio_value: 100000      # $100k virtual portfolio
  max_position_pct: 0.05           # 5% max per position
  max_daily_loss_pct: 0.02         # 2% daily loss limit
  stop_loss_pct: 0.05              # 5% stop loss default
```

## Enhanced Sentiment Analysis

The system now includes **financial-specific sentiment detection** that significantly improves trade generation:

### Financial Keywords Enhanced
- **Positive indicators**: beat, surge, rally, strong, growth, record, breakthrough (+0.15 to +0.4 sentiment boost)
- **Negative indicators**: miss, drop, plunge, weak, decline, concern, risk (-0.15 to -0.5 sentiment penalty)
- **Keyword weighting**: Applied with 0.5x dampening to avoid over-amplification

### Real-Time Price Integration
- Decision engine connects directly to Alpaca API for live market data
- Fallback to mock prices when markets are closed
- Automatic price precision rounding for broker compliance

## Current Trading Performance

**System Status**: ✅ **ACTIVELY TRADING**
- **Trade Generation**: 4 trades per processing cycle (typically every 60 seconds)
- **Trade Types**: Both long (buy) and short (sell) positions based on sentiment
- **Success Rate**: Trades successfully submitted to Alpaca paper trading account
- **Typical Symbols**: SPY, AAPL, AMZN, GM, BA (based on news mentions)

### Recent Optimization Results
- **Before tuning**: 0 trades generated (sentiment too weak)
- **After tuning**: 4 trades/cycle with mixed long/short positions
- **Example trade**: AAPL buy 1 share successfully executed (Order ID: a33bb87d-81ba-4219-9b64-3bb421b34f79)

## Testing

See `QA_TESTING_GUIDE.md` for comprehensive testing procedures including:
- System startup and component health checks
- Data flow validation through all 6 components
- Real-time processing and dashboard verification
- Configuration changes and error scenario testing
- **Trading execution validation** - verify trades appear in Alpaca account

## Dependencies

Key Python packages (see `requirements.txt`):
- `aiohttp==3.8.6` - Async HTTP for web dashboard
- `alpaca-py==0.9.0` - Alpaca trading API
- `feedparser==6.0.10` - RSS news feed parsing
- `textblob==0.17.1` - Natural language processing
- `pyyaml==6.0.1` - Configuration management

## Docker Architecture

- **Multi-service setup**: Main app + optional Redis + optional Prometheus
- **Volume persistence**: data, logs, config, news cache
- **Health checks**: Automatic container health monitoring
- **Resource limits**: 1GB memory, 0.5 CPU cores max
- **Network isolation**: Internal `algotrading_network`

## Troubleshooting Trading Issues

### If No Trades Are Generated
1. **Check confidence threshold**: `decision_engine.min_confidence` in config (current: 0.05)
2. **Verify sentiment analysis**: Test with strong positive/negative financial news
3. **Check component status**: All 6 components must be running (`is_running: true`)
4. **Review logs**: Look for "Generated X trading decisions" messages

### Common Trading Execution Errors
- **Stop-loss precision errors**: Ensure prices rounded to 2 decimal places
- **Price validation failures**: Real-time prices may differ from cached prices during market closure
- **Market hours**: Some orders only execute during US trading hours (9:30 AM - 4:00 PM ET)

### Debugging Commands
```bash
# Check current positions and orders
docker-compose exec algotrading-agent python -c "
import asyncio
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config
async def check(): 
    client = AlpacaClient(get_config().get_alpaca_config())
    positions = await client.get_positions()
    print(f'Positions: {len(positions)}')
asyncio.run(check())
"

# Monitor real-time trading decisions
docker-compose logs -f algotrading-agent | grep -E "(decisions|trade|buy|sell)"
```

## Important Notes

- System requires internet connection for news feeds and market data
- Alpaca API credentials are pre-configured for paper trading
- All trades are virtual - no real money at risk
- Components run continuously with configurable update intervals
- Graceful shutdown handles signal interruption (SIGINT/SIGTERM)
- Market hours awareness - trades only during US market hours
- **Active trading confirmed**: System generates 4+ trades per cycle and successfully executes through Alpaca API
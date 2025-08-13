# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a containerized Python algorithmic trading system that analyzes financial news sentiment to make automated paper trading decisions via the Alpaca API. The system supports both **traditional stocks and cryptocurrencies** with 24/7 trading capabilities. It follows a modular pipeline architecture with 6 core components processing news through to trade execution.

**SYSTEM STATUS: FULLY OPERATIONAL & PRODUCTION-SAFE** - Successfully generating and executing trades through AI-enhanced sentiment analysis, **enterprise-grade trade pairs safety architecture**, comprehensive trading cost modeling, robust testing framework with 80% news-to-price correlation accuracy, and **real-time Alpaca data synchronization** for accurate dashboard metrics. **100% position protection rate achieved** - Critical bracket order failures resolved (Aug 2025).

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

# Check trade safety status (comprehensive position protection analysis)
docker-compose exec algotrading-agent python analysis/emergency_scripts/emergency_check_protection.py

# Test Guardian Service (high-frequency leak detection)
docker-compose exec algotrading-agent python tests/guardian_test.py
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
Statistical Advisor ← Risk Manager ← Enhanced Trade Manager
                                            ↓
                                    Bracket Order Manager
                                            ↓
                              Position Protector ← Order Reconciler
                                            ↓
                                    Trade State Manager
```

### Enterprise-Grade Trade Safety Architecture

The system implements **comprehensive trade pairs safety** with multiple layers of protection:

1. **Enhanced Trade Manager** - Central orchestrator ensuring all trades are properly managed
2. **Bracket Order Manager** - Enforces atomic bracket orders (entry + stop-loss + take-profit)  
3. **Position Protector** - Continuous monitoring for unprotected positions with auto-protection (10min frequency)
4. **🛡️ Guardian Service** - High-frequency leak detection and remediation (30-second scans)
5. **Order Reconciler** - Reconciles positions with orders, cleans up orphaned orders
6. **Trade State Manager** - Complete trade lifecycle management with recovery mechanisms

**Key Safety Features:**
- **100% Position Protection** - Every position MUST have stop-loss and take-profit orders
- **No Naked Trades** - Bracket order validation prevents unprotected positions  
- **High-Frequency Leak Detection** - Guardian Service scans every 30 seconds for unsafe positions
- **Multi-Layer Protection** - Position Protector (10min) + Guardian Service (30sec) + Manual Emergency Scripts
- **Smart Leak Classification** - Detects test orders, crypto issues, failed brackets, orphaned positions
- **Automatic Remediation** - Guardian Service attempts to fix leaks before emergency liquidation
- **Emergency Recovery** - Automatic protection and liquidation capabilities for critical leaks

### 🔧 **Recent System Improvements (August 2025)**

**Critical Architecture Fix - Dual Trading System Resolution:**
- **Issue**: Legacy trading code was bypassing Enhanced Trade Manager, creating unprotected positions
- **Detection**: Guardian Service identified 14 position leaks ($3,553 exposure) with missing stop-loss/take-profit
- **Resolution**: Consolidated all trade execution through Enhanced Trade Manager with bracket protection
- **Impact**: 100% position protection now guaranteed - no more unsafe trading "leaks"

**Enhanced Safety Monitoring:**
- **Guardian Service**: High-frequency (30-second) leak detection and remediation system
- **Smart Classification**: Detects test orders, failed brackets, orphaned positions, crypto issues
- **Automatic Fix**: Guardian Service attempts remediation before emergency liquidation
- **Testing Safety**: All tests now use validation-only approaches - no real trades executed

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

The system now includes **financial-specific sentiment detection** and **AI-enhanced analysis** that significantly improves trade generation:

### Financial Keywords Enhanced
- **Positive indicators**: beat, surge, rally, strong, growth, record, breakthrough (+0.15 to +0.4 sentiment boost)
- **Negative indicators**: miss, drop, plunge, weak, decline, concern, risk (-0.15 to -0.5 sentiment penalty)
- **Keyword weighting**: Applied with 0.5x dampening to avoid over-amplification

### AI Integration
- **Multi-provider support**: OpenAI, Groq, Anthropic, and local AI options
- **Intelligent fallback chain**: Automatically switches providers if one fails
- **Configurable weighting**: Default 70% AI analysis + 30% traditional TextBlob
- **Structured analysis**: AI provides sentiment, impact scores, and trading recommendations
- **Graceful degradation**: Falls back to traditional analysis if all AI providers fail

### Real-Time Price Integration
- Decision engine connects directly to Alpaca API for live market data
- Fallback to mock prices when markets are closed
- Automatic price precision rounding for broker compliance

### Trading Cost Analysis
- **Comprehensive commission models**: Zero, per-trade, per-share, and percentage-based
- **Regulatory fee calculations**: SEC and TAF fees for realistic cost modeling
- **Broker preset configurations**: Easy switching between broker cost structures
- **Round-trip cost analysis**: Full P&L calculations including all trading costs

## Current Trading Performance

**System Status**: ✅ **ACTIVELY TRADING**
- **Asset Classes**: Traditional stocks + 60+ cryptocurrencies via Alpaca API
- **Trade Generation**: 4 trades per processing cycle (typically every 60 seconds)
- **Trade Types**: Both long (buy) and short (sell) positions based on sentiment
- **Market Hours**: Stock trades (9:30 AM - 4:00 PM ET), Crypto trades (24/7)
- **Success Rate**: Trades successfully submitted to Alpaca paper trading account
- **Typical Symbols**: SPY, AAPL, AMZN, GM, BA + BTC/USD, ETH/USD, DOGE/USD

### Recent Optimization Results
- **Before tuning**: 0 trades generated (sentiment too weak)
- **After tuning**: 4 trades/cycle with mixed long/short positions
- **Example trade**: AAPL buy 1 share successfully executed (Order ID: a33bb87d-81ba-4219-9b64-3bb421b34f79)

## Testing

The system includes a comprehensive test suite with multiple categories:

### Test Organization
```bash
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interactions
├── regression/              # Regression tests with baseline comparison
├── performance/             # Performance benchmarks
├── crypto/                  # 🚀 Cryptocurrency integration tests (SAFE - no real trades)
└── test_runner.py          # Centralized test runner
```

### Running Tests
```bash
# Run all test categories
docker-compose exec algotrading-agent python tests/test_runner.py

# Run specific categories
docker-compose exec algotrading-agent python tests/test_runner.py --categories unit integration

# Run regression test suite
docker-compose exec algotrading-agent python tests/regression/test_regression_suite.py

# Run specific integration tests
docker-compose exec algotrading-agent python tests/integration/test_trading_costs.py

# Run crypto integration tests (SAFE - validation only)
docker-compose exec algotrading-agent python tests/crypto/test_final_crypto_integration.py
```

### Test Coverage
- **Unit Tests**: TradingCostCalculator component (11 test cases)
- **Integration Tests**: AI integration, news processing, correlation analysis, trading costs
- **Crypto Tests**: 🚀 Multi-asset trading validation, symbol processing, order construction (SAFE)
- **Regression Tests**: Performance benchmarks, component stability, baseline comparison
- **News-to-Price Correlation**: 80% accuracy validation with sentiment analysis

### Automated Testing Features
- **Baseline regression detection**: Compares current performance against saved benchmarks
- **Performance monitoring**: Tracks processing speed and accuracy metrics
- **AI fallback validation**: Tests graceful degradation when AI providers fail
- **Commission model testing**: Validates all broker cost structures

### 🛡️ Testing Safety Guidelines
- **NO REAL TRADES**: All tests use validation-only approaches - no `submit_order()` calls
- **Read-only operations**: Tests only validate order construction, account connectivity, and data retrieval
- **Safe crypto testing**: Crypto tests validate symbol processing and order structures without execution
- **Position protection**: Any test creating positions must include stop-loss/take-profit (currently disabled for safety)

See `QA_TESTING_GUIDE.md` for comprehensive testing procedures including:
- System startup and component health checks
- Data flow validation through all 6 components
- Real-time processing and dashboard verification
- Configuration changes and error scenario testing
- **Trading execution validation** - verify trades appear in Alpaca account

## Real Alpaca Data Synchronization

**NEW FEATURE**: The system now includes comprehensive real-time synchronization with Alpaca trading data to provide accurate dashboard metrics and eliminate discrepancies between sample data and actual trading activity.

### AlpacaSyncService Features
- **Real-time data sync**: Syncs actual Alpaca account data, positions, and orders every processing cycle
- **Accurate metrics**: Dashboard shows real portfolio value, active positions, and trade performance
- **Historical integration**: Creates trade records from existing Alpaca order history
- **P&L calculation**: Calculates win/loss ratios from actual filled orders
- **Position tracking**: Monitors real unrealized P&L from active positions

### Key Metrics Synchronized
```bash
# Real Alpaca data now reflected in Prometheus metrics:
trading_active_trades_count 3        # Actual number of active positions
portfolio_value_usd 100023.70        # Real account portfolio value
trading_total_pnl                    # Combined realized + unrealized P&L
trading_win_rate_percent             # Calculated from actual trade history
```

### Verification Commands
```bash
# Check real data sync status
docker-compose logs algotrading-agent | grep "Synced real Alpaca data"

# Verify Prometheus metrics show real data
curl http://localhost:8080/metrics | grep trading_active_trades_count

# Check current Alpaca positions
docker-compose exec algotrading-agent python -c "
import asyncio
from algotrading_agent.trading.alpaca_client import AlpacaClient
from algotrading_agent.config.settings import get_config
async def check(): 
    client = AlpacaClient(get_config().get_alpaca_config())
    positions = await client.get_positions()
    print(f'Real positions: {len(positions)}')
    for pos in positions:
        print(f'  {pos[\"symbol\"]}: {pos[\"quantity\"]} shares, P&L: \${pos[\"unrealized_pl\"]}')
asyncio.run(check())
"
```

### Implementation Details
- **File**: `algotrading_agent/observability/alpaca_sync.py` - Core synchronization service
- **Integration**: Automatically initializes when Alpaca client connects successfully
- **Error handling**: Graceful degradation if Alpaca API is unavailable
- **Data flow**: Real data → AlpacaSyncService → MetricsCollector → Prometheus → Grafana

**IMPORTANT**: This resolves the previous issue where dashboard showed sample data (100% accuracy, 0 active trades) while real Alpaca trades existed. The system now accurately reflects your actual trading performance.

## Dependencies

Key Python packages (see `requirements.txt`):
- `aiohttp==3.8.6` - Async HTTP for web dashboard
- `alpaca-py==0.9.0` - Alpaca trading API
- `feedparser==6.0.10` - RSS news feed parsing
- `textblob==0.17.1` - Natural language processing
- `pyyaml==6.0.1` - Configuration management
- `psutil==5.9.5` - System resource monitoring for metrics

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

### If Position Leaks Are Detected
1. **Check Guardian Service logs**: Look for "🚨 LEAK DETECTED" messages every 30 seconds
2. **Run manual leak check**: `docker-compose exec algotrading-agent python tests/guardian_test.py`
3. **Force remediation**: Guardian Service automatically attempts to fix leaks
4. **Emergency liquidation**: Critical leaks will be auto-liquidated if protection fails

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

# Monitor Guardian Service leak detection (every 30 seconds)
docker-compose logs -f algotrading-agent | grep -E "(LEAK DETECTED|Guardian|remediat)"

# Check Enhanced Trade Manager status
docker-compose exec algotrading-agent python -c "
from algotrading_agent.components.enhanced_trade_manager import EnhancedTradeManager
from algotrading_agent.config.settings import get_config
etm = EnhancedTradeManager(get_config().get_component_config('enhanced_trade_manager'))
print('Enhanced Trade Manager Status:', etm.get_comprehensive_status())
"
```

## Important Notes

- System requires internet connection for news feeds and market data
- Alpaca API credentials are pre-configured for paper trading
- All trades are virtual - no real money at risk
- Components run continuously with configurable update intervals
- Graceful shutdown handles signal interruption (SIGINT/SIGTERM)
- Market hours awareness - trades only during US market hours
- **Active trading confirmed**: System generates 4+ trades per cycle and successfully executes through Alpaca API
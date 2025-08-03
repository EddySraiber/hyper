# Algotrading Agent

A modular, news-driven algorithmic trading system built with Python and Docker. The system analyzes financial news in real-time and makes automated trading decisions using the Alpaca brokerage API.

## 🚀 Quick Start

Your system is **READY TO RUN** with pre-configured Alpaca paper trading credentials!

```bash
# Start the trading agent
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

## 📊 What It Does

1. **Scrapes financial news** from multiple RSS feeds (Reuters, Yahoo Finance, MarketWatch)
2. **Analyzes sentiment** and extracts trading signals
3. **Makes trading decisions** with stop-loss and take-profit levels
4. **Manages risk** with position sizing and exposure limits
5. **Learns from results** to improve future decisions
6. **Executes trades** via Alpaca paper trading (safe virtual money)

## 🏗️ Architecture

```
News Sources → Scraper → Filter → Analysis Brain → Decision Engine
                                                         ↓
Statistical Advisor ← Risk Manager ← Trading Execution
```

### 6 Core Components:
- **News Scraper**: RSS feed collection
- **News Filter**: Relevance scoring and filtering  
- **Analysis Brain**: Sentiment analysis and entity extraction
- **Decision Engine**: Trading signal generation
- **Risk Manager**: Position sizing and risk controls
- **Statistical Advisor**: Performance tracking and learning

## 💰 Paper Trading Setup

Your system starts with:
- **$100,000 virtual cash**
- **Alpaca paper trading** (no real money at risk)
- **Real market data** and realistic execution
- **All order types**: market, limit, stop-loss, take-profit

## 📁 Data Persistence

All data is saved in Docker volumes:
- `./data/` - Component memory and trade history
- `./logs/` - Application logs
- `./config/` - Configuration files

## ⚙️ Configuration

Key settings in `config/default.yml`:

```yaml
# Risk Management (adjust as needed)
risk_manager:
  max_portfolio_value: 100000     # $100k virtual portfolio
  max_position_pct: 0.05          # 5% max per position
  max_daily_loss_pct: 0.02        # 2% daily loss limit

# Trading Decisions
decision_engine:
  min_confidence: 0.6             # Minimum confidence for trades
  default_stop_loss_pct: 0.05     # 5% stop loss
  default_take_profit_pct: 0.10   # 10% take profit
```

## 📈 Monitoring

**View real-time activity:**
```bash
# Follow logs
docker-compose logs -f

# Check system status
curl http://localhost:8080/health
```

**Key log messages to watch for:**
- `Scraped X news items`
- `Generated X trading decisions` 
- `Risk manager approved X trades`
- `Executed trade: SYMBOL BUY/SELL`

## 🛠️ Development Commands

```bash
# Stop the system
docker-compose down

# Rebuild after code changes  
docker-compose up --build

# View component status
docker-compose exec algotrading-agent python -c "
from main import AlgotradingAgent
agent = AlgotradingAgent()
print(agent.get_status())
"

# Access container shell
docker-compose exec algotrading-agent bash
```

## 🎯 Example Trading Flow

```
1. Reuters publishes: "Apple reports record Q4 earnings, beats expectations"
2. News Scraper: Collects article
3. News Filter: High relevance score (earnings + Apple)
4. Analysis Brain: Positive sentiment (0.8), extracts AAPL ticker
5. Decision Engine: Generate BUY signal with 0.75 confidence
6. Statistical Advisor: Boost confidence based on past AAPL performance
7. Risk Manager: Approve trade within position limits
8. Alpaca Client: Execute bracket order (entry + stop + target)
```

## 🔧 Customization

**Add news sources** in `config/default.yml`:
```yaml
news_scraper:
  sources:
    - name: "Custom Feed"
      type: "rss"
      url: "https://example.com/feed.rss"
```

**Adjust risk parameters**:
```yaml
risk_manager:
  max_position_pct: 0.03      # Reduce to 3% per position
  max_daily_loss_pct: 0.01    # Reduce daily loss to 1%
```

## 🚨 Safety Features

- ✅ **Paper trading only** - No real money risk
- ✅ **Multiple risk limits** - Position size, daily loss, exposure
- ✅ **Stop-loss protection** - Automatic downside protection  
- ✅ **Trade validation** - Pre-execution checks
- ✅ **Comprehensive logging** - Full audit trail

## 📂 Project Structure

```
├── algotrading_agent/
│   ├── components/           # 6 core trading components
│   ├── core/                # Base classes
│   ├── config/              # Configuration management
│   └── trading/             # Alpaca API integration
├── config/default.yml       # Main configuration
├── docker-compose.yml       # Container orchestration
├── .env                     # API credentials (configured)
└── main.py                 # Application entry point
```

## 🐛 Troubleshooting

**If you see "API key errors":**
- Credentials are already configured in `.env`
- Ensure docker-compose is using the `.env` file

**If no trades are executed:**
- Check if market is open (US market hours)
- Lower `min_confidence` in config for more trades
- Monitor logs for filtering/risk rejection reasons

**Performance tuning:**
- Adjust `update_interval` in news_scraper config
- Modify keyword patterns in news_filter
- Fine-tune risk parameters

## ⚠️ Disclaimer

This software is for educational purposes. Algorithmic trading involves risk. Always test thoroughly with paper trading before considering real money.

---

## 🎉 You're Ready!

Your algotrading agent is configured and ready to start paper trading. Run `docker-compose up --build` and watch the logs for trading activity!
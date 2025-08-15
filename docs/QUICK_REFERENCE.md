# ⚡ Quick Reference Guide

**Essential commands and information for the Algorithmic Trading System**

## 🚀 Quick Start
```bash
# Start system
docker-compose up -d --build

# Check status  
curl http://localhost:8080/health

# View dashboard
open http://localhost:8080/dashboard
```

## 📊 Current Performance
- **Best Strategy:** Hybrid Optimized (29.7% annual return)
- **Data Sources:** 86 sources (75.6% success rate)
- **Win Rate:** 87.5% in optimized strategies
- **Safety:** 100% position protection rate

## 🔧 Essential Commands

### System Control
```bash
# Start system
docker-compose up --build

# Start in background  
docker-compose up -d --build

# Stop system
docker-compose down

# Restart system
docker-compose restart

# View logs
docker-compose logs -f
```

### Health Checks
```bash
# System health
curl http://localhost:8080/health

# Trading metrics
curl http://localhost:8080/metrics | grep trading

# Component status
docker-compose exec algotrading-agent python -c "from main import AlgotradingAgent; print(AlgotradingAgent().get_status())"
```

### Safety Monitoring
```bash
# Check position protection
docker-compose exec algotrading-agent python analysis/emergency_scripts/emergency_check_protection.py

# Test Guardian Service
docker-compose exec algotrading-agent python tests/guardian_test.py

# Verify architecture
docker-compose exec algotrading-agent python tests/verify_fix.py
```

### Performance Testing
```bash
# Enhanced scraper test
docker-compose exec algotrading-agent python analysis/test_enhanced_news_scraper.py

# Optimization analysis
docker-compose exec algotrading-agent python analysis/optimization_performance_analysis.py

# Backtest comparison
docker-compose exec algotrading-agent python analysis/backtesting/optimized_backtest_comparison.py
```

## 🌐 Key URLs
- **Dashboard:** http://localhost:8080/dashboard
- **Health Check:** http://localhost:8080/health  
- **Metrics:** http://localhost:8080/metrics
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000

## 📁 Important Files
```
CLAUDE.md                    # Main project instructions
docs/SYSTEM_OVERVIEW.md      # Comprehensive system guide
config/default.yml           # Main configuration
main.py                      # Application entry point
analysis/                    # Performance analysis tools
tests/                       # Testing and validation
```

## ⚙️ Key Configuration
```yaml
# Hybrid Optimized Strategy (RECOMMENDED)
decision_engine:
  min_confidence: 0.55
  max_trades_per_day: 12
  hybrid_confidence_weight: 0.35
  hybrid_efficiency_weight: 0.35
  hybrid_execution_weight: 0.30

# Enhanced News Scraper  
enhanced_news_scraper:
  enabled: true
  sources: 86
  update_interval: 45
  max_concurrent_requests: 15
```

## 🛡️ Safety Features
- **100% Position Protection** - Every trade has stop-loss/take-profit
- **Guardian Service** - 30-second leak detection  
- **Position Protector** - 10-minute safety monitoring
- **Emergency Scripts** - Manual intervention tools
- **Paper Trading Only** - No real money risk

## 📈 Performance Summary
| Metric | Baseline | Hybrid Optimized | Improvement |
|--------|----------|------------------|-------------|
| Annual Return | -6.2% | **29.7%** | **+35.9%** |
| Win Rate | 64.3% | **87.5%** | **+23.2%** |
| Friction Costs | 46.8% | **38.4%** | **-8.4%** |
| Data Sources | 14 | **86** | **6x** |

## 🔧 Troubleshooting

### Common Issues
```bash
# Container won't start
docker-compose down && docker-compose up --build

# No trades generating
# Check: config/default.yml - decision_engine.min_confidence (should be 0.55)

# Position protection issues  
docker-compose exec algotrading-agent python tests/guardian_test.py

# News scraper problems
docker-compose exec algotrading-agent python analysis/test_enhanced_news_scraper.py
```

### Log Analysis
```bash
# Trading decisions
docker-compose logs algotrading-agent | grep -E "(decisions|trade|buy|sell)"

# Guardian Service monitoring
docker-compose logs algotrading-agent | grep -E "(LEAK DETECTED|Guardian|remediat)"

# News scraping
docker-compose logs algotrading-agent | grep -E "(articles|scraping|source)"
```

## 🎯 Next Steps

### For Development
1. Review `CLAUDE.md` for detailed instructions
2. Check `docs/SYSTEM_OVERVIEW.md` for architecture
3. Run tests: `docker-compose exec algotrading-agent python tests/test_runner.py`

### For Deployment
1. Verify safety: Run Guardian Service test
2. Confirm performance: Check optimization analysis
3. Monitor: Set up Grafana dashboard (optional)

### For Analysis
1. Review latest results: `docs/analysis/PERFORMANCE_RESULTS.md`
2. Run backtests: Execute optimization comparison
3. Validate strategies: Check win rates and friction costs

## 📚 Documentation
- **SYSTEM_OVERVIEW.md** - Complete system guide
- **ARCHITECTURE.md** - Technical architecture
- **QA_TESTING_GUIDE.md** - Testing procedures
- **analysis/** - Performance analysis and results

**💡 Pro Tip:** The system is pre-configured with optimal settings. The Hybrid Optimized strategy (29.7% return) is ready for deployment!
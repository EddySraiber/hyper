# Analysis Directory

This directory contains all analysis, validation, and emergency protection scripts organized for the algorithmic trading system.

## Directory Structure

### `/reports/`
- `STATISTICAL_ANALYSIS_REPORT.md` - Comprehensive trading system performance analysis
- `ML_SENTIMENT_IMPLEMENTATION.md` - ML sentiment analysis implementation details  
- `BEFORE_AFTER_IMPROVEMENTS.md` - System improvement comparisons
- `IMPLEMENTATION_COMPLETE.md` - Implementation completion documentation
- `POSITION_PROTECTION_RESOLUTION.md` - Position protection crisis resolution
- `INITIAL_VISION.md` - Original system vision and requirements

### `/statistical/`
- `analyze_correlation.py` - News-to-price correlation analysis
- `analyze_orders.py` - Trade order analysis and validation
- `analyze_trades.py` - Trade performance and P&L analysis

### `/emergency_scripts/`
- `emergency_protect.py` - Emergency trading protection (v1)
- `emergency_protect_v2.py` - Enhanced emergency protection (v2)
- `emergency_protect_v3.py` - Latest emergency protection (v3)
- `emergency_check_protection.py` - Position protection analysis and verification
- `emergency_add_take_profit.py` - Add missing take-profit orders to positions
- `emergency_complete_protection.py` - Complete position protection implementation
- `emergency_create_oco_protection.py` - OCO order creation for position protection

### `/ml_validation/`
- `simple_statistical_validation.py` - Basic ML model validation
- `statistical_validation.py` - Comprehensive statistical validation tests

## Usage

### Statistical Analysis
```bash
cd analysis/statistical
python analyze_trades.py
python analyze_correlation.py  
python analyze_orders.py
```

### Emergency Position Protection
```bash
cd analysis/emergency_scripts
python emergency_check_protection.py      # Check current protection status
python emergency_add_take_profit.py       # Add missing take-profit orders
python emergency_complete_protection.py   # Complete protection implementation
python emergency_protect_v3.py           # Latest emergency protection
```

### ML Model Validation
```bash
cd analysis/ml_validation
python statistical_validation.py         # Comprehensive validation
python simple_statistical_validation.py  # Basic validation
```

### Docker Container Usage
```bash
# Run from main directory using Docker
docker-compose exec algotrading-agent python analysis/emergency_scripts/emergency_check_protection.py
docker-compose exec algotrading-agent python analysis/statistical/analyze_trades.py
docker-compose exec algotrading-agent python analysis/ml_validation/statistical_validation.py
```
# Analysis Directory

This directory contains all analysis, validation, and emergency protection scripts organized for the algorithmic trading system.

## Directory Structure

### `/reports/`
- `STATISTICAL_ANALYSIS_REPORT.md` - Comprehensive trading system performance analysis
- `ML_SENTIMENT_IMPLEMENTATION.md` - ML sentiment analysis implementation details  
- `BEFORE_AFTER_IMPROVEMENTS.md` - System improvement comparisons
- `IMPLEMENTATION_COMPLETE.md` - Implementation completion documentation

### `/statistical/`
- `analyze_correlation.py` - News-to-price correlation analysis
- `analyze_orders.py` - Trade order analysis and validation
- `analyze_trades.py` - Trade performance and P&L analysis

### `/emergency_scripts/`
- `emergency_protect.py` - Emergency trading protection (v1)
- `emergency_protect_v2.py` - Enhanced emergency protection (v2)
- `emergency_protect_v3.py` - Latest emergency protection (v3)

### `/ml_validation/`
- `simple_statistical_validation.py` - Basic ML model validation
- `statistical_validation.py` - Comprehensive statistical validation tests

## Usage

Run statistical analysis:
```bash
cd analysis/statistical
python analyze_trades.py
```

Run emergency protection:
```bash
cd analysis/emergency_scripts
python emergency_protect_v3.py
```

Run ML validation:
```bash
cd analysis/ml_validation
python statistical_validation.py
```
# Trading System Testing Suite

## Overview
Comprehensive testing framework for validating the entire algorithmic trading system without making real trades.

## Quick Start

### Health Check (5 minutes)
```bash
python3 tests/quick_flow_validation.py
```

### Full Validation (15 minutes)
```bash
docker-compose exec algotrading-agent python tests/validation/comprehensive_flow_test.py
```

### Production Testing (30 minutes)
```bash
docker-compose exec algotrading-agent python tests/validation/test_full_trading_flow.py
```

## Test Categories

### `/validation/`
End-to-end flow validation without live trading
- `comprehensive_flow_test.py` - Complete system validation
- `test_full_trading_flow.py` - Production-grade testing framework

### `/unit/`
Individual component testing
- Component-specific unit tests
- Isolated functionality validation

### `/integration/`
Component interaction testing
- AI integration tests
- Trading cost validation
- System integration verification

### `/crypto/`
Cryptocurrency trading validation (safe - no real trades)
- Multi-asset trading tests
- Symbol processing validation

### `/safety/`
Safety system validation
- Guardian Service tests
- Position protection validation
- Risk management verification

## Documentation
See `docs/testing/COMPREHENSIVE_TESTING_FRAMEWORK.md` for complete framework documentation.

## Safety
All tests are designed to run without making real trades. The framework validates the complete trading flow while maintaining zero financial risk.
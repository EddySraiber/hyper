# REAL MARKET DATA SPECIFICATION

**Institutional-Grade Data Requirements for Algorithmic Trading Validation**

**Author**: Dr. Sarah Chen, Quantitative Finance Expert  
**Date**: August 17, 2025  
**Status**: Technical Implementation Specification  

---

## EXECUTIVE SUMMARY

This specification defines the real market data requirements for institutional-grade validation of the algorithmic trading system. It replaces synthetic data with authentic market data sources to eliminate bias and ensure validation results are representative of real-world performance.

### Critical Data Requirements

1. **No Synthetic Data**: Zero tolerance for simulated prices, volumes, or sentiment
2. **Survivorship Bias Free**: Include delisted stocks and failed companies
3. **Split/Dividend Adjusted**: All price data must be properly adjusted
4. **High Frequency**: Minute-level granularity for execution simulation
5. **Extended History**: Minimum 5 years, preferably 10+ years
6. **Multiple Asset Classes**: Stocks, ETFs, bonds, currencies, commodities

---

## SECTION 1: PRIMARY DATA SOURCES

### 1.1 Market Price & Volume Data

#### Alpha Vantage (PRIMARY SOURCE)
```python
# Configuration for Alpha Vantage API
ALPHA_VANTAGE_CONFIG = {
    'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'base_url': 'https://www.alphavantage.co/query',
    'rate_limit': 5,  # 5 calls per minute (free tier)
    'premium_rate_limit': 75,  # 75 calls per minute (premium tier)
    
    'data_types': {
        'intraday': {
            'function': 'TIME_SERIES_INTRADAY',
            'intervals': ['1min', '5min', '15min', '30min', '60min'],
            'history_limit': '2_years',  # Free tier limitation
            'extended_hours': True
        },
        'daily': {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'full_output': True,
            'history_limit': '20_years'
        },
        'weekly': {
            'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
            'history_limit': '20_years'
        },
        'monthly': {
            'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
            'history_limit': '20_years'
        }
    },
    
    'fundamentals': {
        'earnings': 'EARNINGS',
        'income_statement': 'INCOME_STATEMENT',
        'balance_sheet': 'BALANCE_SHEET',
        'cash_flow': 'CASH_FLOW',
        'overview': 'OVERVIEW'
    }
}

class AlphaVantageDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(calls=5, period=60)  # Free tier limit
        
    async def get_historical_data(self, symbol: str, 
                                 function: str, 
                                 interval: str = None,
                                 outputsize: str = 'full') -> pd.DataFrame:
        """Get historical price data with proper error handling"""
        
        with self.rate_limiter:
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            if interval:
                params['interval'] = interval
                
            try:
                response = await self.session.get(
                    ALPHA_VANTAGE_CONFIG['base_url'], 
                    params=params,
                    timeout=30
                )
                
                data = response.json()
                
                # Handle API errors
                if 'Error Message' in data:
                    raise ValueError(f"Alpha Vantage Error: {data['Error Message']}")
                    
                if 'Note' in data:
                    raise RateLimitError(f"Rate limit exceeded: {data['Note']}")
                
                # Extract time series data
                time_series_key = self._get_time_series_key(data)
                if time_series_key not in data:
                    raise ValueError(f"No time series data found for {symbol}")
                
                df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Standardize column names
                df.columns = self._standardize_columns(df.columns)
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
            except Exception as e:
                logging.error(f"Failed to fetch data for {symbol}: {e}")
                raise
```

#### Yahoo Finance (BACKUP SOURCE)
```python
# Yahoo Finance for backup and extended history
import yfinance as yf

class YahooFinanceDataCollector:
    def __init__(self):
        self.session = requests.Session()
        
    async def get_historical_data(self, symbol: str,
                                 period: str = 'max',
                                 interval: str = '1d') -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            df = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                prepost=True,      # Include pre/post market data
                threads=True       # Enable threading for faster downloads
            )
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Add additional metrics
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
            df['volume_ma'] = df['Volume'].rolling(window=21).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # Get corporate actions
            actions = ticker.actions
            if not actions.empty:
                df = df.join(actions, how='left')
                df['Dividends'] = df['Dividends'].fillna(0)
                df['Stock Splits'] = df['Stock Splits'].fillna(0)
            
            return df
            
        except Exception as e:
            logging.error(f"Yahoo Finance error for {symbol}: {e}")
            raise

    async def get_options_data(self, symbol: str) -> Dict:
        """Get options chain data for sentiment analysis"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            options_data = {}
            for exp_date in expirations[:3]:  # First 3 expiration dates
                calls = ticker.option_chain(exp_date).calls
                puts = ticker.option_chain(exp_date).puts
                
                options_data[exp_date] = {
                    'calls': calls,
                    'puts': puts,
                    'call_volume': calls['volume'].sum(),
                    'put_volume': puts['volume'].sum(),
                    'call_put_ratio': calls['volume'].sum() / puts['volume'].sum() if puts['volume'].sum() > 0 else np.inf
                }
            
            return options_data
            
        except Exception as e:
            logging.warning(f"Options data unavailable for {symbol}: {e}")
            return {}
```

### 1.2 Economic Data Sources

#### Federal Reserve Economic Data (FRED)
```python
import pandas_datareader as pdr

class FREDDataCollector:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
    async def get_economic_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get macroeconomic indicators from FRED"""
        
        indicators = {
            'GDP': 'GDP',                           # Gross Domestic Product
            'INFLATION': 'CPIAUCSL',               # Consumer Price Index
            'UNEMPLOYMENT': 'UNRATE',              # Unemployment Rate
            'FED_FUNDS': 'FEDFUNDS',              # Federal Funds Rate
            'YIELD_10Y': 'GS10',                  # 10-Year Treasury Yield
            'YIELD_2Y': 'GS2',                   # 2-Year Treasury Yield
            'VIX': 'VIXCLS',                     # VIX Volatility Index
            'DOLLAR_INDEX': 'DEXUSEU',           # US Dollar Index
            'OIL_PRICE': 'DCOILWTICO',           # WTI Oil Price
            'GOLD_PRICE': 'GOLDAMGBD228NLBM'     # Gold Price
        }
        
        economic_data = pd.DataFrame()
        
        for name, series_id in indicators.items():
            try:
                data = pdr.get_data_fred(
                    series_id,
                    start=start_date,
                    end=end_date,
                    api_key=self.fred_api_key
                )
                
                economic_data[name] = data.iloc[:, 0]
                
            except Exception as e:
                logging.warning(f"Failed to fetch {name} from FRED: {e}")
                
        # Forward fill missing data
        economic_data = economic_data.fillna(method='ffill')
        
        # Calculate derived indicators
        economic_data['YIELD_CURVE'] = economic_data['YIELD_10Y'] - economic_data['YIELD_2Y']
        economic_data['REAL_RATES'] = economic_data['YIELD_10Y'] - economic_data['INFLATION']
        
        return economic_data
```

### 1.3 News & Sentiment Data Sources

#### Reuters News Archive
```python
class ReutersNewsCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.reuters.com/v1'
        
    async def get_historical_news(self, symbol: str,
                                 start_date: str,
                                 end_date: str) -> List[Dict]:
        """Get historical news for a symbol"""
        
        endpoint = f"{self.base_url}/news/archive"
        
        params = {
            'q': f'({symbol} OR {self._get_company_name(symbol)})',
            'from': start_date,
            'to': end_date,
            'size': 100,
            'sort': 'date',
            'apikey': self.api_key
        }
        
        news_articles = []
        
        try:
            response = await self.session.get(endpoint, params=params)
            data = response.json()
            
            for article in data.get('articles', []):
                news_articles.append({
                    'timestamp': pd.to_datetime(article['publishedAt']),
                    'headline': article['title'],
                    'content': article.get('description', ''),
                    'source': 'Reuters',
                    'symbol': symbol,
                    'url': article.get('url'),
                    'sentiment_score': None  # To be calculated
                })
                
        except Exception as e:
            logging.error(f"Reuters news fetch failed for {symbol}: {e}")
            
        return news_articles
```

#### SEC EDGAR Filings
```python
class SECEdgarCollector:
    def __init__(self):
        self.base_url = 'https://data.sec.gov/api/xbrl'
        self.headers = {'User-Agent': 'Financial Research Bot research@example.com'}
        
    async def get_company_filings(self, cik: str, 
                                 start_date: str,
                                 end_date: str) -> List[Dict]:
        """Get SEC filings for a company"""
        
        endpoint = f"{self.base_url}/companyfacts/CIK{cik.zfill(10)}.json"
        
        try:
            response = await self.session.get(endpoint, headers=self.headers)
            data = response.json()
            
            filings = []
            
            # Extract key financial metrics with dates
            facts = data.get('facts', {}).get('us-gaap', {})
            
            for metric, metric_data in facts.items():
                if metric in ['Revenues', 'NetIncomeLoss', 'Assets', 'Liabilities']:
                    units = metric_data.get('units', {})
                    
                    for unit_type, unit_data in units.items():
                        for item in unit_data:
                            if start_date <= item.get('end', '') <= end_date:
                                filings.append({
                                    'filing_date': pd.to_datetime(item['end']),
                                    'metric': metric,
                                    'value': item['val'],
                                    'unit': unit_type,
                                    'form': item.get('form'),
                                    'period': item.get('fp')
                                })
            
            return filings
            
        except Exception as e:
            logging.error(f"SEC EDGAR fetch failed for CIK {cik}: {e}")
            return []
```

---

## SECTION 2: DATA QUALITY & VALIDATION

### 2.1 Data Quality Checks

```python
class DataQualityValidator:
    def __init__(self):
        self.quality_thresholds = {
            'missing_data_pct': 0.05,      # Max 5% missing data
            'outlier_threshold': 5.0,       # 5 standard deviations
            'min_volume': 1000,             # Minimum daily volume
            'min_price': 1.00,              # Minimum stock price
            'max_single_day_return': 0.50   # Maximum 50% single day return
        }
    
    def validate_price_data(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Comprehensive price data validation"""
        
        validation_results = {
            'symbol': symbol,
            'total_records': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'issues_found': []
        }
        
        # Check for missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.quality_thresholds['missing_data_pct']:
            validation_results['issues_found'].append(
                f"High missing data: {missing_pct:.2%} (threshold: {self.quality_thresholds['missing_data_pct']:.2%})"
            )
        
        # Check for price outliers
        returns = df['Close'].pct_change().dropna()
        outliers = np.abs(returns) > self.quality_thresholds['max_single_day_return']
        if outliers.sum() > 0:
            validation_results['issues_found'].append(
                f"Found {outliers.sum()} extreme return outliers (>{self.quality_thresholds['max_single_day_return']:.0%})"
            )
        
        # Check for volume anomalies
        if 'Volume' in df.columns:
            zero_volume_days = (df['Volume'] == 0).sum()
            if zero_volume_days > len(df) * 0.01:  # More than 1% zero volume days
                validation_results['issues_found'].append(
                    f"High zero-volume days: {zero_volume_days} ({zero_volume_days/len(df):.2%})"
                )
        
        # Check for stock splits (large overnight returns)
        overnight_returns = df['Open'] / df['Close'].shift(1) - 1
        potential_splits = np.abs(overnight_returns) > 0.25
        if potential_splits.sum() > 0:
            validation_results['issues_found'].append(
                f"Potential unaccounted stock splits: {potential_splits.sum()} events"
            )
        
        # Check data continuity
        date_gaps = pd.date_range(df.index.min(), df.index.max(), freq='D')
        missing_dates = set(date_gaps) - set(df.index)
        weekday_gaps = [d for d in missing_dates if d.weekday() < 5]  # Exclude weekends
        
        if len(weekday_gaps) > len(df) * 0.05:  # More than 5% missing weekdays
            validation_results['issues_found'].append(
                f"High missing trading days: {len(weekday_gaps)} weekdays missing"
            )
        
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        validation_results['approved'] = validation_results['quality_score'] >= 0.85
        
        return validation_results
    
    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """Calculate overall data quality score (0-1)"""
        base_score = 1.0
        
        # Deduct points for each issue
        for issue in validation_results['issues_found']:
            if 'missing data' in issue.lower():
                base_score -= 0.2
            elif 'outlier' in issue.lower():
                base_score -= 0.1
            elif 'split' in issue.lower():
                base_score -= 0.15
            elif 'volume' in issue.lower():
                base_score -= 0.05
            else:
                base_score -= 0.1
        
        return max(0.0, base_score)
```

### 2.2 Survivorship Bias Correction

```python
class SurvivorshipBiasCorrector:
    def __init__(self):
        self.delisted_stocks_url = 'https://www.nasdaq.com/api/v1/historical/delisted-stocks'
        
    async def get_delisted_stocks(self, start_date: str, end_date: str) -> List[Dict]:
        """Get list of delisted stocks in time period"""
        
        try:
            # Get delisted stocks from NASDAQ API
            params = {
                'from': start_date,
                'to': end_date,
                'limit': 5000
            }
            
            response = await self.session.get(self.delisted_stocks_url, params=params)
            delisted_data = response.json()
            
            delisted_stocks = []
            for stock in delisted_data.get('data', []):
                delisted_stocks.append({
                    'symbol': stock['symbol'],
                    'company_name': stock['companyName'],
                    'delisting_date': pd.to_datetime(stock['delistingDate']),
                    'reason': stock.get('reason', 'Unknown')
                })
            
            return delisted_stocks
            
        except Exception as e:
            logging.error(f"Failed to fetch delisted stocks: {e}")
            return []
    
    def include_delisted_in_universe(self, stock_universe: List[str],
                                   delisted_stocks: List[Dict],
                                   backtest_start: str) -> List[str]:
        """Include delisted stocks that were active during backtest period"""
        
        backtest_start_date = pd.to_datetime(backtest_start)
        
        # Add delisted stocks that were active during our backtest period
        for stock in delisted_stocks:
            if (stock['delisting_date'] > backtest_start_date and 
                stock['symbol'] not in stock_universe):
                stock_universe.append(stock['symbol'])
        
        logging.info(f"Added {len([s for s in delisted_stocks if s['symbol'] in stock_universe])} delisted stocks to universe")
        
        return stock_universe
```

---

## SECTION 3: DATA INTEGRATION PIPELINE

### 3.1 Unified Data Collection Framework

```python
class UnifiedDataCollector:
    def __init__(self):
        self.collectors = {
            'alpha_vantage': AlphaVantageDataCollector(os.getenv('ALPHA_VANTAGE_API_KEY')),
            'yahoo_finance': YahooFinanceDataCollector(),
            'fred': FREDDataCollector(),
            'reuters': ReutersNewsCollector(os.getenv('REUTERS_API_KEY')),
            'sec_edgar': SECEdgarCollector()
        }
        
        self.quality_validator = DataQualityValidator()
        self.survivorship_corrector = SurvivorshipBiasCorrector()
        
    async def collect_comprehensive_dataset(self, symbols: List[str],
                                          start_date: str,
                                          end_date: str) -> Dict:
        """Collect comprehensive dataset for validation"""
        
        dataset = {
            'price_data': {},
            'volume_data': {},
            'economic_data': None,
            'news_data': {},
            'fundamentals_data': {},
            'options_data': {},
            'quality_reports': {},
            'metadata': {
                'collection_date': datetime.now(),
                'symbols': symbols,
                'date_range': (start_date, end_date),
                'data_sources': list(self.collectors.keys())
            }
        }
        
        # Correct for survivorship bias
        delisted_stocks = await self.survivorship_corrector.get_delisted_stocks(start_date, end_date)
        symbols = self.survivorship_corrector.include_delisted_in_universe(symbols, delisted_stocks, start_date)
        
        # Collect economic data (once for all symbols)
        try:
            dataset['economic_data'] = await self.collectors['fred'].get_economic_indicators(start_date, end_date)
            logging.info("Economic data collection completed")
        except Exception as e:
            logging.error(f"Economic data collection failed: {e}")
        
        # Collect data for each symbol
        for i, symbol in enumerate(symbols):
            logging.info(f"Collecting data for {symbol} ({i+1}/{len(symbols)})")
            
            try:
                # Primary source: Alpha Vantage for daily data
                try:
                    price_data = await self.collectors['alpha_vantage'].get_historical_data(
                        symbol, 'TIME_SERIES_DAILY_ADJUSTED'
                    )
                    dataset['price_data'][symbol] = price_data
                    
                except Exception as e:
                    logging.warning(f"Alpha Vantage failed for {symbol}, trying Yahoo Finance: {e}")
                    
                    # Fallback: Yahoo Finance
                    price_data = await self.collectors['yahoo_finance'].get_historical_data(symbol)
                    dataset['price_data'][symbol] = price_data
                
                # Validate data quality
                quality_report = self.quality_validator.validate_price_data(price_data, symbol)
                dataset['quality_reports'][symbol] = quality_report
                
                if not quality_report['approved']:
                    logging.warning(f"Data quality issues for {symbol}: {quality_report['issues_found']}")
                
                # Collect additional data
                if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']:  # High-priority symbols
                    try:
                        # Options data
                        options_data = await self.collectors['yahoo_finance'].get_options_data(symbol)
                        if options_data:
                            dataset['options_data'][symbol] = options_data
                        
                        # News data
                        news_data = await self.collectors['reuters'].get_historical_news(symbol, start_date, end_date)
                        if news_data:
                            dataset['news_data'][symbol] = news_data
                            
                    except Exception as e:
                        logging.warning(f"Additional data collection failed for {symbol}: {e}")
                
            except Exception as e:
                logging.error(f"Complete data collection failed for {symbol}: {e}")
                continue
            
            # Rate limiting
            await asyncio.sleep(0.2)  # 200ms delay between symbols
        
        # Generate collection summary
        dataset['collection_summary'] = self._generate_collection_summary(dataset)
        
        return dataset
    
    def _generate_collection_summary(self, dataset: Dict) -> Dict:
        """Generate summary of data collection results"""
        
        return {
            'total_symbols_requested': len(dataset['metadata']['symbols']),
            'symbols_with_price_data': len(dataset['price_data']),
            'symbols_with_news_data': len(dataset['news_data']),
            'symbols_with_options_data': len(dataset['options_data']),
            'economic_indicators_available': len(dataset['economic_data'].columns) if dataset['economic_data'] is not None else 0,
            'data_quality_summary': {
                'high_quality': len([r for r in dataset['quality_reports'].values() if r['quality_score'] >= 0.9]),
                'medium_quality': len([r for r in dataset['quality_reports'].values() if 0.7 <= r['quality_score'] < 0.9]),
                'low_quality': len([r for r in dataset['quality_reports'].values() if r['quality_score'] < 0.7])
            },
            'date_coverage': {
                'start_date': min([df.index.min() for df in dataset['price_data'].values()]),
                'end_date': max([df.index.max() for df in dataset['price_data'].values()]),
                'total_days': None  # Calculate based on date range
            }
        }
```

### 3.2 Data Storage & Caching

```python
class DataStorageManager:
    def __init__(self, storage_path: str = '/app/data/validation'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_dataset(self, dataset: Dict, dataset_name: str) -> str:
        """Save dataset with compression and metadata"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"{dataset_name}_{timestamp}.pkl.gz"
        dataset_path = self.storage_path / dataset_filename
        
        # Compress and save
        with gzip.open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata separately for quick access
        metadata_path = self.storage_path / f"{dataset_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset['metadata'], f, indent=2, default=str)
        
        logging.info(f"Dataset saved to {dataset_path}")
        return str(dataset_path)
    
    def load_dataset(self, dataset_path: str) -> Dict:
        """Load compressed dataset"""
        
        with gzip.open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        logging.info(f"Dataset loaded from {dataset_path}")
        return dataset
    
    def list_available_datasets(self) -> List[Dict]:
        """List all available datasets with metadata"""
        
        datasets = []
        
        for metadata_file in self.storage_path.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                dataset_file = metadata_file.name.replace('_metadata.json', '.pkl.gz')
                dataset_path = self.storage_path / dataset_file
                
                if dataset_path.exists():
                    datasets.append({
                        'dataset_path': str(dataset_path),
                        'metadata_path': str(metadata_file),
                        'metadata': metadata,
                        'file_size_mb': dataset_path.stat().st_size / (1024 * 1024),
                        'created_date': datetime.fromtimestamp(dataset_path.stat().st_mtime)
                    })
                    
            except Exception as e:
                logging.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        return sorted(datasets, key=lambda x: x['created_date'], reverse=True)
```

---

## SECTION 4: IMPLEMENTATION CHECKLIST

### 4.1 Phase 1: Data Source Setup (Week 1)

- [ ] **Alpha Vantage Integration**
  - [ ] Obtain API key (free tier for testing)
  - [ ] Implement rate limiting (5 calls/minute)
  - [ ] Test historical data collection for 10 symbols
  - [ ] Validate data quality and completeness

- [ ] **Yahoo Finance Backup**
  - [ ] Implement yfinance integration
  - [ ] Test extended historical data (10+ years)
  - [ ] Validate split/dividend adjustments
  - [ ] Test options data collection

- [ ] **FRED Economic Data**
  - [ ] Implement pandas_datareader integration
  - [ ] Collect key economic indicators
  - [ ] Validate data alignment with market dates

### 4.2 Phase 2: Quality & Validation (Week 2)

- [ ] **Data Quality Framework**
  - [ ] Implement quality validation checks
  - [ ] Test outlier detection algorithms
  - [ ] Validate survivorship bias correction
  - [ ] Test data completeness metrics

- [ ] **Integration Testing**
  - [ ] Test full data collection pipeline
  - [ ] Validate data storage and compression
  - [ ] Test data retrieval and caching
  - [ ] Performance optimization

### 4.3 Success Criteria

- ✅ **Data Coverage**: 95%+ successful data collection for target symbols
- ✅ **Data Quality**: 90%+ of symbols pass quality validation
- ✅ **Historical Depth**: Minimum 5 years of daily data
- ✅ **Speed**: Complete dataset collection within 2 hours
- ✅ **Reliability**: Successful failover to backup sources

---

## SECTION 5: COST & RESOURCE REQUIREMENTS

### 5.1 API Costs (Monthly)

| Data Source | Free Tier | Premium Tier | Usage Estimate |
|-------------|-----------|--------------|----------------|
| Alpha Vantage | 5 calls/min | $49.99/month | $50/month |
| Yahoo Finance | Free | Free | $0/month |
| FRED | Free | Free | $0/month |
| Reuters News | N/A | $500/month | $500/month |
| SEC EDGAR | Free | Free | $0/month |

**Total Monthly Cost**: ~$550 for comprehensive data

### 5.2 Storage Requirements

- **Raw Data**: 5GB per 100 symbols (5 years daily data)
- **Compressed Storage**: 1GB per 100 symbols  
- **Total Storage**: 20-50GB for comprehensive validation dataset

### 5.3 Processing Requirements

- **CPU**: 4+ cores for parallel data collection
- **Memory**: 8GB+ RAM for large dataset processing
- **Network**: Stable internet for API calls
- **Time**: 2-4 hours for complete dataset collection

---

## CONCLUSION

This real market data specification ensures that the validation framework operates on authentic market data, eliminating the synthetic data bias that plagued previous validation attempts. The comprehensive data collection pipeline provides the foundation for statistically rigorous backtesting that will accurately reflect real-world trading performance.

**Next Steps**: Implement Phase 1 data source integrations and begin collecting the comprehensive validation dataset.
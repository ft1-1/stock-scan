# EODHD API User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [Core APIs](#core-apis)
5. [Implementation Best Practices](#implementation-best-practices)
6. [Production Considerations](#production-considerations)
7. [Common Use Cases](#common-use-cases)
8. [Troubleshooting](#troubleshooting)

## Introduction

EODHD (End of Day Historical Data) is a comprehensive financial data provider offering extensive market data through their APIs. This guide provides practical implementation insights based on production usage in the PMCC Scanner application.

### Key Strengths
- **Excellent Stock Screening**: Native screener API with powerful filtering capabilities
- **Comprehensive Coverage**: 6000+ US stocks with fundamental and historical data
- **Official Python Library**: Well-maintained `eodhd` package for easy integration
- **Reliable EOD Data**: Consistent end-of-day pricing and volume data
- **Cost-Effective**: Competitive pricing for the data coverage provided

### API Capabilities
- Stock screening with market cap, sector, and financial filters
- Historical price data (EOD and intraday)
- Options data with full Greeks calculations
- Fundamental company data and financials
- Calendar events (earnings, dividends)
- Technical indicators
- Live price data (with appropriate subscription)

## Getting Started

### Installation

#### Using the Official Python Library (Recommended)
```bash
pip install eodhd
```

#### Using Raw API (Alternative)
```bash
pip install aiohttp  # For async requests
pip install requests  # For sync requests
```

### Quick Start Example
```python
from eodhd import APIClient

# Initialize client
api = APIClient("your_api_key")

# Get stock quote
quote = api.get_eod_historical_stock_market_data("AAPL.US", "d", None, None, 1)
print(f"AAPL last price: ${quote[-1]['adjusted_close']}")

# Screen stocks by market cap
# Note: Use raw API for screening as library support varies
```

## Authentication

### API Token Management
```python
import os
from eodhd import APIClient

# Best practice: Store token in environment variable
api_token = os.getenv('EODHD_API_TOKEN')
if not api_token:
    raise ValueError("EODHD_API_TOKEN environment variable not set")

# Initialize client
api = APIClient(api_token)
```

### Environment Configuration
```bash
# .env file
EODHD_API_TOKEN=your_api_token_here
EODHD_API_BASE_URL=https://eodhd.com/api  # Optional custom URL
```

## Core APIs

### 1. Stock Screening API

The screener is EODHD's standout feature for filtering stocks efficiently.

#### Basic Screening Request
```python
import aiohttp
import json

async def screen_stocks(api_token, min_market_cap=50000000, max_market_cap=5000000000):
    """Screen US stocks by market cap"""
    url = "https://eodhd.com/api/screener"
    
    filters = [
        ["market_capitalization", ">=", min_market_cap],
        ["market_capitalization", "<=", max_market_cap],
        ["exchange", "=", "us"]
    ]
    
    params = {
        'api_token': api_token,
        'filters': json.dumps(filters),
        'sort': 'market_capitalization.desc',
        'limit': 100
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            return await response.json()
```

#### Advanced Screening with Multiple Criteria
```python
def build_screening_filters(criteria):
    """Build EODHD filter array from criteria"""
    filters = []
    
    # Market cap filter
    if criteria.get('min_market_cap'):
        filters.append(["market_capitalization", ">=", criteria['min_market_cap']])
    if criteria.get('max_market_cap'):
        filters.append(["market_capitalization", "<=", criteria['max_market_cap']])
    
    # Price filter
    if criteria.get('min_price'):
        filters.append(["adjusted_close", ">=", criteria['min_price']])
    if criteria.get('max_price'):
        filters.append(["adjusted_close", "<=", criteria['max_price']])
    
    # Volume filter
    if criteria.get('min_volume'):
        filters.append(["avgvol_200d", ">=", criteria['min_volume']])
    
    # Exchange filter
    if criteria.get('exchanges'):
        for exchange in criteria['exchanges']:
            filters.append(["exchange", "=", exchange])
    
    return filters
```

#### Handling EODHD's 1000-Result Limit

EODHD limits each query to 1000 results. For comprehensive screening, split queries by market cap ranges:

```python
async def screen_stocks_comprehensive(api_token, min_cap=50000000, max_cap=5000000000):
    """Screen stocks with automatic range splitting to bypass 1000-result limit"""
    
    def get_range_size(current_cap):
        """Dynamic range sizing based on market cap"""
        if current_cap < 100_000_000:  # Under 100M
            return 50_000_000  # 50M ranges
        elif current_cap < 500_000_000:  # 100M-500M
            return 250_000_000  # 250M ranges
        elif current_cap < 1_000_000_000:  # 500M-1B
            return 500_000_000  # 500M ranges
        elif current_cap < 5_000_000_000:  # 1B-5B
            return 1_000_000_000  # 1B ranges
        else:
            return 2_500_000_000  # 2.5B+ ranges
    
    # Generate market cap ranges
    ranges = []
    current = min_cap
    while current < max_cap:
        range_size = get_range_size(current)
        range_end = min(current + range_size, max_cap)
        ranges.append((current, range_end))
        current = range_end
    
    all_results = []
    for range_min, range_max in ranges:
        # Screen each range separately
        results = await screen_range(api_token, range_min, range_max)
        all_results.extend(results)
    
    return all_results
```

### 2. Historical Price Data

#### Using the Official Library
```python
from eodhd import APIClient
import pandas as pd

api = APIClient(api_token)

# Get daily historical data
historical = api.get_eod_historical_stock_market_data(
    symbol="AAPL.US",
    period="d",  # daily
    from_date="2023-01-01",
    to_date="2023-12-31",
    order="d"  # descending
)

# Convert to DataFrame
df = pd.DataFrame(historical)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Calculate technical indicators
df['SMA_20'] = df['adjusted_close'].rolling(window=20).mean()
df['SMA_50'] = df['adjusted_close'].rolling(window=50).mean()
```

#### Getting Latest Quote with Change Calculation
```python
async def get_enhanced_quote(api, symbol):
    """Get quote with calculated change values"""
    # Get last 2 days of data
    from datetime import datetime, timedelta
    today = datetime.now()
    two_days_ago = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    
    data = api.get_eod_historical_stock_market_data(
        f"{symbol}.US", "d", two_days_ago, None, "d"
    )
    
    if len(data) >= 2:
        current = data[-1]
        previous = data[-2]
        
        change = current['adjusted_close'] - previous['adjusted_close']
        change_percent = (change / previous['adjusted_close']) * 100
        
        return {
            'symbol': symbol,
            'last': current['adjusted_close'],
            'volume': current['volume'],
            'change': change,
            'change_percent': change_percent,
            'date': current['date']
        }
```

### 3. Options Data API

**CRITICAL**: EODHD options data includes historical transactions from multiple trading days. Always filter by tradetime or sort by date to ensure you're getting current prices, not stale data from previous days.

#### Tradetime Filtering Parameters
- `filter[tradetime_eq]`: Exact trade date (e.g., "2025-03-19")
- `filter[tradetime_from]`: Minimum trade date (e.g., "2025-01-01")
- `filter[tradetime_to]`: Maximum trade date (e.g., "2025-06-01")

#### Fetching Current Options Chains
```python
async def get_options_chain(symbol, api_token):
    """Get complete options chain with CURRENT prices only"""
    from datetime import datetime, timedelta
    
    # Get the most recent trading day to ensure current data
    today = datetime.now()
    if today.weekday() >= 5:  # Weekend
        days_back = today.weekday() - 4
        last_trading_day = today - timedelta(days=days_back)
    else:
        last_trading_day = today
    
    url = "https://eodhd.com/api/mp/unicornbay/options/eod"
    
    params = {
        'api_token': api_token,
        'filter[underlying_symbol]': symbol,
        'filter[type]': 'call',  # or 'put'
        'filter[tradetime_eq]': last_trading_day.strftime('%Y-%m-%d'),  # CRITICAL: Current data only
        'page[limit]': 1000,
        'sort': 'exp_date'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get('data', [])
```

#### PMCC-Specific Options Filtering
```python
async def get_pmcc_options(symbol, current_price, api_token):
    """Get options suitable for PMCC strategy with CURRENT prices"""
    from datetime import datetime, timedelta
    
    today = datetime.now()
    
    # Calculate the most recent trading day for current prices
    if today.weekday() >= 5:  # Weekend
        days_back = today.weekday() - 4
        last_trading_day = today - timedelta(days=days_back)
    else:
        last_trading_day = today
    
    # For better performance, filter to recent trading days (last 5 days)
    tradetime_from = (last_trading_day - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # LEAPS: 6-12 months out, deep ITM
    leaps_params = {
        'api_token': api_token,
        'filter[underlying_symbol]': symbol,
        'filter[type]': 'call',
        'filter[exp_date_from]': (today + timedelta(days=180)).strftime('%Y-%m-%d'),
        'filter[exp_date_to]': (today + timedelta(days=365)).strftime('%Y-%m-%d'),
        'filter[strike_from]': current_price * 0.60,  # Deep ITM
        'filter[strike_to]': current_price * 0.85,
        'filter[tradetime_from]': tradetime_from,  # IMPORTANT: Recent data only
        'page[limit]': 500,
        'sort': '-tradetime'  # Sort by trade time descending for most recent first
    }
    
    # Short calls: 30-45 DTE, OTM
    short_params = {
        'api_token': api_token,
        'filter[underlying_symbol]': symbol,
        'filter[type]': 'call',
        'filter[exp_date_from]': (today + timedelta(days=30)).strftime('%Y-%m-%d'),
        'filter[exp_date_to]': (today + timedelta(days=45)).strftime('%Y-%m-%d'),
        'filter[strike_from]': current_price * 1.05,  # OTM
        'filter[strike_to]': current_price * 1.20,
        'filter[tradetime_from]': tradetime_from,  # IMPORTANT: Recent data only
        'page[limit]': 500,
        'sort': '-tradetime'  # Sort by trade time descending for most recent first
    }
    
    # Fetch both in parallel
    leaps_task = fetch_options(leaps_params)
    short_task = fetch_options(short_params)
    
    leaps_options, short_options = await asyncio.gather(leaps_task, short_task)
    
    return {
        'leaps': filter_by_greeks(leaps_options, min_delta=0.70),
        'short_calls': filter_by_greeks(short_options, min_delta=0.15, max_delta=0.40)
    }
```

### 4. Fundamental Data

```python
# Using official library
api = APIClient(api_token)

# Get comprehensive fundamental data
fundamentals = api.get_fundamentals_data('AAPL')

# Extract key metrics
general = fundamentals.get('General', {})
financials = fundamentals.get('Financials', {})
valuation = fundamentals.get('Valuation', {})

company_info = {
    'name': general.get('Name'),
    'sector': general.get('Sector'),
    'industry': general.get('Industry'),
    'market_cap': valuation.get('MarketCapitalization'),
    'pe_ratio': valuation.get('TrailingPE'),
    'dividend_yield': fundamentals.get('SplitsDividends', {}).get('ForwardDividendYield'),
    'eps': financials.get('EPS', {}).get('TTM')
}
```

## Implementation Best Practices

### 1. Error Handling and Retry Logic

```python
import asyncio
from typing import Optional, Dict, Any

class EODHDAPIError(Exception):
    """Custom exception for EODHD API errors"""
    def __init__(self, message: str, code: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after

async def make_eodhd_request(url: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Make EODHD API request with retry logic"""
    retry_delays = [1, 2, 5]  # Exponential backoff
    
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        return data
                    elif response.status == 401:
                        raise EODHDAPIError("Authentication failed", code=401)
                    elif response.status == 402:
                        raise EODHDAPIError("Plan limit exceeded", code=402)
                    elif response.status == 429:
                        retry_after = response.headers.get('Retry-After', 60)
                        raise EODHDAPIError("Rate limit exceeded", code=429, retry_after=float(retry_after))
                    else:
                        error_msg = data.get('error', 'Unknown error')
                        raise EODHDAPIError(f"API error: {error_msg}", code=response.status)
                        
        except asyncio.TimeoutError:
            if attempt < max_retries:
                await asyncio.sleep(retry_delays[attempt])
                continue
            raise EODHDAPIError("Request timeout")
        except aiohttp.ClientError as e:
            if attempt < max_retries:
                await asyncio.sleep(retry_delays[attempt])
                continue
            raise EODHDAPIError(f"Network error: {str(e)}")
```

### 2. Rate Limiting and Credit Management

```python
class EODHDRateLimiter:
    """Rate limiter for EODHD API requests"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self._lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                # Wait until oldest request is 1 minute old
                sleep_time = 60 - (now - self.request_times[0]) + 0.1
                await asyncio.sleep(sleep_time)
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]
            
            self.request_times.append(now)

# Usage
rate_limiter = EODHDRateLimiter(requests_per_minute=60)

async def fetch_with_rate_limit(url, params):
    await rate_limiter.acquire()
    return await make_eodhd_request(url, params)
```

### 3. Caching Strategy

```python
import hashlib
from datetime import datetime, timedelta

class EODHDCache:
    """Simple cache for EODHD API responses"""
    
    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl_hours = ttl_hours
    
    def _generate_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and params"""
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached data if valid"""
        key = self._generate_key(endpoint, params)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(hours=self.ttl_hours):
                return entry['data']
            else:
                del self.cache[key]
        
        return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Any):
        """Cache data with timestamp"""
        key = self._generate_key(endpoint, params)
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

# Different TTLs for different data types
screening_cache = EODHDCache(ttl_hours=24)  # Daily refresh
fundamentals_cache = EODHDCache(ttl_hours=168)  # Weekly refresh
options_cache = EODHDCache(ttl_hours=1)  # Hourly refresh
```

### 4. Connection Pooling

```python
class EODHDConnectionPool:
    """Reusable connection pool for EODHD API"""
    
    def __init__(self, pool_size: int = 10):
        self.connector = aiohttp.TCPConnector(
            limit=pool_size,
            limit_per_host=pool_size,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'PMCC-Scanner/1.0',
                'Accept': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with self.session.get(url, params=params) as response:
            return await response.json()
```

## Production Considerations

### 1. API Credit Management

```python
class EODHDCreditTracker:
    """Track API credit usage"""
    
    CREDIT_COSTS = {
        'screener': 5,
        'options': 1,
        'eod': 1,
        'fundamentals': 1,
        'live': 1
    }
    
    def __init__(self, daily_limit: int = 100000):
        self.daily_limit = daily_limit
        self.usage = {}
        self.reset_time = None
    
    def track_usage(self, endpoint_type: str, count: int = 1):
        """Track credit usage"""
        today = datetime.now().date()
        
        if today not in self.usage:
            self.usage = {today: 0}  # Reset for new day
        
        credits = self.CREDIT_COSTS.get(endpoint_type, 1) * count
        self.usage[today] += credits
        
        if self.usage[today] > self.daily_limit * 0.8:
            logger.warning(f"API credit usage at 80%: {self.usage[today]}/{self.daily_limit}")
```

### 2. Multi-Provider Architecture

```python
from abc import ABC, abstractmethod

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def screen_stocks(self, criteria: Dict[str, Any]) -> List[str]:
        pass

class EODHDProvider(DataProvider):
    """EODHD implementation of DataProvider"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.client = APIClient(api_token)
    
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        # Implementation here
        pass
    
    async def screen_stocks(self, criteria: Dict[str, Any]) -> List[str]:
        # Implementation here
        pass

# Provider factory pattern
class ProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> DataProvider:
        if provider_type == "eodhd":
            return EODHDProvider(config['api_token'])
        else:
            raise ValueError(f"Unknown provider: {provider_type}")
```

### 3. Monitoring and Alerting

```python
import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class EODHDMetrics:
    """Metrics for EODHD API usage"""
    total_requests: int = 0
    failed_requests: int = 0
    rate_limit_hits: int = 0
    average_latency_ms: float = 0.0
    credits_used: int = 0

class EODHDMonitor:
    """Monitor EODHD API health and usage"""
    
    def __init__(self):
        self.metrics = EODHDMetrics()
        self.logger = logging.getLogger(__name__)
    
    async def record_request(self, endpoint: str, latency_ms: float, 
                           success: bool, credits: int = 1):
        """Record API request metrics"""
        self.metrics.total_requests += 1
        self.metrics.credits_used += credits
        
        if not success:
            self.metrics.failed_requests += 1
        
        # Update average latency
        n = self.metrics.total_requests
        self.metrics.average_latency_ms = (
            (self.metrics.average_latency_ms * (n - 1) + latency_ms) / n
        )
        
        # Alert on high failure rate
        if self.metrics.total_requests > 100:
            failure_rate = self.metrics.failed_requests / self.metrics.total_requests
            if failure_rate > 0.1:  # 10% failure rate
                self.logger.error(f"High EODHD API failure rate: {failure_rate:.1%}")
```

## Common Use Cases

### 1. Daily Stock Universe Update

```python
async def update_stock_universe():
    """Update daily stock universe for PMCC scanning"""
    
    # Screen stocks with PMCC-suitable criteria
    criteria = {
        'min_market_cap': 50_000_000,
        'max_market_cap': 5_000_000_000,
        'min_volume': 100_000,
        'min_price': 5.0
    }
    
    # Use comprehensive screening to get all results
    stocks = await screen_stocks_comprehensive(
        api_token=os.getenv('EODHD_API_TOKEN'),
        min_cap=criteria['min_market_cap'],
        max_cap=criteria['max_market_cap']
    )
    
    # Filter and sort results
    qualified_stocks = []
    for stock in stocks:
        if (stock.get('avgvol_200d', 0) >= criteria['min_volume'] and
            stock.get('adjusted_close', 0) >= criteria['min_price']):
            qualified_stocks.append({
                'symbol': stock['code'],
                'market_cap': stock['market_capitalization'],
                'price': stock['adjusted_close'],
                'volume': stock['avgvol_200d']
            })
    
    # Sort by market cap descending
    qualified_stocks.sort(key=lambda x: x['market_cap'], reverse=True)
    
    return qualified_stocks
```

### 2. Options Chain Analysis

```python
async def analyze_pmcc_opportunity(symbol: str, current_price: float):
    """Analyze PMCC opportunity for a stock"""
    
    # Get options data
    options_data = await get_pmcc_options(symbol, current_price, api_token)
    
    pmcc_opportunities = []
    
    for leaps in options_data['leaps']:
        for short_call in options_data['short_calls']:
            # PMCC validation
            if short_call['strike'] > leaps['strike'] + leaps['ask']:
                opportunity = {
                    'symbol': symbol,
                    'leaps': {
                        'strike': leaps['strike'],
                        'expiration': leaps['exp_date'],
                        'delta': leaps['delta'],
                        'cost': leaps['ask']
                    },
                    'short_call': {
                        'strike': short_call['strike'],
                        'expiration': short_call['exp_date'],
                        'delta': short_call['delta'],
                        'premium': short_call['bid']
                    },
                    'net_debit': leaps['ask'] - short_call['bid'],
                    'max_profit': short_call['strike'] - leaps['strike'] - (leaps['ask'] - short_call['bid'])
                }
                
                pmcc_opportunities.append(opportunity)
    
    return pmcc_opportunities
```

### 3. Historical Analysis

```python
async def analyze_stock_volatility(symbol: str, lookback_days: int = 30):
    """Analyze stock volatility for options strategies"""
    
    api = APIClient(api_token)
    
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    data = api.get_eod_historical_stock_market_data(
        f"{symbol}.US",
        "d",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        "a"
    )
    
    df = pd.DataFrame(data)
    
    # Calculate metrics
    df['returns'] = df['adjusted_close'].pct_change()
    
    metrics = {
        'symbol': symbol,
        'volatility_30d': df['returns'].std() * (252 ** 0.5),  # Annualized
        'average_volume': df['volume'].mean(),
        'price_range': (df['high'].max() - df['low'].min()) / df['adjusted_close'].mean(),
        'trend': 'up' if df['adjusted_close'].iloc[-1] > df['adjusted_close'].iloc[0] else 'down'
    }
    
    return metrics
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Stale Options Data (Critical Issue)
```python
# PROBLEM: Options data includes historical transactions
# Without tradetime filtering, you may get prices from days or weeks ago!

# BAD - Returns mixed historical data
params = {
    'filter[underlying_symbol]': 'AAPL',
    'filter[type]': 'call'
}

# GOOD - Returns only current data
params = {
    'filter[underlying_symbol]': 'AAPL',
    'filter[type]': 'call',
    'filter[tradetime_eq]': '2025-03-19'  # Specific trading day
}

# BETTER - Returns recent data with fallback
def get_tradetime_filter(lookback_days=5):
    """Get tradetime filter for recent trading days"""
    from datetime import datetime, timedelta
    
    today = datetime.now()
    # Skip to last Friday if weekend
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    
    # Look back N trading days for better data coverage
    tradetime_from = today - timedelta(days=lookback_days)
    return tradetime_from.strftime('%Y-%m-%d')

params = {
    'filter[underlying_symbol]': 'AAPL',
    'filter[type]': 'call',
    'filter[tradetime_from]': get_tradetime_filter(),
    'sort': '-tradetime'  # Most recent first
}
```

#### 2. Authentication Errors (401)
```python
# Check API token is valid
if response.status == 401:
    # Verify token format
    if not api_token or api_token == 'YOUR_API_TOKEN':
        raise ValueError("Invalid API token - please set EODHD_API_TOKEN")
    
    # Token might be expired or invalid
    logger.error("EODHD authentication failed - check API token")
```

#### 3. Rate Limiting (429)
```python
# Handle rate limits gracefully
if response.status == 429:
    retry_after = int(response.headers.get('Retry-After', 60))
    logger.warning(f"Rate limited - waiting {retry_after} seconds")
    await asyncio.sleep(retry_after)
    # Retry request
```

#### 4. Empty or Incomplete Data
```python
def validate_eodhd_response(data: Any, expected_fields: List[str]) -> bool:
    """Validate EODHD response has expected data"""
    if not data:
        return False
    
    if isinstance(data, dict):
        return all(field in data for field in expected_fields)
    elif isinstance(data, list):
        return len(data) > 0 and all(
            all(field in item for field in expected_fields)
            for item in data
        )
    
    return False
```

#### 5. Timezone Handling
```python
from zoneinfo import ZoneInfo

def convert_to_market_timezone(date_str: str) -> datetime:
    """Convert EODHD date to market timezone"""
    # EODHD returns dates in US Eastern timezone
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.replace(tzinfo=ZoneInfo('America/New_York'))
```

### Debug Logging

```python
import logging

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log all API requests
logger = logging.getLogger('eodhd_api')

async def debug_request(url: str, params: Dict[str, Any]):
    """Make request with debug logging"""
    logger.debug(f"Request URL: {url}")
    logger.debug(f"Request params: {params}")
    
    start_time = time.time()
    response = await make_eodhd_request(url, params)
    elapsed = (time.time() - start_time) * 1000
    
    logger.debug(f"Response time: {elapsed:.1f}ms")
    logger.debug(f"Response size: {len(json.dumps(response))} bytes")
    
    return response
```

## Summary

EODHD provides a robust and comprehensive financial data API that's particularly strong for:
- Stock screening with complex filters
- Historical price data and fundamentals
- Options data with Greeks calculations
- Cost-effective access to market data

Key implementation tips:
1. Use the official Python library when possible
2. Implement proper error handling and retry logic
3. Cache responses appropriately based on data type
4. Split large queries to handle API limits
5. Monitor usage to stay within credit limits
6. Use connection pooling for better performance

This guide provides battle-tested patterns from production usage. Adapt these examples to your specific needs while maintaining the core principles of reliability, efficiency, and proper error handling.
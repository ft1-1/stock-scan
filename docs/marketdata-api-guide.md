# MarketData.app API User Guide for Options Trading Applications

This comprehensive guide is based on real-world learnings from implementing the MarketData.app API in a production options trading scanner (PMCC Scanner). It covers best practices, optimization techniques, and important gotchas discovered during development.

## Table of Contents
1. [Introduction](#introduction)
2. [Authentication & Setup](#authentication--setup)
3. [API Credit System](#api-credit-system)
4. [Core Concepts](#core-concepts)
5. [Options Chain API](#options-chain-api)
6. [Stock Quotes API](#stock-quotes-api)
7. [Rate Limiting & Error Handling](#rate-limiting--error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Data Structures](#data-structures)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Example Implementations](#example-implementations)

## Introduction

MarketData.app is a modern financial data API that provides real-time and cached market data. While powerful, it has unique characteristics that differ from other financial APIs:

- **No native stock screener** - You need to implement screening by fetching quotes for a predefined universe
- **Credit-based pricing** - Different endpoints consume different amounts of credits
- **Cached vs Live feeds** - Critical choice that impacts both cost and freshness
- **Array-based responses** - Data is returned in parallel arrays, not objects

## Authentication & Setup

### API Token Configuration

```python
# Bearer token authentication
headers = {
    'Authorization': f'Bearer {api_token}',
    'Accept': 'application/json',
    'User-Agent': 'YourApp/1.0'
}
```

### Base URL Structure

```python
base_url = 'https://api.marketdata.app/v1'
# Always ensure endpoints end with '/'
endpoint_url = f"{base_url}/options/chain/AAPL/"
```

**Important**: MarketData.app requires trailing slashes on all endpoints. Missing slashes will result in 404 errors.

## API Credit System

Understanding the credit system is crucial for cost management:

### Credit Costs by Endpoint

| Endpoint | Feed Type | Credit Cost |
|----------|-----------|-------------|
| Stock Quote | Any | 1 credit per symbol |
| Options Chain | Cached | 1 credit per request |
| Options Chain | Live | 1 credit per contract |
| Option Expirations | Any | 1 credit |

### Cached vs Live Feed

The most critical decision when using options endpoints:

```python
# Cached feed - 1 credit total regardless of contracts returned
params = {'feed': 'cached'}  # Returns data up to 15 minutes old

# Live feed - 1 credit PER CONTRACT (can be thousands!)
params = {'feed': 'live'}  # Real-time data
```

**Real-world example**: SPX options chain can have 5,000+ contracts. Using live feed = 5,000 credits vs cached feed = 1 credit.

## Core Concepts

### 1. No Native Stock Screener

Unlike EODHD or other providers, MarketData.app doesn't offer a stock screening endpoint. You must:

```python
# Define your stock universe
stock_universe = ['AAPL', 'MSFT', 'GOOGL', ...]  # Pre-defined list

# Fetch quotes for all symbols
quotes = await get_stock_quotes(stock_universe)

# Apply screening criteria locally
filtered_stocks = [q for q in quotes if q.volume > 1000000]
```

### 2. Array-Based Response Format

MarketData.app returns data as parallel arrays:

```json
{
  "optionSymbol": ["AAPL240119C150000", "AAPL240119C155000"],
  "bid": [25.50, 20.30],
  "ask": [25.70, 20.50],
  "delta": [0.85, 0.75]
}
```

Each index across all arrays represents one contract.

### 3. Date Handling

- Dates can be Unix timestamps or ISO strings
- Always validate date parsing as formats may vary
- Expiration dates are typically returned as Unix timestamps

## Options Chain API

### Basic Options Chain Request

```python
async def get_option_chain(symbol: str, **params):
    """Fetch options chain with optimal parameters."""
    
    default_params = {
        'feed': 'cached',        # ALWAYS use cached unless real-time critical
        'side': 'call',          # 'call', 'put', or omit for both
        'minOpenInterest': 5,    # Filter out illiquid options
    }
    
    # Merge with user params
    final_params = {**default_params, **params}
    
    response = await make_request(f'/options/chain/{symbol}/', final_params)
    return parse_option_chain(response)
```

### Optimized PMCC Strategy Calls

For strategies like PMCC that need specific option types, make targeted calls:

```python
async def get_pmcc_chains(symbol: str):
    """Fetch LEAPS and short calls in 2 efficient requests."""
    
    today = datetime.now().date()
    
    # Request 1: LEAPS (6-12 months, deep ITM)
    leaps_params = {
        'from': (today + timedelta(days=180)).isoformat(),
        'to': (today + timedelta(days=365)).isoformat(),
        'side': 'call',
        'delta': '.70-.95',      # Deep ITM
        'minOpenInterest': 10,
        'feed': 'cached'
    }
    
    # Request 2: Short calls (21-45 days, OTM)
    short_params = {
        'from': (today + timedelta(days=21)).isoformat(),
        'to': (today + timedelta(days=45)).isoformat(),
        'side': 'call',
        'delta': '.15-.40',      # OTM
        'minOpenInterest': 5,
        'feed': 'cached'
    }
    
    # Make both requests concurrently
    leaps, shorts = await asyncio.gather(
        make_request(f'/options/chain/{symbol}/', leaps_params),
        make_request(f'/options/chain/{symbol}/', short_params)
    )
    
    return {'leaps': leaps, 'short': shorts}
```

### Filtering Parameters

Key parameters for efficient data retrieval:

```python
params = {
    # Date filtering (use these instead of DTE)
    'from': '2024-06-01',        # ISO date format
    'to': '2024-12-31',
    
    # Greeks filtering
    'delta': '.70-.95',          # Range format
    'iv': '.20-.50',             # Implied volatility range
    
    # Liquidity filtering
    'minOpenInterest': 10,
    'minVolume': 5,
    
    # Strike filtering
    'strikeLimit': 10,           # Limits strikes around ATM
    'strike': '150-200',         # Specific strike range
    
    # ALWAYS specify for cost savings
    'feed': 'cached'
}
```

## Stock Quotes API

### Single Quote

```python
async def get_stock_quote(symbol: str):
    """Get quote for a single symbol."""
    response = await make_request(f'/stocks/quotes/{symbol}/')
    return parse_stock_quote(response)
```

### Batch Quotes

Since there's no bulk endpoint, use concurrent requests:

```python
async def get_stock_quotes(symbols: List[str], max_concurrent: int = 50):
    """Fetch quotes for multiple symbols concurrently."""
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_limit(symbol):
        async with semaphore:
            return await get_stock_quote(symbol)
    
    # Fetch all quotes concurrently
    tasks = [fetch_with_limit(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    quotes = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {symbol}: {result}")
        else:
            quotes[symbol] = result
    
    return quotes
```

## Rate Limiting & Error Handling

### Rate Limit Headers

MarketData.app provides rate limit info in response headers:

```python
def parse_rate_limit_headers(headers):
    return {
        'limit': int(headers.get('X-Api-Ratelimit-Limit', 0)),
        'remaining': int(headers.get('X-Api-Ratelimit-Remaining', 0)),
        'reset': int(headers.get('X-Api-Ratelimit-Reset', 0)),
        'consumed': int(headers.get('X-Api-Ratelimit-Consumed', 0))
    }
```

### Error Response Codes

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 200 | Success | Process data |
| 203 | Success (cached) | Process data (note it's cached) |
| 204 | No cached data | Need to use live feed |
| 401 | Authentication failed | Check API token |
| 402 | Plan limit exceeded | Upgrade plan or wait |
| 429 | Rate limited | Implement backoff |

### Retry Strategy

```python
async def make_request_with_retry(endpoint, params, max_retries=3):
    """Make request with exponential backoff retry."""
    
    for attempt in range(max_retries + 1):
        try:
            response = await make_request(endpoint, params)
            
            if response.status == 429:  # Rate limited
                if attempt < max_retries:
                    # Exponential backoff
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)
                    continue
            
            return response
            
        except aiohttp.ClientError as e:
            if attempt < max_retries:
                delay = 2 ** attempt
                logger.warning(f"Retry {attempt + 1} after {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                raise
    
    raise Exception("Max retries exceeded")
```

## Performance Optimization

### 1. Connection Pooling

```python
# Reuse connections with proper limits
connector = aiohttp.TCPConnector(
    limit=50,                    # Total connection pool size
    limit_per_host=50,          # Connections per host
    ttl_dns_cache=300,          # DNS cache TTL
    use_dns_cache=True
)

session = aiohttp.ClientSession(
    connector=connector,
    timeout=aiohttp.ClientTimeout(total=30)
)
```

### 2. Concurrent Request Management

```python
class RateLimitedClient:
    def __init__(self, requests_per_second=10):
        self.semaphore = asyncio.Semaphore(requests_per_second)
        self.request_times = []
    
    async def make_request(self, *args, **kwargs):
        async with self.semaphore:
            # Ensure rate limit
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 1]
            
            if len(self.request_times) >= 10:
                sleep_time = 1 - (now - self.request_times[0])
                await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.time())
            return await self._make_actual_request(*args, **kwargs)
```

### 3. Minimize API Calls

```python
# Bad: Fetch entire chain then filter
chain = await get_option_chain(symbol)
filtered = [c for c in chain if c.delta > 0.7]

# Good: Filter at API level
chain = await get_option_chain(symbol, delta='.70-1.0')
```

## Data Structures

### Parsing Option Chains

```python
def parse_option_chain(data: dict) -> List[OptionContract]:
    """Parse array-based response into option contracts."""
    
    contracts = []
    
    # Get array lengths (all should be same)
    length = len(data.get('optionSymbol', []))
    
    for i in range(length):
        contract = OptionContract(
            option_symbol=data['optionSymbol'][i],
            underlying=data['underlying'][i],
            strike=Decimal(str(data['strike'][i])),
            expiration=datetime.fromtimestamp(data['expiration'][i]),
            side='call' if data['side'][i] == 'call' else 'put',
            
            # Pricing
            bid=Decimal(str(data['bid'][i])) if data['bid'][i] else None,
            ask=Decimal(str(data['ask'][i])) if data['ask'][i] else None,
            last=Decimal(str(data['last'][i])) if data['last'][i] else None,
            
            # Greeks
            delta=Decimal(str(data['delta'][i])) if data['delta'][i] else None,
            gamma=Decimal(str(data['gamma'][i])) if data['gamma'][i] else None,
            theta=Decimal(str(data['theta'][i])) if data['theta'][i] else None,
            vega=Decimal(str(data['vega'][i])) if data['vega'][i] else None,
            
            # Market data
            volume=data['volume'][i],
            open_interest=data['openInterest'][i],
            
            # Calculated
            dte=data['dte'][i],
            intrinsic_value=calculate_intrinsic_value(...)
        )
        contracts.append(contract)
    
    return contracts
```

### Handling Missing Data

```python
def safe_get_array_value(data: dict, key: str, index: int, default=None):
    """Safely extract value from array with defaults."""
    
    arr = data.get(key, [])
    if isinstance(arr, list) and len(arr) > index:
        value = arr[index]
        # Handle null/None values
        return value if value is not None else default
    return default
```

## Best Practices

### 1. Always Use Cached Feed for Scanning

```python
# For scanning/analysis where 15-min delay is acceptable
params = {'feed': 'cached'}  # 1 credit total

# Only use live for execution/final checks
params = {'feed': 'live'}  # 1 credit per contract!
```

### 2. Filter at API Level

```python
# Reduce data transfer and parsing
params = {
    'delta': '.70-.95',          # API filters
    'minOpenInterest': 10,
    'from': start_date,
    'to': end_date
}
```

### 3. Implement Caching

```python
class CachedMarketDataClient:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    async def get_option_chain(self, symbol, **params):
        cache_key = f"{symbol}:{hash(frozenset(params.items()))}"
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return data
        
        # Fetch fresh data
        data = await self._fetch_option_chain(symbol, **params)
        self.cache[cache_key] = (data, time.time())
        
        return data
```

### 4. Handle Array Misalignment

```python
def validate_array_response(data: dict) -> bool:
    """Ensure all arrays have same length."""
    
    lengths = set()
    for key, value in data.items():
        if isinstance(value, list):
            lengths.add(len(value))
    
    if len(lengths) > 1:
        logger.error(f"Array length mismatch: {lengths}")
        return False
    
    return True
```

## Common Pitfalls

### 1. Forgetting Trailing Slashes

```python
# Wrong - will get 404
url = 'https://api.marketdata.app/v1/options/chain/AAPL'

# Correct
url = 'https://api.marketdata.app/v1/options/chain/AAPL/'
```

### 2. Using Live Feed for Scanning

```python
# DON'T DO THIS - Can consume thousands of credits
chain = await get_option_chain('SPX', feed='live')  # 5000+ credits!

# DO THIS
chain = await get_option_chain('SPX', feed='cached')  # 1 credit
```

### 3. Not Handling 204 Responses

```python
if response.status == 204:
    # No cached data available
    logger.info(f"No cached data for {symbol}, consider using live feed")
    return None  # or fetch with live feed if critical
```

### 4. Ignoring Rate Limits

```python
# Always check rate limit headers
if rate_limit['remaining'] < 100:
    logger.warning(f"Low on API credits: {rate_limit['remaining']} remaining")
```

## Example Implementations

### Complete Options Scanner

```python
class MarketDataOptionsScanner:
    """Production-ready options scanner using MarketData.app."""
    
    def __init__(self, api_token: str):
        self.client = MarketDataClient(api_token)
        self.cache = {}
        
    async def scan_pmcc_opportunities(self, symbols: List[str]):
        """Scan symbols for PMCC opportunities."""
        
        opportunities = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Concurrent processing of batch
            tasks = [self._analyze_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect valid opportunities
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {symbol}: {result}")
                elif result:
                    opportunities.append(result)
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.1)
        
        return opportunities
    
    async def _analyze_symbol(self, symbol: str):
        """Analyze single symbol for PMCC setup."""
        
        try:
            # Get current quote
            quote = await self.client.get_stock_quote(symbol)
            if not quote or not quote.last:
                return None
            
            # Fetch PMCC-optimized chains (2 requests)
            chains = await self.client.get_pmcc_option_chains(symbol)
            
            # Find best LEAPS
            leaps = self._find_best_leaps(chains['leaps'], quote.last)
            if not leaps:
                return None
            
            # Find best short call
            short_call = self._find_best_short_call(
                chains['short'], 
                quote.last, 
                leaps.strike
            )
            if not short_call:
                return None
            
            # Calculate PMCC metrics
            return self._calculate_pmcc_metrics(
                symbol, quote, leaps, short_call
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
```

### Error Recovery Pattern

```python
class ResilientMarketDataClient:
    """Client with comprehensive error handling."""
    
    def __init__(self, primary_token: str, backup_token: str = None):
        self.primary = MarketDataClient(primary_token)
        self.backup = MarketDataClient(backup_token) if backup_token else None
        self.circuit_breaker = CircuitBreaker()
    
    async def get_option_chain(self, symbol: str, **params):
        """Get option chain with fallback and circuit breaker."""
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            if self.backup:
                return await self._fetch_with_client(self.backup, symbol, **params)
            raise Exception("Service unavailable")
        
        try:
            # Try primary
            result = await self._fetch_with_client(self.primary, symbol, **params)
            self.circuit_breaker.record_success()
            return result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            
            # Try backup if available
            if self.backup:
                logger.warning(f"Primary failed, using backup: {e}")
                return await self._fetch_with_client(self.backup, symbol, **params)
            
            raise
```

## Summary

Key takeaways for MarketData.app API:

1. **Always use cached feed** unless real-time data is critical
2. **No native screener** - implement your own with batch quotes
3. **Filter at API level** to minimize data transfer and credits
4. **Handle array-based responses** carefully with validation
5. **Implement proper error handling** including 204 responses
6. **Use connection pooling** and concurrent requests wisely
7. **Monitor rate limits** and implement appropriate backoff
8. **Cache responses** where appropriate to minimize API calls

The MarketData.app API is powerful and cost-effective when used correctly, but requires careful attention to its unique characteristics and credit system.
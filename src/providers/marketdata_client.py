"""
MarketData.app API client implementation for options trading applications.

This module provides a complete implementation of the MarketData.app API for:
- Stock quotes and market data
- Comprehensive options chains with Greeks
- Options expirations and contract details
- Real-time and cached data feeds
- Credit-efficient data retrieval
- Array-based response parsing

Based on production usage patterns from the PMCC Scanner application.
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal

import aiohttp
from pydantic import ValidationError

from src.providers.base_provider import MarketDataProvider
from src.models import (
    ScreeningCriteria,
    StockQuote,
    OptionContract,
    FundamentalData,
    ProviderResponse,
    ProviderType,
    HealthCheckResult,
    ProviderStatus,
    OptionType
)
from src.providers.exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidSymbolError,
    InsufficientDataError,
    DataQualityError,
    TimeoutError,
    QuotaExceededError
)

logger = logging.getLogger(__name__)


class MarketDataClient(MarketDataProvider):
    """
    MarketData.app API client optimized for options trading applications.
    
    Key Features:
    - Credit-efficient operations (cached vs live feeds)
    - Array-based response parsing
    - Options chain filtering and analysis
    - Stock quotes for pre-defined universe
    - Rate limiting and error handling
    - Comprehensive options Greeks
    
    Important Notes:
    - No native stock screener (requires predefined symbol list)
    - Cached feed = 1 credit per request vs Live feed = 1 credit per contract
    - All endpoints require trailing slashes
    - Response data is in parallel arrays, not objects
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("MarketData.app API key is required")
        
        self.base_url = config.get("base_url", "https://api.marketdata.app/v1")
        self.provider_type = ProviderType.MARKETDATA
        
        # MarketData.app specific settings
        self.default_feed = config.get("default_feed", "cached")  # cached or live
        self.max_concurrent_requests = config.get("max_concurrent_requests", 50)
        
        # Credit costs by endpoint
        self.credit_costs = {
            'stock_quote': 1,
            'options_chain_cached': 1,
            'options_chain_live_per_contract': 1,
            'options_expirations': 1
        }
        
        # Credit tracking
        self._daily_credits_used = 0
        self._daily_credit_limit = config.get("daily_credit_limit", 10000)
        
        logger.info(f"MarketData client initialized with base URL: {self.base_url}")
    
    async def get_health_status(self) -> HealthCheckResult:
        """Check MarketData.app API health and availability."""
        try:
            # Test with a simple stock quote
            start_time = datetime.now()
            response = await self._make_request(
                "GET",
                f"{self.base_url}/stocks/quotes/AAPL/",
                headers=self._get_headers()
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.success:
                return HealthCheckResult(
                    provider=self.provider_type,
                    status=ProviderStatus.HEALTHY,
                    response_time_ms=response_time,
                    message="MarketData.app API is responding normally"
                )
            else:
                return HealthCheckResult(
                    provider=self.provider_type,
                    status=ProviderStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    message=f"MarketData.app API error: {response.error}"
                )
                
        except Exception as e:
            logger.error(f"MarketData health check failed: {e}")
            return HealthCheckResult(
                provider=self.provider_type,
                status=ProviderStatus.UNHEALTHY,
                message=str(e)
            )
    
    async def screen_stocks(self, criteria: ScreeningCriteria) -> List[str]:
        """
        MarketData.app has no native screener - requires predefined symbol universe.
        This method fetches quotes for a predefined list and applies local filtering.
        """
        logger.warning("MarketData.app has no native stock screener. Using predefined universe with local filtering.")
        
        # You would need to provide a predefined universe of symbols
        # This is a limitation of MarketData.app vs other providers
        predefined_universe = self._get_predefined_universe()
        
        if not predefined_universe:
            logger.error("No predefined stock universe configured for MarketData.app screening")
            return []
        
        # Fetch quotes for universe
        quotes = await self.get_stock_quotes(predefined_universe)
        
        # Apply local filtering based on criteria
        filtered_symbols = []
        for symbol, quote in quotes.items():
            if self._meets_screening_criteria(quote, criteria):
                filtered_symbols.append(symbol)
        
        logger.info(f"MarketData screening returned {len(filtered_symbols)} stocks from {len(predefined_universe)} universe")
        return filtered_symbols
    
    async def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get current stock quote for a single symbol."""
        try:
            response = await self._make_request(
                "GET",
                f"{self.base_url}/stocks/quotes/{symbol}/",
                headers=self._get_headers()
            )
            
            if not response.success or not response.data:
                if response.error and "not found" in response.error.lower():
                    raise InvalidSymbolError(symbol, provider="marketdata")
                return None
            
            quote_data = response.data
            
            # MarketData.app returns arrays even for single quotes
            # Extract first element from arrays
            def get_first(data, key, default=None):
                val = data.get(key, default)
                if isinstance(val, list) and len(val) > 0:
                    return val[0]
                return val if val is not None else default
            
            quote = StockQuote(
                symbol=symbol,
                last_price=get_first(quote_data, 'last', 0.0),
                change=get_first(quote_data, 'change', 0.0),
                change_percent=get_first(quote_data, 'changepct', 0.0),
                volume=get_first(quote_data, 'volume', 0),
                bid=get_first(quote_data, 'bid'),
                ask=get_first(quote_data, 'ask'),
                high=get_first(quote_data, 'high'),
                low=get_first(quote_data, 'low'),
                open=get_first(quote_data, 'open'),
                timestamp=datetime.now()  # MarketData doesn't always provide timestamp
            )
            
            self._daily_credits_used += self.credit_costs['stock_quote']
            return quote
            
        except ValidationError as e:
            logger.error(f"Quote validation error for {symbol}: {e}")
            raise DataQualityError(f"Invalid quote data for {symbol}: {str(e)}", provider="marketdata")
        except Exception as e:
            if isinstance(e, InvalidSymbolError):
                raise
            logger.debug(f"Error getting quote for {symbol}: {e}")
            return None
    
    async def get_stock_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """
        Get multiple stock quotes using concurrent requests.
        MarketData.app has no bulk quote endpoint, so we use concurrent individual requests.
        """
        try:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            async def fetch_with_limit(symbol):
                async with semaphore:
                    return await self.get_stock_quote(symbol)
            
            # Fetch all quotes concurrently
            logger.info(f"Fetching {len(symbols)} stock quotes concurrently")
            tasks = [fetch_with_limit(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            quotes = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch quote for {symbol}: {result}")
                elif result is not None:
                    quotes[symbol] = result
            
            logger.info(f"Successfully retrieved {len(quotes)} quotes from {len(symbols)} requested")
            return quotes
            
        except Exception as e:
            logger.error(f"Error getting bulk quotes: {e}")
            raise ProviderError(f"Bulk quote retrieval failed: {str(e)}", provider="marketdata")
    
    async def get_options_chain(
        self, 
        symbol: str, 
        expiration_date: Optional[str] = None,
        **filters
    ) -> List[OptionContract]:
        """
        Get options chain with comprehensive filtering.
        
        Uses cached feed by default to minimize credit consumption.
        Live feed would cost 1 credit per contract returned!
        """
        try:
            # Build query parameters
            params = {}
            # DO NOT set feed parameter - let API default to cached data (status 203)
            
            # Date filtering
            if expiration_date:
                params['expiration'] = expiration_date
            elif filters.get('from_date') or filters.get('to_date'):
                if filters.get('from_date'):
                    params['from'] = filters['from_date']
                if filters.get('to_date'):
                    params['to'] = filters['to_date']
            
            # Option type filtering
            if filters.get('option_type'):
                option_type = filters['option_type']
                if isinstance(option_type, OptionType):
                    params['side'] = option_type.value
                else:
                    params['side'] = str(option_type).lower()
            
            # Greeks filtering
            if filters.get('min_delta') or filters.get('max_delta'):
                min_delta = filters.get('min_delta', 0)
                max_delta = filters.get('max_delta', 1)
                params['delta'] = f"{min_delta}-{max_delta}"
            
            if filters.get('min_iv') or filters.get('max_iv'):
                min_iv = filters.get('min_iv', 0)
                max_iv = filters.get('max_iv', 2)
                params['iv'] = f"{min_iv}-{max_iv}"
            
            # Liquidity filtering
            if filters.get('min_volume'):
                params['minVolume'] = filters['min_volume']
            if filters.get('min_open_interest'):
                params['minOpenInterest'] = filters['min_open_interest']
            
            # Strike filtering
            if filters.get('strike_limit'):
                params['strikeLimit'] = filters['strike_limit']
            elif filters.get('min_strike') or filters.get('max_strike'):
                min_strike = filters.get('min_strike', 0)
                max_strike = filters.get('max_strike', 9999)
                params['strike'] = f"{min_strike}-{max_strike}"
            
            # Make API request
            response = await self._make_request(
                "GET",
                f"{self.base_url}/options/chain/{symbol}/",  # Note: trailing slash required!
                headers=self._get_headers(),
                params=params
            )
            
            if not response.success:
                if response.error and "not found" in response.error.lower():
                    raise InvalidSymbolError(symbol, provider="marketdata")
                elif response.error and "204" in str(response.error):
                    logger.info(f"No cached options data for {symbol}. Consider using live feed if critical.")
                    return []
                logger.warning(f"Options chain request failed for {symbol}: {response.error}")
                return []
            
            if not response.data:
                logger.warning(f"No options data returned for {symbol}")
                return []
            
            # Parse array-based response
            contracts = self._parse_options_chain_response(response.data, symbol)
            
            # Update credit usage based on response status
            # Status 203 = cached data (1 credit per request)
            # Status 200 = live data (1 credit per contract)
            # Check if status is stored in the response object
            status_code = getattr(response, '_status_code', 203)  # Default to cached if not set
            
            if status_code == 203:
                self._daily_credits_used += self.credit_costs['options_chain_cached']
                feed_type = 'cached'
                logger.info(f"✓ Using CACHED feed for {symbol}: {len(contracts)} contracts for 1 credit")
            else:
                # Live feed costs 1 credit per contract - WARN!
                credits_used = len(contracts) * self.credit_costs['options_chain_live_per_contract']
                self._daily_credits_used += credits_used
                feed_type = 'live'
                logger.warning(f"⚠️ Using LIVE feed for {symbol}: {len(contracts)} contracts = {credits_used} credits!")
            
            logger.info(f"Retrieved {len(contracts)} option contracts for {symbol} using {feed_type} feed (status {status_code})")
            return contracts
            
        except Exception as e:
            if isinstance(e, InvalidSymbolError):
                raise
            logger.debug(f"Error getting options chain for {symbol}: {e}")
            raise ProviderError(f"Options chain retrieval failed: {str(e)}", provider="marketdata")
    
    async def get_pmcc_option_chains(self, symbol: str) -> Dict[str, List[OptionContract]]:
        """
        Get options chains optimized for PMCC strategy analysis.
        Makes 2 targeted requests: LEAPS and short calls.
        """
        try:
            today = datetime.now().date()
            
            # Request 1: LEAPS (6-12 months, deep ITM)
            leaps_params = {
                'from': (today + timedelta(days=180)).isoformat(),
                'to': (today + timedelta(days=365)).isoformat(),
                'side': 'call',
                'delta': '.70-.95',  # Deep ITM
                'minOpenInterest': 10
                # NO feed parameter - let API default to cached
            }
            
            # Request 2: Short calls (21-45 days, OTM)
            short_params = {
                'from': (today + timedelta(days=21)).isoformat(),
                'to': (today + timedelta(days=45)).isoformat(),
                'side': 'call',
                'delta': '.15-.40',  # OTM
                'minOpenInterest': 5
                # NO feed parameter - let API default to cached
            }
            
            # Make both requests concurrently
            leaps_task = self.get_options_chain(symbol, **leaps_params)
            shorts_task = self.get_options_chain(symbol, **short_params)
            
            leaps, shorts = await asyncio.gather(leaps_task, shorts_task)
            
            return {
                'leaps': leaps,
                'short_calls': shorts
            }
            
        except Exception as e:
            logger.error(f"Error getting PMCC chains for {symbol}: {e}")
            raise ProviderError(f"PMCC options chain retrieval failed: {str(e)}", provider="marketdata")
    
    async def get_options_expirations(self, symbol: str) -> List[str]:
        """Get available option expiration dates for a symbol."""
        try:
            response = await self._make_request(
                "GET",
                f"{self.base_url}/options/expirations/{symbol}/",  # Note: trailing slash required!
                headers=self._get_headers()
            )
            
            if not response.success or not response.data:
                if response.error and "not found" in response.error.lower():
                    raise InvalidSymbolError(symbol, provider="marketdata")
                return []
            
            expirations = response.data
            if isinstance(expirations, list):
                # Convert timestamps to ISO dates if needed
                formatted_expirations = []
                for exp in expirations:
                    if isinstance(exp, (int, float)):
                        # Timestamp
                        date_obj = datetime.fromtimestamp(exp).date()
                        formatted_expirations.append(date_obj.isoformat())
                    else:
                        # Already string
                        formatted_expirations.append(str(exp))
                
                self._daily_credits_used += self.credit_costs['options_expirations']
                return formatted_expirations
            
            return []
            
        except Exception as e:
            if isinstance(e, InvalidSymbolError):
                raise
            logger.error(f"Error getting expirations for {symbol}: {e}")
            return []
    
    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """
        MarketData.app has limited fundamental data.
        This method returns None as fundamental data should come from EODHD.
        """
        logger.warning("MarketData.app has limited fundamental data. Use EODHD for comprehensive fundamentals.")
        return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with Bearer token authentication."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'User-Agent': 'OptionsScanner/1.0'
        }
    
    def _get_predefined_universe(self) -> List[str]:
        """
        Get predefined stock universe for screening.
        In production, this would come from configuration or external source.
        """
        # This is a sample universe - in production you'd load from config
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'ORCL', 'IBM',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'COF',
            'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'UNH', 'CVS', 'AMGN',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO',
            'WMT', 'HD', 'LOW', 'TGT', 'COST', 'AMZN', 'EBAY', 'ETSY'
        ]
    
    def _meets_screening_criteria(self, quote: StockQuote, criteria: ScreeningCriteria) -> bool:
        """Apply screening criteria to a stock quote (local filtering)."""
        # Price filters
        if criteria.min_price and quote.last_price < criteria.min_price:
            return False
        if criteria.max_price and quote.last_price > criteria.max_price:
            return False
        
        # Volume filter
        if criteria.min_volume and quote.volume < criteria.min_volume:
            return False
        
        # Symbol exclusion
        if quote.symbol in criteria.exclude_symbols:
            return False
        
        # Note: Market cap, sectors, and other filters would require additional data
        # that's not available in basic stock quotes from MarketData.app
        
        return True
    
    def _parse_options_chain_response(self, data: Dict[str, Any], underlying_symbol: str) -> List[OptionContract]:
        """
        Parse MarketData.app's array-based options chain response.
        
        MarketData.app returns data as parallel arrays where each index
        across all arrays represents one option contract.
        """
        try:
            contracts = []
            
            # Validate that all arrays have the same length
            if not self._validate_array_response(data):
                logger.error("Array length mismatch in options chain response")
                return []
            
            # Get array length (all should be same)
            array_keys = ['optionSymbol', 'strike', 'expiration', 'side']
            length = 0
            for key in array_keys:
                if key in data and isinstance(data[key], list):
                    length = len(data[key])
                    break
            
            if length == 0:
                logger.warning("No valid options data arrays found")
                return []
            
            # Parse each contract
            for i in range(length):
                try:
                    contract = self._parse_single_option_contract(data, i, underlying_symbol)
                    if contract:
                        contracts.append(contract)
                except Exception as e:
                    logger.warning(f"Skipping invalid option contract at index {i}: {e}")
                    continue
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error parsing options chain response: {e}")
            return []
    
    def _validate_array_response(self, data: Dict[str, Any]) -> bool:
        """Ensure all arrays in response have same length."""
        lengths = set()
        for key, value in data.items():
            if isinstance(value, list):
                lengths.add(len(value))
        
        if len(lengths) > 1:
            logger.error(f"Array length mismatch in response: {lengths}")
            return False
        
        return True
    
    def _parse_single_option_contract(self, data: Dict[str, Any], index: int, underlying_symbol: str) -> Optional[OptionContract]:
        """Parse a single option contract from array data at given index."""
        try:
            # Required fields
            option_symbol = self._safe_get_array_value(data, 'optionSymbol', index)
            strike = self._safe_get_array_value(data, 'strike', index)
            expiration_ts = self._safe_get_array_value(data, 'expiration', index)
            side = self._safe_get_array_value(data, 'side', index)
            
            if not all([option_symbol, strike is not None, expiration_ts, side]):
                return None
            
            # Parse expiration date
            if isinstance(expiration_ts, (int, float)):
                expiration = datetime.fromtimestamp(expiration_ts).date()
            else:
                expiration = datetime.fromisoformat(str(expiration_ts)).date()
            
            # Parse option type
            option_type = OptionType.CALL if side.lower() == 'call' else OptionType.PUT
            
            # Calculate days to expiration
            dte = (expiration - datetime.now().date()).days
            
            # Parse pricing data
            bid = self._safe_get_array_value(data, 'bid', index)
            ask = self._safe_get_array_value(data, 'ask', index)
            last = self._safe_get_array_value(data, 'last', index)
            
            # Parse Greeks
            delta = self._safe_get_array_value(data, 'delta', index)
            gamma = self._safe_get_array_value(data, 'gamma', index)
            theta = self._safe_get_array_value(data, 'theta', index)
            vega = self._safe_get_array_value(data, 'vega', index)
            rho = self._safe_get_array_value(data, 'rho', index)
            iv = self._safe_get_array_value(data, 'iv', index)
            
            # Parse market data
            volume = self._safe_get_array_value(data, 'volume', index)
            open_interest = self._safe_get_array_value(data, 'openInterest', index)
            
            contract = OptionContract(
                option_symbol=option_symbol,
                underlying_symbol=underlying_symbol,
                strike=Decimal(str(strike)),
                expiration=expiration,
                option_type=option_type,
                bid=Decimal(str(bid)) if bid is not None else None,
                ask=Decimal(str(ask)) if ask is not None else None,
                last=Decimal(str(last)) if last is not None else None,
                volume=volume,
                open_interest=open_interest,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                implied_volatility=iv,
                days_to_expiration=dte if dte >= 0 else 0
            )
            
            return contract
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing option contract at index {index}: {e}")
            return None
    
    def _safe_get_array_value(self, data: Dict[str, Any], key: str, index: int, default=None):
        """Safely extract value from array with defaults."""
        arr = data.get(key, [])
        if isinstance(arr, list) and len(arr) > index:
            value = arr[index]
            # Handle null/None values
            return value if value is not None else default
        return default
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Override base _make_request to handle MarketData.app-specific errors.
        """
        try:
            response = await super()._make_request(method, url, headers, params, data, **kwargs)
            
            # Handle MarketData.app-specific status codes
            if hasattr(response, '_status_code'):
                status_code = response._status_code
                
                if status_code == 203:
                    # Success with cached data
                    logger.debug("Received cached data (203)")
                elif status_code == 204:
                    # No cached data available
                    logger.info("No cached data available (204). Consider using live feed.")
                    return ProviderResponse(
                        success=False,
                        error="No cached data available",
                        provider=self.provider_type
                    )
                elif status_code == 402:
                    raise QuotaExceededError(
                        "MarketData.app plan limit exceeded",
                        provider="marketdata"
                    )
            
            return response
            
        except Exception as e:
            if isinstance(e, (RateLimitError, AuthenticationError, QuotaExceededError)):
                raise
            
            logger.error(f"MarketData request failed: {e}")
            return ProviderResponse(
                success=False,
                error=str(e),
                provider=self.provider_type
            )
"""
EODHD API client implementation for comprehensive market data access.

This module provides a complete implementation of the EODHD API for:
- Stock screening with market cap, volume, and price filters
- Historical and real-time price data
- Technical indicators (RSI, ADX, ATR, Bollinger Bands)
- Fundamental data and financial metrics
- Corporate calendar events (earnings dates)
- Market news and macro context data
- Options chains and Greeks data

Based on production usage patterns and the official EODHD API documentation.
"""

import asyncio
import json
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
    TechnicalIndicators,
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


class EODHDClient(MarketDataProvider):
    """
    EODHD API client implementing comprehensive market data access.
    
    Features:
    - Stock screening with comprehensive filters
    - Batch price data retrieval
    - Technical indicators calculation
    - Fundamental data access
    - Corporate calendar events
    - News and macro context
    - Options chains with Greeks
    - Rate limiting and error handling
    - Response caching
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("EODHD API key is required")
        
        self.base_url = config.get("base_url", "https://eodhd.com/api")
        self.provider_type = ProviderType.EODHD
        
        # EODHD specific settings
        self.screener_credit_cost = config.get("screener_credit_cost", 5)
        self.max_screening_results = config.get("max_screening_results", 500)  # API max is 500
        
        # Credit tracking
        self._daily_credits_used = 0
        self._daily_credit_limit = config.get("daily_credit_limit", 100000)
        
        logger.info(f"EODHD client initialized with base URL: {self.base_url}")
    
    async def get_health_status(self) -> HealthCheckResult:
        """Check EODHD API health and availability."""
        try:
            # Test with a simple API call
            test_params = {
                'api_token': self.api_key,
                'limit': 1
            }
            
            start_time = datetime.now()
            response = await self._make_request(
                "GET",
                f"{self.base_url}/screener",
                params=test_params
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.success:
                return HealthCheckResult(
                    provider=self.provider_type,
                    status=ProviderStatus.HEALTHY,
                    response_time_ms=response_time,
                    message="EODHD API is responding normally"
                )
            else:
                return HealthCheckResult(
                    provider=self.provider_type,
                    status=ProviderStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    message=f"EODHD API error: {response.error}"
                )
                
        except Exception as e:
            logger.error(f"EODHD health check failed: {e}")
            return HealthCheckResult(
                provider=self.provider_type,
                status=ProviderStatus.UNHEALTHY,
                message=str(e)
            )
    
    async def screen_stocks(self, criteria: ScreeningCriteria) -> List[str]:
        """
        Screen stocks using EODHD's powerful screening API.
        
        Handles the 1000-result limit by automatically splitting queries
        into market cap ranges when needed.
        """
        try:
            logger.info(f"Starting EODHD stock screening with criteria: {criteria}")
            
            # Build EODHD filter array
            filters = self._build_screening_filters(criteria)
            
            # Use comprehensive screening for large result sets
            if (criteria.min_market_cap and criteria.max_market_cap and 
                criteria.max_market_cap - criteria.min_market_cap > 1_000_000_000):
                
                logger.info("Using range-based screening for large market cap range")
                results = await self._screen_stocks_comprehensive(
                    filters, 
                    criteria.min_market_cap or 50_000_000,
                    criteria.max_market_cap or 5_000_000_000
                )
            else:
                # Single query screening
                results = await self._screen_stocks_single(filters)
            
            # Extract symbols from results
            symbols = []
            for stock in results:
                if stock.get('code'):
                    symbols.append(stock['code'])
            
            logger.info(f"EODHD screening returned {len(symbols)} stocks")
            return symbols
            
        except Exception as e:
            logger.debug(f"EODHD screening failed: {e}")
            raise ProviderError(f"Stock screening failed: {str(e)}", provider="eodhd")
    
    async def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get current stock quote using real-time endpoint."""
        try:
            # Use real-time endpoint for current price
            params = {
                'api_token': self.api_key,
                'fmt': 'json'
            }
            
            response = await self._make_request(
                "GET",
                f"{self.base_url}/real-time/{symbol}.US",
                params=params
            )
            
            if not response.success or not response.data:
                return None
            
            data = response.data
            
            # Real-time endpoint returns a single dict with current data
            quote = StockQuote(
                symbol=symbol,
                last_price=data.get('close', 0.0),
                change=data.get('change', 0.0),
                change_percent=data.get('change_p', 0.0),
                volume=data.get('volume', 0),
                high=data.get('high', data.get('close', 0.0)),
                low=data.get('low', data.get('close', 0.0)),
                open=data.get('open', data.get('close', 0.0)),
                timestamp=datetime.now()  # Real-time data is current
            )
            
            return quote
            
        except ValidationError as e:
            logger.error(f"Quote validation error for {symbol}: {e}")
            raise DataQualityError(f"Invalid quote data for {symbol}: {str(e)}", provider="eodhd")
        except Exception as e:
            logger.debug(f"Error getting quote for {symbol}: {e}")
            return None
    
    async def get_stock_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get multiple stock quotes efficiently using bulk endpoint."""
        try:
            # EODHD bulk endpoint for multiple symbols
            params = {
                'api_token': self.api_key,
                'fmt': 'json'
            }
            
            response = await self._make_request(
                "GET",
                f"{self.base_url}/eod-bulk-last-day/US",
                params=params
            )
            
            if not response.success or not response.data:
                logger.error("Failed to get bulk quotes from EODHD")
                return {}
            
            # Filter to requested symbols and create quotes
            symbol_set = set(symbols)
            quotes = {}
            
            for stock_data in response.data:
                symbol = stock_data.get('code', '')
                if symbol in symbol_set:
                    try:
                        quote = StockQuote(
                            symbol=symbol,
                            last_price=stock_data['adjusted_close'],
                            change=stock_data.get('change', 0.0),
                            change_percent=stock_data.get('change_p', 0.0),
                            volume=stock_data.get('volume', 0),
                            high=stock_data.get('high', stock_data['adjusted_close']),
                            low=stock_data.get('low', stock_data['adjusted_close']),
                            open=stock_data.get('open', stock_data['adjusted_close']),
                            timestamp=datetime.strptime(stock_data['date'], '%Y-%m-%d')
                        )
                        quotes[symbol] = quote
                    except (ValidationError, KeyError) as e:
                        logger.warning(f"Skipping invalid quote data for {symbol}: {e}")
                        continue
            
            logger.info(f"Retrieved {len(quotes)} quotes from EODHD bulk endpoint")
            return quotes
            
        except Exception as e:
            logger.error(f"Error getting bulk quotes: {e}")
            raise ProviderError(f"Bulk quote retrieval failed: {str(e)}", provider="eodhd")
    
    async def get_options_chain(
        self, 
        symbol: str, 
        expiration_date: Optional[str] = None,
        **filters
    ) -> List[OptionContract]:
        """
        Get options chain for symbol with comprehensive filtering.
        
        CRITICAL: Always filters by recent tradetime to ensure current prices,
        not stale historical data from previous trading days.
        """
        try:
            # Calculate recent tradetime filter
            tradetime_filter = self._get_tradetime_filter()
            
            base_params = {
                'api_token': self.api_key,
                'filter[underlying_symbol]': symbol,
                'filter[tradetime_from]': tradetime_filter,
                'sort': '-tradetime',
                'page[limit]': 1000
            }
            
            # Add expiration filter if specified
            if expiration_date:
                base_params['filter[exp_date]'] = expiration_date
            
            # Apply additional filters
            if filters.get('option_type'):
                option_type = filters['option_type']
                if isinstance(option_type, OptionType):
                    base_params['filter[type]'] = option_type.value
                else:
                    base_params['filter[type]'] = str(option_type).lower()
            
            if filters.get('min_strike'):
                base_params['filter[strike_from]'] = filters['min_strike']
            if filters.get('max_strike'):
                base_params['filter[strike_to]'] = filters['max_strike']
            
            if filters.get('min_volume'):
                base_params['filter[volume_from]'] = filters['min_volume']
            if filters.get('min_open_interest'):
                base_params['filter[open_interest_from]'] = filters['min_open_interest']
            
            # Make API request
            response = await self._make_request(
                "GET",
                f"{self.base_url}/options/{symbol}",
                params=base_params
            )
            
            if not response.success or not response.data:
                logger.warning(f"No options data available for {symbol}")
                return []
            
            # Parse response data
            contracts = []
            data = response.data.get('data', []) if isinstance(response.data, dict) else response.data
            
            for option_data in data:
                try:
                    contract = self._parse_option_contract(option_data, symbol)
                    if contract:
                        contracts.append(contract)
                except Exception as e:
                    logger.warning(f"Skipping invalid option contract: {e}")
                    continue
            
            logger.info(f"Retrieved {len(contracts)} option contracts for {symbol}")
            return contracts
            
        except Exception as e:
            logger.debug(f"Error getting options chain for {symbol}: {e}")
            raise ProviderError(f"Options chain retrieval failed: {str(e)}", provider="eodhd")
    
    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get comprehensive fundamental data for a symbol."""
        try:
            params = {
                'api_token': self.api_key
            }
            
            response = await self._make_request(
                "GET",
                f"{self.base_url}/fundamentals/{symbol}.US",
                params=params
            )
            
            if not response.success or not response.data:
                return None
            
            data = response.data
            
            # Extract data from nested structure
            general = data.get('General', {})
            financials = data.get('Financials', {})
            valuation = data.get('Valuation', {})
            highlights = data.get('Highlights', {})
            
            fundamental_data = FundamentalData(
                symbol=symbol,
                market_cap=valuation.get('MarketCapitalization'),
                pe_ratio=valuation.get('TrailingPE'),
                pb_ratio=valuation.get('PriceBookMRQ'),
                ps_ratio=valuation.get('PriceSalesTTM'),
                roe=highlights.get('ReturnOnEquityTTM'),
                roa=highlights.get('ReturnOnAssetsTTM'),
                profit_margin=highlights.get('ProfitMargin'),
                revenue_growth=highlights.get('RevenueGrowthTTM'),
                earnings_growth=highlights.get('QuarterlyEarningsGrowthYOY'),
                debt_to_equity=highlights.get('TotalDebtToEquity'),
                current_ratio=highlights.get('CurrentRatio'),
                dividend_yield=highlights.get('DividendYield')
            )
            
            return fundamental_data
            
        except ValidationError as e:
            logger.error(f"Fundamental data validation error for {symbol}: {e}")
            raise DataQualityError(f"Invalid fundamental data for {symbol}: {str(e)}", provider="eodhd")
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return None
    
    async def get_technical_indicators(self, symbol: str, period: int = 14) -> Optional[TechnicalIndicators]:
        """Get technical indicators for a symbol."""
        try:
            indicators = {}
            
            # RSI
            rsi_data = await self._get_technical_indicator(symbol, 'rsi', period)
            if rsi_data:
                indicators['rsi_14'] = rsi_data[-1]['rsi'] if rsi_data else None
            
            # Moving averages
            sma_20 = await self._get_technical_indicator(symbol, 'sma', 20)
            if sma_20:
                indicators['sma_20'] = sma_20[-1]['sma']
            
            sma_50 = await self._get_technical_indicator(symbol, 'sma', 50)
            if sma_50:
                indicators['sma_50'] = sma_50[-1]['sma']
            
            sma_200 = await self._get_technical_indicator(symbol, 'sma', 200)
            if sma_200:
                indicators['sma_200'] = sma_200[-1]['sma']
            
            # Bollinger Bands
            bb_data = await self._get_technical_indicator(symbol, 'bbands', 20)
            if bb_data:
                bb = bb_data[-1]
                indicators.update({
                    'bollinger_upper': bb.get('upper_band'),
                    'bollinger_middle': bb.get('middle_band'),
                    'bollinger_lower': bb.get('lower_band')
                })
            
            # ATR
            atr_data = await self._get_technical_indicator(symbol, 'atr', 14)
            if atr_data:
                indicators['atr_14'] = atr_data[-1]['atr']
            
            technical_indicators = TechnicalIndicators(
                symbol=symbol,
                **indicators
            )
            
            return technical_indicators
            
        except ValidationError as e:
            logger.error(f"Technical indicators validation error for {symbol}: {e}")
            raise DataQualityError(f"Invalid technical indicators for {symbol}: {str(e)}", provider="eodhd")
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {e}")
            return None
    
    async def get_earnings_calendar(self, from_date: date, to_date: date) -> List[Dict[str, Any]]:
        """Get earnings calendar for date range."""
        try:
            params = {
                'api_token': self.api_key,
                'from': from_date.isoformat(),
                'to': to_date.isoformat()
            }
            
            response = await self._make_request(
                "GET",
                f"{self.base_url}/calendar/earnings",
                params=params
            )
            
            if response.success and response.data:
                return response.data
            return []
            
        except Exception as e:
            logger.error(f"Error getting earnings calendar: {e}")
            return []
    
    async def get_news(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent news for a symbol."""
        try:
            params = {
                'api_token': self.api_key,
                's': symbol,
                'limit': limit
            }
            
            response = await self._make_request(
                "GET",
                f"{self.base_url}/news",
                params=params
            )
            
            if response.success and response.data:
                return response.data
            return []
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []
    
    def _build_screening_filters(self, criteria: ScreeningCriteria) -> List[List[str]]:
        """Build EODHD filter array from screening criteria."""
        filters = []
        
        # Market cap filters
        if criteria.min_market_cap:
            filters.append(["market_capitalization", ">=", criteria.min_market_cap])
        if criteria.max_market_cap:
            filters.append(["market_capitalization", "<=", criteria.max_market_cap])
        
        # Price filters
        if criteria.min_price:
            filters.append(["adjusted_close", ">=", criteria.min_price])
        if criteria.max_price:
            filters.append(["adjusted_close", "<=", criteria.max_price])
        
        # Volume filter
        if criteria.min_volume:
            filters.append(["avgvol_200d", ">=", criteria.min_volume])
        
        # Exchange filter (US stocks only)
        filters.append(["exchange", "=", "us"])
        
        # Exclude sectors
        for sector in criteria.exclude_sectors:
            filters.append(["sector", "!=", sector])
        
        # Exclude symbols
        for symbol in criteria.exclude_symbols:
            filters.append(["code", "!=", symbol])
        
        return filters
    
    async def _screen_stocks_single(self, filters: List[List[str]]) -> List[Dict[str, Any]]:
        """Perform single screening query."""
        params = {
            'api_token': self.api_key,
            'filters': json.dumps(filters),
            'sort': 'market_capitalization.desc',
            'limit': self.max_screening_results
        }
        
        logger.info(f"EODHD screener filters: {json.dumps(filters)}")
        logger.debug(f"Making EODHD screener API call...")
        response = await self._make_request(
            "GET",
            f"{self.base_url}/screener",
            params=params
        )
        
        if response.success and response.data:
            self._daily_credits_used += self.screener_credit_cost
            result_data = response.data.get('data', []) if isinstance(response.data, dict) else response.data
            logger.info(f"EODHD screener returned {len(result_data)} results")
            if result_data and len(result_data) > 0:
                logger.info(f"First result: {result_data[0].get('code', 'N/A')} - ${result_data[0].get('market_capitalization', 0)/1e9:.1f}B")
            return result_data
        else:
            logger.warning(f"EODHD screener failed - success: {response.success}, has_data: {response.data is not None}, error: {response.error if hasattr(response, 'error') else 'N/A'}")
        
        logger.debug("EODHD screener returned no results")
        return []
    
    async def _screen_stocks_comprehensive(
        self, 
        base_filters: List[List[str]], 
        min_cap: float, 
        max_cap: float
    ) -> List[Dict[str, Any]]:
        """
        Perform comprehensive screening by splitting into market cap ranges
        to handle EODHD's 1000-result limit.
        """
        ranges = self._generate_market_cap_ranges(min_cap, max_cap)
        all_results = []
        
        total_ranges = len(ranges)
        print(f"\n📈 Starting comprehensive screening across {total_ranges} market cap ranges...", flush=True)
        logger.info(f"Starting comprehensive screening with {total_ranges} ranges")
        for idx, (range_min, range_max) in enumerate(ranges, 1):
            # Show progress
            logger.info(f"📊 Screening range {idx}/{total_ranges}: ${range_min/1e9:.1f}B - ${range_max/1e9:.1f}B")
            print(f"  → Screening market cap range {idx}/{total_ranges}: ${range_min/1e9:.1f}B - ${range_max/1e9:.1f}B", flush=True)
            
            range_filters = base_filters.copy()
            range_filters.append(["market_capitalization", ">=", range_min])
            range_filters.append(["market_capitalization", "<=", range_max])
            
            try:
                # Add timeout for the API call
                results = await asyncio.wait_for(
                    self._screen_stocks_single(range_filters),
                    timeout=30.0  # 30 second timeout per range
                )
                if results:
                    all_results.extend(results)
                    print(f"    ✓ Found {len(results)} stocks in this range", flush=True)
                else:
                    print(f"    ○ No stocks found in this range", flush=True)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout screening range ${range_min/1e9:.1f}B - ${range_max/1e9:.1f}B")
                print(f"    ⚠ Timeout for this range, continuing...", flush=True)
            except Exception as e:
                logger.debug(f"Error screening range: {e}")
                print(f"    ⚠ Error for this range, continuing...", flush=True)
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Remove duplicates and sort by market cap
        unique_results = {}
        for stock in all_results:
            symbol = stock.get('code')
            if symbol and symbol not in unique_results:
                unique_results[symbol] = stock
        
        sorted_results = list(unique_results.values())
        sorted_results.sort(key=lambda x: x.get('market_capitalization', 0), reverse=True)
        
        return sorted_results
    
    def _generate_market_cap_ranges(self, min_cap: float, max_cap: float) -> List[tuple]:
        """Generate optimal market cap ranges for screening."""
        ranges = []
        current = min_cap
        
        while current < max_cap:
            # Dynamic range sizing based on market cap
            if current < 100_000_000:  # Under 100M
                range_size = 50_000_000  # 50M ranges
            elif current < 500_000_000:  # 100M-500M
                range_size = 250_000_000  # 250M ranges
            elif current < 1_000_000_000:  # 500M-1B
                range_size = 500_000_000  # 500M ranges
            elif current < 5_000_000_000:  # 1B-5B
                range_size = 2_000_000_000  # 2B ranges
            elif current < 10_000_000_000:  # 5B-10B
                range_size = 5_000_000_000  # 5B ranges
            elif current < 50_000_000_000:  # 10B-50B
                range_size = 10_000_000_000  # 10B ranges
            elif current < 100_000_000_000:  # 50B-100B
                range_size = 50_000_000_000  # 50B ranges
            elif current < 500_000_000_000:  # 100B-500B
                range_size = 100_000_000_000  # 100B ranges
            elif current < 1_000_000_000_000:  # 500B-1T
                range_size = 500_000_000_000  # 500B ranges
            else:  # Above 1T
                range_size = 1_000_000_000_000  # 1T ranges
            
            range_end = min(current + range_size, max_cap)
            ranges.append((current, range_end))
            current = range_end
            
            # Safety check: limit to reasonable number of ranges
            if len(ranges) > 50:
                logger.warning(f"Too many ranges ({len(ranges)}), capping at 50")
                break
        
        return ranges
    
    def _get_tradetime_filter(self, lookback_days: int = 5) -> str:
        """Get tradetime filter for recent trading days to ensure current data."""
        today = datetime.now()
        
        # Skip to last Friday if weekend
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        
        # Look back N trading days for better data coverage
        tradetime_from = today - timedelta(days=lookback_days)
        return tradetime_from.strftime('%Y-%m-%d')
    
    def _parse_option_contract(self, option_data: Dict[str, Any], underlying_symbol: str) -> Optional[OptionContract]:
        """Parse EODHD option data into OptionContract model."""
        try:
            # Parse expiration date
            exp_date_str = option_data.get('exp_date')
            if isinstance(exp_date_str, str):
                expiration = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
            else:
                # Assume timestamp
                expiration = datetime.fromtimestamp(exp_date_str).date()
            
            # Parse option type
            option_type_str = option_data.get('type', '').lower()
            option_type = OptionType.CALL if option_type_str == 'call' else OptionType.PUT
            
            # Calculate days to expiration
            dte = (expiration - datetime.now().date()).days
            
            contract = OptionContract(
                option_symbol=option_data.get('option_id', ''),
                underlying_symbol=underlying_symbol,
                strike=Decimal(str(option_data.get('strike', 0))),
                expiration=expiration,
                option_type=option_type,
                bid=Decimal(str(option_data.get('bid', 0))) if option_data.get('bid') is not None else None,
                ask=Decimal(str(option_data.get('ask', 0))) if option_data.get('ask') is not None else None,
                last=Decimal(str(option_data.get('last', 0))) if option_data.get('last') is not None else None,
                volume=option_data.get('volume'),
                open_interest=option_data.get('open_interest'),
                delta=option_data.get('delta'),
                gamma=option_data.get('gamma'),
                theta=option_data.get('theta'),
                vega=option_data.get('vega'),
                rho=option_data.get('rho'),
                implied_volatility=option_data.get('implied_volatility'),
                days_to_expiration=dte if dte >= 0 else 0
            )
            
            return contract
            
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error parsing option contract: {e}")
            return None
    
    async def _get_technical_indicator(
        self, 
        symbol: str, 
        indicator: str, 
        period: int,
        lookback_days: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """Get specific technical indicator data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            params = {
                'api_token': self.api_key,
                'function': indicator,
                'symbol': f'{symbol}.US',
                'period': period,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            response = await self._make_request(
                "GET",
                f"{self.base_url}/technical",
                params=params
            )
            
            if response.success and response.data:
                return response.data
            return None
            
        except Exception as e:
            logger.warning(f"Error getting {indicator} for {symbol}: {e}")
            return None
    
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
        Override base _make_request to handle EODHD-specific errors.
        """
        try:
            response = await super()._make_request(method, url, headers, params, data, **kwargs)
            
            # Handle EODHD-specific error codes
            if not response.success and hasattr(response, 'data') and response.data:
                error_data = response.data
                if isinstance(error_data, dict):
                    error_code = error_data.get('code')
                    error_message = error_data.get('message', response.error)
                    
                    if error_code == 402:
                        raise QuotaExceededError(
                            f"EODHD plan limit exceeded: {error_message}",
                            provider="eodhd"
                        )
                    elif error_code == 403:
                        raise AuthenticationError(
                            f"EODHD authentication failed: {error_message}",
                            provider="eodhd"
                        )
            
            return response
            
        except Exception as e:
            if isinstance(e, (RateLimitError, AuthenticationError, QuotaExceededError)):
                raise
            
            logger.debug(f"EODHD request failed: {e}")
            return ProviderResponse(
                success=False,
                error=str(e),
                provider=self.provider_type
            )
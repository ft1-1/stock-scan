"""Data enrichment step executor for comprehensive market data collection.

This module implements the DataEnrichmentExecutor that integrates all existing providers
to collect comprehensive data following the enhanced data patterns from examples.
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Union
import traceback
import pandas as pd
import aiohttp

from src.screener.workflow_engine import WorkflowStepExecutor
from src.models import (
    WorkflowStep,
    WorkflowStepStatus,
    WorkflowStepResult,
    WorkflowExecutionContext,
    ScreeningCriteria,
    ProviderType
)
from src.providers.provider_manager import ProviderManager, ProviderConfig, DataSource, ProviderStrategy
from src.providers.eodhd_client import EODHDClient
from src.providers.marketdata_client import MarketDataClient
from src.providers.exceptions import ProviderError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataEnrichmentExecutor(WorkflowStepExecutor):
    """
    Data enrichment executor that collects comprehensive market data for each symbol.
    
    This executor integrates:
    - EODHD API for fundamentals, technicals, news, macro data
    - MarketData API for options chains with Greeks
    - Enhanced data collection patterns from examples
    - Holiday-aware trading date logic
    - Comprehensive error handling and retries
    """
    
    def __init__(self, step: WorkflowStep):
        super().__init__(step)
        self.provider_manager: Optional[ProviderManager] = None
        self._trading_dates_cache: Optional[Dict[str, str]] = None
        self._holiday_dates_cache: Optional[set] = None
        
    async def execute_step(
        self, 
        input_data: Any, 
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute data enrichment for all symbols.
        
        Args:
            input_data: List of symbols from screening step
            context: Workflow execution context
            
        Returns:
            Dict containing enriched data for all symbols
        """
        # Input validation
        if not isinstance(input_data, list):
            raise ValueError("Input data must be a list of symbols")
            
        symbols = input_data
        if not symbols:
            logger.warning("No symbols provided for data enrichment")
            return {'enriched_data': {}, 'symbols_processed': 0, 'symbols_failed': 0}
        
        logger.info(f"Starting MINIMAL data collection for local rating of {len(symbols)} symbols")
        logger.info("Collecting only: historical prices, options chain, and current quote")
        
        # Initialize provider manager
        await self._initialize_providers()
        
        # Prepare trading dates
        await self._prepare_trading_dates()
        
        enriched_data = {}
        symbols_processed = 0
        symbols_failed = 0
        
        # Process symbols with controlled concurrency
        # Reduce concurrency to avoid EODHD rate limits (15 calls/second)
        # Further reduce to 1 concurrent symbol to avoid circuit breaker opening
        max_concurrent = 1  # Process one symbol at a time to avoid overwhelming EODHD
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_symbol_with_semaphore(symbol: str) -> tuple[str, Optional[Dict[str, Any]]]:
            async with semaphore:
                # Add 1 second delay between each stock to respect MarketData rate limits
                await asyncio.sleep(1.0)
                # Use minimal data collection for local rating instead of full enhanced data
                return symbol, await self._collect_minimal_data_for_rating(symbol, context)
        
        # Process all symbols concurrently
        tasks = [process_symbol_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Symbol processing failed: {result}")
                symbols_failed += 1
                continue
                
            symbol, symbol_data = result
            if symbol_data is not None:
                enriched_data[symbol] = symbol_data
                symbols_processed += 1
                logger.debug(f"Successfully enriched data for {symbol}")
            else:
                symbols_failed += 1
                logger.warning(f"Failed to enrich data for {symbol}")
        
        # Update context
        context.completed_symbols += symbols_processed
        context.failed_symbols += symbols_failed
        
        logger.info(f"Minimal data collection completed: {symbols_processed} successful, {symbols_failed} failed")
        
        # Clean up provider sessions to avoid unclosed session warnings
        try:
            if self.provider_manager:
                await self.provider_manager.__aexit__(None, None, None)
        except Exception as e:
            logger.debug(f"Provider cleanup warning: {e}")
        
        return {
            'enriched_data': enriched_data,
            'symbols_processed': symbols_processed,
            'symbols_failed': symbols_failed,
            'total_symbols': len(symbols),
            'trading_dates': self._trading_dates_cache
        }
    
    async def _initialize_providers(self) -> None:
        """Initialize the provider manager with both EODHD and MarketData clients."""
        if self.provider_manager is not None:
            return
        
        try:
            # Provider configurations
            provider_configs = []
            
            # EODHD configuration
            eodhd_config = ProviderConfig(
                provider_type=ProviderType.EODHD,
                config={
                    'type': 'eodhd',  # Add type field for BaseProvider (lowercase to match enum)
                    'api_key': self.settings.eodhd_api_key,  # Changed from api_token to api_key
                    'base_url': self.settings.eodhd_base_url,
                    'max_requests_per_minute': self.settings.eodhd_requests_per_minute,
                    'timeout_seconds': self.settings.request_timeout,
                    'requests_per_minute': 900  # EODHD allows 15 requests/second = 900/minute
                },
                priority=1,
                capabilities=[
                    'stock_screening', 'stock_quotes', 'fundamental_data',
                    'technical_indicators', 'earnings_calendar', 'news',
                    'economic_events', 'sentiment_analysis'
                ],
                max_requests_per_minute=900,  # EODHD allows 15 requests/second
                cost_per_request=0.01,
                enabled=True
            )
            provider_configs.append(eodhd_config)
            
            # MarketData configuration  
            marketdata_config = ProviderConfig(
                provider_type=ProviderType.MARKETDATA,
                config={
                    'type': 'marketdata',  # Add type field for BaseProvider (lowercase to match enum)
                    'api_key': self.settings.marketdata_api_key,  # Changed from api_token to api_key
                    'base_url': self.settings.marketdata_base_url,
                    'max_requests_per_minute': self.settings.marketdata_requests_per_minute,
                    'timeout_seconds': self.settings.request_timeout
                },
                priority=2,
                capabilities=['stock_quotes', 'options_chains', 'options_greeks'],
                max_requests_per_minute=100,
                cost_per_request=0.005,
                enabled=True
            )
            provider_configs.append(marketdata_config)
            
            # Initialize provider manager
            self.provider_manager = ProviderManager(
                provider_configs=provider_configs,
                strategy=ProviderStrategy.FAILOVER,
                data_source_preference=DataSource.EODHD_PRIMARY,
                enable_caching=True,
                enable_validation=True,
                min_quality_score=80.0,
                max_retry_attempts=3,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=300
            )
            
            # Initialize providers
            async with self.provider_manager:
                pass  # Context manager handles initialization
            
            logger.info("Provider manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize providers: {e}")
            raise ProviderError(f"Provider initialization failed: {str(e)}")
    
    async def _prepare_trading_dates(self) -> None:
        """Prepare trading dates with holiday awareness following the enhanced example."""
        if self._trading_dates_cache is not None:
            return
        
        try:
            today = datetime.now()
            
            # Get last trading day with holiday handling
            last_trading_day = await self._get_last_trading_day(today)
            last_trading_date = datetime.strptime(last_trading_day, '%Y-%m-%d')
            
            self._trading_dates_cache = {
                'today': last_trading_day,
                'thirty_days_ago': (last_trading_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'sixty_days_ago': (last_trading_date - timedelta(days=60)).strftime('%Y-%m-%d'),
                'six_months_ago': (last_trading_date - timedelta(days=180)).strftime('%Y-%m-%d'),
                'one_year_ago': (last_trading_date - timedelta(days=365)).strftime('%Y-%m-%d'),
                'hundred_days_ago': (last_trading_date - timedelta(days=100)).strftime('%Y-%m-%d'),
                'two_hundred_fifty_days_ago': (last_trading_date - timedelta(days=250)).strftime('%Y-%m-%d'),
                'ninety_days_future': (last_trading_date + timedelta(days=90)).strftime('%Y-%m-%d'),
                'one_year_future': (last_trading_date + timedelta(days=365)).strftime('%Y-%m-%d')
            }
            
            logger.info(f"Trading dates prepared with last trading day: {last_trading_day}")
            
        except Exception as e:
            logger.warning(f"Failed to prepare trading dates: {e}, using fallback dates")
            # Fallback to simple date calculation
            today = datetime.now()
            self._trading_dates_cache = {
                'today': today.strftime('%Y-%m-%d'),
                'thirty_days_ago': (today - timedelta(days=30)).strftime('%Y-%m-%d'),
                'sixty_days_ago': (today - timedelta(days=60)).strftime('%Y-%m-%d'),
                'six_months_ago': (today - timedelta(days=180)).strftime('%Y-%m-%d'),
                'one_year_ago': (today - timedelta(days=365)).strftime('%Y-%m-%d'),
                'hundred_days_ago': (today - timedelta(days=100)).strftime('%Y-%m-%d'),
                'two_hundred_fifty_days_ago': (today - timedelta(days=250)).strftime('%Y-%m-%d'),
                'ninety_days_future': (today + timedelta(days=90)).strftime('%Y-%m-%d'),
                'one_year_future': (today + timedelta(days=365)).strftime('%Y-%m-%d')
            }
    
    async def _get_last_trading_day(self, today: Optional[datetime] = None) -> str:
        """Get the most recent trading day, accounting for market holidays."""
        if today is None:
            today = datetime.now()
        
        # Look back 10 days to catch any holidays
        ten_days_ago = (today - timedelta(days=10)).strftime('%Y-%m-%d')
        today_str = today.strftime('%Y-%m-%d')
        
        try:
            # Get market holidays if EODHD provider is available
            if (self.provider_manager and 
                ProviderType.EODHD in self.provider_manager.providers):
                
                eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
                
                # Get holidays data
                holidays_data = await eodhd_client._make_request(
                    'GET',
                    '/calendar/holidays',
                    params={
                        'country': 'US',
                        'from': ten_days_ago,
                        'to': today_str,
                        'fmt': 'json'
                    }
                )
                
                # Extract holiday dates
                holiday_dates = set()
                if holidays_data and isinstance(holidays_data, list):
                    for holiday in holidays_data:
                        if isinstance(holiday, dict) and 'date' in holiday:
                            holiday_dates.add(holiday['date'])
                
                self._holiday_dates_cache = holiday_dates
                
        except Exception as e:
            logger.warning(f"Could not fetch holiday data: {e}")
            holiday_dates = set()
        
        # Find last trading day
        current_date = today
        while True:
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends (Saturday=5, Sunday=6) and holidays
            if (current_date.weekday() < 5 and 
                current_date_str not in (self._holiday_dates_cache or set())):
                return current_date_str
                
            current_date = current_date - timedelta(days=1)
            
            # Safety check - don't go back more than 10 days
            if (today - current_date).days > 10:
                break
        
        # Fallback to simple business day logic
        return (today - pd.BDay(1)).strftime('%Y-%m-%d')
    
    async def _collect_live_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect live price data from EODHD."""
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            dates = self._trading_dates_cache
            
            response = await eodhd_client._make_request(
                'GET',
                f"{eodhd_client.base_url}/eod-bulk-last-day/US",
                params={
                    'api_token': eodhd_client.api_key,
                    'symbols': f'{symbol}.US',
                    'filter': 'extended',
                    'fmt': 'json'
                }
            )
            
            if response.success and response.data:
                if isinstance(response.data, list) and len(response.data) > 0:
                    return response.data[0]
            return None
            
        except Exception as e:
            logger.debug(f"Live price collection failed for {symbol}: {e}")
            return None

    async def _collect_minimal_data_for_rating(
        self,
        symbol: str,
        context: WorkflowExecutionContext
    ) -> Optional[Dict[str, Any]]:
        """
        Collect essential data needed for comprehensive local rating.
        Using EODHD all-in-one plan capabilities for technical analysis.
        
        Essential data for local rating:
        1. Historical prices (250 days) - for technical indicators
        2. Options chain - for options quality scoring
        3. Current quote - for price/volume
        4. Technical indicators - RSI, ADX, ATR, Bollinger Bands for enhanced scoring
        """
        logger.debug(f"Collecting comprehensive local rating data: {symbol}")
        
        minimal_data = {
            'symbol': symbol,
            'quote': None,
            'historical_prices': None,
            'options_chain': [],
            'technicals': {},
            'trading_dates': self._trading_dates_cache,
            'data_collection_timestamp': datetime.now().isoformat(),
            'data_sources': []
        }
        
        if not self.provider_manager:
            logger.error("Provider manager not initialized")
            return None
            
        try:
            # Collect essential data for comprehensive local rating
            tasks = []
            
            # 1. Historical Price Data - CRITICAL for technical analysis
            tasks.append(self._collect_historical_data(symbol))
            
            # 2. Options Chain - CRITICAL for options scoring
            tasks.append(self._collect_options_data(symbol))
            
            # 3. Current Quote - for current price/volume
            tasks.append(self._collect_quote_data(symbol))
            
            # 4. Technical Indicators - for enhanced scoring
            tasks.append(self._collect_technical_indicators(symbol))
            
            # Execute tasks with minimal concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            data_keys = ['historical_prices', 'options_chain', 'quote', 'technicals']
            
            for i, result in enumerate(results):
                key = data_keys[i]
                
                if isinstance(result, Exception):
                    logger.warning(f"Failed to collect {key} for {symbol}: {result}")
                    minimal_data[key] = None
                else:
                    minimal_data[key] = result
                    if result is not None:
                        minimal_data['data_sources'].append(key)
                        logger.info(f"✅ Collected {key} for {symbol}")
                    else:
                        logger.warning(f"❌ Got None for {key} collection for {symbol}")
            
            # Check if we have minimum required data
            if minimal_data['historical_prices'] is None:
                logger.warning(f"Missing critical historical data for {symbol}")
                # Continue anyway - will use simplified scoring
            
            return minimal_data
            
        except Exception as e:
            logger.error(f"Failed to collect minimal data for {symbol}: {e}")
            return None
    
    async def _collect_enhanced_data(
        self, 
        symbol: str, 
        context: WorkflowExecutionContext
    ) -> Optional[Dict[str, Any]]:
        """
        Collect all enhanced data for a given symbol following the enhanced example pattern.
        
        Returns comprehensive data structure matching examples/eodhd-enhanced-data.py
        """
        logger.debug(f"Collecting enhanced data for {symbol}")
        
        enhanced_data = {
            'symbol': symbol,
            'quote': None,
            'historical': None,
            'fundamentals': None,
            'technicals': {},
            'economic_events': None,
            'news': None,
            'earnings': None,
            'sentiment': None,
            'options_chain': [],
            'trading_dates': self._trading_dates_cache,
            'data_collection_timestamp': datetime.now().isoformat(),
            'data_sources': []
        }
        
        if not self.provider_manager:
            logger.error("Provider manager not initialized")
            return None
        
        try:
            # Collect data from multiple sources concurrently
            tasks = []
            
            # 1. Market Context (VIX, SPY, market trend) - EODHD
            tasks.append(self._collect_market_context(symbol))
            
            # 2. Recent News - EODHD  
            tasks.append(self._collect_news(symbol))
            
            # 3. Fundamental Data - EODHD
            tasks.append(self._collect_fundamental_data(symbol))
            
            # 4. Live Quote - Both providers with fallback
            tasks.append(self._collect_quote_data(symbol))
            
            # 5. Earnings Data - EODHD
            tasks.append(self._collect_earnings_data(symbol))
            
            # 6. Historical Price Data - EODHD
            tasks.append(self._collect_historical_data(symbol))
            
            # 7. Sentiment Analysis - EODHD
            tasks.append(self._collect_sentiment_data(symbol))
            
            # 8. Technical Indicators - EODHD
            tasks.append(self._collect_technical_indicators(symbol))
            
            # 9. Options Chain with Greeks - MarketData preferred
            tasks.append(self._collect_options_data(symbol))
            
            # 10. Live Price Data - EODHD
            tasks.append(self._collect_live_price(symbol))
            
            # 11. Risk Metrics - EODHD (Sharpe, Sortino, Max Drawdown)
            tasks.append(self._collect_risk_metrics(symbol))
            
            # Execute data collection with rate limiting for EODHD
            # EODHD allows 15 calls/second - we need to stagger the calls
            # Split tasks into EODHD-heavy and other providers
            eodhd_tasks = tasks[0:8] + tasks[9:11]  # All except options (index 8)
            other_tasks = [tasks[8]]  # Options chain (MarketData preferred)
            
            # Execute EODHD tasks with delays to avoid rate limits
            eodhd_results = []
            for task in eodhd_tasks:
                try:
                    result = await task
                    eodhd_results.append(result)
                except Exception as e:
                    eodhd_results.append(e)
                await asyncio.sleep(0.07)  # 70ms delay between EODHD calls
            
            # Execute other tasks concurrently
            other_results = await asyncio.gather(*other_tasks, return_exceptions=True)
            
            # Combine results in original order
            results = eodhd_results[:8] + other_results + eodhd_results[8:]
            
            # Process results - use consistent field names
            data_keys = [
                'market_context', 'news', 'fundamentals', 'quote', 
                'earnings', 'historical_prices', 'sentiment', 'technicals', 'options_chain', 'live_price',
                'risk_metrics'
            ]
            
            for i, result in enumerate(results):
                key = data_keys[i]
                
                if isinstance(result, Exception):
                    logger.warning(f"Failed to collect {key} for {symbol}: {result}")
                    enhanced_data[key] = None
                else:
                    enhanced_data[key] = result
                    if result is not None:
                        enhanced_data['data_sources'].append(key)
                        logger.info(f"✅ Successfully collected {key} for {symbol}")
                    else:
                        logger.warning(f"❌ Got None for {key} collection for {symbol}")
            
            # Data quality check
            collected_sources = len([v for v in enhanced_data.values() 
                                   if v is not None and v != [] and v != {}])
            
            if collected_sources < 3:  # Minimum data threshold
                logger.warning(f"Insufficient data collected for {symbol}: {collected_sources} sources")
                return None
            
            logger.debug(f"Enhanced data collection completed for {symbol}: {collected_sources} data sources")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced data for {symbol}: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return None
    
    async def collect_enhanced_data_for_qualified_stocks(
        self,
        opportunities: List[Dict[str, Any]],
        context: WorkflowExecutionContext
    ) -> List[Dict[str, Any]]:
        """
        Collect enhanced data ONLY for stocks that passed local rating threshold.
        This is called after local ranking, before AI analysis.
        
        Enhanced data includes:
        - Market context (VIX, SPY)
        - Recent news
        - Fundamental data
        - Earnings calendar
        - Sentiment analysis
        """
        logger.info(f"Collecting enhanced data for {len(opportunities)} qualified stocks")
        
        if not self.provider_manager:
            await self._initialize_providers()
            await self._prepare_trading_dates()
        
        enhanced_opportunities = []
        
        try:
            for opp in opportunities:
                symbol = opp.get('symbol')
                if not symbol:
                    enhanced_opportunities.append(opp)
                    continue
                    
                logger.info(f"Enriching data for qualified stock: {symbol}")
                
                # Add 1 second delay between each stock to respect MarketData rate limits
                await asyncio.sleep(1.0)
                
                try:
                    # Collect additional enhanced data
                    tasks = []
                    
                    # 1. Market Context - for AI understanding of market conditions
                    tasks.append(self._collect_market_context(symbol))
                    
                    # 2. Recent News - for AI sentiment and event analysis
                    tasks.append(self._collect_news(symbol))
                    
                    # 3. Fundamental Data - for AI valuation analysis
                    tasks.append(self._collect_fundamental_data(symbol))
                    
                    # 4. Earnings Data - for AI timing analysis
                    tasks.append(self._collect_earnings_data(symbol))
                    
                    # 5. Sentiment Data - for AI market sentiment understanding
                    tasks.append(self._collect_sentiment_data(symbol))
                    
                    # Execute enhancement tasks
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Merge enhanced data with existing opportunity data
                    enhanced_opp = opp.copy()
                    enhanced_data = enhanced_opp.get('enhanced_data', {})
                    
                    # Process results
                    data_keys = ['market_context', 'news', 'fundamentals', 'earnings', 'sentiment']
                    
                    for i, result in enumerate(results):
                        key = data_keys[i]
                        
                        if isinstance(result, Exception):
                            logger.warning(f"Failed to collect {key} for {symbol}: {result}")
                        elif result is not None:
                            enhanced_data[key] = result
                            logger.info(f"✅ Added {key} to {symbol}")
                    
                    enhanced_opp['enhanced_data'] = enhanced_data
                    enhanced_opportunities.append(enhanced_opp)
                    
                    logger.info(f"Enhanced data collection complete for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to enhance data for {symbol}: {e}")
                    # Keep the original opportunity even if enhancement fails
                    enhanced_opportunities.append(opp)
        finally:
            # Clean up provider sessions
            if self.provider_manager:
                try:
                    await self.provider_manager.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Provider cleanup warning: {e}")
        
        return enhanced_opportunities
    
    async def _collect_market_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect market context including VIX, SPY performance, and market trends."""
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            market_context = {}
            
            # 1. Get VIX (volatility index) - fear gauge
            try:
                vix_response = await eodhd_client._make_request(
                    'GET',
                    f'{eodhd_client.base_url}/real-time/VIX.INDX',
                    params={
                        'api_token': eodhd_client.api_key,
                        'fmt': 'json'
                    }
                )
                if vix_response.success and vix_response.data:
                    vix_data = vix_response.data
                    market_context['vix'] = {
                        'level': vix_data.get('close', 0),
                        'change': vix_data.get('change', 0),
                        'change_percent': vix_data.get('change_p', 0),
                        'interpretation': self._interpret_vix_level(vix_data.get('close', 0))
                    }
            except Exception as e:
                logger.debug(f"VIX fetch failed: {e}")
            
            # Add delay to avoid EODHD rate limits
            await asyncio.sleep(0.07)  # 70ms delay
            
            # 2. Get SPY (S&P 500 ETF) for market trend
            try:
                spy_response = await eodhd_client._make_request(
                    'GET',
                    f'{eodhd_client.base_url}/real-time/SPY.US',
                    params={
                        'api_token': eodhd_client.api_key,
                        'fmt': 'json'
                    }
                )
                if spy_response.success and spy_response.data:
                    spy_data = spy_response.data
                    market_context['spy'] = {
                        'price': spy_data.get('close', 0),
                        'change': spy_data.get('change', 0),
                        'change_percent': spy_data.get('change_p', 0),
                        'volume': spy_data.get('volume', 0)
                    }
                    
                    # Determine market trend based on SPY movement
                    change_pct = spy_data.get('change_p', 0)
                    if change_pct > 1:
                        market_context['market_trend'] = 'bullish'
                    elif change_pct < -1:
                        market_context['market_trend'] = 'bearish'
                    else:
                        market_context['market_trend'] = 'neutral'
            except Exception as e:
                logger.debug(f"SPY fetch failed: {e}")
            
            # Add delay to avoid EODHD rate limits
            await asyncio.sleep(0.07)  # 70ms delay
            
            # 3. Get DXY (Dollar Index) for currency strength
            try:
                dxy_response = await eodhd_client._make_request(
                    'GET',
                    f'{eodhd_client.base_url}/real-time/DX-Y.NYB',
                    params={
                        'api_token': eodhd_client.api_key,
                        'fmt': 'json'
                    }
                )
                if dxy_response.success and dxy_response.data:
                    dxy_data = dxy_response.data
                    market_context['dollar_index'] = {
                        'level': dxy_data.get('close', 0),
                        'change': dxy_data.get('change', 0),
                        'change_percent': dxy_data.get('change_p', 0)
                    }
            except Exception as e:
                logger.debug(f"DXY fetch failed: {e}")
            
            # 4. Add market hours status
            from datetime import datetime
            import pytz
            
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            hour = now_et.hour
            weekday = now_et.weekday()
            
            if weekday >= 5:  # Weekend
                market_context['market_status'] = 'closed_weekend'
            elif hour < 9 or (hour == 9 and now_et.minute < 30):
                market_context['market_status'] = 'pre_market'
            elif hour >= 16:
                market_context['market_status'] = 'after_hours'
            else:
                market_context['market_status'] = 'open'
            
            market_context['timestamp'] = datetime.now().isoformat()
            
            return market_context if market_context else None
            
        except Exception as e:
            logger.debug(f"Market context collection failed: {e}")
            return None
    
    def _interpret_vix_level(self, vix_level: float) -> str:
        """Interpret VIX level for market sentiment."""
        if vix_level < 12:
            return "very_low_volatility"
        elif vix_level < 20:
            return "normal_volatility"
        elif vix_level < 30:
            return "elevated_volatility"
        elif vix_level < 40:
            return "high_volatility"
        else:
            return "extreme_volatility"
    
    async def _collect_news(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Collect recent news from EODHD."""
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            dates = self._trading_dates_cache
            
            # Get recent news for the symbol
            response = await eodhd_client._make_request(
                'GET',
                f'{eodhd_client.base_url}/news',
                params={
                    'api_token': eodhd_client.api_key,
                    's': f'{symbol}.US',
                    'from': dates['thirty_days_ago'],
                    'to': dates['today'],
                    'limit': 5,
                    'offset': 0,
                    'fmt': 'json'
                }
            )
            
            if response.success and response.data:
                return response.data if isinstance(response.data, list) else None
            return None
            
        except Exception as e:
            logger.debug(f"News collection failed for {symbol}: {e}")
            return None
    
    async def _collect_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect and filter fundamental data from EODHD."""
        try:
            # Get raw fundamental data directly from EODHD API
            from src.providers.provider_manager import ProviderType
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
                
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            
            response = await eodhd_client._make_request(
                'GET',
                f'{eodhd_client.base_url}/fundamentals/{symbol}.US',
                params={
                    'api_token': eodhd_client.api_key,
                    'fmt': 'json'
                }
            )
            
            if response.success and response.data:
                # Apply filtering logic from the enhanced example
                return self._filter_fundamental_data(response.data)
            
            return None
            
        except Exception as e:
            logger.debug(f"Fundamental data collection failed for {symbol}: {e}")
            return None
    
    async def _collect_quote_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect live quote data with provider fallback."""
        try:
            quote = await self.provider_manager.get_stock_quote(symbol)
            
            if quote:
                # Convert StockQuote model to dict
                if hasattr(quote, 'dict'):
                    quote_dict = quote.dict()
                elif hasattr(quote, '__dict__'):
                    quote_dict = quote.__dict__
                else:
                    quote_dict = quote
                
                # Add a 'close' field that mirrors last_price for compatibility
                if isinstance(quote_dict, dict) and 'last_price' in quote_dict:
                    quote_dict['close'] = quote_dict['last_price']
                
                return quote_dict
            
            return None
            
        except Exception as e:
            logger.debug(f"Quote collection failed for {symbol}: {e}")
            return None
    
    async def _collect_earnings_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Collect earnings calendar data from EODHD."""
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            dates = self._trading_dates_cache
            
            # Get FUTURE earnings data (not past)
            response = await eodhd_client._make_request(
                'GET',
                f'{eodhd_client.base_url}/calendar/earnings',
                params={
                    'api_token': eodhd_client.api_key,
                    'symbols': symbol,
                    'from': dates['today'],
                    'to': dates.get('ninety_days_future', dates['today']),
                    'fmt': 'json'
                }
            )
            
            if response.success and response.data:
                # Earnings endpoint returns {"earnings": [...]}
                if isinstance(response.data, dict) and 'earnings' in response.data:
                    return response.data['earnings']
                elif isinstance(response.data, list):
                    return response.data
            return None
            
        except Exception as e:
            logger.debug(f"Earnings collection failed for {symbol}: {e}")
            return None
    
    async def _collect_historical_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Collect historical price data from EODHD."""
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            dates = self._trading_dates_cache
            
            # Build proper API URL for historical data
            url = f"https://eodhd.com/api/eod/{symbol}.US"
            params = {
                'api_token': eodhd_client.api_key,
                'period': 'd',
                'from': dates['two_hundred_fifty_days_ago'],
                'to': dates['today'],
                'order': 'a',  # Ascending order (oldest to newest)
                'fmt': 'json'
            }
            
            # Make direct HTTP request using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            return data
                    else:
                        logger.warning(f"EODHD API returned status {response.status} for {symbol}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Historical data collection failed for {symbol}: {e}")
            return None
    
    async def _collect_sentiment_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect sentiment analysis data from EODHD."""
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            dates = self._trading_dates_cache
            
            # Get sentiment data
            response = await eodhd_client._make_request(
                'GET',
                f'{eodhd_client.base_url}/sentiments',
                params={
                    'api_token': eodhd_client.api_key,
                    's': f'{symbol}.US',
                    'from': dates['thirty_days_ago'],
                    'to': dates['today'],
                    'fmt': 'json'
                }
            )
            
            if response.success and response.data:
                # Sentiment returns {symbol: [{date, count, normalized}, ...]}
                if isinstance(response.data, dict):
                    # Get the sentiment data for this symbol
                    for key in response.data:
                        if key.upper().startswith(symbol.upper()):
                            return response.data[key]
                return response.data if isinstance(response.data, dict) else None
            return None
            
        except Exception as e:
            logger.debug(f"Sentiment collection failed for {symbol}: {e}")
            return None
    
    async def _collect_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Collect technical indicators from EODHD following the enhanced example."""
        technical_indicators = {}
        
        try:
            if ProviderType.EODHD not in self.provider_manager.providers:
                return technical_indicators
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            dates = self._trading_dates_cache
            
            # List of indicators - expanded to include all required for completeness
            indicators = [
                {'name': 'rsi', 'function': 'rsi', 'period': 14, 'days_back': 60},
                {'name': 'volatility', 'function': 'volatility', 'period': 30, 'days_back': 60},
                {'name': 'atr', 'function': 'atr', 'period': 14, 'days_back': 60},
                {'name': 'macd', 'function': 'macd', 'period': 12, 'days_back': 60},  # MACD with standard period
                {'name': 'bbands', 'function': 'bbands', 'period': 20, 'days_back': 60},  # Bollinger Bands
                {'name': 'sma', 'function': 'sma', 'period': 20, 'days_back': 60},  # 20-day SMA
                {'name': 'sma', 'function': 'sma', 'period': 50, 'days_back': 90}   # 50-day SMA
            ]
            
            for indicator in indicators:
                try:
                    days_back = (datetime.now() - timedelta(days=indicator['days_back'])).strftime('%Y-%m-%d')
                    
                    response = await eodhd_client._make_request(
                        'GET',
                        f'{eodhd_client.base_url}/technical',
                        params={
                            'api_token': eodhd_client.api_key,
                            'ticker': f'{symbol}.US',
                            'function': indicator['function'],
                            'period': indicator['period'],
                            'from': days_back,
                            'to': dates['today'],
                            'order': 'd',
                            'splitadjusted_only': '0',
                            'fmt': 'json'
                        }
                    )
                    
                    if response.success and response.data:
                        # Special handling for SMA with different periods
                        if indicator['name'] == 'sma':
                            key_name = f"sma_{indicator['period']}"
                            technical_indicators[key_name] = response.data
                        else:
                            technical_indicators[indicator['name']] = response.data
                    else:
                        if indicator['name'] == 'sma':
                            key_name = f"sma_{indicator['period']}"
                            technical_indicators[key_name] = None
                        else:
                            technical_indicators[indicator['name']] = None
                    
                    logger.debug(f"Collected {indicator['name']} (period {indicator['period']}) for {symbol}")
                    
                except Exception as e:
                    logger.debug(f"Failed to fetch {indicator['name']} for {symbol}: {e}")
                    technical_indicators[indicator['name']] = None
            
            return technical_indicators
            
        except Exception as e:
            logger.debug(f"Technical indicators collection failed for {symbol}: {e}")
            return technical_indicators
    
    async def _collect_options_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect comprehensive options chain with Greeks from MarketData API."""
        try:
            # Get all options in a SINGLE API call for efficiency
            # This uses cached feed (status 203) = 1 credit per request
            # Instead of live feed = 1 credit per contract!
            from datetime import datetime, timedelta
            
            # Get options expiring in next 90 days to cover our selection window (45-75 days)
            # Single call instead of 2 separate PMCC calls
            filters = {
                'from_date': (datetime.now().date() + timedelta(days=40)).isoformat(),  # Start from 40 days out
                'to_date': (datetime.now().date() + timedelta(days=90)).isoformat(),    # Up to 90 days out
                'min_open_interest': 50,  # Good liquidity
                'option_type': 'call'  # Focus on calls for PMCC
            }
            
            options_chain = await self.provider_manager.get_options_chain(symbol, **filters)
            
            if not options_chain:
                return []
            
            # Convert to dict format
            all_contracts = []
            for contract in options_chain:
                contract_dict = contract.dict() if hasattr(contract, 'dict') else contract
                
                # Classify contract type based on DTE
                if hasattr(contract, 'days_to_expiration'):
                    dte = contract.days_to_expiration
                elif 'days_to_expiration' in contract_dict:
                    dte = contract_dict['days_to_expiration']
                else:
                    dte = 30  # Default
                
                # Mark LEAPS (>180 days) vs short calls (<45 days)
                if dte > 180:
                    contract_dict['contract_type'] = 'LEAP'
                else:
                    contract_dict['contract_type'] = 'SHORT_CALL'
                    
                all_contracts.append(contract_dict)
            
            logger.debug(f"Collected {len(all_contracts)} option contracts for {symbol} in 1 API call")
            return all_contracts
            
        except Exception as e:
            logger.debug(f"Options data collection failed for {symbol}: {e}")
            return []
    
    async def _collect_risk_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Calculate risk metrics for the stock based on historical data.
        Includes Sharpe ratio, Sortino ratio, max drawdown, etc.
        """
        try:
            from src.providers.provider_manager import ProviderType
            
            if ProviderType.EODHD not in self.provider_manager.providers:
                return None
            
            eodhd_client = self.provider_manager.providers[ProviderType.EODHD]
            
            # Get 1 year of historical data for calculations
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            response = await eodhd_client._make_request(
                'GET',
                f'{eodhd_client.base_url}/eod/{symbol}.US',
                params={
                    'api_token': eodhd_client.api_key,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'fmt': 'json'
                }
            )
            
            if not response.success or not response.data:
                return None
            
            prices = response.data
            if len(prices) < 30:  # Need at least 30 data points
                return None
            
            # Calculate returns
            closes = [float(p['adjusted_close']) for p in prices]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            
            if not returns:
                return None
            
            # Risk-free rate (approximate US Treasury 1-year)
            risk_free_rate = 0.045 / 252  # Daily risk-free rate
            
            # Calculate metrics
            import numpy as np
            returns_array = np.array(returns)
            
            # Sharpe Ratio (annualized)
            excess_returns = returns_array - risk_free_rate
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            # Sortino Ratio (annualized) - uses downside deviation
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.001
            sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Volatility (annualized)
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # Beta (vs SPY as market proxy) - simplified calculation
            # In production, you'd fetch SPY data and calculate covariance
            beta = 1.0  # Default to market beta
            
            risk_metrics = {
                'sharpe_ratio': round(float(sharpe_ratio), 3),
                'sortino_ratio': round(float(sortino_ratio), 3),
                'max_drawdown': round(float(max_drawdown), 3),
                'volatility_annual': round(float(volatility), 3),
                'downside_deviation': round(float(downside_deviation * np.sqrt(252)), 3),
                'beta': beta,
                'calculation_period': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                'data_points': len(returns)
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.debug(f"Risk metrics calculation failed for {symbol}: {e}")
            return None
    
    def _filter_fundamental_data(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter fundamental data to essential metrics following the enhanced example.
        
        This method replicates the filtering logic from examples/eodhd-enhanced-data.py
        """
        if not fundamentals:
            return {}
        
        try:
            filtered = {}
            
            # Company Info
            if 'General' in fundamentals:
                general = fundamentals['General']
                filtered['company_info'] = {
                    'name': general.get('Name'),
                    'sector': general.get('Sector'),
                    'industry': general.get('Industry'),
                    'market_cap_mln': general.get('MarketCapitalization', 0) / 1000000 if general.get('MarketCapitalization') else None,
                    'employees': general.get('FullTimeEmployees'),
                    'description': general.get('Description', '')[:300]  # Brief description
                }
            
            # Financial Health
            if 'Highlights' in fundamentals:
                highlights = fundamentals['Highlights']
                filtered['financial_health'] = {
                    # Profitability metrics
                    'eps_ttm': highlights.get('EarningsShare'),
                    'profit_margin': highlights.get('ProfitMargin'),
                    'operating_margin': highlights.get('OperatingMarginTTM'),
                    'roe': highlights.get('ReturnOnEquityTTM'),
                    'roa': highlights.get('ReturnOnAssetsTTM'),
                    
                    # Growth metrics
                    'revenue_growth_yoy': highlights.get('QuarterlyRevenueGrowthYOY'),
                    'earnings_growth_yoy': highlights.get('QuarterlyEarningsGrowthYOY'),
                    'eps_estimate_current_year': highlights.get('EPSEstimateCurrentYear'),
                    'eps_estimate_next_year': highlights.get('EPSEstimateNextYear'),
                    
                    # Dividend information (critical for PMCC)
                    'dividend_yield': highlights.get('DividendYield'),
                    'dividend_per_share': highlights.get('DividendShare'),
                    
                    # Revenue and earnings
                    'revenue_ttm': highlights.get('RevenueTTM'),
                    'revenue_per_share': highlights.get('RevenuePerShareTTM'),
                    'most_recent_quarter': highlights.get('MostRecentQuarter')
                }
            
            # Valuation Metrics
            if 'Valuation' in fundamentals:
                valuation = fundamentals['Valuation']
                filtered['valuation_metrics'] = {
                    'pe_ratio': valuation.get('TrailingPE'),
                    'forward_pe': valuation.get('ForwardPE'),
                    'price_to_sales': valuation.get('PriceSalesTTM'),
                    'price_to_book': valuation.get('PriceBookMRQ'),
                    'enterprise_value': valuation.get('EnterpriseValue'),
                    'ev_to_revenue': valuation.get('EnterpriseValueRevenue'),
                    'ev_to_ebitda': valuation.get('EnterpriseValueEbitda')
                }
            
            # Stock Technicals
            if 'Technicals' in fundamentals:
                technicals = fundamentals['Technicals']
                filtered['stock_technicals'] = {
                    'beta': technicals.get('Beta'),
                    '52_week_high': technicals.get('52WeekHigh'),
                    '52_week_low': technicals.get('52WeekLow'),
                    '50_day_ma': technicals.get('50DayMA'),
                    '200_day_ma': technicals.get('200DayMA'),
                    'short_interest': technicals.get('ShortPercent'),
                    'short_ratio': technicals.get('ShortRatio')
                }
            
            # Dividend Information
            if 'SplitsDividends' in fundamentals:
                dividends = fundamentals['SplitsDividends']
                filtered['dividend_info'] = {
                    'forward_dividend_rate': dividends.get('ForwardAnnualDividendRate'),
                    'forward_dividend_yield': dividends.get('ForwardAnnualDividendYield'),
                    'payout_ratio': dividends.get('PayoutRatio'),
                    'dividend_date': dividends.get('DividendDate'),
                    'ex_dividend_date': dividends.get('ExDividendDate'),
                    'last_split_date': dividends.get('LastSplitDate'),
                    'last_split_factor': dividends.get('LastSplitFactor')
                }
            
            # Analyst Sentiment
            if 'AnalystRatings' in fundamentals:
                ratings = fundamentals['AnalystRatings']
                filtered['analyst_sentiment'] = {
                    'avg_rating': ratings.get('Rating'),  # 1=Strong Buy, 5=Strong Sell
                    'target_price': ratings.get('TargetPrice'),
                    'strong_buy': ratings.get('StrongBuy'),
                    'buy': ratings.get('Buy'),
                    'hold': ratings.get('Hold'),
                    'sell': ratings.get('Sell'),
                    'strong_sell': ratings.get('StrongSell')
                }
            
            # Ownership Structure
            if 'SharesStats' in fundamentals:
                shares = fundamentals['SharesStats']
                filtered['ownership_structure'] = {
                    'shares_outstanding': shares.get('SharesOutstanding'),
                    'percent_institutions': shares.get('PercentInstitutions'),
                    'percent_insiders': shares.get('PercentInsiders'),
                    'shares_float': shares.get('SharesFloat')
                }
            
            # Financial Statements (most recent quarter)
            if 'Financials' in fundamentals:
                filtered.update(self._extract_financial_statements(fundamentals['Financials']))
            
            return filtered
            
        except Exception as e:
            logger.warning(f"Error filtering fundamental data: {e}")
            return fundamentals  # Return original if filtering fails
    
    def _extract_financial_statements(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial statement data from the most recent quarter."""
        extracted = {}
        
        try:
            # Balance Sheet - Financial Strength Indicators
            if 'Balance_Sheet' in financials and 'quarterly' in financials['Balance_Sheet']:
                bs_data = financials['Balance_Sheet']['quarterly']
                latest_quarter = max(bs_data.keys()) if bs_data else None
                
                if latest_quarter:
                    bs = bs_data[latest_quarter]
                    extracted['balance_sheet'] = {
                        'total_assets': float(bs.get('totalAssets', 0)) / 1000000 if bs.get('totalAssets') else None,
                        'total_debt': float(bs.get('shortLongTermDebtTotal', 0)) / 1000000 if bs.get('shortLongTermDebtTotal') else None,
                        'cash_and_equivalents': float(bs.get('cashAndEquivalents', 0)) / 1000000 if bs.get('cashAndEquivalents') else None,
                        'net_debt': float(bs.get('netDebt', 0)) / 1000000 if bs.get('netDebt') else None,
                        'working_capital': float(bs.get('netWorkingCapital', 0)) / 1000000 if bs.get('netWorkingCapital') else None,
                        'shareholders_equity': float(bs.get('totalStockholderEquity', 0)) / 1000000 if bs.get('totalStockholderEquity') else None,
                        'quarter_date': latest_quarter
                    }
                    
                    # Calculate debt-to-equity ratio
                    if (extracted['balance_sheet']['total_debt'] and 
                        extracted['balance_sheet']['shareholders_equity']):
                        extracted['balance_sheet']['debt_to_equity'] = round(
                            extracted['balance_sheet']['total_debt'] / 
                            extracted['balance_sheet']['shareholders_equity'], 2
                        )
            
            # Income Statement - Profitability and Revenue Trends
            if 'Income_Statement' in financials and 'quarterly' in financials['Income_Statement']:
                is_data = financials['Income_Statement']['quarterly']
                latest_quarter = max(is_data.keys()) if is_data else None
                
                if latest_quarter:
                    is_ = is_data[latest_quarter]
                    extracted['income_statement'] = {
                        'total_revenue': float(is_.get('totalRevenue', 0)) / 1000000 if is_.get('totalRevenue') else None,
                        'gross_profit': float(is_.get('grossProfit', 0)) / 1000000 if is_.get('grossProfit') else None,
                        'operating_income': float(is_.get('operatingIncome', 0)) / 1000000 if is_.get('operatingIncome') else None,
                        'net_income': float(is_.get('netIncome', 0)) / 1000000 if is_.get('netIncome') else None,
                        'ebitda': float(is_.get('ebitda', 0)) / 1000000 if is_.get('ebitda') else None,
                        'quarter_date': latest_quarter
                    }
                    
                    # Calculate margins
                    if (extracted['income_statement']['total_revenue'] and 
                        extracted['income_statement']['total_revenue'] > 0):
                        revenue = extracted['income_statement']['total_revenue']
                        
                        if extracted['income_statement']['gross_profit']:
                            extracted['income_statement']['gross_margin'] = round(
                                (extracted['income_statement']['gross_profit'] / revenue) * 100, 2
                            )
                        
                        if extracted['income_statement']['operating_income']:
                            extracted['income_statement']['operating_margin'] = round(
                                (extracted['income_statement']['operating_income'] / revenue) * 100, 2
                            )
                        
                        if extracted['income_statement']['net_income']:
                            extracted['income_statement']['net_margin'] = round(
                                (extracted['income_statement']['net_income'] / revenue) * 100, 2
                            )
            
            # Cash Flow - Critical for PMCC (company sustainability)
            if 'Cash_Flow' in financials and 'quarterly' in financials['Cash_Flow']:
                cf_data = financials['Cash_Flow']['quarterly']
                latest_quarter = max(cf_data.keys()) if cf_data else None
                
                if latest_quarter:
                    cf = cf_data[latest_quarter]
                    extracted['cash_flow'] = {
                        'operating_cash_flow': float(cf.get('totalCashFromOperatingActivities', 0)) / 1000000 if cf.get('totalCashFromOperatingActivities') else None,
                        'free_cash_flow': float(cf.get('freeCashFlow', 0)) / 1000000 if cf.get('freeCashFlow') else None,
                        'capex': float(cf.get('capitalExpenditures', 0)) / 1000000 if cf.get('capitalExpenditures') else None,
                        'net_income': float(cf.get('netIncome', 0)) / 1000000 if cf.get('netIncome') else None,
                        'cash_change': float(cf.get('changeInCash', 0)) / 1000000 if cf.get('changeInCash') else None,
                        'dividends_paid': float(cf.get('dividendsPaid', 0)) / 1000000 if cf.get('dividendsPaid') else None,
                        'quarter_date': latest_quarter
                    }
            
        except Exception as e:
            logger.warning(f"Error extracting financial statements: {e}")
        
        return extracted
    
    async def validate_input(self, input_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate input data for the data enrichment step."""
        if not isinstance(input_data, list):
            raise ValueError("Input data must be a list of symbols")
        
        if not input_data:
            raise ValueError("Symbol list cannot be empty")
        
        # Validate symbol format
        for symbol in input_data:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"Invalid symbol format: {symbol}")
            
            # Basic symbol validation (alphanumeric with possible dots/dashes)
            if not all(c.isalnum() or c in '.-' for c in symbol):
                raise ValueError(f"Invalid symbol characters: {symbol}")
    
    async def validate_output(self, output_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate output data from the data enrichment step."""
        if not isinstance(output_data, dict):
            raise ValueError("Output data must be a dictionary")
        
        required_keys = ['enriched_data', 'symbols_processed', 'symbols_failed', 'total_symbols']
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")
        
        # Validate enriched data structure
        enriched_data = output_data['enriched_data']
        if not isinstance(enriched_data, dict):
            raise ValueError("Enriched data must be a dictionary")
        
        # Validate that we have some successful results
        if output_data['symbols_processed'] == 0 and output_data['total_symbols'] > 0:
            logger.warning("No symbols were successfully processed")
    
    def get_records_processed(self, output_data: Any) -> Optional[int]:
        """Get number of records processed by this step."""
        if isinstance(output_data, dict):
            return output_data.get('symbols_processed', 0)
        return 0
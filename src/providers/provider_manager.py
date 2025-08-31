"""
Provider manager for orchestrating multiple data providers with intelligent failover.

This module provides a comprehensive provider management system that:
- Orchestrates multiple providers (EODHD, MarketData, Claude)
- Implements intelligent failover and load balancing
- Provides unified data access interfaces
- Handles provider-specific optimizations
- Manages provider health and circuit breaking
- Implements data validation and quality assurance
- Provides comprehensive monitoring and metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import time

from src.providers.base_provider import BaseProvider, MarketDataProvider, AIProvider, ProviderFactory
from src.providers.eodhd_client import EODHDClient
from src.providers.marketdata_client import MarketDataClient
from src.providers.cache import CacheManager, ProviderCacheWrapper, get_cache_manager
from src.providers.validators import DataValidator, ValidationResult, get_validator
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
    AIAnalysisResult,
    ScreeningResult
)
from src.providers.exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    CircuitBreakerOpenError,
    QuotaExceededError,
    InsufficientDataError
)

logger = logging.getLogger(__name__)


class ProviderStrategy(str, Enum):
    """Provider selection strategies."""
    PRIMARY_ONLY = "primary_only"
    FAILOVER = "failover"
    LOAD_BALANCE = "load_balance"
    BEST_QUALITY = "best_quality"
    FASTEST = "fastest"
    COST_OPTIMIZED = "cost_optimized"


class DataSource(str, Enum):
    """Data source preferences."""
    EODHD_ONLY = "eodhd_only"
    MARKETDATA_ONLY = "marketdata_only"
    EODHD_PRIMARY = "eodhd_primary"
    MARKETDATA_PRIMARY = "marketdata_primary"
    BEST_AVAILABLE = "best_available"


@dataclass
class ProviderConfig:
    """Provider configuration with capabilities."""
    provider_type: ProviderType
    config: Dict[str, Any]
    priority: int = 1
    capabilities: List[str] = field(default_factory=list)
    max_requests_per_minute: int = 60
    cost_per_request: float = 0.0
    enabled: bool = True


@dataclass
class ProviderMetrics:
    """Provider performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    total_cost: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate


class ProviderManager:
    """
    Advanced provider manager with intelligent failover and optimization.
    
    Key Features:
    - Multi-provider orchestration with failover
    - Provider health monitoring and circuit breaking
    - Intelligent data source selection
    - Cost optimization and rate limiting
    - Data validation and quality assurance
    - Comprehensive metrics and monitoring
    - Caching integration
    - Provider-specific optimizations
    """
    
    def __init__(
        self,
        provider_configs: List[ProviderConfig],
        strategy: ProviderStrategy = ProviderStrategy.FAILOVER,
        data_source_preference: DataSource = DataSource.EODHD_PRIMARY,
        enable_caching: bool = True,
        enable_validation: bool = True,
        min_quality_score: float = 80.0,
        max_retry_attempts: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 300
    ):
        self.strategy = strategy
        self.data_source_preference = data_source_preference
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        self.min_quality_score = min_quality_score
        self.max_retry_attempts = max_retry_attempts
        
        # Initialize providers
        self.providers: Dict[ProviderType, BaseProvider] = {}
        self.provider_configs: Dict[ProviderType, ProviderConfig] = {}
        self.provider_metrics: Dict[ProviderType, ProviderMetrics] = {}
        
        # Initialize cache and validator
        self.cache_manager = get_cache_manager() if enable_caching else None
        self.validator = get_validator() if enable_validation else None
        
        # Provider capabilities mapping
        self.capabilities = {
            'stock_screening': [ProviderType.EODHD],  # Only EODHD has native screening
            'stock_quotes': [ProviderType.EODHD, ProviderType.MARKETDATA],
            'options_chains': [ProviderType.EODHD, ProviderType.MARKETDATA],
            'fundamental_data': [ProviderType.EODHD],
            'technical_indicators': [ProviderType.EODHD],
            'earnings_calendar': [ProviderType.EODHD],
            'news': [ProviderType.EODHD],
            'ai_analysis': [ProviderType.CLAUDE]
        }
        
        # Initialize providers from configs
        for provider_config in provider_configs:
            self._initialize_provider(provider_config)
        
        logger.info(f"Provider manager initialized with {len(self.providers)} providers, strategy: {strategy}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        for provider in self.providers.values():
            await provider.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        for provider in self.providers.values():
            await provider.cleanup()
    
    async def screen_stocks(self, criteria: ScreeningCriteria) -> List[str]:
        """
        Screen stocks with intelligent provider selection.
        Primarily uses EODHD due to its native screening capability.
        """
        # For screening, EODHD is the clear choice due to native API
        if ProviderType.EODHD in self.providers:
            provider = self.providers[ProviderType.EODHD]
            
            try:
                start_time = time.time()
                
                # Try with caching if enabled
                if self.cache_manager:
                    cache_wrapper = ProviderCacheWrapper(provider, self.cache_manager)
                    # For screening, we need a custom cache key based on criteria
                    cache_key = self._generate_screening_cache_key(criteria)
                    
                    cached_result = await self.cache_manager.get(
                        cache_key,
                        provider=ProviderType.EODHD,
                        data_type='screening'
                    )
                    
                    if cached_result is not None:
                        logger.info(f"Retrieved {len(cached_result)} symbols from cache")
                        return cached_result
                
                # Fetch fresh data
                symbols = await provider.screen_stocks(criteria)
                
                # Cache the result if successful
                if self.cache_manager and symbols:
                    await self.cache_manager.set(
                        cache_key,
                        symbols,
                        provider=ProviderType.EODHD,
                        data_type='screening',
                        tags=['screening', 'stock_symbols']
                    )
                
                # Update metrics
                await self._update_provider_metrics(
                    ProviderType.EODHD,
                    True,
                    time.time() - start_time,
                    0.05  # Estimated cost for screening
                )
                
                logger.info(f"Successfully screened {len(symbols)} stocks using EODHD")
                return symbols
                
            except Exception as e:
                await self._update_provider_metrics(ProviderType.EODHD, False, time.time() - start_time, 0)
                logger.debug(f"EODHD screening failed: {e}")
                
                # Fallback to MarketData with predefined universe (limited capability)
                if ProviderType.MARKETDATA in self.providers:
                    logger.warning("Falling back to MarketData with limited screening capability")
                    provider = self.providers[ProviderType.MARKETDATA]
                    return await provider.screen_stocks(criteria)
                
                raise ProviderError(f"Stock screening failed: {str(e)}")
        
        raise InsufficientDataError("No providers available for stock screening")
    
    async def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get stock quote with provider selection based on strategy."""
        providers = await self._select_providers_for_capability('stock_quotes')
        
        for provider_type in providers:
            try:
                provider = self.providers[provider_type]
                start_time = time.time()
                
                # Try with caching if enabled
                if self.cache_manager:
                    cache_wrapper = ProviderCacheWrapper(provider, self.cache_manager)
                    quote = await cache_wrapper.get_stock_quote(symbol)
                else:
                    quote = await provider.get_stock_quote(symbol)
                
                if quote is None:
                    continue
                
                # Validate data if enabled
                if self.enable_validation and self.validator:
                    validation_result = self.validator.validate_stock_quote(quote, symbol)
                    
                    if not validation_result.is_valid or validation_result.quality_score < self.min_quality_score:
                        logger.warning(f"Quote validation failed for {symbol} from {provider_type}: score={validation_result.quality_score}")
                        continue
                
                # Update metrics
                await self._update_provider_metrics(
                    provider_type,
                    True,
                    time.time() - start_time,
                    0.01  # Estimated cost per quote
                )
                
                logger.debug(f"Successfully retrieved quote for {symbol} from {provider_type}")
                return quote
                
            except Exception as e:
                await self._update_provider_metrics(provider_type, False, time.time() - start_time, 0)
                logger.warning(f"Failed to get quote for {symbol} from {provider_type}: {e}")
                continue
        
        logger.error(f"Failed to get quote for {symbol} from all providers")
        return None
    
    async def get_stock_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get multiple stock quotes efficiently."""
        providers = await self._select_providers_for_capability('stock_quotes')
        
        for provider_type in providers:
            try:
                provider = self.providers[provider_type]
                start_time = time.time()
                
                quotes = await provider.get_stock_quotes(symbols)
                
                if not quotes:
                    continue
                
                # Validate quotes if enabled
                if self.enable_validation and self.validator:
                    validated_quotes = {}
                    for symbol, quote in quotes.items():
                        validation_result = self.validator.validate_stock_quote(quote, symbol)
                        
                        if validation_result.is_valid and validation_result.quality_score >= self.min_quality_score:
                            validated_quotes[symbol] = quote
                        else:
                            logger.debug(f"Quote validation failed for {symbol}: score={validation_result.quality_score}")
                    
                    quotes = validated_quotes
                
                # Update metrics
                await self._update_provider_metrics(
                    provider_type,
                    True,
                    time.time() - start_time,
                    len(quotes) * 0.01  # Cost per quote
                )
                
                logger.info(f"Successfully retrieved {len(quotes)} quotes from {provider_type}")
                return quotes
                
            except Exception as e:
                await self._update_provider_metrics(provider_type, False, time.time() - start_time, 0)
                logger.warning(f"Failed to get bulk quotes from {provider_type}: {e}")
                continue
        
        logger.error(f"Failed to get bulk quotes from all providers")
        return {}
    
    async def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        **filters
    ) -> List[OptionContract]:
        """Get options chain with intelligent provider selection."""
        # ALWAYS use MarketData only for options data
        providers = [ProviderType.MARKETDATA] if ProviderType.MARKETDATA in self.providers else []
        
        if not providers:
            logger.error(f"MarketData provider not available for options chain")
            return []
        
        for provider_type in providers:
            try:
                provider = self.providers[provider_type]
                start_time = time.time()
                
                # Try with caching if enabled
                if self.cache_manager:
                    cache_wrapper = ProviderCacheWrapper(provider, self.cache_manager)
                    contracts = await cache_wrapper.get_options_chain(symbol, expiration_date=expiration_date, **filters)
                else:
                    contracts = await provider.get_options_chain(symbol, expiration_date=expiration_date, **filters)
                
                if not contracts:
                    continue
                
                # Validate contracts if enabled
                if self.enable_validation and self.validator:
                    validation_result = self.validator.validate_options_chain(contracts, symbol)
                    
                    if not validation_result.is_valid or validation_result.quality_score < self.min_quality_score:
                        logger.warning(f"Options chain validation failed for {symbol} from {provider_type}: score={validation_result.quality_score}")
                        continue
                
                # Update metrics
                cost = 0.01  # MarketData cost per options request
                await self._update_provider_metrics(
                    provider_type,
                    True,
                    time.time() - start_time,
                    cost
                )
                
                logger.info(f"Successfully retrieved {len(contracts)} option contracts for {symbol} from {provider_type}")
                return contracts
                
            except Exception as e:
                await self._update_provider_metrics(provider_type, False, time.time() - start_time, 0)
                logger.warning(f"Failed to get options chain for {symbol} from {provider_type}: {e}")
                continue
        
        logger.error(f"Failed to get options chain for {symbol} from all providers")
        return []
    
    async def get_pmcc_option_chains(self, symbol: str) -> Dict[str, List[OptionContract]]:
        """Get PMCC-optimized options chains."""
        # For PMCC strategies, prefer MarketData for better filtering
        if ProviderType.MARKETDATA in self.providers:
            try:
                provider = self.providers[ProviderType.MARKETDATA]
                
                if hasattr(provider, 'get_pmcc_option_chains'):
                    start_time = time.time()
                    chains = await provider.get_pmcc_option_chains(symbol)
                    
                    if chains and (chains.get('leaps') or chains.get('short_calls')):
                        await self._update_provider_metrics(
                            ProviderType.MARKETDATA,
                            True,
                            time.time() - start_time,
                            0.02  # 2 requests
                        )
                        
                        logger.info(f"Retrieved PMCC chains for {symbol}: {len(chains.get('leaps', []))} LEAPS, {len(chains.get('short_calls', []))} shorts")
                        return chains
                
            except Exception as e:
                logger.warning(f"MarketData PMCC chains failed for {symbol}: {e}")
        
        # Fallback to manual PMCC filtering with regular options chain
        try:
            all_contracts = await self.get_options_chain(
                symbol,
                option_type='call',
                min_open_interest=5
            )
            
            if not all_contracts:
                return {'leaps': [], 'short_calls': []}
            
            # Manual PMCC filtering
            today = date.today()
            leaps_contracts = []
            short_contracts = []
            
            for contract in all_contracts:
                if contract.expiration:
                    dte = (contract.expiration - today).days
                    
                    # LEAPS: 180-365 days, high delta
                    if 180 <= dte <= 365 and contract.delta and contract.delta >= 0.7:
                        leaps_contracts.append(contract)
                    
                    # Short calls: 21-45 days, medium delta
                    elif 21 <= dte <= 45 and contract.delta and 0.15 <= contract.delta <= 0.4:
                        short_contracts.append(contract)
            
            logger.info(f"Manual PMCC filtering for {symbol}: {len(leaps_contracts)} LEAPS, {len(short_contracts)} shorts")
            return {'leaps': leaps_contracts, 'short_calls': short_contracts}
            
        except Exception as e:
            logger.error(f"Failed to get PMCC chains for {symbol}: {e}")
            return {'leaps': [], 'short_calls': []}
    
    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data (primarily from EODHD)."""
        # Fundamental data is primarily available from EODHD
        if ProviderType.EODHD not in self.providers:
            logger.warning("No provider available for fundamental data")
            return None
        
        try:
            provider = self.providers[ProviderType.EODHD]
            start_time = time.time()
            
            # Try with caching if enabled
            if self.cache_manager:
                cache_wrapper = ProviderCacheWrapper(provider, self.cache_manager)
                data = await cache_wrapper.get_fundamental_data(symbol)
            else:
                data = await provider.get_fundamental_data(symbol)
            
            if data is None:
                return None
            
            # Validate data if enabled
            if self.enable_validation and self.validator:
                validation_result = self.validator.validate_fundamental_data(data, symbol)
                
                if not validation_result.is_valid or validation_result.quality_score < self.min_quality_score:
                    logger.warning(f"Fundamental data validation failed for {symbol}: score={validation_result.quality_score}")
                    return None
            
            # Update metrics
            await self._update_provider_metrics(
                ProviderType.EODHD,
                True,
                time.time() - start_time,
                0.02
            )
            
            logger.debug(f"Successfully retrieved fundamental data for {symbol}")
            return data
            
        except Exception as e:
            await self._update_provider_metrics(ProviderType.EODHD, False, time.time() - start_time, 0)
            logger.error(f"Failed to get fundamental data for {symbol}: {e}")
            return None
    
    async def get_technical_indicators(self, symbol: str, period: int = 14) -> Optional[TechnicalIndicators]:
        """Get technical indicators (primarily from EODHD)."""
        if ProviderType.EODHD not in self.providers:
            logger.warning("No provider available for technical indicators")
            return None
        
        try:
            provider = self.providers[ProviderType.EODHD]
            
            if not hasattr(provider, 'get_technical_indicators'):
                return None
            
            start_time = time.time()
            indicators = await provider.get_technical_indicators(symbol, period)
            
            if indicators is None:
                return None
            
            # Validate indicators if enabled
            if self.enable_validation and self.validator:
                validation_result = self.validator.validate_technical_indicators(indicators, symbol)
                
                if not validation_result.is_valid or validation_result.quality_score < self.min_quality_score:
                    logger.warning(f"Technical indicators validation failed for {symbol}: score={validation_result.quality_score}")
                    return None
            
            # Update metrics
            await self._update_provider_metrics(
                ProviderType.EODHD,
                True,
                time.time() - start_time,
                0.05
            )
            
            logger.debug(f"Successfully retrieved technical indicators for {symbol}")
            return indicators
            
        except Exception as e:
            await self._update_provider_metrics(ProviderType.EODHD, False, time.time() - start_time, 0)
            logger.error(f"Failed to get technical indicators for {symbol}: {e}")
            return None
    
    async def get_provider_health_status(self) -> Dict[ProviderType, HealthCheckResult]:
        """Get health status for all providers."""
        health_results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                health_result = await provider.get_health_status()
                health_results[provider_type] = health_result
            except Exception as e:
                logger.error(f"Health check failed for {provider_type}: {e}")
                health_results[provider_type] = HealthCheckResult(
                    provider=provider_type,
                    status=ProviderStatus.UNHEALTHY,
                    message=str(e)
                )
        
        return health_results
    
    async def get_provider_metrics(self) -> Dict[ProviderType, ProviderMetrics]:
        """Get performance metrics for all providers."""
        return self.provider_metrics.copy()
    
    async def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        total_requests = sum(metrics.total_requests for metrics in self.provider_metrics.values())
        total_cost = sum(metrics.total_cost for metrics in self.provider_metrics.values())
        overall_success_rate = 0.0
        
        if total_requests > 0:
            total_successful = sum(metrics.successful_requests for metrics in self.provider_metrics.values())
            overall_success_rate = total_successful / total_requests
        
        # Cache statistics
        cache_stats = None
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
        
        return {
            'total_requests': total_requests,
            'total_cost': total_cost,
            'overall_success_rate': overall_success_rate,
            'provider_metrics': {pt.value: metrics for pt, metrics in self.provider_metrics.items()},
            'cache_stats': {
                'hit_rate': cache_stats.hit_rate if cache_stats else 0.0,
                'total_size_mb': (cache_stats.total_size_bytes / 1024 / 1024) if cache_stats else 0.0,
                'entries_count': cache_stats.entries_count if cache_stats else 0
            } if cache_stats else None,
            'active_providers': list(self.providers.keys())
        }
    
    def _initialize_provider(self, config: ProviderConfig) -> None:
        """Initialize a single provider from configuration."""
        try:
            if config.provider_type == ProviderType.EODHD:
                provider = EODHDClient(config.config)
            elif config.provider_type == ProviderType.MARKETDATA:
                provider = MarketDataClient(config.config)
            else:
                logger.warning(f"Unknown provider type: {config.provider_type}")
                return
            
            self.providers[config.provider_type] = provider
            self.provider_configs[config.provider_type] = config
            self.provider_metrics[config.provider_type] = ProviderMetrics()
            
            logger.info(f"Initialized provider: {config.provider_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize provider {config.provider_type}: {e}")
    
    async def _select_providers_for_capability(self, capability: str) -> List[ProviderType]:
        """Select providers for a specific capability based on strategy."""
        available_providers = [
            pt for pt in self.capabilities.get(capability, [])
            if pt in self.providers and self.provider_configs[pt].enabled
        ]
        
        if not available_providers:
            return []
        
        # Filter by data source preference
        if capability in ['stock_quotes', 'options_chains']:
            if self.data_source_preference == DataSource.EODHD_ONLY:
                available_providers = [pt for pt in available_providers if pt == ProviderType.EODHD]
            elif self.data_source_preference == DataSource.MARKETDATA_ONLY:
                available_providers = [pt for pt in available_providers if pt == ProviderType.MARKETDATA]
            elif self.data_source_preference == DataSource.EODHD_PRIMARY:
                available_providers = sorted(available_providers, key=lambda pt: 0 if pt == ProviderType.EODHD else 1)
            elif self.data_source_preference == DataSource.MARKETDATA_PRIMARY:
                available_providers = sorted(available_providers, key=lambda pt: 0 if pt == ProviderType.MARKETDATA else 1)
        
        # Apply strategy
        if self.strategy == ProviderStrategy.PRIMARY_ONLY:
            return available_providers[:1]
        elif self.strategy == ProviderStrategy.LOAD_BALANCE:
            # Simple random selection for load balancing
            return [random.choice(available_providers)]
        elif self.strategy == ProviderStrategy.BEST_QUALITY:
            # Sort by success rate
            available_providers.sort(
                key=lambda pt: self.provider_metrics[pt].success_rate,
                reverse=True
            )
        elif self.strategy == ProviderStrategy.FASTEST:
            # Sort by average response time
            available_providers.sort(
                key=lambda pt: self.provider_metrics[pt].average_response_time_ms
            )
        elif self.strategy == ProviderStrategy.COST_OPTIMIZED:
            # Sort by cost per request
            available_providers.sort(
                key=lambda pt: self.provider_configs[pt].cost_per_request
            )
        
        # For failover strategy, return all available providers in order
        return available_providers
    
    async def _update_provider_metrics(
        self,
        provider_type: ProviderType,
        success: bool,
        response_time_seconds: float,
        cost: float
    ) -> None:
        """Update provider performance metrics."""
        metrics = self.provider_metrics[provider_type]
        
        metrics.total_requests += 1
        metrics.total_cost += cost
        
        if success:
            metrics.successful_requests += 1
            metrics.last_success = datetime.now()
            metrics.consecutive_failures = 0
        else:
            metrics.failed_requests += 1
            metrics.last_failure = datetime.now()
            metrics.consecutive_failures += 1
        
        # Update rolling average response time
        response_time_ms = response_time_seconds * 1000
        if metrics.total_requests == 1:
            metrics.average_response_time_ms = response_time_ms
        else:
            # Rolling average
            alpha = 0.1  # Smoothing factor
            metrics.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * metrics.average_response_time_ms
            )
    
    def _generate_screening_cache_key(self, criteria: ScreeningCriteria) -> str:
        """Generate cache key for screening criteria."""
        # Create a deterministic key based on criteria
        key_parts = []
        
        if criteria.min_market_cap:
            key_parts.append(f"min_cap:{criteria.min_market_cap}")
        if criteria.max_market_cap:
            key_parts.append(f"max_cap:{criteria.max_market_cap}")
        if criteria.min_price:
            key_parts.append(f"min_price:{criteria.min_price}")
        if criteria.max_price:
            key_parts.append(f"max_price:{criteria.max_price}")
        if criteria.min_volume:
            key_parts.append(f"min_vol:{criteria.min_volume}")
        if criteria.exclude_sectors:
            key_parts.append(f"ex_sectors:{','.join(sorted(criteria.exclude_sectors))}")
        if criteria.exclude_symbols:
            key_parts.append(f"ex_symbols:{','.join(sorted(criteria.exclude_symbols))}")
        
        key = f"screening:{'|'.join(key_parts)}"
        return key
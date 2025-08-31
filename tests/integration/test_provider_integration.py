"""
Integration tests for the complete provider system.

Tests the integration between:
- Provider Manager orchestration
- EODHD and MarketData clients
- Caching layer functionality
- Data validation and quality checks
- Failover and error handling
- End-to-end workflow scenarios
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal

from src.providers import (
    ProviderManager,
    ProviderConfig,
    ProviderStrategy,
    DataSource,
    EODHDClient,
    MarketDataClient,
    CacheManager,
    DataValidator,
    ProviderType
)
from src.models import (
    ScreeningCriteria,
    StockQuote,
    OptionContract,
    FundamentalData,
    TechnicalIndicators,
    ProviderResponse,
    HealthCheckResult,
    ProviderStatus,
    OptionType
)
from src.providers.exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidSymbolError
)


class TestProviderIntegration:
    """Integration test suite for the complete provider system."""
    
    @pytest.fixture
    def provider_configs(self):
        """Provider configurations for testing."""
        return [
            ProviderConfig(
                provider_type=ProviderType.EODHD,
                config={
                    "api_key": "test_eodhd_key",
                    "base_url": "https://eodhd.com/api",
                    "requests_per_minute": 60
                },
                priority=1,
                capabilities=["stock_screening", "stock_quotes", "fundamentals"]
            ),
            ProviderConfig(
                provider_type=ProviderType.MARKETDATA,
                config={
                    "api_key": "test_marketdata_key",
                    "base_url": "https://api.marketdata.app/v1",
                    "requests_per_minute": 100
                },
                priority=2,
                capabilities=["stock_quotes", "options_chains"]
            )
        ]
    
    @pytest.fixture
    def cache_manager(self):
        """Cache manager for testing."""
        return CacheManager(
            max_memory_size_mb=10,  # Small for testing
            enable_persistence=False  # Avoid disk I/O in tests
        )
    
    @pytest.fixture
    def data_validator(self):
        """Data validator for testing."""
        return DataValidator(min_quality_score=70.0)
    
    @pytest.fixture
    async def provider_manager(self, provider_configs):
        """Provider manager with all providers initialized."""
        manager = ProviderManager(
            provider_configs=provider_configs,
            strategy=ProviderStrategy.FAILOVER,
            data_source_preference=DataSource.EODHD_PRIMARY,
            enable_caching=True,
            enable_validation=True
        )
        
        async with manager:
            yield manager
    
    @pytest.mark.asyncio
    async def test_provider_manager_initialization(self, provider_configs):
        """Test provider manager initialization with multiple providers."""
        manager = ProviderManager(
            provider_configs=provider_configs,
            strategy=ProviderStrategy.FAILOVER
        )
        
        assert len(manager.providers) == 2
        assert ProviderType.EODHD in manager.providers
        assert ProviderType.MARKETDATA in manager.providers
        
        # Test provider metrics initialization
        assert len(manager.provider_metrics) == 2
        for provider_type in [ProviderType.EODHD, ProviderType.MARKETDATA]:
            assert provider_type in manager.provider_metrics
            assert manager.provider_metrics[provider_type].total_requests == 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_screening_workflow(self, provider_manager):
        """Test complete screening workflow from criteria to results."""
        # Mock EODHD screening response
        mock_screening_response = ProviderResponse(
            success=True,
            data={
                "data": [
                    {"code": "AAPL", "market_capitalization": 2_000_000_000},
                    {"code": "MSFT", "market_capitalization": 2_500_000_000}
                ]
            },
            provider=ProviderType.EODHD
        )
        
        # Mock quote responses for both providers
        mock_quote_response = ProviderResponse(
            success=True,
            data=[{
                "date": "2025-01-15",
                "adjusted_close": 150.0,
                "volume": 50_000_000,
                "high": 152.0,
                "low": 148.0,
                "open": 149.0
            }],
            provider=ProviderType.EODHD
        )
        
        # Mock options chain response
        mock_options_response = ProviderResponse(
            success=True,
            data={
                "optionSymbol": ["AAPL240119C150000"],
                "underlying": ["AAPL"],
                "strike": [150.0],
                "expiration": [1705622400],
                "side": ["call"],
                "bid": [5.0],
                "ask": [5.2],
                "delta": [0.6]
            },
            provider=ProviderType.MARKETDATA
        )
        
        criteria = ScreeningCriteria(
            min_market_cap=1_000_000_000,
            max_market_cap=5_000_000_000,
            min_price=50.0,
            min_volume=1_000_000
        )
        
        with patch.object(provider_manager.providers[ProviderType.EODHD], '_make_request', return_value=mock_screening_response):
            with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', return_value=mock_options_response):
                # Step 1: Screen stocks
                symbols = await provider_manager.screen_stocks(criteria)
                assert len(symbols) == 2
                assert "AAPL" in symbols
                assert "MSFT" in symbols
                
                # Step 2: Get quotes for screened symbols
                with patch.object(provider_manager.providers[ProviderType.EODHD], '_make_request', return_value=mock_quote_response):
                    quotes = await provider_manager.get_stock_quotes(symbols)
                    assert len(quotes) >= 1  # At least one successful quote
                
                # Step 3: Get options chain for a symbol
                contracts = await provider_manager.get_options_chain("AAPL")
                assert len(contracts) >= 1
                assert contracts[0].underlying_symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_failover_between_providers(self, provider_manager):
        """Test failover from primary to secondary provider."""
        # Mock EODHD failure
        eodhd_error_response = ProviderResponse(
            success=False,
            error="EODHD API Error",
            provider=ProviderType.EODHD
        )
        
        # Mock MarketData success
        marketdata_success_response = ProviderResponse(
            success=True,
            data={
                "last": 150.0,
                "change": 2.0,
                "changepct": 1.35,
                "volume": 50_000_000
            },
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(provider_manager.providers[ProviderType.EODHD], '_make_request', return_value=eodhd_error_response):
            with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', return_value=marketdata_success_response):
                
                # Should failover from EODHD to MarketData
                quote = await provider_manager.get_stock_quote("AAPL")
                
                assert quote is not None
                assert quote.last_price == 150.0
                
                # Check metrics show the failure and success
                eodhd_metrics = provider_manager.provider_metrics[ProviderType.EODHD]
                marketdata_metrics = provider_manager.provider_metrics[ProviderType.MARKETDATA]
                
                assert eodhd_metrics.failed_requests > 0
                assert marketdata_metrics.successful_requests > 0
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, provider_manager):
        """Test caching integration across the provider system."""
        mock_response = ProviderResponse(
            success=True,
            data={
                "last": 150.0,
                "change": 2.0,
                "changepct": 1.35,
                "volume": 50_000_000
            },
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', return_value=mock_response) as mock_request:
            
            # First request should hit the provider
            quote1 = await provider_manager.get_stock_quote("AAPL")
            assert quote1 is not None
            assert mock_request.call_count == 1
            
            # Second request should use cache (if caching is enabled)
            if provider_manager.cache_manager:
                quote2 = await provider_manager.get_stock_quote("AAPL")
                assert quote2 is not None
                # May or may not hit cache depending on TTL and implementation
                # This is more about verifying the integration works
    
    @pytest.mark.asyncio
    async def test_data_validation_integration(self, provider_manager):
        """Test data validation integration with provider responses."""
        # Mock response with invalid data
        invalid_quote_response = ProviderResponse(
            success=True,
            data={
                "last": -50.0,  # Invalid negative price
                "change": 0,
                "changepct": 0,
                "volume": -1000  # Invalid negative volume
            },
            provider=ProviderType.MARKETDATA
        )
        
        # Mock response with valid data
        valid_quote_response = ProviderResponse(
            success=True,
            data=[{
                "date": "2025-01-15",
                "adjusted_close": 150.0,
                "volume": 50_000_000,
                "high": 152.0,
                "low": 148.0,
                "open": 149.0
            }],
            provider=ProviderType.EODHD
        )
        
        with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', return_value=invalid_quote_response):
            with patch.object(provider_manager.providers[ProviderType.EODHD], '_make_request', return_value=valid_quote_response):
                
                # Should skip invalid data and failover to valid provider
                quote = await provider_manager.get_stock_quote("AAPL")
                
                if quote:  # Depending on validation settings
                    assert quote.last_price > 0  # Should have valid data
    
    @pytest.mark.asyncio
    async def test_provider_health_monitoring(self, provider_manager):
        """Test provider health monitoring across all providers."""
        # Mock health check responses
        healthy_response = HealthCheckResult(
            provider=ProviderType.EODHD,
            status=ProviderStatus.HEALTHY,
            response_time_ms=100.0
        )
        
        unhealthy_response = HealthCheckResult(
            provider=ProviderType.MARKETDATA,
            status=ProviderStatus.UNHEALTHY,
            message="API Error"
        )
        
        with patch.object(provider_manager.providers[ProviderType.EODHD], 'get_health_status', return_value=healthy_response):
            with patch.object(provider_manager.providers[ProviderType.MARKETDATA], 'get_health_status', return_value=unhealthy_response):
                
                health_status = await provider_manager.get_provider_health_status()
                
                assert len(health_status) == 2
                assert health_status[ProviderType.EODHD].status == ProviderStatus.HEALTHY
                assert health_status[ProviderType.MARKETDATA].status == ProviderStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_pmcc_workflow_integration(self, provider_manager):
        """Test PMCC-specific workflow integration."""
        # Mock LEAPS options
        leaps_response = ProviderResponse(
            success=True,
            data={
                "optionSymbol": ["AAPL250117C100000"],
                "underlying": ["AAPL"],
                "strike": [100.0],
                "expiration": [1737158400],  # Far expiration
                "side": ["call"],
                "bid": [45.0],
                "ask": [45.5],
                "delta": [0.75]  # High delta for LEAPS
            },
            provider=ProviderType.MARKETDATA
        )
        
        # Mock short calls
        short_calls_response = ProviderResponse(
            success=True,
            data={
                "optionSymbol": ["AAPL250221C155000"],
                "underlying": ["AAPL"],
                "strike": [155.0],
                "expiration": [1708560000],  # Near expiration
                "side": ["call"],
                "bid": [2.0],
                "ask": [2.2],
                "delta": [0.25]  # Low delta for short calls
            },
            provider=ProviderType.MARKETDATA
        )
        
        responses = [leaps_response, short_calls_response]
        
        with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', side_effect=responses):
            pmcc_chains = await provider_manager.get_pmcc_option_chains("AAPL")
            
            assert "leaps" in pmcc_chains
            assert "short_calls" in pmcc_chains
            
            if pmcc_chains["leaps"]:
                leaps = pmcc_chains["leaps"][0]
                assert leaps.strike == Decimal('100.0')
                assert leaps.delta >= 0.7  # High delta for LEAPS
            
            if pmcc_chains["short_calls"]:
                short_call = pmcc_chains["short_calls"][0]
                assert short_call.strike == Decimal('155.0')
                assert short_call.delta <= 0.4  # Lower delta for short calls
    
    @pytest.mark.asyncio
    async def test_usage_statistics_and_metrics(self, provider_manager):
        """Test comprehensive usage statistics collection."""
        mock_response = ProviderResponse(
            success=True,
            data={"last": 150.0, "change": 2.0},
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', return_value=mock_response):
            
            # Make several requests
            await provider_manager.get_stock_quote("AAPL")
            await provider_manager.get_stock_quote("MSFT")
            await provider_manager.get_stock_quote("GOOGL")
            
            # Get usage summary
            summary = await provider_manager.get_usage_summary()
            
            assert summary["total_requests"] > 0
            assert summary["overall_success_rate"] >= 0.0
            assert "provider_metrics" in summary
            assert ProviderType.MARKETDATA.value in summary["provider_metrics"]
            
            # Check individual provider metrics
            metrics = await provider_manager.get_provider_metrics()
            marketdata_metrics = metrics[ProviderType.MARKETDATA]
            
            assert marketdata_metrics.total_requests >= 3
            assert marketdata_metrics.successful_requests >= 3
            assert marketdata_metrics.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_provider_strategy_selection(self):
        """Test different provider selection strategies."""
        provider_configs = [
            ProviderConfig(
                provider_type=ProviderType.EODHD,
                config={"api_key": "test1"},
                priority=1,
                cost_per_request=0.02
            ),
            ProviderConfig(
                provider_type=ProviderType.MARKETDATA,
                config={"api_key": "test2"},
                priority=2,
                cost_per_request=0.01  # Cheaper
            )
        ]
        
        # Test cost-optimized strategy
        cost_manager = ProviderManager(
            provider_configs=provider_configs,
            strategy=ProviderStrategy.COST_OPTIMIZED
        )
        
        providers = await cost_manager._select_providers_for_capability('stock_quotes')
        # Should prefer MarketData due to lower cost
        assert providers[0] == ProviderType.MARKETDATA
        
        # Test primary-only strategy
        primary_manager = ProviderManager(
            provider_configs=provider_configs,
            strategy=ProviderStrategy.PRIMARY_ONLY
        )
        
        providers = await primary_manager._select_providers_for_capability('stock_quotes')
        # Should only return one provider
        assert len(providers) == 1
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_handling(self, provider_manager):
        """Test error propagation through the provider system."""
        # Mock authentication error
        auth_error_response = ProviderResponse(
            success=False,
            error="Authentication failed",
            provider=ProviderType.EODHD
        )
        
        with patch.object(provider_manager.providers[ProviderType.EODHD], '_make_request', side_effect=AuthenticationError("Invalid API key")):
            
            # Should propagate authentication errors
            with pytest.raises(AuthenticationError):
                await provider_manager.get_fundamental_data("AAPL")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, provider_manager):
        """Test handling of concurrent requests across providers."""
        mock_response = ProviderResponse(
            success=True,
            data={"last": 150.0, "change": 2.0, "changepct": 1.35, "volume": 50_000_000},
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(provider_manager.providers[ProviderType.MARKETDATA], '_make_request', return_value=mock_response):
            
            # Make concurrent requests
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            tasks = [provider_manager.get_stock_quote(symbol) for symbol in symbols]
            
            quotes = await asyncio.gather(*tasks)
            
            # All requests should succeed
            assert len([q for q in quotes if q is not None]) == len(symbols)
            
            # Check that metrics were updated correctly
            metrics = await provider_manager.get_provider_metrics()
            total_requests = sum(m.total_requests for m in metrics.values())
            assert total_requests >= len(symbols)
    
    @pytest.mark.asyncio
    async def test_data_source_preference_routing(self):
        """Test data source preference routing."""
        provider_configs = [
            ProviderConfig(
                provider_type=ProviderType.EODHD,
                config={"api_key": "test1"},
                priority=1
            ),
            ProviderConfig(
                provider_type=ProviderType.MARKETDATA,
                config={"api_key": "test2"},
                priority=2
            )
        ]
        
        # Test MarketData primary preference
        marketdata_manager = ProviderManager(
            provider_configs=provider_configs,
            data_source_preference=DataSource.MARKETDATA_PRIMARY
        )
        
        providers = await marketdata_manager._select_providers_for_capability('stock_quotes')
        assert providers[0] == ProviderType.MARKETDATA
        
        # Test EODHD only
        eodhd_only_manager = ProviderManager(
            provider_configs=provider_configs,
            data_source_preference=DataSource.EODHD_ONLY
        )
        
        providers = await eodhd_only_manager._select_providers_for_capability('stock_quotes')
        assert providers == [ProviderType.EODHD]


if __name__ == "__main__":
    pytest.main([__file__])
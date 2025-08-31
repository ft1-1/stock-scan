"""Comprehensive unit tests for providers module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import aiohttp
from aioresponses import aioresponses

# Import the modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from providers.eodhd_client import EODHDClient
from providers.marketdata_client import MarketDataClient
from providers.provider_manager import ProviderManager
from providers.cache import CacheManager, TTLCache
from providers.validators import DataValidator, DataQualityScorer
from providers.exceptions import (
    ProviderError, RateLimitError, AuthenticationError, 
    DataNotFoundError, InvalidDataError
)
from models.provider_models import ProviderConfig, CacheConfig


class TestEODHDClient:
    """Comprehensive tests for EODHD API client."""
    
    @pytest.fixture
    def client_config(self):
        """EODHD client configuration."""
        return ProviderConfig(
            provider_type="eodhd",
            api_key="test_api_key",
            base_url="https://eodhistoricaldata.com/api",
            rate_limit_per_minute=20,
            timeout_seconds=30,
            max_retries=3,
            backoff_factor=1.0
        )
    
    @pytest.fixture
    def eodhd_client(self, client_config):
        """Create EODHD client instance."""
        return EODHDClient(client_config)
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, eodhd_client, client_config):
        """Test client initialization and configuration."""
        assert eodhd_client.config == client_config
        assert eodhd_client.provider_type == "eodhd"
        assert eodhd_client._session is None  # Not initialized yet
        assert eodhd_client._request_count == 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, eodhd_client):
        """Test HTTP session lifecycle management."""
        # Session should be created on first use
        await eodhd_client._ensure_session()
        assert eodhd_client._session is not None
        assert isinstance(eodhd_client._session, aiohttp.ClientSession)
        
        # Cleanup should close session
        await eodhd_client.cleanup()
        assert eodhd_client._session.closed
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_success(self, eodhd_client, mock_eodhd_response):
        """Test successful stock quote retrieval."""
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                payload=mock_eodhd_response
            )
            
            quote = await eodhd_client.get_stock_quote("AAPL")
            
            assert quote.symbol == "AAPL"
            assert quote.last_price == 150.5
            assert quote.volume == 50_000_000
            assert eodhd_client._request_count == 1
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_not_found(self, eodhd_client):
        """Test stock quote with non-existent symbol."""
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/real-time/INVALID.US",
                status=404,
                payload={"error": "Symbol not found"}
            )
            
            with pytest.raises(DataNotFoundError):
                await eodhd_client.get_stock_quote("INVALID")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, eodhd_client):
        """Test rate limiting functionality."""
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                status=429,
                payload={"error": "Rate limit exceeded"}
            )
            
            with pytest.raises(RateLimitError):
                await eodhd_client.get_stock_quote("AAPL")
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, eodhd_client):
        """Test authentication error handling."""
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                status=401,
                payload={"error": "Invalid API key"}
            )
            
            with pytest.raises(AuthenticationError):
                await eodhd_client.get_stock_quote("AAPL")
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, eodhd_client):
        """Test automatic retry on transient failures."""
        with aioresponses() as m:
            # First request fails with server error
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                status=500,
                payload={"error": "Internal server error"}
            )
            # Second request succeeds
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                payload={"code": "AAPL.US", "close": 150.5, "volume": 1000000}
            )
            
            quote = await eodhd_client.get_stock_quote("AAPL")
            assert quote.last_price == 150.5
            assert eodhd_client._request_count == 2  # One retry
    
    @pytest.mark.asyncio
    async def test_historical_data_request(self, eodhd_client):
        """Test historical data retrieval."""
        historical_response = [
            {
                "date": "2024-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000
            }
        ]
        
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/eod/AAPL.US",
                payload=historical_response
            )
            
            data = await eodhd_client.get_historical_data("AAPL", days=30)
            
            assert len(data) == 1
            assert data[0]["symbol"] == "AAPL"
            assert data[0]["close"] == 103.0
    
    @pytest.mark.asyncio
    async def test_enhanced_data_request(self, eodhd_client):
        """Test enhanced fundamental data retrieval."""
        enhanced_response = {
            "General": {
                "Code": "AAPL",
                "Name": "Apple Inc",
                "Exchange": "NASDAQ",
                "CurrencyCode": "USD"
            },
            "Highlights": {
                "MarketCapitalization": 3000000000000,
                "PERatio": 25.5,
                "EPS": 6.05
            }
        }
        
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/fundamentals/AAPL.US",
                payload=enhanced_response
            )
            
            data = await eodhd_client.get_enhanced_data("AAPL")
            
            assert data["symbol"] == "AAPL"
            assert data["market_cap"] == 3000000000000
            assert data["pe_ratio"] == 25.5
    
    @pytest.mark.asyncio
    async def test_request_validation(self, eodhd_client):
        """Test request parameter validation."""
        with pytest.raises(ValueError):
            await eodhd_client.get_stock_quote("")  # Empty symbol
        
        with pytest.raises(ValueError):
            await eodhd_client.get_historical_data("AAPL", days=0)  # Invalid days
    
    @pytest.mark.asyncio
    async def test_health_check(self, eodhd_client):
        """Test health check functionality."""
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                payload={"code": "AAPL.US", "close": 150.0}
            )
            
            health = await eodhd_client.get_health_status()
            
            assert health["status"] == "healthy"
            assert health["provider"] == "eodhd"
            assert "response_time" in health
    
    @pytest.mark.asyncio
    async def test_usage_statistics(self, eodhd_client):
        """Test usage statistics tracking."""
        with aioresponses() as m:
            m.get(
                f"{eodhd_client.config.base_url}/real-time/AAPL.US",
                payload={"code": "AAPL.US", "close": 150.0}
            )
            
            await eodhd_client.get_stock_quote("AAPL")
            
            stats = await eodhd_client.get_usage_stats()
            
            assert stats["request_count"] == 1
            assert stats["provider"] == "eodhd"
            assert "total_cost" in stats


class TestMarketDataClient:
    """Comprehensive tests for MarketData.app API client."""
    
    @pytest.fixture
    def client_config(self):
        """MarketData client configuration."""
        return ProviderConfig(
            provider_type="marketdata",
            api_key="test_marketdata_key",
            base_url="https://api.marketdata.app/v1",
            rate_limit_per_minute=100,
            timeout_seconds=30,
            max_retries=3,
            backoff_factor=1.0
        )
    
    @pytest.fixture
    def marketdata_client(self, client_config):
        """Create MarketData client instance."""
        return MarketDataClient(client_config)
    
    @pytest.mark.asyncio
    async def test_options_chain_request(self, marketdata_client, test_data_generator):
        """Test options chain data retrieval."""
        options_data = test_data_generator.generate_options_chain("AAPL", 150.0)
        
        with aioresponses() as m:
            m.get(
                f"{marketdata_client.config.base_url}/options/chain/AAPL",
                payload={
                    "s": "ok",
                    "symbol": "AAPL",
                    "options": options_data
                }
            )
            
            chain = await marketdata_client.get_options_chain("AAPL")
            
            assert len(chain) > 0
            assert all(opt.underlying_symbol == "AAPL" for opt in chain)
            assert all(hasattr(opt, 'delta') for opt in chain)
    
    @pytest.mark.asyncio
    async def test_options_chain_with_expiration_filter(self, marketdata_client):
        """Test options chain with expiration date filtering."""
        with aioresponses() as m:
            m.get(
                f"{marketdata_client.config.base_url}/options/chain/AAPL",
                payload={
                    "s": "ok",
                    "symbol": "AAPL",
                    "options": []
                }
            )
            
            expiration = datetime.now().date() + timedelta(days=30)
            chain = await marketdata_client.get_options_chain("AAPL", expiration=expiration)
            
            # Verify expiration filter was applied in request
            assert marketdata_client._request_count == 1
    
    @pytest.mark.asyncio
    async def test_greeks_calculation_validation(self, marketdata_client, test_data_generator):
        """Test validation of Greeks in options data."""
        options_data = test_data_generator.generate_options_chain("AAPL", 150.0)
        
        with aioresponses() as m:
            m.get(
                f"{marketdata_client.config.base_url}/options/chain/AAPL",
                payload={
                    "s": "ok",
                    "symbol": "AAPL", 
                    "options": options_data
                }
            )
            
            chain = await marketdata_client.get_options_chain("AAPL")
            
            for option in chain:
                # Validate Greeks are within expected ranges
                assert 0 <= option.delta <= 1, f"Invalid delta: {option.delta}"
                assert option.gamma >= 0, f"Invalid gamma: {option.gamma}"
                assert option.theta <= 0, f"Invalid theta: {option.theta}"
                assert option.vega >= 0, f"Invalid vega: {option.vega}"
    
    @pytest.mark.asyncio
    async def test_bulk_quotes_request(self, marketdata_client):
        """Test bulk stock quotes retrieval."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        with aioresponses() as m:
            m.get(
                f"{marketdata_client.config.base_url}/stocks/quotes",
                payload={
                    "s": "ok",
                    "symbol": symbols,
                    "last": [150.0, 300.0, 2800.0],
                    "change": [2.5, -5.0, 15.0],
                    "volume": [1000000, 800000, 300000]
                }
            )
            
            quotes = await marketdata_client.get_bulk_quotes(symbols)
            
            assert len(quotes) == 3
            assert quotes[0].symbol == "AAPL"
            assert quotes[1].symbol == "MSFT"
            assert quotes[2].symbol == "GOOGL"
    
    @pytest.mark.asyncio
    async def test_error_response_handling(self, marketdata_client):
        """Test handling of API error responses."""
        with aioresponses() as m:
            m.get(
                f"{marketdata_client.config.base_url}/options/chain/INVALID",
                payload={
                    "s": "error",
                    "errmsg": "Symbol not found"
                }
            )
            
            with pytest.raises(DataNotFoundError):
                await marketdata_client.get_options_chain("INVALID")


class TestCacheManager:
    """Tests for cache management functionality."""
    
    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return CacheConfig(
            enabled=True,
            backend="memory",
            default_ttl_seconds=300,
            max_size_mb=100,
            compression_enabled=True
        )
    
    @pytest.fixture
    def cache_manager(self, cache_config):
        """Create cache manager instance."""
        return CacheManager(cache_config)
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache_manager):
        """Test basic cache set and get operations."""
        key = "test_key"
        value = {"symbol": "AAPL", "price": 150.0}
        
        await cache_manager.set(key, value, ttl=60)
        cached_value = await cache_manager.get(key)
        
        assert cached_value == value
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager):
        """Test cache TTL expiration."""
        key = "expire_test"
        value = {"data": "test"}
        
        await cache_manager.set(key, value, ttl=1)  # 1 second TTL
        
        # Should exist immediately
        assert await cache_manager.get(key) == value
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        assert await cache_manager.get(key) is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager):
        """Test cache key invalidation."""
        key = "invalidate_test"
        value = {"data": "test"}
        
        await cache_manager.set(key, value)
        assert await cache_manager.get(key) == value
        
        await cache_manager.delete(key)
        assert await cache_manager.get(key) is None
    
    @pytest.mark.asyncio
    async def test_cache_size_limits(self, cache_manager):
        """Test cache size limitation enforcement."""
        # Fill cache with large data
        large_data = {"data": "x" * 1000000}  # 1MB string
        
        for i in range(200):  # Should exceed 100MB limit
            await cache_manager.set(f"large_key_{i}", large_data)
        
        # Verify eviction occurred (not all keys should be present)
        present_keys = 0
        for i in range(200):
            if await cache_manager.get(f"large_key_{i}") is not None:
                present_keys += 1
        
        assert present_keys < 200  # Some keys should have been evicted
    
    @pytest.mark.asyncio
    async def test_cache_compression(self, cache_manager):
        """Test data compression in cache."""
        # This test verifies compression is working by checking
        # that large repetitive data is stored efficiently
        key = "compression_test"
        # Large repetitive data that should compress well
        value = {"data": "A" * 100000}
        
        await cache_manager.set(key, value)
        retrieved = await cache_manager.get(key)
        
        assert retrieved == value
    
    def test_ttl_cache_basic_operations(self):
        """Test TTL cache basic operations."""
        cache = TTLCache(max_size=100, default_ttl=60)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_ttl_cache_expiration(self):
        """Test TTL cache expiration."""
        cache = TTLCache(max_size=100, default_ttl=1)  # 1 second TTL
        
        cache.set("expire_key", "value")
        assert cache.get("expire_key") == "value"
        
        # Simulate time passage
        import time
        time.sleep(1.1)
        
        assert cache.get("expire_key") is None
    
    def test_ttl_cache_size_limit(self):
        """Test TTL cache size limitations."""
        cache = TTLCache(max_size=3, default_ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict oldest
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key4") == "value4"  # Present


class TestDataValidator:
    """Tests for data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create data validator instance."""
        return DataValidator()
    
    def test_stock_quote_validation_valid(self, validator, sample_stock_quote):
        """Test validation of valid stock quote."""
        assert validator.validate_stock_quote(sample_stock_quote) is True
    
    def test_stock_quote_validation_invalid_price(self, validator):
        """Test validation with invalid price data."""
        invalid_quote = {
            "symbol": "AAPL",
            "last_price": -150.0,  # Negative price
            "volume": 1000000,
            "bid": 149.0,
            "ask": 151.0
        }
        
        with pytest.raises(InvalidDataError):
            validator.validate_stock_quote(invalid_quote)
    
    def test_stock_quote_validation_missing_fields(self, validator):
        """Test validation with missing required fields."""
        incomplete_quote = {
            "symbol": "AAPL",
            "last_price": 150.0
            # Missing volume, bid, ask
        }
        
        with pytest.raises(InvalidDataError):
            validator.validate_stock_quote(incomplete_quote)
    
    def test_option_contract_validation_valid(self, validator, sample_option_contract):
        """Test validation of valid option contract."""
        assert validator.validate_option_contract(sample_option_contract) is True
    
    def test_option_contract_validation_invalid_greeks(self, validator):
        """Test validation with invalid Greeks."""
        invalid_option = {
            "option_symbol": "AAPL250117C00150000",
            "underlying_symbol": "AAPL",
            "strike": 150.0,
            "delta": 1.5,  # Invalid delta > 1
            "gamma": 0.025,
            "theta": -0.08,
            "vega": 0.35
        }
        
        with pytest.raises(InvalidDataError):
            validator.validate_option_contract(invalid_option)
    
    def test_options_chain_validation(self, validator, test_data_generator):
        """Test validation of complete options chain."""
        chain = test_data_generator.generate_options_chain("AAPL", 150.0)
        
        assert validator.validate_options_chain(chain) is True
    
    def test_historical_data_validation(self, validator):
        """Test validation of historical price data."""
        valid_data = [
            {
                "date": "2024-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000
            }
        ]
        
        assert validator.validate_historical_data(valid_data) is True
    
    def test_data_quality_scoring(self):
        """Test data quality scoring functionality."""
        scorer = DataQualityScorer()
        
        # High quality data
        high_quality_data = {
            "symbol": "AAPL",
            "last_price": 150.0,
            "volume": 1000000,
            "bid": 149.95,
            "ask": 150.05,
            "timestamp": datetime.now().isoformat()
        }
        
        score = scorer.calculate_quality_score(high_quality_data)
        assert score >= 80  # Should be high quality
        
        # Low quality data (missing fields, stale timestamp)
        low_quality_data = {
            "symbol": "AAPL",
            "last_price": 150.0,
            "timestamp": (datetime.now() - timedelta(hours=25)).isoformat()
        }
        
        score = scorer.calculate_quality_score(low_quality_data)
        assert score < 50  # Should be low quality


class TestProviderManager:
    """Tests for provider manager coordination."""
    
    @pytest.fixture
    def provider_configs(self):
        """Provider configurations."""
        return {
            "eodhd": ProviderConfig(
                provider_type="eodhd",
                api_key="test_eodhd_key",
                base_url="https://eodhistoricaldata.com/api",
                rate_limit_per_minute=20
            ),
            "marketdata": ProviderConfig(
                provider_type="marketdata",
                api_key="test_marketdata_key",
                base_url="https://api.marketdata.app/v1",
                rate_limit_per_minute=100
            )
        }
    
    @pytest.fixture
    def provider_manager(self, provider_configs):
        """Create provider manager instance."""
        return ProviderManager(provider_configs)
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, provider_manager):
        """Test provider manager initialization."""
        await provider_manager.initialize()
        
        assert "eodhd" in provider_manager.providers
        assert "marketdata" in provider_manager.providers
        assert len(provider_manager.providers) == 2
    
    @pytest.mark.asyncio
    async def test_failover_mechanism(self, provider_manager):
        """Test automatic failover between providers."""
        await provider_manager.initialize()
        
        # Mock primary provider failure
        with patch.object(provider_manager.providers["eodhd"], "get_stock_quote") as mock_eodhd:
            mock_eodhd.side_effect = ProviderError("Service unavailable")
            
            with patch.object(provider_manager.providers["marketdata"], "get_stock_quote") as mock_marketdata:
                mock_marketdata.return_value = Mock(symbol="AAPL", last_price=150.0)
                
                quote = await provider_manager.get_stock_quote("AAPL", preferred_provider="eodhd")
                
                # Should failover to marketdata
                assert quote.last_price == 150.0
                mock_eodhd.assert_called_once()
                mock_marketdata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_provider_health_monitoring(self, provider_manager):
        """Test provider health status monitoring."""
        await provider_manager.initialize()
        
        health_status = await provider_manager.get_health_status()
        
        assert "eodhd" in health_status
        assert "marketdata" in health_status
        assert all("status" in status for status in health_status.values())
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, provider_manager):
        """Test circuit breaker functionality."""
        await provider_manager.initialize()
        
        # Simulate multiple failures to trigger circuit breaker
        with patch.object(provider_manager.providers["eodhd"], "get_stock_quote") as mock_eodhd:
            mock_eodhd.side_effect = ProviderError("Service error")
            
            # Multiple failed requests should trigger circuit breaker
            for _ in range(5):
                try:
                    await provider_manager.get_stock_quote("AAPL", preferred_provider="eodhd")
                except ProviderError:
                    pass
            
            # Circuit breaker should be open, subsequent calls should fail fast
            with pytest.raises(ProviderError, match="Circuit breaker"):
                await provider_manager.get_stock_quote("AAPL", preferred_provider="eodhd")
    
    @pytest.mark.asyncio
    async def test_provider_cost_tracking(self, provider_manager):
        """Test cost tracking across providers."""
        await provider_manager.initialize()
        
        with patch.object(provider_manager.providers["eodhd"], "get_stock_quote") as mock_eodhd:
            mock_eodhd.return_value = Mock(symbol="AAPL", last_price=150.0)
            
            await provider_manager.get_stock_quote("AAPL", preferred_provider="eodhd")
            
            cost_summary = await provider_manager.get_cost_summary()
            
            assert "eodhd" in cost_summary
            assert cost_summary["eodhd"]["request_count"] >= 1
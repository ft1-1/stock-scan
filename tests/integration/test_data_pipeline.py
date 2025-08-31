"""Integration tests for data pipeline and provider coordination."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from aioresponses import aioresponses

# Import modules for integration testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from providers.provider_manager import ProviderManager
from providers.eodhd_client import EODHDClient
from providers.marketdata_client import MarketDataClient
from providers.cache import CacheManager
from providers.validators import DataValidator, DataQualityScorer
from models.provider_models import ProviderConfig, CacheConfig
from analytics.technical_indicators import TechnicalIndicators
from analytics.momentum_analysis import MomentumAnalyzer
from analytics.options_selector import OptionsSelector, CallOptionCriteria
from ai_analysis.data_packager import DataPackager, PackagingConfig


class TestProviderCoordination:
    """Test coordination between multiple data providers."""
    
    @pytest.fixture
    def provider_configs(self):
        """Provider configurations for testing."""
        return {
            "eodhd": ProviderConfig(
                provider_type="eodhd",
                api_key="test_eodhd_key",
                base_url="https://eodhistoricaldata.com/api",
                rate_limit_per_minute=20,
                timeout_seconds=30,
                max_retries=2
            ),
            "marketdata": ProviderConfig(
                provider_type="marketdata",
                api_key="test_marketdata_key",
                base_url="https://api.marketdata.app/v1",
                rate_limit_per_minute=100,
                timeout_seconds=30,
                max_retries=2
            )
        }
    
    @pytest.fixture
    def cache_config(self):
        """Cache configuration for testing."""
        return CacheConfig(
            enabled=True,
            backend="memory",
            default_ttl_seconds=300,
            max_size_mb=50,
            compression_enabled=True
        )
    
    @pytest.fixture
    async def provider_manager(self, provider_configs, cache_config):
        """Create provider manager with cache."""
        manager = ProviderManager(provider_configs, cache_config)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_provider_quote_fetching(self, provider_manager, test_data_generator):
        """Test fetching quotes from multiple providers with failover."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Mock responses for both providers
        eodhd_responses = {}
        marketdata_responses = {}
        
        for symbol in symbols:
            eodhd_data = test_data_generator.mock_eodhd_response(symbol, "quote")
            marketdata_data = test_data_generator.mock_marketdata_response(symbol, "quote")
            
            eodhd_responses[f"https://eodhistoricaldata.com/api/real-time/{symbol}.US"] = eodhd_data
            marketdata_responses[f"https://api.marketdata.app/v1/stocks/quotes/{symbol}"] = marketdata_data
        
        with aioresponses() as m:
            # Add responses for both providers
            for url, response in eodhd_responses.items():
                m.get(url, payload=response)
            
            for url, response in marketdata_responses.items():
                m.get(url, payload=response)
            
            # Fetch quotes using preferred provider (EODHD)
            quotes = []
            for symbol in symbols:
                quote = await provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
                quotes.append(quote)
            
            assert len(quotes) == len(symbols)
            for quote in quotes:
                assert quote.symbol in symbols
                assert quote.last_price > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_failover_mechanism(self, provider_manager, test_data_generator):
        """Test automatic failover when primary provider fails."""
        symbol = "AAPL"
        
        with aioresponses() as m:
            # EODHD fails
            m.get(
                f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                status=500,
                payload={"error": "Internal server error"}
            )
            
            # MarketData succeeds
            marketdata_response = test_data_generator.mock_marketdata_response(symbol, "quote")
            m.get(
                f"https://api.marketdata.app/v1/stocks/quotes/{symbol}",
                payload=marketdata_response
            )
            
            # Should failover to MarketData
            quote = await provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
            
            assert quote is not None
            assert quote.symbol == symbol
            assert quote.last_price > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cache_integration_across_providers(self, provider_manager, test_data_generator):
        """Test cache integration across multiple providers."""
        symbol = "AAPL"
        
        with aioresponses() as m:
            # Mock successful response
            eodhd_response = test_data_generator.mock_eodhd_response(symbol, "quote")
            m.get(
                f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                payload=eodhd_response
            )
            
            # First request should hit provider
            quote1 = await provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
            
            # Second request should hit cache (no additional mock needed)
            quote2 = await provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
            
            assert quote1.symbol == quote2.symbol
            assert quote1.last_price == quote2.last_price
            
            # Verify cache metrics
            cache_stats = await provider_manager.get_cache_statistics()
            assert cache_stats["hit_rate"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_provider_requests(self, provider_manager, test_data_generator):
        """Test concurrent requests across multiple providers."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        with aioresponses() as m:
            # Mock responses for all symbols
            for symbol in symbols:
                eodhd_response = test_data_generator.mock_eodhd_response(symbol, "quote")
                m.get(
                    f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                    payload=eodhd_response
                )
        
        # Make concurrent requests
        start_time = datetime.now()
        
        tasks = [
            provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
            for symbol in symbols
        ]
        
        quotes = await asyncio.gather(*tasks)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Verify all quotes received
        assert len(quotes) == len(symbols)
        for i, quote in enumerate(quotes):
            assert quote.symbol == symbols[i]
        
        # Concurrent execution should be faster than sequential
        # (This is a rough check, actual timing depends on mock delays)
        assert execution_time < 5  # Should complete quickly with mocks
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limit_coordination(self, provider_manager):
        """Test rate limiting coordination across providers."""
        symbol = "AAPL"
        
        with aioresponses() as m:
            # First few requests succeed
            for i in range(3):
                m.get(
                    f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                    payload={"code": f"{symbol}.US", "close": 150.0}
                )
            
            # Subsequent requests hit rate limit
            m.get(
                f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                status=429,
                payload={"error": "Rate limit exceeded"}
            )
            
            # Make requests that should trigger rate limiting
            quotes = []
            for i in range(4):
                try:
                    quote = await provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
                    quotes.append(quote)
                except Exception as e:
                    # Rate limit errors are expected
                    if "rate limit" not in str(e).lower():
                        raise
            
            # Should have gotten some quotes before hitting limit
            assert len(quotes) >= 3
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_validation_across_providers(self, provider_manager, test_data_generator):
        """Test data validation across different providers."""
        symbol = "AAPL"
        validator = DataValidator()
        
        with aioresponses() as m:
            # Mock valid response from EODHD
            valid_response = test_data_generator.mock_eodhd_response(symbol, "quote")
            m.get(
                f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                payload=valid_response
            )
            
            # Mock invalid response from MarketData (for comparison)
            invalid_response = {
                "s": "ok",
                "symbol": [symbol],
                "last": [-150.0],  # Invalid negative price
                "volume": [1000000]
            }
            m.get(
                f"https://api.marketdata.app/v1/stocks/quotes/{symbol}",
                payload=invalid_response
            )
            
            # Get quote from valid provider
            quote = await provider_manager.get_stock_quote(symbol, preferred_provider="eodhd")
            
            # Validate the data
            quote_dict = {
                "symbol": quote.symbol,
                "last_price": quote.last_price,
                "volume": quote.volume,
                "bid": quote.bid,
                "ask": quote.ask
            }
            
            assert validator.validate_stock_quote(quote_dict) is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_health_monitoring(self, provider_manager):
        """Test health monitoring across providers."""
        with aioresponses() as m:
            # Mock health check responses
            m.get(
                "https://eodhistoricaldata.com/api/real-time/AAPL.US",
                payload={"code": "AAPL.US", "close": 150.0}
            )
            m.get(
                "https://api.marketdata.app/v1/stocks/quotes/AAPL",
                payload={"s": "ok", "symbol": ["AAPL"], "last": [150.0]}
            )
            
            health_status = await provider_manager.get_health_status()
            
            assert "eodhd" in health_status
            assert "marketdata" in health_status
            
            for provider, status in health_status.items():
                assert "status" in status
                assert status["status"] in ["healthy", "degraded", "unhealthy"]
                assert "response_time" in status


class TestDataPipelineIntegration:
    """Test integration of complete data pipeline."""
    
    @pytest.fixture
    def data_pipeline_components(self, test_data_generator):
        """Set up all pipeline components."""
        # Provider manager
        provider_configs = {
            "eodhd": ProviderConfig(
                provider_type="eodhd",
                api_key="test_key",
                base_url="https://eodhistoricaldata.com/api"
            )
        }
        provider_manager = ProviderManager(provider_configs)
        
        # Cache manager
        cache_config = CacheConfig(enabled=True, backend="memory")
        cache_manager = CacheManager(cache_config)
        
        # Analytics components
        momentum_analyzer = MomentumAnalyzer()
        options_selector = OptionsSelector(CallOptionCriteria())
        
        # Data packager
        packaging_config = PackagingConfig(max_payload_size_mb=5)
        data_packager = DataPackager(packaging_config)
        
        # Data validator
        validator = DataValidator()
        quality_scorer = DataQualityScorer()
        
        return {
            "provider_manager": provider_manager,
            "cache_manager": cache_manager,
            "momentum_analyzer": momentum_analyzer,
            "options_selector": options_selector,
            "data_packager": data_packager,
            "validator": validator,
            "quality_scorer": quality_scorer,
            "test_data_generator": test_data_generator
        }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_data_flow(self, data_pipeline_components):
        """Test complete data flow from provider to analysis."""
        components = data_pipeline_components
        symbol = "AAPL"
        
        # Initialize provider manager
        await components["provider_manager"].initialize()
        
        try:
            with aioresponses() as m:
                # Mock quote response
                quote_response = components["test_data_generator"].mock_eodhd_response(symbol, "quote")
                m.get(
                    f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                    payload=quote_response
                )
                
                # Mock historical data response
                hist_response = components["test_data_generator"].mock_eodhd_response(symbol, "historical")
                m.get(
                    f"https://eodhistoricaldata.com/api/eod/{symbol}.US",
                    payload=hist_response
                )
                
                # Step 1: Fetch quote data
                quote = await components["provider_manager"].get_stock_quote(symbol)
                
                # Step 2: Validate quote data
                quote_dict = {
                    "symbol": quote.symbol,
                    "last_price": quote.last_price,
                    "volume": quote.volume,
                    "bid": quote.bid,
                    "ask": quote.ask
                }
                assert components["validator"].validate_stock_quote(quote_dict) is True
                
                # Step 3: Calculate data quality score
                quality_score = components["quality_scorer"].calculate_quality_score(quote_dict)
                assert quality_score > 50  # Should be reasonable quality
                
                # Step 4: Fetch historical data for technical analysis
                historical_data = await components["provider_manager"].get_historical_data(symbol, days=30)
                
                # Step 5: Calculate technical indicators
                # Convert to DataFrame format
                import pandas as pd
                df_data = []
                for record in historical_data:
                    df_data.append({
                        "open": record["open"],
                        "high": record["high"],
                        "low": record["low"],
                        "close": record["close"],
                        "volume": record["volume"]
                    })
                
                df = pd.DataFrame(df_data)
                indicators = TechnicalIndicators.calculate_all_indicators(df)
                
                # Step 6: Perform momentum analysis
                momentum_result = components["momentum_analyzer"].calculate_momentum_score(df)
                
                # Step 7: Package data for analysis
                combined_data = {
                    "stock_quote": quote,
                    "technical_indicators": indicators,
                    "momentum_analysis": momentum_result,
                    "data_quality_score": quality_score
                }
                
                packaged_data = components["data_packager"].package_for_analysis(combined_data)
                
                # Verify complete pipeline
                assert "stock_analysis" in packaged_data
                assert "analysis_context" in packaged_data
                assert packaged_data["analysis_context"]["symbol"] == symbol
                
        finally:
            await components["provider_manager"].cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_options_data_pipeline(self, data_pipeline_components):
        """Test options-specific data pipeline."""
        components = data_pipeline_components
        symbol = "AAPL"
        
        await components["provider_manager"].initialize()
        
        try:
            with aioresponses() as m:
                # Mock options chain response
                options_data = components["test_data_generator"].generate_options_chain(symbol, 150.0)
                options_response = {
                    "s": "ok",
                    "symbol": symbol,
                    "options": options_data
                }
                
                # Mock provider response (assuming MarketData for options)
                components["provider_manager"].providers["marketdata"] = AsyncMock()
                components["provider_manager"].providers["marketdata"].get_options_chain.return_value = [
                    Mock(**opt) for opt in options_data
                ]
                
                # Step 1: Fetch options chain
                options_chain = await components["provider_manager"].get_options_chain(symbol)
                
                # Step 2: Validate options data
                for option in options_chain[:3]:  # Test first few
                    option_dict = {
                        "strike": float(option.strike),
                        "bid": float(option.bid),
                        "ask": float(option.ask),
                        "delta": option.delta,
                        "gamma": option.gamma,
                        "theta": option.theta,
                        "vega": option.vega
                    }
                    assert components["validator"].validate_option_contract(option_dict) is True
                
                # Step 3: Select best options
                best_calls = components["options_selector"].select_best_calls(options_data, top_n=3)
                
                # Step 4: Package options data
                options_package = components["data_packager"].package_options_data({
                    "underlying_symbol": symbol,
                    "options_chain": options_data,
                    "selection_results": {
                        "best_calls": best_calls,
                        "selection_criteria": "test_criteria"
                    }
                })
                
                # Verify options pipeline
                assert "underlying_symbol" in options_package
                assert options_package["underlying_symbol"] == symbol
                assert "options_analysis" in options_package
                assert "best_options" in options_package
                
        finally:
            await components["provider_manager"].cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_in_pipeline(self, data_pipeline_components):
        """Test error handling throughout the data pipeline."""
        components = data_pipeline_components
        symbol = "INVALID"
        
        await components["provider_manager"].initialize()
        
        try:
            with aioresponses() as m:
                # Mock error response
                m.get(
                    f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                    status=404,
                    payload={"error": "Symbol not found"}
                )
                
                # Pipeline should handle errors gracefully
                try:
                    quote = await components["provider_manager"].get_stock_quote(symbol)
                    assert False, "Should have raised an exception"
                except Exception as e:
                    # Error should be appropriate type
                    assert "not found" in str(e).lower() or "404" in str(e)
                
                # Test data validation with invalid data
                invalid_quote = {
                    "symbol": symbol,
                    "last_price": -100.0,  # Invalid negative price
                    "volume": -1000  # Invalid negative volume
                }
                
                try:
                    components["validator"].validate_stock_quote(invalid_quote)
                    assert False, "Should have raised validation error"
                except Exception as e:
                    assert "negative" in str(e).lower() or "positive" in str(e).lower()
                
        finally:
            await components["provider_manager"].cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_performance_monitoring(self, data_pipeline_components):
        """Test performance monitoring throughout pipeline."""
        components = data_pipeline_components
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        await components["provider_manager"].initialize()
        
        try:
            with aioresponses() as m:
                # Mock responses for all symbols
                for symbol in symbols:
                    quote_response = components["test_data_generator"].mock_eodhd_response(symbol, "quote")
                    m.get(
                        f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                        payload=quote_response
                    )
                
                # Track performance metrics
                start_time = datetime.now()
                pipeline_results = []
                
                for symbol in symbols:
                    symbol_start = datetime.now()
                    
                    # Fetch and validate quote
                    quote = await components["provider_manager"].get_stock_quote(symbol)
                    quote_dict = {
                        "symbol": quote.symbol,
                        "last_price": quote.last_price,
                        "volume": quote.volume,
                        "bid": quote.bid,
                        "ask": quote.ask
                    }
                    
                    # Validate and score
                    is_valid = components["validator"].validate_stock_quote(quote_dict)
                    quality_score = components["quality_scorer"].calculate_quality_score(quote_dict)
                    
                    symbol_time = (datetime.now() - symbol_start).total_seconds()
                    
                    pipeline_results.append({
                        "symbol": symbol,
                        "processing_time": symbol_time,
                        "data_valid": is_valid,
                        "quality_score": quality_score
                    })
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                # Verify performance
                assert total_time < 10  # Should complete quickly with mocks
                assert len(pipeline_results) == len(symbols)
                
                for result in pipeline_results:
                    assert result["processing_time"] < 5  # Individual symbol processing
                    assert result["data_valid"] is True
                    assert result["quality_score"] > 0
                
                # Calculate average processing time
                avg_time = sum(r["processing_time"] for r in pipeline_results) / len(pipeline_results)
                print(f"Average processing time per symbol: {avg_time:.3f}s")
                
        finally:
            await components["provider_manager"].cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cache_effectiveness_in_pipeline(self, data_pipeline_components):
        """Test cache effectiveness throughout the pipeline."""
        components = data_pipeline_components
        symbol = "AAPL"
        
        await components["provider_manager"].initialize()
        
        try:
            with aioresponses() as m:
                # Mock response for first request only
                quote_response = components["test_data_generator"].mock_eodhd_response(symbol, "quote")
                m.get(
                    f"https://eodhistoricaldata.com/api/real-time/{symbol}.US",
                    payload=quote_response
                )
                
                # First request - should hit provider
                start_time = datetime.now()
                quote1 = await components["provider_manager"].get_stock_quote(symbol)
                first_request_time = (datetime.now() - start_time).total_seconds()
                
                # Second request - should hit cache (faster)
                start_time = datetime.now()
                quote2 = await components["provider_manager"].get_stock_quote(symbol)
                second_request_time = (datetime.now() - start_time).total_seconds()
                
                # Cache hit should be faster than provider hit
                assert second_request_time < first_request_time
                
                # Data should be identical
                assert quote1.symbol == quote2.symbol
                assert quote1.last_price == quote2.last_price
                
                # Verify cache statistics
                cache_stats = await components["provider_manager"].get_cache_statistics()
                assert cache_stats["hit_rate"] > 0
                assert cache_stats["total_hits"] > 0
                
                print(f"First request: {first_request_time:.3f}s, Second request: {second_request_time:.3f}s")
                print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
                
        finally:
            await components["provider_manager"].cleanup()


class TestDataQualityPipeline:
    """Test data quality assurance throughout the pipeline."""
    
    @pytest.fixture
    def quality_pipeline(self):
        """Set up quality assurance pipeline."""
        validator = DataValidator()
        scorer = DataQualityScorer()
        
        return {
            "validator": validator,
            "scorer": scorer
        }
    
    @pytest.mark.integration
    def test_data_quality_scoring_pipeline(self, quality_pipeline, test_data_generator):
        """Test comprehensive data quality scoring."""
        # Generate test data with varying quality
        high_quality_data = test_data_generator.generate_stock_quote("AAPL", base_price=150.0)
        
        # Create low quality version
        low_quality_data = high_quality_data.copy()
        low_quality_data["volume"] = 0  # Suspicious zero volume
        low_quality_data["bid"] = 0  # Missing bid
        low_quality_data["ask"] = 0  # Missing ask
        
        # Score both datasets
        high_score = quality_pipeline["scorer"].calculate_quality_score(high_quality_data)
        low_score = quality_pipeline["scorer"].calculate_quality_score(low_quality_data)
        
        # High quality should score better
        assert high_score > low_score
        assert high_score >= 80  # Should be high quality
        assert low_score < 60   # Should be flagged as low quality
    
    @pytest.mark.integration
    def test_validation_pipeline_comprehensive(self, quality_pipeline, test_data_generator):
        """Test comprehensive validation pipeline."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        validation_results = []
        
        for symbol in symbols:
            # Generate stock quote
            quote_data = test_data_generator.generate_stock_quote(symbol)
            
            # Generate options chain
            options_chain = test_data_generator.generate_options_chain(symbol, quote_data["last_price"])
            
            # Generate technical indicators
            historical_data = test_data_generator.generate_historical_data(symbol, days=30)
            
            # Validate each component
            quote_valid = quality_pipeline["validator"].validate_stock_quote(quote_data)
            options_valid = quality_pipeline["validator"].validate_options_chain(options_chain)
            historical_valid = quality_pipeline["validator"].validate_historical_data([
                {
                    "date": row.name.strftime('%Y-%m-%d'),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"]
                }
                for _, row in historical_data.iterrows()
            ])
            
            validation_results.append({
                "symbol": symbol,
                "quote_valid": quote_valid,
                "options_valid": options_valid,
                "historical_valid": historical_valid,
                "overall_valid": quote_valid and options_valid and historical_valid
            })
        
        # All data should be valid for well-formed test data
        for result in validation_results:
            assert result["quote_valid"] is True
            assert result["options_valid"] is True
            assert result["historical_valid"] is True
            assert result["overall_valid"] is True
    
    @pytest.mark.integration
    def test_data_consistency_checks(self, quality_pipeline, test_data_generator):
        """Test data consistency validation across pipeline."""
        symbol = "AAPL"
        
        # Generate related data
        quote_data = test_data_generator.generate_stock_quote(symbol, base_price=150.0)
        options_chain = test_data_generator.generate_options_chain(symbol, 150.0)
        
        # Check consistency between quote and options
        spot_price = quote_data["last_price"]
        
        for option in options_chain[:5]:  # Check first 5 options
            # Option underlying should match quote symbol
            assert option["underlying_symbol"] == symbol
            
            # Option pricing should be reasonable relative to spot
            strike = float(option["strike"])
            option_price = float(option["last"])
            
            if option["option_type"] == "call":
                # Call intrinsic value should not exceed option price
                intrinsic_value = max(0, spot_price - strike)
                assert option_price >= intrinsic_value
            
            # Greeks should be reasonable
            assert 0 <= abs(option["delta"]) <= 1
            assert option["gamma"] >= 0
            assert option["vega"] >= 0
    
    @pytest.mark.integration
    def test_quality_threshold_enforcement(self, quality_pipeline, test_data_generator):
        """Test quality threshold enforcement in pipeline."""
        quality_threshold = 70.0
        symbols = ["AAPL", "MSFT", "GOOGL", "INVALID", "TEST"]
        
        passed_symbols = []
        failed_symbols = []
        
        for symbol in symbols:
            try:
                # Generate data (some may be lower quality for invalid symbols)
                if symbol in ["INVALID", "TEST"]:
                    # Create intentionally poor quality data
                    quote_data = {
                        "symbol": symbol,
                        "last_price": 0.01,  # Suspicious low price
                        "volume": 100,       # Low volume
                        "bid": 0,           # Missing bid
                        "ask": 0,           # Missing ask
                        "timestamp": (datetime.now() - timedelta(hours=48)).isoformat()  # Stale data
                    }
                else:
                    quote_data = test_data_generator.generate_stock_quote(symbol)
                
                # Calculate quality score
                quality_score = quality_pipeline["scorer"].calculate_quality_score(quote_data)
                
                if quality_score >= quality_threshold:
                    passed_symbols.append((symbol, quality_score))
                else:
                    failed_symbols.append((symbol, quality_score))
                    
            except Exception as e:
                failed_symbols.append((symbol, 0))  # Failed validation
        
        # Should have some passed and some failed
        assert len(passed_symbols) > 0
        assert len(failed_symbols) > 0
        
        # Passed symbols should meet threshold
        for symbol, score in passed_symbols:
            assert score >= quality_threshold
        
        # Failed symbols should be below threshold
        for symbol, score in failed_symbols:
            assert score < quality_threshold
        
        print(f"Passed quality check: {len(passed_symbols)} symbols")
        print(f"Failed quality check: {len(failed_symbols)} symbols")
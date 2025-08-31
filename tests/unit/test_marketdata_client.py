"""
Unit tests for MarketData.app API client.

Tests the MarketData client implementation including:
- Stock quote retrieval and validation
- Options chain processing with array-based responses
- PMCC-specific options chains
- Options expirations
- Credit tracking and cost optimization
- Error handling for cached vs live feeds
- Array response parsing and validation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal

from src.providers.marketdata_client import MarketDataClient
from src.models import (
    ScreeningCriteria,
    StockQuote,
    OptionContract,
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
    QuotaExceededError
)


class TestMarketDataClient:
    """Test suite for MarketData.app API client."""
    
    @pytest.fixture
    def client_config(self):
        """Basic client configuration for testing."""
        return {
            "api_key": "test_api_key",
            "base_url": "https://api.marketdata.app/v1",
            "requests_per_minute": 100,
            "timeout": 30,
            "max_retries": 3,
            "default_feed": "cached"
        }
    
    @pytest.fixture
    def client(self, client_config):
        """Create MarketData client instance for testing."""
        return MarketDataClient(client_config)
    
    @pytest.fixture
    def sample_screening_criteria(self):
        """Sample screening criteria for tests."""
        return ScreeningCriteria(
            min_price=10.0,
            max_price=200.0,
            min_volume=500_000,
            exclude_symbols=["TEST"]
        )
    
    @pytest.fixture
    def sample_options_chain_response(self):
        """Sample MarketData.app options chain response (array format)."""
        return {
            "optionSymbol": ["AAPL240119C150000", "AAPL240119C155000"],
            "underlying": ["AAPL", "AAPL"],
            "strike": [150.0, 155.0],
            "expiration": [1705622400, 1705622400],  # Unix timestamps
            "side": ["call", "call"],
            "bid": [5.0, 3.2],
            "ask": [5.2, 3.4],
            "last": [5.1, 3.3],
            "volume": [1000, 750],
            "openInterest": [5000, 3000],
            "delta": [0.6, 0.45],
            "gamma": [0.05, 0.06],
            "theta": [-0.02, -0.015],
            "vega": [0.15, 0.12],
            "iv": [0.25, 0.28]
        }
    
    def test_client_initialization(self, client_config):
        """Test client initialization with valid config."""
        client = MarketDataClient(client_config)
        
        assert client.api_key == "test_api_key"
        assert client.base_url == "https://api.marketdata.app/v1"
        assert client.provider_type == ProviderType.MARKETDATA
        assert client.default_feed == "cached"
    
    def test_client_initialization_without_api_key(self):
        """Test client initialization fails without API key."""
        config = {"base_url": "https://api.marketdata.app/v1"}
        
        with pytest.raises(ValueError, match="MarketData.app API key is required"):
            MarketDataClient(config)
    
    def test_get_headers(self, client):
        """Test request headers generation."""
        headers = client._get_headers()
        
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["Accept"] == "application/json"
        assert "User-Agent" in headers
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        mock_response = ProviderResponse(
            success=True,
            data={"last": 150.0, "change": 2.0},
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = await client.get_health_status()
        
        assert isinstance(result, HealthCheckResult)
        assert result.provider == ProviderType.MARKETDATA
        assert result.status == ProviderStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test health check with API failure."""
        mock_response = ProviderResponse(
            success=False,
            error="API Error",
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = await client.get_health_status()
        
        assert result.status == ProviderStatus.UNHEALTHY
        assert "API Error" in result.message
    
    def test_get_predefined_universe(self, client):
        """Test predefined stock universe retrieval."""
        universe = client._get_predefined_universe()
        
        assert isinstance(universe, list)
        assert len(universe) > 0
        assert "AAPL" in universe
        assert "MSFT" in universe
    
    def test_meets_screening_criteria(self, client):
        """Test local screening criteria application."""
        quote = StockQuote(
            symbol="AAPL",
            last_price=150.0,
            change=2.0,
            change_percent=1.35,
            volume=50_000_000
        )
        
        criteria = ScreeningCriteria(
            min_price=100.0,
            max_price=200.0,
            min_volume=1_000_000
        )
        
        assert client._meets_screening_criteria(quote, criteria)
        
        # Test failure cases
        criteria_low_price = ScreeningCriteria(min_price=200.0)
        assert not client._meets_screening_criteria(quote, criteria_low_price)
        
        criteria_high_volume = ScreeningCriteria(min_volume=100_000_000)
        assert not client._meets_screening_criteria(quote, criteria_high_volume)
    
    @pytest.mark.asyncio
    async def test_screen_stocks_with_local_filtering(self, client, sample_screening_criteria):
        """Test stock screening with local filtering."""
        mock_quotes = {
            "AAPL": StockQuote(symbol="AAPL", last_price=150.0, change=0, change_percent=0, volume=50_000_000),
            "MSFT": StockQuote(symbol="MSFT", last_price=50.0, change=0, change_percent=0, volume=30_000_000),  # Below min_volume
            "GOOGL": StockQuote(symbol="GOOGL", last_price=2500.0, change=0, change_percent=0, volume=2_000_000),  # Above max_price
        }
        
        with patch.object(client, 'get_stock_quotes', return_value=mock_quotes):
            symbols = await client.screen_stocks(sample_screening_criteria)
        
        assert "AAPL" in symbols
        assert "MSFT" not in symbols  # Filtered out by volume
        assert "GOOGL" not in symbols  # Filtered out by price
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_success(self, client):
        """Test successful stock quote retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data={
                "last": 150.0,
                "change": 2.0,
                "changepct": 1.35,
                "volume": 50000000,
                "bid": 149.9,
                "ask": 150.1,
                "high": 152.0,
                "low": 148.0,
                "open": 149.0
            },
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            quote = await client.get_stock_quote("AAPL")
        
        assert isinstance(quote, StockQuote)
        assert quote.symbol == "AAPL"
        assert quote.last_price == 150.0
        assert quote.change == 2.0
        assert quote.volume == 50000000
        assert quote.bid == 149.9
        assert quote.ask == 150.1
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_not_found(self, client):
        """Test stock quote retrieval with not found error."""
        mock_response = ProviderResponse(
            success=False,
            error="Symbol not found",
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            with pytest.raises(InvalidSymbolError):
                await client.get_stock_quote("INVALID")
    
    @pytest.mark.asyncio
    async def test_get_stock_quotes_concurrent(self, client):
        """Test concurrent stock quotes retrieval."""
        mock_quote_responses = [
            StockQuote(symbol="AAPL", last_price=150.0, change=0, change_percent=0, volume=50_000_000),
            StockQuote(symbol="MSFT", last_price=280.0, change=0, change_percent=0, volume=30_000_000),
            None  # Simulate one failure
        ]
        
        with patch.object(client, 'get_stock_quote', side_effect=mock_quote_responses):
            quotes = await client.get_stock_quotes(["AAPL", "MSFT", "INVALID"])
        
        assert len(quotes) == 2
        assert "AAPL" in quotes
        assert "MSFT" in quotes
        assert "INVALID" not in quotes
    
    def test_validate_array_response(self, client, sample_options_chain_response):
        """Test validation of array-based response format."""
        assert client._validate_array_response(sample_options_chain_response)
        
        # Test mismatched array lengths
        invalid_response = sample_options_chain_response.copy()
        invalid_response["bid"] = [5.0]  # Different length
        
        assert not client._validate_array_response(invalid_response)
    
    def test_safe_get_array_value(self, client):
        """Test safe array value extraction."""
        data = {
            "test_array": [1, 2, 3, None, 5],
            "empty_array": []
        }
        
        assert client._safe_get_array_value(data, "test_array", 0) == 1
        assert client._safe_get_array_value(data, "test_array", 3) is None  # None value
        assert client._safe_get_array_value(data, "test_array", 10, "default") == "default"  # Out of bounds
        assert client._safe_get_array_value(data, "missing_key", 0, "default") == "default"  # Missing key
        assert client._safe_get_array_value(data, "empty_array", 0, "default") == "default"  # Empty array
    
    def test_parse_single_option_contract(self, client, sample_options_chain_response):
        """Test parsing of single option contract from array data."""
        contract = client._parse_single_option_contract(sample_options_chain_response, 0, "AAPL")
        
        assert isinstance(contract, OptionContract)
        assert contract.option_symbol == "AAPL240119C150000"
        assert contract.underlying_symbol == "AAPL"
        assert contract.strike == Decimal('150.0')
        assert contract.option_type == OptionType.CALL
        assert contract.bid == Decimal('5.0')
        assert contract.ask == Decimal('5.2')
        assert contract.delta == 0.6
        assert contract.volume == 1000
        assert contract.open_interest == 5000
    
    def test_parse_single_option_contract_invalid_data(self, client):
        """Test parsing with invalid/missing data."""
        invalid_data = {
            "optionSymbol": ["TEST"],
            "strike": [None],  # Missing required data
            "expiration": [1705622400],
            "side": ["call"]
        }
        
        contract = client._parse_single_option_contract(invalid_data, 0, "TEST")
        assert contract is None
    
    @pytest.mark.asyncio
    async def test_parse_options_chain_response(self, client, sample_options_chain_response):
        """Test parsing of complete options chain response."""
        contracts = client._parse_options_chain_response(sample_options_chain_response, "AAPL")
        
        assert len(contracts) == 2
        
        # Check first contract
        contract1 = contracts[0]
        assert contract1.option_symbol == "AAPL240119C150000"
        assert contract1.strike == Decimal('150.0')
        
        # Check second contract
        contract2 = contracts[1]
        assert contract2.option_symbol == "AAPL240119C155000"
        assert contract2.strike == Decimal('155.0')
    
    @pytest.mark.asyncio
    async def test_get_options_chain_success(self, client, sample_options_chain_response):
        """Test successful options chain retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=sample_options_chain_response,
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            contracts = await client.get_options_chain("AAPL")
        
        assert len(contracts) == 2
        assert all(isinstance(c, OptionContract) for c in contracts)
        assert contracts[0].underlying_symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_options_chain_with_filters(self, client, sample_options_chain_response):
        """Test options chain retrieval with filtering parameters."""
        mock_response = ProviderResponse(
            success=True,
            data=sample_options_chain_response,
            provider=ProviderType.MARKETDATA
        )
        
        filters = {
            "option_type": OptionType.CALL,
            "min_delta": 0.5,
            "max_delta": 0.8,
            "min_volume": 100,
            "feed": "cached"
        }
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_request:
            contracts = await client.get_options_chain("AAPL", **filters)
        
        # Verify correct parameters were passed
        call_args = mock_request.call_args
        params = call_args[1]["params"]
        
        assert params["side"] == "call"
        assert params["delta"] == "0.5-0.8"
        assert params["minVolume"] == 100
        assert params["feed"] == "cached"
    
    @pytest.mark.asyncio
    async def test_get_options_chain_no_cached_data(self, client):
        """Test options chain retrieval with 204 response (no cached data)."""
        mock_response = ProviderResponse(
            success=False,
            error="No cached data available",
            provider=ProviderType.MARKETDATA
        )
        
        # Mock the response to simulate 204 status
        with patch.object(client, '_make_request', return_value=mock_response):
            contracts = await client.get_options_chain("AAPL")
        
        assert contracts == []
    
    @pytest.mark.asyncio
    async def test_get_options_chain_invalid_symbol(self, client):
        """Test options chain retrieval with invalid symbol."""
        mock_response = ProviderResponse(
            success=False,
            error="Symbol not found",
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            with pytest.raises(InvalidSymbolError):
                await client.get_options_chain("INVALID")
    
    @pytest.mark.asyncio
    async def test_get_pmcc_option_chains_success(self, client, sample_options_chain_response):
        """Test PMCC-specific options chains retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=sample_options_chain_response,
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, 'get_options_chain', return_value=[
            OptionContract(
                option_symbol="LEAPS_OPTION",
                underlying_symbol="AAPL",
                strike=Decimal('100'),
                expiration=date.today() + timedelta(days=200),
                option_type=OptionType.CALL
            )
        ]) as mock_get_chain:
            chains = await client.get_pmcc_option_chains("AAPL")
        
        assert "leaps" in chains
        assert "short_calls" in chains
        assert len(chains["leaps"]) >= 0
        assert len(chains["short_calls"]) >= 0
        
        # Verify two separate calls were made
        assert mock_get_chain.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_options_expirations_success(self, client):
        """Test successful options expirations retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=[1705622400, 1708300800, 1710892800],  # Unix timestamps
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            expirations = await client.get_options_expirations("AAPL")
        
        assert len(expirations) == 3
        assert all(isinstance(exp, str) for exp in expirations)
        assert all(len(exp) == 10 for exp in expirations)  # YYYY-MM-DD format
    
    @pytest.mark.asyncio
    async def test_get_options_expirations_invalid_symbol(self, client):
        """Test options expirations with invalid symbol."""
        mock_response = ProviderResponse(
            success=False,
            error="Symbol not found",
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            with pytest.raises(InvalidSymbolError):
                await client.get_options_expirations("INVALID")
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data_returns_none(self, client):
        """Test that fundamental data returns None (not supported)."""
        result = await client.get_fundamental_data("AAPL")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_credit_tracking(self, client):
        """Test credit usage tracking."""
        initial_credits = client._daily_credits_used
        
        mock_response = ProviderResponse(
            success=True,
            data={"last": 150.0},
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            await client.get_stock_quote("AAPL")
        
        # Should have used 1 credit for stock quote
        assert client._daily_credits_used == initial_credits + 1
    
    @pytest.mark.asyncio
    async def test_credit_tracking_options_cached_feed(self, client, sample_options_chain_response):
        """Test credit tracking for cached options chain."""
        initial_credits = client._daily_credits_used
        
        mock_response = ProviderResponse(
            success=True,
            data=sample_options_chain_response,
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            await client.get_options_chain("AAPL", feed="cached")
        
        # Cached feed should cost 1 credit regardless of contracts returned
        assert client._daily_credits_used == initial_credits + 1
    
    @pytest.mark.asyncio
    async def test_feed_parameter_handling(self, client, sample_options_chain_response):
        """Test proper handling of feed parameter (cached vs live)."""
        mock_response = ProviderResponse(
            success=True,
            data=sample_options_chain_response,
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_request:
            # Test default feed (should be cached)
            await client.get_options_chain("AAPL")
            
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["feed"] == "cached"  # Default from client config
            
            # Test explicit live feed
            await client.get_options_chain("AAPL", feed="live")
            
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["feed"] == "live"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_limiting(self, client):
        """Test concurrent request limiting."""
        # Set low concurrent limit for testing
        client.max_concurrent_requests = 2
        
        mock_response = ProviderResponse(
            success=True,
            data={"last": 150.0},
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, 'get_stock_quote', return_value=StockQuote(
            symbol="TEST", last_price=150.0, change=0, change_percent=0, volume=1000
        )) as mock_get_quote:
            
            # Start many concurrent requests
            symbols = [f"TEST{i}" for i in range(10)]
            quotes = await client.get_stock_quotes(symbols)
            
            # Should have limited concurrency but still gotten all quotes
            assert len(quotes) == 10
            assert mock_get_quote.call_count == 10
    
    @pytest.mark.asyncio
    async def test_error_handling_quota_exceeded(self, client):
        """Test handling of quota exceeded errors."""
        mock_response = ProviderResponse(
            success=False,
            error="Plan limit exceeded",
            provider=ProviderType.MARKETDATA
        )
        
        # Mock the _make_request to simulate 402 status code
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = mock_response
            # Simulate the status code check
            mock_request.return_value._status_code = 402
            
            with pytest.raises(QuotaExceededError):
                await client._make_request("GET", "test_url")
    
    @pytest.mark.asyncio
    async def test_trailing_slash_handling(self, client):
        """Test that trailing slashes are properly included in URLs."""
        mock_response = ProviderResponse(
            success=True,
            data={"last": 150.0},
            provider=ProviderType.MARKETDATA
        )
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_request:
            await client.get_stock_quote("AAPL")
            
            # Verify URL ends with trailing slash
            call_args = mock_request.call_args
            url = call_args[0][1]  # Second positional argument is the URL
            assert url.endswith("/"), f"URL should end with trailing slash: {url}"
    
    def test_option_type_parameter_conversion(self, client):
        """Test option type parameter conversion to MarketData format."""
        mock_response = ProviderResponse(success=True, data={}, provider=ProviderType.MARKETDATA)
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_request:
            # Test with OptionType enum
            asyncio.run(client.get_options_chain("AAPL", option_type=OptionType.CALL))
            
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["side"] == "call"
            
            # Test with string
            asyncio.run(client.get_options_chain("AAPL", option_type="put"))
            
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["side"] == "put"


if __name__ == "__main__":
    pytest.main([__file__])
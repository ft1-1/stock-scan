"""
Unit tests for EODHD API client.

Tests the EODHD client implementation including:
- Stock screening functionality
- Quote retrieval and validation
- Options chain processing
- Fundamental data parsing
- Technical indicators calculation
- Error handling and resilience
- Rate limiting and circuit breaking
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, date, timedelta
from decimal import Decimal

from src.providers.eodhd_client import EODHDClient
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
    DataQualityError,
    QuotaExceededError
)


class TestEODHDClient:
    """Test suite for EODHD API client."""
    
    @pytest.fixture
    def client_config(self):
        """Basic client configuration for testing."""
        return {
            "api_key": "test_api_key",
            "base_url": "https://eodhd.com/api",
            "requests_per_minute": 60,
            "timeout": 30,
            "max_retries": 3
        }
    
    @pytest.fixture
    def client(self, client_config):
        """Create EODHD client instance for testing."""
        return EODHDClient(client_config)
    
    @pytest.fixture
    def sample_screening_criteria(self):
        """Sample screening criteria for tests."""
        return ScreeningCriteria(
            min_market_cap=50_000_000,
            max_market_cap=5_000_000_000,
            min_price=5.0,
            max_price=500.0,
            min_volume=100_000,
            exclude_sectors=["Utilities"],
            exclude_symbols=["TEST"]
        )
    
    def test_client_initialization(self, client_config):
        """Test client initialization with valid config."""
        client = EODHDClient(client_config)
        
        assert client.api_key == "test_api_key"
        assert client.base_url == "https://eodhd.com/api"
        assert client.provider_type == ProviderType.EODHD
    
    def test_client_initialization_without_api_key(self):
        """Test client initialization fails without API key."""
        config = {"base_url": "https://eodhd.com/api"}
        
        with pytest.raises(ValueError, match="EODHD API key is required"):
            EODHDClient(config)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        mock_response = ProviderResponse(
            success=True,
            data={"data": []},
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = await client.get_health_status()
        
        assert isinstance(result, HealthCheckResult)
        assert result.provider == ProviderType.EODHD
        assert result.status == ProviderStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test health check with API failure."""
        mock_response = ProviderResponse(
            success=False,
            error="API Error",
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = await client.get_health_status()
        
        assert result.status == ProviderStatus.UNHEALTHY
        assert "API Error" in result.message
    
    @pytest.mark.asyncio
    async def test_build_screening_filters(self, client, sample_screening_criteria):
        """Test building EODHD filter array from criteria."""
        filters = client._build_screening_filters(sample_screening_criteria)
        
        # Check that filters are properly constructed
        assert ["market_capitalization", ">=", 50_000_000] in filters
        assert ["market_capitalization", "<=", 5_000_000_000] in filters
        assert ["adjusted_close", ">=", 5.0] in filters
        assert ["adjusted_close", "<=", 500.0] in filters
        assert ["avgvol_200d", ">=", 100_000] in filters
        assert ["exchange", "=", "us"] in filters
        assert ["sector", "!=", "Utilities"] in filters
        assert ["code", "!=", "TEST"] in filters
    
    @pytest.mark.asyncio
    async def test_screen_stocks_single_success(self, client, sample_screening_criteria):
        """Test successful single screening query."""
        mock_response = ProviderResponse(
            success=True,
            data={
                "data": [
                    {"code": "AAPL", "market_capitalization": 2_000_000_000},
                    {"code": "MSFT", "market_capitalization": 2_500_000_000}
                ]
            },
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            symbols = await client.screen_stocks(sample_screening_criteria)
        
        assert symbols == ["AAPL", "MSFT"]
    
    @pytest.mark.asyncio
    async def test_screen_stocks_comprehensive_range_splitting(self, client):
        """Test comprehensive screening with range splitting."""
        criteria = ScreeningCriteria(
            min_market_cap=50_000_000,
            max_market_cap=10_000_000_000  # Large range triggers splitting
        )
        
        mock_responses = [
            {"code": "AAPL", "market_capitalization": 100_000_000},
            {"code": "MSFT", "market_capitalization": 2_000_000_000}
        ]
        
        with patch.object(client, '_screen_stocks_single', side_effect=mock_responses):
            with patch.object(client, '_generate_market_cap_ranges', return_value=[(50_000_000, 1_000_000_000), (1_000_000_000, 10_000_000_000)]):
                symbols = await client.screen_stocks(criteria)
        
        assert "AAPL" in symbols
        assert "MSFT" in symbols
    
    def test_generate_market_cap_ranges(self, client):
        """Test market cap range generation."""
        ranges = client._generate_market_cap_ranges(50_000_000, 1_000_000_000)
        
        assert len(ranges) > 0
        assert ranges[0][0] == 50_000_000
        assert ranges[-1][1] == 1_000_000_000
        
        # Ensure no gaps or overlaps
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_success(self, client):
        """Test successful stock quote retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=[
                {
                    "date": "2025-01-15",
                    "adjusted_close": 150.0,
                    "volume": 50000000,
                    "high": 152.0,
                    "low": 148.0,
                    "open": 149.0
                },
                {
                    "date": "2025-01-14",
                    "adjusted_close": 148.0,
                    "volume": 45000000,
                    "high": 150.0,
                    "low": 147.0,
                    "open": 148.5
                }
            ],
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            quote = await client.get_stock_quote("AAPL")
        
        assert isinstance(quote, StockQuote)
        assert quote.symbol == "AAPL"
        assert quote.last_price == 150.0
        assert quote.volume == 50000000
        assert quote.change == 2.0  # 150.0 - 148.0
        assert abs(quote.change_percent - 1.35) < 0.1  # (2/148)*100
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_no_data(self, client):
        """Test stock quote retrieval with no data."""
        mock_response = ProviderResponse(
            success=True,
            data=[],
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            quote = await client.get_stock_quote("INVALID")
        
        assert quote is None
    
    @pytest.mark.asyncio
    async def test_get_stock_quotes_bulk_success(self, client):
        """Test successful bulk stock quotes retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=[
                {
                    "code": "AAPL",
                    "date": "2025-01-15",
                    "adjusted_close": 150.0,
                    "change": 2.0,
                    "change_p": 1.35,
                    "volume": 50000000,
                    "high": 152.0,
                    "low": 148.0,
                    "open": 149.0
                },
                {
                    "code": "MSFT",
                    "date": "2025-01-15",
                    "adjusted_close": 280.0,
                    "change": -1.5,
                    "change_p": -0.53,
                    "volume": 30000000,
                    "high": 282.0,
                    "low": 278.0,
                    "open": 281.0
                }
            ],
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            quotes = await client.get_stock_quotes(["AAPL", "MSFT"])
        
        assert len(quotes) == 2
        assert "AAPL" in quotes
        assert "MSFT" in quotes
        assert quotes["AAPL"].last_price == 150.0
        assert quotes["MSFT"].last_price == 280.0
    
    def test_get_tradetime_filter(self, client):
        """Test tradetime filter generation."""
        tradetime_filter = client._get_tradetime_filter(5)
        
        # Should be a valid date string
        assert isinstance(tradetime_filter, str)
        assert len(tradetime_filter) == 10  # YYYY-MM-DD format
        
        # Should be within the last week
        filter_date = datetime.strptime(tradetime_filter, '%Y-%m-%d').date()
        today = datetime.now().date()
        assert (today - filter_date).days <= 7
    
    @pytest.mark.asyncio
    async def test_get_options_chain_success(self, client):
        """Test successful options chain retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data={
                "data": [
                    {
                        "option_id": "AAPL240119C150000",
                        "type": "call",
                        "strike": 150.0,
                        "exp_date": "2024-01-19",
                        "bid": 5.0,
                        "ask": 5.2,
                        "last": 5.1,
                        "volume": 1000,
                        "open_interest": 5000,
                        "delta": 0.6,
                        "gamma": 0.05,
                        "theta": -0.02,
                        "vega": 0.15,
                        "implied_volatility": 0.25
                    }
                ]
            },
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            contracts = await client.get_options_chain("AAPL")
        
        assert len(contracts) == 1
        contract = contracts[0]
        assert isinstance(contract, OptionContract)
        assert contract.underlying_symbol == "AAPL"
        assert contract.option_type == OptionType.CALL
        assert contract.strike == Decimal('150.0')
        assert contract.delta == 0.6
    
    @pytest.mark.asyncio
    async def test_parse_option_contract_invalid_data(self, client):
        """Test option contract parsing with invalid data."""
        invalid_data = {
            "type": "call",
            "strike": "invalid",
            "exp_date": "2024-01-19"
        }
        
        contract = client._parse_option_contract(invalid_data, "AAPL")
        assert contract is None
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data_success(self, client):
        """Test successful fundamental data retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data={
                "General": {"Name": "Apple Inc", "Sector": "Technology"},
                "Valuation": {
                    "MarketCapitalization": 2000000000,
                    "TrailingPE": 25.5,
                    "PriceBookMRQ": 6.2
                },
                "Highlights": {
                    "ReturnOnEquityTTM": 0.15,
                    "ProfitMargin": 0.25,
                    "DividendYield": 0.006
                }
            },
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            data = await client.get_fundamental_data("AAPL")
        
        assert isinstance(data, FundamentalData)
        assert data.symbol == "AAPL"
        assert data.market_cap == 2000000000
        assert data.pe_ratio == 25.5
        assert data.roe == 0.15
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators_success(self, client):
        """Test successful technical indicators retrieval."""
        mock_rsi_response = [{"rsi": 65.0}]
        mock_sma_response = [{"sma": 145.0}]
        
        with patch.object(client, '_get_technical_indicator', side_effect=[
            mock_rsi_response,  # RSI
            mock_sma_response,  # SMA 20
            mock_sma_response,  # SMA 50
            mock_sma_response,  # SMA 200
            None,              # Bollinger Bands
            [{"atr": 2.5}]     # ATR
        ]):
            indicators = await client.get_technical_indicators("AAPL")
        
        assert isinstance(indicators, TechnicalIndicators)
        assert indicators.symbol == "AAPL"
        assert indicators.rsi_14 == 65.0
        assert indicators.sma_20 == 145.0
        assert indicators.atr_14 == 2.5
    
    @pytest.mark.asyncio
    async def test_earnings_calendar_success(self, client):
        """Test successful earnings calendar retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=[
                {"symbol": "AAPL", "date": "2025-01-20", "eps_estimate": 1.50},
                {"symbol": "MSFT", "date": "2025-01-21", "eps_estimate": 2.25}
            ],
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            calendar = await client.get_earnings_calendar(
                date(2025, 1, 20),
                date(2025, 1, 25)
            )
        
        assert len(calendar) == 2
        assert calendar[0]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_news_success(self, client):
        """Test successful news retrieval."""
        mock_response = ProviderResponse(
            success=True,
            data=[
                {"title": "Apple Reports Strong Q4", "date": "2025-01-15"},
                {"title": "New iPhone Launch", "date": "2025-01-14"}
            ],
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            news = await client.get_news("AAPL", limit=2)
        
        assert len(news) == 2
        assert "Apple Reports Strong Q4" in news[0]["title"]
    
    @pytest.mark.asyncio
    async def test_error_handling_authentication_error(self, client):
        """Test handling of authentication errors."""
        mock_response = ProviderResponse(
            success=False,
            data={"code": 403, "message": "Invalid API key"},
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            with pytest.raises(AuthenticationError):
                await client._make_request("GET", "test_url")
    
    @pytest.mark.asyncio
    async def test_error_handling_quota_exceeded(self, client):
        """Test handling of quota exceeded errors."""
        mock_response = ProviderResponse(
            success=False,
            data={"code": 402, "message": "Plan limit exceeded"},
            provider=ProviderType.EODHD
        )
        
        with patch.object(client, '_make_request', return_value=mock_response):
            with pytest.raises(QuotaExceededError):
                await client._make_request("GET", "test_url")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, client):
        """Test rate limiting functionality."""
        # Set a very low rate limit for testing
        client.rate_limiter.state.requests_per_window = 2
        client.rate_limiter.state.window_duration = 1
        
        # Make requests quickly
        start_time = asyncio.get_event_loop().time()
        
        mock_response = ProviderResponse(success=True, data={}, provider=ProviderType.EODHD)
        
        with patch.object(client, '_make_request', return_value=mock_response):
            # First two requests should succeed quickly
            await client._make_request("GET", "test1")
            await client._make_request("GET", "test2")
            
            # Third request should trigger rate limiting
            with pytest.raises(RateLimitError):
                await client._make_request("GET", "test3")
    
    def test_market_cap_ranges_no_gaps(self, client):
        """Test that market cap ranges have no gaps."""
        ranges = client._generate_market_cap_ranges(50_000_000, 5_000_000_000)
        
        for i in range(len(ranges) - 1):
            current_end = ranges[i][1]
            next_start = ranges[i + 1][0]
            assert current_end == next_start, f"Gap found between ranges: {ranges[i]} and {ranges[i + 1]}"
    
    def test_market_cap_ranges_covers_full_range(self, client):
        """Test that market cap ranges cover the full requested range."""
        min_cap = 50_000_000
        max_cap = 5_000_000_000
        
        ranges = client._generate_market_cap_ranges(min_cap, max_cap)
        
        assert ranges[0][0] == min_cap
        assert ranges[-1][1] == max_cap
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, client):
        """Test that circuit breaker opens after consecutive failures."""
        # Configure circuit breaker for testing
        client.circuit_breaker.failure_threshold = 2
        
        # Simulate failures
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = Exception("Connection error")
            
            # First failure
            result1 = await client._make_request("GET", "test")
            assert not result1.success
            
            # Second failure - should trigger circuit breaker
            result2 = await client._make_request("GET", "test")
            assert not result2.success
            
            # Third request should be blocked by circuit breaker
            with pytest.raises(CircuitBreakerOpenError):
                await client._make_request("GET", "test")


if __name__ == "__main__":
    pytest.main([__file__])
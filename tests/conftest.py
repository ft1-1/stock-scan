"""Pytest configuration and shared fixtures."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

# Test-specific imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import Settings
from models import (
    ScreeningCriteria,
    StockQuote,
    OptionContract,
    TechnicalIndicators,
    WorkflowConfig,
    WorkflowExecutionContext
)
from providers.base_provider import BaseProvider


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Provide test-specific settings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        return Settings(
            environment="testing",
            debug=True,
            log_level="DEBUG",
            eodhd_api_key="test_eodhd_key",
            marketdata_api_key="test_marketdata_key", 
            claude_api_key="test_claude_key",
            output_directory=f"{temp_dir}/output",
            log_directory=f"{temp_dir}/logs",
            cache_directory=f"{temp_dir}/cache",
            test_mode=True,
            test_symbol_limit=10,
            max_stocks_to_screen=100,
            enable_caching=False,
            enable_checkpointing=False
        )


@pytest.fixture
def sample_screening_criteria() -> ScreeningCriteria:
    """Provide sample screening criteria for testing."""
    return ScreeningCriteria(
        min_market_cap=1_000_000_000,
        max_market_cap=100_000_000_000,
        min_price=10.0,
        max_price=500.0,
        min_volume=500_000,
        min_rsi=30.0,
        max_rsi=70.0,
        min_option_volume=100,
        min_open_interest=500,
        max_days_to_expiration=45,
        exclude_sectors=["Utilities"],
        exclude_symbols=["TSLA"]  # Exclude for testing
    )


@pytest.fixture
def sample_stock_quote() -> StockQuote:
    """Provide sample stock quote for testing."""
    return StockQuote(
        symbol="AAPL",
        last_price=150.50,
        change=2.25,
        change_percent=1.52,
        volume=50_000_000,
        bid=150.45,
        ask=150.55,
        high=152.00,
        low=149.00,
        open=150.00
    )


@pytest.fixture
def sample_option_contract() -> OptionContract:
    """Provide sample option contract for testing."""
    from decimal import Decimal
    from datetime import date, timedelta
    
    return OptionContract(
        option_symbol="AAPL250117C00150000",
        underlying_symbol="AAPL",
        strike=Decimal("150.00"),
        expiration=date.today() + timedelta(days=30),
        option_type="call",
        bid=Decimal("5.20"),
        ask=Decimal("5.30"),
        last=Decimal("5.25"),
        volume=1500,
        open_interest=8500,
        delta=0.55,
        gamma=0.025,
        theta=-0.08,
        vega=0.35,
        implied_volatility=0.28,
        days_to_expiration=30
    )


@pytest.fixture
def sample_technical_indicators() -> TechnicalIndicators:
    """Provide sample technical indicators for testing."""
    return TechnicalIndicators(
        symbol="AAPL",
        sma_20=148.50,
        sma_50=145.20,
        sma_200=140.80,
        ema_12=149.75,
        ema_26=147.30,
        rsi_14=58.5,
        macd=2.45,
        macd_signal=1.80,
        macd_histogram=0.65,
        bollinger_upper=155.20,
        bollinger_middle=148.50,
        bollinger_lower=141.80,
        atr_14=3.25,
        volume_sma_20=45_000_000,
        volume_ratio=1.12
    )


@pytest.fixture
def workflow_config() -> WorkflowConfig:
    """Provide test workflow configuration."""
    return WorkflowConfig(
        max_concurrent_stocks=10,
        max_retry_attempts=2,
        step_timeout_seconds=60,
        continue_on_errors=True,
        error_threshold_percent=20.0,
        enable_ai_analysis=False,
        ai_batch_size=5,
        max_ai_cost_dollars=10.0,
        save_intermediate_results=True,
        min_data_quality_score=70.0
    )


@pytest.fixture
def workflow_context(workflow_config) -> WorkflowExecutionContext:
    """Provide test workflow execution context."""
    return WorkflowExecutionContext(
        workflow_id="test-workflow-123",
        config=workflow_config
    )


@pytest.fixture
def mock_provider() -> Mock:
    """Provide a mock provider for testing."""
    provider = AsyncMock(spec=BaseProvider)
    provider.provider_type = "mock"
    provider.get_health_status = AsyncMock()
    provider.get_usage_stats = AsyncMock()
    provider._request_count = 0
    provider._total_cost = 0.0
    return provider


@pytest.fixture
def mock_eodhd_response() -> dict:
    """Provide mock EODHD API response."""
    return {
        "code": "AAPL.US",
        "timestamp": 1640995200,
        "gmtoffset": 0,
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 150.5,
        "volume": 50000000,
        "previousClose": 148.25,
        "change": 2.25,
        "change_p": 1.52
    }


@pytest.fixture
def mock_marketdata_response() -> dict:
    """Provide mock MarketData.app API response."""
    return {
        "s": "ok",
        "symbol": ["AAPL"],
        "last": [150.5],
        "change": [2.25],
        "changepct": [1.52],
        "volume": [50000000],
        "updated": [1640995200]
    }


@pytest.fixture
def mock_options_chain() -> list:
    """Provide mock options chain data."""
    return [
        {
            "contractSymbol": "AAPL250117C00150000",
            "strike": 150.0,
            "currency": "USD",
            "lastPrice": 5.25,
            "change": 0.15,
            "percentChange": 2.94,
            "volume": 1500,
            "openInterest": 8500,
            "bid": 5.20,
            "ask": 5.30,
            "contractSize": "REGULAR",
            "expiration": 1737158400,
            "lastTradeDate": 1640995200,
            "impliedVolatility": 0.28,
            "inTheMoney": True
        }
    ]


@pytest.fixture
async def mock_http_session():
    """Provide mock HTTP session for testing."""
    from aiohttp import ClientSession
    from aioresponses import aioresponses
    
    with aioresponses() as m:
        yield m


@pytest.fixture
def sample_test_data_dir(tmp_path) -> Path:
    """Create temporary directory with sample test data."""
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    
    # Create sample CSV files
    (test_data_dir / "sample_stocks.csv").write_text(
        "symbol,price,volume,market_cap\n"
        "AAPL,150.50,50000000,2500000000000\n"
        "MSFT,300.25,25000000,2200000000000\n"
        "GOOGL,2800.75,1500000,1800000000000\n"
    )
    
    (test_data_dir / "sample_options.csv").write_text(
        "symbol,strike,expiration,option_type,volume,open_interest\n"
        "AAPL,150.0,2025-01-17,call,1500,8500\n"
        "MSFT,300.0,2025-01-17,call,800,4200\n"
        "GOOGL,2800.0,2025-01-17,call,300,1500\n"
    )
    
    return test_data_dir


# Async fixtures
@pytest.fixture
async def async_mock_provider() -> AsyncGenerator[AsyncMock, None]:
    """Provide async mock provider."""
    provider = AsyncMock(spec=BaseProvider)
    provider.initialize = AsyncMock()
    provider.cleanup = AsyncMock()
    
    async with provider:
        yield provider


# Pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "api: Tests requiring API access")
    config.addinivalue_line("markers", "ai: Tests requiring AI/LLM access")


# Test data generators
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_stock_symbols(count: int = 10) -> list[str]:
        """Generate list of test stock symbols."""
        return [f"TEST{i:03d}" for i in range(count)]
    
    @staticmethod
    def generate_price_data(symbol: str, days: int = 30) -> list[dict]:
        """Generate historical price data for testing."""
        import random
        from datetime import date, timedelta
        
        data = []
        base_price = random.uniform(50, 300)
        
        for i in range(days):
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            data.append({
                "symbol": symbol,
                "date": (date.today() - timedelta(days=days-i)).isoformat(),
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.97,
                "close": price,
                "volume": random.randint(100_000, 10_000_000)
            })
            base_price = price
            
        return data
    
    @staticmethod
    def generate_options_chain(symbol: str, spot_price: float, days_to_exp: int = 30) -> list[dict]:
        """Generate realistic options chain data."""
        import random
        from datetime import date, timedelta
        from decimal import Decimal
        
        chain = []
        exp_date = date.today() + timedelta(days=days_to_exp)
        
        # Generate strikes around current price
        strikes = [spot_price * (1 + i * 0.05) for i in range(-10, 11)]
        
        for strike in strikes:
            # Calculate realistic Greeks and prices
            moneyness = spot_price / strike
            time_value = max(0.01, days_to_exp / 365.0)
            iv = random.uniform(0.15, 0.45)
            
            # Basic Black-Scholes approximation for testing
            intrinsic = max(0, spot_price - strike)
            option_price = intrinsic + (strike * iv * (time_value ** 0.5) * 0.4)
            
            delta = max(0.01, min(0.99, moneyness ** 0.5)) if moneyness > 0.8 else 0.01
            gamma = 0.5 / (spot_price * iv * (time_value ** 0.5)) if time_value > 0 else 0.01
            theta = -option_price * 0.05 / days_to_exp if days_to_exp > 0 else -0.01
            vega = spot_price * (time_value ** 0.5) * 0.01
            
            chain.append({
                "option_symbol": f"{symbol}{exp_date.strftime('%y%m%d')}C{int(strike*1000):08d}",
                "underlying_symbol": symbol,
                "strike": Decimal(str(round(strike, 2))),
                "expiration": exp_date,
                "option_type": "call",
                "bid": Decimal(str(round(max(0.01, option_price - 0.05), 2))),
                "ask": Decimal(str(round(option_price + 0.05, 2))),
                "last": Decimal(str(round(option_price, 2))),
                "volume": random.randint(0, 5000),
                "open_interest": random.randint(100, 20000),
                "delta": round(delta, 3),
                "gamma": round(gamma, 3),
                "theta": round(theta, 3),
                "vega": round(vega, 3),
                "implied_volatility": round(iv, 3),
                "days_to_expiration": days_to_exp
            })
        
        return chain
    
    @staticmethod
    def generate_market_conditions(scenario: str = "normal") -> dict:
        """Generate market condition data for different scenarios."""
        import random
        
        scenarios = {
            "bull": {"trend": 1.5, "volatility": 0.15, "volume_mult": 1.2},
            "bear": {"trend": -1.8, "volatility": 0.25, "volume_mult": 1.5},
            "sideways": {"trend": 0.1, "volatility": 0.12, "volume_mult": 0.8},
            "volatile": {"trend": 0.5, "volatility": 0.35, "volume_mult": 2.0},
            "normal": {"trend": 0.8, "volatility": 0.18, "volume_mult": 1.0}
        }
        
        config = scenarios.get(scenario, scenarios["normal"])
        
        return {
            "market_trend": config["trend"],
            "volatility_regime": config["volatility"],
            "volume_multiplier": config["volume_mult"],
            "risk_free_rate": random.uniform(0.01, 0.05),
            "dividend_yield": random.uniform(0.0, 0.04),
            "scenario": scenario
        }
    
    @staticmethod
    def generate_claude_response(analysis_type: str = "comprehensive") -> dict:
        """Generate mock Claude API responses."""
        responses = {
            "comprehensive": {
                "momentum_score": 75,
                "technical_score": 82,
                "options_attractiveness": 68,
                "risk_assessment": "moderate",
                "overall_rating": 74,
                "confidence": 85,
                "reasoning": "Strong technical indicators with good momentum. Options show reasonable premium levels.",
                "key_factors": ["Rising RSI", "Bullish MACD crossover", "High options volume"],
                "risks": ["Market volatility", "Earnings uncertainty"],
                "recommendation": "Consider for watchlist"
            },
            "momentum": {
                "momentum_score": 68,
                "trend_strength": "strong",
                "momentum_direction": "bullish", 
                "sustainability": "high",
                "reasoning": "Consistent upward price movement with increasing volume"
            },
            "technical": {
                "technical_score": 79,
                "support_resistance": "strong support at $145",
                "trend_analysis": "uptrend intact",
                "indicator_consensus": "bullish",
                "reasoning": "Multiple indicators confirm bullish bias"
            },
            "options": {
                "options_attractiveness": 72,
                "iv_assessment": "moderate",
                "liquidity_score": 85,
                "value_proposition": "good",
                "reasoning": "Decent liquidity with fair pricing"
            }
        }
        
        return responses.get(analysis_type, responses["comprehensive"])


@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator."""
    return TestDataGenerator()


# Additional fixtures for comprehensive testing
@pytest.fixture
def mock_redis():
    """Mock Redis client for cache testing."""
    from unittest.mock import Mock
    redis_mock = Mock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.ttl = AsyncMock(return_value=-1)
    return redis_mock


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for HTTP testing."""
    from unittest.mock import Mock
    session = AsyncMock()
    
    # Mock response object
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"status": "ok"})
    response.text = AsyncMock(return_value='{"status": "ok"}')
    response.headers = {"Content-Type": "application/json"}
    
    session.get = AsyncMock(return_value=response)
    session.post = AsyncMock(return_value=response)
    session.close = AsyncMock()
    
    return session


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        return end_time - start_time
    
    return timer


@pytest.fixture
def market_data_scenarios(test_data_generator):
    """Provide various market data scenarios for testing."""
    scenarios = {}
    
    for scenario in ["bull", "bear", "sideways", "volatile", "normal"]:
        conditions = test_data_generator.generate_market_conditions(scenario)
        scenarios[scenario] = {
            "conditions": conditions,
            "stocks": [
                test_data_generator.generate_price_data(f"TEST{i}", 60) 
                for i in range(5)
            ],
            "options": [
                test_data_generator.generate_options_chain(f"TEST{i}", 100 + i*10)
                for i in range(5)
            ]
        }
    
    return scenarios


@pytest.fixture
def error_scenarios():
    """Provide error scenarios for testing error handling."""
    return {
        "network_timeout": {"error": "TimeoutError", "message": "Request timed out"},
        "api_rate_limit": {"error": "RateLimitError", "status": 429, "message": "Rate limit exceeded"},
        "invalid_api_key": {"error": "AuthenticationError", "status": 401, "message": "Invalid API key"},
        "server_error": {"error": "ServerError", "status": 500, "message": "Internal server error"},
        "malformed_response": {"error": "JSONDecodeError", "message": "Invalid JSON response"},
        "missing_data": {"error": "DataError", "message": "Required field missing"},
        "calculation_error": {"error": "CalculationError", "message": "Mathematical error in calculation"}
    }


@pytest.fixture
def memory_monitor():
    """Memory monitoring fixture for performance tests."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    def get_memory_usage():
        return process.memory_info().rss / 1024 / 1024  # MB
    
    return get_memory_usage


@pytest.fixture
def database_mock():
    """Mock database connections for testing."""
    from unittest.mock import AsyncMock
    
    db = AsyncMock()
    db.execute = AsyncMock(return_value=None)
    db.fetchone = AsyncMock(return_value=None)
    db.fetchall = AsyncMock(return_value=[])
    db.commit = AsyncMock(return_value=None)
    db.rollback = AsyncMock(return_value=None)
    db.close = AsyncMock(return_value=None)
    
    return db


# Test utilities
class TestAssertions:
    """Custom assertion helpers for financial data testing."""
    
    @staticmethod
    def assert_price_data_valid(price_data: dict):
        """Assert price data meets financial data requirements."""
        required_fields = ["open", "high", "low", "close", "volume"]
        for field in required_fields:
            assert field in price_data, f"Missing required field: {field}"
            assert price_data[field] > 0, f"Invalid {field} value: {price_data[field]}"
        
        # OHLC relationship validation
        assert price_data["high"] >= price_data["open"], "High must be >= Open"
        assert price_data["high"] >= price_data["close"], "High must be >= Close"
        assert price_data["low"] <= price_data["open"], "Low must be <= Open"
        assert price_data["low"] <= price_data["close"], "Low must be <= Close"
    
    @staticmethod
    def assert_option_data_valid(option_data: dict):
        """Assert option data meets options trading requirements."""
        required_fields = ["strike", "bid", "ask", "delta", "gamma", "theta", "vega"]
        for field in required_fields:
            assert field in option_data, f"Missing required field: {field}"
        
        # Options pricing relationships
        assert option_data["ask"] >= option_data["bid"], "Ask must be >= Bid"
        assert 0 <= option_data["delta"] <= 1, f"Invalid delta: {option_data['delta']}"
        assert option_data["gamma"] >= 0, f"Invalid gamma: {option_data['gamma']}"
        assert option_data["vega"] >= 0, f"Invalid vega: {option_data['vega']}"
    
    @staticmethod
    def assert_technical_indicators_valid(indicators: dict):
        """Assert technical indicators are within expected ranges."""
        if "rsi_14" in indicators:
            assert 0 <= indicators["rsi_14"] <= 100, f"Invalid RSI: {indicators['rsi_14']}"
        
        if "atr_14" in indicators:
            assert indicators["atr_14"] > 0, f"Invalid ATR: {indicators['atr_14']}"
        
        # Bollinger Bands relationship
        if all(key in indicators for key in ["bollinger_upper", "bollinger_middle", "bollinger_lower"]):
            assert indicators["bollinger_upper"] > indicators["bollinger_middle"]
            assert indicators["bollinger_middle"] > indicators["bollinger_lower"]
    
    @staticmethod
    def assert_score_in_range(score: float, min_val: float = 0.0, max_val: float = 100.0):
        """Assert score is within expected range."""
        assert min_val <= score <= max_val, f"Score {score} not in range [{min_val}, {max_val}]"


@pytest.fixture
def test_assertions() -> TestAssertions:
    """Provide test assertion helpers."""
    return TestAssertions()
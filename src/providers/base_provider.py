"""Base provider interfaces and common functionality."""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import aiohttp
import logging

from src.models import (
    ScreeningCriteria,
    StockQuote,
    OptionContract,
    FundamentalData,
    ProviderResponse,
    ProviderError,
    HealthCheckResult,
    ProviderStatus,
    ProviderType,
    AIAnalysisResult,
    ScreeningResult
)
from .exceptions import (
    ProviderError as ProviderException,
    RateLimitError,
    CircuitBreakerOpenError,
    AuthenticationError
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Track rate limiting state for a provider."""
    requests_made: int = 0
    window_start: float = field(default_factory=time.time)
    requests_per_window: int = 60
    window_duration: int = 60  # seconds


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.state = RateLimitState(requests_per_window=requests_per_minute)
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission for an API request."""
        async with self._lock:
            now = time.time()
            
            # Reset window if expired
            if now - self.state.window_start >= self.state.window_duration:
                self.state.requests_made = 0
                self.state.window_start = now
            
            # Check if we can make the request
            if self.state.requests_made >= self.state.requests_per_window:
                # Calculate wait time
                wait_time = self.state.window_duration - (now - self.state.window_start)
                if wait_time > 0:
                    logger.info(f"Rate limit reached ({self.state.requests_made}/{self.state.requests_per_window}). Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                    # Reset the window after waiting
                    self.state.requests_made = 0
                    self.state.window_start = time.time()
            
            self.state.requests_made += 1
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        return max(0, self.state.requests_per_window - self.state.requests_made)
    
    def get_reset_time(self) -> float:
        """Get timestamp when rate limit resets."""
        return self.state.window_start + self.state.window_duration


class CircuitBreaker:
    """Circuit breaker for provider reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # Reduced from 300 to 60 seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                logger.info("Circuit breaker reset to CLOSED")
    
    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.debug(f"Circuit breaker opened after {self.failure_count} failures")


class BaseProvider(ABC):
    """Abstract base class for all data providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_type = ProviderType(config.get("type", "unknown"))
        self.rate_limiter = RateLimiter(config.get("requests_per_minute", 60))
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 50),  # Increase threshold to be more tolerant
            recovery_timeout=config.get("recovery_timeout", 300)
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._total_cost = 0.0
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @abstractmethod
    async def get_health_status(self) -> HealthCheckResult:
        """Check provider health and availability."""
        pass
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "provider": self.provider_type.value,
            "requests_made": self._request_count,
            "total_cost": self._total_cost,
            "rate_limit_remaining": self.rate_limiter.get_remaining_requests(),
            "circuit_breaker_state": self.circuit_breaker.state,
            "last_health_check": datetime.now().isoformat()
        }
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ProviderResponse:
        """Make an HTTP request with rate limiting and error handling."""
        if not self.session:
            await self.initialize()
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Prepare request
        if headers is None:
            headers = {}
        
        start_time = time.time()
        
        try:
            async def _request():
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data,
                    **kwargs
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    self._request_count += 1
                    
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key or authentication failed")
                    elif response.status == 429:
                        raise RateLimitError("Rate limit exceeded by provider")
                    elif response.status >= 400:
                        error_text = await response.text()
                        raise ProviderException(
                            f"HTTP {response.status}: {error_text}",
                            status_code=response.status
                        )
                    
                    response_data = await response.json()
                    
                    return ProviderResponse(
                        success=True,
                        data=response_data,
                        provider=self.provider_type,
                        response_time_ms=response_time,
                        request_id=response.headers.get("X-Request-ID")
                    )
            
            # Execute with circuit breaker protection
            return await self.circuit_breaker.call(_request)
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(e, (RateLimitError, AuthenticationError, CircuitBreakerOpenError)):
                raise
            
            return ProviderResponse(
                success=False,
                error=str(e),
                provider=self.provider_type,
                response_time_ms=response_time
            )


class MarketDataProvider(BaseProvider):
    """Interface for market data providers (EODHD, MarketData.app)."""
    
    @abstractmethod
    async def screen_stocks(self, criteria: ScreeningCriteria) -> List[str]:
        """Screen stocks based on technical/fundamental criteria."""
        pass
    
    @abstractmethod
    async def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get current stock quote."""
        pass
    
    @abstractmethod
    async def get_stock_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get multiple stock quotes efficiently."""
        pass
    
    @abstractmethod
    async def get_options_chain(
        self, 
        symbol: str, 
        expiration_date: Optional[str] = None,
        **filters
    ) -> List[OptionContract]:
        """Get options chain for symbol with filtering."""
        pass
    
    @abstractmethod
    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental financial data."""
        pass
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and has data."""
        try:
            quote = await self.get_stock_quote(symbol)
            return quote is not None
        except Exception:
            return False


class AIProvider(BaseProvider):
    """Interface for AI analysis providers."""
    
    @abstractmethod
    async def analyze_opportunities(
        self,
        opportunities: List[ScreeningResult],
        context_data: Optional[Dict[str, Any]] = None
    ) -> List[AIAnalysisResult]:
        """Analyze screening opportunities with AI."""
        pass
    
    @abstractmethod
    async def estimate_cost(self, data_size: int) -> float:
        """Estimate analysis cost for given data size."""
        pass
    
    async def check_cost_limit(self, estimated_cost: float) -> bool:
        """Check if estimated cost is within daily limits."""
        daily_limit = self.config.get("daily_cost_limit", 50.0)
        return self._total_cost + estimated_cost <= daily_limit


class ProviderFactory:
    """Factory for creating provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a provider implementation."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> BaseProvider:
        """Create a provider instance."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider types."""
        return list(cls._providers.keys())
    
    @classmethod
    async def create_provider_with_health_check(
        cls, 
        provider_type: str, 
        config: Dict[str, Any]
    ) -> BaseProvider:
        """Create provider and verify it's healthy."""
        provider = cls.create_provider(provider_type, config)
        
        try:
            async with provider:
                health = await provider.get_health_status()
                if health.status != ProviderStatus.HEALTHY:
                    logger.warning(f"Provider {provider_type} is not healthy: {health.message}")
                
                return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider {provider_type}: {e}")
            raise
"""Provider interfaces and factory for market data and AI services."""

from .base_provider import (
    BaseProvider,
    MarketDataProvider,
    AIProvider,
    ProviderFactory,
    CircuitBreaker,
    RateLimiter
)
from .eodhd_client import EODHDClient
from .marketdata_client import MarketDataClient
from .provider_manager import ProviderManager, ProviderConfig, ProviderStrategy, DataSource
from .cache import CacheManager, ProviderCacheWrapper, get_cache_manager, initialize_cache_manager
from .validators import DataValidator, ValidationResult, get_validator, initialize_validator
from .exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    CircuitBreakerOpenError,
    DataQualityError,
    TimeoutError,
    QuotaExceededError,
    InvalidSymbolError,
    InsufficientDataError,
    ConfigurationError
)

# Register providers with factory
ProviderFactory.register_provider("eodhd", EODHDClient)
ProviderFactory.register_provider("marketdata", MarketDataClient)

__all__ = [
    # Base classes
    "BaseProvider",
    "MarketDataProvider", 
    "AIProvider",
    "ProviderFactory",
    "RateLimiter",
    "CircuitBreaker",
    
    # Concrete providers
    "EODHDClient",
    "MarketDataClient",
    
    # Management
    "ProviderManager",
    "ProviderConfig", 
    "ProviderStrategy",
    "DataSource",
    
    # Utilities
    "CacheManager",
    "ProviderCacheWrapper",
    "get_cache_manager",
    "initialize_cache_manager",
    "DataValidator",
    "ValidationResult", 
    "get_validator",
    "initialize_validator",
    
    # Exceptions
    "ProviderError",
    "RateLimitError",
    "AuthenticationError", 
    "CircuitBreakerOpenError",
    "DataQualityError",
    "TimeoutError",
    "QuotaExceededError",
    "InvalidSymbolError",
    "InsufficientDataError",
    "ConfigurationError"
]
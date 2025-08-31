"""Provider-specific exceptions."""

from typing import Optional


class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        return " | ".join(parts)


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, wait_time: Optional[float] = None, provider: Optional[str] = None):
        super().__init__(message, provider=provider, recoverable=True)
        self.wait_time = wait_time


class AuthenticationError(ProviderError):
    """Authentication/authorization error."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, provider=provider, status_code=401, recoverable=False)


class CircuitBreakerOpenError(ProviderError):
    """Circuit breaker is open, preventing requests."""
    
    def __init__(self, message: str = "Circuit breaker is open", provider: Optional[str] = None):
        super().__init__(message, provider=provider, recoverable=False)


class DataQualityError(ProviderError):
    """Data quality validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None, provider: Optional[str] = None):
        super().__init__(message, provider=provider, recoverable=True)
        self.field = field


class TimeoutError(ProviderError):
    """Request timeout error."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, provider: Optional[str] = None):
        super().__init__(message, provider=provider, recoverable=True)
        self.timeout_seconds = timeout_seconds


class QuotaExceededError(ProviderError):
    """API quota exceeded error."""
    
    def __init__(self, message: str, quota_type: Optional[str] = None, provider: Optional[str] = None):
        super().__init__(message, provider=provider, recoverable=False)
        self.quota_type = quota_type


class InvalidSymbolError(ProviderError):
    """Invalid or unknown symbol error."""
    
    def __init__(self, symbol: str, provider: Optional[str] = None):
        message = f"Invalid or unknown symbol: {symbol}"
        super().__init__(message, provider=provider, recoverable=False)
        self.symbol = symbol


class InsufficientDataError(ProviderError):
    """Insufficient data available error."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, provider: Optional[str] = None):
        super().__init__(message, provider=provider, recoverable=True)
        self.data_type = data_type


class ConfigurationError(ProviderError):
    """Provider configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, recoverable=False)
        self.config_key = config_key
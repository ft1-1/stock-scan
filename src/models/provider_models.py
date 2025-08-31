"""Provider-specific data models and interfaces."""

from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported provider types."""
    EODHD = "eodhd"
    MARKETDATA = "marketdata"
    CLAUDE = "claude"


class ProviderStatus(str, Enum):
    """Provider operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProviderResponse(BaseModel):
    """Standard response wrapper for provider operations."""
    
    success: bool = Field(..., description="Operation success status")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Metadata
    provider: ProviderType = Field(..., description="Provider that generated response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    # Performance metrics
    response_time_ms: Optional[float] = Field(None, ge=0, description="Response time in milliseconds")
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens consumed (for AI providers)")
    cost: Optional[float] = Field(None, ge=0, description="Operation cost in USD")


class ProviderError(BaseModel):
    """Detailed provider error information."""
    
    provider: ProviderType = Field(..., description="Provider that generated error")
    error_code: Optional[str] = Field(None, description="Provider-specific error code")
    error_message: str = Field(..., description="Error description")
    
    # HTTP details (if applicable)
    status_code: Optional[int] = Field(None, description="HTTP status code")
    response_body: Optional[str] = Field(None, description="Raw response body")
    
    # Context information
    endpoint: Optional[str] = Field(None, description="API endpoint that failed")
    request_params: Optional[Dict[str, Any]] = Field(None, description="Request parameters")
    
    # Timing and retry information
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    retry_count: int = Field(0, ge=0, description="Number of retries attempted")
    is_recoverable: bool = Field(True, description="Whether the error is recoverable")


class HealthCheckResult(BaseModel):
    """Provider health check result."""
    
    provider: ProviderType = Field(..., description="Provider being checked")
    status: ProviderStatus = Field(..., description="Current provider status")
    
    # Response metrics
    response_time_ms: Optional[float] = Field(None, ge=0, description="Health check response time")
    last_success: Optional[datetime] = Field(None, description="Last successful operation")
    last_failure: Optional[datetime] = Field(None, description="Last failed operation")
    
    # Usage statistics
    requests_in_last_hour: Optional[int] = Field(None, ge=0, description="Requests in past hour")
    success_rate_24h: Optional[float] = Field(None, ge=0, le=1, description="24-hour success rate")
    
    # Rate limiting information
    rate_limit_remaining: Optional[int] = Field(None, ge=0, description="Remaining rate limit")
    rate_limit_reset: Optional[datetime] = Field(None, description="Rate limit reset time")
    
    # Additional details
    message: Optional[str] = Field(None, description="Additional status information")
    check_timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class RateLimitInfo(BaseModel):
    """Rate limiting information for providers."""
    
    provider: ProviderType = Field(..., description="Provider type")
    
    # Current limits
    requests_per_minute: Optional[int] = Field(None, gt=0, description="Requests per minute limit")
    requests_per_hour: Optional[int] = Field(None, gt=0, description="Requests per hour limit")
    requests_per_day: Optional[int] = Field(None, gt=0, description="Requests per day limit")
    
    # Current usage
    current_minute_usage: int = Field(0, ge=0, description="Current minute usage")
    current_hour_usage: int = Field(0, ge=0, description="Current hour usage")
    current_day_usage: int = Field(0, ge=0, description="Current day usage")
    
    # Reset times
    minute_reset: Optional[datetime] = Field(None, description="Minute counter reset time")
    hour_reset: Optional[datetime] = Field(None, description="Hour counter reset time")
    day_reset: Optional[datetime] = Field(None, description="Day counter reset time")
    
    # Status
    is_limited: bool = Field(False, description="Whether rate limit is currently active")
    estimated_reset: Optional[datetime] = Field(None, description="Estimated time until reset")


class CacheInfo(BaseModel):
    """Cache information for provider responses."""
    
    cache_key: str = Field(..., description="Cache key identifier")
    provider: ProviderType = Field(..., description="Provider type")
    
    # Cache metadata
    cached_at: datetime = Field(default_factory=datetime.now, description="Cache creation time")
    expires_at: Optional[datetime] = Field(None, description="Cache expiration time")
    ttl_seconds: Optional[int] = Field(None, gt=0, description="Time to live in seconds")
    
    # Data information
    data_size_bytes: Optional[int] = Field(None, ge=0, description="Cached data size")
    hit_count: int = Field(0, ge=0, description="Number of cache hits")
    
    # Validation
    is_stale: bool = Field(False, description="Whether cache data is stale")
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")


class UsageStatistics(BaseModel):
    """Provider usage statistics."""
    
    provider: ProviderType = Field(..., description="Provider type")
    period_start: datetime = Field(..., description="Statistics period start")
    period_end: datetime = Field(..., description="Statistics period end")
    
    # Request statistics
    total_requests: int = Field(0, ge=0, description="Total requests made")
    successful_requests: int = Field(0, ge=0, description="Successful requests")
    failed_requests: int = Field(0, ge=0, description="Failed requests")
    
    # Performance metrics
    average_response_time_ms: Optional[float] = Field(None, ge=0, description="Average response time")
    min_response_time_ms: Optional[float] = Field(None, ge=0, description="Minimum response time")
    max_response_time_ms: Optional[float] = Field(None, ge=0, description="Maximum response time")
    
    # Cost information (if applicable)
    total_cost: Optional[float] = Field(None, ge=0, description="Total cost for period")
    cost_per_request: Optional[float] = Field(None, ge=0, description="Average cost per request")
    
    # Data transfer
    bytes_received: Optional[int] = Field(None, ge=0, description="Total bytes received")
    bytes_sent: Optional[int] = Field(None, ge=0, description="Total bytes sent")
    
    # Error breakdown
    error_types: Dict[str, int] = Field(default_factory=dict, description="Error counts by type")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
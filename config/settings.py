"""Centralized configuration management for the options screening application."""

import os
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    """Application configuration with environment variable support."""
    
    # Environment
    environment: str = Field("development", description="Application environment")
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    verbose: bool = Field(True, description="Enable verbose console output")
    
    # API Configuration
    eodhd_api_key: str = Field(..., description="EODHD API key")
    marketdata_api_key: str = Field(..., description="MarketData.app API key") 
    claude_api_key: Optional[str] = Field(None, description="Claude AI API key")
    
    # API Base URLs
    eodhd_base_url: str = Field("https://eodhd.com/api", description="EODHD API base URL")
    marketdata_base_url: str = Field("https://api.marketdata.app/v1", description="MarketData.app base URL")
    claude_base_url: str = Field("https://api.anthropic.com", description="Claude API base URL")
    
    # Screening Parameters
    max_stocks_to_screen: int = Field(5000, gt=0, le=10000, description="Maximum stocks to screen")
    min_market_cap: float = Field(50_000_000, gt=0, description="Minimum market cap filter")
    max_market_cap: float = Field(500_000_000_000, gt=0, description="Maximum market cap filter")
    min_stock_price: float = Field(5.0, gt=0, description="Minimum stock price")
    max_stock_price: float = Field(1000.0, gt=0, description="Maximum stock price")
    min_daily_volume: int = Field(100_000, gt=0, description="Minimum daily volume")
    
    # Technical Analysis Parameters
    rsi_oversold_threshold: float = Field(30.0, ge=0, le=100, description="RSI oversold level")
    rsi_overbought_threshold: float = Field(70.0, ge=0, le=100, description="RSI overbought level")
    bollinger_period: int = Field(20, gt=0, description="Bollinger Bands period")
    bollinger_std_dev: float = Field(2.0, gt=0, description="Bollinger Bands standard deviations")
    
    # Options Filtering
    min_option_volume: int = Field(100, ge=0, description="Minimum option volume")
    min_open_interest: int = Field(500, ge=0, description="Minimum open interest")
    max_days_to_expiration: int = Field(45, gt=0, description="Maximum days to expiration")
    min_days_to_expiration: int = Field(7, gt=0, description="Minimum days to expiration")
    
    # Options Greeks Filters
    min_delta_calls: float = Field(0.3, ge=0, le=1, description="Minimum delta for call options")
    max_delta_calls: float = Field(0.8, ge=0, le=1, description="Maximum delta for call options")
    min_delta_puts: float = Field(-0.8, ge=-1, le=0, description="Minimum delta for put options")
    max_delta_puts: float = Field(-0.3, ge=-1, le=0, description="Maximum delta for put options")
    
    # AI Configuration
    claude_model: str = Field("claude-3-5-sonnet-20241022", description="Claude model to use")
    claude_max_tokens: int = Field(4000, gt=0, le=8192, description="Maximum tokens per request")
    claude_temperature: float = Field(0.1, ge=0, le=1, description="Claude temperature setting")
    claude_daily_cost_limit: float = Field(50.0, gt=0, description="Daily AI cost limit in USD")
    claude_request_timeout: int = Field(60, gt=0, description="AI request timeout in seconds")
    ai_analysis_min_score: float = Field(70.0, ge=0, le=100, description="Minimum local score to qualify for AI analysis")
    ai_analysis_max_opportunities: int = Field(10, gt=0, description="Maximum number of opportunities to send to AI")
    
    # Local Rating Eligibility Gates
    rating_min_price: float = Field(5.0, gt=0, description="Minimum price for rating eligibility")
    rating_min_market_cap: float = Field(2000000000, gt=0, description="Minimum market cap for rating eligibility")
    rating_min_adv_shares: float = Field(1000000, gt=0, description="Minimum ADV shares for rating eligibility")
    rating_min_adv_dollars: float = Field(20000000, gt=0, description="Minimum ADV dollars for rating eligibility")
    rating_min_earnings_days: int = Field(7, gt=0, description="Minimum days to earnings for rating eligibility")
    rating_max_data_missing_pct: float = Field(0.20, ge=0, le=1, description="Maximum data missing percentage")
    
    # Local Rating Options Requirements
    rating_min_dte: int = Field(45, gt=0, description="Minimum days to expiration for valid options")
    rating_max_dte: int = Field(120, gt=0, description="Maximum days to expiration for valid options")
    rating_min_oi: int = Field(250, ge=0, description="Minimum open interest for valid options")
    rating_max_spread_pct: float = Field(3.5, gt=0, description="Maximum bid-ask spread percentage")
    
    # Performance Settings
    max_concurrent_requests: int = Field(50, gt=0, le=200, description="Max concurrent API requests")
    request_timeout: int = Field(30, gt=0, description="API request timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(2.0, gt=1, description="Retry backoff multiplier")
    
    # Caching Configuration
    enable_caching: bool = Field(True, description="Enable response caching")
    cache_ttl: int = Field(3600, gt=0, description="Cache TTL in seconds")
    max_cache_size_mb: int = Field(500, gt=0, description="Maximum cache size in MB")
    
    # Rate Limiting
    eodhd_requests_per_minute: int = Field(1000, gt=0, description="EODHD rate limit per minute")
    marketdata_requests_per_minute: int = Field(100, gt=0, description="MarketData rate limit per minute")
    claude_requests_per_minute: int = Field(50, gt=0, description="Claude rate limit per minute")
    
    # Output Configuration
    output_directory: str = Field("data/output", description="Output directory path")
    log_directory: str = Field("logs", description="Log directory path")
    cache_directory: str = Field("data/cache", description="Cache directory path")
    export_formats: List[str] = Field(["json", "csv"], description="Export formats")
    
    # Quality Settings
    min_data_quality_score: float = Field(80.0, ge=0, le=100, description="Minimum data quality threshold")
    require_fundamental_data: bool = Field(False, description="Require fundamental data")
    require_options_data: bool = Field(True, description="Require options data")
    
    # Monitoring and Alerting
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    metrics_collection_interval: int = Field(60, gt=0, description="Metrics collection interval in seconds")
    health_check_interval: int = Field(300, gt=0, description="Health check interval in seconds")
    
    # Security Settings
    encrypt_cache: bool = Field(False, description="Encrypt cached data")
    api_key_rotation_days: int = Field(90, gt=0, description="API key rotation period")
    
    # Workflow Settings
    enable_checkpointing: bool = Field(True, description="Enable workflow checkpointing")
    checkpoint_interval: int = Field(100, gt=0, description="Checkpoint every N symbols")
    max_workflow_duration_hours: int = Field(4, gt=0, description="Maximum workflow duration")
    
    # Exclusion Lists
    excluded_sectors: List[str] = Field(
        default_factory=lambda: ["Utilities", "Real Estate"], 
        description="Sectors to exclude from screening"
    )
    excluded_symbols: List[str] = Field(
        default_factory=list, 
        description="Specific symbols to exclude"
    )
    
    # Specific Symbol Screening
    specific_symbols: Optional[str] = Field(
        None,
        description="Comma-separated list of specific symbols to screen (overrides general screening)"
    )
    
    @property
    def symbols_list(self) -> Optional[List[str]]:
        """Parse specific_symbols into a list if provided."""
        if self.specific_symbols:
            return [s.strip().upper() for s in self.specific_symbols.split(',') if s.strip()]
        return None
    
    # Testing Configuration
    test_mode: bool = Field(False, description="Enable test mode with mock data")
    test_symbol_limit: int = Field(10, gt=0, description="Symbol limit in test mode")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = "SCREENER_"
    
    @validator('max_market_cap')
    def validate_market_cap_range(cls, v, values):
        if 'min_market_cap' in values and v <= values['min_market_cap']:
            raise ValueError('max_market_cap must be greater than min_market_cap')
        return v
    
    @validator('max_stock_price')
    def validate_price_range(cls, v, values):
        if 'min_stock_price' in values and v <= values['min_stock_price']:
            raise ValueError('max_stock_price must be greater than min_stock_price')
        return v
    
    @validator('rsi_overbought_threshold')
    def validate_rsi_thresholds(cls, v, values):
        if 'rsi_oversold_threshold' in values and v <= values['rsi_oversold_threshold']:
            raise ValueError('rsi_overbought_threshold must be greater than rsi_oversold_threshold')
        return v
    
    @validator('max_days_to_expiration')
    def validate_expiration_range(cls, v, values):
        if 'min_days_to_expiration' in values and v <= values['min_days_to_expiration']:
            raise ValueError('max_days_to_expiration must be greater than min_days_to_expiration')
        return v
    
    @validator('max_delta_calls')
    def validate_call_delta_range(cls, v, values):
        if 'min_delta_calls' in values and v <= values['min_delta_calls']:
            raise ValueError('max_delta_calls must be greater than min_delta_calls')
        return v
    
    @validator('min_delta_puts')
    def validate_put_delta_range(cls, v, values):
        if 'max_delta_puts' in values and v >= values['max_delta_puts']:
            raise ValueError('min_delta_puts must be less than max_delta_puts')
        return v
    
    @property
    def output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_directory)
    
    @property
    def log_path(self) -> Path:
        """Get log directory as Path object."""
        return Path(self.log_directory)
    
    @property
    def cache_path(self) -> Path:
        """Get cache directory as Path object."""
        return Path(self.cache_directory)
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for a specific provider."""
        configs = {
            "eodhd": {
                "api_key": self.eodhd_api_key,
                "base_url": self.eodhd_base_url,
                "requests_per_minute": self.eodhd_requests_per_minute,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries
            },
            "marketdata": {
                "api_key": self.marketdata_api_key,
                "base_url": self.marketdata_base_url,
                "requests_per_minute": self.marketdata_requests_per_minute,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries
            },
            "claude": {
                "api_key": self.claude_api_key,
                "base_url": self.claude_base_url,
                "model": self.claude_model,
                "max_tokens": self.claude_max_tokens,
                "temperature": self.claude_temperature,
                "requests_per_minute": self.claude_requests_per_minute,
                "timeout": self.claude_request_timeout,
                "daily_cost_limit": self.claude_daily_cost_limit
            }
        }
        return configs.get(provider, {})


# Helper function to get env var with prefix
def get_env(key: str, default=None):
    """Get environment variable with SCREENER_ prefix."""
    return os.getenv(f"SCREENER_{key}", default)

# Global settings instance - load from environment
def load_settings_from_env() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        # Environment
        environment=get_env("ENVIRONMENT", "development"),
        debug=get_env("DEBUG", "false").lower() == "true",
        log_level=get_env("LOG_LEVEL", "INFO"),
        verbose=get_env("VERBOSE", "true").lower() == "true",
        
        # API Configuration
        eodhd_api_key=get_env("EODHD_API_KEY", ""),
        marketdata_api_key=get_env("MARKETDATA_API_KEY", ""),
        claude_api_key=get_env("CLAUDE_API_KEY"),
        
        # AI Analysis settings
        ai_analysis_min_score=float(get_env("AI_ANALYSIS_MIN_SCORE", "70.0")),
        ai_analysis_max_opportunities=int(get_env("AI_ANALYSIS_MAX_OPPORTUNITIES", "10")),
        
        # Local Rating Eligibility Gates
        rating_min_price=float(get_env("RATING_MIN_PRICE", "5.0")),
        rating_min_market_cap=float(get_env("RATING_MIN_MARKET_CAP", "2000000000")),
        rating_min_adv_shares=float(get_env("RATING_MIN_ADV_SHARES", "1000000")),
        rating_min_adv_dollars=float(get_env("RATING_MIN_ADV_DOLLARS", "20000000")),
        rating_min_earnings_days=int(get_env("RATING_MIN_EARNINGS_DAYS", "7")),
        rating_max_data_missing_pct=float(get_env("RATING_MAX_DATA_MISSING_PCT", "0.20")),
        
        # Local Rating Options Requirements
        rating_min_dte=int(get_env("RATING_MIN_DTE", "45")),
        rating_max_dte=int(get_env("RATING_MAX_DTE", "120")),
        rating_min_oi=int(get_env("RATING_MIN_OI", "250")),
        rating_max_spread_pct=float(get_env("RATING_MAX_SPREAD_PCT", "3.5")),
        
        # Add other settings as needed from environment
        specific_symbols=get_env("SPECIFIC_SYMBOLS"),
        min_market_cap=float(get_env("MIN_MARKET_CAP", "50000000")),
        max_market_cap=float(get_env("MAX_MARKET_CAP", "500000000000")),
        min_stock_price=float(get_env("MIN_STOCK_PRICE", "5.0")),
        max_stock_price=float(get_env("MAX_STOCK_PRICE", "1000.0")),
        min_daily_volume=int(get_env("MIN_DAILY_VOLUME", "100000")),
    )

# Global settings instance
settings = load_settings_from_env()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    settings = load_settings_from_env()
    return settings
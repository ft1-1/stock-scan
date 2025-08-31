# Data Models, API Contracts & Configuration Specifications

## Complete Data Model Specifications

### Core Stock Market Models

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from datetime import datetime, date
from enum import Enum
import uuid

# Enumerations for type safety
class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class ScreeningStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class DataSource(Enum):
    EODHD = "eodhd"
    MARKETDATA = "marketdata"
    CLAUDE = "claude"
    CALCULATED = "calculated"

# Base Models
@dataclass
class BaseModel:
    """Base class for all data models with common fields."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    data_source: Optional[DataSource] = None
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()

# Stock Market Data Models
@dataclass
class StockQuote(BaseModel):
    """Real-time or delayed stock quote information."""
    symbol: str
    last_price: Decimal
    change: Decimal
    change_percent: float
    volume: int
    
    # OHLC data
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    previous_close: Optional[Decimal] = None
    
    # Market microstructure
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    # Extended data
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    # Metadata
    exchange: Optional[str] = None
    currency: str = "USD"
    market_session: str = "regular"  # regular, pre, post
    quote_timestamp: Optional[datetime] = None
    
    @property
    def spread_percent(self) -> Optional[float]:
        """Calculate bid-ask spread as percentage of mid-price."""
        if self.bid and self.ask and self.bid > 0:
            mid = (self.bid + self.ask) / 2
            return float((self.ask - self.bid) / mid * 100)
        return None

@dataclass
class FundamentalData(BaseModel):
    """Comprehensive fundamental analysis data."""
    symbol: str
    
    # Valuation metrics
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[int] = None
    ev_to_ebitda: Optional[float] = None
    
    # Profitability metrics
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roi: Optional[float] = None  # Return on Investment
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    
    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # Growth metrics
    revenue_growth_1y: Optional[float] = None
    earnings_growth_1y: Optional[float] = None
    eps_growth_quarterly: Optional[float] = None
    
    # Per-share metrics
    earnings_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    cash_per_share: Optional[float] = None
    
    # Company information
    sector: Optional[str] = None
    industry: Optional[str] = None
    employees: Optional[int] = None
    description: Optional[str] = None
    
    # Calendar events
    next_earnings_date: Optional[date] = None
    ex_dividend_date: Optional[date] = None
    dividend_amount: Optional[float] = None
    
    data_date: Optional[date] = None

@dataclass 
class TechnicalIndicators(BaseModel):
    """Technical analysis indicators and oscillators."""
    symbol: str
    price: Decimal  # Reference price for calculations
    
    # Moving averages
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum indicators
    rsi_14: Optional[float] = None
    rsi_30: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_width: Optional[float] = None
    atr_14: Optional[float] = None
    
    # Volume indicators
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None  # Current vs average
    
    # Price patterns
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    trend_direction: Optional[str] = None  # "up", "down", "sideways"
    
    # Market metrics
    beta: Optional[float] = None
    correlation_spy: Optional[float] = None
    
    calculation_period: int = 252  # Trading days for calculations
    calculation_timestamp: Optional[datetime] = None

@dataclass
class OptionContract(BaseModel):
    """Option contract with full Greeks and market data."""
    option_symbol: str
    underlying_symbol: str
    strike: Decimal
    expiration: date
    option_type: OptionType
    
    # Market data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    mark: Optional[Decimal] = None  # Mid-price
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    epsilon: Optional[float] = None  # Dividend sensitivity
    
    # Volatility metrics
    implied_volatility: Optional[float] = None
    historical_volatility: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    
    # Calculated fields
    days_to_expiration: Optional[int] = None
    intrinsic_value: Optional[Decimal] = None
    time_value: Optional[Decimal] = None
    moneyness: Optional[float] = None
    
    # Liquidity metrics
    bid_ask_spread: Optional[Decimal] = None
    spread_percent: Optional[float] = None
    liquidity_score: Optional[float] = None
    
    # Risk metrics
    probability_itm: Optional[float] = None  # Probability in-the-money
    probability_profit: Optional[float] = None
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money."""
        if not hasattr(self, '_underlying_price'):
            return False
        if self.option_type == OptionType.CALL:
            return self._underlying_price > self.strike
        else:
            return self._underlying_price < self.strike
    
    def set_underlying_price(self, price: Decimal):
        """Set underlying price for ITM calculations."""
        self._underlying_price = price

@dataclass
class OptionsChain(BaseModel):
    """Complete options chain for a symbol."""
    symbol: str
    underlying_price: Decimal
    expiration_dates: List[date]
    
    # Options by expiration and type
    calls: Dict[date, List[OptionContract]] = field(default_factory=dict)
    puts: Dict[date, List[OptionContract]] = field(default_factory=dict)
    
    # Chain metadata
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_open_interest: int = 0
    total_put_open_interest: int = 0
    put_call_ratio: Optional[float] = None
    
    # Volatility surface data
    iv_surface: Optional[Dict[str, Any]] = None
    term_structure: Optional[Dict[date, float]] = None
    
    def get_options_by_criteria(
        self,
        option_type: OptionType,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        min_volume: Optional[int] = None,
        min_open_interest: Optional[int] = None
    ) -> List[OptionContract]:
        """Filter options by various criteria."""
        options = []
        option_dict = self.calls if option_type == OptionType.CALL else self.puts
        
        for exp_date, contracts in option_dict.items():
            for contract in contracts:
                # Apply filters
                if min_dte and contract.days_to_expiration and contract.days_to_expiration < min_dte:
                    continue
                if max_dte and contract.days_to_expiration and contract.days_to_expiration > max_dte:
                    continue
                if min_delta and contract.delta and abs(contract.delta) < min_delta:
                    continue
                if max_delta and contract.delta and abs(contract.delta) > max_delta:
                    continue
                if min_volume and contract.volume and contract.volume < min_volume:
                    continue
                if min_open_interest and contract.open_interest and contract.open_interest < min_open_interest:
                    continue
                
                options.append(contract)
        
        return options

# Screening and Analysis Models
@dataclass
class ScreeningCriteria(BaseModel):
    """Comprehensive screening criteria for stock selection."""
    name: str
    description: Optional[str] = None
    
    # Market filters
    min_market_cap: Optional[int] = None
    max_market_cap: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    max_volume: Optional[int] = None
    
    # Technical filters
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    price_above_sma_periods: Optional[List[int]] = None
    price_below_sma_periods: Optional[List[int]] = None
    min_atr_percent: Optional[float] = None
    max_atr_percent: Optional[float] = None
    
    # Fundamental filters
    min_pe_ratio: Optional[float] = None
    max_pe_ratio: Optional[float] = None
    min_roe: Optional[float] = None
    max_debt_to_equity: Optional[float] = None
    min_current_ratio: Optional[float] = None
    
    # Options-specific filters
    min_option_volume: Optional[int] = None
    min_open_interest: Optional[int] = None
    max_bid_ask_spread_percent: Optional[float] = None
    min_iv_rank: Optional[float] = None
    max_iv_rank: Optional[float] = None
    
    # Date filters
    earnings_exclude_days: Optional[int] = None  # Exclude if earnings within N days
    dividend_exclude_days: Optional[int] = None
    
    # Inclusion/exclusion lists
    include_sectors: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None
    include_symbols: Optional[List[str]] = None
    exclude_symbols: Optional[List[str]] = None
    
    # Geographic filters
    exchanges: Optional[List[str]] = None
    countries: Optional[List[str]] = None

@dataclass
class ScreeningResult(BaseModel):
    """Result of screening analysis for a single symbol."""
    symbol: str
    screening_criteria_id: str
    
    # Core data
    stock_quote: StockQuote
    fundamental_data: Optional[FundamentalData] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    options_chain: Optional[OptionsChain] = None
    
    # Selected options (strategy-specific)
    selected_calls: List[OptionContract] = field(default_factory=list)
    selected_puts: List[OptionContract] = field(default_factory=list)
    
    # Quantitative scores (0-100)
    liquidity_score: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    options_quality_score: float = 0.0
    overall_quantitative_score: float = 0.0
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # AI analysis (if enabled)
    ai_analysis: Optional['AIAnalysisResult'] = None
    combined_score: Optional[float] = None  # Quantitative + AI combined
    
    # Processing metadata
    processing_time_seconds: Optional[float] = None
    data_completeness_percent: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

# AI Analysis Models
@dataclass
class AIAnalysisResult(BaseModel):
    """Results from AI analysis of screening opportunities."""
    symbol: str
    model_used: str
    
    # AI scores (0-100)
    ai_overall_score: float
    confidence_level: float
    
    # Component scores
    fundamental_assessment: float
    technical_assessment: float
    options_strategy_assessment: float
    market_context_assessment: float
    risk_assessment: float
    
    # Qualitative analysis
    key_strengths: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    market_commentary: Optional[str] = None
    strategy_recommendation: Optional[str] = None
    
    # Specific insights
    entry_timing_assessment: Optional[str] = None
    exit_strategy_suggestions: List[str] = field(default_factory=list)
    position_sizing_recommendation: Optional[str] = None
    
    # Analysis metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    
    # Supporting data used in analysis
    data_sources_used: List[DataSource] = field(default_factory=list)
    data_completeness_score: float = 0.0

# Workflow and Execution Models
@dataclass
class WorkflowExecution(BaseModel):
    """Tracks execution of the complete screening workflow."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    screening_criteria: ScreeningCriteria
    status: ScreeningStatus = ScreeningStatus.PENDING
    
    # Execution timeline
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Step tracking
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    
    # Results
    total_symbols_screened: int = 0
    symbols_passed_screening: int = 0
    final_results: List[ScreeningResult] = field(default_factory=list)
    
    # Resource usage
    api_calls_made: Dict[str, int] = field(default_factory=dict)
    api_costs_usd: Dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    # Configuration snapshot
    configuration_snapshot: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DailyScreeningReport(BaseModel):
    """Daily summary report of screening results."""
    report_date: date
    execution_id: str
    
    # Summary statistics
    total_opportunities_found: int
    top_opportunities: List[ScreeningResult]
    
    # Performance metrics
    average_scores: Dict[str, float] = field(default_factory=dict)
    score_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Market analysis
    market_summary: Optional[str] = None
    sector_breakdown: Dict[str, int] = field(default_factory=dict)
    volatility_environment: Optional[str] = None
    
    # Operational metrics
    execution_time_seconds: float
    data_quality_score: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Comparison to previous days
    performance_vs_yesterday: Optional[Dict[str, float]] = None
    performance_vs_week_ago: Optional[Dict[str, float]] = None
```

## API Contract Specifications

### Provider Interface Contracts

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class MarketDataProviderProtocol(Protocol):
    """Type protocol for market data providers."""
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Return provider health and status information."""
        ...
    
    async def screen_stocks(
        self, 
        criteria: ScreeningCriteria
    ) -> List[str]:
        """Screen stocks and return list of symbols."""
        ...
    
    async def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get current stock quote."""
        ...
    
    async def get_stock_quotes(
        self, 
        symbols: List[str]
    ) -> Dict[str, StockQuote]:
        """Get multiple stock quotes efficiently."""
        ...
    
    async def get_options_chain(
        self, 
        symbol: str, 
        **filters
    ) -> OptionsChain:
        """Get options chain with filtering."""
        ...
    
    async def get_fundamental_data(
        self, 
        symbol: str
    ) -> Optional[FundamentalData]:
        """Get fundamental analysis data."""
        ...

@runtime_checkable  
class AIProviderProtocol(Protocol):
    """Type protocol for AI analysis providers."""
    
    async def analyze_opportunities(
        self,
        opportunities: List[ScreeningResult],
        context: Dict[str, Any]
    ) -> List[AIAnalysisResult]:
        """Analyze screening opportunities."""
        ...
    
    async def estimate_cost(
        self,
        opportunities: List[ScreeningResult]
    ) -> float:
        """Estimate analysis cost in USD."""
        ...
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        ...

# Response Format Standards
@dataclass
class APIResponse:
    """Standardized API response format."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def success_response(cls, data: Any, **metadata) -> 'APIResponse':
        """Create success response."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_response(
        cls, 
        message: str, 
        code: Optional[str] = None
    ) -> 'APIResponse':
        """Create error response."""
        return cls(success=False, error_message=message, error_code=code)

# Health Check Contract
@dataclass
class HealthStatus:
    """Health status for components."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    dependencies: List['HealthStatus'] = field(default_factory=list)
```

## Configuration Management System

```python
from pydantic import BaseSettings, Field, validator
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

class DatabaseConfig(BaseSettings):
    """Database configuration."""
    url: str = "sqlite:///./data/screening.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20

class EODHDConfig(BaseSettings):
    """EODHD API configuration."""
    api_key: str = Field(..., env="EODHD_API_KEY")
    base_url: str = "https://eodhd.com/api"
    timeout: int = 30
    max_retries: int = 3
    requests_per_minute: int = 60
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v == "your_api_key_here":
            raise ValueError("Valid EODHD API key required")
        return v

class MarketDataConfig(BaseSettings):
    """MarketData.app API configuration."""
    api_key: str = Field(..., env="MARKETDATA_API_KEY")
    base_url: str = "https://api.marketdata.app/v1"
    timeout: int = 30
    max_retries: int = 3
    use_cached_feed: bool = True  # Critical for cost control
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v == "your_api_key_here":
            raise ValueError("Valid MarketData API key required")
        return v

class ClaudeConfig(BaseSettings):
    """Claude AI configuration."""
    api_key: Optional[str] = Field(None, env="CLAUDE_API_KEY")
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60
    daily_cost_limit: float = 50.0
    max_opportunities_per_batch: int = 20
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if v and (not v.startswith('sk-') or len(v) < 20):
            raise ValueError("Invalid Claude API key format")
        return v

class ScreeningConfig(BaseSettings):
    """Screening process configuration."""
    max_stocks_to_screen: int = 5000
    max_concurrent_requests: int = 50
    cache_ttl_seconds: int = 3600
    
    # Default screening criteria
    default_min_market_cap: int = 50_000_000
    default_max_market_cap: int = 500_000_000_000
    default_min_volume: int = 100_000
    default_min_price: float = 5.0
    default_max_price: float = 1000.0
    
    # Options filtering
    min_option_volume: int = 100
    min_open_interest: int = 500
    max_days_to_expiration: int = 45
    max_bid_ask_spread_percent: float = 5.0

class AIConfig(BaseSettings):
    """AI analysis configuration."""
    enabled: bool = Field(False, env="AI_ANALYSIS_ENABLED")
    min_data_completeness: float = 60.0
    min_confidence_threshold: float = 70.0
    top_n_opportunities: int = 10
    cost_per_1k_tokens: float = 0.003  # Claude pricing

class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "1 day"
    retention: str = "30 days"
    log_directory: Path = Path("data/logs")

class Settings(BaseSettings):
    """Main application settings."""
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    eodhd: EODHDConfig = EODHDConfig()
    marketdata: MarketDataConfig = MarketDataConfig()
    claude: ClaudeConfig = ClaudeConfig()
    screening: ScreeningConfig = ScreeningConfig()
    ai: AIConfig = AIConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Output configuration
    output_directory: Path = Path("data/results")
    export_formats: List[str] = ["json", "csv"]
    
    # Monitoring
    enable_monitoring: bool = True
    health_check_interval: int = 300  # seconds
    alert_webhooks: List[str] = []
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('output_directory')
    def create_output_directory(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_provider_config(self, provider_type: str) -> Dict[str, Any]:
        """Get configuration for specific provider."""
        configs = {
            "eodhd": self.eodhd.dict(),
            "marketdata": self.marketdata.dict(),
            "claude": self.claude.dict()
        }
        return configs.get(provider_type, {})
    
    def validate_configuration(self) -> List[str]:
        """Validate complete configuration and return warnings."""
        warnings = []
        
        # Check API keys
        if not self.eodhd.api_key:
            warnings.append("EODHD API key not configured")
        if not self.marketdata.api_key:
            warnings.append("MarketData API key not configured")
        if self.ai.enabled and not self.claude.api_key:
            warnings.append("AI enabled but Claude API key not configured")
        
        # Check directories
        if not self.output_directory.exists():
            warnings.append(f"Output directory {self.output_directory} does not exist")
        
        # Check cost limits
        if self.claude.daily_cost_limit > 100:
            warnings.append("Claude daily cost limit is very high (>$100)")
        
        return warnings

# Configuration factory
def get_settings() -> Settings:
    """Get application settings with caching."""
    return Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    screening: ScreeningConfig = ScreeningConfig(max_stocks_to_screen=100)
    claude: ClaudeConfig = ClaudeConfig(daily_cost_limit=5.0)

class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    logging: LoggingConfig = LoggingConfig(level="WARNING")
    screening: ScreeningConfig = ScreeningConfig(max_stocks_to_screen=10000)
```

## Configuration File Examples

### .env.example
```bash
# Environment
ENVIRONMENT=development
DEBUG=false

# API Keys (REQUIRED)
EODHD_API_KEY=your_eodhd_api_key_here
MARKETDATA_API_KEY=your_marketdata_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here

# Screening Configuration
MAX_STOCKS_TO_SCREEN=5000
MIN_MARKET_CAP=50000000
MAX_MARKET_CAP=500000000000
MIN_VOLUME=100000

# AI Configuration
AI_ANALYSIS_ENABLED=true
CLAUDE_DAILY_COST_LIMIT=50.0
TOP_N_OPPORTUNITIES=10

# Performance
MAX_CONCURRENT_REQUESTS=50
CACHE_TTL_SECONDS=3600

# Output
OUTPUT_DIRECTORY=data/results
EXPORT_FORMATS=json,csv
```

### production.yaml
```yaml
database:
  url: "postgresql://user:pass@localhost/screening_prod"
  pool_size: 20
  max_overflow: 40

screening:
  max_stocks_to_screen: 10000
  max_concurrent_requests: 100

logging:
  level: "WARNING"
  retention: "90 days"

monitoring:
  enable_monitoring: true
  health_check_interval: 60
  alert_webhooks:
    - "https://hooks.slack.com/services/..."
```

This comprehensive data model and configuration system provides:
1. **Type Safety**: Pydantic models with validation
2. **Flexibility**: Environment-based configuration 
3. **Scalability**: Support for different deployment environments
4. **Maintainability**: Clear contracts and interfaces
5. **Monitoring**: Built-in health checks and metrics
6. **Cost Control**: AI usage limits and monitoring
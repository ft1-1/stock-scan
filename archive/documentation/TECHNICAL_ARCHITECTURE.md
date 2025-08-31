# Technical Architecture Decisions

## Core Technology Stack

### Python Framework & Libraries
- **Python 3.11+**: Latest stable version for performance and type hints
- **AsyncIO**: Native async/await for concurrent API calls and processing
- **Pydantic v2**: Data validation, serialization, and type safety
- **aiohttp**: Async HTTP client for API integrations
- **pandas**: Data manipulation and technical indicator calculations
- **numpy**: Numerical computations for financial metrics
- **loguru**: Advanced logging with structured output
- **tenacity**: Retry mechanisms with exponential backoff

### Data Processing & Storage
- **JSON**: Primary data exchange format for flexibility
- **Parquet**: Efficient storage for large datasets (future enhancement)
- **SQLite**: Local caching and data persistence
- **Redis**: Optional distributed caching (production)

### AI & API Integration
- **anthropic**: Official Claude AI client library
- **eodhd**: Official EODHD Python library
- **Custom aiohttp clients**: For MarketData.app and other APIs

### Deployment & Operations
- **Docker**: Containerization for consistent deployment
- **systemd**: Service management for Linux deployments
- **APScheduler**: Advanced Python job scheduling
- **Prometheus**: Metrics collection (optional)

## Component Interface Specifications

### 1. Provider Interface Architecture

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ProviderType(Enum):
    EODHD = "eodhd"
    MARKETDATA = "marketdata"
    CLAUDE = "claude"

class DataProvider(ABC):
    """Abstract base class for all data providers."""
    
    @abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """Check provider health and availability."""
        pass
    
    @abstractmethod
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        pass

class MarketDataProvider(DataProvider):
    """Interface for market data providers (EODHD, MarketData.app)."""
    
    @abstractmethod
    async def screen_stocks(self, criteria: ScreeningCriteria) -> List[StockSymbol]:
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
    async def get_options_chain(self, symbol: str, **filters) -> OptionsChain:
        """Get options chain for symbol with filtering."""
        pass
    
    @abstractmethod
    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental financial data."""
        pass

class AIProvider(DataProvider):
    """Interface for AI analysis providers."""
    
    @abstractmethod
    async def analyze_opportunities(
        self, 
        opportunities: List[ScreeningResult],
        context_data: Dict[str, Any]
    ) -> AIAnalysisResult:
        """Analyze screening opportunities with AI."""
        pass
    
    @abstractmethod
    async def estimate_cost(self, data_size: int) -> float:
        """Estimate analysis cost for given data size."""
        pass
```

### 2. Core Workflow Engine

```python
class WorkflowStep(Enum):
    STOCK_SCREENING = "stock_screening"
    DATA_COLLECTION = "data_collection" 
    TECHNICAL_CALCULATION = "technical_calculation"
    OPTION_SELECTION = "option_selection"
    LLM_PACKAGING = "llm_packaging"
    AI_ANALYSIS = "ai_analysis"
    RESULT_PROCESSING = "result_processing"

class WorkflowManager:
    """Orchestrates the 7-step options screening workflow."""
    
    async def execute_workflow(
        self, 
        criteria: ScreeningCriteria
    ) -> WorkflowResult:
        """Execute complete screening workflow with error recovery."""
        
        results = {}
        
        for step in WorkflowStep:
            try:
                result = await self._execute_step(step, results)
                results[step] = result
                await self._checkpoint_progress(step, result)
                
            except Exception as e:
                await self._handle_step_failure(step, e)
                if self._is_critical_step(step):
                    raise
                    
        return self._compile_final_results(results)
    
    async def _execute_step(
        self, 
        step: WorkflowStep, 
        previous_results: Dict[WorkflowStep, Any]
    ) -> Any:
        """Execute individual workflow step."""
        pass
```

### 3. Data Models & Type System

```python
@dataclass
class ScreeningCriteria:
    """Comprehensive screening criteria."""
    # Market filters
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    
    # Technical filters
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    price_above_sma: Optional[int] = None  # SMA period
    
    # Fundamental filters
    min_pe_ratio: Optional[float] = None
    max_pe_ratio: Optional[float] = None
    min_roe: Optional[float] = None
    
    # Options-specific filters
    min_option_volume: Optional[int] = None
    min_open_interest: Optional[int] = None
    
    # Exclusions
    exclude_sectors: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)

@dataclass
class StockQuote:
    """Standardized stock quote data."""
    symbol: str
    last_price: float
    change: float
    change_percent: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OptionContract:
    """Option contract with full Greeks and market data."""
    option_symbol: str
    underlying_symbol: str
    strike: Decimal
    expiration: date
    option_type: str  # 'call' or 'put'
    
    # Market data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    # Calculated fields
    days_to_expiration: Optional[int] = None
    intrinsic_value: Optional[Decimal] = None
    time_value: Optional[Decimal] = None
    moneyness: Optional[float] = None  # Strike/Spot ratio

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    symbol: str
    
    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    
    # Volume indicators
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScreeningResult:
    """Result of options screening analysis."""
    symbol: str
    stock_quote: StockQuote
    technical_indicators: TechnicalIndicators
    fundamental_data: Optional[FundamentalData]
    
    # Selected options
    selected_calls: List[OptionContract]
    selected_puts: List[OptionContract]
    
    # Quantitative scores
    liquidity_score: float
    volatility_score: float
    momentum_score: float
    overall_score: float
    
    # AI analysis (if available)
    ai_analysis: Optional[AIAnalysisResult] = None
    
    screening_timestamp: datetime = field(default_factory=datetime.now)
```

### 4. Error Handling & Resilience Architecture

```python
class ScreeningError(Exception):
    """Base exception for screening errors."""
    def __init__(self, message: str, step: WorkflowStep, recoverable: bool = True):
        super().__init__(message)
        self.step = step
        self.recoverable = recoverable

class ProviderError(ScreeningError):
    """Provider-specific errors."""
    def __init__(self, provider: str, message: str, status_code: Optional[int] = None):
        super().__init__(f"Provider {provider} error: {message}", None)
        self.provider = provider
        self.status_code = status_code

class CircuitBreaker:
    """Circuit breaker for provider reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ProviderError("Circuit breaker OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

class RetryManager:
    """Advanced retry logic with exponential backoff."""
    
    @staticmethod
    async def retry_with_backoff(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """Retry function with exponential backoff and jitter."""
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries:
                    raise
                
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                if jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                await asyncio.sleep(delay)
```

### 5. Configuration Management Architecture

```python
class Settings(BaseSettings):
    """Centralized configuration management."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # API Configuration
    eodhd_api_key: str
    marketdata_api_key: str
    claude_api_key: Optional[str] = None
    
    # Screening Parameters
    max_stocks_to_screen: int = 5000
    min_market_cap: float = 50_000_000
    max_market_cap: float = 500_000_000_000
    
    # Options Filtering
    min_option_volume: int = 100
    min_open_interest: int = 500
    max_days_to_expiration: int = 45
    
    # AI Configuration
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_max_tokens: int = 4000
    claude_temperature: float = 0.1
    claude_daily_cost_limit: float = 50.0
    
    # Performance Settings
    max_concurrent_requests: int = 50
    request_timeout: int = 30
    cache_ttl: int = 3600
    
    # Output Configuration
    output_directory: str = "data/results"
    export_formats: List[str] = ["json", "csv"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

## Architecture Decision Rationale

### 1. **Async-First Design**
- **Decision**: Use asyncio throughout the application
- **Rationale**: Options screening requires many concurrent API calls; async provides better resource utilization
- **Impact**: 5-10x performance improvement for I/O-bound operations

### 2. **Provider Abstraction Pattern**
- **Decision**: Abstract interface for all data providers
- **Rationale**: Enables easy provider switching, testing, and failover
- **Impact**: Improved reliability and flexibility

### 3. **Workflow Engine Architecture**
- **Decision**: Step-based workflow with checkpointing
- **Rationale**: Complex 7-step process needs error recovery and resume capability
- **Impact**: Better fault tolerance and debugging

### 4. **Type-Safe Data Models**
- **Decision**: Pydantic v2 for all data structures
- **Rationale**: Runtime validation, IDE support, and serialization
- **Impact**: Fewer runtime errors, better maintainability

### 5. **Configurable AI Integration**
- **Decision**: Optional AI analysis with cost controls
- **Rationale**: Users can gradually adopt AI features based on budget
- **Impact**: Broader adoption potential, controlled costs

### 6. **Circuit Breaker Pattern**
- **Decision**: Circuit breakers for all external API calls
- **Rationale**: Financial APIs can be unreliable; need protection from cascading failures
- **Impact**: Improved system stability

### 7. **Horizontal Scaling Ready**
- **Decision**: Stateless design with external configuration
- **Rationale**: Enables multiple instances for high-volume screening
- **Impact**: Supports growth and reliability requirements
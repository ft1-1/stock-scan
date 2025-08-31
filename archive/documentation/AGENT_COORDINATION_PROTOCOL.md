# Agent Coordination Protocol - Phase 1

## Overview
This document establishes the coordination protocol between specialist agents during Phase 1 implementation of the options screening application.

## Foundation Status ✅
The lead agent has completed the core foundation infrastructure:

### Completed Infrastructure
- **Data Models**: Complete Pydantic models with validation (`/src/models/`)
- **Configuration**: Centralized settings with environment support (`/config/settings.py`)
- **Provider Framework**: Abstract interfaces with rate limiting and circuit breakers (`/src/providers/`)
- **Workflow Engine**: 7-step orchestration with checkpointing (`/src/screener/`)
- **Error Handling**: Comprehensive exception framework
- **Testing Foundation**: pytest configuration and fixtures (`/tests/conftest.py`)
- **Application Entry**: CLI interface and health checks (`/src/main.py`)

## Agent Task Assignments

### 🚀 IMMEDIATE PRIORITY TASKS

Each specialist agent should start with their Task 1 immediately, as these have no cross-dependencies.

## 📊 market-data-specialist: PRIORITY 1

### Task 1: EODHD Provider Implementation
**Files to Create**: 
- `/src/providers/eodhd_provider.py`
- `/src/providers/eodhd_client.py`

**Implementation Requirements**:
1. Inherit from `MarketDataProvider` base class
2. Implement all abstract methods:
   - `screen_stocks(criteria: ScreeningCriteria) -> List[str]`
   - `get_stock_quote(symbol: str) -> Optional[StockQuote]`
   - `get_stock_quotes(symbols: List[str]) -> Dict[str, StockQuote]`
   - `get_options_chain(symbol: str, **filters) -> List[OptionContract]`
   - `get_fundamental_data(symbol: str) -> Optional[FundamentalData]`

3. Use the configuration system: `self.config = settings.get_provider_config("eodhd")`
4. Implement proper error handling with provider exceptions
5. Follow the rate limiting: 1000 requests/minute
6. Return standardized data models (StockQuote, OptionContract, etc.)

**Integration Points**:
- Uses: `config.settings`, `models.*`, `providers.base_provider`
- Provides: Market data for analytics and workflow

### Task 2: MarketData.app Provider Implementation  
**Files to Create**:
- `/src/providers/marketdata_provider.py`
- `/src/providers/marketdata_client.py`

**Dependencies**: Task 1 completion for provider factory registration

## 🧮 options-quant-analyst: PRIORITY 1

### Task 1: Technical Indicator Engine
**Files to Create**:
- `/src/analytics/__init__.py`
- `/src/analytics/technical_indicators.py`
- `/src/analytics/calculation_engine.py`

**Implementation Requirements**:
1. Create indicator calculation functions returning `TechnicalIndicators` model
2. Functions needed:
   - `calculate_rsi(prices: List[float], period: int = 14) -> float`
   - `calculate_macd(prices: List[float], fast=12, slow=26, signal=9) -> tuple`
   - `calculate_bollinger_bands(prices: List[float], period=20, std_dev=2) -> tuple`
   - `calculate_sma(prices: List[float], period: int) -> float`
   - `calculate_ema(prices: List[float], period: int) -> float`

3. Use pandas/numpy for efficient calculations
4. Return `TechnicalIndicators` model with validation
5. Handle edge cases (insufficient data, NaN values)

**Integration Points**:
- Uses: `models.TechnicalIndicators`, pandas, numpy
- Provides: Technical analysis for workflow step 3

### Task 2: Options Analysis Framework
**Dependencies**: Task 1 + market data provider availability

## 🤖 ai-integration-architect: PRIORITY 1

### Task 1: Claude AI Integration Framework
**Files to Create**:
- `/src/ai_analysis/__init__.py`
- `/src/ai_analysis/claude_client.py`
- `/src/ai_analysis/ai_provider.py`

**Implementation Requirements**:
1. Inherit from `AIProvider` base class
2. Implement required methods:
   - `analyze_opportunities(opportunities: List[ScreeningResult]) -> List[AIAnalysisResult]`
   - `estimate_cost(data_size: int) -> float`

3. Use anthropic library for Claude API integration
4. Implement cost tracking: `self._total_cost += response_cost`
5. Return `AIAnalysisResult` model with proper validation
6. Enforce daily cost limits from settings

**Integration Points**:
- Uses: `config.settings`, `models.AIAnalysisResult`, anthropic library
- Provides: AI analysis for workflow step 6

### Task 2: Data Formatting Pipeline
**Dependencies**: Task 1 completion for testing integration

## 🧪 options-qa-tester: PRIORITY 1

### Task 1: Provider Testing Framework
**Files to Create**:
- `/tests/unit/test_providers.py`
- `/tests/integration/test_eodhd.py`
- `/tests/integration/test_marketdata.py`

**Implementation Requirements**:
1. Unit tests for provider base classes
2. Integration tests with live APIs (marked with `@pytest.mark.api`)
3. Mock provider implementations for isolated testing
4. Test cases for error handling and edge cases
5. Performance benchmarking tests

**Integration Points**:
- Tests: All provider implementations
- Uses: `tests/conftest.py` fixtures

### Task 2: Analytics Testing Framework  
**Dependencies**: Analytics implementation from options-quant-analyst

## Coordination Workflow

### Daily Sync Process
1. **Morning Check-in (9:00 AM)**:
   - Each agent updates progress in shared todo tracking
   - Identifies any blockers or dependencies
   - Lead agent reviews and coordinates solutions

2. **Integration Check (2:00 PM)**:
   - Test cross-component integrations
   - Validate interfaces are working correctly
   - Address any compatibility issues

3. **Evening Review (5:00 PM)**:
   - Review day's progress
   - Plan next day's priorities
   - Update task assignments if needed

### Communication Protocol

#### For Technical Issues:
- **Provider-related issues**: Coordinate with market-data-specialist
- **Analytics/calculation issues**: Coordinate with options-quant-analyst  
- **AI integration issues**: Coordinate with ai-integration-architect
- **Testing/quality issues**: Coordinate with options-qa-tester

#### For Integration Points:
- **Provider → Analytics**: market-data-specialist + options-quant-analyst
- **Analytics → AI**: options-quant-analyst + ai-integration-architect
- **Testing → All**: options-qa-tester coordinates with all agents

### Code Integration Rules

1. **Interface Compliance**: All implementations must use the provided base classes and models
2. **Error Handling**: Use the established exception hierarchy in `providers.exceptions`
3. **Configuration**: Use `config.settings.get_provider_config()` for provider setup
4. **Logging**: Use `utils.logging_config.get_logger(__name__)` for consistent logging
5. **Testing**: All new code requires corresponding test coverage

### File Registration Process

When creating new providers or components:

1. **Provider Registration**:
```python
from providers import ProviderFactory
ProviderFactory.register_provider("eodhd", EodhdProvider)
```

2. **Workflow Step Registration**:
```python
from screener import WorkflowEngine  
engine = WorkflowEngine()
engine.register_step_executor(WorkflowStep.TECHNICAL_CALCULATION, TechnicalAnalysisExecutor())
```

3. **Model Exports**: Add new models to appropriate `__init__.py` files

### Success Criteria Checkpoints

Each agent should validate their implementation against these criteria:

#### market-data-specialist:
- [ ] EODHD provider passes health check
- [ ] Can retrieve quotes for 100 symbols in <10 seconds
- [ ] Error handling prevents crashes on bad data
- [ ] Rate limiting prevents API overuse

#### options-quant-analyst:
- [ ] Technical indicators match reference implementations (±0.1%)
- [ ] Can process 1000 symbols in <30 seconds
- [ ] Handles edge cases (insufficient data, etc.)
- [ ] Memory usage stays under 500MB

#### ai-integration-architect:
- [ ] Claude API integration works with cost tracking
- [ ] Response parsing accuracy >99%
- [ ] Daily cost limits enforced
- [ ] Error handling prevents cascading failures

#### options-qa-tester:
- [ ] Test suite runs in <60 seconds
- [ ] Integration tests pass with live APIs
- [ ] Test coverage >80% for implemented components
- [ ] Performance benchmarks established

### Blocker Resolution Process

If an agent encounters a blocker:

1. **Document the issue**: Specific error, attempted solutions, impact
2. **Check dependencies**: Is it waiting on another agent's work?
3. **Escalate to lead**: Post issue details for coordination
4. **Parallel work**: Continue on non-blocked tasks while waiting for resolution

### Integration Testing Schedule

- **Day 2**: Basic provider health checks
- **Day 3**: Provider + analytics integration
- **Day 4**: Analytics + AI integration  
- **Day 5**: End-to-end workflow testing
- **Day 6-7**: Performance testing and optimization

### Quality Gates

No agent should consider their Phase 1 tasks complete until:

1. All unit tests pass
2. Integration tests pass with live APIs (where applicable)
3. Performance benchmarks are met
4. Code review completed by lead agent
5. Documentation updated

This protocol ensures coordinated development while maintaining autonomy for each specialist agent to implement their domain expertise effectively.
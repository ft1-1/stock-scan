# Options Screening Application - Project Structure

## Complete Directory Structure

```
options-screener/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                           # Core business logic
│   │   ├── __init__.py
│   │   ├── screener.py                 # Main screening orchestrator
│   │   ├── option_selector.py          # Option selection algorithms
│   │   ├── technical_calculator.py     # Technical indicator calculations
│   │   └── workflow_manager.py         # 7-step workflow coordinator
│   │
│   ├── data_providers/                 # External API integrations
│   │   ├── __init__.py
│   │   ├── base_provider.py           # Abstract provider interface
│   │   ├── eodhd_provider.py          # EODHD API integration
│   │   ├── marketdata_provider.py     # MarketData.app integration
│   │   ├── provider_factory.py        # Provider instantiation
│   │   └── provider_manager.py        # Multi-provider coordination
│   │
│   ├── ai_integration/                 # LLM integration components
│   │   ├── __init__.py
│   │   ├── claude_client.py           # Claude API client
│   │   ├── data_formatter.py          # Data formatting for LLM
│   │   ├── prompt_builder.py          # Dynamic prompt construction
│   │   ├── response_parser.py         # LLM response processing
│   │   └── cost_manager.py            # Cost control and monitoring
│   │
│   ├── models/                         # Data models and schemas
│   │   ├── __init__.py
│   │   ├── stock_models.py            # Stock and market data models
│   │   ├── option_models.py           # Option contract models
│   │   ├── screening_models.py        # Screening criteria and results
│   │   ├── analysis_models.py         # Analysis results and insights
│   │   └── config_models.py           # Configuration data models
│   │
│   ├── utils/                          # Utility functions and helpers
│   │   ├── __init__.py
│   │   ├── logging_utils.py           # Centralized logging configuration
│   │   ├── retry_utils.py             # Retry mechanisms and circuit breakers
│   │   ├── validation_utils.py        # Data validation helpers
│   │   ├── date_utils.py              # Date/time utilities
│   │   └── financial_utils.py         # Financial calculations
│   │
│   ├── config/                         # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py                # Main configuration loader
│   │   ├── validation.py              # Configuration validation
│   │   └── defaults.py                # Default configuration values
│   │
│   ├── storage/                        # Data persistence layer
│   │   ├── __init__.py
│   │   ├── file_storage.py            # File-based storage (JSON, CSV)
│   │   ├── cache_manager.py           # Caching implementation
│   │   └── data_serializer.py         # Data serialization utilities
│   │
│   └── scheduler/                      # Job scheduling and execution
│       ├── __init__.py
│       ├── daily_scheduler.py         # Daily job coordination
│       ├── task_executor.py           # Task execution engine
│       └── monitoring.py              # Job monitoring and alerts
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration
│   │
│   ├── unit/                          # Unit tests
│   │   ├── test_screener.py
│   │   ├── test_option_selector.py
│   │   ├── test_technical_calculator.py
│   │   ├── test_data_providers.py
│   │   ├── test_ai_integration.py
│   │   └── test_models.py
│   │
│   ├── integration/                   # Integration tests
│   │   ├── test_provider_integration.py
│   │   ├── test_ai_workflow.py
│   │   ├── test_end_to_end.py
│   │   └── test_error_scenarios.py
│   │
│   └── fixtures/                      # Test data fixtures
│       ├── sample_stock_data.json
│       ├── sample_options_data.json
│       └── mock_responses/
│
├── scripts/                           # Deployment and utility scripts
│   ├── deploy.sh
│   ├── setup_environment.sh
│   ├── run_daily_scan.sh
│   ├── backup_data.sh
│   └── health_check.py
│
├── configs/                           # Configuration files
│   ├── production.yaml
│   ├── staging.yaml
│   ├── development.yaml
│   └── logging.yaml
│
├── data/                              # Data directory
│   ├── raw/                          # Raw API responses
│   ├── processed/                    # Processed screening results
│   ├── cache/                        # Cached data
│   ├── logs/                         # Application logs
│   └── exports/                      # Final output data
│
├── docs/                              # Documentation
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── deployment_guide.md
│   ├── user_manual.md
│   └── troubleshooting.md
│
└── monitoring/                        # Monitoring and alerting
    ├── dashboard_config.json
    ├── alert_rules.yaml
    └── metrics_collection.py
```

## Module Responsibilities

### Core Modules (`src/core/`)
- **screener.py**: Orchestrates the complete screening workflow, coordinates between providers
- **option_selector.py**: Implements option selection algorithms, Greeks analysis, liquidity filtering
- **technical_calculator.py**: Calculates technical indicators locally (RSI, MACD, Bollinger Bands, etc.)
- **workflow_manager.py**: Manages the 7-step workflow execution and error handling

### Data Providers (`src/data_providers/`)
- **base_provider.py**: Abstract interface ensuring consistent provider behavior
- **eodhd_provider.py**: EODHD API integration with screening, fundamentals, and options data
- **marketdata_provider.py**: MarketData.app integration with focus on options chains
- **provider_factory.py**: Creates appropriate provider instances based on configuration
- **provider_manager.py**: Manages multiple providers, failover, and data aggregation

### AI Integration (`src/ai_integration/`)
- **claude_client.py**: Claude API client with rate limiting and error handling
- **data_formatter.py**: Formats screening results for LLM consumption
- **prompt_builder.py**: Constructs dynamic prompts based on market data
- **response_parser.py**: Parses and validates LLM responses
- **cost_manager.py**: Monitors and controls AI API costs

### Models (`src/models/`)
- **stock_models.py**: Stock quotes, fundamental data, technical indicators
- **option_models.py**: Option contracts, Greeks, chains, strategies
- **screening_models.py**: Screening criteria, filters, results
- **analysis_models.py**: AI analysis results, ratings, insights
- **config_models.py**: Configuration validation and type safety

### Storage (`src/storage/`)
- **file_storage.py**: JSON/CSV export functionality
- **cache_manager.py**: Intelligent caching for API responses
- **data_serializer.py**: Consistent data serialization across modules

### Scheduler (`src/scheduler/`)
- **daily_scheduler.py**: Coordinates daily screening execution
- **task_executor.py**: Executes individual workflow tasks
- **monitoring.py**: Monitors job execution and sends alerts

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Injection**: Providers and services are injected for testability
3. **Interface Segregation**: Small, focused interfaces rather than large ones
4. **Configuration-Driven**: Behavior controlled through configuration files
5. **Error Isolation**: Failures in one component don't cascade to others
6. **Async-First**: Built for concurrent processing and high throughput
7. **Test-Driven**: Comprehensive test coverage for all components
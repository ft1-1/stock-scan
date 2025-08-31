# Claude Code Assistant Configuration

This file contains essential information for Claude Code to understand and work with the Options Screening Application codebase effectively.

## Project Overview

This is a Python-based options screening application that analyzes stocks and options data to identify profitable trading opportunities. The system integrates with multiple data providers (EODHD, MarketData API) and uses AI analysis (Claude) for enhanced scoring.

### Key Directories
- `src/` - Main application source code
- `tests/` - Test suites (unit, integration, e2e, performance)
- `config/` - Configuration management
- `data/` - Output and cache directories
- `examples/` - Example scripts and integrations
- `docs/` - Documentation

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m e2e           # End-to-end tests only

# Run with coverage
pytest --cov=src --cov-report=html

# Test specific components
./test_app.sh           # Run all application tests
python test_minimal.py  # Test basic imports
```

### Application Entry Points

#### Simplified Testing (Recommended for Development)
```bash
# Health check
python main_simple.py health

# Screen specific symbols
python main_simple.py screen --symbols AAPL,MSFT,NVDA

# Run all tests
python main_simple.py test
```

#### Full Application
```bash
# Initialize application
python src/main.py init

# Health check
python src/main.py health

# Screen with AI analysis
python src/main.py screen --symbols AAPL,TSLA --enable-ai

# Screen with custom parameters
python src/main.py screen --min-price 50 --max-price 200
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration

### Environment Setup
1. Copy `.env.example` to `.env`
2. Configure required API keys:
   - `SCREENER_EODHD_API_KEY` (Required)
   - `SCREENER_MARKETDATA_API_KEY` (Required)  
   - `SCREENER_CLAUDE_API_KEY` (Optional, for AI analysis)

### Key Settings
- `SCREENER_SPECIFIC_SYMBOLS` - Override screening to specific symbols
- `SCREENER_VERBOSE=true` - Enable detailed logging
- `SCREENER_TEST_MODE=true` - Enable test mode
- `SCREENER_CLAUDE_DAILY_COST_LIMIT` - AI cost limits

## Project Architecture

### Core Modules
- `src/main.py` - Main application CLI
- `src/screener/` - Workflow engine and coordination
- `src/providers/` - Data provider integrations (EODHD, MarketData)
- `src/analytics/` - Technical analysis and scoring
- `src/ai_analysis/` - Claude AI integration
- `src/models/` - Data models and contracts
- `src/utils/` - Utilities and helpers

### Key Classes
- `OptionsScreenerApp` - Main application controller (src/main.py:34)
- `WorkflowEngine` - Orchestrates screening workflow  
- `ScreeningCoordinator` - Coordinates screening steps
- `ClaudeClient` - AI analysis client (src/ai_analysis/claude_client.py)

## Development Workflow

### Making Changes
1. Always start with health check: `python main_simple.py health`
2. Test changes with simplified app first
3. Run relevant tests before committing
4. Use the full application for complete workflow testing

### Adding New Features
1. Check existing patterns in similar modules
2. Follow the established project structure
3. Add appropriate tests (unit/integration)
4. Update configuration if needed
5. Document API changes

### Debugging
- Use `SCREENER_VERBOSE=true` for detailed logs
- Check `logs/` directory for application logs
- Use `python main_simple.py health` for configuration issues
- Test individual components with simplified scripts

## Common Tasks

### Add New Technical Indicator
1. Add indicator to `src/analytics/technical_indicators.py`
2. Update scoring in `src/analytics/scoring_models.py`
3. Add tests in `tests/unit/analytics/`

### Add New Data Provider
1. Extend `src/providers/base_provider.py`
2. Implement provider-specific client
3. Update workflow configuration
4. Add integration tests

### Modify AI Analysis
1. Update prompts in `src/ai_analysis/prompt_templates.py`
2. Modify parsing in `src/ai_analysis/response_parser.py`
3. Test with mock responses first

## Dependencies

### Core Dependencies
- `aiohttp` - Async HTTP requests
- `pydantic` - Data validation and models
- `pandas`/`numpy` - Data processing
- `anthropic` - Claude AI client
- `click` - CLI framework

### Development Dependencies  
- `pytest` - Testing framework
- `black` - Code formatting
- `mypy` - Type checking
- `flake8` - Linting

## Testing Strategy

The project uses pytest with comprehensive test markers:
- `@pytest.mark.unit` - Fast isolated tests
- `@pytest.mark.integration` - Component integration tests
- `@pytest.mark.api` - Tests requiring live API access
- `@pytest.mark.ai` - Tests requiring AI/Claude access
- `@pytest.mark.slow` - Long-running tests

## Important Notes

### Security
- Never commit API keys or sensitive data
- Use environment variables for all configuration
- API keys are stored in `.env` (git-ignored)

### Performance
- Application supports concurrent processing
- Caching is enabled by default for API responses
- Rate limiting is configured for all providers

### AI Integration
- Claude analysis is optional (requires API key)
- Cost limits are enforced (`SCREENER_CLAUDE_DAILY_COST_LIMIT`)
- Mock mode available for development

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure you're in the project root and venv is activated
2. **API key errors**: Check `.env` file and key validity
3. **Path errors**: Always run from project root directory
4. **Module not found**: Install requirements: `pip install -r requirements.txt`

### Getting Help
- Check `RUN_INSTRUCTIONS.md` for setup guidance
- Review `DEVELOPMENT_PLAN.md` for architecture details
- Look at example scripts in `examples/` directory
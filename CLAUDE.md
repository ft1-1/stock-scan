# Claude Code Assistant Configuration

This file contains essential information for Claude Code to understand and work with the Stock Screening Application codebase effectively.

## Project Overview

This is a Python-based stock screening application that analyzes stocks using momentum-squeeze technical patterns to identify profitable stock purchase opportunities. The system integrates with multiple data providers (EODHD, MarketData API) and uses AI analysis (Claude) for enhanced scoring and risk assessment.

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

#### Production Running (Recommended)
```bash
# Run full production screening with all symbols
python run_production.py

# Run with specific symbols
SCREENER_SPECIFIC_SYMBOLS="AAPL,TSLA" python run_production.py

# Run with AI analysis filtering (minimum score threshold)
SCREENER_SPECIFIC_SYMBOLS="AAPL,TSLA" SCREENER_AI_ANALYSIS_MIN_SCORE=40.0 python run_production.py
```

#### Simplified Testing (Development)
```bash
# Health check
python main_simple.py health

# Screen specific symbols
python main_simple.py screen --symbols AAPL,MSFT,NVDA

# Run all tests
python main_simple.py test
```

#### Full Application (Legacy)
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
- `SCREENER_SPECIFIC_SYMBOLS` - Override screening to specific symbols (e.g., "AAPL,TSLA")
- `SCREENER_AI_ANALYSIS_MIN_SCORE` - Minimum score threshold for AI analysis (default 30.0)
- `SCREENER_VERBOSE=true` - Enable detailed logging
- `SCREENER_TEST_MODE=true` - Enable test mode
- `SCREENER_CLAUDE_DAILY_COST_LIMIT` - AI cost limits
- `SCREENER_MAILGUN_DOMAIN` - Email notification domain
- `SCREENER_MAILGUN_API_KEY` - Email notification API key

## Project Architecture

### Core Modules
- `run_production.py` - Production workflow runner
- `src/main.py` - Main application CLI (legacy)
- `src/screener/` - Workflow engine and coordination
- `src/providers/` - Data provider integrations (EODHD, MarketData)
- `src/analytics/` - Technical analysis, scoring, and risk assessment
- `src/ai_analysis/` - Claude AI integration for stock analysis
- `src/models/` - Data models and contracts
- `src/notifications/` - Email notification system
- `src/utils/` - Utilities and helpers

### Key Classes
- `StockScreenerApp` - Main application controller (src/main.py:34)
- `WorkflowEngine` - Orchestrates screening workflow  
- `ScreeningCoordinator` - Coordinates screening steps
- `LocalRatingSystem` - Stock scoring and eligibility (src/analytics/local_rating_system.py)
- `ClaudeClient` - AI analysis client (src/ai_analysis/claude_client.py)
- `MailgunClient` - Email notification client (src/notifications/mailgun_client.py)

## Development Workflow

### Making Changes
1. Always start with health check: `python main_simple.py health`
2. Test changes with specific symbols first: `SCREENER_SPECIFIC_SYMBOLS="AAPL,TSLA" python run_production.py`
3. Run relevant tests before committing
4. Use the full production workflow for complete testing: `python run_production.py`

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
2. Update scoring in `src/analytics/scoring_models.py` and `src/analytics/local_rating_system.py`
3. Add tests in `tests/unit/analytics/`

### Add New Data Provider
1. Extend `src/providers/base_provider.py`
2. Implement provider-specific client
3. Update workflow configuration
4. Add integration tests

### Modify AI Analysis
1. Update prompts in `src/ai_analysis/prompt_templates.py`
2. Modify parsing in `src/ai_analysis/response_parser.py`
3. Test with specific symbols first: `SCREENER_SPECIFIC_SYMBOLS="AAPL" python run_production.py`

### Update Risk Assessment
1. Modify risk calculations in `src/analytics/risk_assessment.py`
2. Update email formatting in `src/notifications/mailgun_client.py`
3. Test position sizing calculations

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
- Claude analysis is used for stock evaluation and risk assessment
- Cost limits are enforced (`SCREENER_CLAUDE_DAILY_COST_LIMIT`)
- Minimum score threshold filtering (`SCREENER_AI_ANALYSIS_MIN_SCORE`)
- Mock mode available for development

### Stock Screening Strategy
- Focus on momentum-squeeze technical patterns
- TTM Squeeze indicator for volatility contraction
- Momentum analysis across multiple timeframes (21D, 63D, 126D)
- Risk assessment using ATR (Average True Range)
- Position sizing based on Value at Risk (VaR) calculations

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
- Review recent workflow results in `data/output/` directory
- Check AI analysis details in `data/ai_analysis/` directory

## Recent Changes (Stock-Focused Transformation)

### Major Updates
- **Removed Options Dependency**: Transformed from options-centric to pure stock-purchase workflow
- **Updated Scoring Model**: Redistributed weights (35% technical, 35% momentum, 20% squeeze, 10% quality)
- **Enhanced Risk Assessment**: Added ATR-based stop-loss and position sizing calculations
- **AI Prompt Updates**: Claude prompts now focus on stock analysis and risk management
- **Email Notifications**: Updated to show risk metrics and position sizing instead of options contracts

### Key Files Modified
- `src/analytics/scoring_models.py` - Removed options scoring component
- `src/analytics/local_rating_system.py` - Enhanced stock eligibility and quality scoring
- `src/screener/steps/technical_analysis_step.py` - Added comprehensive risk assessment
- `src/screener/steps/local_ranking_step.py` - Fixed validation for new score components
- `src/ai_analysis/prompt_templates.py` - Updated for stock-focused analysis
- `src/notifications/mailgun_client.py` - Updated email format for risk metrics

### Workflow Steps
1. **Stock Screening** - Filter by basic criteria (market cap, price, volume)
2. **Data Collection** - Gather market data and fundamentals
3. **Technical Analysis** - Calculate indicators and risk metrics
4. **Local Ranking** - Score and rank opportunities, select top candidates
5. **AI Analysis** - Claude evaluates qualified stocks for purchase decisions
6. **Result Processing** - Generate reports and send email notifications

### Testing
- Successfully tested with AAPL and TSLA
- Complete end-to-end workflow validation
- Email delivery with updated stock-focused formatting
- Claude AI cost: ~$0.05 per analysis session
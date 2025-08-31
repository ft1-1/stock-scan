# Quick Start Guide

## Prerequisites Completed ✅
- `.env` file configured with API keys
- Python environment ready

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Initialize the application:**
```bash
python src/main.py init
```

## Running the Application

### 1. Test with Specific Symbols (Recommended First Run)
```bash
# Test with a few liquid stocks
export SCREENER_SPECIFIC_SYMBOLS=AAPL,MSFT,NVDA
python src/main.py screen

# Or pass symbols directly
python src/main.py screen --symbols AAPL,MSFT,NVDA
```

### 2. Run Full Market Screening
```bash
# Screen entire market with default criteria
python src/main.py screen

# With AI analysis enabled (requires Claude API key)
python src/main.py screen --enable-ai
```

### 3. Custom Screening Criteria
```bash
# Custom price range
python src/main.py screen --min-price 10 --max-price 100

# Specific symbols with AI
python src/main.py screen --symbols TSLA,AMD,META --enable-ai
```

## Health Check
```bash
# Verify configuration and API connectivity
python src/main.py health
```

## Output
- Results are saved to: `data/output/results_YYYYMMDD_HHMMSS.json`
- Logs are saved to: `logs/application.log`
- Errors are saved to: `logs/errors.log`

## Monitoring Progress
The application provides detailed console output showing:
- 7-step workflow progress
- API calls and cache hits
- Symbol processing status
- Results summary table

## Troubleshooting

### No results found?
- Check if symbols meet filter criteria in `.env`
- Verify API keys are correct
- Check `logs/errors.log` for issues

### API errors?
- Verify API keys in `.env`
- Check rate limits aren't exceeded
- Ensure internet connectivity

### To see more detail:
```bash
# Enable debug mode
export SCREENER_DEBUG=true
export SCREENER_LOG_LEVEL=DEBUG
python src/main.py screen --symbols AAPL
```

## Daily Workflow Example
```bash
# Morning scan for opportunities
export SCREENER_SPECIFIC_SYMBOLS=SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,AMZN
python src/main.py screen --enable-ai

# Review results
cat data/output/results_*.json | jq '.[0:5]'  # Top 5 results
```

## Environment Variables Reference
Key settings you can adjust in `.env`:
- `SCREENER_SPECIFIC_SYMBOLS`: Comma-separated symbols to screen
- `SCREENER_MIN_MARKET_CAP`: Minimum market cap filter
- `SCREENER_MIN_OPTION_VOLUME`: Minimum option volume required
- `SCREENER_VERBOSE`: Enable/disable detailed console output
- `SCREENER_CLAUDE_DAILY_COST_LIMIT`: Max AI spending per day

## Next Steps
1. Start with 2-3 known symbols to test
2. Verify results match expectations
3. Gradually expand to more symbols
4. Enable AI analysis for top opportunities
5. Schedule daily runs with cron (see `docs/`)
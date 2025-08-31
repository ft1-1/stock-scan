# Running the Options Screening Application

## ✅ Setup Complete - Ready to Test!

### Quick Start (Simplified Version)

The application has been set up with multiple entry points. Start with the simplified version for testing:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run health check
python main_simple.py health

# 3. Test with mock data
python main_simple.py test

# 4. Screen specific symbols
python main_simple.py screen --symbols AAPL,MSFT,NVDA
```

### Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `main_simple.py` | Simplified version for testing | `python main_simple.py [health\|screen\|test]` |
| `test_minimal.py` | Test basic imports | `python test_minimal.py` |
| `test_app.sh` | Run all tests | `./test_app.sh` |
| `run.py` | Production launcher (when ready) | `python run.py [command]` |

### Testing Workflow

1. **Verify Installation**:
```bash
python test_minimal.py
```
Expected: All imports should pass

2. **Check Configuration**:
```bash
python main_simple.py health
```
Expected: Shows API key status and directories

3. **Run Mock Screening**:
```bash
python main_simple.py screen --symbols AAPL,TSLA
```
Expected: Creates test results in `data/output/`

### Environment Variables

Make sure your `.env` file has:
```env
SCREENER_EODHD_API_KEY=your_actual_key
SCREENER_MARKETDATA_API_KEY=your_actual_key
SCREENER_SPECIFIC_SYMBOLS=AAPL,MSFT,NVDA  # Optional
SCREENER_VERBOSE=true  # For detailed output
```

### Troubleshooting

**Import Errors?**
- Make sure you're in the venv: `source venv/bin/activate`
- Install packages: `pip install -r requirements-minimal.txt`

**API Key Errors?**
- Check `.env` file exists and has your keys
- Keys should start with `SCREENER_` prefix

**Path Errors?**
- Always run from project root: `/home/deployuser/stock-scan/stock-scanner/`
- Use the provided scripts, not `src/main.py` directly

### Next Steps

Once basic tests pass:

1. **Add Real API Testing** (when ready):
   - The simplified app currently uses mock data
   - Real API integration is in the full app

2. **Run Full Application** (when debugged):
   - Use `python run.py` for the complete workflow
   - Includes all 7 steps of processing

3. **Schedule Daily Runs**:
   - Use cron for automated daily screening
   - Example: `0 9 * * * cd /path/to/project && ./run_daily.sh`

### Current Status

✅ **Working**:
- Basic imports and configuration
- Health check functionality  
- Mock screening with output

⚠️ **In Progress**:
- Full workflow with real APIs
- Complex module dependencies

❌ **Not Yet Tested**:
- Real EODHD/MarketData API calls
- AI integration with Claude
- Full 7-step workflow

### Commands Summary

```bash
# Always start with:
cd /home/deployuser/stock-scan/stock-scanner
source venv/bin/activate

# Then run:
python main_simple.py health        # Check setup
python main_simple.py test          # Run all tests
python main_simple.py screen        # Screen with env symbols
python main_simple.py screen --symbols AAPL,MSFT  # Screen specific

# For debugging:
./test_app.sh                       # Run test suite
python test_minimal.py               # Test imports only
```
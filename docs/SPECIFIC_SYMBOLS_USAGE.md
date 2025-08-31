# Running with Specific Symbols

The options screening application supports running with specific symbols instead of screening the entire market. This is useful for:
- Testing individual stocks
- Focused analysis on a watchlist
- Debugging specific edge cases
- Running quick scans on known opportunities

## Methods to Specify Symbols

### 1. Environment Variable (Recommended for Regular Use)

Set the `SCREENER_SPECIFIC_SYMBOLS` environment variable in your `.env` file:

```bash
# .env file
SCREENER_SPECIFIC_SYMBOLS=AAPL,MSFT,GOOGL,TSLA,NVDA
```

Then run the application normally:

```bash
python src/main.py screen
```

### 2. Command Line Option (For One-off Runs)

Pass symbols directly via the CLI:

```bash
python src/main.py screen --symbols AAPL,MSFT,GOOGL
```

The CLI option overrides the environment variable if both are set.

### 3. Programmatic Usage

When using the application programmatically:

```python
from src.models import ScreeningCriteria
from src.main import OptionsScreenerApp

app = OptionsScreenerApp()
await app.initialize()

criteria = ScreeningCriteria(
    specific_symbols=["AAPL", "MSFT", "GOOGL"],
    # Other criteria still apply to the specific symbols
    min_option_volume=100,
    min_open_interest=500
)

result = await app.run_screening(criteria)
```

## Behavior with Specific Symbols

When specific symbols are provided:

1. **Market Screening is Bypassed**: The app won't run general market screening
2. **Direct Analysis**: Goes directly to analyzing the specified symbols
3. **Filters Still Apply**: Technical and options filters still apply to the specific symbols
4. **Faster Execution**: Much faster since it's not screening thousands of stocks
5. **Same Output Format**: Results are in the same format as full screening

## Example Use Cases

### Testing a New Strategy
```bash
# Test on a few liquid stocks first
SCREENER_SPECIFIC_SYMBOLS=SPY,QQQ,AAPL python src/main.py screen --enable-ai
```

### Daily Watchlist
```bash
# Your regular watchlist
export SCREENER_SPECIFIC_SYMBOLS=NVDA,AMD,TSLA,META,AMZN,GOOGL,MSFT,AAPL
python src/main.py screen
```

### Debugging a Specific Stock
```bash
# Debug why a stock isn't showing up
python src/main.py screen --symbols KSS --min-price 0 --max-price 1000
```

### Batch Processing
```bash
# Process different groups separately
SCREENER_SPECIFIC_SYMBOLS=AAPL,MSFT,GOOGL python src/main.py screen > tech_giants.json
SCREENER_SPECIFIC_SYMBOLS=JPM,BAC,WFC python src/main.py screen > banks.json
SCREENER_SPECIFIC_SYMBOLS=XOM,CVX,COP python src/main.py screen > energy.json
```

## Integration with Other Features

- **AI Analysis**: Works normally with specific symbols
- **Technical Indicators**: All calculations run on specified symbols
- **Options Selection**: Finds best options for each symbol
- **Caching**: Cached data is used if available
- **Rate Limiting**: Same rate limits apply

## Performance Considerations

- Specific symbol screening is **10-100x faster** than full market screening
- Ideal for development and testing
- Can run more frequently without hitting API limits
- Lower AI costs when using Claude analysis

## Tips

1. **Start Small**: Test with 2-3 symbols before running on larger lists
2. **Use Known Liquid Stocks**: For testing, use highly liquid stocks like SPY, AAPL
3. **Monitor Logs**: Check logs to ensure symbols are being processed correctly
4. **Validate Symbols**: Ensure symbols are valid and traded on supported exchanges
5. **Case Insensitive**: Symbols are automatically converted to uppercase

## Troubleshooting

**No results for specific symbols?**
- Check if the symbols meet your filter criteria (price, volume, etc.)
- Verify the symbols are valid and currently traded
- Check logs for any API errors
- Ensure options data is available for the symbols

**Environment variable not working?**
- Ensure the `.env` file is in the project root
- Check the variable name is exactly `SCREENER_SPECIFIC_SYMBOLS`
- Verify there are no spaces around the commas
- Restart the application after changing `.env`
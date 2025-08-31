#!/usr/bin/env python3
"""Add comprehensive logging to track MarketData usage."""

import os

def add_logging_to_production():
    """Add logging to show when MarketData is being used."""
    
    print("Adding MarketData usage logging...")
    
    # Read the current file
    with open("run_simple_production.py", "r") as f:
        lines = f.readlines()
    
    # Find and update the options fetching section
    updated = False
    for i, line in enumerate(lines):
        if "📊 Fetching options chain from MarketData..." in line:
            # Add more detailed logging after this line
            insert_lines = [
                '        print(f"   🔍 MarketData API Call:")\n',
                f'        print(f"      URL: {{url}}")\n',
                f'        print(f"      Date Range: {{from_date}} to {{to_date}}")\n',
                f'        print(f"      Filters: minOI=100, side=call")\n',
            ]
            # Insert after the current line
            if i + 1 < len(lines) and "🔍 MarketData API Call:" not in lines[i + 1]:
                lines[i:i+1] = [line] + insert_lines
                updated = True
                break
    
    if updated:
        # Write back
        with open("run_simple_production.py", "w") as f:
            f.writelines(lines)
        print("✅ Added detailed MarketData logging to run_simple_production.py")
    else:
        print("⚠️  Logging already exists or pattern not found")
    
    # Create a summary script
    summary = '''#!/usr/bin/env python3
"""Summary of MarketData Integration Status."""

print("""
================================================================================
MARKETDATA INTEGRATION STATUS SUMMARY
================================================================================

✅ WHAT'S WORKING:
-----------------
1. MarketData API is accessible and responding
2. Options data is being fetched (contracts, strikes, Greeks)
3. Authentication is working with your API key
4. The integration code is calling MarketData (not just EODHD)

⚠️ CURRENT BEHAVIOR:
--------------------
1. Getting LIVE data (status 200) instead of CACHED (status 203)
2. This costs 1 credit PER CONTRACT (46 credits for AAPL)
3. Cached feed would cost only 1 credit total

📊 CREDIT USAGE:
----------------
- Each run with 4 symbols (AAPL, MSFT, GOOGL, TSLA) uses ~200 credits
- With cached feed, this would be only 4 credits
- Check your usage at: https://www.marketdata.app/dashboard

🔧 POSSIBLE SOLUTIONS:
----------------------
1. Your account might not have cached feed access
   → Check subscription at https://www.marketdata.app/account
   
2. Use more aggressive filtering to reduce contracts:
   → Specific strikes instead of date ranges
   → Higher minOpenInterest (500+)
   → Tighter moneyness filters
   
3. Cache results locally to avoid repeated API calls
   → Save options data with timestamp
   → Reuse if less than 1 hour old

📝 RECOMMENDATIONS:
-------------------
1. For testing: Use single strike/expiration to minimize credits
2. For production: Consider upgrading for cached feed access
3. Add local caching layer to prevent redundant calls
4. Monitor credit usage closely at the dashboard

The good news: MarketData IS being called and working!
The optimization: Need to reduce credit consumption.
================================================================================
""")
'''
    
    with open("marketdata_status_summary.py", "w") as f:
        f.write(summary)
    
    print("\n✅ Created marketdata_status_summary.py")
    print("\nRun it to see the full status summary:")
    print("  python3 marketdata_status_summary.py")

if __name__ == "__main__":
    add_logging_to_production()
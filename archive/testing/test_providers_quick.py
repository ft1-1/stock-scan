#!/usr/bin/env python3
"""Quick test of provider initialization."""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

async def test_providers():
    """Test provider initialization."""
    
    # Check API keys
    eodhd_key = os.getenv("SCREENER_EODHD_API_KEY")
    marketdata_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    
    print("API Keys:")
    print(f"  EODHD: {eodhd_key[:10]}..." if eodhd_key else "  EODHD: Missing")
    print(f"  MarketData: {marketdata_key[:10]}..." if marketdata_key else "  MarketData: Missing")
    
    # Test EODHD
    print("\nTesting EODHD...")
    from src.providers.eodhd_client import EODHDClient
    
    eodhd_config = {
        'type': 'eodhd',
        'api_key': eodhd_key,
        'base_url': 'https://eodhd.com/api'
    }
    
    try:
        eodhd = EODHDClient(eodhd_config)
        quote = await eodhd.get_stock_quote("AAPL")
        if quote:
            print(f"  ✅ EODHD working: AAPL ${quote.price}")
        else:
            print(f"  ❌ EODHD: No quote returned")
    except Exception as e:
        print(f"  ❌ EODHD error: {e}")
    
    # Test MarketData
    print("\nTesting MarketData...")
    from src.providers.marketdata_client import MarketDataClient
    
    marketdata_config = {
        'type': 'marketdata',
        'api_key': marketdata_key,
        'base_url': 'https://api.marketdata.app/v1'
    }
    
    try:
        marketdata = MarketDataClient(marketdata_config)
        # Test with specific expiration to get cached data
        options = await marketdata.get_options_chain(
            symbol='AAPL',
            expiration_date='2025-09-19'
        )
        if options:
            print(f"  ✅ MarketData working: {len(options)} options")
        else:
            print(f"  ❌ MarketData: No options returned")
    except Exception as e:
        print(f"  ❌ MarketData error: {e}")
    
    print("\n" + "="*50)
    print("Provider test complete")

if __name__ == "__main__":
    asyncio.run(test_providers())
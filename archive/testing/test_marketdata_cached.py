#!/usr/bin/env python3
"""Test MarketData API with CACHED feed (paid subscription)."""

import os
import sys
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
load_dotenv()

async def test_marketdata_cached():
    """Test MarketData API with cached feed."""
    
    api_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    if not api_key:
        print("❌ SCREENER_MARKETDATA_API_KEY not found")
        return
    
    print(f"✅ API Key: {api_key[:15]}...")
    base_url = "https://api.marketdata.app/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print("\n" + "="*60)
    print("Testing MarketData API with CACHED feed")
    print("="*60)
    
    # Test 1: Get options chain with cached feed
    print("\n1. Testing Options Chain with CACHED feed...")
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/options/chain/AAPL/"
        
        # Get options 45-75 days out
        today = datetime.now().date()
        from_date = (today + timedelta(days=45)).isoformat()
        to_date = (today + timedelta(days=75)).isoformat()
        
        params = {
            'from': from_date,
            'to': to_date,
            'side': 'call',
            'minOpenInterest': 100,
            'feed': 'cached'  # Explicitly request cached feed
        }
        
        print(f"   URL: {url}")
        print(f"   Params: {json.dumps(params, indent=2)}")
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                print(f"   Status: {resp.status}")
                
                if resp.status == 203:
                    print("   ✅ Using CACHED feed (status 203)")
                elif resp.status == 200:
                    print("   ⚠️  Using LIVE feed (status 200)")
                
                data = await resp.json()
                
                if data.get('s') == 'ok':
                    strikes = data.get('strike', [])
                    deltas = data.get('delta', [])
                    ivs = data.get('iv', [])
                    
                    print(f"   ✅ Options found: {len(strikes)} contracts")
                    print(f"   💰 Credits used: 1 (cached feed)")
                    
                    if strikes and len(strikes) > 0:
                        print(f"\n   Sample contracts:")
                        for i in range(min(3, len(strikes))):
                            print(f"     Strike ${strikes[i]}: Delta={deltas[i] if i < len(deltas) else 'N/A':.3f}, IV={ivs[i] if i < len(ivs) else 'N/A':.2f}")
                    
                    # Check if Greeks are present
                    print(f"\n   Greeks available:")
                    print(f"     Delta: {'✅' if deltas else '❌'}")
                    print(f"     Gamma: {'✅' if data.get('gamma') else '❌'}")
                    print(f"     Theta: {'✅' if data.get('theta') else '❌'}")
                    print(f"     Vega: {'✅' if data.get('vega') else '❌'}")
                    
                else:
                    print(f"   ❌ Failed: {data}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 2: Direct provider test
    print("\n2. Testing through MarketDataClient...")
    try:
        from src.providers.marketdata_client import MarketDataClient
        
        config = {
            'type': 'MARKETDATA',  # Add required type field
            'api_key': api_key,
            'base_url': base_url,
            'default_feed': 'cached',  # Ensure cached feed
            'daily_credit_limit': 10000
        }
        
        client = MarketDataClient(config)
        
        # Test options chain
        options = await client.get_options_chain(
            symbol='AAPL',
            expiration_date=None,
            feed='cached'  # Explicitly use cached
        )
        
        print(f"   ✅ Client returned: {len(options)} contracts")
        
        if options and len(options) > 0:
            contract = options[0]
            print(f"   Sample: {contract.option_symbol}")
            print(f"     Strike: ${contract.strike}")
            print(f"     Delta: {contract.delta}")
            print(f"     Bid/Ask: ${contract.bid}/{contract.ask}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✅ If you see options data above, MarketData API is working!")
    print("📊 Check https://www.marketdata.app/dashboard for credit usage")
    print("\nWith cached feed (paid subscription):")
    print("  - 1 credit per request (regardless of contracts)")
    print("  - Status code 203 indicates cached data")
    print("  - Much more cost-effective for scanning")

if __name__ == "__main__":
    asyncio.run(test_marketdata_cached())
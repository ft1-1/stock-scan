#!/usr/bin/env python3
"""Final MarketData API test - determine best approach."""

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

async def test_marketdata_approaches():
    """Test different MarketData API approaches."""
    
    api_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    if not api_key:
        print("❌ SCREENER_MARKETDATA_API_KEY not found")
        return
    
    print(f"✅ API Key: {api_key[:15]}...")
    base_url = "https://api.marketdata.app/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print("\n" + "="*60)
    print("Testing MarketData API - Finding Best Approach")
    print("="*60)
    
    # Test 1: No feed parameter (let API decide)
    print("\n1. Testing with NO feed parameter (API default)...")
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/options/chain/AAPL/"
        
        params = {
            'expiration': '2025-09-19',  # Specific expiration
            'side': 'call',
            'strike': '220,225,230'  # Limit strikes
            # NO feed parameter
        }
        
        print(f"   Params: {json.dumps(params, indent=2)}")
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                print(f"   Status: {resp.status}")
                
                if resp.status == 203:
                    print("   ✅ Got CACHED data (status 203) - Good!")
                elif resp.status == 200:
                    print("   ⚠️  Got LIVE data (status 200)")
                
                data = await resp.json()
                
                if data.get('s') == 'ok':
                    strikes = data.get('strike', [])
                    print(f"   ✅ Success: {len(strikes)} contracts returned")
                else:
                    print(f"   ❌ Failed: {data.get('errmsg', 'Unknown error')}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 2: Explicitly request live feed
    print("\n2. Testing with feed='live' (explicit)...")
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/options/chain/AAPL/"
        
        params = {
            'expiration': '2025-09-19',
            'side': 'call',
            'strike': '225',  # Single strike to minimize cost
            'feed': 'live'
        }
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                print(f"   Status: {resp.status}")
                data = await resp.json()
                
                if data.get('s') == 'ok':
                    strikes = data.get('strike', [])
                    print(f"   ✅ Success: {len(strikes)} contracts (1 credit each)")
                else:
                    print(f"   ❌ Failed: {data.get('errmsg', 'Unknown error')}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 3: Through MarketDataClient
    print("\n3. Testing through MarketDataClient class...")
    try:
        from src.providers.marketdata_client import MarketDataClient
        
        # Try without specifying feed (let it use default)
        config = {
            'type': 'MARKETDATA',
            'api_key': api_key,
            'base_url': base_url,
            # Don't specify default_feed, let it use code default
            'daily_credit_limit': 10000
        }
        
        client = MarketDataClient(config)
        
        # Test with specific expiration to minimize credits
        options = await client.get_options_chain(
            symbol='AAPL',
            expiration_date='2025-09-19'
            # Don't specify feed parameter
        )
        
        if options:
            print(f"   ✅ Success: {len(options)} contracts returned")
            if len(options) > 0:
                contract = options[0]
                print(f"   Sample: Strike ${contract.strike}, Delta {contract.delta}")
        else:
            print(f"   ❌ No options returned")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Check what the production code is doing
    print("\n4. Checking production configuration...")
    from config.settings import get_settings
    settings = get_settings()
    
    print(f"   EODHD API Key: {'✅' if settings.eodhd_api_key else '❌'}")
    print(f"   MarketData API Key: {'✅' if settings.marketdata_api_key else '❌'}")
    print(f"   MarketData Base URL: {settings.marketdata_base_url}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\nBased on the tests above:")
    print("1. If status 203 works → You're getting cached data (good!)")
    print("2. If only status 200 works → You're on free tier (1 credit/contract)")
    print("3. If cached feed error → Your subscription doesn't include it")
    print("\nFor production:")
    print("- Use NO feed parameter (let API choose)")
    print("- Or explicitly use 'live' if on free tier")
    print("- Filter aggressively (specific strikes, expirations)")
    print("\n📊 Check credits at: https://www.marketdata.app/dashboard")

if __name__ == "__main__":
    asyncio.run(test_marketdata_approaches())
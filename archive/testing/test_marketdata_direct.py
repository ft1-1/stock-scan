#!/usr/bin/env python3
"""Test MarketData API connectivity and functionality directly."""

import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_marketdata_api():
    """Test MarketData API with various endpoints."""
    
    api_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    if not api_key:
        print("❌ SCREENER_MARKETDATA_API_KEY not found in environment")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    base_url = "https://api.marketdata.app/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Test 1: Get stock quote
    print("\n1. Testing Stock Quote (AAPL)...")
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/stocks/quotes/AAPL/"
        print(f"   URL: {url}")
        
        try:
            async with session.get(url, headers=headers) as resp:
                print(f"   Status: {resp.status}")
                data = await resp.json()
                print(f"   Response: {json.dumps(data, indent=2)[:500]}")
                
                if resp.status == 200 and data.get('s') == 'ok':
                    print(f"   ✅ Stock quote successful: ${data.get('last', [0])[0]}")
                else:
                    print(f"   ❌ Stock quote failed: {data}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 2: Get options expirations
    print("\n2. Testing Options Expirations (AAPL)...")
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/options/expirations/AAPL/"
        print(f"   URL: {url}")
        
        try:
            async with session.get(url, headers=headers) as resp:
                print(f"   Status: {resp.status}")
                data = await resp.json()
                
                if resp.status in [200, 203] and data.get('s') == 'ok':
                    expirations = data.get('expirations', [])
                    print(f"   ✅ Expirations found: {len(expirations)}")
                    print(f"   First 5: {expirations[:5]}")
                else:
                    print(f"   ❌ Expirations failed: {data}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 3: Get options chain (with Greeks)
    print("\n3. Testing Options Chain with Greeks (AAPL)...")
    async with aiohttp.ClientSession() as session:
        # Get options 45-75 days out
        today = datetime.now().date()
        from_date = (today + timedelta(days=45)).isoformat()
        to_date = (today + timedelta(days=75)).isoformat()
        
        url = f"{base_url}/options/chain/AAPL/"
        params = {
            'from': from_date,
            'to': to_date,
            'side': 'call',
            'minOpenInterest': 100,
            'feed': 'cached'  # Use cached for lower cost
        }
        
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                print(f"   Status: {resp.status}")
                data = await resp.json()
                
                if resp.status in [200, 203] and data.get('s') == 'ok':
                    # MarketData returns parallel arrays
                    strikes = data.get('strike', [])
                    deltas = data.get('delta', [])
                    ivs = data.get('iv', [])
                    
                    print(f"   ✅ Options found: {len(strikes)} contracts")
                    if strikes:
                        print(f"   Sample contract:")
                        print(f"     Strike: ${strikes[0]}")
                        print(f"     Delta: {deltas[0] if deltas else 'N/A'}")
                        print(f"     IV: {ivs[0] if ivs else 'N/A'}")
                else:
                    print(f"   ❌ Options chain failed: {data}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 4: Check which feed is being used
    print("\n4. Testing Feed Type (cached vs live)...")
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/options/chain/AAPL/"
        
        # Test cached feed
        params_cached = {
            'expiration': '2025-09-19',
            'side': 'call',
            'feed': 'cached'
        }
        
        try:
            async with session.get(url, headers=headers, params=params_cached) as resp:
                if resp.status == 203:
                    print(f"   ✅ Using CACHED feed (status 203) - 1 credit per request")
                elif resp.status == 200:
                    print(f"   ⚠️  Using LIVE feed (status 200) - 1 credit per contract!")
                else:
                    print(f"   ❌ Unexpected status: {resp.status}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("MarketData API Test Complete")
    print("="*60)
    
    # Check if API key is actually being used in production
    print("\n⚠️  IMPORTANT: Check your MarketData dashboard at:")
    print("   https://www.marketdata.app/dashboard")
    print("   to verify if credits are being consumed.")
    print("\n   If credits show 0 usage, the production code may not be")
    print("   actually calling MarketData API for options data.")

if __name__ == "__main__":
    asyncio.run(test_marketdata_api())
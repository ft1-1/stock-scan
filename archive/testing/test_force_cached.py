#!/usr/bin/env python3
"""Test forcing cached feed with explicit parameter."""

import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

async def test_cached_variations():
    api_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    url = "https://api.marketdata.app/v1/options/chain/AAPL/"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print("Testing different approaches to get CACHED data:\n")
    
    # Test 1: Explicit feed=cached
    print("1. WITH feed='cached' parameter:")
    params = {
        'expiration': '2025-09-19',
        'side': 'call', 
        'strike': '225',
        'feed': 'cached'  # Explicitly request cached
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as resp:
            print(f"   Status: {resp.status}")
            data = await resp.json()
            if resp.status == 402:
                print(f"   Error: {data.get('errmsg')}")
            elif data.get('s') == 'ok':
                print(f"   Success: {len(data.get('strike', []))} contracts")
    
    # Test 2: No feed parameter  
    print("\n2. WITHOUT feed parameter (should default to cached):")
    params = {
        'expiration': '2025-09-19',
        'side': 'call',
        'strike': '225'
        # NO feed parameter
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as resp:
            print(f"   Status: {resp.status}")
            if resp.status == 203:
                print("   ✅ CACHED data!")
            elif resp.status == 200:
                print("   ❌ LIVE data")
            data = await resp.json()
            if data.get('s') == 'ok':
                print(f"   Contracts: {len(data.get('strike', []))}")
    
    # Test 3: Different date to check if it's data availability
    print("\n3. Testing different expiration date:")
    params = {
        'expiration': '2025-08-22',  # Nearer expiration
        'side': 'call',
        'strike': '225'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as resp:
            print(f"   Status: {resp.status}")
            if resp.status == 203:
                print("   ✅ CACHED data!")
            elif resp.status == 200:
                print("   ❌ LIVE data")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    print("\nIf all tests return status 200 (live):")
    print("  → Your account may default to live feed")
    print("  → Contact MarketData support about cached feed access")
    print("\nIf feed='cached' returns 402 error:")
    print("  → Your subscription doesn't include cached feed")
    print("  → Need to upgrade subscription tier")
    print("\nCheck your plan at: https://www.marketdata.app/account/billing")

asyncio.run(test_cached_variations())
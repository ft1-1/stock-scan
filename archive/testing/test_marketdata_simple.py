#!/usr/bin/env python3
"""Simple test to verify MarketData cached feed access."""

import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

async def test():
    api_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    
    # Test with minimal parameters
    url = "https://api.marketdata.app/v1/options/chain/AAPL/"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Minimal params - no feed
    today = datetime.now().date()
    params = {
        'from': (today + timedelta(days=45)).isoformat(),
        'to': (today + timedelta(days=75)).isoformat(),
        'side': 'call',
        'minOpenInterest': 100
    }
    
    print(f"URL: {url}")
    print(f"Params: {json.dumps(params, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as resp:
            print(f"Status: {resp.status}")
            
            if resp.status == 203:
                print("✅ CACHED data (1 credit total)")
            elif resp.status == 200:
                print("⚠️  LIVE data (1 credit per contract!)")
            
            data = await resp.json()
            if data.get('s') == 'ok':
                strikes = data.get('strike', [])
                print(f"Contracts: {len(strikes)}")

asyncio.run(test())
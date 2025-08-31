#!/usr/bin/env python3
"""Minimal test without Claude to check data collection."""

import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

async def test_minimal():
    """Test data collection only."""
    
    eodhd_key = os.getenv("SCREENER_EODHD_API_KEY")
    marketdata_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    
    print("Testing AAPL data collection...")
    
    # Test EODHD
    print("\n1. EODHD Quote:")
    async with aiohttp.ClientSession() as session:
        url = f"https://eodhd.com/api/real-time/AAPL.US"
        params = {"api_token": eodhd_key, "fmt": "json"}
        
        try:
            async with session.get(url, params=params) as resp:
                print(f"   Status: {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   Price: ${data.get('close', 0)}")
                else:
                    text = await resp.text()
                    print(f"   Error: {text[:100]}")
        except Exception as e:
            print(f"   Exception: {e}")
    
    # Test MarketData
    print("\n2. MarketData Options:")
    async with aiohttp.ClientSession() as session:
        url = f"https://api.marketdata.app/v1/options/expirations/AAPL/"
        headers = {"Authorization": f"Bearer {marketdata_key}"}
        
        try:
            async with session.get(url, headers=headers) as resp:
                print(f"   Status: {resp.status}")
                if resp.status in [200, 203]:
                    data = await resp.json()
                    if data.get('s') == 'ok':
                        exps = data.get('expirations', [])
                        print(f"   Expirations: {len(exps)}")
                        if exps:
                            print(f"   First: {exps[0]}")
                else:
                    text = await resp.text()
                    print(f"   Error: {text[:100]}")
        except Exception as e:
            print(f"   Exception: {e}")
    
    print("\nDone!")

asyncio.run(test_minimal())

#!/usr/bin/env python
"""Test real API connections with your configured keys."""

import sys
import os
from pathlib import Path
import json
import asyncio
import aiohttp
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

class APITester:
    """Test real API connections."""
    
    def __init__(self):
        self.eodhd_key = os.getenv("SCREENER_EODHD_API_KEY")
        self.marketdata_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
        
    async def test_eodhd(self):
        """Test EODHD API connection."""
        print("\n📊 Testing EODHD API...")
        
        if not self.eodhd_key:
            print("❌ EODHD API key not found")
            return False
            
        url = f"https://eodhd.com/api/real-time/AAPL.US"
        params = {
            "api_token": self.eodhd_key,
            "fmt": "json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ EODHD API working!")
                        print(f"   AAPL Price: ${data.get('close', 'N/A')}")
                        print(f"   Volume: {data.get('volume', 'N/A'):,}")
                        return True
                    else:
                        print(f"❌ EODHD API error: Status {response.status}")
                        text = await response.text()
                        print(f"   Response: {text[:200]}")
                        return False
        except Exception as e:
            print(f"❌ EODHD API error: {e}")
            return False
    
    async def test_marketdata(self):
        """Test MarketData.app API connection."""
        print("\n📈 Testing MarketData.app API...")
        
        if not self.marketdata_key:
            print("❌ MarketData API key not found")
            return False
            
        url = "https://api.marketdata.app/v1/stocks/quotes/AAPL/"
        headers = {
            "Authorization": f"Bearer {self.marketdata_key}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status in [200, 203]:  # 203 = cached data
                        data = await response.json()
                        print(f"✅ MarketData API working! (Status {response.status})")
                        if 's' in data and data['s'] == 'ok':
                            # MarketData returns single values, not arrays for quotes
                            if 'last' in data:
                                last_price = data['last'][0] if isinstance(data['last'], list) else data['last']
                                print(f"   AAPL Last: ${last_price}")
                            if 'volume' in data:
                                volume = data['volume'][0] if isinstance(data['volume'], list) else data['volume']
                                print(f"   Volume: {volume:,}")
                        return True
                    else:
                        print(f"❌ MarketData API error: Status {response.status}")
                        text = await response.text()
                        print(f"   Response: {text[:200]}")
                        return False
        except Exception as e:
            print(f"❌ MarketData API error: {e}")
            return False
    
    async def test_eodhd_screening(self):
        """Test EODHD screening endpoint."""
        print("\n🔍 Testing EODHD Screening...")
        
        if not self.eodhd_key:
            print("❌ EODHD API key not found")
            return False
            
        url = "https://eodhd.com/api/screener"
        # Use simpler filter format that EODHD expects
        filters = {
            "market_capitalization": {"min": 1000000000},
            "close": {"min": 10},
            "volume": {"min": 100000}
        }
        params = {
            "api_token": self.eodhd_key,
            "sort": "market_capitalization.desc",
            "limit": 5,
            "offset": 0,
            "filters": json.dumps(filters)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        count = data.get('count', 0)
                        stocks = data.get('data', {}).get('data', [])
                        print(f"✅ EODHD Screening working!")
                        print(f"   Found {count} stocks matching criteria")
                        if stocks:
                            print(f"   Sample: {stocks[0].get('code', 'N/A')}")
                        return True
                    else:
                        print(f"❌ EODHD Screening error: Status {response.status}")
                        return False
        except Exception as e:
            print(f"❌ EODHD Screening error: {e}")
            return False
    
    async def test_marketdata_options(self):
        """Test MarketData options chain endpoint."""
        print("\n📊 Testing MarketData Options Chain...")
        
        if not self.marketdata_key:
            print("❌ MarketData API key not found")
            return False
            
        url = "https://api.marketdata.app/v1/options/expirations/AAPL/"
        headers = {
            "Authorization": f"Bearer {self.marketdata_key}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status in [200, 203]:  # 203 = cached data (preferred for cost)
                        data = await response.json()
                        print(f"✅ MarketData Options working! (Using cached data - Status {response.status})")
                        if 's' in data and data['s'] == 'ok':
                            expirations = data.get('expirations', [])
                            if expirations:
                                print(f"   Found {len(expirations)} expiration dates")
                                print(f"   Next: {expirations[0] if expirations else 'N/A'}")
                        return True
                    else:
                        print(f"❌ MarketData Options error: Status {response.status}")
                        return False
        except Exception as e:
            print(f"❌ MarketData Options error: {e}")
            return False

async def main():
    """Run all API tests."""
    print("="*60)
    print("REAL API CONNECTION TESTS")
    print("="*60)
    
    tester = APITester()
    
    results = {
        "EODHD Basic": await tester.test_eodhd(),
        "MarketData Basic": await tester.test_marketdata(),
        "EODHD Screening": await tester.test_eodhd_screening(),
        "MarketData Options": await tester.test_marketdata_options()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test:.<30} {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL API TESTS PASSED!")
        print("Your API keys are working correctly.")
    else:
        print("⚠️  Some API tests failed.")
        print("Check your API keys and try again.")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
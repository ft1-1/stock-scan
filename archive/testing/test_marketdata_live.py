#!/usr/bin/env python3
"""Test MarketData API with LIVE feed (free tier)."""

import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_marketdata_live_feed():
    """Test MarketData API with live feed for free tier."""
    
    api_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
    if not api_key:
        print("❌ SCREENER_MARKETDATA_API_KEY not found in environment")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    base_url = "https://api.marketdata.app/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print("\n" + "="*60)
    print("Testing MarketData API with LIVE feed (free tier)")
    print("="*60)
    
    # Test 1: Get specific expiration options with LIVE feed
    print("\n1. Testing Options Chain with LIVE feed (limited contracts)...")
    async with aiohttp.ClientSession() as session:
        # Get next monthly expiration
        url = f"{base_url}/options/chain/AAPL/"
        params = {
            'expiration': '2025-09-19',  # Specific expiration
            'side': 'call',
            'strike': '220,225,230',  # Limit to 3 strikes to minimize credits
            # NO 'feed' parameter - defaults to live
        }
        
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        print(f"   Note: Using LIVE feed (1 credit per contract)")
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                print(f"   Status: {resp.status}")
                data = await resp.json()
                
                if resp.status == 200 and data.get('s') == 'ok':
                    # MarketData returns parallel arrays
                    strikes = data.get('strike', [])
                    bids = data.get('bid', [])
                    asks = data.get('ask', [])
                    deltas = data.get('delta', [])
                    gammas = data.get('gamma', [])
                    thetas = data.get('theta', [])
                    vegas = data.get('vega', [])
                    ivs = data.get('iv', [])
                    volumes = data.get('volume', [])
                    ois = data.get('openInterest', [])
                    
                    print(f"   ✅ Options found: {len(strikes)} contracts")
                    
                    # Display the data in a table format
                    if strikes:
                        print("\n   Option Chain Data:")
                        print("   " + "-"*80)
                        print(f"   {'Strike':<10} {'Bid':<8} {'Ask':<8} {'Delta':<8} {'IV':<8} {'Volume':<10} {'OI':<10}")
                        print("   " + "-"*80)
                        
                        for i in range(len(strikes)):
                            strike = strikes[i] if i < len(strikes) else 'N/A'
                            bid = f"${bids[i]:.2f}" if i < len(bids) and bids[i] else 'N/A'
                            ask = f"${asks[i]:.2f}" if i < len(asks) and asks[i] else 'N/A'
                            delta = f"{deltas[i]:.3f}" if i < len(deltas) and deltas[i] else 'N/A'
                            iv = f"{ivs[i]:.2f}" if i < len(ivs) and ivs[i] else 'N/A'
                            volume = f"{volumes[i]:,}" if i < len(volumes) and volumes[i] else '0'
                            oi = f"{ois[i]:,}" if i < len(ois) and ois[i] else '0'
                            
                            print(f"   ${strike:<9} {bid:<8} {ask:<8} {delta:<8} {iv:<8} {volume:<10} {oi:<10}")
                        
                        print("\n   Greeks Available:")
                        print(f"     Gamma: {'✅' if gammas else '❌'}")
                        print(f"     Theta: {'✅' if thetas else '❌'}")
                        print(f"     Vega: {'✅' if vegas else '❌'}")
                        
                    print(f"\n   💰 Credits used: ~{len(strikes)} (1 per contract with live feed)")
                    
                else:
                    print(f"   ❌ Options chain failed: {data}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 2: Get options chain for date range (be careful with credits!)
    print("\n2. Testing Date Range Options (MINIMAL for credit conservation)...")
    async with aiohttp.ClientSession() as session:
        today = datetime.now().date()
        from_date = (today + timedelta(days=45)).isoformat()
        to_date = (today + timedelta(days=50)).isoformat()  # Only 5 days range
        
        url = f"{base_url}/options/chain/AAPL/"
        params = {
            'from': from_date,
            'to': to_date,
            'side': 'call',
            'moneyness': 'itm',  # Limit to ITM only
            'minOpenInterest': 1000,  # High OI filter
            'maxBidAskSpread': 0.10,  # Tight spread filter
            # NO 'feed' parameter - defaults to live
        }
        
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        print(f"   Note: Heavily filtered to minimize credit usage")
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                print(f"   Status: {resp.status}")
                data = await resp.json()
                
                if resp.status == 200 and data.get('s') == 'ok':
                    strikes = data.get('strike', [])
                    expirations = data.get('expiration', [])
                    
                    print(f"   ✅ Options found: {len(strikes)} contracts")
                    
                    if expirations:
                        unique_expirations = list(set(expirations))
                        print(f"   Expirations: {unique_expirations}")
                    
                    print(f"   💰 Credits used: ~{len(strikes)} (be careful with live feed!)")
                    
                else:
                    print(f"   ❌ Options chain failed: {data}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("IMPORTANT NOTES:")
    print("="*60)
    print("1. Free tier uses LIVE feed: 1 credit per contract")
    print("2. Cached feed (1 credit per request) requires paid subscription")
    print("3. Be VERY careful with filters to limit contracts returned")
    print("4. Consider using specific strikes/expirations to minimize credits")
    print("\n⚠️  Check https://www.marketdata.app/dashboard for credit usage")

if __name__ == "__main__":
    asyncio.run(test_marketdata_live_feed())
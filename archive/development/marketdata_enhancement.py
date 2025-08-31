# Enhanced MarketData fetching for production
# Add this to run_simple_production.py


async def fetch_marketdata_options_enhanced(self, symbol):
    """Fetch full options chain from MarketData with Greeks."""
    print(f"   📊 Fetching options chain from MarketData...")
    
    async with aiohttp.ClientSession() as session:
        # Get options 45-75 days out for best liquidity
        today = datetime.now().date()
        from_date = (today + timedelta(days=45)).isoformat()
        to_date = (today + timedelta(days=75)).isoformat()
        
        url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
        headers = {"Authorization": f"Bearer {self.marketdata_key}"}
        
        # DO NOT set feed parameter - gets cached data automatically
        params = {
            'from': from_date,
            'to': to_date,
            'side': 'call',
            'minOpenInterest': 100
        }
        
        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status in [200, 203]:
                data = await resp.json()
                if data.get('s') == 'ok':
                    strikes = data.get('strike', [])
                    
                    if resp.status == 203:
                        print(f"   ✅ Options (cached): {len(strikes)} contracts (1 credit total)")
                    else:
                        print(f"   ⚠️  Options (live): {len(strikes)} contracts ({len(strikes)} credits!)")
                    
                    # Return structured options data
                    return {
                        'contracts': len(strikes),
                        'strikes': strikes[:10],  # Sample strikes
                        'deltas': data.get('delta', [])[:10],
                        'ivs': data.get('iv', [])[:10],
                        'data_type': 'cached' if resp.status == 203 else 'live'
                    }
    
    return None

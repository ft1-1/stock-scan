#!/usr/bin/env python3
"""Test EODHD enhanced data collection directly."""

import asyncio
import aiohttp
from datetime import datetime, timedelta


async def test_eodhd_endpoints():
    """Test EODHD API endpoints directly."""
    
    api_key = "688d9b4a6b88c1.45343789"
    base_url = "https://eodhd.com/api"
    
    dates = {
        'today': datetime.now().strftime('%Y-%m-%d'),
        'six_months_ago': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    }
    
    async with aiohttp.ClientSession() as session:
        
        # Test economic events
        try:
            print('Testing economic events...')
            url = f'{base_url}/economic-events'
            params = {
                'api_token': api_key,
                'from': dates['six_months_ago'],
                'to': dates['today'],
                'country': 'US',
                'comparison': 'mom',
                'limit': 5,
                'fmt': 'json'
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f'✅ Economic events: {len(data) if isinstance(data, list) else "OK"}')
                else:
                    print(f'❌ Economic events: HTTP {resp.status}')
                    text = await resp.text()
                    print(f'   Response: {text[:200]}')
        except Exception as e:
            print(f'❌ Economic events failed: {e}')
        
        # Test news
        try:
            print('Testing news...')
            url = f'{base_url}/news'
            params = {
                'api_token': api_key,
                's': 'AAPL.US',
                'limit': 5,
                'fmt': 'json'
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f'✅ News: {len(data) if isinstance(data, list) else "OK"}')
                else:
                    print(f'❌ News: HTTP {resp.status}')
                    text = await resp.text()
                    print(f'   Response: {text[:200]}')
        except Exception as e:
            print(f'❌ News failed: {e}')
        
        # Test sentiment
        try:
            print('Testing sentiment...')
            url = f'{base_url}/sentiments'
            params = {
                'api_token': api_key,
                's': 'AAPL.US',
                'from': dates['six_months_ago'],
                'to': dates['today'],
                'fmt': 'json'
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f'✅ Sentiment: OK')
                else:
                    print(f'❌ Sentiment: HTTP {resp.status}')
                    text = await resp.text()
                    print(f'   Response: {text[:200]}')
        except Exception as e:
            print(f'❌ Sentiment failed: {e}')
        
        # Test fundamentals
        try:
            print('Testing fundamentals...')
            url = f'{base_url}/fundamentals/AAPL.US'
            params = {
                'api_token': api_key,
                'fmt': 'json'
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f'✅ Fundamentals: OK')
                else:
                    print(f'❌ Fundamentals: HTTP {resp.status}')
                    text = await resp.text()
                    print(f'   Response: {text[:200]}')
        except Exception as e:
            print(f'❌ Fundamentals failed: {e}')
        
        # Test earnings
        try:
            print('Testing earnings...')
            url = f'{base_url}/calendar/earnings'
            params = {
                'api_token': api_key,
                'symbols': 'AAPL.US',
                'from': dates['six_months_ago'],
                'to': dates['today'],
                'fmt': 'json'
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f'✅ Earnings: {len(data.get("earnings", [])) if isinstance(data, dict) else "OK"}')
                else:
                    print(f'❌ Earnings: HTTP {resp.status}')
                    text = await resp.text()
                    print(f'   Response: {text[:200]}')
        except Exception as e:
            print(f'❌ Earnings failed: {e}')


if __name__ == "__main__":
    asyncio.run(test_eodhd_endpoints())
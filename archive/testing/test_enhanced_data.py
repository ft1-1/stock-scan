#!/usr/bin/env python3
"""Test enhanced data collection from EODHD."""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.providers.provider_manager import ProviderManager
from config.settings import get_settings


async def test_enhanced_data():
    settings = get_settings()
    
    # Initialize provider
    provider_configs = {
        'eodhd': {
            'api_key': settings.eodhd_api_key,
            'base_url': settings.eodhd_base_url,
            'type': 'eodhd',
            'requests_per_minute': settings.eodhd_requests_per_minute
        }
    }
    
    provider_manager = ProviderManager(provider_configs)
    await provider_manager.initialize()
    
    eodhd = provider_manager.get_provider('eodhd')
    
    # Test economic events
    try:
        dates = {
            'today': datetime.now().strftime('%Y-%m-%d'),
            'six_months_ago': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        }
        
        print('Testing economic events...')
        response = await eodhd._make_request(
            'GET',
            f'{eodhd.base_url}/economic-events',
            params={
                'api_token': eodhd.api_key,
                'from': dates['six_months_ago'],
                'to': dates['today'],
                'country': 'US',
                'comparison': 'mom',
                'limit': 5,
                'fmt': 'json'
            }
        )
        if response:
            print(f'✅ Economic events: {len(response) if isinstance(response, list) else "OK"}')
        else:
            print('❌ Economic events: Empty response')
    except Exception as e:
        print(f'❌ Economic events failed: {e}')
    
    # Test news
    try:
        print('Testing news...')
        response = await eodhd._make_request(
            'GET',
            f'{eodhd.base_url}/news',
            params={
                'api_token': eodhd.api_key,
                's': 'AAPL.US',
                'limit': 5,
                'fmt': 'json'
            }
        )
        if response:
            print(f'✅ News: {len(response) if isinstance(response, list) else "OK"}')
        else:
            print('❌ News: Empty response')
    except Exception as e:
        print(f'❌ News failed: {e}')
    
    # Test sentiment
    try:
        print('Testing sentiment...')
        response = await eodhd._make_request(
            'GET',
            f'{eodhd.base_url}/sentiments',
            params={
                'api_token': eodhd.api_key,
                's': 'AAPL.US',
                'from': dates['six_months_ago'],
                'to': dates['today'],
                'fmt': 'json'
            }
        )
        if response:
            print(f'✅ Sentiment: OK')
        else:
            print('❌ Sentiment: Empty response')
    except Exception as e:
        print(f'❌ Sentiment failed: {e}')
    
    # Test fundamentals
    try:
        print('Testing fundamentals...')
        response = await eodhd._make_request(
            'GET',
            f'{eodhd.base_url}/fundamentals/AAPL.US',
            params={
                'api_token': eodhd.api_key,
                'fmt': 'json'
            }
        )
        if response:
            print(f'✅ Fundamentals: OK')
        else:
            print('❌ Fundamentals: Empty response')
    except Exception as e:
        print(f'❌ Fundamentals failed: {e}')
    
    # Test earnings
    try:
        print('Testing earnings...')
        response = await eodhd._make_request(
            'GET',
            f'{eodhd.base_url}/calendar/earnings',
            params={
                'api_token': eodhd.api_key,
                'symbols': 'AAPL.US',
                'from': dates['six_months_ago'],
                'to': dates['today'],
                'fmt': 'json'
            }
        )
        if response:
            print(f'✅ Earnings: {len(response.get("earnings", [])) if isinstance(response, dict) else "OK"}')
        else:
            print('❌ Earnings: Empty response')
    except Exception as e:
        print(f'❌ Earnings failed: {e}')

if __name__ == "__main__":
    asyncio.run(test_enhanced_data())
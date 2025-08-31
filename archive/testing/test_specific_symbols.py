#!/usr/bin/env python
"""Test real API calls for specific symbols only (no screening)."""

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

class SymbolTester:
    """Test APIs with specific symbols."""
    
    def __init__(self, symbols=None):
        self.eodhd_key = os.getenv("SCREENER_EODHD_API_KEY")
        self.marketdata_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
        self.symbols = symbols or ["AAPL", "MSFT", "NVDA"]
        
    async def get_eodhd_quote(self, symbol):
        """Get stock quote from EODHD."""
        url = f"https://eodhd.com/api/real-time/{symbol}.US"
        params = {
            "api_token": self.eodhd_key,
            "fmt": "json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "symbol": symbol,
                            "price": data.get('close'),
                            "volume": data.get('volume'),
                            "change": data.get('change_p'),
                            "source": "EODHD"
                        }
        except Exception as e:
            print(f"   Error fetching {symbol}: {e}")
        return None
    
    async def get_marketdata_options_expirations(self, symbol):
        """Get options expirations from MarketData."""
        url = f"https://api.marketdata.app/v1/options/expirations/{symbol}/"
        headers = {
            "Authorization": f"Bearer {self.marketdata_key}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status in [200, 203]:  # Accept cached
                        data = await response.json()
                        if 's' in data and data['s'] == 'ok':
                            expirations = data.get('expirations', [])
                            # Filter to next 3 months
                            if expirations:
                                return expirations[:6]  # Roughly 6 weekly/monthly expirations
        except Exception as e:
            print(f"   Error fetching options for {symbol}: {e}")
        return None
    
    async def test_symbol_workflow(self, symbol):
        """Test basic workflow for a single symbol."""
        print(f"\n📊 Testing {symbol}...")
        
        # 1. Get stock quote
        quote = await self.get_eodhd_quote(symbol)
        if quote:
            print(f"   ✅ Price: ${quote['price']:.2f}")
            print(f"   📈 Change: {quote['change']:.2f}%")
            print(f"   📊 Volume: {quote['volume']:,}")
        else:
            print(f"   ❌ Could not fetch quote")
            return False
        
        # 2. Get options expirations
        expirations = await self.get_marketdata_options_expirations(symbol)
        if expirations:
            print(f"   ✅ Options: {len(expirations)} expirations available")
            print(f"   📅 Next: {expirations[0]}")
        else:
            print(f"   ⚠️  No options data (may not have options)")
        
        return True

async def main():
    """Run tests for specific symbols."""
    print("="*60)
    print("SPECIFIC SYMBOL API TESTS")
    print("="*60)
    
    # Get symbols from environment or use defaults
    symbols_env = os.getenv("SCREENER_SPECIFIC_SYMBOLS")
    if symbols_env:
        symbols = [s.strip() for s in symbols_env.split(',')]
        print(f"Using symbols from environment: {', '.join(symbols)}")
    else:
        symbols = ["AAPL", "MSFT", "NVDA", "TSLA"]
        print(f"Using default test symbols: {', '.join(symbols)}")
    
    tester = SymbolTester(symbols)
    
    # Test each symbol
    results = {}
    for symbol in symbols:
        results[symbol] = await tester.test_symbol_workflow(symbol)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for symbol, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {symbol}")
    
    success_rate = sum(results.values()) / len(results) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All symbols tested successfully!")
    elif success_rate >= 75:
        print("✅ Most symbols working correctly")
    else:
        print("⚠️  Some issues detected")
    
    # Save results
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "symbols_tested": symbols,
        "success_rate": success_rate,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*60)
    
    return 0 if success_rate >= 75 else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
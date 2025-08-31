#!/usr/bin/env python3
"""Fix MarketData integration to ensure it's actually being used for options."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def analyze_current_implementation():
    """Analyze how options data is currently being fetched."""
    
    print("="*60)
    print("MARKETDATA INTEGRATION ANALYSIS")
    print("="*60)
    
    issues_found = []
    
    # 1. Check MarketDataClient implementation
    print("\n1. Checking MarketDataClient feed configuration...")
    with open("src/providers/marketdata_client.py", "r") as f:
        content = f.read()
        
        # Check if default_feed is being used
        if "self.default_feed = config.get(\"default_feed\", \"cached\")" in content:
            print("   ✅ Default feed set to 'cached'")
        else:
            print("   ❌ Default feed not properly configured")
            issues_found.append("Default feed configuration")
        
        # Check if feed parameter is being passed
        if "'feed': filters.get('feed', self.default_feed)" in content:
            print("   ⚠️  Feed parameter is being set - should NOT set for cached data")
            issues_found.append("Feed parameter should not be set")
    
    # 2. Check simple production script
    print("\n2. Checking run_simple_production.py...")
    with open("run_simple_production.py", "r") as f:
        content = f.read()
        
        if "fetch_marketdata_options" in content:
            print("   ✅ Has MarketData options fetching method")
            
            # Check if it's actually fetching option chains
            if "options/expirations" in content and "options/chain" not in content:
                print("   ❌ Only fetching expirations, not full option chains!")
                issues_found.append("Not fetching full option chains")
        else:
            print("   ❌ No MarketData options fetching")
            issues_found.append("Missing MarketData integration")
    
    # 3. Check DataEnrichmentExecutor
    print("\n3. Checking DataEnrichmentExecutor...")
    enrichment_file = "src/screener/steps/data_enrichment_step.py"
    if os.path.exists(enrichment_file):
        with open(enrichment_file, "r") as f:
            content = f.read()
            
            if "get_pmcc_option_chains" in content:
                print("   ✅ Uses get_pmcc_option_chains method")
            else:
                print("   ❌ Not using MarketData for options")
                issues_found.append("DataEnrichmentExecutor not using MarketData")
    
    return issues_found

def create_fixes():
    """Create fixes for identified issues."""
    
    print("\n" + "="*60)
    print("CREATING FIXES")
    print("="*60)
    
    # Fix 1: Update MarketDataClient to not send feed parameter
    print("\n1. Creating fixed MarketDataClient method...")
    
    fixed_method = '''
    async def get_options_chain_fixed(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        **filters
    ) -> List[OptionContract]:
        """
        Get options chain with proper configuration for cached data.
        FIXED VERSION: Doesn't send feed parameter to get cached data.
        """
        try:
            if not symbol:
                raise ValueError("Symbol is required")
            
            url = f"{self.base_url}/options/chain/{symbol}/"
            
            # Build query parameters
            params = {}
            
            # DO NOT SET feed parameter - let API use cached by default
            # This gives us status 203 with cached data
            
            if expiration_date:
                params['expiration'] = expiration_date
            
            # Apply other filters
            if 'side' in filters:
                params['side'] = filters['side']
            if 'strike' in filters:
                params['strike'] = filters['strike']
            if 'minOpenInterest' in filters:
                params['minOpenInterest'] = filters['minOpenInterest']
            if 'moneyness' in filters:
                params['moneyness'] = filters['moneyness']
            
            # Make request
            response = await self._make_request(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                params=params
            )
            
            if not response.success:
                logger.error(f"Failed to get options chain: {response.error}")
                return []
            
            # Parse response - MarketData returns parallel arrays
            data = response.data
            if data.get('s') != 'ok':
                return []
            
            # Convert arrays to OptionContract objects
            contracts = []
            strikes = data.get('strike', [])
            
            for i in range(len(strikes)):
                contract = OptionContract(
                    option_symbol=data.get('optionSymbol', [])[i] if i < len(data.get('optionSymbol', [])) else f"{symbol}_{strikes[i]}",
                    underlying_symbol=symbol,
                    strike=strikes[i],
                    expiration_date=datetime.fromtimestamp(data.get('expiration', [])[i]) if i < len(data.get('expiration', [])) else None,
                    option_type=OptionType.CALL if data.get('side', [])[i] == 'call' else OptionType.PUT,
                    bid=data.get('bid', [])[i] if i < len(data.get('bid', [])) else 0,
                    ask=data.get('ask', [])[i] if i < len(data.get('ask', [])) else 0,
                    last=data.get('last', [])[i] if i < len(data.get('last', [])) else 0,
                    volume=data.get('volume', [])[i] if i < len(data.get('volume', [])) else 0,
                    open_interest=data.get('openInterest', [])[i] if i < len(data.get('openInterest', [])) else 0,
                    implied_volatility=data.get('iv', [])[i] if i < len(data.get('iv', [])) else 0,
                    delta=data.get('delta', [])[i] if i < len(data.get('delta', [])) else None,
                    gamma=data.get('gamma', [])[i] if i < len(data.get('gamma', [])) else None,
                    theta=data.get('theta', [])[i] if i < len(data.get('theta', [])) else None,
                    vega=data.get('vega', [])[i] if i < len(data.get('vega', [])) else None,
                )
                contracts.append(contract)
            
            # Log success
            logger.info(f"Retrieved {len(contracts)} option contracts for {symbol}")
            
            # Track credit usage (1 credit for cached request)
            self._daily_credits_used += 1
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return []
'''
    
    # Save the fixed method
    with open("marketdata_fix.py", "w") as f:
        f.write("# Fixed MarketData options chain method\n")
        f.write("# Copy this to src/providers/marketdata_client.py\n\n")
        f.write(fixed_method)
    
    print("   ✅ Fixed method saved to marketdata_fix.py")
    
    # Fix 2: Update simple production script
    print("\n2. Creating enhanced fetch_marketdata_options...")
    
    enhanced_method = '''
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
'''
    
    with open("marketdata_enhancement.py", "w") as f:
        f.write("# Enhanced MarketData fetching for production\n")
        f.write("# Add this to run_simple_production.py\n\n")
        f.write(enhanced_method)
    
    print("   ✅ Enhanced method saved to marketdata_enhancement.py")
    
    print("\n" + "="*60)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. DON'T send 'feed' parameter - let API default to cached")
    print("2. Use specific date ranges to limit data")
    print("3. Add filtering (minOpenInterest, moneyness) to reduce contracts")
    print("4. Log whether you got cached (203) or live (200) data")
    print("5. Track credit usage - cached = 1 credit total")

def main():
    """Main analysis and fix creation."""
    
    print("\nANALYZING MARKETDATA INTEGRATION\n")
    
    # Analyze current implementation
    issues = analyze_current_implementation()
    
    if issues:
        print("\n" + "="*60)
        print("ISSUES FOUND")
        print("="*60)
        for issue in issues:
            print(f"  ❌ {issue}")
    
    # Create fixes
    create_fixes()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Review the generated fix files")
    print("2. Apply fixes to the actual source files")
    print("3. Test with test_marketdata_final.py")
    print("4. Check https://www.marketdata.app/dashboard for credit usage")
    print("\nKey insight: DO NOT send 'feed' parameter for cached data!")

if __name__ == "__main__":
    main()
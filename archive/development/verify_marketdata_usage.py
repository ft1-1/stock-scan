#!/usr/bin/env python3
"""Verify MarketData API usage in the production workflow."""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
load_dotenv()

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_provider_manager():
    """Test ProviderManager to see which APIs are actually called."""
    from src.providers.provider_manager import ProviderManager, ProviderConfig, ProviderType
    from config.settings import get_settings
    
    settings = get_settings()
    
    print("\n" + "="*60)
    print("Testing Provider Manager Integration")
    print("="*60)
    
    # Create provider configs
    provider_configs = []
    
    # EODHD config
    if settings.eodhd_api_key:
        eodhd_config = ProviderConfig(
            provider_type=ProviderType.EODHD,
            config={
                'api_key': settings.eodhd_api_key,
                'base_url': 'https://eodhd.com/api',
                'requests_per_minute': 60
            },
            priority=1,
            capabilities=['screening', 'quotes', 'fundamentals', 'technicals'],
            enabled=True
        )
        provider_configs.append(eodhd_config)
        print(f"✅ EODHD configured: {settings.eodhd_api_key[:10]}...")
    
    # MarketData config
    if settings.marketdata_api_key:
        marketdata_config = ProviderConfig(
            provider_type=ProviderType.MARKETDATA,
            config={
                'api_key': settings.marketdata_api_key,
                'base_url': 'https://api.marketdata.app/v1',
                'default_feed': 'live',  # Use live for free tier
                'requests_per_minute': 100
            },
            priority=2,
            capabilities=['options_chains', 'options_greeks'],
            enabled=True
        )
        provider_configs.append(marketdata_config)
        print(f"✅ MarketData configured: {settings.marketdata_api_key[:10]}...")
    
    # Initialize provider manager
    provider_manager = ProviderManager(provider_configs=provider_configs)
    # ProviderManager doesn't have initialize method, providers are initialized on first use
    
    print("\nProvider Manager initialized with providers:")
    for provider_type in provider_manager.providers:
        print(f"  - {provider_type.value}")
    
    # Test 1: Get stock quote (should use EODHD)
    print("\n1. Testing Stock Quote (should use EODHD)...")
    try:
        quote = await provider_manager.get_stock_quote("AAPL")
        if quote:
            print(f"   ✅ Quote received: ${quote.price}")
            print(f"   Provider used: Check logs above")
        else:
            print(f"   ❌ No quote received")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Get options chain (should use MarketData)
    print("\n2. Testing Options Chain (should use MarketData)...")
    try:
        # Use specific expiration to minimize credits
        options = await provider_manager.get_options_chain(
            "AAPL",
            expiration_date="2025-09-19"
        )
        if options:
            print(f"   ✅ Options received: {len(options)} contracts")
            print(f"   Provider used: Check logs above")
            if options:
                contract = options[0]
                print(f"   Sample: Strike ${contract.strike}, Delta {contract.delta}")
        else:
            print(f"   ❌ No options received")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Get PMCC chains (should use MarketData)
    print("\n3. Testing PMCC Chains (should use MarketData)...")
    try:
        chains = await provider_manager.get_pmcc_option_chains("AAPL")
        if chains:
            leaps = chains.get('leaps', [])
            shorts = chains.get('short_calls', [])
            print(f"   ✅ PMCC chains: {len(leaps)} LEAPS, {len(shorts)} short calls")
            print(f"   Provider used: Check logs above")
        else:
            print(f"   ❌ No PMCC chains received")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check metrics
    print("\n4. Provider Metrics:")
    metrics = provider_manager.get_metrics()
    for provider_type, provider_metrics in metrics.items():
        print(f"\n   {provider_type}:")
        print(f"     Requests: {provider_metrics.get('total_requests', 0)}")
        print(f"     Success rate: {provider_metrics.get('success_rate', 0):.1%}")
        print(f"     Avg response time: {provider_metrics.get('avg_response_time', 0):.2f}s")
    
    await provider_manager.cleanup()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if any('marketdata' in str(m).lower() for m in metrics):
        print("✅ MarketData provider is configured and available")
    else:
        print("❌ MarketData provider not found in metrics")
    
    print("\n⚠️  Check the logs above to see which provider was actually used")
    print("    Look for lines like: 'Using provider: MARKETDATA'")
    print("\n📊 Check MarketData dashboard for credit usage:")
    print("    https://www.marketdata.app/dashboard")

async def test_data_enrichment_step():
    """Test the DataEnrichmentExecutor to see if it uses MarketData."""
    print("\n" + "="*60)
    print("Testing DataEnrichmentExecutor")
    print("="*60)
    
    try:
        from src.screener.steps.data_enrichment_step import DataEnrichmentExecutor
        from src.models import WorkflowStep, WorkflowExecutionContext, WorkflowConfig
        
        # Create executor
        executor = DataEnrichmentExecutor(WorkflowStep.DATA_COLLECTION)
        
        # Create context
        context = WorkflowExecutionContext(
            workflow_id="test_123",
            config=WorkflowConfig()
        )
        
        # Test with single symbol
        print("\nTesting data collection for AAPL...")
        result = await executor.execute_step(
            input_data={'symbols': ['AAPL']},
            context=context
        )
        
        if result and result.output_data:
            data = result.output_data.get('AAPL', {})
            
            # Check what data was collected
            print("\nData collected:")
            print(f"  Quote: {'✅' if data.get('quote') else '❌'}")
            print(f"  Historical: {'✅' if data.get('historical') else '❌'}")
            print(f"  Fundamentals: {'✅' if data.get('fundamentals') else '❌'}")
            print(f"  News: {'✅' if data.get('news') else '❌'}")
            print(f"  Options Chain: {'✅' if data.get('options_chain') else '❌'}")
            
            if data.get('options_chain'):
                options = data['options_chain']
                print(f"\n  Options Details:")
                print(f"    Contracts: {len(options)}")
                if options:
                    contract = options[0]
                    print(f"    Has Greeks: {'✅' if 'delta' in contract else '❌'}")
                    print(f"    Source: {'MarketData' if 'contract_type' in contract else 'Unknown'}")
            
            # Check data sources
            sources = data.get('data_sources', [])
            print(f"\n  Data Sources Used: {sources}")
            
            if 'MARKETDATA' in sources:
                print("  ✅ MarketData was used!")
            else:
                print("  ❌ MarketData was NOT used")
        
    except Exception as e:
        print(f"❌ Error testing DataEnrichmentExecutor: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests."""
    await test_provider_manager()
    await test_data_enrichment_step()
    
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    print("\n1. Check logs above for 'MARKETDATA' mentions")
    print("2. Check if options_chain data was collected")
    print("3. Visit https://www.marketdata.app/dashboard")
    print("4. Look for credit usage - if 0, MarketData is NOT being used")
    print("\nIf MarketData is not being used, the issue is likely:")
    print("  - Provider not initialized properly")
    print("  - Options data coming from EODHD instead")
    print("  - Provider manager not routing to MarketData for options")

if __name__ == "__main__":
    asyncio.run(main())
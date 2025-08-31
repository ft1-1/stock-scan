"""
Example demonstrating how to integrate DataEnrichmentExecutor with the WorkflowEngine.

This example shows how to:
1. Create and register the DataEnrichmentExecutor
2. Execute data enrichment as part of the workflow
3. Handle the comprehensive data structure output
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.screener.workflow_engine import WorkflowEngine, WorkflowStepExecutor
from src.screener.steps.data_enrichment_step import DataEnrichmentExecutor
from src.models.workflow_models import (
    WorkflowStep,
    WorkflowConfig,
    WorkflowExecutionContext
)
from src.models.base_models import ScreeningCriteria
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MockStockScreeningExecutor(WorkflowStepExecutor):
    """Mock stock screening executor for demonstration purposes."""
    
    async def execute_step(self, input_data, context):
        """Return a sample list of symbols for testing."""
        # In a real implementation, this would be the stock screening step
        sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        logger.info(f"Mock screening returned {len(sample_symbols)} symbols")
        return sample_symbols


async def demonstrate_data_enrichment():
    """Demonstrate the DataEnrichmentExecutor integration."""
    
    # Initialize workflow engine
    engine = WorkflowEngine()
    
    # Register step executors
    stock_screening_executor = MockStockScreeningExecutor(WorkflowStep.STOCK_SCREENING)
    data_enrichment_executor = DataEnrichmentExecutor(WorkflowStep.DATA_COLLECTION)
    
    engine.register_step_executor(WorkflowStep.STOCK_SCREENING, stock_screening_executor)
    engine.register_step_executor(WorkflowStep.DATA_COLLECTION, data_enrichment_executor)
    
    # Create screening criteria
    criteria = ScreeningCriteria(
        min_market_cap=1_000_000_000,  # $1B
        max_market_cap=100_000_000_000,  # $100B
        min_price=10.0,
        max_price=500.0,
        min_volume=1_000_000,
        min_option_volume=100,
        min_open_interest=500
    )
    
    # Configure workflow
    config = WorkflowConfig(
        max_concurrent_stocks=5,  # Limit for testing
        max_retry_attempts=2,
        step_timeout_seconds=300,
        continue_on_errors=True,
        enable_caching=True,
        enable_ai_analysis=False  # Disable AI for this demo
    )
    
    print("=== Data Enrichment Integration Demo ===")
    print(f"Screening criteria: Market cap ${criteria.min_market_cap:,} - ${criteria.max_market_cap:,}")
    print(f"Price range: ${criteria.min_price} - ${criteria.max_price}")
    print(f"Min volume: {criteria.min_volume:,}")
    print()
    
    try:
        # For this demo, we'll manually execute the steps to show the data flow
        
        # Step 1: Stock Screening (mock)
        print("Step 1: Stock Screening")
        context = WorkflowExecutionContext(config=config)
        symbols = await stock_screening_executor.execute_step(criteria, context)
        print(f"Screened symbols: {symbols}")
        print()
        
        # Step 2: Data Enrichment
        print("Step 2: Data Enrichment")
        enrichment_result = await data_enrichment_executor.execute_step(symbols, context)
        
        if enrichment_result:
            print("=== Data Enrichment Results ===")
            result_data = enrichment_result
            
            print(f"Total symbols processed: {result_data['total_symbols']}")
            print(f"Successfully enriched: {result_data['symbols_processed']}")
            print(f"Failed to enrich: {result_data['symbols_failed']}")
            print(f"Success rate: {(result_data['symbols_processed'] / result_data['total_symbols']) * 100:.1f}%")
            print()
            
            # Show sample enriched data structure
            enriched_data = result_data['enriched_data']
            if enriched_data:
                sample_symbol = list(enriched_data.keys())[0]
                sample_data = enriched_data[sample_symbol]
                
                print(f"=== Sample Enriched Data for {sample_symbol} ===")
                print(f"Data sources collected: {len(sample_data.get('data_sources', []))}")
                print(f"Available data fields:")
                
                for field, value in sample_data.items():
                    if field == 'data_sources':
                        continue
                    
                    if isinstance(value, dict):
                        print(f"  {field}: {len(value)} fields")
                    elif isinstance(value, list):
                        print(f"  {field}: {len(value)} items")
                    elif value is not None:
                        print(f"  {field}: Available")
                    else:
                        print(f"  {field}: None")
                
                # Show sample fundamental data if available
                if sample_data.get('fundamentals'):
                    print(f"\n=== Sample Fundamental Data Structure ===")
                    fundamentals = sample_data['fundamentals']
                    for category, data in fundamentals.items():
                        if isinstance(data, dict):
                            print(f"  {category}: {len(data)} metrics")
                        else:
                            print(f"  {category}: {type(data).__name__}")
                
                # Show sample options chain if available
                options_chain = sample_data.get('options_chain', [])
                if options_chain:
                    print(f"\n=== Sample Options Chain ===")
                    print(f"Total contracts: {len(options_chain)}")
                    
                    # Group by contract type
                    leaps = [c for c in options_chain if c.get('contract_type') == 'LEAP']
                    short_calls = [c for c in options_chain if c.get('contract_type') == 'SHORT_CALL']
                    
                    print(f"  LEAPS contracts: {len(leaps)}")
                    print(f"  Short call contracts: {len(short_calls)}")
                    
                    if leaps:
                        sample_leap = leaps[0]
                        print(f"  Sample LEAP: {sample_leap.get('contract_symbol', 'N/A')}")
                        print(f"    Strike: ${sample_leap.get('strike', 'N/A')}")
                        print(f"    Delta: {sample_leap.get('delta', 'N/A')}")
                        print(f"    Expiration: {sample_leap.get('expiration', 'N/A')}")
                
                # Show trading dates
                trading_dates = sample_data.get('trading_dates', {})
                if trading_dates:
                    print(f"\n=== Trading Dates ===")
                    print(f"  Last trading day: {trading_dates.get('today')}")
                    print(f"  30 days ago: {trading_dates.get('thirty_days_ago')}")
                    print(f"  6 months ago: {trading_dates.get('six_months_ago')}")
        
        print("\n✓ Data enrichment integration demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def show_data_structure_example():
    """Show the expected data structure from DataEnrichmentExecutor."""
    
    print("\n=== Expected Data Structure from DataEnrichmentExecutor ===")
    
    example_structure = {
        'enriched_data': {
            'SYMBOL': {
                'symbol': 'SYMBOL',
                'quote': {
                    'price': 150.25,
                    'volume': 50000000,
                    'market_cap': 2500000000000,
                    # ... other quote fields
                },
                'historical': [
                    # List of historical price records
                ],
                'fundamentals': {
                    'company_info': {
                        'name': 'Company Name',
                        'sector': 'Technology',
                        'industry': 'Software',
                        'market_cap_mln': 2500000,
                        # ... other company info
                    },
                    'financial_health': {
                        'eps_ttm': 6.15,
                        'profit_margin': 0.25,
                        'dividend_yield': 0.005,
                        # ... other financial metrics
                    },
                    'valuation_metrics': {
                        'pe_ratio': 24.5,
                        'forward_pe': 22.1,
                        # ... other valuation metrics
                    },
                    'balance_sheet': {
                        'total_assets': 350000,  # in millions
                        'total_debt': 15000,
                        'cash_and_equivalents': 165000,
                        # ... other balance sheet items
                    },
                    # ... other fundamental categories
                },
                'technicals': {
                    'rsi': [
                        # RSI indicator data points
                    ],
                    'volatility': [
                        # Volatility data points
                    ],
                    'atr': [
                        # ATR indicator data points
                    ]
                },
                'economic_events': [
                    # US macro economic events
                ],
                'news': [
                    # Recent news articles
                ],
                'earnings': [
                    # Earnings calendar data
                ],
                'sentiment': {
                    # Sentiment analysis data
                },
                'options_chain': [
                    {
                        'contract_symbol': 'SYMBOL240315C00100000',
                        'contract_type': 'LEAP',
                        'option_type': 'call',
                        'strike': 100.0,
                        'expiration': '2024-03-15',
                        'bid': 52.5,
                        'ask': 53.0,
                        'delta': 0.85,
                        'gamma': 0.003,
                        'theta': -0.12,
                        'vega': 0.45,
                        'implied_volatility': 0.28,
                        'open_interest': 1250,
                        'volume': 45
                    },
                    # ... more option contracts
                ],
                'trading_dates': {
                    'today': '2024-01-15',
                    'thirty_days_ago': '2023-12-16',
                    'sixty_days_ago': '2023-11-16',
                    'six_months_ago': '2023-07-15',
                    'one_year_ago': '2023-01-15',
                    'hundred_days_ago': '2023-10-07'
                },
                'data_collection_timestamp': '2024-01-15T10:30:00',
                'data_sources': [
                    'quote', 'fundamentals', 'technicals', 'options_chain',
                    'news', 'earnings', 'economic_events'
                ]
            }
        },
        'symbols_processed': 5,
        'symbols_failed': 0,
        'total_symbols': 5,
        'trading_dates': {
            # Same trading dates as above
        }
    }
    
    def print_structure(obj, indent=0):
        """Recursively print data structure."""
        spaces = "  " * indent
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"{spaces}{key}:")
                    print_structure(value, indent + 1)
                else:
                    print(f"{spaces}{key}: {type(value).__name__}")
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                print(f"{spaces}[")
                print(f"{spaces}  {type(obj[0]).__name__} with fields:")
                print_structure(obj[0], indent + 2)
                print(f"{spaces}  ... {len(obj)} items total")
                print(f"{spaces}]")
            else:
                print(f"{spaces}[{len(obj)} items of type {type(obj[0]).__name__ if obj else 'unknown'}]")
    
    print_structure(example_structure)
    
    print("\nKey Features of the Enriched Data:")
    print("1. Multi-source integration (EODHD + MarketData APIs)")
    print("2. Comprehensive fundamental analysis with filtered metrics")
    print("3. Technical indicators from EODHD API")
    print("4. Complete options chains with Greeks for PMCC analysis")
    print("5. Macro economic context and recent news")
    print("6. Holiday-aware trading date calculations")
    print("7. Structured data validation and quality assurance")
    print("8. Error handling with graceful degradation")


async def main():
    """Main demonstration function."""
    await demonstrate_data_enrichment()
    await show_data_structure_example()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""Test enhanced data collection for AAPL."""

import asyncio
from src.screener.steps.data_enrichment_step import DataEnrichmentStep
from src.screener.workflow_engine import WorkflowExecutionContext

async def test_data_collection():
    """Test data collection for AAPL."""
    
    # Create context
    context = WorkflowExecutionContext()
    
    # Create step
    step = DataEnrichmentStep()
    
    # Create input with AAPL
    input_data = {
        'symbols': ['AAPL']
    }
    
    # Execute
    result = await step.execute_step(input_data, context)
    
    print(f"Result keys: {list(result.keys())}")
    
    if 'enriched_data' in result:
        for symbol, data in result['enriched_data'].items():
            print(f"\n{symbol} data keys: {list(data.keys())}")
            
            # Check for enhanced data fields
            for field in ['economic_events', 'news', 'fundamentals', 'sentiment', 'earnings']:
                if field in data:
                    if data[field]:
                        print(f"  ✅ {field}: Present")
                    else:
                        print(f"  ❌ {field}: Empty")
                else:
                    print(f"  ❌ {field}: Missing")

if __name__ == "__main__":
    asyncio.run(test_data_collection())
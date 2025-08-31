#!/usr/bin/env python3
"""Test the complete production workflow with a single symbol."""

import sys
import os
import asyncio
from pathlib import Path

# Setup Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Load environment
from dotenv import load_dotenv
load_dotenv()

async def test_single_symbol_workflow():
    """Test the complete workflow with a single symbol."""
    
    print("🧪 Testing Complete Workflow with Single Symbol")
    print("=" * 50)
    
    try:
        # Import required components
        from src.screener.workflow_orchestrator import WorkflowOrchestrator
        from src.models import ScreeningCriteria, WorkflowConfig
        
        # Create criteria for single symbol test
        criteria = ScreeningCriteria(
            specific_symbols=["AAPL"],  # Single symbol for testing
            min_market_cap=1000000000,
            max_market_cap=10000000000000,
            min_price=5.0,
            max_price=1000.0,
            min_volume=100000,
            min_option_volume=100,
            min_open_interest=50,
            exclude_sectors=[]
        )
        
        # Create lightweight workflow config for testing
        config = WorkflowConfig(
            max_retry_attempts=2,
            step_timeout_seconds=300,  # 5 minutes
            checkpoint_interval=1,
            max_concurrent_stocks=1,
            enable_caching=True,
            enable_validation=True
        )
        
        # Initialize orchestrator
        orchestrator = WorkflowOrchestrator(config)
        workflow_id = "test_single_symbol"
        
        print(f"🚀 Starting workflow: {workflow_id}")
        print(f"📊 Testing with symbol: AAPL")
        print("-" * 50)
        
        # Execute workflow
        result = await orchestrator.execute_workflow(
            criteria=criteria,
            workflow_id=workflow_id
        )
        
        # Display results
        print(f"\n✅ Workflow completed successfully!")
        print(f"   Workflow ID: {result.workflow_id}")
        print(f"   Execution Time: {result.execution_time_seconds:.1f}s")
        print(f"   Stocks Screened: {result.total_stocks_screened}")
        print(f"   Opportunities Found: {len(result.qualifying_results)}")
        success_rate = result.success_rate or 0.0
        print(f"   Success Rate: {(success_rate * 100):.1f}%")
        
        if result.qualifying_results:
            print(f"\n📈 Results:")
            for opp in result.qualifying_results:
                symbol = opp.get('symbol', 'N/A')
                local_score = opp.get('local_score', 0)
                ai_rating = opp.get('ai_rating', 'N/A')
                print(f"   {symbol}: Local={local_score:.1f}, AI={ai_rating}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings: {len(result.warnings)}")
        
        if result.errors_encountered:
            print(f"\n❌ Errors: {len(result.errors_encountered)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""
    # Check for required environment variables
    required_vars = ["SCREENER_EODHD_API_KEY", "SCREENER_MARKETDATA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the test.")
        return 1
    
    success = await test_single_symbol_workflow()
    return 0 if success else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
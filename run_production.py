#!/usr/bin/env python
"""Production runner for the options screening application using integrated workflow."""

import sys
import os
from pathlib import Path
import asyncio
from datetime import datetime
import json
import uuid

# Setup Python path properly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import workflow components
from src.screener.workflow_orchestrator import WorkflowOrchestrator
from src.models import ScreeningCriteria, WorkflowConfig
from src.utils.logging_config import get_logger, print_info, print_success, print_error
from config.settings import get_settings

logger = get_logger(__name__)

async def run_integrated_workflow():
    """Run the complete 7-step integrated workflow."""
    
    # Set to CRITICAL level to suppress all but the most critical issues
    import os
    os.environ['SCREENER_LOG_LEVEL'] = 'CRITICAL'
    
    # Re-setup logging with new level
    from src.utils.logging_config import setup_logging
    setup_logging()
    
    print("\n🎯 OPTIONS SCREENING WORKFLOW")
    
    try:
        # Get configuration from environment
        settings = get_settings()
        
        # Create screening criteria
        symbols_str = os.getenv("SCREENER_SPECIFIC_SYMBOLS")
        
        if symbols_str:
            # Use specific symbols if provided
            symbols = [s.strip() for s in symbols_str.split(',')]
            print(f"📌 Symbols: {', '.join(symbols)}")
            
            criteria = ScreeningCriteria(
                specific_symbols=symbols,
                # Other criteria will be ignored when specific_symbols is provided
                min_market_cap=1000000000,  # $1B minimum
                max_market_cap=10000000000000,  # $10T maximum
                min_price=5.0,
                max_price=1000.0,
                min_volume=100000,
                min_option_volume=100,
                min_open_interest=50,
                exclude_sectors=[]
            )
        else:
            # Use market screening criteria
            print_info("Using market-wide screening criteria")
            
            criteria = ScreeningCriteria(
                specific_symbols=None,
                min_market_cap=1000000000,  # $1B minimum
                max_market_cap=10000000000000,  # $10T maximum
                min_price=10.0,  # Higher minimum for options liquidity
                max_price=500.0,
                min_volume=500000,  # Higher volume for liquidity
                min_option_volume=500,
                min_open_interest=100,
                exclude_sectors=[]  # Include all sectors
            )
        
        # Create workflow configuration
        workflow_config = WorkflowConfig(
            max_retry_attempts=3,
            step_timeout_seconds=3600,  # 1 hour per step
            checkpoint_interval=5,  # Save checkpoint every 5 symbols
            max_concurrent_stocks=3,  # Conservative concurrency
            enable_caching=True,
            enable_validation=True
        )
        
        # Create workflow orchestrator
        orchestrator = WorkflowOrchestrator(workflow_config)
        
        # Generate unique workflow ID
        workflow_id = f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        print(f"📊 Running analysis...")
        print("")
        
        # Execute the complete workflow
        result = await orchestrator.execute_workflow(
            criteria=criteria,
            workflow_id=workflow_id
        )
        
        # Display results summary
        print("\n✅ ANALYSIS COMPLETE")
        
        print_success(f"Workflow ID: {result.workflow_id}")
        print_info(f"Execution Time: {result.execution_time_seconds:.1f} seconds")
        print_info(f"Stocks Screened: {result.total_stocks_screened}")
        print_info(f"Opportunities Found: {len(result.qualifying_results)}")
        print_info(f"API Calls Made: {result.api_calls_made or 0}")
        print_info(f"Total Cost: ${result.total_cost:.4f}" if result.total_cost is not None else "Total Cost: $0.0000")
        print_info(f"Success Rate: {(result.success_rate * 100):.1f}%" if result.success_rate is not None else "Success Rate: 0.0%")
        
        # Display top opportunities
        if result.qualifying_results:
            print("\n📈 Top Opportunities:")
            for i, opportunity in enumerate(result.qualifying_results[:10], 1):
                symbol = opportunity.get('symbol', 'N/A')
                local_score = opportunity.get('local_score', 0)
                ai_rating = opportunity.get('ai_rating', 'N/A')
                print_info(f"{i:2d}. {symbol:5s} | Local: {local_score:5.1f} | AI: {ai_rating}")
        
        # Display warnings and errors
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings[-5:]:  # Show last 5 warnings
                print(f"   - {warning}")
        
        if result.errors_encountered:
            print("\n❌ Errors:")
            for error in result.errors_encountered[-5:]:  # Show last 5 errors
                print(f"   - {error}")
        
        # Save summary results
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary_data = {
            "workflow_id": result.workflow_id,
            "timestamp": result.completed_at.isoformat() if result.completed_at else datetime.now().isoformat(),
            "screening_criteria": criteria.model_dump() if hasattr(criteria, 'model_dump') else str(criteria),
            "execution_summary": {
                "execution_time_seconds": result.execution_time_seconds,
                "stocks_screened": result.total_stocks_screened,
                "opportunities_found": len(result.qualifying_results),
                "api_calls_made": result.api_calls_made,
                "total_cost_usd": result.total_cost,
                "success_rate": result.success_rate
            },
            "top_opportunities": result.qualifying_results[:20],  # Save top 20
            "warnings": result.warnings,
            "errors": result.errors_encountered
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print_success(f"\n💾 Summary saved to: {summary_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise

async def main():
    """Main entry point for integrated workflow execution."""
    try:
        await run_integrated_workflow()
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
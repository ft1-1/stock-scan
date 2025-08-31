#!/usr/bin/env python3
"""Test enhanced data collection and save to JSON."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.screener.workflow_engine import WorkflowExecutionContext
from src.screener.steps.data_enrichment_step import DataEnrichmentExecutor
from src.screener.steps.technical_analysis_step import TechnicalAnalysisExecutor
from src.screener.steps.local_ranking_step import LocalRankingExecutor
from src.screener.steps.claude_analysis_step import ClaudeAnalysisExecutor


async def test_full_workflow():
    """Test full workflow with enhanced data."""
    
    # Create context
    context = WorkflowExecutionContext()
    
    # Step 1: Data Collection
    print("Step 1: Collecting enhanced data...")
    data_step = DataEnrichmentExecutor()
    data_result = await data_step.execute_step({'symbols': ['AAPL']}, context)
    
    if 'enriched_data' not in data_result or not data_result['enriched_data']:
        print("❌ Data collection failed")
        return
    
    print(f"✅ Data collected for {len(data_result['enriched_data'])} symbols")
    
    # Step 2: Technical Analysis
    print("\nStep 2: Technical analysis...")
    tech_step = TechnicalAnalysisExecutor()
    tech_result = await tech_step.execute_step(data_result, context)
    
    if 'analyzed_opportunities' not in tech_result:
        print("❌ Technical analysis failed")
        return
    
    print(f"✅ Analyzed {len(tech_result['analyzed_opportunities'])} opportunities")
    
    # Step 3: Local Ranking
    print("\nStep 3: Local ranking...")
    rank_step = LocalRankingExecutor()
    rank_result = await rank_step.execute_step(tech_result, context)
    
    if 'top_opportunities' not in rank_result:
        print("❌ Ranking failed")
        return
    
    print(f"✅ Selected {len(rank_result['top_opportunities'])} top opportunities")
    
    # Step 4: Claude Analysis
    print("\nStep 4: Claude AI analysis...")
    claude_step = ClaudeAnalysisExecutor()
    claude_result = await claude_step.execute_step(rank_result, context)
    
    if 'ai_results' not in claude_result:
        print("❌ Claude analysis failed")
        return
    
    print(f"✅ AI analyzed {len(claude_result['ai_results'])} opportunities")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(f'data/test_enhanced_data_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(claude_result, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Check what enhanced data was included
    if claude_result['ai_results']:
        result = claude_result['ai_results'][0]
        if 'data_package' in result and 'enhanced_stock_data' in result['data_package']:
            esd = result['data_package']['enhanced_stock_data']
            print("\nEnhanced data fields in Claude submission:")
            for key in esd:
                if isinstance(esd[key], dict):
                    print(f"  - {key}: {len(esd[key])} items")
                elif isinstance(esd[key], list):
                    print(f"  - {key}: {len(esd[key])} items")
                else:
                    print(f"  - {key}: present")


if __name__ == "__main__":
    asyncio.run(test_full_workflow())
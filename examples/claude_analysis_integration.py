#!/usr/bin/env python3
"""
Claude Analysis Integration Example

Demonstrates how to integrate ClaudeAnalysisExecutor into the complete workflow
for AI-powered options screening analysis.

This example shows:
1. Complete workflow from screening to AI analysis
2. Data flow between all components
3. Individual JSON persistence per opportunity
4. Cost management and rate limiting
5. Error handling and validation
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Import all workflow components
from src.screener.steps import (
    DataEnrichmentExecutor,
    TechnicalAnalysisExecutor, 
    LocalRankingExecutor,
    ClaudeAnalysisExecutor
)
from src.models import (
    WorkflowStep,
    WorkflowExecutionContext,
    WorkflowConfig
)


async def run_complete_workflow_with_ai():
    """
    Demonstrate complete workflow from screening to AI analysis.
    
    This simulates the full pipeline:
    1. Stock screening (simulated with sample symbols)
    2. Data enrichment (EODHD + MarketData APIs)
    3. Technical analysis (momentum, squeeze, scoring)
    4. Local ranking (filter to top N)
    5. Claude AI analysis (only top opportunities)
    """
    print("=" * 60)
    print("COMPLETE OPTIONS SCREENING WORKFLOW WITH AI ANALYSIS")
    print("=" * 60)
    
    # Sample symbols (in production, these come from screening step)
    sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Create workflow context
    config = WorkflowConfig(
        max_concurrent_stocks=5,
        enable_ai_analysis=True,
        ai_batch_size=3,  # Limit AI analysis to top 3
        max_ai_cost_dollars=10.0  # $10 daily limit
    )
    
    context = WorkflowExecutionContext(
        config=config,
        total_symbols=len(sample_symbols)
    )
    
    print(f"Processing {len(sample_symbols)} symbols: {', '.join(sample_symbols)}")
    print(f"AI analysis enabled with ${config.max_ai_cost_dollars} daily limit")
    print()
    
    try:
        # Phase 1: Data Enrichment
        print("PHASE 1: Data Enrichment")
        print("-" * 30)
        
        data_enrichment = DataEnrichmentExecutor(WorkflowStep.DATA_COLLECTION)
        enriched_result = await data_enrichment.execute_step(sample_symbols, context)
        
        print(f"✓ Data enrichment completed:")
        print(f"  Symbols processed: {enriched_result['symbols_processed']}")
        print(f"  Symbols failed: {enriched_result['symbols_failed']}")
        print()
        
        # Phase 2: Technical Analysis & Scoring
        print("PHASE 2: Technical Analysis & Scoring")
        print("-" * 40)
        
        technical_analysis = TechnicalAnalysisExecutor(WorkflowStep.TECHNICAL_CALCULATION)
        technical_result = await technical_analysis.execute_step(enriched_result, context)
        
        print(f"✓ Technical analysis completed:")
        print(f"  Opportunities analyzed: {len(technical_result.get('analyzed_opportunities', {}))}")
        print()
        
        # Phase 3: Local Ranking
        print("PHASE 3: Local Ranking & Filtering")
        print("-" * 35)
        
        local_ranking = LocalRankingExecutor(WorkflowStep.LLM_PACKAGING, top_n=3)
        ranking_result = await local_ranking.execute_step(technical_result, context)
        
        top_opportunities = ranking_result['top_opportunities']
        print(f"✓ Local ranking completed:")
        print(f"  Top opportunities selected: {len(top_opportunities)}")
        print(f"  Score threshold: {ranking_result['ranking_summary']['score_threshold']:.1f}")
        print()
        
        # Display top opportunities
        print("TOP OPPORTUNITIES FOR AI ANALYSIS:")
        for opp in top_opportunities:
            symbol = opp['symbol']
            score = opp['composite_score']
            ranking = opp['ranking']
            proceed = opp.get('proceed_to_ai', False)
            print(f"  #{ranking}: {symbol} (score: {score:.1f}) - AI: {'YES' if proceed else 'NO'}")
        print()
        
        # Phase 4: Claude AI Analysis
        print("PHASE 4: Claude AI Analysis")
        print("-" * 30)
        
        claude_analysis = ClaudeAnalysisExecutor(WorkflowStep.AI_ANALYSIS)
        ai_result = await claude_analysis.execute_step(ranking_result, context)
        
        print(f"✓ Claude AI analysis completed:")
        print(f"  Opportunities analyzed: {ai_result['opportunities_analyzed']}")
        print(f"  Opportunities failed: {ai_result['opportunities_failed']}")
        print(f"  Total cost: ${ai_result['total_cost']:.4f}")
        print(f"  Success rate: {ai_result['analysis_summary']['success_rate']:.1f}%")
        print(f"  Output directory: {ai_result['analysis_summary']['output_directory']}")
        print()
        
        # Display AI analysis results
        print("AI ANALYSIS RESULTS:")
        for symbol, result in ai_result['ai_analysis_results'].items():
            if result['success']:
                claude_response = result['claude_response']
                parsed = claude_response.get('parsed_analysis', {})
                rating = parsed.get('rating', 'N/A')
                confidence = parsed.get('confidence', 'N/A')
                thesis = parsed.get('thesis', 'N/A')[:80] + '...' if parsed.get('thesis') else 'N/A'
                
                print(f"  {symbol}:")
                print(f"    Rating: {rating}/100 (confidence: {confidence})")
                print(f"    Thesis: {thesis}")
                print(f"    Cost: ${result.get('cost', 0):.4f}")
                print(f"    File saved: {symbol}_{datetime.now().strftime('%Y%m%d')}_*.json")
            else:
                print(f"  {symbol}: FAILED - {result.get('error', 'Unknown error')}")
        print()
        
        # Final Summary
        print("WORKFLOW SUMMARY:")
        print(f"  Total symbols: {context.total_symbols}")
        print(f"  Completed: {context.completed_symbols}")
        print(f"  Failed: {context.failed_symbols}")
        print(f"  Total cost: ${context.total_cost:.4f}")
        print(f"  Duration: {context.duration_seconds:.1f} seconds")
        
        return ai_result
        
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        raise


async def analyze_single_opportunity():
    """
    Demonstrate analyzing a single pre-ranked opportunity.
    
    Useful for testing individual components or processing specific opportunities.
    """
    print("=" * 50)
    print("SINGLE OPPORTUNITY AI ANALYSIS")
    print("=" * 50)
    
    # Sample top opportunity data (from LocalRankingExecutor)
    sample_opportunity = {
        'symbol': 'AAPL',
        'composite_score': 87.5,
        'ranking': 1,
        'proceed_to_ai': True,
        'current_price': 175.50,
        'score_breakdown': {
            'technical': 88,
            'momentum': 85,
            'squeeze': 90,
            'options': 87,
            'quality': 89
        },
        'technical_data': {
            'rsi': 58.2,
            'macd': 1.25,
            'sma_20': 172.30,
            'bollinger_position': 0.65
        },
        'momentum_data': {
            'momentum_21d': 0.045,
            'momentum_score': 85.0,
            'relative_strength': 1.15
        },
        'squeeze_data': {
            'is_squeeze': True,
            'squeeze_strength': 0.75,
            'expansion_probability': 0.82
        },
        'best_call': {
            'symbol': 'AAPL240315C00170000',
            'strike': 170.0,
            'expiration': '2024-03-15',
            'dte': 45,
            'bid': 8.50,
            'ask': 8.75,
            'last': 8.60,
            'delta': 0.65,
            'gamma': 0.03,
            'theta': -0.08,
            'vega': 0.25,
            'implied_volatility': 0.28,
            'volume': 850,
            'open_interest': 1250,
            'score': 87
        },
        'enhanced_data': {
            'fundamentals': {
                'company_info': {
                    'name': 'Apple Inc.',
                    'sector': 'Technology',
                    'industry': 'Consumer Electronics'
                },
                'financial_health': {
                    'eps_ttm': 5.95,
                    'profit_margin': 0.253,
                    'roe': 0.147
                }
            },
            'news': [
                {
                    'title': 'Apple announces strong quarterly results',
                    'date': '2024-01-15',
                    'sentiment': 'positive'
                }
            ],
            'earnings': [
                {
                    'date': '2024-04-25',
                    'estimate': 1.35
                }
            ]
        },
        'warnings': [],
        'score_percentile': 95.5,
        'selection_rationale': 'Ranked #1 of 50 opportunities with excellent composite score'
    }
    
    # Prepare input data structure
    input_data = {
        'top_opportunities': [sample_opportunity],
        'ranking_summary': {
            'total_analyzed': 50,
            'top_n_selected': 1,
            'score_threshold': 87.5
        }
    }
    
    # Create context
    context = WorkflowExecutionContext(
        config=WorkflowConfig(enable_ai_analysis=True)
    )
    
    # Execute Claude analysis
    claude_analysis = ClaudeAnalysisExecutor(WorkflowStep.AI_ANALYSIS)
    
    try:
        print(f"Analyzing opportunity: {sample_opportunity['symbol']}")
        print(f"Composite score: {sample_opportunity['composite_score']}")
        print(f"Best call option: {sample_opportunity['best_call']['symbol']}")
        print()
        
        result = await claude_analysis.execute_step(input_data, context)
        
        if result['opportunities_analyzed'] > 0:
            symbol = sample_opportunity['symbol']
            analysis = result['ai_analysis_results'][symbol]
            
            if analysis['success']:
                print("✓ AI Analysis Completed Successfully")
                
                claude_response = analysis['claude_response']['parsed_analysis']
                print(f"  Claude Rating: {claude_response.get('rating', 'N/A')}/100")
                print(f"  Confidence: {claude_response.get('confidence', 'N/A')}")
                print(f"  Thesis: {claude_response.get('thesis', 'N/A')}")
                
                # Component scores
                component_scores = claude_response.get('component_scores', {})
                if component_scores:
                    print(f"  Component Scores:")
                    for component, score in component_scores.items():
                        print(f"    {component.replace('_', ' ').title()}: {score}")
                
                # Opportunities and risks
                opportunities = claude_response.get('opportunities', [])
                if opportunities:
                    print(f"  Opportunities:")
                    for opp in opportunities[:3]:
                        print(f"    • {opp}")
                
                risks = claude_response.get('risks', [])
                if risks:
                    print(f"  Risks:")
                    for risk in risks[:3]:
                        print(f"    • {risk}")
                
                print(f"  Cost: ${analysis.get('cost', 0):.4f}")
                print(f"  Data saved to: data/ai_analysis/{datetime.now().strftime('%Y%m%d')}/")
            else:
                print(f"✗ Analysis failed: {analysis.get('error', 'Unknown error')}")
        else:
            print("✗ No opportunities were analyzed")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def check_claude_integration():
    """Check Claude client integration and cost management."""
    print("=" * 40)
    print("CLAUDE INTEGRATION CHECK")
    print("=" * 40)
    
    from src.ai_analysis.claude_client import create_claude_client
    
    # Test client creation
    try:
        # This will use mock client if no API key is available
        client = create_claude_client("test-key", use_mock=True)
        
        print("✓ Claude client created successfully")
        
        # Check usage stats
        stats = client.get_usage_stats()
        print(f"  Daily cost limit: ${stats['daily_limit']}")
        print(f"  Remaining budget: ${stats['remaining_budget']}")
        print(f"  Hourly limit: {stats['hourly_limit']} requests")
        
        # Check if can make request
        can_request, reason = client.can_make_request()
        print(f"  Can make request: {can_request} ({reason})")
        
        # Test rate limiting
        print(f"  Rate limiting: {client.config.min_request_interval} seconds between requests")
        
    except Exception as e:
        print(f"✗ Claude integration error: {e}")


def display_saved_analysis_files():
    """Display information about saved AI analysis files."""
    print("=" * 40)
    print("SAVED ANALYSIS FILES")
    print("=" * 40)
    
    ai_analysis_dir = Path('data/ai_analysis')
    
    if ai_analysis_dir.exists():
        total_files = 0
        for date_dir in ai_analysis_dir.iterdir():
            if date_dir.is_dir():
                files = list(date_dir.glob('*.json'))
                if files:
                    print(f"Date: {date_dir.name}")
                    for file in files:
                        print(f"  {file.name} ({file.stat().st_size} bytes)")
                        total_files += 1
        
        print(f"\nTotal analysis files: {total_files}")
    else:
        print("No analysis files found. Run workflow to generate analyses.")


async def main():
    """Main example runner."""
    print("Claude Analysis Integration Examples")
    print("Select example to run:")
    print("1. Complete workflow with AI analysis")
    print("2. Single opportunity analysis") 
    print("3. Claude integration check")
    print("4. Display saved analysis files")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        await run_complete_workflow_with_ai()
    elif choice == '2':
        await analyze_single_opportunity()
    elif choice == '3':
        await check_claude_integration()
    elif choice == '4':
        display_saved_analysis_files()
    else:
        print("Invalid choice. Running Claude integration check...")
        await check_claude_integration()


if __name__ == "__main__":
    asyncio.run(main())
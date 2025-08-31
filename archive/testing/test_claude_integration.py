#!/usr/bin/env python3
"""Test Claude integration with the production script."""

import os
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

async def test_claude_integration():
    """Test the Claude integration functionality."""
    print("Testing Claude Integration")
    print("=" * 50)
    
    # Import the production app
    from run_simple_production import SimpleProductionApp
    
    # Create app instance
    app = SimpleProductionApp()
    
    # Override symbols to test just one
    app.symbols = ["AAPL"]
    
    print(f"Claude API Key present: {'Yes' if app.claude_key else 'No'}")
    print(f"Claude client available: {'Yes' if app.claude_client else 'No'}")
    
    if app.claude_client:
        stats = app.claude_client.get_usage_stats()
        print(f"Daily cost limit: ${stats['daily_limit']:.2f}")
        print(f"Current usage: ${stats['cost_today']:.4f}")
        
        can_request, reason = app.claude_client.can_make_request()
        print(f"Can make request: {can_request} ({reason})")
    
    print("\nTesting with AAPL...")
    result = await app.process_symbol("AAPL")
    
    print(f"\nResult Status: {result.get('status')}")
    if result.get('status') == 'success':
        ai_analysis = result.get('ai_analysis', {})
        print(f"Rating: {ai_analysis.get('rating', 'N/A')}")
        print(f"Claude analyzed: {ai_analysis.get('claude_analyzed', False)}")
        print(f"Confidence: {ai_analysis.get('confidence', 'N/A')}")
        print(f"Thesis: {ai_analysis.get('thesis', 'N/A')}")
        
        if ai_analysis.get('claude_analyzed'):
            print(f"Tokens used: {ai_analysis.get('tokens_used', 0)}")
            print(f"Cost: ${ai_analysis.get('cost_estimate', 0.0):.4f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_claude_integration())
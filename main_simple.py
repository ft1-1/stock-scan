#!/usr/bin/env python
"""Simplified main entry point for testing."""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Load environment
from dotenv import load_dotenv
load_dotenv()

def setup_minimal_logging():
    """Setup basic logging without complex dependencies."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_minimal_logging()

class SimpleApp:
    """Simplified application for testing."""
    
    def __init__(self):
        """Initialize the application."""
        from config.settings import Settings
        
        # Create settings with environment variables
        self.settings = Settings(
            eodhd_api_key=os.getenv("SCREENER_EODHD_API_KEY", ""),
            marketdata_api_key=os.getenv("SCREENER_MARKETDATA_API_KEY", ""),
            claude_api_key=os.getenv("SCREENER_CLAUDE_API_KEY"),
            specific_symbols=os.getenv("SCREENER_SPECIFIC_SYMBOLS"),
            min_market_cap=float(os.getenv("SCREENER_MIN_MARKET_CAP", "50000000")),
            max_market_cap=float(os.getenv("SCREENER_MAX_MARKET_CAP", "500000000000")),
            verbose=os.getenv("SCREENER_VERBOSE", "true").lower() == "true"
        )
        
        # Create output directory
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SimpleApp initialized")
    
    def health_check(self):
        """Perform a simple health check."""
        print("\n" + "="*60)
        print("HEALTH CHECK")
        print("="*60)
        
        # Check API keys
        print("\n📋 Configuration Status:")
        print(f"  • EODHD API Key: {'✅ Configured' if self.settings.eodhd_api_key else '❌ Missing'}")
        print(f"  • MarketData API Key: {'✅ Configured' if self.settings.marketdata_api_key else '❌ Missing'}")
        print(f"  • Claude API Key: {'✅ Configured' if self.settings.claude_api_key else '❌ Missing (optional)'}")
        
        # Check directories
        print("\n📁 Directory Status:")
        print(f"  • Output directory: {'✅ Exists' if self.output_dir.exists() else '❌ Missing'}")
        print(f"  • Logs directory: {'✅ Exists' if Path('logs').exists() else '⚠️ Will be created'}")
        
        # Check specific symbols
        if self.settings.specific_symbols:
            symbols = [s.strip() for s in self.settings.specific_symbols.split(',')]
            print(f"\n🎯 Specific Symbols Configured: {', '.join(symbols)}")
        else:
            print("\n🎯 No specific symbols set (will use general screening)")
        
        # Overall status
        if self.settings.eodhd_api_key and self.settings.marketdata_api_key:
            print("\n✅ System is ready to run!")
            return True
        else:
            print("\n❌ Missing required API keys. Please configure in .env file")
            return False
    
    def screen_symbols(self, symbols=None):
        """Run a simple screening test."""
        print("\n" + "="*60)
        print("SCREENING TEST")
        print("="*60)
        
        # Determine symbols to screen
        if not symbols and self.settings.specific_symbols:
            symbols = [s.strip().upper() for s in self.settings.specific_symbols.split(',')]
        elif not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL"]  # Default test symbols
        
        print(f"\n🔍 Screening symbols: {', '.join(symbols)}")
        
        # Mock results for testing
        results = []
        for symbol in symbols:
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": 150.00 + (ord(symbol[0]) - 65) * 10,  # Fake price
                "score": 75 + (ord(symbol[0]) - 65) % 20,  # Fake score
                "status": "screened"
            }
            results.append(result)
            print(f"  • {symbol}: Price=${result['price']:.2f}, Score={result['score']}")
        
        # Save results
        output_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"✅ Screening completed: {len(results)} symbols processed")
        
        return results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Options Screening Application (Simplified)")
    parser.add_argument('command', choices=['health', 'screen', 'test'], help='Command to run')
    parser.add_argument('--symbols', help='Comma-separated list of symbols')
    
    args = parser.parse_args()
    
    # Create app
    app = SimpleApp()
    
    # Execute command
    if args.command == 'health':
        success = app.health_check()
        return 0 if success else 1
    
    elif args.command == 'screen':
        symbols = None
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        app.screen_symbols(symbols)
        return 0
    
    elif args.command == 'test':
        print("Running all tests...")
        app.health_check()
        print("\n" + "-"*60)
        app.screen_symbols(["AAPL", "TSLA"])
        return 0
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
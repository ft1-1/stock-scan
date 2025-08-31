#!/usr/bin/env python
"""Simple production run with real API calls."""

import os
import sys
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import logging
import time

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import Claude client
try:
    from src.ai_analysis.claude_client import create_claude_client, ClaudeConfig
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logging.warning("Claude client not available - will use mock analysis")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleProductionApp:
    """Simplified production app with real API calls."""
    
    def __init__(self):
        self.eodhd_key = os.getenv("SCREENER_EODHD_API_KEY")
        self.marketdata_key = os.getenv("SCREENER_MARKETDATA_API_KEY")
        self.claude_key = os.getenv("SCREENER_CLAUDE_API_KEY")
        self.symbols = os.getenv("SCREENER_SPECIFIC_SYMBOLS", "AAPL,MSFT,GOOGL,TSLA").split(',')
        self.symbols = [s.strip() for s in self.symbols]
        
        # Initialize Claude client
        self.claude_client = None
        if CLAUDE_AVAILABLE and self.claude_key:
            try:
                daily_limit = float(os.getenv("CLAUDE_DAILY_COST_LIMIT", "5.0"))
                self.claude_client = create_claude_client(
                    api_key=self.claude_key,
                    daily_cost_limit=daily_limit,
                    use_mock=False
                )
                logger.info("Claude client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                self.claude_client = None
        else:
            if not self.claude_key:
                logger.warning("SCREENER_CLAUDE_API_KEY not found - using mock analysis")
            if not CLAUDE_AVAILABLE:
                logger.warning("Claude dependencies not available - using mock analysis")
        
    async def fetch_eodhd_data(self, symbol):
        """Fetch comprehensive data from EODHD."""
        print(f"\n📊 Fetching EODHD data for {symbol}...")
        
        data = {}
        
        # 1. Real-time quote
        async with aiohttp.ClientSession() as session:
            url = f"https://eodhd.com/api/real-time/{symbol}.US"
            params = {"api_token": self.eodhd_key, "fmt": "json"}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    quote = await resp.json()
                    data['quote'] = quote
                    print(f"   ✅ Price: ${quote.get('close', 0):.2f} | Volume: {quote.get('volume', 0):,}")
        
        # 2. Historical data for technicals
        async with aiohttp.ClientSession() as session:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            url = f"https://eodhd.com/api/eod/{symbol}.US"
            params = {
                "api_token": self.eodhd_key,
                "from": start_date.strftime('%Y-%m-%d'),
                "to": end_date.strftime('%Y-%m-%d'),
                "fmt": "json"
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    historical = await resp.json()
                    data['historical'] = historical
                    print(f"   ✅ Historical: {len(historical)} days")
        
        # 3. Fundamentals
        async with aiohttp.ClientSession() as session:
            url = f"https://eodhd.com/api/fundamentals/{symbol}.US"
            params = {"api_token": self.eodhd_key}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    fundamentals = await resp.json()
                    if fundamentals and 'General' in fundamentals:
                        data['fundamentals'] = {
                            'market_cap': fundamentals['General'].get('MarketCapitalization'),
                            'pe_ratio': fundamentals.get('Highlights', {}).get('PERatio'),
                            'dividend_yield': fundamentals.get('Highlights', {}).get('DividendYield'),
                            'sector': fundamentals['General'].get('Sector'),
                            'industry': fundamentals['General'].get('Industry')
                        }
                        print(f"   ✅ Fundamentals: {data['fundamentals']['sector']}")
        
        return data
    
    async def fetch_marketdata_options(self, symbol):
        """Fetch full options chain from MarketData with Greeks."""
        print(f"   📊 Fetching options chain from MarketData...")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
            headers = {"Authorization": f"Bearer {self.marketdata_key}"}
            
            # First get expirations to find the best one (45-75 days out)
            exp_url = f"https://api.marketdata.app/v1/options/expirations/{symbol}/"
            async with session.get(exp_url, headers=headers) as resp:
                if resp.status in [200, 203]:
                    exp_data = await resp.json()
                    if exp_data.get('s') == 'ok':
                        expirations = exp_data.get('expirations', [])
                        
                        # Find expiration closest to 60 days out
                        today = datetime.now().date()
                        target_date = today + timedelta(days=60)
                        
                        best_expiration = None
                        for exp in expirations:
                            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                            days_out = (exp_date - today).days
                            if 45 <= days_out <= 75:
                                best_expiration = exp
                                break
                        
                        if not best_expiration and expirations:
                            best_expiration = expirations[2] if len(expirations) > 2 else expirations[0]
                        
                        if best_expiration:
                            # Get options for specific expiration (gets cached data!)
                            params = {
                                'expiration': best_expiration,
                                'side': 'call',
                                'minOpenInterest': 100
                                # NO feed parameter - gets cached automatically
                            }
                            
                            print(f"   🔍 MarketData API Call:")
                            print(f"      URL: {url}")
                            print(f"      Expiration: {best_expiration}")
                            print(f"      Filters: minOI=100, side=call")
                            
                            async with session.get(url, headers=headers, params=params) as resp:
                                if resp.status in [200, 203]:
                                    data = await resp.json()
                                    if data.get('s') == 'ok':
                                        strikes = data.get('strike', [])
                                        
                                        if resp.status == 203:
                                            print(f"   ✅ Options (CACHED): {len(strikes)} contracts (1 credit total)")
                                        else:
                                            print(f"   ⚠️  Options (live): {len(strikes)} contracts ({len(strikes)} credits!)")
                                        
                                        # Return structured options data with Greeks
                                        return {
                                            'expiration': best_expiration,
                                            'contracts': len(strikes),
                                            'strikes': strikes[:10] if strikes else [],
                                            'deltas': data.get('delta', [])[:10] if data.get('delta') else [],
                                            'ivs': data.get('iv', [])[:10] if data.get('iv') else [],
                                            'data_source': 'MarketData',
                                            'data_type': 'cached' if resp.status == 203 else 'live'
                                        }
        
        return None
    
    def calculate_technicals(self, historical_data):
        """Calculate technical indicators."""
        if not historical_data or len(historical_data) < 20:
            return {}
            
        df = pd.DataFrame(historical_data)
        df['close'] = pd.to_numeric(df['close'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean() if len(df) >= 50 else None
        
        return {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'sma_20': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else df['close'].iloc[-1],
            'sma_50': sma_50.iloc[-1] if sma_50 is not None and not pd.isna(sma_50.iloc[-1]) else df['close'].iloc[-1],
            'price_vs_sma20': ((df['close'].iloc[-1] / sma_20.iloc[-1]) - 1) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
        }
    
    def generate_ai_package(self, symbol_data):
        """Generate data package for AI analysis."""
        package = {
            'symbol': symbol_data['symbol'],
            'timestamp': datetime.now().isoformat(),
            'stock_data': {
                'price': symbol_data.get('quote', {}).get('close', 0),
                'volume': symbol_data.get('quote', {}).get('volume', 0),
                'change_percent': symbol_data.get('quote', {}).get('change_p', 0)
            },
            'technicals': symbol_data.get('technicals', {}),
            'fundamentals': symbol_data.get('fundamentals', {}),
            'options': symbol_data.get('options', {}),
            'score': 75  # Mock score for now
        }
        return package
    
    def build_claude_prompt(self, symbol_data):
        """Build comprehensive prompt for Claude analysis."""
        symbol = symbol_data['symbol']
        quote = symbol_data.get('quote', {})
        technicals = symbol_data.get('technicals', {})
        fundamentals = symbol_data.get('fundamentals', {})
        options = symbol_data.get('options', {})
        
        prompt = f"""You are an expert stock analyst specializing in options trading opportunities. Analyze {symbol} and provide a comprehensive 0-100 rating.

## CURRENT MARKET DATA
**Symbol**: {symbol}
**Price**: ${quote.get('close', 0):.2f}
**Volume**: {quote.get('volume', 0):,}
**Change**: {quote.get('change_p', 0):.2f}%

## TECHNICAL INDICATORS
**RSI (14)**: {f"{technicals.get('rsi', 50):.1f}" if isinstance(technicals.get('rsi'), (int, float)) else 'N/A'}
**SMA 20**: ${technicals.get('sma_20', 0):.2f}
**SMA 50**: ${technicals.get('sma_50', 0):.2f}
**Price vs SMA20**: {technicals.get('price_vs_sma20', 0):.1f}%

## FUNDAMENTAL DATA
**Market Cap**: {f'${fundamentals.get("market_cap", 0):,}' if fundamentals.get('market_cap') else 'N/A'}
**P/E Ratio**: {fundamentals.get('pe_ratio', 'N/A')}
**Dividend Yield**: {fundamentals.get('dividend_yield', 'N/A')}
**Sector**: {fundamentals.get('sector', 'N/A')}
**Industry**: {fundamentals.get('industry', 'N/A')}

## OPTIONS AVAILABILITY
**Options Available**: {'Yes' if options else 'No'}
**Expirations**: {len(options.get('expirations', [])) if options else 0}

## SCORING FRAMEWORK (0-100 Total)

**1. TREND & MOMENTUM (35 points)**
- Uptrend strength and consistency
- Relative performance vs market
- Breakout quality and volume confirmation

**2. OPTIONS QUALITY (20 points)** 
- Bid-ask spreads and liquidity
- Open interest and volume
- Delta/theta optimization potential

**3. IV VALUE ASSESSMENT (15 points)**
- Implied volatility percentile
- Premium vs historical levels
- Volatility expansion potential

**4. SQUEEZE/VOLATILITY (10 points)**
- Bollinger band squeeze patterns
- Volatility contraction/expansion
- Breakout setup quality

**5. FUNDAMENTAL HEALTH (10 points)**
- Margin stability and growth
- Revenue growth consistency
- Balance sheet strength

**6. EVENTS & NEWS (10 points)**
- Earnings timing and expectations
- Catalyst timing and potential
- News sentiment and momentum

## ANALYSIS INSTRUCTIONS
1. Focus on options trading viability (liquidity, IV, Greeks)
2. Weight technical setup heavily for near-term trades
3. Consider fundamental backdrop for risk assessment
4. Evaluate volatility environment for options strategies
5. Rate overall opportunity quality from 0-100

Respond ONLY with JSON in this exact format:
{{
    "symbol": "{symbol}",
    "rating": 85,
    "confidence": "high",
    "thesis": "One sentence summary of why this is or isn't a good opportunity",
    "opportunities": ["bullet point 1", "bullet point 2", "bullet point 3"],
    "risks": ["risk factor 1", "risk factor 2", "risk factor 3"],
    "option_contract": {{
        "recommendation": "Specific options strategy recommendation",
        "entry_timing": "Timing guidance",
        "risk_management": "Position sizing and risk controls"
    }},
    "red_flags": ["any critical red flags if present"],
    "notes": "Additional context or considerations"
}}""".replace('{symbol}', symbol)

        return prompt
    
    async def analyze_with_claude(self, symbol_data):
        """Analyze stock with Claude AI or fallback to mock analysis."""
        symbol = symbol_data['symbol']
        
        # Try Claude analysis if available
        if self.claude_client:
            try:
                # Check if client can make request
                can_request, reason = self.claude_client.can_make_request()
                if not can_request:
                    logger.warning(f"Claude request blocked for {symbol}: {reason}")
                    return self._create_mock_analysis(symbol_data)
                
                # Build prompt and analyze
                prompt = self.build_claude_prompt(symbol_data)
                response = await self.claude_client.analyze_opportunity(prompt, symbol)
                
                if response.success and response.parsed_json:
                    logger.info(f"Claude analysis successful for {symbol} (rating: {response.parsed_json.get('rating', 'N/A')})")
                    # Add some metadata
                    analysis = response.parsed_json.copy()
                    analysis['claude_analyzed'] = True
                    analysis['analysis_timestamp'] = datetime.now().isoformat()
                    analysis['tokens_used'] = response.tokens_used
                    analysis['cost_estimate'] = response.cost_estimate
                    return analysis
                else:
                    logger.error(f"Claude analysis failed for {symbol}: {response.error}")
                    return self._create_mock_analysis(symbol_data)
                    
            except Exception as e:
                logger.error(f"Error in Claude analysis for {symbol}: {e}")
                return self._create_mock_analysis(symbol_data)
        else:
            # No Claude client available
            return self._create_mock_analysis(symbol_data)
    
    def _create_mock_analysis(self, symbol_data):
        """Create mock analysis when Claude is not available."""
        symbol = symbol_data['symbol']
        technicals = symbol_data.get('technicals', {})
        fundamentals = symbol_data.get('fundamentals', {})
        options = symbol_data.get('options', {})
        
        # Create a deterministic but varied rating based on available data
        base_rating = 60
        
        # Technical scoring
        rsi = technicals.get('rsi', 50)
        if 30 <= rsi <= 70:  # Good RSI range
            base_rating += 10
        
        price_vs_sma = technicals.get('price_vs_sma20', 0)
        if price_vs_sma > 0:  # Above moving average
            base_rating += 10
        
        # Fundamental scoring
        if fundamentals.get('pe_ratio') and 10 <= fundamentals.get('pe_ratio') <= 30:
            base_rating += 5
        
        if fundamentals.get('sector'):
            base_rating += 5
        
        # Options availability
        if options and options.get('expirations'):
            base_rating += 10
        
        # Add symbol-based variation for consistency
        symbol_modifier = hash(symbol) % 20 - 10
        final_rating = max(0, min(100, base_rating + symbol_modifier))
        
        return {
            'symbol': symbol,
            'rating': final_rating,
            'confidence': 'medium',
            'thesis': f"Mock analysis for {symbol} based on technical indicators (RSI: {rsi:.1f}, trend: {'up' if price_vs_sma > 0 else 'down'})",
            'opportunities': [
                f"RSI at {rsi:.1f} indicating {'oversold' if rsi < 30 else 'neutral' if rsi < 70 else 'overbought'} conditions",
                f"Price {'above' if price_vs_sma > 0 else 'below'} 20-day moving average",
                f"Options {'available' if options else 'not available'} for strategy implementation"
            ],
            'risks': [
                "Mock analysis - not based on real AI insights",
                "Limited technical data available",
                "Market volatility and general conditions"
            ],
            'option_contract': {
                'recommendation': f"Mock recommendation for {symbol} options",
                'entry_timing': "Technical analysis suggests current levels",
                'risk_management': "Standard position sizing recommended"
            },
            'red_flags': ["This is mock analysis - use real Claude analysis for trading decisions"],
            'notes': "Generated by fallback analysis system",
            'claude_analyzed': False,
            'analysis_timestamp': datetime.now().isoformat(),
            'tokens_used': 0,
            'cost_estimate': 0.0
        }
    
    async def process_symbol(self, symbol):
        """Process a single symbol through all steps."""
        print(f"\n{'='*50}")
        print(f"Processing: {symbol}")
        print('='*50)
        
        result = {'symbol': symbol, 'timestamp': datetime.now().isoformat()}
        
        try:
            # Step 1-3: Get EODHD data
            eodhd_data = await self.fetch_eodhd_data(symbol)
            result.update(eodhd_data)
            
            # Step 4: Calculate technicals
            if 'historical' in eodhd_data:
                technicals = self.calculate_technicals(eodhd_data['historical'])
                result['technicals'] = technicals
                print(f"   ✅ Technicals: RSI={technicals.get('rsi', 0):.1f}")
            
            # Step 5: Get options data
            options = await self.fetch_marketdata_options(symbol)
            if options:
                result['options'] = options
            
            # Step 6: Create AI package
            ai_package = self.generate_ai_package(result)
            result['ai_package_size'] = len(json.dumps(ai_package))
            print(f"   ✅ AI Package: {result['ai_package_size']} bytes")
            
            # Step 7: AI Analysis with Claude (60-second rate limiting)
            print(f"   🤖 Starting AI analysis for {symbol}...")
            
            # Add 60-second delay between Claude calls as required
            if hasattr(self, '_last_claude_call'):
                time_since_last = asyncio.get_event_loop().time() - self._last_claude_call
                if time_since_last < 60:
                    wait_time = 60 - time_since_last
                    print(f"   ⏱️ Rate limiting: waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
            
            ai_analysis = await self.analyze_with_claude(result)
            result['ai_analysis'] = ai_analysis
            self._last_claude_call = asyncio.get_event_loop().time()
            
            claude_used = ai_analysis.get('claude_analyzed', False)
            rating = ai_analysis.get('rating', 0)
            print(f"   ✅ AI Analysis: {rating}/100 ({'Claude' if claude_used else 'Mock'})")
            
            if claude_used:
                tokens = ai_analysis.get('tokens_used', 0)
                cost = ai_analysis.get('cost_estimate', 0.0)
                print(f"   💰 Cost: {tokens} tokens, ${cost:.4f}")
            
            # Show Claude usage stats if available
            if self.claude_client:
                stats = self.claude_client.get_usage_stats()
                print(f"   📊 Daily usage: ${stats['cost_today']:.4f}/${stats['daily_limit']:.2f}")
            
            result['status'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    async def run(self):
        """Run the complete workflow."""
        print("="*60)
        print("PRODUCTION RUN - FULL WORKFLOW")
        print("="*60)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Process all symbols
        results = []
        for symbol in self.symbols:
            result = await self.process_symbol(symbol)
            results.append(result)
        
        # Save results
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create summary
        summary = {
            'run_timestamp': datetime.now().isoformat(),
            'symbols': self.symbols,
            'results': results,
            'summary': {
                'total': len(results),
                'successful': len([r for r in results if r.get('status') == 'success']),
                'failed': len([r for r in results if r.get('status') == 'error']),
                'average_rating': np.mean([r.get('ai_analysis', {}).get('rating', 0) for r in results if r.get('status') == 'success']),
                'claude_analyses': len([r for r in results if r.get('ai_analysis', {}).get('claude_analyzed', False)]),
                'mock_analyses': len([r for r in results if r.get('status') == 'success' and not r.get('ai_analysis', {}).get('claude_analyzed', False)]),
                'total_tokens_used': sum([r.get('ai_analysis', {}).get('tokens_used', 0) for r in results if r.get('status') == 'success']),
                'total_cost': sum([r.get('ai_analysis', {}).get('cost_estimate', 0.0) for r in results if r.get('status') == 'success'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Display summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print('='*60)
        
        for result in results:
            if result.get('status') == 'success':
                price = result.get('quote', {}).get('close', 0)
                rsi = result.get('technicals', {}).get('rsi', 0)
                rating = result.get('ai_analysis', {}).get('rating', 0)
                claude_used = result.get('ai_analysis', {}).get('claude_analyzed', False)
                ai_type = "🤖" if claude_used else "🔧"
                print(f"✅ {result['symbol']}: ${price:.2f} | RSI: {rsi:.1f} | Rating: {rating}/100 {ai_type}")
            else:
                print(f"❌ {result['symbol']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Processed: {summary['summary']['total']}")
        print(f"   Successful: {summary['summary']['successful']}")
        print(f"   Failed: {summary['summary']['failed']}")
        print(f"   Avg Rating: {summary['summary']['average_rating']:.1f}/100")
        print(f"\n🤖 AI Analysis:")
        print(f"   Claude Analyses: {summary['summary']['claude_analyses']}")
        print(f"   Mock Analyses: {summary['summary']['mock_analyses']}")
        print(f"   Total Tokens: {summary['summary']['total_tokens_used']:,}")
        print(f"   Total Cost: ${summary['summary']['total_cost']:.4f}")
        
        # Show final Claude usage stats if available
        if hasattr(self, 'claude_client') and self.claude_client:
            final_stats = self.claude_client.get_usage_stats()
            print(f"\n💰 Claude Usage Today:")
            print(f"   Daily Budget: ${final_stats['daily_limit']:.2f}")
            print(f"   Used: ${final_stats['cost_today']:.4f}")
            print(f"   Remaining: ${final_stats['remaining_budget']:.4f}")
            print(f"   Requests: {final_stats['requests_today']}")
        
        print(f"\n💾 Results saved to: {output_file}")
        print("="*60)
        
        return summary

async def main():
    """Main entry point."""
    app = SimpleProductionApp()
    return await app.run()

if __name__ == "__main__":
    asyncio.run(main())
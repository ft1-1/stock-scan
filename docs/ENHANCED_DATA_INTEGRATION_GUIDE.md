# Enhanced Data Integration Guide for PMCC Scanner

## Overview

This guide provides a complete understanding of how the PMCC Scanner retrieves, processes, and packages enhanced financial data for Claude AI analysis. The system fetches 8 types of fundamental data from EODHD, combines it with PMCC options analysis, and formats it for AI consumption.

## Table of Contents
1. [Data Retrieval Process](#1-data-retrieval-process)
2. [Data Processing Pipeline](#2-data-processing-pipeline)
3. [Data Packaging for Claude](#3-data-packaging-for-claude)
4. [Complete Code Examples](#4-complete-code-examples)
5. [Integration Points](#5-integration-points)

---

## 1. Data Retrieval Process

### 1.1 Enhanced EODHD Provider Setup

The enhanced data retrieval starts with the `EnhancedEODHDProvider` class which fetches comprehensive fundamental data:

```python
# src/api/providers/enhanced_eodhd_provider.py

from eodhd import APIClient as EODHDAPIClient

class EnhancedEODHDProvider(DataProvider):
    def __init__(self, provider_type: ProviderType, config: Dict[str, Any]):
        super().__init__(provider_type, config)
        
        # Initialize official EODHD client
        api_token = config.get('api_token')
        self.client = EODHDAPIClient(api_token)
        
        # Define supported operations for enhanced data
        self._supported_operations = {
            'get_stock_quote', 
            'get_fundamental_data', 
            'get_calendar_events', 
            'get_technical_indicators', 
            'get_risk_metrics',
            'get_enhanced_stock_data'  # Main method that combines all
        }
```

### 1.2 The 8 Types of Enhanced Data Retrieved

```python
async def get_enhanced_stock_data(self, symbol: str) -> APIResponse:
    """
    Get comprehensive enhanced stock data combining all available data sources.
    Lines 1545-1637 in enhanced_eodhd_provider.py
    """
    
    # Gather all 8 data types in parallel
    tasks = [
        self.get_stock_quote(symbol),           # 1. Current quote data
        self.get_fundamental_data(symbol),      # 2. Company fundamentals
        self.get_calendar_events(symbol),       # 3. Earnings/dividends
        self.get_technical_indicators(symbol),  # 4. Technical analysis
        self.get_risk_metrics(symbol),          # 5. Risk measurements
        # Additional calls for complete data:
        self.get_company_news(symbol),          # 6. Recent news
        self.get_historical_prices(symbol),     # 7. Price history
        self.get_economic_events()              # 8. Macro context
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Create enhanced stock data object
    enhanced_data = EnhancedStockData(
        quote=quote,
        fundamentals=fundamentals,
        calendar_events=calendar_events,
        technical_indicators=technical_indicators,
        risk_metrics=risk_metrics,
        options_chain=None  # Options come from MarketData.app
    )
    
    return APIResponse(
        status=APIStatus.OK,
        data=enhanced_data,
        provider_metadata=metadata
    )
```

### 1.3 Example: Fetching Fundamental Data

```python
async def get_fundamental_data(self, symbol: str) -> APIResponse:
    """
    Lines 633-799 in enhanced_eodhd_provider.py
    """
    # Fetch fundamentals from EODHD
    response = await asyncio.get_event_loop().run_in_executor(
        None, self.client.get_fundamentals_data, f'{symbol}.US'
    )
    
    # Process into structured format
    fundamentals = FundamentalMetrics(
        # Company information
        market_cap=Decimal(str(general.get('MarketCapitalization', 0))),
        enterprise_value=Decimal(str(valuation.get('EnterpriseValue', 0))),
        
        # Financial ratios
        pe_ratio=Decimal(str(valuation.get('TrailingPE', 0))),
        peg_ratio=Decimal(str(valuation.get('PEGRatio', 0))),
        price_to_book=Decimal(str(valuation.get('PriceBookMRQ', 0))),
        
        # Profitability metrics
        profit_margin=Decimal(str(financials.get('ProfitMargin', 0))),
        operating_margin=Decimal(str(financials.get('OperatingMarginTTM', 0))),
        roe=Decimal(str(financials.get('ReturnOnEquityTTM', 0))),
        roa=Decimal(str(financials.get('ReturnOnAssetsTTM', 0))),
        
        # Growth metrics
        revenue_growth=Decimal(str(financials.get('QuarterlyRevenueGrowthYOY', 0))),
        earnings_growth=Decimal(str(financials.get('QuarterlyEarningsGrowthYOY', 0))),
        
        # Dividend information
        dividend_yield=Decimal(str(dividend_data.get('Yield', 0))),
        payout_ratio=Decimal(str(dividend_data.get('PayoutRatio', 0)))
    )
    
    return APIResponse(
        status=APIStatus.OK,
        data=fundamentals
    )
```

---

## 2. Data Processing Pipeline

### 2.1 PMCC Opportunity Preparation

The `ClaudeIntegrationManager` prepares PMCC opportunities with complete options data:

```python
# src/analysis/claude_integration.py

def prepare_opportunities_for_claude(self, opportunities) -> Dict[str, Any]:
    """
    Lines 413-685 in claude_integration.py
    Prepare PMCC opportunities data for Claude AI analysis.
    """
    
    prepared_opportunities = []
    
    for opp in opportunities:
        # Extract PMCC position details
        symbol = opp.symbol
        underlying_price = float(opp.underlying_price)
        
        # Extract LEAPS option (long call) with ALL Greeks
        long_call = opp.analysis.long_call
        leaps_data = {
            'option_symbol': long_call.option_symbol,
            'strike': float(long_call.strike),
            'expiration': long_call.expiration.isoformat(),
            'dte': long_call.dte,
            
            # Complete Greeks package
            'delta': float(long_call.delta),
            'gamma': float(long_call.gamma),
            'theta': float(long_call.theta),
            'vega': float(long_call.vega),
            'iv': float(long_call.iv),
            
            # Market data
            'bid': float(long_call.bid),
            'ask': float(long_call.ask),
            'volume': long_call.volume,
            'open_interest': long_call.open_interest
        }
        
        # Extract short call option with ALL Greeks
        short_call = opp.analysis.short_call
        short_data = {
            'option_symbol': short_call.option_symbol,
            'strike': float(short_call.strike),
            'expiration': short_call.expiration.isoformat(),
            'dte': short_call.dte,
            
            # Complete Greeks package
            'delta': float(short_call.delta),
            'gamma': float(short_call.gamma),
            'theta': float(short_call.theta),
            'vega': float(short_call.vega),
            'iv': float(short_call.iv),
            
            # Market data
            'bid': float(short_call.bid),
            'ask': float(short_call.ask),
            'volume': short_call.volume,
            'open_interest': short_call.open_interest
        }
        
        # Create comprehensive prepared opportunity
        prepared_opp = {
            'symbol': symbol,
            'underlying_price': underlying_price,
            'pmcc_score': float(opp.total_score),
            
            # Complete strategy details
            'strategy_details': {
                'net_debit': float(opp.analysis.net_debit),
                'credit_received': float(opp.analysis.credit_received),
                'max_profit': float(opp.analysis.risk_metrics.max_profit),
                'max_loss': float(opp.analysis.risk_metrics.max_loss),
                'breakeven_price': float(opp.analysis.risk_metrics.breakeven),
                'risk_reward_ratio': float(opp.analysis.risk_metrics.risk_reward_ratio)
            },
            
            # Complete options data
            'leaps_option': leaps_data,
            'short_option': short_data,
            
            # Include complete option chain if available
            'complete_option_chain': opp.get('complete_option_chain', {})
        }
        
        prepared_opportunities.append(prepared_opp)
    
    return {
        'opportunities': prepared_opportunities,
        'market_context': {
            'total_opportunities': len(prepared_opportunities),
            'average_pmcc_score': avg_score,
            'highest_pmcc_score': max_score
        },
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'prepared_for_claude': True
        }
    }
```

### 2.2 Enhanced Stock Data Integration

```python
async def analyze_single_opportunity_with_claude(
    self,
    opportunity_data: Dict[str, Any],
    enhanced_stock_data: Dict[str, Any],
    claude_provider,
    market_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Lines 815-871 in claude_integration.py
    Analyze a single PMCC opportunity using Claude AI.
    """
    
    # Call Claude provider for single opportunity analysis
    response = await claude_provider.analyze_single_pmcc_opportunity(
        opportunity_data,     # PMCC position with Greeks
        enhanced_stock_data,  # 8 types of EODHD data
        market_context       # Market conditions
    )
    
    # Integrate Claude's analysis with original opportunity
    integrated_opportunity = {
        **opportunity_data,
        
        # Claude analysis results
        'claude_analyzed': True,
        'claude_score': claude_analysis.get('pmcc_score', 0),
        'claude_analysis_summary': claude_analysis.get('analysis_summary', ''),
        'claude_recommendation': claude_analysis.get('recommendation', 'neutral'),
        
        # Combined scoring (60% PMCC, 40% AI by default)
        'combined_score': (original_pmcc_score * 0.6) + (claude_score * 0.4),
        
        # Detailed Claude scores breakdown
        'claude_scores_breakdown': {
            'risk_score': claude_analysis.get('risk_score', 0),
            'fundamental_score': claude_analysis.get('fundamental_score', 0),
            'technical_score': claude_analysis.get('technical_score', 0),
            'calendar_score': claude_analysis.get('calendar_score', 0),
            'strategy_score': claude_analysis.get('strategy_score', 0)
        }
    }
    
    return integrated_opportunity
```

---

## 3. Data Packaging for Claude

### 3.1 Prompt Building Process

The `ClaudeClient` builds comprehensive prompts with all available data:

```python
# src/api/claude_client.py

def _build_single_opportunity_prompt(
    self,
    opportunity_data: Dict[str, Any],
    enhanced_stock_data: Dict[str, Any],
    market_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Lines 305-1700+ in claude_client.py
    Build the analysis prompt for a single PMCC opportunity.
    """
    
    # Extract all data components
    symbol = opportunity_data.get('symbol', 'Unknown')
    underlying_price = opportunity_data.get('underlying_price', 0)
    
    # Extract enhanced data with comprehensive coverage
    fundamentals = enhanced_stock_data.get('fundamentals', {})
    tech_indicators = enhanced_stock_data.get('technical_indicators', {})
    news = enhanced_stock_data.get('recent_news', [])
    earnings = enhanced_stock_data.get('earnings_calendar', [])
    economic = enhanced_stock_data.get('economic_context', [])
    balance_sheet = enhanced_stock_data.get('balance_sheet', {})
    cash_flow = enhanced_stock_data.get('cash_flow', {})
    historical_prices = enhanced_stock_data.get('historical_prices', [])
    
    # Helper function for meaningful value filtering
    def get_meaningful_value(value, default_zeros=[0, 0.0]):
        """Return value if meaningful, None if zero/N/A"""
        if value is None or value == 'N/A' or value in default_zeros:
            return None
        return value
    
    # Build financial health section
    financial_fields = []
    
    # P/E Ratio - try multiple field names
    pe_candidates = ['pe_ratio', 'pe', 'trailing_pe', 'TrailingPE']
    for candidate in pe_candidates:
        pe_value = financial_health.get(candidate) or fundamentals.get(candidate)
        if pe_value:
            pe_field = f"P/E Ratio: {pe_value:.2f}"
            financial_fields.append(pe_field)
            break
    
    # EPS - try multiple field names
    eps_candidates = ['eps', 'eps_ttm', 'earnings_per_share', 'EPS']
    for candidate in eps_candidates:
        eps_value = financial_health.get(candidate) or fundamentals.get(candidate)
        if eps_value:
            eps_field = f"EPS: ${eps_value:.2f}"
            financial_fields.append(eps_field)
            break
    
    # Build the prompt with conditional sections
    prompt_sections = []
    
    # Always include strategy setup
    prompt_sections.append(f"""## PMCC OPPORTUNITY: {symbol}

**STRATEGY SETUP:**
- Current Stock Price: ${underlying_price}
- Net Debit: ${net_debit:.2f}
- LEAPS Strike: ${leaps_strike} | Expiration: {leaps_expiration} | DTE: {leaps_dte} | Delta: {leaps_delta:.3f}
- Short Call Strike: ${short_strike} | Expiration: {short_expiration} | DTE: {short_dte} | Delta: {short_delta:.3f}

**LIQUIDITY ASSESSMENT:**
- LEAPS: Volume {leaps_volume} | OI {leaps_oi} | Bid/Ask: ${leaps_bid:.2f}/${leaps_ask:.2f}
- Short Call: Volume {short_volume} | OI {short_oi} | Bid/Ask: ${short_bid:.2f}/${short_ask:.2f}""")
    
    # Add financial health if data exists
    if financial_fields:
        prompt_sections.append(f"""
**FINANCIAL HEALTH:**
- {' | '.join(financial_fields)}""")
    
    # Add technical indicators if data exists
    if technical_fields:
        prompt_sections.append(f"""
**TECHNICAL INDICATORS:**
- {' | '.join(technical_fields)}""")
    
    # Add dividend information if applicable
    if dividend_fields:
        prompt_sections.append(f"""
**DIVIDEND & CALENDAR:**
- {' | '.join(dividend_fields)}""")
    
    # Combine all sections
    final_prompt = '\n'.join(prompt_sections)
    
    return final_prompt
```

### 3.2 Complete Prompt Structure Example

```python
"""
## PMCC OPPORTUNITY: AAPL

**STRATEGY SETUP:**
- Current Stock Price: $175.50
- Daily Change: $2.30 (+1.33%)
- Daily Range: $173.20 - $176.80 | Open: $174.00
- Volume: 48,234,500
- Net Debit: $4,250.00
- LEAPS Strike: $150 | Expiration: 2025-01-17 | DTE: 365 | Delta: 0.750
- Short Call Strike: $180 | Expiration: 2024-02-16 | DTE: 45 | Delta: 0.300

**LIQUIDITY ASSESSMENT:**
- LEAPS: Volume 523 | OI 12,450 | Bid/Ask: $28.20/$28.50
- Short Call: Volume 1,234 | OI 8,765 | Bid/Ask: $3.40/$3.50

## COMPREHENSIVE ANALYSIS DATA

**COMPANY OVERVIEW:**
- Company: Apple Inc. | Sector: Technology | Industry: Consumer Electronics

**FINANCIAL HEALTH:**
- P/E Ratio: 29.45 | EPS: $5.95 | Revenue Growth: 8.2% | Profit Margin: 24.3%

**VALUATION METRICS:**
- Market Cap: $2.8T | PEG Ratio: 2.85 | P/B Ratio: 45.2 | EV/EBITDA: 22.5

**TECHNICAL INDICATORS:**
- Beta: 1.25 | 52-Week Range: $164.08 - $199.62 | RSI(14): 58.3
- 50-Day MA: $172.45 | 200-Day MA: $168.90

**DIVIDEND & CALENDAR:**
- Dividend Yield: 0.44% | Payout Ratio: 15.2%
- Ex-Dividend: 2024-02-09 | Next Earnings: 2024-01-25 (+14 days)

**EARNINGS CALENDAR:**
• 2024-01-25 (+14 days): EPS $2.10 (estimated)
• 2023-10-26 (-88 days): EPS $1.46 vs $1.39 est (+5.0% surprise)

**ANALYST SENTIMENT:**
- Avg Rating: 4.5/5 | Target Price: $195.00
- Total Analysts: 38 | Buy: 30 | Hold: 7 | Sell: 1

**BALANCE SHEET STRENGTH:**
- Total Cash: $65.2B | Total Debt: $109.3B | Debt/Equity: 1.95
- Working Capital: $58.3B | Current Ratio: 0.98

**OPTIONS ANALYSIS:**
- LEAPS Greeks: Delta: 0.750 (Moderate ITM) | Theta: -0.045 ($0.05/day decay)
- Short Call Greeks: Delta: 0.300 | Theta: -0.085 ($0.09/day decay)
- Net Greeks: Delta: 0.450 | Theta: +0.040 (positive time decay)
"""
```

---

## 4. Complete Code Examples

### 4.1 Full Enhanced Data Retrieval Flow

```python
# Example: Complete flow from data retrieval to Claude analysis

async def analyze_stock_with_enhanced_data(symbol: str):
    """
    Complete example of enhanced data retrieval and analysis
    """
    
    # Step 1: Initialize providers
    eodhd_provider = EnhancedEODHDProvider(
        ProviderType.EODHD,
        {'api_token': 'your_eodhd_token'}
    )
    
    claude_provider = ClaudeProvider(
        ProviderType.CLAUDE,
        {'api_key': 'your_claude_key'}
    )
    
    # Step 2: Get enhanced stock data (8 types)
    enhanced_data_response = await eodhd_provider.get_enhanced_stock_data(symbol)
    
    if not enhanced_data_response.is_success:
        print(f"Failed to get enhanced data: {enhanced_data_response.error}")
        return
    
    enhanced_stock_data = enhanced_data_response.data
    
    # Step 3: Get PMCC opportunities (from your scanner)
    pmcc_opportunities = await scanner.find_pmcc_opportunities(symbol)
    
    # Step 4: Prepare opportunities for Claude
    integration_manager = ClaudeIntegrationManager()
    prepared_data = integration_manager.prepare_opportunities_for_claude(
        pmcc_opportunities
    )
    
    # Step 5: Package enhanced data for Claude
    enhanced_data_dict = {
        'fundamentals': enhanced_stock_data.fundamentals.to_dict(),
        'technical_indicators': enhanced_stock_data.technical_indicators.to_dict(),
        'calendar_events': [event.to_dict() for event in enhanced_stock_data.calendar_events],
        'risk_metrics': enhanced_stock_data.risk_metrics.to_dict(),
        'recent_news': enhanced_stock_data.news_articles,
        'historical_prices': enhanced_stock_data.historical_prices,
        'balance_sheet': enhanced_stock_data.balance_sheet_data,
        'cash_flow': enhanced_stock_data.cash_flow_data,
        'earnings_calendar': enhanced_stock_data.earnings_events,
        'economic_context': enhanced_stock_data.economic_events,
        'analyst_sentiment': enhanced_stock_data.analyst_data,
        'quote': enhanced_stock_data.quote.to_dict()
    }
    
    # Step 6: Analyze with Claude
    for opportunity in prepared_data['opportunities']:
        claude_response = await integration_manager.analyze_single_opportunity_with_claude(
            opportunity,
            enhanced_data_dict,
            claude_provider,
            market_context={'vix_level': 15.5, 'market_trend': 'bullish'}
        )
        
        print(f"""
        Symbol: {claude_response['symbol']}
        Original PMCC Score: {claude_response['pmcc_score']}
        Claude AI Score: {claude_response['claude_score']}
        Combined Score: {claude_response['combined_score']}
        Recommendation: {claude_response['claude_recommendation']}
        Analysis: {claude_response['claude_analysis_summary']}
        """)
```

### 4.2 Data Structure Examples

```python
# Example of EnhancedStockData structure
enhanced_stock_data = {
    'quote': {
        'symbol': 'AAPL',
        'price': 175.50,
        'change': 2.30,
        'change_percent': 1.33,
        'volume': 48234500,
        'market_cap': 2800000000000
    },
    
    'fundamentals': {
        'company_info': {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'employees': 164000
        },
        'financial_health': {
            'pe_ratio': 29.45,
            'eps': 5.95,
            'revenue_growth': 0.082,
            'profit_margin': 0.243
        },
        'valuation_metrics': {
            'peg_ratio': 2.85,
            'price_to_book': 45.2,
            'ev_to_ebitda': 22.5
        },
        'dividend_info': {
            'dividend_yield': 0.0044,
            'payout_ratio': 0.152,
            'ex_dividend_date': '2024-02-09'
        }
    },
    
    'technical_indicators': {
        'rsi': 58.3,
        'rsi_date': '2024-01-11',
        'sma_50': 172.45,
        'sma_200': 168.90,
        'atr': 3.25,
        'volatility_30d': 18.5,
        'beta': 1.25
    },
    
    'calendar_events': [
        {
            'event_type': 'earnings',
            'date': '2024-01-25',
            'eps_estimate': 2.10,
            'revenue_estimate': 117500000000
        },
        {
            'event_type': 'dividend',
            'date': '2024-02-09',
            'amount': 0.24
        }
    ],
    
    'risk_metrics': {
        'sharpe_ratio': 1.85,
        'sortino_ratio': 2.10,
        'max_drawdown': -0.185,
        'downside_deviation': 0.012
    },
    
    'historical_prices': [
        {'date': '2024-01-11', 'open': 174.00, 'high': 176.80, 'low': 173.20, 'close': 175.50, 'volume': 48234500},
        {'date': '2024-01-10', 'open': 172.50, 'high': 174.50, 'low': 172.00, 'close': 173.20, 'volume': 42156000},
        # ... more price data
    ],
    
    'recent_news': [
        {
            'date': '2024-01-11',
            'title': 'Apple Announces New Product Line',
            'sentiment': 'positive',
            'relevance_score': 0.95
        }
    ],
    
    'earnings_calendar': [
        {
            'report_date': '2024-01-25',
            'quarter_date': '2023-12-31',
            'eps_actual': None,  # Future earnings
            'eps_estimate': 2.10,
            'revenue_actual': None,
            'revenue_estimate': 117500000000
        },
        {
            'report_date': '2023-10-26',
            'quarter_date': '2023-09-30',
            'eps_actual': 1.46,
            'eps_estimate': 1.39,
            'revenue_actual': 89498000000,
            'revenue_estimate': 89280000000
        }
    ]
}
```

---

## 5. Integration Points

### 5.1 Provider Factory Integration

```python
# src/api/provider_factory.py

class ProviderFactory:
    def get_enhanced_data_provider(self) -> DataProvider:
        """Get the appropriate provider for enhanced fundamental data"""
        
        # Enhanced EODHD is the primary source for fundamental data
        if self._is_provider_available(ProviderType.EODHD):
            return self._providers[ProviderType.EODHD]
        
        raise ValueError("No enhanced data provider available")
    
    def get_options_provider(self) -> DataProvider:
        """Get the appropriate provider for options data"""
        
        # MarketData.app is primary for options
        if self._is_provider_available(ProviderType.MARKETDATA):
            return self._providers[ProviderType.MARKETDATA]
        
        # Fallback to EODHD for options if needed
        if self._is_provider_available(ProviderType.EODHD):
            return self._providers[ProviderType.EODHD]
        
        raise ValueError("No options provider available")
```

### 5.2 Scanner Integration

```python
# src/analysis/scanner.py

async def scan_with_enhanced_analysis(self):
    """Run complete scan with enhanced data and Claude AI analysis"""
    
    # Step 1: Screen stocks
    stocks = await self.stock_screener.screen_stocks()
    
    # Step 2: Find PMCC opportunities
    pmcc_opportunities = []
    for stock in stocks:
        opportunities = await self.options_analyzer.find_pmcc_opportunities(stock)
        pmcc_opportunities.extend(opportunities)
    
    # Step 3: Get enhanced data for each stock
    enhanced_data_map = {}
    for symbol in set(opp.symbol for opp in pmcc_opportunities):
        enhanced_response = await self.eodhd_provider.get_enhanced_stock_data(symbol)
        if enhanced_response.is_success:
            enhanced_data_map[symbol] = enhanced_response.data
    
    # Step 4: Prepare for Claude analysis
    prepared_data = self.integration_manager.prepare_opportunities_for_claude(
        pmcc_opportunities
    )
    
    # Step 5: Analyze with Claude
    analyzed_opportunities = await self.integration_manager.analyze_opportunities_individually(
        prepared_data['opportunities'],
        enhanced_data_map,
        self.claude_provider,
        market_context={'vix_level': 15.5}
    )
    
    # Step 6: Sort by combined score
    analyzed_opportunities.sort(
        key=lambda x: x.get('combined_score', 0), 
        reverse=True
    )
    
    return analyzed_opportunities
```

### 5.3 Configuration Example

```python
# .env configuration

# EODHD for enhanced fundamental data
EODHD_API_TOKEN=your_eodhd_token
EODHD_ENABLE_ENHANCED_DATA=true

# Claude for AI analysis
CLAUDE_API_KEY=your_claude_api_key
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_MAX_TOKENS=4000

# MarketData.app for options data
MARKETDATA_API_TOKEN=your_marketdata_token

# Analysis settings
SCAN_USE_ENHANCED_DATA=true
SCAN_USE_CLAUDE_ANALYSIS=true
SCAN_TRADITIONAL_PMCC_WEIGHT=0.6
SCAN_AI_ANALYSIS_WEIGHT=0.4
```

## Summary

The enhanced data integration process:

1. **Retrieves** 8 types of fundamental data from EODHD in parallel
2. **Processes** the data into structured formats with meaningful value filtering
3. **Combines** with PMCC options analysis including complete Greeks
4. **Packages** into comprehensive prompts with conditional sections
5. **Sends** to Claude for AI analysis with 0-100 scoring
6. **Integrates** Claude's analysis back with original PMCC scores

The system ensures Claude receives complete financial context for accurate PMCC opportunity analysis, combining traditional quantitative metrics with AI-powered insights.

## Key Files Reference

- `src/api/providers/enhanced_eodhd_provider.py` - Enhanced data retrieval
- `src/analysis/claude_integration.py` - Data preparation and integration
- `src/api/claude_client.py` - Prompt building and Claude API calls
- `src/models/api_models.py` - Data structure definitions
- `src/api/provider_factory.py` - Provider orchestration
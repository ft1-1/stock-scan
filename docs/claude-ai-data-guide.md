# Claude AI Data Integration Guide for Trading Strategy Analysis

This guide explains how to effectively provide data to Claude AI for trading strategy analysis, based on the PMCC Scanner implementation. Use this as a template when building similar financial analysis applications.

## Table of Contents
1. [Overview](#overview)
2. [Data Structure Requirements](#data-structure-requirements)
3. [Essential Data Categories](#essential-data-categories)
4. [Data Formatting Best Practices](#data-formatting-best-practices)
5. [Prompt Engineering for Trading Analysis](#prompt-engineering-for-trading-analysis)
6. [Response Format Specification](#response-format-specification)
7. [Implementation Architecture](#implementation-architecture)
8. [Cost Optimization Strategies](#cost-optimization-strategies)
9. [Example Implementation](#example-implementation)

## Overview

The PMCC Scanner demonstrates a comprehensive approach to providing financial data to Claude AI for sophisticated trading analysis. The key principles are:

- **Comprehensive Data Package**: Provide all relevant data in a single, well-structured prompt
- **Domain-Specific Context**: Frame the AI as an expert in your specific trading strategy
- **Structured Output**: Request JSON responses with specific scoring frameworks
- **Data Quality Filtering**: Only send high-quality, complete data to maximize analysis value

## Data Structure Requirements

### 1. Trading Opportunity Data
Each trading opportunity should include:

```python
{
    "symbol": "AAPL",
    "underlying_price": 175.50,
    "strategy_score": 82.5,  # Pre-calculated by your system
    "liquidity_score": 95.0,
    
    # Strategy-specific details (PMCC example)
    "strategy_details": {
        "net_debit": 4500.00,
        "credit_received": 150.00,
        "max_profit": "unlimited",
        "max_loss": 4350.00,
        "breakeven_price": 168.50,
        "risk_reward_ratio": 3.2,
        "strategy_type": "Poor_Mans_Covered_Call"
    },
    
    # Complete options data with ALL Greeks
    "long_option": {
        "option_symbol": "AAPL240621C00160000",
        "strike": 160.0,
        "expiration": "2024-06-21",
        "dte": 180,
        "delta": 0.85,
        "gamma": 0.002,
        "theta": -0.05,
        "vega": 0.15,
        "iv": 0.28,
        "bid": 18.50,
        "ask": 18.75,
        "volume": 1250,
        "open_interest": 5600
    },
    
    "short_option": {
        "option_symbol": "AAPL240119C00185000",
        "strike": 185.0,
        "expiration": "2024-01-19",
        "dte": 30,
        "delta": 0.25,
        "gamma": 0.015,
        "theta": -0.08,
        "vega": 0.22,
        "iv": 0.32,
        "bid": 1.45,
        "ask": 1.55,
        "volume": 3200,
        "open_interest": 12000
    }
}
```

### 2. Enhanced Stock Data Package
Provide comprehensive fundamental and technical data:

```python
{
    "quote": {
        "last": 175.50,
        "change": 2.30,
        "change_percent": 1.33,
        "volume": 45000000,
        "high": 176.80,
        "low": 173.20,
        "open": 173.50
    },
    
    "fundamentals": {
        "market_cap": 2800000000000,  # In dollars
        "pe_ratio": 29.5,
        "eps_ttm": 5.95,
        "revenue_ttm": 385000000000,
        "profit_margin": 0.253,
        "roe": 0.147,
        "debt_to_equity": 1.95,
        "free_cash_flow": 99800000000,
        "dividend_yield": 0.0044
    },
    
    "technical_indicators": {
        "sma_20": 172.30,
        "sma_50": 168.50,
        "sma_200": 165.20,
        "rsi_14": 58.5,
        "macd": 1.25,
        "macd_signal": 0.98,
        "bollinger_upper": 178.50,
        "bollinger_lower": 166.10,
        "atr_14": 3.25,
        "volume_sma_20": 52000000
    },
    
    "calendar_events": {
        "next_earnings_date": "2024-01-25",
        "next_ex_dividend_date": "2024-02-09",
        "recent_earnings": [
            {
                "date": "2023-10-26",
                "eps_actual": 1.46,
                "eps_estimate": 1.39,
                "revenue_actual": 89500000000,
                "revenue_estimate": 89300000000
            }
        ]
    },
    
    "analyst_sentiment": {
        "avg_rating": 4.2,
        "target_price": 195.00,
        "buy_count": 28,
        "hold_count": 8,
        "sell_count": 1
    },
    
    "news": [
        {
            "date": "2023-12-20",
            "title": "Apple Announces New Product Line",
            "content": "Full article content..."
        }
    ]
}
```

## Essential Data Categories

### 1. Market Data (Required)
- Current price and daily price action
- Volume and liquidity metrics
- Bid/ask spreads for all instruments

### 2. Strategy-Specific Metrics (Required)
- Position setup details (strikes, expirations, etc.)
- Risk/reward calculations
- Greeks for all options legs
- Liquidity scores for entry/exit

### 3. Fundamental Data (Highly Recommended)
- Financial health indicators (cash flow, debt, margins)
- Valuation metrics (P/E, P/S, PEG)
- Growth metrics (revenue/earnings growth)

### 4. Technical Indicators (Recommended)
- Moving averages and trend indicators
- Momentum oscillators (RSI, MACD)
- Volatility measures (ATR, Bollinger Bands)

### 5. Calendar & Event Data (Critical for Options)
- Earnings dates and estimates
- Dividend dates and amounts
- Economic events affecting the sector

### 6. Sentiment Data (Optional but Valuable)
- Analyst ratings and price targets
- News sentiment and recent articles
- Options flow and put/call ratios

## Data Formatting Best Practices

### 1. Meaningful Value Filtering
Filter out null, zero, or meaningless values before sending to Claude:

```python
def get_meaningful_value(value):
    """Only include values that add analytical value"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 0 or value == -9999 or value == 9999:
            return None
    return value
```

### 2. Consistent Number Formatting
Format numbers appropriately for their context:

```python
def format_field(label, value, unit="", format_type="default"):
    meaningful_val = get_meaningful_value(value)
    if meaningful_val is None:
        return None
        
    if format_type == 'currency':
        return f"{label}: ${meaningful_val:.2f}{unit}"
    elif format_type == 'percentage':
        return f"{label}: {meaningful_val * 100:.2f}%{unit}"
    elif format_type == 'number':
        return f"{label}: {meaningful_val:,.0f}{unit}"
    else:
        return f"{label}: {meaningful_val}{unit}"
```

### 3. Hierarchical Data Organization
Structure data in logical sections:

```
## STRATEGY SETUP
- Core position details
- Risk/reward metrics
- Liquidity assessment

## COMPREHENSIVE ANALYSIS DATA
**COMPANY OVERVIEW**
- Business description and sector
- Market cap and employee count

**FINANCIAL HEALTH**
- Profitability metrics
- Growth indicators
- Cash flow strength

**TECHNICAL INDICATORS**
- Price trends and momentum
- Volatility measures

**CALENDAR EVENTS**
- Earnings and dividends
- Risk event timing
```

## Prompt Engineering for Trading Analysis

### 1. Expert Framing
Start your prompt by establishing Claude as a domain expert:

```
You are an expert options strategist specializing in [YOUR STRATEGY]. 
Analyze this specific opportunity using the comprehensive dataset provided 
and score it from 0-100.
```

### 2. Scoring Framework
Provide a clear scoring framework tailored to your strategy:

```
## SCORING FRAMEWORK (0-100 Total)

**1. EXECUTION RISK (30 points)**
- Entry/exit liquidity
- Spread management
- Position sizing considerations

**2. FUNDAMENTAL STRENGTH (25 points)**
- Financial health metrics
- Growth prospects
- Competitive position

**3. TECHNICAL SETUP (25 points)**
- Entry timing quality
- Trend alignment
- Risk/reward profile

**4. EVENT RISK (20 points)**
- Earnings proximity
- Dividend considerations
- Sector-specific catalysts
```

### 3. Critical Considerations
Include strategy-specific red and green flags:

```
## CRITICAL CONSIDERATIONS

**RED FLAGS (Avoid if present):**
- Liquidity below minimum thresholds
- Deteriorating fundamentals
- High event risk in near term

**GREEN FLAGS (Favorable conditions):**
- Strong technical setup at support
- Improving fundamental trends
- Favorable volatility environment
```

### 4. Analysis Instructions
Guide Claude's analytical approach:

```
## ANALYSIS INSTRUCTIONS

1. Prioritize strategy-specific factors over general analysis
2. Quantify risks with specific probabilities when possible
3. Focus on the strategy's typical time horizon
4. Consider position sizing based on risk profile
5. Weight recent events more heavily than historical data
```

## Response Format Specification

### JSON Response Structure
Always request structured JSON output:

```json
{
    "symbol": "AAPL",
    "strategy_score": 85,
    "component_scores": {
        "execution_risk": 26,
        "fundamental_strength": 22,
        "technical_setup": 21,
        "event_risk": 16
    },
    "recommendation": "buy",
    "confidence_level": 82,
    "key_risks": [
        "Earnings in 15 days may cause volatility",
        "Tech sector rotation risk",
        "Premium collection below average"
    ],
    "key_opportunities": [
        "Strong support at 170 level",
        "IV elevated vs historical",
        "Positive momentum indicators"
    ],
    "position_management": {
        "entry_timing": "Immediate - favorable setup",
        "position_size": "2-3% of portfolio",
        "exit_conditions": [
            "Break below 170 support",
            "IV collapse below 20%",
            "Adverse earnings surprise"
        ]
    }
}
```

### Enforcing JSON Output
Add clear instructions:

```
Respond ONLY with the JSON structure above - no additional commentary.
Base your analysis strictly on the provided dataset.
```

## Implementation Architecture

### 1. Data Pipeline Structure
```
Data Collection → Quality Filtering → Formatting → 
Prompt Construction → Claude API → Response Parsing → 
Integration with Trading System
```

### 2. Provider Pattern
Use a provider pattern for flexibility:

```python
class ClaudeProvider(DataProvider):
    def __init__(self, config):
        self.client = ClaudeClient(
            api_key=config.get('api_key'),
            model='claude-3-5-sonnet-20241022',
            max_tokens=4000,
            temperature=0.1  # Low for consistency
        )
    
    async def analyze_opportunity(self, opportunity_data, stock_data):
        prompt = self._build_analysis_prompt(opportunity_data, stock_data)
        response = await self.client.analyze(prompt)
        return self._parse_response(response)
```

### 3. Error Handling
Implement robust error handling:

```python
try:
    response = await claude_provider.analyze_opportunity(data)
    if not response.is_success:
        # Fallback to traditional scoring
        return self._calculate_traditional_score(data)
except (RateLimitError, APIError) as e:
    logger.error(f"Claude analysis failed: {e}")
    return self._create_failed_analysis_result(data)
```

## Cost Optimization Strategies

### 1. Data Completeness Filtering
Only send opportunities with sufficient data:

```python
def filter_by_data_quality(opportunities):
    return [opp for opp in opportunities 
            if opp.calculate_completeness_score() >= 60.0]
```

### 2. Batch Size Limits
Limit concurrent analyses:

```python
MAX_CONCURRENT_ANALYSES = 3
async with asyncio.Semaphore(MAX_CONCURRENT_ANALYSES):
    results = await asyncio.gather(*analysis_tasks)
```

### 3. Token Estimation
Estimate costs before sending:

```python
def estimate_tokens(data):
    # Rough estimation: ~200 tokens per stock
    estimated_input = len(data) * 200
    estimated_output = 500  # Per response
    
    input_cost = estimated_input * 0.000003   # $3/1M tokens
    output_cost = estimated_output * 0.000015  # $15/1M tokens
    
    return input_cost + output_cost
```

### 4. Caching Strategy
Cache analyses for similar setups:

```python
cache_key = f"{symbol}_{strategy_hash}_{date.today()}"
if cache_key in analysis_cache:
    return analysis_cache[cache_key]
```

## Example Implementation

### Complete Analysis Function
```python
async def analyze_trading_opportunity(
    opportunity: Dict[str, Any],
    enhanced_data: Dict[str, Any],
    claude_provider: ClaudeProvider
) -> Dict[str, Any]:
    """
    Analyze a trading opportunity with Claude AI.
    
    Args:
        opportunity: Strategy-specific opportunity data
        enhanced_data: Complete fundamental/technical data
        claude_provider: Initialized Claude provider
        
    Returns:
        Enhanced opportunity with AI insights
    """
    # Check data quality
    completeness = calculate_data_completeness(enhanced_data)
    if completeness < 60.0:
        logger.warning(f"Insufficient data for {opportunity['symbol']}")
        return add_default_scores(opportunity)
    
    # Build comprehensive prompt
    prompt = build_strategy_prompt(
        opportunity=opportunity,
        enhanced_data=enhanced_data,
        strategy_type="YOUR_STRATEGY"
    )
    
    # Get Claude analysis
    try:
        response = await claude_provider.analyze_single_opportunity(
            prompt=prompt,
            timeout=30.0
        )
        
        if response.is_success:
            # Merge AI insights with original data
            return merge_ai_insights(opportunity, response.data)
        else:
            logger.error(f"Claude analysis failed: {response.error}")
            return add_default_scores(opportunity)
            
    except Exception as e:
        logger.error(f"Error analyzing {opportunity['symbol']}: {e}")
        return add_default_scores(opportunity)


def build_strategy_prompt(opportunity, enhanced_data, strategy_type):
    """Build comprehensive prompt for strategy analysis."""
    
    sections = []
    
    # Strategy setup section
    sections.append(format_strategy_section(opportunity))
    
    # Add data sections based on availability
    if enhanced_data.get('fundamentals'):
        sections.append(format_fundamentals_section(enhanced_data['fundamentals']))
    
    if enhanced_data.get('technicals'):
        sections.append(format_technicals_section(enhanced_data['technicals']))
    
    if enhanced_data.get('calendar_events'):
        sections.append(format_events_section(enhanced_data['calendar_events']))
    
    # Construct final prompt
    return f"""
You are an expert {strategy_type} strategist. Analyze this opportunity:

{''.join(sections)}

## SCORING FRAMEWORK (0-100)
[Your specific scoring criteria]

## RESPONSE FORMAT
[JSON structure specification]

Respond only with JSON based on the provided data.
"""
```

## Key Takeaways

1. **Comprehensive Data**: Provide all relevant data in a single prompt to avoid multiple API calls
2. **Quality Over Quantity**: Filter and format data to ensure high signal-to-noise ratio
3. **Structured Output**: Always request specific JSON output format for easy integration
4. **Domain Expertise**: Frame Claude as an expert in your specific strategy
5. **Cost Management**: Implement filtering, batching, and caching to optimize API usage
6. **Error Resilience**: Always have fallback scoring mechanisms when AI analysis fails
7. **Clear Instructions**: Provide explicit analysis guidelines and scoring frameworks

This approach enables sophisticated AI-enhanced analysis while maintaining system reliability and cost efficiency. Adapt these patterns to your specific trading strategy and data sources.
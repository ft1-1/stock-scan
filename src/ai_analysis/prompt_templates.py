"""
Prompt Templates for Claude AI Options Analysis

Implements the structured 0-100 rating system:
- Trend & Momentum (35 points)
- Options Quality (20 points) 
- IV Value (15 points)
- Squeeze/Volatility (10 points)
- Fundamentals (10 points)
- Event & News (10 points)

Designed for consistent, parseable JSON responses.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RatingCriteria:
    """Defines the 0-100 rating system breakdown"""
    trend_momentum: int = 35
    options_quality: int = 20
    iv_value: int = 15
    squeeze_volatility: int = 10
    fundamentals: int = 10
    event_news: int = 10
    
    def get_total(self) -> int:
        return (self.trend_momentum + self.options_quality + self.iv_value + 
                self.squeeze_volatility + self.fundamentals + self.event_news)


class PromptTemplates:
    """
    Structured prompt templates for consistent Claude AI analysis.
    
    Provides templates optimized for options screening with detailed
    scoring rubrics and clear JSON response requirements.
    """
    
    def __init__(self):
        self.rating_criteria = RatingCriteria()
        
    def create_analysis_prompt(
        self,
        data_package: Dict[str, Any],
        analysis_context: Optional[str] = None
    ) -> str:
        """
        Create comprehensive analysis prompt for Claude.
        
        Args:
            data_package: Complete data package from DataPackager
            analysis_context: Optional context (market conditions, special notes)
            
        Returns:
            Formatted prompt ready for Claude API
        """
        symbol = data_package.get('metadata', {}).get('symbol', 'UNKNOWN')
        
        # Build prompt sections
        sections = []
        
        # System prompt and role definition
        sections.append(self._get_system_prompt())
        
        # Scoring framework
        sections.append(self._get_scoring_framework())
        
        # Data sections
        sections.append(self._format_opportunity_data(data_package.get('opportunity_analysis', {})))
        sections.append(self._format_option_contract_data(data_package.get('selected_option_contract', {})))
        sections.append(self._format_enhanced_data(data_package.get('enhanced_stock_data', {})))
        
        # Analysis context if provided
        if analysis_context:
            sections.append(f"\n## SPECIAL ANALYSIS CONTEXT\n{analysis_context}\n")
        
        # Response format requirements
        sections.append(self._get_response_format())
        
        # Final instructions
        sections.append(self._get_analysis_instructions(symbol))
        
        return "\n".join(sections)
    
    def _get_system_prompt(self) -> str:
        """Define Claude's role and expertise"""
        return """You are an expert quantitative options analyst specializing in systematic options screening and strategy evaluation. Your role is to analyze options opportunities using a data-driven approach that combines technical analysis, fundamental research, volatility assessment, and risk management principles.

Your analysis should be objective, systematic, and focused on the specific opportunity presented. Base all conclusions strictly on the provided data, and maintain consistency in your scoring methodology across different opportunities."""
    
    def _get_scoring_framework(self) -> str:
        """Define the detailed 0-100 scoring rubric"""
        return f"""## SCORING FRAMEWORK (0-100 TOTAL POINTS)

**1. TREND & MOMENTUM ANALYSIS ({self.rating_criteria.trend_momentum} points)**
   • Price Trend Strength (15 points):
     - Strong uptrend with momentum: 13-15 points
     - Moderate uptrend: 10-12 points  
     - Sideways/weak trend: 5-9 points
     - Downtrend: 0-4 points
   
   • Momentum Acceleration (10 points):
     - Strong accelerating momentum: 8-10 points
     - Steady momentum: 6-7 points
     - Weak/declining momentum: 3-5 points
     - Negative momentum: 0-2 points
   
   • Relative Strength (10 points):
     - Significantly outperforming market/sector: 8-10 points
     - Moderate outperformance: 6-7 points
     - In-line performance: 4-5 points
     - Underperforming: 0-3 points

**2. OPTIONS QUALITY ASSESSMENT ({self.rating_criteria.options_quality} points)**
   • Liquidity and Spread Quality (10 points):
     - Tight spreads (<2%), high volume/OI: 8-10 points
     - Moderate spreads (2-4%), decent liquidity: 6-7 points
     - Wide spreads (4-6%), limited liquidity: 3-5 points
     - Very wide spreads (>6%), poor liquidity: 0-2 points
   
   • Greeks Profile Optimization (10 points):
     - Optimal delta/theta/vega profile for strategy: 8-10 points
     - Good Greeks alignment: 6-7 points
     - Acceptable Greeks: 4-5 points
     - Poor Greeks profile: 0-3 points

**3. IMPLIED VOLATILITY VALUE ({self.rating_criteria.iv_value} points)**
   • IV Percentile Assessment (10 points):
     - IV in favorable range for strategy: 8-10 points
     - Moderately attractive IV: 6-7 points
     - Neutral IV conditions: 4-5 points
     - Unfavorable IV levels: 0-3 points
   
   • IV/HV Relationship (5 points):
     - Favorable IV vs HV for strategy: 4-5 points
     - Neutral IV/HV: 2-3 points
     - Unfavorable IV/HV: 0-1 points

**4. SQUEEZE/VOLATILITY DYNAMICS ({self.rating_criteria.squeeze_volatility} points)**
   • Volatility Compression Assessment (5 points):
     - Strong squeeze with expansion potential: 4-5 points
     - Moderate compression: 3 points
     - Normal volatility: 2 points
     - High volatility: 0-1 points
   
   • Breakout Probability (5 points):
     - High probability directional move: 4-5 points
     - Moderate breakout potential: 2-3 points
     - Low breakout probability: 0-1 points

**5. FUNDAMENTAL HEALTH ({self.rating_criteria.fundamentals} points)**
   • Financial Strength (5 points):
     - Strong margins, cash flow, balance sheet: 4-5 points
     - Adequate financial health: 3 points
     - Weak but stable: 1-2 points
     - Poor financial condition: 0 points
   
   • Growth Trajectory (5 points):
     - Strong consistent growth: 4-5 points
     - Moderate growth: 2-3 points
     - Flat/declining: 0-1 points

**6. EVENT RISK & NEWS SENTIMENT ({self.rating_criteria.event_news} points)**
   • Earnings Safety (5 points):
     - No earnings for >30 days: 4-5 points
     - Earnings 15-30 days: 2-3 points
     - Earnings <15 days: 0-1 points
   
   • News Sentiment (5 points):
     - Positive news flow: 4-5 points
     - Neutral news: 2-3 points
     - Negative news: 0-1 points

**TOTAL POSSIBLE: {self.rating_criteria.get_total()} POINTS**"""
    
    def _format_opportunity_data(self, opportunity_data: Dict[str, Any]) -> str:
        """Format opportunity data section"""
        if not opportunity_data:
            return "\n## OPPORTUNITY DATA\nNo opportunity data provided.\n"
        
        sections = ["\n## OPPORTUNITY ANALYSIS DATA"]
        
        # Basic info
        if 'basic_info' in opportunity_data:
            basic = opportunity_data['basic_info']
            sections.append(f"""
**OPPORTUNITY OVERVIEW**
• Symbol: {basic.get('symbol', 'N/A')}
• Current Price: ${basic.get('current_price', 'N/A')}
• Overall Score: {basic.get('overall_score', 'N/A')}
• Strategy Type: {basic.get('strategy_type', 'N/A')}""")
        
        # Technical analysis
        if 'technical_analysis' in opportunity_data:
            sections.append(self._format_technical_data(opportunity_data['technical_analysis']))
        
        # Momentum metrics
        if 'momentum_metrics' in opportunity_data:
            sections.append(self._format_momentum_data(opportunity_data['momentum_metrics']))
        
        # Squeeze detection
        if 'squeeze_detection' in opportunity_data:
            sections.append(self._format_squeeze_data(opportunity_data['squeeze_detection']))
        
        # Quantitative scores
        if 'quantitative_scores' in opportunity_data:
            sections.append(self._format_score_breakdown(opportunity_data['quantitative_scores']))
        
        return "\n".join(sections)
    
    def _format_option_contract_data(self, option_data: Dict[str, Any]) -> str:
        """Format selected option contract data"""
        if not option_data or 'error' in option_data:
            return "\n## SELECTED OPTION CONTRACT\nNo option contract data provided.\n"
        
        sections = ["\n## SELECTED OPTION CONTRACT"]
        
        # Contract details
        if 'contract_details' in option_data:
            details = option_data['contract_details']
            sections.append(f"""
**CONTRACT SPECIFICATIONS**
• Option Symbol: {details.get('symbol', 'N/A')}
• Underlying: {details.get('underlying_symbol', 'N/A')}
• Strike Price: ${details.get('strike_price', 'N/A')}
• Expiration: {details.get('expiration_date', 'N/A')}
• Days to Expiration: {details.get('days_to_expiration', 'N/A')}
• Type: {details.get('contract_type', 'N/A').upper()}""")
        
        # Pricing data
        if 'pricing_data' in option_data:
            pricing = option_data['pricing_data']
            sections.append(f"""
**PRICING INFORMATION**
• Bid: ${pricing.get('bid', 'N/A')}
• Ask: ${pricing.get('ask', 'N/A')}
• Last: ${pricing.get('last_price', 'N/A')}
• Mid: ${pricing.get('mid_price', 'N/A')}
• Spread: {pricing.get('spread_percent', 'N/A')}%""")
        
        # Greeks
        if 'greeks' in option_data:
            greeks = option_data['greeks']
            sections.append(f"""
**GREEKS ANALYSIS**
• Delta: {greeks.get('delta', 'N/A')}
• Gamma: {greeks.get('gamma', 'N/A')}
• Theta: {greeks.get('theta', 'N/A')}
• Vega: {greeks.get('vega', 'N/A')}
• Rho: {greeks.get('rho', 'N/A')}""")
        
        # Volatility metrics
        if 'volatility_metrics' in option_data:
            vol = option_data['volatility_metrics']
            sections.append(f"""
**VOLATILITY ANALYSIS**
• Implied Volatility: {vol.get('implied_volatility', 'N/A')}
• IV Percentile: {vol.get('iv_percentile', 'N/A')}
• IV Rank: {vol.get('iv_rank', 'N/A')}
• Historical Volatility: {vol.get('historical_volatility', 'N/A')}
• IV/HV Ratio: {vol.get('iv_hv_ratio', 'N/A')}""")
        
        # Liquidity metrics
        if 'liquidity_metrics' in option_data:
            liq = option_data['liquidity_metrics']
            sections.append(f"""
**LIQUIDITY ASSESSMENT**
• Volume: {liq.get('volume', 'N/A')}
• Open Interest: {liq.get('open_interest', 'N/A')}
• Volume/OI Ratio: {liq.get('volume_oi_ratio', 'N/A')}
• Average Volume: {liq.get('avg_volume', 'N/A')}
• Liquidity Score: {liq.get('liquidity_score', 'N/A')}""")
        
        return "\n".join(sections)
    
    def _format_enhanced_data(self, enhanced_data: Dict[str, Any]) -> str:
        """Format EODHD enhanced data section"""
        if not enhanced_data or 'error' in enhanced_data:
            return "\n## ENHANCED MARKET DATA\nNo enhanced data provided.\n"
        
        sections = ["\n## COMPREHENSIVE MARKET ANALYSIS"]
        
        # Current market data
        if 'current_market_data' in enhanced_data:
            market = enhanced_data['current_market_data']
            sections.append(f"""
**LIVE MARKET DATA**
• Current Price: ${market.get('current_price', 'N/A')}
• Change: ${market.get('change', 'N/A')} ({market.get('change_percent', 'N/A')}%)
• Volume: {market.get('volume', 'N/A')}""")
        
        # Fundamental analysis
        if 'fundamental_analysis' in enhanced_data:
            sections.append(self._format_fundamentals_section(enhanced_data['fundamental_analysis']))
        
        # Recent news
        if 'recent_news' in enhanced_data:
            sections.append(self._format_news_section(enhanced_data['recent_news']))
        
        # Earnings calendar
        if 'earnings_calendar' in enhanced_data:
            sections.append(self._format_earnings_section(enhanced_data['earnings_calendar']))
        
        # Sentiment analysis
        if 'sentiment_analysis' in enhanced_data:
            sections.append(self._format_sentiment_section(enhanced_data['sentiment_analysis']))
        
        # Economic context
        if 'macro_economic_context' in enhanced_data:
            sections.append(self._format_economic_context(enhanced_data['macro_economic_context']))
        
        return "\n".join(sections)
    
    def _format_technical_data(self, technical_data: Dict[str, Any]) -> str:
        """Format technical analysis section"""
        if not technical_data:
            return ""
        
        items = []
        for key, value in technical_data.items():
            if value is not None:
                items.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        if items:
            return f"\n**TECHNICAL INDICATORS**\n" + "\n".join(items)
        return ""
    
    def _format_momentum_data(self, momentum_data: Dict[str, Any]) -> str:
        """Format momentum analysis section"""
        if not momentum_data:
            return ""
        
        items = []
        for key, value in momentum_data.items():
            if value is not None:
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float) and 'return' in key.lower():
                    items.append(f"• {formatted_key}: {value:.2%}")
                else:
                    items.append(f"• {formatted_key}: {value}")
        
        if items:
            return f"\n**MOMENTUM ANALYSIS**\n" + "\n".join(items)
        return ""
    
    def _format_squeeze_data(self, squeeze_data: Dict[str, Any]) -> str:
        """Format squeeze detection section"""
        if not squeeze_data:
            return ""
        
        items = []
        for key, value in squeeze_data.items():
            if value is not None:
                items.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        if items:
            return f"\n**VOLATILITY SQUEEZE ANALYSIS**\n" + "\n".join(items)
        return ""
    
    def _format_score_breakdown(self, scores: Dict[str, Any]) -> str:
        """Format quantitative score breakdown"""
        if not scores:
            return ""
        
        items = []
        for key, value in scores.items():
            if value is not None:
                items.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        if items:
            return f"\n**QUANTITATIVE SCORING**\n" + "\n".join(items)
        return ""
    
    def _format_fundamentals_section(self, fundamentals: Dict[str, Any]) -> str:
        """Format fundamentals data"""
        if not fundamentals:
            return ""
        
        sections = ["\n**FUNDAMENTAL ANALYSIS**"]
        
        # Company overview
        if 'company_overview' in fundamentals:
            overview = fundamentals['company_overview']
            sections.append(f"• Company: {overview.get('name', 'N/A')}")
            sections.append(f"• Sector: {overview.get('sector', 'N/A')}")
            sections.append(f"• Industry: {overview.get('industry', 'N/A')}")
        
        # Financial metrics
        if 'financial_metrics' in fundamentals:
            financial = fundamentals['financial_metrics']
            if financial.get('eps_ttm'):
                sections.append(f"• EPS (TTM): ${financial['eps_ttm']}")
            if financial.get('profit_margin'):
                sections.append(f"• Profit Margin: {financial['profit_margin']:.1%}")
            if financial.get('roe'):
                sections.append(f"• ROE: {financial['roe']:.1%}")
        
        # Valuation
        if 'valuation' in fundamentals:
            valuation = fundamentals['valuation']
            if valuation.get('pe_ratio'):
                sections.append(f"• P/E Ratio: {valuation['pe_ratio']:.1f}")
        
        return "\n".join(sections) if len(sections) > 1 else ""
    
    def _format_news_section(self, news_data: Dict[str, Any]) -> str:
        """Format news analysis section"""
        if not news_data or not news_data.get('articles'):
            return ""
        
        sections = [f"\n**NEWS ANALYSIS ({news_data.get('articles_count', 0)} articles)**"]
        
        # Recent articles
        for article in news_data['articles'][:5]:  # Show top 5
            if article.get('title'):
                sections.append(f"• {article['date']}: {article['title'][:80]}...")
        
        # Sentiment summary
        if 'sentiment_summary' in news_data:
            sentiment = news_data['sentiment_summary']
            sections.append(f"• Overall Sentiment: {sentiment.get('overall', 'neutral')}")
        
        return "\n".join(sections)
    
    def _format_earnings_section(self, earnings_data: Dict[str, Any]) -> str:
        """Format earnings calendar section"""
        if not earnings_data:
            return ""
        
        sections = ["\n**EARNINGS CALENDAR**"]
        
        if earnings_data.get('next_earnings_date'):
            sections.append(f"• Next Earnings: {earnings_data['next_earnings_date']}")
        
        return "\n".join(sections) if len(sections) > 1 else ""
    
    def _format_sentiment_section(self, sentiment_data: Dict[str, Any]) -> str:
        """Format sentiment analysis section"""
        if not sentiment_data:
            return ""
        
        sections = ["\n**MARKET SENTIMENT**"]
        
        if sentiment_data.get('overall_sentiment'):
            sections.append(f"• Overall Sentiment: {sentiment_data['overall_sentiment']}")
        if sentiment_data.get('sentiment_score'):
            sections.append(f"• Sentiment Score: {sentiment_data['sentiment_score']}")
        
        return "\n".join(sections) if len(sections) > 1 else ""
    
    def _format_economic_context(self, economic_data: Dict[str, Any]) -> str:
        """Format economic context section"""
        if not economic_data:
            return ""
        
        sections = ["\n**ECONOMIC CONTEXT**"]
        
        if economic_data.get('recent_us_economic_events'):
            events = economic_data['recent_us_economic_events'][:3]  # Top 3 events
            for event in events:
                if event.get('event'):
                    sections.append(f"• {event['date']}: {event['event']}")
        
        return "\n".join(sections) if len(sections) > 1 else ""
    
    def _get_response_format(self) -> str:
        """Define required JSON response format"""
        return '''## REQUIRED RESPONSE FORMAT

You MUST respond with ONLY a valid JSON object in exactly this format:

```json
{
    "symbol": "TICKER",
    "rating": 85,
    "component_scores": {
        "trend_momentum": 28,
        "options_quality": 18,
        "iv_value": 12,
        "squeeze_volatility": 8,
        "fundamentals": 9,
        "event_news": 10
    },
    "confidence": "high",
    "thesis": "2-3 sentence overall investment thesis based on the data",
    "opportunities": [
        "Specific positive factor 1",
        "Specific positive factor 2",
        "Specific positive factor 3"
    ],
    "risks": [
        "Specific risk factor 1", 
        "Specific risk factor 2",
        "Specific risk factor 3"
    ],
    "option_contract": {
        "recommendation": "Specific actionable recommendation",
        "entry_timing": "Assessment of entry timing",
        "risk_management": "Key risk management considerations"
    },
    "red_flags": [
        "Critical warning if any (leave empty array if none)"
    ],
    "notes": "Any additional important context or caveats"
}
```

CRITICAL REQUIREMENTS:
• Respond with ONLY the JSON object - no additional text
• All component scores must sum to your overall rating
• Base ALL analysis strictly on the provided data
• Use specific data points to justify scores
• Confidence levels: "low", "medium", "high"
'''
    
    def _get_analysis_instructions(self, symbol: str) -> str:
        """Final analysis instructions"""
        return f"""## ANALYSIS INSTRUCTIONS FOR {symbol}

1. **Systematic Scoring**: Evaluate each component (trend/momentum, options quality, IV value, squeeze/volatility, fundamentals, event/news) independently using the point allocations above.

2. **Data-Driven Decisions**: Base all scores on specific quantitative metrics provided in the data. Reference actual numbers in your reasoning.

3. **Consistency**: Apply the same scoring standards regardless of the stock. A score of 85 should represent the same quality level across all analyses.

4. **Risk Focus**: Pay special attention to red flags that could invalidate the opportunity (poor liquidity, imminent earnings, deteriorating fundamentals).

5. **Strategy Alignment**: Consider how well this opportunity fits options strategies (directional bias, volatility conditions, time decay considerations).

6. **Completeness Check**: If critical data is missing, note it in your analysis but don't penalize the opportunity excessively for data gaps.

Analyze the {symbol} opportunity now using this framework and respond with the required JSON format."""


def create_analysis_prompt(
    data_package: Dict[str, Any],
    analysis_context: Optional[str] = None
) -> str:
    """
    Convenience function to create analysis prompt.
    
    Args:
        data_package: Complete data package from DataPackager
        analysis_context: Optional additional context
        
    Returns:
        Formatted prompt ready for Claude API
    """
    templates = PromptTemplates()
    return templates.create_analysis_prompt(data_package, analysis_context)


if __name__ == "__main__":
    # Example usage
    sample_package = {
        "metadata": {"symbol": "AAPL"},
        "opportunity_analysis": {
            "basic_info": {
                "symbol": "AAPL",
                "current_price": 175.50,
                "overall_score": 82.5
            },
            "technical_analysis": {
                "rsi": 58.2,
                "macd": 1.25
            }
        },
        "selected_option_contract": {
            "contract_details": {
                "symbol": "AAPL240315C00170000",
                "strike_price": 170.0
            },
            "pricing_data": {
                "bid": 8.50,
                "ask": 8.75
            }
        }
    }
    
    prompt = create_analysis_prompt(sample_package)
    print(f"Generated prompt length: {len(prompt)} characters")
    print("\nFirst 500 characters:")
    print(prompt[:500] + "...")
"""
Prompt Templates for Claude AI Stock Analysis

Implements the structured 0-100 rating system for stock purchases:
- Trend & Momentum (35 points)
- Technical Quality (35 points)
- Squeeze/Volatility (20 points)
- Fundamentals & Quality (10 points)

Designed for consistent, parseable JSON responses focused on stock purchase decisions.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RatingCriteria:
    """Defines the 0-100 rating system breakdown for stock analysis"""
    trend_momentum: int = 35
    technical_quality: int = 35
    squeeze_volatility: int = 20
    fundamentals_quality: int = 10
    
    def get_total(self) -> int:
        return (self.trend_momentum + self.technical_quality + 
                self.squeeze_volatility + self.fundamentals_quality)


class PromptTemplates:
    """
    Structured prompt templates for consistent Claude AI analysis.
    
    Provides templates optimized for stock purchase screening with detailed
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
        sections.append(self._format_risk_metrics_data(data_package.get('risk_metrics', {})))
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
        return """You are an expert quantitative stock analyst specializing in systematic stock screening and momentum-squeeze strategy evaluation. Your role is to analyze stock purchase opportunities using a data-driven approach that combines technical analysis, fundamental research, volatility assessment, and risk management principles.

Your analysis should be objective, systematic, and focused on the specific stock purchase opportunity presented. Evaluate stocks for their momentum potential, technical strength, volatility patterns, and overall quality. Base all conclusions strictly on the provided data, and maintain consistency in your scoring methodology across different opportunities."""
    
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

**2. TECHNICAL QUALITY ASSESSMENT ({self.rating_criteria.technical_quality} points)**
   • Chart Pattern & Support/Resistance (15 points):
     - Strong bullish patterns, clean breakouts: 13-15 points
     - Moderate technical setup: 10-12 points
     - Neutral/consolidating patterns: 5-9 points
     - Bearish patterns, weak support: 0-4 points
   
   • Volume & Liquidity Quality (10 points):
     - High volume confirmation, excellent liquidity: 8-10 points
     - Good volume, adequate liquidity: 6-7 points
     - Moderate volume/liquidity: 3-5 points
     - Low volume, poor liquidity: 0-2 points
     
   • Risk/Reward Profile (10 points):
     - Excellent risk/reward setup (>3:1): 8-10 points
     - Good risk/reward (2-3:1): 6-7 points
     - Adequate risk/reward (1.5-2:1): 3-5 points
     - Poor risk/reward (<1.5:1): 0-2 points

**3. SQUEEZE/VOLATILITY DYNAMICS ({self.rating_criteria.squeeze_volatility} points)**
   • Volatility Compression Assessment (10 points):
     - Strong squeeze with expansion potential: 8-10 points
     - Moderate compression: 6-7 points
     - Normal volatility: 3-5 points
     - High/unstable volatility: 0-2 points
   
   • Breakout Probability (10 points):
     - High probability directional move: 8-10 points
     - Moderate breakout potential: 5-7 points
     - Low breakout probability: 2-4 points
     - No clear directional bias: 0-1 points

**4. FUNDAMENTALS & QUALITY ({self.rating_criteria.fundamentals_quality} points)**
   • Financial Health & Position Sizing (5 points):
     - Strong balance sheet, manageable risk metrics: 4-5 points
     - Adequate financial health: 3 points
     - Weak but acceptable: 1-2 points
     - Poor financial condition: 0 points
   
   • Event Risk & News Sentiment (5 points):
     - Positive news, no near-term events: 4-5 points
     - Neutral conditions: 2-3 points
     - Negative sentiment or event risk: 0-1 points

**TOTAL POSSIBLE: {self.rating_criteria.get_total()} POINTS**

## KEY FOCUS AREAS FOR STOCK ANALYSIS:
- **Position Sizing**: Evaluate ATR-based stop losses and position sizing recommendations
- **Risk Management**: Assess Value-at-Risk (VaR) and volatility metrics for trade sizing
- **Entry Timing**: Analyze squeeze patterns and momentum for optimal entry points
- **Exit Strategy**: Consider technical levels and risk/reward for profit targets"""
    
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
    
    def _format_risk_metrics_data(self, risk_data: Dict[str, Any]) -> str:
        """Format stock risk assessment and position sizing data"""
        if not risk_data:
            return "\n## RISK ASSESSMENT & POSITION SIZING\nNo risk metrics data provided.\n"
        
        sections = ["\n## RISK ASSESSMENT & POSITION SIZING"]
        
        # Current price and volatility
        if 'current_price' in risk_data:
            sections.append(f"""
**CURRENT MARKET DATA**
• Current Price: ${risk_data.get('current_price', 'N/A')}
• Daily Volatility: {risk_data.get('daily_volatility', 'N/A'):.2%}
• Annualized Volatility: {risk_data.get('annualized_volatility', 'N/A'):.2%}""")
        
        # ATR-based risk metrics
        if 'atr_value' in risk_data:
            sections.append(f"""
**ATR RISK METRICS**
• ATR Value: ${risk_data.get('atr_value', 'N/A'):.2f}
• ATR Percentage: {risk_data.get('atr_percent', 'N/A'):.2%}
• Suggested Stop Loss: ${risk_data.get('suggested_stop_loss', 'N/A'):.2f}
• Stop Loss Risk: ${risk_data.get('stop_loss_risk_dollars', 'N/A'):.2f} ({risk_data.get('stop_loss_risk_percent', 'N/A'):.2%})""")
        
        # Value at Risk
        if 'value_at_risk_95' in risk_data:
            sections.append(f"""
**VALUE AT RISK (95% CONFIDENCE)**
• Daily VaR: ${risk_data.get('value_at_risk_95', 'N/A'):.2f}
• Position Risk Assessment: {"High" if risk_data.get('atr_percent', 0) > 0.04 else "Moderate" if risk_data.get('atr_percent', 0) > 0.02 else "Low"} volatility stock""")
        
        # Position sizing recommendation
        if 'position_size_shares' in risk_data:
            total_position_value = risk_data.get('current_price', 0) * risk_data.get('position_size_shares', 0)
            sections.append(f"""
**POSITION SIZING RECOMMENDATION**
• Recommended Shares: {risk_data.get('position_size_shares', 'N/A')} shares
• Total Position Value: ${total_position_value:.2f}
• Based on 2% portfolio risk with ATR stop loss
• Risk per trade optimized for momentum strategy""")
        
        return "\n".join(sections)
    
    def _format_enhanced_data(self, enhanced_data: Dict[str, Any]) -> str:
        """Format comprehensive enhanced data section"""
        if not enhanced_data or 'error' in enhanced_data:
            return "\n## COMPREHENSIVE MARKET ANALYSIS\nNo enhanced data provided.\n"
        
        sections = ["\n## COMPREHENSIVE MARKET ANALYSIS"]
        
        # Market context (VIX, S&P500, dollar strength)
        if 'market_context' in enhanced_data:
            sections.append(self._format_market_context(enhanced_data['market_context']))
        
        # Fundamental analysis (complete financials)
        if 'fundamental_analysis' in enhanced_data:
            sections.append(self._format_detailed_fundamentals(enhanced_data['fundamental_analysis']))
        
        # Recent news with full content
        if 'recent_news' in enhanced_data:
            sections.append(self._format_detailed_news(enhanced_data['recent_news']))
        
        # Earnings calendar
        if 'earnings_calendar' in enhanced_data:
            sections.append(self._format_earnings_section(enhanced_data['earnings_calendar']))
        
        # Price history analysis
        if 'price_history' in enhanced_data:
            sections.append(self._format_price_history(enhanced_data['price_history']))
        
        # Trend momentum analysis
        if 'trend_momentum_analysis' in enhanced_data:
            sections.append(self._format_trend_momentum_analysis(enhanced_data['trend_momentum_analysis']))
        
        # Squeeze breakout analysis
        if 'squeeze_breakout_analysis' in enhanced_data:
            sections.append(self._format_squeeze_breakout_analysis(enhanced_data['squeeze_breakout_analysis']))
        
        # Liquidity risk analysis
        if 'liquidity_risk_analysis' in enhanced_data:
            sections.append(self._format_liquidity_risk_analysis(enhanced_data['liquidity_risk_analysis']))
        
        # Local rating analysis
        if 'local_rating_analysis' in enhanced_data:
            sections.append(self._format_local_rating_analysis(enhanced_data['local_rating_analysis']))
        
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
    
    def _format_market_context(self, market_data: Dict[str, Any]) -> str:
        """Format market context section"""
        if not market_data:
            return ""
            
        sections = ["\n**MARKET CONTEXT**"]
        
        # Market status and trend
        sections.append(f"• Market Status: {market_data.get('market_status', 'N/A')}")
        sections.append(f"• Market Trend: {market_data.get('market_trend', 'N/A')}")
        
        # VIX data
        if 'volatility_index' in market_data:
            vix = market_data['volatility_index']
            sections.append(f"• VIX Level: {vix.get('level', 'N/A')} (Risk: {vix.get('risk_level', 'N/A')})")
        
        # S&P500 context
        if 'sp500' in market_data:
            sp = market_data['sp500']
            sections.append(f"• S&P500: ${sp.get('price', 'N/A')} ({sp.get('change_percent', 'N/A'):.2f}%)")
            sections.append(f"• Market Momentum: {sp.get('momentum', 'N/A')}")
        
        return "\n".join(sections)
    
    def _format_detailed_fundamentals(self, fundamentals: Dict[str, Any]) -> str:
        """Format comprehensive fundamentals data"""
        if not fundamentals:
            return ""
        
        sections = ["\n**COMPREHENSIVE FUNDAMENTAL ANALYSIS**"]
        
        # Company overview
        if 'company_overview' in fundamentals:
            overview = fundamentals['company_overview']
            sections.append(f"• Company: {overview.get('name', 'N/A')}")
            sections.append(f"• Sector: {overview.get('sector', 'N/A')} | Industry: {overview.get('industry', 'N/A')}")
            if overview.get('employees'):
                sections.append(f"• Employees: {overview['employees']:,}")
        
        # Financial metrics
        if 'financial_metrics' in fundamentals:
            financial = fundamentals['financial_metrics']
            sections.append("• **Key Financials:**")
            if financial.get('eps_ttm'):
                sections.append(f"  - EPS (TTM): ${financial['eps_ttm']}")
            if financial.get('profit_margin'):
                sections.append(f"  - Profit Margin: {financial['profit_margin']:.1%}")
            if financial.get('operating_margin'):
                sections.append(f"  - Operating Margin: {financial['operating_margin']:.1%}")
            if financial.get('roe'):
                sections.append(f"  - ROE: {financial['roe']:.1%}")
            if financial.get('revenue_growth_yoy') is not None:
                sections.append(f"  - Revenue Growth (YoY): {financial['revenue_growth_yoy']:.1%}")
            if financial.get('earnings_growth_yoy') is not None:
                sections.append(f"  - Earnings Growth (YoY): {financial['earnings_growth_yoy']:.1%}")
        
        # Valuation metrics
        if 'valuation' in fundamentals:
            valuation = fundamentals['valuation']
            sections.append("• **Valuation:**")
            if valuation.get('pe_ratio'):
                sections.append(f"  - P/E Ratio: {valuation['pe_ratio']:.1f}")
            if valuation.get('forward_pe'):
                sections.append(f"  - Forward P/E: {valuation['forward_pe']:.1f}")
            if valuation.get('price_to_sales'):
                sections.append(f"  - P/S Ratio: {valuation['price_to_sales']:.1f}")
            if valuation.get('price_to_book'):
                sections.append(f"  - P/B Ratio: {valuation['price_to_book']:.1f}")
        
        # Technical data (beta, 52-week range)
        if 'technical_data' in fundamentals:
            tech = fundamentals['technical_data']
            if tech.get('beta'):
                sections.append(f"• Beta: {tech['beta']:.2f}")
            if tech.get('52_week_high') and tech.get('52_week_low'):
                sections.append(f"• 52-Week Range: ${tech['52_week_low']:.2f} - ${tech['52_week_high']:.2f}")
        
        # Analyst ratings
        if 'analyst_ratings' in fundamentals:
            ratings = fundamentals['analyst_ratings']
            if ratings.get('avg_rating') and ratings.get('target_price'):
                sections.append(f"• Analyst Rating: {ratings['avg_rating']:.1f} | Target: ${ratings['target_price']:.2f}")
                total_analysts = sum([ratings.get('strong_buy', 0), ratings.get('buy', 0), 
                                    ratings.get('hold', 0), ratings.get('sell', 0), ratings.get('strong_sell', 0)])
                if total_analysts > 0:
                    sections.append(f"  - Coverage: {total_analysts} analysts")
        
        # Balance sheet highlights
        if 'balance_latest_sheet' in fundamentals:
            balance = fundamentals['balance_latest_sheet']
            if balance.get('debt_to_equity') is not None:
                sections.append(f"• Debt-to-Equity: {balance['debt_to_equity']:.2f}")
            if balance.get('cash_and_equivalents'):
                sections.append(f"• Cash: ${balance['cash_and_equivalents']:.0f}M")
        
        return "\n".join(sections)
    
    def _format_detailed_news(self, news_data: Dict[str, Any]) -> str:
        """Format detailed news with content"""
        if not news_data or not news_data.get('articles'):
            return ""
        
        sections = [f"\n**RECENT NEWS ANALYSIS ({news_data.get('articles_count', 0)} articles)**"]
        
        # Show top 3 articles with content excerpts
        for i, article in enumerate(news_data['articles'][:3]):
            if article.get('title') and article.get('content'):
                sections.append(f"• **{article['date'][:10]}**: {article['title']}")
                # First 200 characters of content
                content_excerpt = article['content'][:200] + "..." if len(article['content']) > 200 else article['content']
                sections.append(f"  Summary: {content_excerpt}")
                sections.append("")  # Add spacing
        
        return "\n".join(sections)
    
    def _format_price_history(self, price_data: Dict[str, Any]) -> str:
        """Format price history analysis"""
        if not price_data or not price_data.get('daily_prices'):
            return ""
        
        sections = ["\n**PRICE HISTORY ANALYSIS**"]
        
        prices = price_data['daily_prices']
        if len(prices) >= 5:
            recent_prices = prices[-5:]  # Last 5 days
            sections.append("• **Recent Price Action (Last 5 days):**")
            for price_data_point in recent_prices:
                change = price_data_point['close'] - price_data_point['open']
                change_pct = (change / price_data_point['open']) * 100
                sections.append(f"  - {price_data_point['date']}: ${price_data_point['close']:.2f} ({change_pct:+.1f}%) Vol: {price_data_point['volume']:,}")
        
        return "\n".join(sections)
    
    def _format_trend_momentum_analysis(self, trend_data: Dict[str, Any]) -> str:
        """Format trend momentum analysis"""
        if not trend_data:
            return ""
        
        sections = ["\n**TREND & MOMENTUM DETAILS**"]
        
        if trend_data.get('return_21d') is not None:
            sections.append(f"• 21-Day Return: {trend_data['return_21d']:.2f}%")
        if trend_data.get('return_63d') is not None:
            sections.append(f"• 63-Day Return: {trend_data['return_63d']:.2f}%")
        if trend_data.get('rsi') is not None:
            sections.append(f"• RSI(14): {trend_data['rsi']:.1f}")
        if trend_data.get('adx') is not None:
            sections.append(f"• ADX: {trend_data['adx']:.1f}")
        if trend_data.get('trend_quality_score') is not None:
            sections.append(f"• Trend Quality Score: {trend_data['trend_quality_score']:.2f}")
        
        return "\n".join(sections)
    
    def _format_squeeze_breakout_analysis(self, squeeze_data: Dict[str, Any]) -> str:
        """Format squeeze breakout analysis"""
        if not squeeze_data:
            return ""
        
        sections = ["\n**SQUEEZE & BREAKOUT ANALYSIS**"]
        
        if squeeze_data.get('atr_pct') is not None:
            sections.append(f"• ATR Percentage: {squeeze_data['atr_pct']:.2f}%")
        if squeeze_data.get('in_squeeze') is not None:
            sections.append(f"• In Squeeze: {squeeze_data['in_squeeze']}")
        if squeeze_data.get('bb_width') is not None:
            sections.append(f"• Bollinger Band Width: {squeeze_data['bb_width']:.2f}")
        if squeeze_data.get('keltner_width') is not None:
            sections.append(f"• Keltner Channel Width: {squeeze_data['keltner_width']:.2f}")
        
        return "\n".join(sections)
    
    def _format_liquidity_risk_analysis(self, liquidity_data: Dict[str, Any]) -> str:
        """Format liquidity risk analysis"""
        if not liquidity_data:
            return ""
        
        sections = ["\n**LIQUIDITY & RISK ANALYSIS**"]
        
        if liquidity_data.get('adv_dollars') is not None:
            sections.append(f"• Avg Daily Value: ${liquidity_data['adv_dollars']:,.0f}")
        if liquidity_data.get('liquidity_tier'):
            sections.append(f"• Liquidity Tier: {liquidity_data['liquidity_tier']}")
        if liquidity_data.get('news_activity'):
            sections.append(f"• News Activity: {liquidity_data['news_activity']}")
        
        return "\n".join(sections)
    
    def _format_local_rating_analysis(self, rating_data: Dict[str, Any]) -> str:
        """Format local rating analysis"""
        if not rating_data:
            return ""
        
        sections = ["\n**LOCAL QUANTITATIVE RATING**"]
        
        if rating_data.get('final_score') is not None:
            sections.append(f"• Local Score: {rating_data['final_score']:.1f}")
        
        if 'sub_scores' in rating_data:
            sub_scores = rating_data['sub_scores']
            sections.append("• **Component Breakdown:**")
            for component, score in sub_scores.items():
                sections.append(f"  - {component.replace('_', ' ').title()}: {score:.1f}")
        
        if 'key_features' in rating_data:
            features = rating_data['key_features']
            sections.append("• **Key Technical Features:**")
            for feature, value in features.items():
                if isinstance(value, float) and 'ret_' in feature:
                    sections.append(f"  - {feature}: {value:.2f}%")
                else:
                    sections.append(f"  - {feature}: {value}")
        
        return "\n".join(sections)
    
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
        "technical_quality": 25,
        "squeeze_volatility": 15,
        "fundamentals_quality": 7
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
    "position_sizing": {
        "recommended_action": "Buy/Hold/Avoid with specific reasoning",
        "entry_timing": "Assessment of optimal entry timing",
        "risk_management": "Stop-loss levels and position sizing guidance"
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

1. **Systematic Scoring**: Evaluate each component (trend/momentum, technical quality, squeeze/volatility, fundamentals/quality) independently using the point allocations above.

2. **Data-Driven Decisions**: Base all scores on specific quantitative metrics provided in the data. Reference actual numbers from the comprehensive dataset.

3. **Consistency**: Apply the same scoring standards regardless of the stock. A score of 85 should represent the same quality level across all analyses.

4. **Risk Focus**: Pay special attention to red flags that could invalidate the opportunity (poor fundamentals, negative news sentiment, high volatility, earnings risk).

5. **Stock Purchase Strategy**: Consider momentum patterns, squeeze setups, risk/reward ratios, and position sizing for stock purchase decisions.

6. **Enhanced Data Utilization**: Leverage all available data including market context, news sentiment, detailed financials, price history, and technical analysis.

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
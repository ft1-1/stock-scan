"""
Data Packager for Claude AI Integration

Creates comprehensive JSON data packages combining:
1. Locally calculated opportunity data (technicals, momentum, squeeze, etc.)
2. Option chain data with full Greeks and volatility metrics  
3. EODHD enhanced data (fundamentals, news, calendar events)

Designed for single opportunity analysis with 60-second rate limiting.
"""

import json
import logging
from datetime import datetime, date
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class DataCompletenessMetrics:
    """Tracks completeness of data package for quality assessment"""
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    options_score: float = 0.0
    calendar_score: float = 0.0
    news_score: float = 0.0
    overall_score: float = 0.0
    missing_fields: List[str] = None
    
    def __post_init__(self):
        if self.missing_fields is None:
            self.missing_fields = []


class DataPackager:
    """
    Combines multiple data sources into comprehensive packages for Claude AI analysis.
    
    Handles:
    - Opportunity data with quantitative scores
    - Complete option chain data with Greeks
    - EODHD enhanced data (fundamentals, news, calendar)
    - Data quality assessment and filtering
    """
    
    def __init__(self, min_completeness_threshold: float = 60.0):
        """
        Initialize data packager.
        
        Args:
            min_completeness_threshold: Minimum data completeness score to send to Claude
        """
        self.min_completeness_threshold = min_completeness_threshold
        
    def create_analysis_package(
        self,
        opportunity_data: Dict[str, Any],
        option_chain_data: Dict[str, Any],
        enhanced_eodhd_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive data package for Claude analysis.
        
        Args:
            opportunity_data: Locally calculated opportunity metrics
            option_chain_data: Selected option contract with full Greeks
            enhanced_eodhd_data: EODHD enhanced data per example
            
        Returns:
            Formatted data package ready for Claude API
        """
        logger.info(f"Creating analysis package for {opportunity_data.get('symbol', 'UNKNOWN')}")
        
        # Calculate data completeness
        completeness = self._assess_data_completeness(
            opportunity_data, option_chain_data, enhanced_eodhd_data
        )
        
        if completeness.overall_score < self.min_completeness_threshold:
            logger.warning(
                f"Data completeness {completeness.overall_score:.1f}% below threshold "
                f"{self.min_completeness_threshold}% for {opportunity_data.get('symbol')}"
            )
        
        # Build comprehensive package
        package = {
            "metadata": self._create_metadata(opportunity_data, completeness),
            "opportunity_analysis": self._format_opportunity_data(opportunity_data),
            "selected_option_contract": self._format_option_data(option_chain_data),
            "enhanced_stock_data": self._format_enhanced_data(enhanced_eodhd_data),
            "data_quality": asdict(completeness)
        }
        
        logger.debug(f"Package created with {len(json.dumps(package, default=str))} characters")
        return package
    
    def _create_metadata(self, opportunity_data: Dict[str, Any], completeness: DataCompletenessMetrics) -> Dict[str, Any]:
        """Create package metadata"""
        return {
            "symbol": opportunity_data.get('symbol', 'UNKNOWN'),
            "package_timestamp": datetime.now().isoformat(),
            "analysis_type": "options_screening_ai_rating",
            "data_completeness_score": completeness.overall_score,
            "meets_quality_threshold": completeness.overall_score >= self.min_completeness_threshold,
            "package_version": "1.0"
        }
    
    def _format_opportunity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format locally calculated opportunity data for Claude.
        
        Includes all technical indicators, momentum analysis, squeeze detection,
        and quantitative scoring results.
        """
        formatted = {
            "basic_info": {
                "symbol": data.get('symbol'),
                "current_price": data.get('current_price'),
                "overall_score": data.get('overall_score'),
                "confidence_level": data.get('confidence_level'),
                "strategy_type": data.get('strategy_type', 'options_screening')
            }
        }
        
        # Technical indicators
        if 'technical_indicators' in data:
            formatted["technical_analysis"] = self._clean_dict(data['technical_indicators'])
        
        # Momentum analysis
        if 'momentum_analysis' in data:
            formatted["momentum_metrics"] = self._clean_dict(data['momentum_analysis'])
        
        # Squeeze detection
        if 'squeeze_analysis' in data:
            formatted["squeeze_detection"] = self._clean_dict(data['squeeze_analysis'])
        
        # Scoring breakdown
        if 'score_breakdown' in data:
            formatted["quantitative_scores"] = self._clean_dict(data['score_breakdown'])
        
        # Risk metrics
        if 'risk_metrics' in data:
            formatted["risk_assessment"] = self._clean_dict(data['risk_metrics'])
        
        return formatted
    
    def _format_option_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format selected option contract data with all Greeks and metrics.
        
        Includes pricing, Greeks, liquidity, and IV analysis.
        """
        if not data:
            return {"error": "No option data provided"}
        
        formatted = {
            "contract_details": {
                "symbol": data.get('option_symbol'),
                "underlying_symbol": data.get('underlying_symbol'),
                "strike_price": data.get('strike'),
                "expiration_date": data.get('expiration'),
                "days_to_expiration": data.get('dte'),
                "contract_type": data.get('option_type', 'call')
            },
            
            "pricing_data": {
                "bid": data.get('bid'),
                "ask": data.get('ask'),
                "last_price": data.get('last'),
                "mid_price": self._calculate_mid_price(data.get('bid'), data.get('ask')),
                "spread_percent": self._calculate_spread_percent(
                    data.get('bid'), data.get('ask'), data.get('last')
                )
            },
            
            "greeks": {
                "delta": data.get('delta'),
                "gamma": data.get('gamma'),
                "theta": data.get('theta'),
                "vega": data.get('vega'),
                "rho": data.get('rho')
            },
            
            "volatility_metrics": {
                "implied_volatility": data.get('implied_volatility'),
                "iv_percentile": data.get('iv_percentile'),
                "iv_rank": data.get('iv_rank'),
                "historical_volatility": data.get('historical_volatility'),
                "iv_hv_ratio": self._calculate_iv_hv_ratio(
                    data.get('implied_volatility'), data.get('historical_volatility')
                )
            },
            
            "liquidity_metrics": {
                "volume": data.get('volume'),
                "open_interest": data.get('open_interest'),
                "volume_oi_ratio": self._calculate_volume_oi_ratio(
                    data.get('volume'), data.get('open_interest')
                ),
                "avg_volume": data.get('avg_volume'),
                "liquidity_score": data.get('liquidity_score')
            },
            
            "selection_metrics": {
                "option_score": data.get('option_score'),
                "selection_reason": data.get('selection_reason'),
                "risk_reward_ratio": data.get('risk_reward'),
                "probability_profit": data.get('prob_profit')
            }
        }
        
        return self._clean_dict(formatted)
    
    def _format_enhanced_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format EODHD enhanced data following the example structure.
        
        Includes 10 news articles (increased from 5), economic events,
        fundamentals, calendar events, and sentiment data.
        """
        if not data:
            logger.warning("No enhanced data provided to _format_enhanced_data")
            return {"error": "No enhanced data provided"}
        
        logger.info(f"Formatting enhanced data with keys: {list(data.keys())}")
        formatted = {}
        
        # Market context (VIX, SPY, market trend)
        if 'market_context' in data and data['market_context']:
            formatted["market_context"] = self._format_market_context(data['market_context'])
        
        # News (increased to 10 articles)
        if 'news' in data and data['news']:
            formatted["recent_news"] = self._format_news_data(data['news'])
        
        # Filtered fundamentals
        if 'fundamentals' in data and data['fundamentals']:
            formatted["fundamental_analysis"] = self._format_fundamentals(data['fundamentals'])
        
        # Live price data
        if 'live_price' in data and data['live_price']:
            formatted["current_market_data"] = self._format_live_price(data['live_price'])
        
        # Earnings calendar
        if 'earnings' in data and data['earnings']:
            formatted["earnings_calendar"] = self._format_earnings_data(data['earnings'])
        
        # Historical prices (30 days)
        if 'historical_prices' in data and data['historical_prices']:
            formatted["price_history"] = self._format_historical_prices(data['historical_prices'])
        
        # REMOVED: Sentiment analysis - Claude will determine sentiment from news articles
        
        # Technical indicators from EODHD
        if 'technical_indicators' in data and data['technical_indicators']:
            formatted["eodhd_technical_indicators"] = self._format_technical_indicators(data['technical_indicators'])
        
        # Also include technicals if present (different key name)
        if 'technicals' in data and data['technicals']:
            formatted["technical_data"] = data['technicals']
        
        # Risk metrics (Sharpe, Sortino, Max Drawdown)
        if 'risk_metrics' in data and data['risk_metrics']:
            formatted["risk_metrics"] = self._format_risk_metrics(data['risk_metrics'])
        
        # ENHANCED TECHNICAL ANALYSIS - NEW FIELDS
        # Trend & Momentum indicators
        if 'trend_momentum' in data and data['trend_momentum']:
            formatted["trend_momentum_analysis"] = self._clean_dict(data['trend_momentum'])
        
        # Squeeze & Breakout indicators
        if 'squeeze_breakout' in data and data['squeeze_breakout']:
            formatted["squeeze_breakout_analysis"] = self._clean_dict(data['squeeze_breakout'])
        
        # Liquidity & Risk metrics
        if 'liquidity_risk' in data and data['liquidity_risk']:
            formatted["liquidity_risk_analysis"] = self._clean_dict(data['liquidity_risk'])
        
        # Options scoring details
        if 'options_scoring' in data and data['options_scoring']:
            formatted["options_scoring_details"] = self._clean_dict(data['options_scoring'])
        
        # LOCAL RATING SYSTEM RESULTS
        if 'local_rating' in data and data['local_rating']:
            formatted["local_rating_analysis"] = {
                'pre_score': data['local_rating'].get('pre_score'),
                'final_score': data['local_rating'].get('final_score'),
                'sub_scores': data['local_rating'].get('sub_scores'),
                'red_flags': data['local_rating'].get('red_flags'),
                'penalties_applied': data['local_rating'].get('penalties'),
                'key_features': {
                    'ret_21d': data['local_rating'].get('features', {}).get('ret_21d'),
                    'ret_63d': data['local_rating'].get('features', {}).get('ret_63d'),
                    'rsi14': data['local_rating'].get('features', {}).get('rsi14'),
                    'adx14': data['local_rating'].get('features', {}).get('adx14'),
                    'iv_percentile': data['local_rating'].get('features', {}).get('iv_percentile_atm'),
                    'days_to_earnings': data['local_rating'].get('features', {}).get('days_to_earnings')
                }
            }
        
        logger.info(f"Formatted enhanced data contains: {list(formatted.keys())}")
        return self._clean_dict(formatted)
    
    def _format_market_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format market context data including VIX, SPY, and market trends"""
        if not context:
            return {}
        
        # Helper function to safely convert to float
        def safe_float(val, default=0):
            if val is None or val == 'NA' or val == 'N/A':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        formatted = {
            "timestamp": context.get('timestamp'),
            "market_status": context.get('market_status'),
            "market_trend": context.get('market_trend', 'unknown')
        }
        
        # VIX data
        if 'vix' in context:
            vix = context['vix']
            
            vix_level = safe_float(vix.get('level'), 0)
            formatted["volatility_index"] = {
                "level": vix_level,
                "change": safe_float(vix.get('change'), 0),
                "change_percent": safe_float(vix.get('change_percent'), 0),
                "interpretation": vix.get('interpretation'),
                "risk_level": "high" if vix_level > 30 else "moderate" if vix_level > 20 else "low"
            }
        
        # SPY/Market data
        if 'spy' in context:
            spy = context['spy']
            # Use the safe_float function defined above
            spy_change_pct = safe_float(spy.get('change_percent'), 0)
            formatted["sp500"] = {
                "price": safe_float(spy.get('price'), 0),
                "change": safe_float(spy.get('change'), 0),
                "change_percent": spy_change_pct,
                "volume": int(safe_float(spy.get('volume'), 0)),
                "momentum": "bullish" if spy_change_pct > 0 else "bearish"
            }
        
        # Dollar Index
        if 'dollar_index' in context:
            dxy = context['dollar_index']
            # Use the safe_float function defined above
            dxy_change = safe_float(dxy.get('change'), 0)
            formatted["dollar_strength"] = {
                "level": safe_float(dxy.get('level'), 0),
                "change": dxy_change,
                "change_percent": safe_float(dxy.get('change_percent'), 0),
                "trend": "strengthening" if dxy_change > 0 else "weakening"
            }
        
        return formatted
    
    def _format_news_data(self, news: List[Dict]) -> Dict[str, Any]:
        """Format news data - full articles, no truncation"""
        if not news:
            return {}
        
        articles = []
        for article in news[:10]:  # Keep 10 articles limit
            if isinstance(article, dict):
                articles.append({
                    "date": article.get('date'),
                    "title": article.get('title'),
                    "content": article.get('content'),  # Full content, no truncation
                    "url": article.get('url')
                    # Removed sentiment - Claude will determine on its own
                })
        
        return {
            "articles": articles,
            "articles_count": len(articles)
            # Removed sentiment_summary
        }
    
    def _format_fundamentals(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Format filtered fundamentals data"""
        if not fundamentals:
            return {}
        
        # Use the filtered structure from the EODHD example
        formatted = {}
        
        if 'company_info' in fundamentals:
            formatted["company_overview"] = fundamentals['company_info']
        
        if 'financial_health' in fundamentals:
            formatted["financial_metrics"] = fundamentals['financial_health']
        
        if 'valuation_metrics' in fundamentals:
            formatted["valuation"] = fundamentals['valuation_metrics']
        
        if 'stock_technicals' in fundamentals:
            formatted["technical_data"] = fundamentals['stock_technicals']
        
        if 'dividend_info' in fundamentals:
            formatted["dividend_data"] = fundamentals['dividend_info']
        
        if 'analyst_sentiment' in fundamentals:
            formatted["analyst_ratings"] = fundamentals['analyst_sentiment']
        
        if 'ownership_structure' in fundamentals:
            formatted["institutional_data"] = fundamentals['ownership_structure']
        
        # Financial statements (quarterly data)
        for statement in ['balance_sheet', 'income_statement', 'cash_flow']:
            if statement in fundamentals:
                formatted[statement.replace('_', '_latest_')] = fundamentals[statement]
        
        return formatted
    
    def _format_live_price(self, live_price: Dict[str, Any]) -> Dict[str, Any]:
        """Format live price data"""
        if not live_price:
            return {}
        
        return {
            "current_price": live_price.get('price'),
            "change": live_price.get('change'),
            "change_percent": live_price.get('change_p'),
            "volume": live_price.get('volume'),
            "timestamp": live_price.get('timestamp')
        }
    
    def _format_earnings_data(self, earnings: List[Dict]) -> Dict[str, Any]:
        """Format earnings calendar data"""
        if not earnings:
            return {}
        
        upcoming = []
        for earning in earnings:
            if isinstance(earning, dict):
                upcoming.append({
                    "date": earning.get('date'),
                    "estimate": earning.get('estimate'),
                    "symbol": earning.get('symbol')
                })
        
        return {
            "upcoming_earnings": upcoming,
            "next_earnings_date": upcoming[0].get('date') if upcoming else None
        }
    
    def _format_historical_prices(self, prices: List[Dict]) -> Dict[str, Any]:
        """Format 30-day historical price data"""
        if not prices:
            return {}
        
        # Take the most recent 30 days (last 30 items since data is in ascending order)
        price_data = []
        recent_prices = prices[-30:] if len(prices) > 30 else prices
        for price in recent_prices:
            if isinstance(price, dict):
                price_data.append({
                    "date": price.get('date'),
                    "open": price.get('open'),
                    "high": price.get('high'),
                    "low": price.get('low'),
                    "close": price.get('close'),
                    "volume": price.get('volume')
                })
        
        return {
            "daily_prices": price_data,
            "data_points": len(price_data),
            "date_range": {
                "start": price_data[0].get('date') if price_data else None,
                "end": price_data[-1].get('date') if price_data else None
            }
        }
    
    def _format_sentiment_data(self, sentiment: Any) -> Dict[str, Any]:
        """Format sentiment analysis data - handles both dict and list formats"""
        if not sentiment:
            return {}
        
        # Handle list format from sentiments endpoint
        if isinstance(sentiment, list) and sentiment:
            # Get most recent sentiment data
            latest = sentiment[0] if sentiment else {}
            if isinstance(latest, dict):
                # Calculate average sentiment over the period
                avg_sentiment = sum(s.get('normalized', 0) for s in sentiment[:7]) / min(len(sentiment), 7)
                return {
                    "latest_date": latest.get('date'),
                    "latest_score": latest.get('normalized'),
                    "avg_7day_score": avg_sentiment,
                    "data_points": len(sentiment)
                }
        
        # Handle dict format (legacy)
        elif isinstance(sentiment, dict):
            return {
                "overall_sentiment": sentiment.get('sentiment'),
                "sentiment_score": sentiment.get('sentiment_score'),
                "buzz": sentiment.get('buzz'),
                "sentiment_change": sentiment.get('sentiment_change')
            }
        
        return {}
    
    def _format_technical_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Format EODHD technical indicators"""
        if not indicators:
            return {}
        
        formatted = {}
        for indicator_name, data in indicators.items():
            if data and isinstance(data, list) and len(data) > 0:
                latest = data[0]  # Most recent data point
                if isinstance(latest, dict):
                    formatted[indicator_name] = {
                        "current_value": latest.get(indicator_name),
                        "date": latest.get('date'),
                        "data_available": True
                    }
            else:
                formatted[indicator_name] = {
                    "data_available": False
                }
        
        return formatted
    
    def _format_risk_metrics(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format risk metrics data"""
        if not risk_metrics:
            return {}
        
        return {
            "sharpe_ratio": risk_metrics.get('sharpe_ratio'),
            "sortino_ratio": risk_metrics.get('sortino_ratio'),
            "max_drawdown": risk_metrics.get('max_drawdown'),
            "volatility_annual": risk_metrics.get('volatility_annual'),
            "downside_deviation": risk_metrics.get('downside_deviation'),
            "beta": risk_metrics.get('beta'),
            "calculation_period": risk_metrics.get('calculation_period'),
            "data_points": risk_metrics.get('data_points')
        }
    
    def _assess_data_completeness(
        self, 
        opportunity_data: Dict[str, Any], 
        option_data: Dict[str, Any], 
        enhanced_data: Dict[str, Any]
    ) -> DataCompletenessMetrics:
        """
        Assess completeness of data package to determine analysis quality.
        
        Returns completeness metrics for filtering low-quality packages.
        """
        missing_fields = []
        
        # Fundamental data completeness
        fundamental_score = 0.0
        if enhanced_data.get('fundamentals'):
            fundamental_keys = ['company_info', 'financial_health', 'valuation_metrics']
            available = sum(1 for key in fundamental_keys if enhanced_data['fundamentals'].get(key))
            fundamental_score = (available / len(fundamental_keys)) * 100
        else:
            missing_fields.append('fundamentals')
        
        # Technical data completeness  
        technical_score = 0.0
        if opportunity_data.get('technical_indicators'):
            required_indicators = ['rsi', 'macd', 'sma_20', 'sma_50', 'bollinger_bands']
            available = sum(1 for ind in required_indicators 
                          if opportunity_data['technical_indicators'].get(ind) is not None)
            technical_score = (available / len(required_indicators)) * 100
        else:
            missing_fields.append('technical_indicators')
        
        # Options data completeness
        options_score = 0.0
        if option_data:
            required_fields = ['bid', 'ask', 'delta', 'implied_volatility', 'volume']
            available = sum(1 for field in required_fields if option_data.get(field) is not None)
            options_score = (available / len(required_fields)) * 100
        else:
            missing_fields.append('option_data')
        
        # Calendar data completeness
        calendar_score = 0.0
        if enhanced_data.get('earnings'):
            calendar_score = 100.0
        elif enhanced_data.get('economic_events'):
            calendar_score = 50.0
        else:
            missing_fields.append('calendar_events')
        
        # News data completeness
        news_score = 0.0
        if enhanced_data.get('news'):
            news_count = len(enhanced_data['news'])
            news_score = min(news_count / 10 * 100, 100.0)  # Up to 10 articles = 100%
        else:
            missing_fields.append('news')
        
        # Calculate overall score (weighted average)
        weights = {
            'fundamental': 0.25,
            'technical': 0.25,
            'options': 0.25,
            'calendar': 0.125,
            'news': 0.125
        }
        
        overall_score = (
            fundamental_score * weights['fundamental'] +
            technical_score * weights['technical'] +
            options_score * weights['options'] +
            calendar_score * weights['calendar'] +
            news_score * weights['news']
        )
        
        return DataCompletenessMetrics(
            fundamental_score=fundamental_score,
            technical_score=technical_score,
            options_score=options_score,
            calendar_score=calendar_score,
            news_score=news_score,
            overall_score=overall_score,
            missing_fields=missing_fields
        )
    
    # Utility methods
    
    def _clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and clean data for JSON serialization"""
        if not isinstance(data, dict):
            return data
        
        cleaned = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned_value = self._clean_dict(value)
                    if cleaned_value:  # Only add non-empty dicts
                        cleaned[key] = cleaned_value
                elif isinstance(value, (list, tuple)):
                    cleaned_list = [item for item in value if item is not None]
                    if cleaned_list:
                        cleaned[key] = cleaned_list
                elif isinstance(value, (Decimal, float)) and str(value).lower() in ['nan', 'inf', '-inf']:
                    continue  # Skip invalid numbers
                else:
                    cleaned[key] = value
        
        return cleaned
    
    def _calculate_mid_price(self, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        """Calculate mid price from bid/ask"""
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    def _calculate_spread_percent(self, bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
        """Calculate bid/ask spread as percentage"""
        if bid is not None and ask is not None and ask > 0:
            return ((ask - bid) / ask) * 100
        return None
    
    def _calculate_iv_hv_ratio(self, iv: Optional[float], hv: Optional[float]) -> Optional[float]:
        """Calculate IV/HV ratio"""
        if iv is not None and hv is not None and hv > 0:
            return iv / hv
        return None
    
    def _calculate_volume_oi_ratio(self, volume: Optional[int], oi: Optional[int]) -> Optional[float]:
        """Calculate volume to open interest ratio"""
        if volume is not None and oi is not None and oi > 0:
            return volume / oi
        return None
    
    def _calculate_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate news sentiment"""
        if not articles:
            return {"overall": "neutral", "positive_count": 0, "negative_count": 0, "neutral_count": 0}
        
        sentiments = []
        for article in articles:
            sentiment = article.get('sentiment')
            if sentiment:
                # Handle both string and dict sentiment formats
                if isinstance(sentiment, dict):
                    # Use polarity score to determine sentiment
                    polarity = sentiment.get('polarity', 0)
                    if polarity > 0.6:
                        sentiments.append('positive')
                    elif polarity < -0.6:
                        sentiments.append('negative')
                    else:
                        sentiments.append('neutral')
                elif isinstance(sentiment, str):
                    sentiments.append(sentiment)
        
        counts = {
            'positive': sum(1 for s in sentiments if s.lower() in ['positive', 'bullish']),
            'negative': sum(1 for s in sentiments if s.lower() in ['negative', 'bearish']),
            'neutral': sum(1 for s in sentiments if s.lower() == 'neutral')
        }
        
        if counts['positive'] > counts['negative']:
            overall = 'positive'
        elif counts['negative'] > counts['positive']:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            "overall": overall,
            "positive_count": counts['positive'],
            "negative_count": counts['negative'],
            "neutral_count": counts['neutral'],
            "total_analyzed": len(sentiments)
        }


def serialize_for_json(obj: Any) -> str:
    """
    Custom JSON serializer for complex data types.
    
    Handles datetime, date, Decimal, and other non-JSON-serializable types.
    """
    def default_serializer(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        elif isinstance(o, Decimal):
            return float(o)
        elif hasattr(o, '__dict__'):
            return asdict(o) if hasattr(o, '__dataclass_fields__') else vars(o)
        else:
            return str(o)
    
    return json.dumps(obj, default=default_serializer, indent=2)


def estimate_token_count(package: Dict[str, Any]) -> int:
    """
    Estimate token count for cost management.
    
    Rough estimation: 1 token ≈ 4 characters for English text.
    """
    json_str = serialize_for_json(package)
    return len(json_str) // 4


if __name__ == "__main__":
    # Example usage and testing
    packager = DataPackager(min_completeness_threshold=60.0)
    
    # Sample data for testing
    sample_opportunity = {
        "symbol": "AAPL",
        "current_price": 175.50,
        "overall_score": 82.5,
        "technical_indicators": {
            "rsi": 58.2,
            "macd": 1.25,
            "sma_20": 172.30,
            "sma_50": 168.50,
            "bollinger_bands": {"upper": 178.50, "lower": 166.10}
        },
        "momentum_analysis": {
            "21_day_return": 0.045,
            "momentum_score": 75.0
        }
    }
    
    sample_option = {
        "option_symbol": "AAPL240315C00170000",
        "strike": 170.0,
        "bid": 8.50,
        "ask": 8.75,
        "delta": 0.65,
        "implied_volatility": 0.28
    }
    
    sample_enhanced = {
        "fundamentals": {
            "company_info": {"name": "Apple Inc.", "sector": "Technology"},
            "financial_health": {"eps_ttm": 5.95, "profit_margin": 0.253}
        },
        "news": [{"title": "Apple announces...", "date": "2024-01-15"}]
    }
    
    package = packager.create_analysis_package(
        sample_opportunity, sample_option, sample_enhanced
    )
    
    print(f"Package created with {estimate_token_count(package)} estimated tokens")
    print(f"Data completeness: {package['data_quality']['overall_score']:.1f}%")
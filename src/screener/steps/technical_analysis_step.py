"""Technical analysis step executor for comprehensive quantitative scoring.

This module implements the TechnicalAnalysisExecutor that integrates all existing
analytics modules to calculate local scores and select best options for each symbol.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Union
import traceback

from src.screener.workflow_engine import WorkflowStepExecutor
from src.models import (
    WorkflowStep,
    WorkflowStepStatus,
    WorkflowStepResult,
    WorkflowExecutionContext,
    OptionContract,
    OptionType
)

# Import all analytics modules
from src.analytics.technical_indicators import TechnicalIndicators
from src.analytics.momentum_analysis import MomentumAnalysis, PercentileCalculator
from src.analytics.squeeze_detector import SqueezeDetector, VolatilityRegimeDetector
from src.analytics.scoring_models import QuantitativeScorer, RiskAssessment
from src.analytics.enhanced_technical_analysis import EnhancedTechnicalAnalysis
from src.analytics.local_rating_system import LocalRatingSystem, RatingResult

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TechnicalAnalysisExecutor(WorkflowStepExecutor):
    """
    Technical Analysis executor that uses existing analytics modules to:
    1. Calculate technical indicators and momentum analysis
    2. Detect TTM squeeze conditions
    3. Score options opportunities 
    4. Generate comprehensive quantitative scores (0-100) for each symbol
    5. Select best call options using OptionsSelector
    """
    
    def __init__(self, step: WorkflowStep):
        super().__init__(step)
        self.quantitative_scorer = QuantitativeScorer()
        self.local_rating_system = LocalRatingSystem()
        
    async def execute_step(
        self, 
        input_data: Any, 
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute technical analysis for all symbols from DataEnrichmentExecutor.
        
        Args:
            input_data: Output from DataEnrichmentExecutor with enriched data
            context: Workflow execution context
            
        Returns:
            Dict containing analyzed opportunities with scores and best options
        """
        # Input validation
        if not isinstance(input_data, dict) or 'enriched_data' not in input_data:
            raise ValueError("Input must be enriched data from DataEnrichmentExecutor")
            
        enriched_data = input_data['enriched_data']
        if not enriched_data:
            logger.warning("No enriched data provided for technical analysis")
            return {
                'analyzed_opportunities': {},
                'symbols_analyzed': 0,
                'symbols_failed': 0,
                'analysis_summary': {}
            }
        
        logger.info(f"Starting technical analysis for {len(enriched_data)} symbols")
        
        analyzed_opportunities = {}
        symbols_analyzed = 0
        symbols_failed = 0
        analysis_summary = {
            'total_symbols': len(enriched_data),
            'high_score_count': 0,  # Score >= 70
            'medium_score_count': 0,  # Score 50-69
            'low_score_count': 0,  # Score < 50
            'options_available_count': 0,
            'squeeze_detected_count': 0,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Process symbols with controlled concurrency
        max_concurrent = min(context.config.max_concurrent_stocks, 5)  # Conservative for analysis
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_symbol_with_semaphore(symbol: str, symbol_data: Dict) -> tuple[str, Optional[Dict]]:
            async with semaphore:
                return symbol, await self._analyze_single_symbol(symbol, symbol_data)
        
        # Process all symbols concurrently
        tasks = [
            analyze_symbol_with_semaphore(symbol, symbol_data) 
            for symbol, symbol_data in enriched_data.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Symbol analysis failed: {result}")
                symbols_failed += 1
                continue
                
            symbol, analysis_result = result
            if analysis_result is not None:
                analyzed_opportunities[symbol] = analysis_result
                symbols_analyzed += 1
                
                # Update summary statistics
                score = analysis_result.get('composite_score', 0)
                if score >= 70:
                    analysis_summary['high_score_count'] += 1
                elif score >= 50:
                    analysis_summary['medium_score_count'] += 1
                else:
                    analysis_summary['low_score_count'] += 1
                
                # Options counting removed for stock-only workflow
                
                if analysis_result.get('squeeze_data', {}).get('is_squeeze'):
                    analysis_summary['squeeze_detected_count'] += 1
                    
                logger.debug(f"Successfully analyzed {symbol} with score {score:.1f}")
            else:
                symbols_failed += 1
                logger.warning(f"Failed to analyze {symbol}")
        
        # Update context
        context.completed_symbols += symbols_analyzed
        context.failed_symbols += symbols_failed
        
        logger.info(f"Technical analysis completed: {symbols_analyzed} successful, {symbols_failed} failed")
        logger.info(f"Score distribution - High: {analysis_summary['high_score_count']}, "
                   f"Medium: {analysis_summary['medium_score_count']}, "
                   f"Low: {analysis_summary['low_score_count']}")
        
        return {
            'analyzed_opportunities': analyzed_opportunities,
            'symbols_analyzed': symbols_analyzed,
            'symbols_failed': symbols_failed,
            'total_symbols': len(enriched_data),
            'analysis_summary': analysis_summary
        }
    
    async def _analyze_single_symbol(self, symbol: str, symbol_data: Dict) -> Optional[Dict]:
        """
        Perform comprehensive technical analysis for a single symbol.
        
        Args:
            symbol: Stock symbol
            symbol_data: Enriched data for the symbol
            
        Returns:
            Dictionary with analysis results and scores
        """
        try:
            # Initialize result structure
            analysis_result = {
                'symbol': symbol,
                'composite_score': 0.0,
                'score_breakdown': {
                    'technical': 0.0,
                    'momentum': 0.0,
                    'squeeze': 0.0,
                    'quality': 0.0
                },
                'risk_metrics': {},  # Stock risk assessment
                'technical_indicators': {},
                'momentum_data': {},
                'squeeze_data': {},
                'options_analysis': {},
                # Options analysis removed for stock-only workflow
                'data_quality': {},
                'analysis_timestamp': datetime.now().isoformat(),
                'warnings': [],
                # IMPORTANT: Pass through enhanced data for Claude
                'enhanced_data': {
                    'quote': symbol_data.get('quote'),
                    'fundamentals': symbol_data.get('fundamentals'),
                    'news': symbol_data.get('news'),
                    'earnings': symbol_data.get('earnings'),
                    'sentiment': symbol_data.get('sentiment'),
                    'technicals': symbol_data.get('technicals'),
                    'economic_events': symbol_data.get('economic_events'),
                    # Options data removed for stock-only workflow
                }
            }
            
            # Extract and validate historical data (check both field names for compatibility)
            historical_data = symbol_data.get('historical_prices') or symbol_data.get('historical')
            if not historical_data or not isinstance(historical_data, list):
                analysis_result['warnings'].append("No historical price data available")
                # Use simplified scoring based on available data
                analysis_result['composite_score'] = self._calculate_simplified_score(symbol_data)
                analysis_result['score_breakdown'] = {
                    'technical': 0,  # No technical due to missing historical data
                    'momentum': 0,   # No momentum due to missing historical data
                    'squeeze': 0,    # No squeeze due to missing historical data
                    'quality': analysis_result['composite_score']  # Use full score for quality
                }
                return analysis_result
            
            # Convert to DataFrame
            ohlcv_df = self._prepare_ohlcv_data(historical_data)
            if ohlcv_df.empty or len(ohlcv_df) < 20:
                analysis_result['warnings'].append("Insufficient historical data for analysis")
                # Use simplified scoring based on available data
                analysis_result['composite_score'] = self._calculate_simplified_score(symbol_data)
                analysis_result['score_breakdown'] = {
                    'technical': 0,  # No technical due to missing historical data
                    'momentum': 0,   # No momentum due to missing historical data
                    'squeeze': 0,    # No squeeze due to missing historical data
                    'quality': analysis_result['composite_score']  # Use full score for quality
                }
                return analysis_result
            
            # Get current stock price
            quote = symbol_data.get('quote')
            current_price = None
            if quote and isinstance(quote, dict):
                current_price = quote.get('close') or quote.get('price')
            if current_price is None and not ohlcv_df.empty:
                current_price = ohlcv_df['close'].iloc[-1]
            
            # Store current price in result
            analysis_result['current_price'] = current_price
            
            # === USE LOCAL RATING SYSTEM ===
            # Get options chain
            options_chain = symbol_data.get('options_chain', [])
            
            # Apply comprehensive local rating
            rating_result = self.local_rating_system.rate_opportunity(
                symbol=symbol,
                ohlcv_df=ohlcv_df,
                enhanced_data=symbol_data,
                options_chain=options_chain,
                current_price=current_price,
                is_earnings_strategy=False
            )
            
            # Check if eligibility failed
            if not rating_result.eligibility.passed:
                logger.info(f"Symbol {symbol} failed eligibility: {rating_result.eligibility.reason}")
                analysis_result['warnings'].append(f"Eligibility failed: {rating_result.eligibility.reason}")
                analysis_result['composite_score'] = 0
                analysis_result['red_flags'] = rating_result.red_flags
                # Still return result for transparency
                return analysis_result
            
            # Store rating results
            analysis_result['local_rating'] = {
                'pre_score': rating_result.pre_score,
                'final_score': rating_result.final_score,
                'sub_scores': rating_result.sub_scores,
                'red_flags': rating_result.red_flags,
                'penalties': rating_result.penalties_applied,
                'features': rating_result.feature_values
            }
            
            # Use local rating score as composite score
            analysis_result['composite_score'] = rating_result.final_score
            
            # Update score breakdown with local rating sub-scores (stock-focused)
            if rating_result.sub_scores:
                analysis_result['score_breakdown'] = {
                    'trend_momentum': rating_result.sub_scores.get('trend_momentum', 0),
                    'squeeze_breakout': rating_result.sub_scores.get('squeeze_breakout', 0),
                    'fundamentals': rating_result.sub_scores.get('fundamentals', 0),
                    'market_quality': rating_result.sub_scores.get('market_quality', 0)
                }
            
            # Options analysis removed - focusing on stock metrics only
            logger.info(f"Stock analysis completed for {symbol} with score {rating_result.final_score:.1f}")
            
            # Add red flags to warnings
            if rating_result.red_flags:
                analysis_result['warnings'].extend(rating_result.red_flags)
            
            # 1. Calculate traditional technical indicators
            try:
                technical_indicators = TechnicalIndicators.calculate_all_indicators(ohlcv_df)
                analysis_result['technical_indicators'] = technical_indicators
            except Exception as e:
                logger.warning(f"Technical indicators calculation failed for {symbol}: {e}")
                analysis_result['warnings'].append(f"Technical indicators error: {str(e)}")
                technical_indicators = {}
            
            # 2. Calculate momentum analysis
            try:
                momentum_data = MomentumAnalysis.calculate_all_momentum_indicators(ohlcv_df)
                analysis_result['momentum_data'] = momentum_data
            except Exception as e:
                logger.warning(f"Momentum analysis failed for {symbol}: {e}")
                analysis_result['warnings'].append(f"Momentum analysis error: {str(e)}")
                momentum_data = {}
            
            # 3. Detect TTM squeeze
            try:
                squeeze_data = SqueezeDetector.comprehensive_squeeze_analysis(ohlcv_df)
                analysis_result['squeeze_data'] = squeeze_data
            except Exception as e:
                logger.warning(f"Squeeze detection failed for {symbol}: {e}")
                analysis_result['warnings'].append(f"Squeeze detection error: {str(e)}")
                squeeze_data = {}
            
            # 3a. Calculate ENHANCED technical analysis (new requirements)
            try:
                # Calculate trend & momentum indicators
                trend_momentum = EnhancedTechnicalAnalysis.calculate_trend_momentum(
                    ohlcv_df, 
                    current_price,
                    None  # No sector ETF data for now
                )
                analysis_result['trend_momentum'] = trend_momentum
                
                # Calculate squeeze & breakout indicators
                squeeze_breakout = EnhancedTechnicalAnalysis.calculate_squeeze_breakout(ohlcv_df)
                analysis_result['squeeze_breakout'] = squeeze_breakout
                
                # Calculate liquidity & risk metrics
                fundamentals = symbol_data.get('fundamentals', {})
                shares_float = fundamentals.get('SharesStats', {}).get('SharesFloat') if isinstance(fundamentals, dict) else None
                shares_short = fundamentals.get('SharesStats', {}).get('ShortPercent') if isinstance(fundamentals, dict) else None
                news_list = symbol_data.get('news', [])
                news_count = len(news_list) if isinstance(news_list, list) else None
                
                liquidity_risk = EnhancedTechnicalAnalysis.calculate_liquidity_risk(
                    ohlcv_df,
                    shares_float=shares_float,
                    shares_short=shares_short,
                    news_count=news_count
                )
                analysis_result['liquidity_risk'] = liquidity_risk
                
                logger.info(f"Enhanced technical analysis calculated for {symbol}")
            except Exception as e:
                logger.warning(f"Enhanced technical analysis failed for {symbol}: {e}")
                analysis_result['warnings'].append(f"Enhanced technical analysis error: {str(e)}")
            
            # 4. Calculate Risk Metrics for Stock Position Sizing
            if current_price and not ohlcv_df.empty:
                try:
                    # Calculate risk metrics using RiskAssessment
                    risk_metrics = RiskAssessment.calculate_risk_metrics(
                        ohlcv_df, 
                        position_size=10000.0  # Default $10K position
                    )
                    analysis_result['risk_metrics'] = risk_metrics
                    
                    # Add position sizing recommendations
                    suggested_position = RiskAssessment.suggest_position_size(
                        account_balance=100000.0,  # Assume $100K account
                        risk_per_trade=0.02,       # 2% risk per trade
                        risk_metrics=risk_metrics
                    )
                    analysis_result['position_sizing'] = suggested_position
                    
                    logger.info(f"Risk metrics calculated for {symbol}: "
                              f"ATR={risk_metrics.get('atr_percent', 0):.2f}%, "
                              f"Stop={risk_metrics.get('suggested_stop_loss', 0):.2f}")
                        
                except Exception as e:
                    logger.warning(f"Risk metrics calculation failed for {symbol}: {e}")
                    analysis_result['warnings'].append(f"Risk metrics error: {str(e)}")
            
            # 5. Calculate comprehensive scores using QuantitativeScorer (stock-only)
            try:
                scoring_result = self.quantitative_scorer.calculate_comprehensive_score(
                    ohlcv_data=ohlcv_df,
                    options_chain=None,  # No options for stock-only workflow
                    current_stock_price=current_price,
                    iv_percentile=None   # No IV analysis needed
                )
                
                # Extract scores (stock-focused components only)
                if 'component_scores' in scoring_result:
                    for component, score_data in scoring_result['component_scores'].items():
                        if component in ['technical', 'momentum', 'squeeze', 'quality']:  # Only stock components
                            score_key = f'{component}_score'
                            if score_key in score_data:
                                analysis_result['score_breakdown'][component] = score_data[score_key]
                
                analysis_result['composite_score'] = scoring_result.get('overall_score', 0.0)
                analysis_result['data_quality'] = scoring_result.get('data_quality', {})
                
                # Add any warnings from scoring
                if 'warnings' in scoring_result:
                    analysis_result['warnings'].extend(scoring_result['warnings'])
                    
            except Exception as e:
                logger.warning(f"Comprehensive scoring failed for {symbol}: {e}")
                analysis_result['warnings'].append(f"Scoring error: {str(e)}")
                # Fallback scoring (stock-only)
                analysis_result['composite_score'] = self._calculate_fallback_score(
                    technical_indicators, momentum_data, squeeze_data, False  # No options
                )
            
            # 6. Add ALL enhanced data from symbol_data
            # This is critical - we need to pass through all the enhanced data collected
            enhanced_fields = [
                'quote', 'fundamentals', 'news', 'earnings', 
                'sentiment', 'technicals', 'market_context',  # Changed from economic_events
                'live_price', 'historical_prices', 'technical_indicators',
                # Options chain removed for stock-only workflow
                'risk_metrics'   # Add risk metrics (Sharpe, Sortino, etc.)
            ]
            
            analysis_result['enhanced_data'] = {}
            for field in enhanced_fields:
                if field in symbol_data and symbol_data[field]:
                    analysis_result['enhanced_data'][field] = symbol_data[field]
            
            # Add the NEW enhanced technical analysis fields
            if 'trend_momentum' in analysis_result:
                analysis_result['enhanced_data']['trend_momentum'] = analysis_result['trend_momentum']
            if 'squeeze_breakout' in analysis_result:
                analysis_result['enhanced_data']['squeeze_breakout'] = analysis_result['squeeze_breakout']
            if 'liquidity_risk' in analysis_result:
                analysis_result['enhanced_data']['liquidity_risk'] = analysis_result['liquidity_risk']
            # Add risk metrics to enhanced data
            if 'risk_metrics' in analysis_result:
                analysis_result['enhanced_data']['risk_metrics'] = analysis_result['risk_metrics']
            if 'position_sizing' in analysis_result:
                analysis_result['enhanced_data']['position_sizing'] = analysis_result['position_sizing']
            
            # Add local rating results to enhanced data
            if 'local_rating' in analysis_result:
                analysis_result['enhanced_data']['local_rating'] = analysis_result['local_rating']
            
            # Log what enhanced data we're passing through
            if analysis_result['enhanced_data']:
                available_fields = [k for k, v in analysis_result['enhanced_data'].items() if v]
                logger.info(f"Passing enhanced data for {symbol} with fields: {available_fields}")
            
            # 7. Apply weighted scoring for stock-only workflow
            # technical (35%), momentum (35%), squeeze (20%), quality (10%)
            weighted_score = (
                analysis_result['score_breakdown']['technical'] * 0.35 +
                analysis_result['score_breakdown']['momentum'] * 0.35 +
                analysis_result['score_breakdown']['squeeze'] * 0.20 +
                analysis_result['score_breakdown']['quality'] * 0.10
            )
            analysis_result['composite_score'] = min(100.0, max(0.0, weighted_score))
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze symbol {symbol}: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return None
    
    def _prepare_ohlcv_data(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Convert historical data to OHLCV DataFrame."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Remove any rows with NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare OHLCV data: {e}")
            return pd.DataFrame()
    
    def _convert_options_data(self, options_data: List[Dict]) -> List[OptionContract]:
        """Convert options data to OptionContract objects."""
        option_contracts = []
        
        for option_dict in options_data:
            try:
                # Extract option type
                option_type = OptionType.CALL if option_dict.get('option_type') == 'call' else OptionType.PUT
                
                # Extract underlying symbol
                underlying_symbol = option_dict.get('symbol', option_dict.get('underlying_symbol', ''))
                if not underlying_symbol:
                    continue
                
                # Create option symbol if not provided
                option_symbol = option_dict.get('option_symbol', f"{underlying_symbol}_{option_dict.get('expiration', '')}_{option_dict.get('strike', '')}_{option_type.value}")
                
                # Handle expiration date (could be string or already a date object)
                expiration_raw = option_dict.get('expiration')
                expiration_date = None
                if expiration_raw:
                    if isinstance(expiration_raw, str):
                        expiration_date = datetime.strptime(expiration_raw, '%Y-%m-%d').date()
                    elif isinstance(expiration_raw, date):
                        expiration_date = expiration_raw
                    elif isinstance(expiration_raw, datetime):
                        expiration_date = expiration_raw.date()
                
                # Create OptionContract
                contract = OptionContract(
                    option_symbol=option_symbol,
                    underlying_symbol=underlying_symbol,
                    option_type=option_type,
                    strike=float(option_dict.get('strike', 0)),
                    expiration=expiration_date,
                    bid=float(option_dict.get('bid', 0)) if option_dict.get('bid') else None,
                    ask=float(option_dict.get('ask', 0)) if option_dict.get('ask') else None,
                    last=float(option_dict.get('last', 0)) if option_dict.get('last') else None,
                    volume=int(option_dict.get('volume', 0)) if option_dict.get('volume') else None,
                    open_interest=int(option_dict.get('open_interest', 0)) if option_dict.get('open_interest') else None,
                    implied_volatility=float(option_dict.get('implied_volatility', 0)) if option_dict.get('implied_volatility') else None,
                    delta=float(option_dict.get('delta', 0)) if option_dict.get('delta') else None,
                    gamma=float(option_dict.get('gamma', 0)) if option_dict.get('gamma') else None,
                    theta=float(option_dict.get('theta', 0)) if option_dict.get('theta') else None,
                    vega=float(option_dict.get('vega', 0)) if option_dict.get('vega') else None
                )
                
                # Calculate days to expiration
                if contract.expiration:
                    contract.days_to_expiration = (contract.expiration - date.today()).days
                
                option_contracts.append(contract)
                
            except Exception as e:
                logger.debug(f"Failed to convert option data: {e}")
                continue
        
        return option_contracts
    
    def _calculate_iv_percentile(self, option_contracts: List[OptionContract], symbol_data: Dict) -> Optional[float]:
        """Calculate IV percentile using historical data."""
        try:
            if not option_contracts:
                return None
            
            # Get current average IV
            ivs = [opt.implied_volatility for opt in option_contracts if opt.implied_volatility is not None]
            if not ivs:
                return None
            
            current_iv = np.mean(ivs)
            
            # Try to get historical IV from technical indicators or calculate from price data
            historical_data = symbol_data.get('historical_prices') or symbol_data.get('historical', [])
            if historical_data and len(historical_data) >= 252:  # Need 1 year of data
                # Calculate historical volatility from price data
                df = self._prepare_ohlcv_data(historical_data)
                if not df.empty:
                    returns = df['close'].pct_change().dropna()
                    historical_vols = []
                    
                    # Rolling 30-day historical volatility over past year
                    for i in range(30, len(returns)):
                        vol_period = returns.iloc[i-30:i]
                        historical_vol = vol_period.std() * np.sqrt(252)  # Annualized
                        historical_vols.append(historical_vol)
                    
                    if historical_vols:
                        return PercentileCalculator.calculate_percentile(current_iv, historical_vols)
            
            # Fallback: Use simple comparison with 6-month average
            if len(ivs) >= 10:  # If we have enough current IV data points
                avg_iv = np.mean(ivs)
                std_iv = np.std(ivs)
                if std_iv > 0:
                    # Assume normal distribution for rough percentile
                    z_score = (current_iv - avg_iv) / std_iv
                    percentile = 50 + (z_score * 15)  # Rough approximation
                    return max(0, min(100, percentile))
            
            return None
            
        except Exception as e:
            logger.debug(f"IV percentile calculation failed: {e}")
            return None
    
    def _calculate_fallback_score(self, technical_indicators: Dict, momentum_data: Dict, 
                                squeeze_data: Dict, has_options: bool = False) -> float:
        """Calculate a simple fallback score when comprehensive scoring fails."""
        score = 50.0  # Start with neutral score
        
        # Technical component (simple)
        rsi = technical_indicators.get('rsi_14')
        if rsi is not None and 30 <= rsi <= 70:
            score += 10
        
        # Momentum component
        momentum_21d = momentum_data.get('momentum_21d')
        if momentum_21d is not None and momentum_21d > 0:
            score += 15
        
        # Squeeze component
        if squeeze_data.get('is_squeeze'):
            score += 10
        
        # Quality component (replaces options)
        score += 5  # Base quality score
        
        return min(100.0, max(0.0, score))
    
    def _calculate_simplified_score(self, symbol_data: Dict) -> float:
        """
        Calculate a simplified score when historical price data is unavailable.
        Uses only available data like options, quotes, and fundamentals.
        
        Args:
            symbol_data: Enriched symbol data dictionary
            
        Returns:
            Float score between 0-100
        """
        score = 0.0
        
        # 1. Options liquidity scoring (up to 40 points)
        options_data = symbol_data.get('options_chain')
        if options_data and isinstance(options_data, list) and len(options_data) > 0:
            # Check for active options with good volume
            total_volume = sum(opt.get('volume', 0) for opt in options_data[:10])
            total_oi = sum(opt.get('openInterest', 0) for opt in options_data[:10])
            
            if total_volume > 1000:
                score += 20  # Good volume
            elif total_volume > 100:
                score += 10  # Moderate volume
            
            if total_oi > 500:
                score += 20  # Good open interest
            elif total_oi > 50:
                score += 10  # Moderate open interest
        
        # 2. Quote data quality (up to 20 points)
        quote_data = symbol_data.get('quote', {})
        if quote_data:
            # Check bid-ask spread
            bid = quote_data.get('bid', 0)
            ask = quote_data.get('ask', 0)
            price = quote_data.get('price', 0)
            
            if bid > 0 and ask > 0 and price > 0:
                spread_pct = ((ask - bid) / price) * 100 if price > 0 else 100
                if spread_pct < 0.5:
                    score += 10  # Tight spread
                elif spread_pct < 2.0:
                    score += 5   # Acceptable spread
            
            # Check volume
            volume = quote_data.get('volume', 0)
            if volume > 1000000:
                score += 10  # High volume
            elif volume > 100000:
                score += 5   # Moderate volume
        
        # 3. Fundamentals scoring (up to 20 points)
        fundamentals = symbol_data.get('fundamentals', {})
        if fundamentals:
            # P/E ratio check
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio and 5 < pe_ratio < 30:
                score += 10  # Reasonable P/E
            
            # Market cap check
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap > 10_000_000_000:  # > $10B
                score += 10  # Large cap
            elif market_cap > 2_000_000_000:  # > $2B
                score += 5   # Mid cap
        
        # 4. Any available technical indicators (up to 20 points)
        technicals = symbol_data.get('technical_indicators', {})
        if technicals:
            # RSI check
            rsi = technicals.get('rsi_14')
            if rsi and 30 <= rsi <= 70:
                score += 10  # Not overbought/oversold
            
            # Check for any momentum indicators
            if technicals.get('momentum_5d') or technicals.get('momentum_21d'):
                score += 10  # Has momentum data
        
        # Ensure score is within bounds
        return min(100.0, max(0.0, score))
    
    async def validate_input(self, input_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate input data from DataEnrichmentExecutor."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'enriched_data' not in input_data:
            raise ValueError("Input must contain 'enriched_data' key")
        
        enriched_data = input_data['enriched_data']
        if not isinstance(enriched_data, dict):
            raise ValueError("Enriched data must be a dictionary")
    
    async def validate_output(self, output_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate output data from technical analysis."""
        if not isinstance(output_data, dict):
            raise ValueError("Output data must be a dictionary")
        
        required_keys = ['analyzed_opportunities', 'symbols_analyzed', 'symbols_failed', 'total_symbols']
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")
        
        # Validate analyzed opportunities structure
        opportunities = output_data['analyzed_opportunities']
        if not isinstance(opportunities, dict):
            raise ValueError("Analyzed opportunities must be a dictionary")
        
        # Validate that we have some results or proper failure handling
        if output_data['symbols_analyzed'] == 0 and output_data['total_symbols'] > 0:
            logger.warning("No symbols were successfully analyzed")
    
    def get_records_processed(self, output_data: Any) -> Optional[int]:
        """Get number of records processed by this step."""
        if isinstance(output_data, dict):
            return output_data.get('symbols_analyzed', 0)
        return 0
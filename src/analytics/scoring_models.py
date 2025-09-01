"""Quantitative scoring models and comprehensive analysis algorithms.

This module provides unified scoring systems that combine technical indicators,
momentum analysis, squeeze detection, and options metrics into comprehensive
opportunity scores for stock screening.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, date, timedelta
import warnings

from .technical_indicators import TechnicalIndicators
from .momentum_analysis import MomentumAnalysis, PercentileCalculator
from .squeeze_detector import SqueezeDetector, VolatilityRegimeDetector
from .options_selector import OptionsSelector
from .greeks_calculator import GreeksCalculator
from src.models.base_models import OptionContract, OptionType

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class QuantitativeScorer:
    """Comprehensive quantitative scoring system."""
    
    def __init__(self):
        """Initialize scorer with sub-components."""
        self.options_selector = OptionsSelector()
        self.greeks_calculator = GreeksCalculator()
        
        # Scoring weights (must sum to 100) - Stock-focused weights
        self.weights = {
            'technical': 35.0,      # Technical indicators
            'momentum': 35.0,       # Momentum and trend
            'squeeze': 20.0,        # Squeeze and volatility
            'quality': 10.0         # Data quality and filters
        }
    
    def calculate_technical_score(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate technical analysis score (0-100).
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary with technical score and components
        """
        score = 0.0
        components = {}
        
        # RSI Score (20 points) - prefer 30-70 range, avoid extremes
        rsi = indicators.get('rsi_14')
        if rsi is not None and not np.isnan(rsi):
            if 40 <= rsi <= 60:  # Neutral zone
                rsi_score = 20
            elif 30 <= rsi <= 70:  # Good zone
                rsi_score = 15
            elif 20 <= rsi <= 80:  # OK zone
                rsi_score = 10
            else:  # Extreme zones
                rsi_score = 5
        else:
            rsi_score = 10  # Neutral for missing data
        
        components['rsi_score'] = rsi_score
        score += rsi_score
        
        # ADX Score (20 points) - trend strength
        adx = indicators.get('adx_14')
        if adx is not None and not np.isnan(adx):
            if adx >= 30:  # Strong trend
                adx_score = 20
            elif adx >= 20:  # Moderate trend
                adx_score = 15
            elif adx >= 10:  # Weak trend
                adx_score = 10
            else:  # No trend
                adx_score = 5
        else:
            adx_score = 10
        
        components['adx_score'] = adx_score
        score += adx_score
        
        # Moving Average Score (20 points)
        pct_above_50 = indicators.get('pct_above_sma_50', 0)
        pct_above_200 = indicators.get('pct_above_sma_200', 0)
        
        ma_score = 0
        if not np.isnan(pct_above_50):
            if pct_above_50 > 5:  # Well above 50-day MA
                ma_score += 10
            elif pct_above_50 > 0:  # Above 50-day MA
                ma_score += 7
            elif pct_above_50 > -5:  # Close to 50-day MA
                ma_score += 4
        
        if not np.isnan(pct_above_200):
            if pct_above_200 > 10:  # Well above 200-day MA
                ma_score += 10
            elif pct_above_200 > 0:  # Above 200-day MA
                ma_score += 7
            elif pct_above_200 > -10:  # Close to 200-day MA
                ma_score += 4
        
        components['ma_score'] = ma_score
        score += ma_score
        
        # Bollinger Bands Score (20 points) - position within bands
        bb_upper = indicators.get('bollinger_upper')
        bb_lower = indicators.get('bollinger_lower')
        
        if bb_upper is not None and bb_lower is not None:
            # Would need current price to calculate position
            # For now, award neutral score
            bb_score = 10
        else:
            bb_score = 10
        
        components['bb_score'] = bb_score
        score += bb_score
        
        # Volume Score (20 points)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if not np.isnan(volume_ratio):
            if volume_ratio >= 2.0:  # High volume
                volume_score = 20
            elif volume_ratio >= 1.5:  # Above average
                volume_score = 15
            elif volume_ratio >= 1.0:  # Average
                volume_score = 10
            else:  # Below average
                volume_score = 5
        else:
            volume_score = 10
        
        components['volume_score'] = volume_score
        score += volume_score
        
        return {
            'technical_score': min(100, score),
            'components': components
        }
    
    def calculate_momentum_score(self, momentum_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate momentum analysis score (0-100).
        
        Args:
            momentum_data: Dictionary of momentum indicators
            
        Returns:
            Dictionary with momentum score and components
        """
        score = 0.0
        components = {}
        
        # Short-term momentum (25 points)
        momentum_21d = momentum_data.get('momentum_21d')
        if momentum_21d is not None and not np.isnan(momentum_21d):
            if momentum_21d > 10:  # Strong positive momentum
                short_momentum_score = 25
            elif momentum_21d > 5:  # Moderate positive
                short_momentum_score = 20
            elif momentum_21d > 0:  # Weak positive
                short_momentum_score = 15
            elif momentum_21d > -5:  # Slightly negative
                short_momentum_score = 10
            else:  # Negative
                short_momentum_score = 5
        else:
            short_momentum_score = 12
        
        components['short_momentum'] = short_momentum_score
        score += short_momentum_score
        
        # Medium-term momentum (25 points)
        momentum_63d = momentum_data.get('momentum_63d')
        if momentum_63d is not None and not np.isnan(momentum_63d):
            if momentum_63d > 20:  # Strong positive
                med_momentum_score = 25
            elif momentum_63d > 10:  # Moderate positive
                med_momentum_score = 20
            elif momentum_63d > 0:  # Weak positive
                med_momentum_score = 15
            elif momentum_63d > -10:  # Slightly negative
                med_momentum_score = 10
            else:  # Negative
                med_momentum_score = 5
        else:
            med_momentum_score = 12
        
        components['med_momentum'] = med_momentum_score
        score += med_momentum_score
        
        # Momentum acceleration (20 points)
        acceleration = momentum_data.get('momentum_acceleration')
        if acceleration is not None and not np.isnan(acceleration):
            if acceleration > 5:  # Accelerating
                accel_score = 20
            elif acceleration > 0:  # Slightly accelerating
                accel_score = 15
            elif acceleration > -5:  # Stable
                accel_score = 10
            else:  # Decelerating
                accel_score = 5
        else:
            accel_score = 10
        
        components['acceleration'] = accel_score
        score += accel_score
        
        # Trend strength (15 points)
        trend_strength = momentum_data.get('trend_strength_20d')
        if trend_strength is not None and not np.isnan(trend_strength):
            if trend_strength > 0.8:  # Very strong trend
                trend_score = 15
            elif trend_strength > 0.6:  # Strong trend
                trend_score = 12
            elif trend_strength > 0.4:  # Moderate trend
                trend_score = 9
            elif trend_strength > 0.2:  # Weak trend
                trend_score = 6
            else:  # No trend
                trend_score = 3
        else:
            trend_score = 7
        
        components['trend_strength'] = trend_score
        score += trend_score
        
        # Relative strength (15 points)
        rel_strength = momentum_data.get('relative_strength_63d')
        if rel_strength is not None and not np.isnan(rel_strength):
            if rel_strength > 10:  # Strong outperformance
                rel_score = 15
            elif rel_strength > 5:  # Moderate outperformance
                rel_score = 12
            elif rel_strength > 0:  # Slight outperformance
                rel_score = 9
            elif rel_strength > -5:  # Slight underperformance
                rel_score = 6
            else:  # Significant underperformance
                rel_score = 3
        else:
            rel_score = 7
        
        components['relative_strength'] = rel_score
        score += rel_score
        
        return {
            'momentum_score': min(100, score),
            'components': components
        }
    
    def calculate_squeeze_score(self, squeeze_data: Dict[str, Union[float, bool, int]]) -> Dict[str, float]:
        """Calculate squeeze detection score (0-100).
        
        Args:
            squeeze_data: Dictionary of squeeze analysis results
            
        Returns:
            Dictionary with squeeze score and components
        """
        score = 0.0
        components = {}
        
        # Basic squeeze condition (40 points)
        is_squeeze = squeeze_data.get('is_squeeze', False)
        if is_squeeze:
            squeeze_score = 40
        else:
            squeeze_score = 0
        
        components['squeeze_condition'] = squeeze_score
        score += squeeze_score
        
        # Squeeze intensity (30 points)
        intensity = squeeze_data.get('squeeze_intensity')
        if intensity is not None and not np.isnan(intensity):
            intensity_score = min(30, intensity * 0.3)  # Scale to 30 points
        else:
            intensity_score = 0
        
        components['squeeze_intensity'] = intensity_score
        score += intensity_score
        
        # Volatility compression (15 points)
        compression = squeeze_data.get('volatility_compression')
        if compression is not None and not np.isnan(compression):
            compression_score = min(15, compression * 0.15)  # Scale to 15 points
        else:
            compression_score = 0
        
        components['volatility_compression'] = compression_score
        score += compression_score
        
        # Multi-timeframe confirmation (15 points)
        mtf_count = squeeze_data.get('multi_timeframe_squeeze_count', 0)
        mtf_score = min(15, mtf_count * 5)  # Up to 15 points for 3+ timeframes
        
        components['multi_timeframe'] = mtf_score
        score += mtf_score
        
        return {
            'squeeze_score': min(100, score),
            'components': components
        }
    
    def calculate_options_score(self, best_calls: List[Dict], best_puts: List[Dict],
                              iv_percentile: float = None) -> Dict[str, float]:
        """Calculate options opportunity score (0-100).
        
        Args:
            best_calls: List of best call options
            best_puts: List of best put options
            iv_percentile: IV percentile for additional scoring
            
        Returns:
            Dictionary with options score and components
        """
        score = 0.0
        components = {}
        
        # Call options availability and quality (50 points)
        if best_calls:
            best_call_score = best_calls[0]['scoring']['total_score']
            # Scale from 100-point system to 50 points
            call_score = (best_call_score / 100) * 50
        else:
            call_score = 0
        
        components['call_options'] = call_score
        score += call_score
        
        # Put options availability and quality (30 points)
        if best_puts:
            best_put_score = best_puts[0]['scoring']['total_score']
            # Scale from 100-point system to 30 points
            put_score = (best_put_score / 100) * 30
        else:
            put_score = 0
        
        components['put_options'] = put_score
        score += put_score
        
        # IV percentile consideration (20 points)
        if iv_percentile is not None and not np.isnan(iv_percentile):
            if iv_percentile < 30:  # Low IV - good for buying
                iv_score = 20
            elif iv_percentile < 50:  # Moderate IV
                iv_score = 15
            elif iv_percentile < 70:  # High IV
                iv_score = 10
            else:  # Very high IV
                iv_score = 5
        else:
            iv_score = 10  # Neutral
        
        components['iv_percentile'] = iv_score
        score += iv_score
        
        return {
            'options_score': min(100, score),
            'components': components
        }
    
    def calculate_quality_score(self, data_quality: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data quality and filter score (0-100).
        
        Args:
            data_quality: Dictionary with data quality metrics
            
        Returns:
            Dictionary with quality score and components
        """
        score = 100.0  # Start with perfect score and deduct
        components = {}
        
        # Data completeness (40 points max deduction)
        missing_indicators = data_quality.get('missing_indicators', 0)
        total_indicators = data_quality.get('total_indicators', 20)
        
        if total_indicators > 0:
            completeness_pct = 1 - (missing_indicators / total_indicators)
            completeness_score = completeness_pct * 40
        else:
            completeness_score = 0
        
        components['data_completeness'] = completeness_score
        score = min(score, 60 + completeness_score)  # Max 40 point deduction
        
        # Volume adequacy (20 points)
        volume_ratio = data_quality.get('volume_ratio', 1.0)
        if volume_ratio >= 0.5:  # Adequate volume
            volume_adequacy = 20
        elif volume_ratio >= 0.3:  # Low volume
            volume_adequacy = 15
        elif volume_ratio >= 0.1:  # Very low volume
            volume_adequacy = 10
        else:  # Insufficient volume
            volume_adequacy = 5
        
        components['volume_adequacy'] = volume_adequacy
        
        # Price action quality (20 points)
        price_volatility = data_quality.get('price_volatility', 0.02)
        if 0.01 <= price_volatility <= 0.05:  # Normal volatility
            price_quality = 20
        elif 0.005 <= price_volatility <= 0.10:  # Acceptable volatility
            price_quality = 15
        else:  # Too high or too low volatility
            price_quality = 10
        
        components['price_quality'] = price_quality
        
        # Market timing quality (20 points)
        market_timing = data_quality.get('market_timing_score', 15)
        components['market_timing'] = market_timing
        
        # Calculate final quality score
        quality_components = [volume_adequacy, price_quality, market_timing]
        final_score = min(100, np.mean(quality_components) + completeness_score * 0.4)
        
        return {
            'quality_score': final_score,
            'components': components
        }
    
    def calculate_comprehensive_score(self, ohlcv_data: pd.DataFrame,
                                    options_chain: Optional[List[OptionContract]] = None,
                                    benchmark_data: Optional[pd.DataFrame] = None,
                                    current_stock_price: Optional[float] = None,
                                    iv_percentile: Optional[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive quantitative score.
        
        Args:
            ohlcv_data: OHLCV DataFrame
            options_chain: Optional list of option contracts
            benchmark_data: Optional benchmark data for relative strength
            current_stock_price: Current stock price
            iv_percentile: IV percentile
            
        Returns:
            Dictionary with comprehensive scoring results
        """
        try:
            results = {
                'timestamp': datetime.now(),
                'overall_score': 0.0,
                'component_scores': {},
                'analysis_details': {},
                'warnings': [],
                'data_quality': {}
            }
            
            # Use closing price if current price not provided
            if current_stock_price is None and not ohlcv_data.empty:
                current_stock_price = ohlcv_data['close'].iloc[-1]
            
            # Calculate technical indicators
            try:
                technical_indicators = TechnicalIndicators.calculate_all_indicators(ohlcv_data)
                tech_score = self.calculate_technical_score(technical_indicators)
                results['component_scores']['technical'] = tech_score
                results['analysis_details']['technical_indicators'] = technical_indicators
            except Exception as e:
                results['warnings'].append(f"Technical analysis error: {str(e)}")
                results['component_scores']['technical'] = {'technical_score': 50}
            
            # Calculate momentum analysis
            try:
                momentum_data = MomentumAnalysis.calculate_all_momentum_indicators(
                    ohlcv_data, benchmark_data)
                momentum_score = self.calculate_momentum_score(momentum_data)
                results['component_scores']['momentum'] = momentum_score
                results['analysis_details']['momentum_data'] = momentum_data
            except Exception as e:
                results['warnings'].append(f"Momentum analysis error: {str(e)}")
                results['component_scores']['momentum'] = {'momentum_score': 50}
            
            # Calculate squeeze analysis
            try:
                squeeze_data = SqueezeDetector.comprehensive_squeeze_analysis(ohlcv_data)
                squeeze_score = self.calculate_squeeze_score(squeeze_data)
                results['component_scores']['squeeze'] = squeeze_score
                results['analysis_details']['squeeze_data'] = squeeze_data
            except Exception as e:
                results['warnings'].append(f"Squeeze analysis error: {str(e)}")
                results['component_scores']['squeeze'] = {'squeeze_score': 50}
            
            # Options analysis removed - focusing on stock-only workflow
            
            # Calculate data quality score
            try:
                quality_metrics = self._assess_data_quality(
                    ohlcv_data, options_chain, technical_indicators)
                quality_score = self.calculate_quality_score(quality_metrics)
                results['component_scores']['quality'] = quality_score
                results['data_quality'] = quality_metrics
            except Exception as e:
                results['warnings'].append(f"Quality assessment error: {str(e)}")
                results['component_scores']['quality'] = {'quality_score': 80}
            
            # Calculate weighted overall score
            overall_score = 0.0
            total_weight = 0.0
            
            for component, weight in self.weights.items():
                if component in results['component_scores']:
                    component_score = results['component_scores'][component]
                    score_key = f'{component}_score'
                    if score_key in component_score:
                        overall_score += component_score[score_key] * (weight / 100)
                        total_weight += weight
            
            # Normalize if weights don't sum to 100
            if total_weight > 0:
                results['overall_score'] = (overall_score / total_weight) * 100
            else:
                results['overall_score'] = 0.0
            
            # Add score interpretation
            results['score_interpretation'] = self._interpret_score(results['overall_score'])
            
            return results
            
        except Exception as e:
            return {
                'error': f"Comprehensive scoring error: {str(e)}",
                'timestamp': datetime.now(),
                'overall_score': 0.0
            }
    
    def _assess_data_quality(self, ohlcv_data: pd.DataFrame,
                           options_chain: Optional[List[OptionContract]],
                           technical_indicators: Dict[str, float]) -> Dict[str, Any]:
        """Assess data quality metrics."""
        quality = {}
        
        # Data completeness
        nan_count = sum(1 for value in technical_indicators.values() 
                       if value is None or (isinstance(value, float) and np.isnan(value)))
        quality['missing_indicators'] = nan_count
        quality['total_indicators'] = len(technical_indicators)
        
        # Volume analysis
        if not ohlcv_data.empty:
            recent_volume = ohlcv_data['volume'].tail(20).mean()
            avg_volume = ohlcv_data['volume'].mean()
            quality['volume_ratio'] = recent_volume / avg_volume if avg_volume > 0 else 0
        else:
            quality['volume_ratio'] = 0
        
        # Price volatility
        if not ohlcv_data.empty and len(ohlcv_data) > 1:
            returns = ohlcv_data['close'].pct_change().dropna()
            quality['price_volatility'] = returns.std() if len(returns) > 0 else 0.02
        else:
            quality['price_volatility'] = 0.02
        
        # Market timing assessment (replaces options availability)
        # Simple market timing based on recent price action and volume
        if not ohlcv_data.empty:
            recent_close = ohlcv_data['close'].tail(5)
            recent_volume = ohlcv_data['volume'].tail(5)
            # Basic scoring: positive trend + above average volume
            price_trend = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0]
            volume_trend = recent_volume.mean() / ohlcv_data['volume'].mean()
            timing_score = 50 + (price_trend * 100) + min(25, (volume_trend - 1) * 50)
            quality['market_timing_score'] = max(0, min(100, timing_score))
        else:
            quality['market_timing_score'] = 50
        
        return quality
    
    def _interpret_score(self, score: float) -> str:
        """Interpret overall score into categories."""
        if score >= 80:
            return "Excellent - Strong opportunity with multiple confirming factors"
        elif score >= 70:
            return "Very Good - Solid opportunity with good setup"
        elif score >= 60:
            return "Good - Decent opportunity, monitor for entry"
        elif score >= 50:
            return "Average - Mixed signals, requires caution"
        elif score >= 40:
            return "Below Average - Weak setup, limited opportunity"
        else:
            return "Poor - Avoid, unfavorable conditions"


class RiskAssessment:
    """Risk assessment and position sizing calculations."""
    
    @staticmethod
    def calculate_risk_metrics(ohlcv_data: pd.DataFrame, 
                             position_size: float = 1000.0) -> Dict[str, float]:
        """Calculate risk metrics for position sizing.
        
        Args:
            ohlcv_data: OHLCV DataFrame
            position_size: Position size in dollars
            
        Returns:
            Dictionary with risk metrics
        """
        if ohlcv_data.empty:
            return {}
        
        close = ohlcv_data['close']
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        
        # Price-based risk metrics
        current_price = close.iloc[-1]
        returns = close.pct_change().dropna()
        
        # Volatility measures
        daily_vol = returns.std() if len(returns) > 0 else 0.02
        annualized_vol = daily_vol * np.sqrt(252)
        
        # ATR-based stop levels
        atr = TechnicalIndicators.atr(high, low, close, 14)
        current_atr = atr.iloc[-1] if not atr.empty else current_price * 0.02
        
        # Risk calculations
        shares = position_size / current_price
        
        # Stop loss levels (2x ATR)
        stop_loss_price = current_price - (2 * current_atr)
        stop_loss_risk = (current_price - stop_loss_price) * shares
        
        # VaR calculation (95% confidence)
        var_95 = position_size * 1.65 * daily_vol  # 95% 1-day VaR
        
        return {
            'current_price': current_price,
            'daily_volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'atr_value': current_atr,
            'atr_percent': (current_atr / current_price) * 100,
            'suggested_stop_loss': stop_loss_price,
            'stop_loss_risk_dollars': stop_loss_risk,
            'stop_loss_risk_percent': (stop_loss_risk / position_size) * 100,
            'value_at_risk_95': var_95,
            'position_size_shares': shares
        }
    
    @staticmethod
    def suggest_position_size(account_balance: float, risk_per_trade: float,
                            risk_metrics: Dict[str, float]) -> Dict[str, float]:
        """Suggest position size based on risk management rules.
        
        Args:
            account_balance: Total account balance
            risk_per_trade: Risk per trade as percentage (e.g., 0.02 for 2%)
            risk_metrics: Risk metrics from calculate_risk_metrics
            
        Returns:
            Dictionary with position sizing recommendations
        """
        max_risk_dollars = account_balance * risk_per_trade
        
        # Position sizing based on ATR stop loss
        stop_loss_risk_pct = risk_metrics.get('stop_loss_risk_percent', 5.0) / 100
        if stop_loss_risk_pct > 0:
            suggested_position = max_risk_dollars / stop_loss_risk_pct
        else:
            suggested_position = account_balance * 0.1  # Conservative fallback
        
        # Risk-adjusted position size
        volatility = risk_metrics.get('daily_volatility', 0.02)
        vol_adjustment = min(1.0, 0.02 / volatility)  # Reduce size for high volatility
        
        adjusted_position = suggested_position * vol_adjustment
        
        return {
            'max_risk_dollars': max_risk_dollars,
            'suggested_position_dollars': min(suggested_position, account_balance * 0.20),
            'volatility_adjusted_position': min(adjusted_position, account_balance * 0.15),
            'position_as_percent_of_account': (adjusted_position / account_balance) * 100,
            'volatility_adjustment_factor': vol_adjustment
        }
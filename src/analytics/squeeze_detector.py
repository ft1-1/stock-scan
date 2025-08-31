"""TTM Squeeze and volatility compression detection algorithms.

This module implements John Carter's TTM (Time The Market) Squeeze indicator
and other volatility compression detection methods used in options trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

from .technical_indicators import TechnicalIndicators
from .momentum_analysis import PercentileCalculator

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class SqueezeDetector:
    """TTM Squeeze and volatility compression detection."""
    
    @staticmethod
    def ttm_squeeze_basic(bb_upper: pd.Series, bb_lower: pd.Series,
                         kc_upper: pd.Series, kc_lower: pd.Series) -> pd.Series:
        """Basic TTM Squeeze detection (Bollinger Bands inside Keltner Channels).
        
        Formula:
        Squeeze = (BB_Upper < KC_Upper) AND (BB_Lower > KC_Lower)
        
        Args:
            bb_upper: Bollinger Band upper series
            bb_lower: Bollinger Band lower series  
            kc_upper: Keltner Channel upper series
            kc_lower: Keltner Channel lower series
            
        Returns:
            Boolean series indicating squeeze conditions
        """
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        return squeeze
    
    @staticmethod
    def ttm_squeeze_momentum(close: pd.Series, bb_middle: pd.Series,
                           kc_middle: pd.Series, period: int = 20) -> pd.Series:
        """TTM Squeeze momentum histogram.
        
        Formula:
        Momentum = Linear Regression of (Close - Average of BB_Middle and KC_Middle)
        
        Args:
            close: Close price series
            bb_middle: Bollinger Band middle (SMA) series
            kc_middle: Keltner Channel middle (EMA) series
            period: Period for linear regression (default 20)
            
        Returns:
            Series with momentum values
        """
        if len(close) < period:
            return pd.Series([np.nan] * len(close), index=close.index)
        
        # Average of the two middle lines
        avg_middle = (bb_middle + kc_middle) / 2
        
        # Price deviation from average middle
        price_deviation = close - avg_middle
        
        # Calculate linear regression slope over rolling window
        momentum = price_deviation.rolling(window=period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
        )
        
        return momentum
    
    @staticmethod
    def squeeze_intensity(bb_upper: pd.Series, bb_lower: pd.Series,
                         kc_upper: pd.Series, kc_lower: pd.Series,
                         lookback_period: int = 252) -> pd.Series:
        """Calculate squeeze intensity based on band width percentiles.
        
        Args:
            bb_upper: Bollinger Band upper series
            bb_lower: Bollinger Band lower series
            kc_upper: Keltner Channel upper series
            kc_lower: Keltner Channel lower series
            lookback_period: Lookback period for percentile calculation
            
        Returns:
            Series with squeeze intensity (0-100, higher = tighter squeeze)
        """
        # Calculate band widths
        bb_width = ((bb_upper - bb_lower) / bb_upper) * 100
        kc_width = ((kc_upper - kc_lower) / kc_upper) * 100
        
        # Average width as overall volatility measure
        avg_width = (bb_width + kc_width) / 2
        
        # Calculate rolling percentile (lower percentile = tighter squeeze)
        intensity = avg_width.rolling(window=lookback_period).apply(
            lambda x: (1 - (x.iloc[-1] - x.min()) / (x.max() - x.min()) 
                      if x.max() > x.min() else 0.5) * 100
        )
        
        return intensity
    
    @staticmethod
    def volatility_compression_score(atr: pd.Series, volume: pd.Series,
                                   atr_period: int = 20, volume_period: int = 20) -> pd.Series:
        """Calculate volatility compression score.
        
        Args:
            atr: Average True Range series
            volume: Volume series
            atr_period: Period for ATR percentile calculation
            volume_period: Period for volume analysis
            
        Returns:
            Series with compression scores (0-100)
        """
        if len(atr) < atr_period or len(volume) < volume_period:
            return pd.Series([np.nan] * len(atr), index=atr.index)
        
        score = pd.Series(index=atr.index, dtype=float)
        
        for i in range(max(atr_period, volume_period), len(atr)):
            current_atr = atr.iloc[i]
            current_volume = volume.iloc[i]
            
            # ATR percentile (lower = more compressed)
            historical_atr = atr.iloc[i-atr_period:i]
            atr_percentile = PercentileCalculator.calculate_percentile(current_atr, historical_atr)
            
            # Volume analysis (declining volume adds to compression)
            recent_volume = volume.iloc[i-volume_period:i]
            volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
            
            # Score calculation
            atr_score = (100 - atr_percentile) * 0.7  # Lower ATR percentile = higher score
            volume_score = max(0, -volume_trend / recent_volume.mean() * 100) * 0.3  # Declining volume = higher score
            
            total_score = min(100, max(0, atr_score + volume_score))
            score.iloc[i] = total_score
        
        return score
    
    @staticmethod
    def consolidation_pattern(high: pd.Series, low: pd.Series,
                            period: int = 20, threshold: float = 0.02) -> pd.Series:
        """Detect price consolidation patterns.
        
        Args:
            high: High price series
            low: Low price series
            period: Period for consolidation analysis
            threshold: Maximum range as percentage of price (default 2%)
            
        Returns:
            Boolean series indicating consolidation periods
        """
        if len(high) < period:
            return pd.Series([False] * len(high), index=high.index)
        
        consolidation = pd.Series(index=high.index, dtype=bool)
        
        for i in range(period, len(high)):
            recent_high = high.iloc[i-period:i].max()
            recent_low = low.iloc[i-period:i].min()
            
            if recent_high > 0:
                price_range_pct = (recent_high - recent_low) / recent_high
                consolidation.iloc[i] = price_range_pct <= threshold
            else:
                consolidation.iloc[i] = False
        
        return consolidation
    
    @staticmethod
    def squeeze_duration(squeeze_series: pd.Series) -> pd.Series:
        """Calculate the duration of current squeeze periods.
        
        Args:
            squeeze_series: Boolean series of squeeze conditions
            
        Returns:
            Series with squeeze duration (number of periods)
        """
        duration = pd.Series(index=squeeze_series.index, dtype=int)
        current_duration = 0
        
        for i, is_squeeze in enumerate(squeeze_series):
            if is_squeeze:
                current_duration += 1
            else:
                current_duration = 0
            duration.iloc[i] = current_duration
        
        return duration
    
    @staticmethod
    def squeeze_breakout_prediction(momentum: pd.Series, squeeze_duration: pd.Series,
                                  min_duration: int = 6) -> pd.Series:
        """Predict squeeze breakout direction and probability.
        
        Args:
            momentum: TTM squeeze momentum series
            squeeze_duration: Squeeze duration series
            min_duration: Minimum squeeze duration for breakout prediction
            
        Returns:
            Series with breakout scores (-100 to 100, positive = bullish)
        """
        if len(momentum) != len(squeeze_duration):
            raise ValueError("Momentum and duration series must have same length")
        
        breakout_score = pd.Series(index=momentum.index, dtype=float)
        
        for i in range(len(momentum)):
            if squeeze_duration.iloc[i] >= min_duration:
                # Analyze recent momentum trend
                if i >= 3:  # Need at least 3 points for trend
                    recent_momentum = momentum.iloc[i-3:i+1]
                    
                    # Calculate momentum trend
                    if not recent_momentum.isna().all():
                        trend = np.polyfit(range(len(recent_momentum)), 
                                         recent_momentum.fillna(0), 1)[0]
                        
                        # Scale to -100 to 100
                        score = min(100, max(-100, trend * 1000))
                        breakout_score.iloc[i] = score
                    else:
                        breakout_score.iloc[i] = 0
                else:
                    breakout_score.iloc[i] = 0
            else:
                breakout_score.iloc[i] = 0
        
        return breakout_score
    
    @staticmethod
    def multi_timeframe_squeeze(ohlcv_data: pd.DataFrame, 
                               timeframes: List[int] = [5, 10, 20]) -> Dict[str, pd.Series]:
        """Detect squeeze across multiple timeframes.
        
        Args:
            ohlcv_data: OHLCV DataFrame
            timeframes: List of timeframe periods
            
        Returns:
            Dictionary with squeeze data for each timeframe
        """
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        
        squeeze_data = {}
        
        for tf in timeframes:
            # Calculate indicators for this timeframe
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close, tf)
            kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(
                high, low, close, tf)
            
            # Basic squeeze detection
            squeeze = SqueezeDetector.ttm_squeeze_basic(bb_upper, bb_lower, kc_upper, kc_lower)
            
            # Momentum
            momentum = SqueezeDetector.ttm_squeeze_momentum(close, bb_middle, kc_middle, tf)
            
            # Intensity
            intensity = SqueezeDetector.squeeze_intensity(bb_upper, bb_lower, kc_upper, kc_lower)
            
            squeeze_data[f'tf_{tf}'] = {
                'squeeze': squeeze,
                'momentum': momentum,
                'intensity': intensity,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'kc_upper': kc_upper,
                'kc_lower': kc_lower
            }
        
        return squeeze_data
    
    @staticmethod
    def comprehensive_squeeze_analysis(ohlcv_data: pd.DataFrame) -> Dict[str, Union[float, bool, int]]:
        """Comprehensive squeeze analysis for current market state.
        
        Args:
            ohlcv_data: OHLCV DataFrame
            
        Returns:
            Dictionary with comprehensive squeeze metrics
        """
        try:
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            close = ohlcv_data['close']
            volume = ohlcv_data['volume']
            
            # Calculate required indicators
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close, 20)
            kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(high, low, close, 20)
            atr = TechnicalIndicators.atr(high, low, close, 14)
            
            analysis = {}
            
            # Basic squeeze detection
            squeeze = SqueezeDetector.ttm_squeeze_basic(bb_upper, bb_lower, kc_upper, kc_lower)
            analysis['is_squeeze'] = squeeze.iloc[-1] if not squeeze.empty else False
            
            # Squeeze intensity
            intensity = SqueezeDetector.squeeze_intensity(bb_upper, bb_lower, kc_upper, kc_lower)
            analysis['squeeze_intensity'] = intensity.iloc[-1] if not intensity.empty else np.nan
            
            # Squeeze duration
            duration = SqueezeDetector.squeeze_duration(squeeze)
            analysis['squeeze_duration'] = duration.iloc[-1] if not duration.empty else 0
            
            # Momentum
            momentum = SqueezeDetector.ttm_squeeze_momentum(close, bb_middle, kc_middle)
            analysis['squeeze_momentum'] = momentum.iloc[-1] if not momentum.empty else np.nan
            
            # Volatility compression score
            compression = SqueezeDetector.volatility_compression_score(atr, volume)
            analysis['volatility_compression'] = compression.iloc[-1] if not compression.empty else np.nan
            
            # Consolidation pattern
            consolidation = SqueezeDetector.consolidation_pattern(high, low)
            analysis['is_consolidating'] = consolidation.iloc[-1] if not consolidation.empty else False
            
            # Breakout prediction
            if not momentum.empty and not duration.empty:
                breakout_score = SqueezeDetector.squeeze_breakout_prediction(momentum, duration)
                analysis['breakout_score'] = breakout_score.iloc[-1] if not breakout_score.empty else 0
            else:
                analysis['breakout_score'] = 0
            
            # Band width percentiles
            if len(bb_upper) >= 252 and len(kc_upper) >= 252:
                bb_width_pct = PercentileCalculator.bollinger_width_percentile(bb_upper, bb_lower, 252)
                analysis['bb_width_percentile'] = bb_width_pct
            else:
                analysis['bb_width_percentile'] = np.nan
            
            # ATR percentile
            if len(atr) >= 252:
                current_atr = atr.iloc[-1]
                atr_pct = PercentileCalculator.atr_percentile(current_atr, atr, 252)
                analysis['atr_percentile'] = atr_pct
            else:
                analysis['atr_percentile'] = np.nan
            
            # Multi-timeframe confirmation
            mtf_data = SqueezeDetector.multi_timeframe_squeeze(ohlcv_data, [10, 20, 30])
            squeeze_count = sum(1 for tf_data in mtf_data.values() 
                              if not tf_data['squeeze'].empty and tf_data['squeeze'].iloc[-1])
            analysis['multi_timeframe_squeeze_count'] = squeeze_count
            
            return analysis
            
        except Exception as e:
            raise ValueError(f"Error in squeeze analysis: {str(e)}")


class VolatilityRegimeDetector:
    """Detect different volatility regimes for options trading."""
    
    @staticmethod
    def volatility_regime(atr_percentile: float, volume_percentile: float) -> str:
        """Classify volatility regime.
        
        Args:
            atr_percentile: ATR percentile (0-100)
            volume_percentile: Volume percentile (0-100)
            
        Returns:
            Volatility regime: 'low', 'normal', 'elevated', 'high'
        """
        if np.isnan(atr_percentile) or np.isnan(volume_percentile):
            return 'unknown'
        
        # Combine ATR and volume percentiles
        combined_score = (atr_percentile * 0.7) + (volume_percentile * 0.3)
        
        if combined_score < 20:
            return 'low'
        elif combined_score < 40:
            return 'normal'  
        elif combined_score < 70:
            return 'elevated'
        else:
            return 'high'
    
    @staticmethod
    def volatility_trend(atr_series: pd.Series, period: int = 20) -> str:
        """Determine volatility trend direction.
        
        Args:
            atr_series: ATR time series
            period: Period for trend analysis
            
        Returns:
            Trend direction: 'increasing', 'decreasing', 'stable'
        """
        if len(atr_series) < period:
            return 'unknown'
        
        recent_atr = atr_series.tail(period)
        
        # Calculate linear regression slope
        x = np.arange(len(recent_atr))
        y = recent_atr.values
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope relative to ATR level
            avg_atr = recent_atr.mean()
            normalized_slope = slope / avg_atr if avg_atr > 0 else 0
            
            if normalized_slope > 0.02:  # 2% increase per period
                return 'increasing'
            elif normalized_slope < -0.02:  # 2% decrease per period
                return 'decreasing'
            else:
                return 'stable'
                
        except:
            return 'unknown'
    
    @staticmethod
    def options_environment_score(squeeze_analysis: Dict[str, Union[float, bool, int]],
                                 iv_percentile: float = None) -> int:
        """Score the current options trading environment (0-100).
        
        Args:
            squeeze_analysis: Results from comprehensive_squeeze_analysis
            iv_percentile: Implied volatility percentile
            
        Returns:
            Environment score (0-100, higher = better for options strategies)
        """
        score = 0
        
        # Squeeze conditions (40 points max)
        if squeeze_analysis.get('is_squeeze', False):
            score += 20
        
        squeeze_intensity = squeeze_analysis.get('squeeze_intensity', 0)
        if not np.isnan(squeeze_intensity):
            score += min(20, squeeze_intensity * 0.2)
        
        # Volatility compression (30 points max)
        compression = squeeze_analysis.get('volatility_compression', 0)
        if not np.isnan(compression):
            score += min(30, compression * 0.3)
        
        # Multi-timeframe confirmation (20 points max)
        mtf_count = squeeze_analysis.get('multi_timeframe_squeeze_count', 0)
        score += min(20, mtf_count * 6.67)  # 3 timeframes max
        
        # IV percentile consideration (10 points max)
        if iv_percentile is not None and not np.isnan(iv_percentile):
            if iv_percentile < 30:  # Low IV is good for buying options
                score += 10
            elif iv_percentile > 70:  # High IV is good for selling options
                score += 5
        
        return min(100, max(0, int(score)))
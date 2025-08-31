"""Momentum and trend analysis for quantitative stock screening.

This module implements momentum calculations including returns analysis,
relative strength comparisons, and trend strength measurements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class MomentumAnalysis:
    """Momentum and trend analysis calculations."""
    
    @staticmethod
    def price_momentum(prices: pd.Series, periods: List[int]) -> Dict[str, float]:
        """Calculate price momentum over multiple periods.
        
        Formula: ((Current Price - Price N periods ago) / Price N periods ago) * 100
        
        Args:
            prices: Price series (typically close prices)
            periods: List of lookback periods (e.g., [21, 63, 252])
            
        Returns:
            Dictionary with momentum values for each period
        """
        momentum = {}
        current_price = prices.iloc[-1]
        
        for period in periods:
            if len(prices) >= period + 1:
                past_price = prices.iloc[-(period + 1)]
                momentum[f'momentum_{period}d'] = ((current_price - past_price) / past_price) * 100
            else:
                momentum[f'momentum_{period}d'] = np.nan
                
        return momentum
    
    @staticmethod
    def acceleration(prices: pd.Series, short_period: int = 21, long_period: int = 63) -> float:
        """Calculate momentum acceleration.
        
        Formula: Short term momentum - Long term momentum
        This measures if momentum is increasing or decreasing.
        
        Args:
            prices: Price series
            short_period: Short momentum period (default 21)
            long_period: Long momentum period (default 63)
            
        Returns:
            Momentum acceleration value
        """
        if len(prices) < long_period + 1:
            return np.nan
        
        # Calculate short and long term momentum
        short_momentum = MomentumAnalysis.price_momentum(prices, [short_period])[f'momentum_{short_period}d']
        long_momentum = MomentumAnalysis.price_momentum(prices, [long_period])[f'momentum_{long_period}d']
        
        if np.isnan(short_momentum) or np.isnan(long_momentum):
            return np.nan
            
        return short_momentum - long_momentum
    
    @staticmethod
    def relative_strength(stock_prices: pd.Series, benchmark_prices: pd.Series, 
                         period: int = 63) -> float:
        """Calculate relative strength vs benchmark over specified period.
        
        Formula: Stock Return - Benchmark Return over period
        
        Args:
            stock_prices: Stock price series
            benchmark_prices: Benchmark price series (e.g., S&P 500)
            period: Lookback period (default 63)
            
        Returns:
            Relative strength value (positive = outperformance)
        """
        if len(stock_prices) < period + 1 or len(benchmark_prices) < period + 1:
            return np.nan
        
        # Align series by their common index
        aligned_stock, aligned_benchmark = stock_prices.align(benchmark_prices, join='inner')
        
        if len(aligned_stock) < period + 1:
            return np.nan
        
        # Calculate returns for both
        stock_return = ((aligned_stock.iloc[-1] - aligned_stock.iloc[-(period + 1)]) / 
                       aligned_stock.iloc[-(period + 1)]) * 100
        
        benchmark_return = ((aligned_benchmark.iloc[-1] - aligned_benchmark.iloc[-(period + 1)]) / 
                           aligned_benchmark.iloc[-(period + 1)]) * 100
        
        return stock_return - benchmark_return
    
    @staticmethod
    def rate_of_change(prices: pd.Series, period: int = 21) -> pd.Series:
        """Rate of Change (ROC) indicator.
        
        Formula: ((Current Price - Price N periods ago) / Price N periods ago) * 100
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Series with ROC values
        """
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        past_prices = prices.shift(period)
        roc = ((prices - past_prices) / past_prices) * 100
        return roc
    
    @staticmethod
    def momentum_oscillator(prices: pd.Series, period: int = 14) -> pd.Series:
        """Momentum oscillator indicator.
        
        Formula: Current Price - Price N periods ago
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Series with momentum oscillator values
        """
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        return prices - prices.shift(period)
    
    @staticmethod
    def price_velocity(prices: pd.Series, period: int = 14) -> pd.Series:
        """Price velocity (rate of price change).
        
        Formula: (Current Price - Previous Price) / Previous Price
        
        Args:
            prices: Price series
            period: Smoothing period for velocity calculation
            
        Returns:
            Series with velocity values
        """
        if len(prices) < 2:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate period-to-period velocity
        velocity = prices.pct_change()
        
        # Smooth with rolling average if period > 1
        if period > 1:
            velocity = velocity.rolling(window=period, min_periods=1).mean()
        
        return velocity
    
    @staticmethod
    def trend_strength(prices: pd.Series, period: int = 20) -> float:
        """Calculate trend strength using R-squared of linear regression.
        
        Args:
            prices: Price series
            period: Period for trend analysis
            
        Returns:
            R-squared value (0-1) indicating trend strength
        """
        if len(prices) < period:
            return np.nan
        
        # Use the last 'period' prices
        recent_prices = prices.tail(period).values
        x = np.arange(len(recent_prices))
        
        try:
            # Calculate linear regression
            correlation_matrix = np.corrcoef(x, recent_prices)
            correlation = correlation_matrix[0, 1]
            r_squared = correlation ** 2
            
            return r_squared if not np.isnan(r_squared) else 0.0
            
        except:
            return 0.0
    
    @staticmethod
    def trend_direction(prices: pd.Series, short_ma: int = 20, long_ma: int = 50) -> int:
        """Determine trend direction using moving average crossover.
        
        Args:
            prices: Price series
            short_ma: Short moving average period
            long_ma: Long moving average period
            
        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways/unclear
        """
        if len(prices) < long_ma:
            return 0
        
        short_sma = prices.rolling(window=short_ma, min_periods=short_ma).mean().iloc[-1]
        long_sma = prices.rolling(window=long_ma, min_periods=long_ma).mean().iloc[-1]
        
        if np.isnan(short_sma) or np.isnan(long_sma):
            return 0
        
        if short_sma > long_sma:
            return 1  # Uptrend
        elif short_sma < long_sma:
            return -1  # Downtrend
        else:
            return 0  # Neutral
    
    @staticmethod
    def moving_average_ribbon_score(prices: pd.Series, 
                                  ma_periods: List[int] = [10, 20, 30, 40, 50]) -> float:
        """Score based on moving average ribbon alignment.
        
        Args:
            prices: Price series
            ma_periods: List of MA periods to calculate
            
        Returns:
            Score from 0-100 based on MA alignment (100 = perfect uptrend)
        """
        if len(prices) < max(ma_periods):
            return np.nan
        
        # Calculate all MAs
        mas = []
        current_price = prices.iloc[-1]
        
        for period in ma_periods:
            ma = prices.rolling(window=period, min_periods=period).mean().iloc[-1]
            if not np.isnan(ma):
                mas.append(ma)
        
        if len(mas) < 2:
            return 50.0  # Neutral score
        
        # Check if price is above all MAs
        price_above_all = all(current_price > ma for ma in mas)
        
        # Check if MAs are in ascending order (shortest to longest)
        mas_ascending = all(mas[i] >= mas[i+1] for i in range(len(mas)-1))
        
        # Calculate score
        score = 50.0  # Base score
        
        if price_above_all:
            score += 30.0
        
        if mas_ascending:
            score += 20.0
        
        # Adjust based on distance from MAs
        if mas:
            avg_ma = np.mean(mas)
            price_distance_pct = ((current_price - avg_ma) / avg_ma) * 100
            # Add bonus for being significantly above MAs (up to 10 points)
            distance_bonus = min(10.0, max(-10.0, price_distance_pct))
            score += distance_bonus
        
        return max(0.0, min(100.0, score))
    
    @staticmethod
    def breakout_strength(high: pd.Series, volume: pd.Series, 
                         lookback_period: int = 20) -> float:
        """Calculate breakout strength based on new highs and volume.
        
        Args:
            high: High price series
            volume: Volume series  
            lookback_period: Period for breakout analysis
            
        Returns:
            Breakout strength score (0-100)
        """
        if len(high) < lookback_period or len(volume) < lookback_period:
            return np.nan
        
        # Check if current high is a new high
        recent_highs = high.tail(lookback_period)
        current_high = high.iloc[-1]
        highest_in_period = recent_highs.max()
        
        is_new_high = current_high >= highest_in_period
        
        if not is_new_high:
            return 0.0
        
        # Calculate volume confirmation
        recent_volume = volume.tail(lookback_period)
        avg_volume = recent_volume.mean()
        current_volume = volume.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Base score for new high
        score = 50.0
        
        # Volume confirmation (up to 50 additional points)
        volume_score = min(50.0, (volume_ratio - 1.0) * 25.0)
        score += max(0.0, volume_score)
        
        return min(100.0, score)
    
    @staticmethod
    def momentum_divergence(prices: pd.Series, momentum_indicator: pd.Series,
                           period: int = 14) -> float:
        """Detect momentum divergence between price and momentum indicator.
        
        Args:
            prices: Price series
            momentum_indicator: Momentum indicator series (e.g., RSI, MACD)
            period: Period for divergence analysis
            
        Returns:
            Divergence score (-100 to 100, positive = bullish divergence)
        """
        if len(prices) < period or len(momentum_indicator) < period:
            return np.nan
        
        # Get recent periods
        recent_prices = prices.tail(period)
        recent_momentum = momentum_indicator.tail(period)
        
        # Calculate price trend
        price_start = recent_prices.iloc[0]
        price_end = recent_prices.iloc[-1]
        price_trend = (price_end - price_start) / price_start
        
        # Calculate momentum trend
        momentum_start = recent_momentum.iloc[0]
        momentum_end = recent_momentum.iloc[-1]
        
        if np.isnan(momentum_start) or np.isnan(momentum_end) or momentum_start == 0:
            return 0.0
        
        momentum_trend = (momentum_end - momentum_start) / abs(momentum_start)
        
        # Calculate divergence
        # Bullish divergence: price down, momentum up
        # Bearish divergence: price up, momentum down
        divergence = momentum_trend - price_trend
        
        # Scale to -100 to 100
        return max(-100.0, min(100.0, divergence * 100))
    
    @staticmethod
    def calculate_all_momentum_indicators(ohlcv_data: pd.DataFrame,
                                        benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate comprehensive momentum analysis.
        
        Args:
            ohlcv_data: OHLCV DataFrame
            benchmark_data: Optional benchmark data for relative strength
            
        Returns:
            Dictionary with all momentum indicators
        """
        try:
            high = ohlcv_data['high']
            low = ohlcv_data['low'] 
            close = ohlcv_data['close']
            volume = ohlcv_data['volume']
            
            indicators = {}
            
            # Basic momentum calculations
            momentum_periods = [21, 63, 126, 252]  # ~1 month, 3 months, 6 months, 1 year
            momentum_results = MomentumAnalysis.price_momentum(close, momentum_periods)
            indicators.update(momentum_results)
            
            # Momentum acceleration
            indicators['momentum_acceleration'] = MomentumAnalysis.acceleration(close, 21, 63)
            
            # Relative strength vs benchmark
            if benchmark_data is not None and 'close' in benchmark_data.columns:
                indicators['relative_strength_63d'] = MomentumAnalysis.relative_strength(
                    close, benchmark_data['close'], 63)
                indicators['relative_strength_252d'] = MomentumAnalysis.relative_strength(
                    close, benchmark_data['close'], 252)
            else:
                indicators['relative_strength_63d'] = np.nan
                indicators['relative_strength_252d'] = np.nan
            
            # Rate of change
            roc_21 = MomentumAnalysis.rate_of_change(close, 21)
            indicators['roc_21d'] = roc_21.iloc[-1] if not roc_21.empty else np.nan
            
            # Trend analysis
            indicators['trend_strength_20d'] = MomentumAnalysis.trend_strength(close, 20)
            indicators['trend_direction'] = MomentumAnalysis.trend_direction(close, 20, 50)
            
            # Moving average ribbon score
            indicators['ma_ribbon_score'] = MomentumAnalysis.moving_average_ribbon_score(close)
            
            # Breakout strength
            indicators['breakout_strength'] = MomentumAnalysis.breakout_strength(high, volume, 20)
            
            # Price velocity
            velocity = MomentumAnalysis.price_velocity(close, 5)
            indicators['price_velocity'] = velocity.iloc[-1] if not velocity.empty else np.nan
            
            return indicators
            
        except Exception as e:
            raise ValueError(f"Error calculating momentum indicators: {str(e)}")


class PercentileCalculator:
    """Calculate percentiles for various metrics with lookback periods."""
    
    @staticmethod
    def calculate_percentile(current_value: float, historical_series: pd.Series) -> float:
        """Calculate percentile rank of current value in historical context.
        
        Args:
            current_value: Current value to rank
            historical_series: Historical data series
            
        Returns:
            Percentile rank (0-100)
        """
        if historical_series.empty or np.isnan(current_value):
            return np.nan
        
        # Remove NaN values
        clean_series = historical_series.dropna()
        
        if len(clean_series) == 0:
            return np.nan
        
        # Calculate percentile rank
        rank = (clean_series < current_value).sum()
        percentile = (rank / len(clean_series)) * 100
        
        return percentile
    
    @staticmethod
    def atr_percentile(current_atr: float, atr_series: pd.Series, 
                      lookback_period: int = 252) -> float:
        """Calculate ATR percentile over lookback period.
        
        Args:
            current_atr: Current ATR value
            atr_series: Historical ATR series
            lookback_period: Lookback period (default 252 trading days = 1 year)
            
        Returns:
            ATR percentile (0-100)
        """
        if len(atr_series) < lookback_period:
            return np.nan
        
        # Use the last 'lookback_period' values
        historical_atr = atr_series.tail(lookback_period)
        
        return PercentileCalculator.calculate_percentile(current_atr, historical_atr)
    
    @staticmethod
    def volume_percentile(current_volume: int, volume_series: pd.Series,
                         lookback_period: int = 63) -> float:
        """Calculate volume percentile.
        
        Args:
            current_volume: Current volume
            volume_series: Historical volume series
            lookback_period: Lookback period (default 63 trading days)
            
        Returns:
            Volume percentile (0-100)
        """
        if len(volume_series) < lookback_period:
            return np.nan
        
        historical_volume = volume_series.tail(lookback_period)
        
        return PercentileCalculator.calculate_percentile(current_volume, historical_volume)
    
    @staticmethod
    def bollinger_width_percentile(bb_upper: pd.Series, bb_lower: pd.Series,
                                  lookback_period: int = 252) -> float:
        """Calculate Bollinger Band width percentile.
        
        Args:
            bb_upper: Bollinger Band upper series
            bb_lower: Bollinger Band lower series
            lookback_period: Lookback period for percentile calculation
            
        Returns:
            BB width percentile (0-100, low values indicate squeeze)
        """
        if len(bb_upper) < lookback_period or len(bb_lower) < lookback_period:
            return np.nan
        
        # Calculate width series
        width_series = ((bb_upper - bb_lower) / bb_upper) * 100
        
        # Current width
        current_width = width_series.iloc[-1]
        
        # Historical widths
        historical_widths = width_series.tail(lookback_period)
        
        return PercentileCalculator.calculate_percentile(current_width, historical_widths)
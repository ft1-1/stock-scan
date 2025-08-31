"""Technical indicators for quantitative analysis of stock data.

This module provides implementations of commonly used technical indicators
including momentum, trend, volatility, and volume-based indicators.
Mathematical formulas are documented for each indicator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


class TechnicalIndicators:
    """Technical analysis indicators with vectorized calculations."""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average.
        
        Formula: SMA = (P1 + P2 + ... + Pn) / n
        
        Args:
            prices: Price series (typically close prices)
            period: Number of periods for averaging
            
        Returns:
            Series with SMA values
        """
        if len(prices) < period:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        return prices.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average.
        
        Formula: EMA = (Close - Previous_EMA) * (2 / (Period + 1)) + Previous_EMA
        
        Args:
            prices: Price series
            period: Number of periods for averaging
            
        Returns:
            Series with EMA values
        """
        if len(prices) < period:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index.
        
        Formula: RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss over period
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            Series with RSI values (0-100)
        """
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence.
        
        Formula: 
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        
        Args:
            prices: Price series
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < slow_period:
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators.ema(prices, fast_period)
        ema_slow = TechnicalIndicators.ema(prices, slow_period)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.
        
        Formula:
        Middle Band = SMA(20)
        Upper Band = SMA(20) + (2 * Standard Deviation)
        Lower Band = SMA(20) - (2 * Standard Deviation)
        
        Args:
            prices: Price series
            period: Period for SMA and standard deviation (default 20)
            std_dev: Number of standard deviations (default 2.0)
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series
        
        # Calculate middle band (SMA)
        middle_band = TechnicalIndicators.sma(prices, period)
        
        # Calculate standard deviation
        rolling_std = prices.rolling(window=period, min_periods=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range.
        
        Formula:
        TR = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
        ATR = Average of TR over period using Wilder's smoothing
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)
            
        Returns:
            Series with ATR values
        """
        if len(high) < period + 1:
            return pd.Series([np.nan] * len(high), index=high.index)
        
        # Calculate True Range components
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR using Wilder's smoothing (EMA with alpha = 1/period)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index and Directional Movement.
        
        Formula:
        +DM = High(t) - High(t-1) if > 0 and > Low(t-1) - Low(t), else 0
        -DM = Low(t-1) - Low(t) if > 0 and > High(t) - High(t-1), else 0
        TR = True Range
        +DI = 100 * (+DM smoothed / TR smoothed)
        -DI = 100 * (-DM smoothed / TR smoothed)
        DX = 100 * abs(+DI - -DI) / (+DI + -DI)
        ADX = Smoothed DX
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period (default 14)
            
        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        if len(high) < period + 1:
            nan_series = pd.Series([np.nan] * len(high), index=high.index)
            return nan_series, nan_series, nan_series
        
        # Calculate directional movements
        high_diff = high.diff()
        low_diff = low.diff()
        
        # +DM and -DM
        plus_dm = pd.Series(index=high.index, dtype=float)
        minus_dm = pd.Series(index=high.index, dtype=float)
        
        plus_dm = np.where((high_diff > low_diff.abs()) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff.abs() > high_diff) & (low_diff < 0), low_diff.abs(), 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # True Range
        tr = TechnicalIndicators._true_range(high, low, close)
        
        # Smooth DM and TR using Wilder's method
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range for ADX calculation."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels.
        
        Formula:
        Middle Line = EMA of typical price
        Upper Channel = Middle Line + (multiplier * ATR)
        Lower Channel = Middle Line - (multiplier * ATR)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for EMA and ATR (default 20)
            multiplier: ATR multiplier (default 2.0)
            
        Returns:
            Tuple of (upper_channel, middle_line, lower_channel)
        """
        if len(high) < period:
            nan_series = pd.Series([np.nan] * len(high), index=high.index)
            return nan_series, nan_series, nan_series
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # Middle line (EMA of typical price)
        middle_line = TechnicalIndicators.ema(typical_price, period)
        
        # ATR
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        # Upper and lower channels
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return upper_channel, middle_line, lower_channel
    
    @staticmethod
    def percent_above_ma(prices: pd.Series, ma_period: int) -> pd.Series:
        """Calculate percentage above moving average.
        
        Formula: ((Current Price - MA) / MA) * 100
        
        Args:
            prices: Price series
            ma_period: Moving average period
            
        Returns:
            Series with percentage above MA
        """
        ma = TechnicalIndicators.sma(prices, ma_period)
        return ((prices - ma) / ma) * 100
    
    @staticmethod
    def price_returns(prices: pd.Series, period: int) -> pd.Series:
        """Calculate price returns over specified period.
        
        Formula: ((Current Price - Price N periods ago) / Price N periods ago) * 100
        
        Args:
            prices: Price series
            period: Number of periods to look back
            
        Returns:
            Series with percentage returns
        """
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        past_prices = prices.shift(period)
        returns = ((prices - past_prices) / past_prices) * 100
        return returns
    
    @staticmethod
    def new_high_flag(high: pd.Series, period: int = 55) -> pd.Series:
        """Flag for new highs over specified period.
        
        Args:
            high: High price series
            period: Lookback period (default 55 days)
            
        Returns:
            Boolean series indicating new highs
        """
        if len(high) < period:
            return pd.Series([False] * len(high), index=high.index)
        
        rolling_max = high.rolling(window=period, min_periods=period).max()
        return high >= rolling_max
    
    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Current volume relative to average volume.
        
        Formula: Current Volume / Average Volume over period
        
        Args:
            volume: Volume series
            period: Period for average calculation
            
        Returns:
            Series with volume ratios
        """
        if len(volume) < period:
            return pd.Series([np.nan] * len(volume), index=volume.index)
        
        avg_volume = volume.rolling(window=period, min_periods=period).mean()
        return volume / avg_volume
    
    @staticmethod
    def calculate_all_indicators(ohlcv_data: pd.DataFrame) -> Dict[str, Union[float, pd.Series]]:
        """Calculate all technical indicators for a given OHLCV dataset.
        
        Args:
            ohlcv_data: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            Dictionary containing all calculated indicators
        """
        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in ohlcv_data.columns for col in required_cols):
                raise ValueError(f"OHLCV data must contain columns: {required_cols}")
            
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            close = ohlcv_data['close']
            volume = ohlcv_data['volume']
            
            # Get the latest values for scalar indicators
            latest_idx = -1
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = TechnicalIndicators.sma(close, 20).iloc[latest_idx] if len(close) >= 20 else np.nan
            indicators['sma_50'] = TechnicalIndicators.sma(close, 50).iloc[latest_idx] if len(close) >= 50 else np.nan
            indicators['sma_200'] = TechnicalIndicators.sma(close, 200).iloc[latest_idx] if len(close) >= 200 else np.nan
            indicators['ema_12'] = TechnicalIndicators.ema(close, 12).iloc[latest_idx] if len(close) >= 12 else np.nan
            indicators['ema_26'] = TechnicalIndicators.ema(close, 26).iloc[latest_idx] if len(close) >= 26 else np.nan
            
            # Momentum Indicators
            indicators['rsi_14'] = TechnicalIndicators.rsi(close, 14).iloc[latest_idx] if len(close) >= 15 else np.nan
            
            # MACD
            if len(close) >= 26:
                macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
                indicators['macd'] = macd_line.iloc[latest_idx] if not macd_line.empty else np.nan
                indicators['macd_signal'] = signal_line.iloc[latest_idx] if not signal_line.empty else np.nan
                indicators['macd_histogram'] = histogram.iloc[latest_idx] if not histogram.empty else np.nan
            else:
                indicators['macd'] = np.nan
                indicators['macd_signal'] = np.nan
                indicators['macd_histogram'] = np.nan
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close, 20)
                indicators['bollinger_upper'] = bb_upper.iloc[latest_idx] if not bb_upper.empty else np.nan
                indicators['bollinger_middle'] = bb_middle.iloc[latest_idx] if not bb_middle.empty else np.nan
                indicators['bollinger_lower'] = bb_lower.iloc[latest_idx] if not bb_lower.empty else np.nan
            else:
                indicators['bollinger_upper'] = np.nan
                indicators['bollinger_middle'] = np.nan
                indicators['bollinger_lower'] = np.nan
            
            # ATR
            indicators['atr_14'] = TechnicalIndicators.atr(high, low, close, 14).iloc[latest_idx] if len(close) >= 15 else np.nan
            
            # ADX
            if len(close) >= 15:
                adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, 14)
                indicators['adx_14'] = adx.iloc[latest_idx] if not adx.empty else np.nan
                indicators['plus_di'] = plus_di.iloc[latest_idx] if not plus_di.empty else np.nan
                indicators['minus_di'] = minus_di.iloc[latest_idx] if not minus_di.empty else np.nan
            else:
                indicators['adx_14'] = np.nan
                indicators['plus_di'] = np.nan
                indicators['minus_di'] = np.nan
            
            # Volume indicators
            indicators['volume_sma_20'] = TechnicalIndicators.sma(volume, 20).iloc[latest_idx] if len(volume) >= 20 else np.nan
            indicators['volume_ratio'] = TechnicalIndicators.volume_ratio(volume, 20).iloc[latest_idx] if len(volume) >= 20 else np.nan
            
            # Custom indicators
            indicators['pct_above_sma_50'] = TechnicalIndicators.percent_above_ma(close, 50).iloc[latest_idx] if len(close) >= 50 else np.nan
            indicators['pct_above_sma_200'] = TechnicalIndicators.percent_above_ma(close, 200).iloc[latest_idx] if len(close) >= 200 else np.nan
            indicators['return_21d'] = TechnicalIndicators.price_returns(close, 21).iloc[latest_idx] if len(close) >= 22 else np.nan
            indicators['return_63d'] = TechnicalIndicators.price_returns(close, 63).iloc[latest_idx] if len(close) >= 64 else np.nan
            indicators['new_high_55d'] = TechnicalIndicators.new_high_flag(high, 55).iloc[latest_idx] if len(high) >= 55 else False
            
            # Keltner Channels for squeeze detection
            if len(close) >= 20:
                kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(high, low, close, 20)
                indicators['keltner_upper'] = kc_upper.iloc[latest_idx] if not kc_upper.empty else np.nan
                indicators['keltner_middle'] = kc_middle.iloc[latest_idx] if not kc_middle.empty else np.nan
                indicators['keltner_lower'] = kc_lower.iloc[latest_idx] if not kc_lower.empty else np.nan
            else:
                indicators['keltner_upper'] = np.nan
                indicators['keltner_middle'] = np.nan
                indicators['keltner_lower'] = np.nan
            
            return indicators
            
        except Exception as e:
            raise ValueError(f"Error calculating technical indicators: {str(e)}")


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Validate OHLCV data format and quality.
    
    Args:
        data: OHLCV DataFrame to validate
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If data is invalid
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
    
    # Check for minimum data points
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for calculations")
    
    # Check for negative values where they shouldn't exist
    if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError("Price data contains non-positive values")
    
    if (data['volume'] < 0).any():
        raise ValueError("Volume data contains negative values")
    
    # Check high >= low
    if (data['high'] < data['low']).any():
        raise ValueError("High prices less than low prices found")
    
    # Check OHLC relationships
    if ((data['open'] > data['high']) | (data['open'] < data['low'])).any():
        raise ValueError("Open prices outside high-low range")
        
    if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
        raise ValueError("Close prices outside high-low range")
    
    return True
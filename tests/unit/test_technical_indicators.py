"""Unit tests for technical indicators module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.analytics.technical_indicators import TechnicalIndicators, validate_ohlcv_data


class TestTechnicalIndicators:
    """Test cases for technical indicators calculations."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data with some trend and volatility
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        price_changes = np.random.normal(0.001, 0.02, 100).cumsum()
        close_prices = base_price * (1 + price_changes)
        
        # Generate OHLC from close prices
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price
        
        # Ensure OHLC relationships are correct
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        volume = np.random.randint(100000, 1000000, 100)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    @pytest.fixture
    def simple_price_series(self):
        """Create simple price series for testing."""
        return pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    
    def test_sma_calculation(self, simple_price_series):
        """Test Simple Moving Average calculation."""
        sma_5 = TechnicalIndicators.sma(simple_price_series, 5)
        
        # First 4 values should be NaN
        assert pd.isna(sma_5.iloc[:4]).all()
        
        # Fifth value should be average of first 5 prices
        expected_sma_5 = (100 + 102 + 101 + 103 + 105) / 5
        assert abs(sma_5.iloc[4] - expected_sma_5) < 1e-10
        
        # Test with period longer than series
        sma_long = TechnicalIndicators.sma(simple_price_series, 20)
        assert pd.isna(sma_long).all()
    
    def test_ema_calculation(self, simple_price_series):
        """Test Exponential Moving Average calculation."""
        ema_5 = TechnicalIndicators.ema(simple_price_series, 5)
        
        # Should not have NaN values (EMA can start from first value)
        assert not pd.isna(ema_5.iloc[-1])
        
        # EMA should respond faster to recent price changes than SMA
        sma_5 = TechnicalIndicators.sma(simple_price_series, 5)
        
        # For upward trending data, EMA should be higher than SMA
        assert ema_5.iloc[-1] > sma_5.iloc[-1]
    
    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI calculation."""
        close_prices = sample_ohlcv_data['close']
        rsi = TechnicalIndicators.rsi(close_prices, 14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
        
        # Should have NaN values for first 14 periods
        assert pd.isna(rsi.iloc[:14]).sum() > 0
        
        # Test edge case with insufficient data
        short_series = pd.Series([100, 101, 102])
        rsi_short = TechnicalIndicators.rsi(short_series, 14)
        assert pd.isna(rsi_short).all()
    
    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD calculation."""
        close_prices = sample_ohlcv_data['close']
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close_prices)
        
        # Check that series have same length
        assert len(macd_line) == len(signal_line) == len(histogram) == len(close_prices)
        
        # Histogram should equal macd_line - signal_line
        valid_indices = ~(pd.isna(macd_line) | pd.isna(signal_line))
        if valid_indices.any():
            expected_histogram = macd_line[valid_indices] - signal_line[valid_indices]
            actual_histogram = histogram[valid_indices]
            assert np.allclose(actual_histogram, expected_histogram, rtol=1e-10)
    
    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        close_prices = sample_ohlcv_data['close']
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close_prices, 20)
        
        # Upper band should be above middle, middle above lower
        valid_indices = ~(pd.isna(bb_upper) | pd.isna(bb_middle) | pd.isna(bb_lower))
        if valid_indices.any():
            assert (bb_upper[valid_indices] >= bb_middle[valid_indices]).all()
            assert (bb_middle[valid_indices] >= bb_lower[valid_indices]).all()
        
        # Middle band should equal SMA
        sma_20 = TechnicalIndicators.sma(close_prices, 20)
        valid_sma = ~pd.isna(sma_20)
        if valid_sma.any():
            assert np.allclose(bb_middle[valid_sma], sma_20[valid_sma], rtol=1e-10)
    
    def test_atr_calculation(self, sample_ohlcv_data):
        """Test Average True Range calculation."""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']
        
        atr = TechnicalIndicators.atr(high, low, close, 14)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
        
        # Should have reasonable values relative to price range
        price_range = (high - low).mean()
        assert valid_atr.mean() <= price_range * 2  # Sanity check
    
    def test_adx_calculation(self, sample_ohlcv_data):
        """Test ADX calculation."""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']
        
        adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, 14)
        
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        if len(valid_adx) > 0:
            assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
        
        # DI values should be positive
        valid_plus_di = plus_di.dropna()
        valid_minus_di = minus_di.dropna()
        if len(valid_plus_di) > 0:
            assert (valid_plus_di >= 0).all()
        if len(valid_minus_di) > 0:
            assert (valid_minus_di >= 0).all()
    
    def test_keltner_channels(self, sample_ohlcv_data):
        """Test Keltner Channels calculation."""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']
        
        kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(high, low, close, 20)
        
        # Upper should be above middle, middle above lower
        valid_indices = ~(pd.isna(kc_upper) | pd.isna(kc_middle) | pd.isna(kc_lower))
        if valid_indices.any():
            assert (kc_upper[valid_indices] > kc_middle[valid_indices]).all()
            assert (kc_middle[valid_indices] > kc_lower[valid_indices]).all()
    
    def test_percent_above_ma(self, simple_price_series):
        """Test percentage above moving average calculation."""
        pct_above = TechnicalIndicators.percent_above_ma(simple_price_series, 3)
        
        # Should have reasonable values
        valid_pct = pct_above.dropna()
        assert len(valid_pct) > 0
        
        # For our upward trending test data, should be mostly positive
        assert valid_pct.mean() > 0
    
    def test_price_returns(self, simple_price_series):
        """Test price returns calculation."""
        returns_3 = TechnicalIndicators.price_returns(simple_price_series, 3)
        
        # Check specific calculation
        # Return from index 0 to 3: (103-100)/100 * 100 = 3%
        expected_return = ((103 - 100) / 100) * 100
        assert abs(returns_3.iloc[3] - expected_return) < 1e-10
    
    def test_new_high_flag(self, simple_price_series):
        """Test new high flag calculation."""
        new_highs = TechnicalIndicators.new_high_flag(simple_price_series, 5)
        
        # Should be boolean series
        assert new_highs.dtype == bool
        
        # Last value (109) should be a new high
        assert new_highs.iloc[-1] == True
    
    def test_volume_ratio(self, sample_ohlcv_data):
        """Test volume ratio calculation."""
        volume = sample_ohlcv_data['volume']
        vol_ratio = TechnicalIndicators.volume_ratio(volume, 20)
        
        # Should have positive values
        valid_ratio = vol_ratio.dropna()
        assert (valid_ratio > 0).all()
        
        # Mean should be around 1.0 for random data
        assert 0.5 < valid_ratio.mean() < 2.0
    
    def test_calculate_all_indicators(self, sample_ohlcv_data):
        """Test comprehensive indicator calculation."""
        indicators = TechnicalIndicators.calculate_all_indicators(sample_ohlcv_data)
        
        # Should return dictionary with expected keys
        expected_keys = [
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr_14', 'adx_14', 'volume_sma_20', 'volume_ratio'
        ]
        
        for key in expected_keys:
            assert key in indicators
        
        # Values should be numeric (or NaN for insufficient data)
        for key, value in indicators.items():
            if value is not None and not pd.isna(value):
                assert isinstance(value, (int, float, np.number))


class TestDataValidation:
    """Test data validation functions."""
    
    def test_valid_ohlcv_data(self, sample_ohlcv_data):
        """Test validation of good OHLCV data.""" 
        # This should not raise an exception
        assert validate_ohlcv_data(sample_ohlcv_data) == True
    
    def test_missing_columns(self):
        """Test validation with missing columns."""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'close': [101, 102]
            # Missing 'low' and 'volume'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_data(invalid_data)
    
    def test_insufficient_data(self):
        """Test validation with insufficient data points."""
        insufficient_data = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101],
            'volume': [1000]
        })
        
        with pytest.raises(ValueError, match="Need at least 2 data points"):
            validate_ohlcv_data(insufficient_data)
    
    def test_negative_prices(self):
        """Test validation with negative prices."""
        invalid_data = pd.DataFrame({
            'open': [100, -50],
            'high': [102, 103],
            'low': [99, 98],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        with pytest.raises(ValueError, match="non-positive values"):
            validate_ohlcv_data(invalid_data)
    
    def test_negative_volume(self):
        """Test validation with negative volume."""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 98],
            'close': [101, 102],
            'volume': [1000, -500]
        })
        
        with pytest.raises(ValueError, match="negative values"):
            validate_ohlcv_data(invalid_data)
    
    def test_invalid_ohlc_relationship(self):
        """Test validation with invalid OHLC relationships."""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [99, 103],  # High less than open
            'low': [99, 98],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        with pytest.raises(ValueError, match="outside high-low range"):
            validate_ohlcv_data(invalid_data)


@pytest.fixture
def sample_ohlcv_data():
    """Global fixture for sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0.001, 0.02, 100).cumsum()
    close_prices = base_price * (1 + price_changes)
    
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    volume = np.random.randint(100000, 1000000, 100)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
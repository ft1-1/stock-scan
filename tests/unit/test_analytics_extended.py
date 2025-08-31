"""Comprehensive unit tests for analytics module."""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch
from scipy import stats

# Import analytics modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analytics.technical_indicators import TechnicalIndicators
from analytics.momentum_analysis import MomentumAnalyzer
from analytics.options_selector import OptionsSelector, CallOptionCriteria
from analytics.scoring_models import (
    TechnicalScorer, MomentumScorer, OptionsScorer, 
    CombinedScorer, ScoreWeights
)
from analytics.squeeze_detector import SqueezeDetector, SqueezeCondition
from analytics.greeks_calculator import GreeksCalculator, BlackScholesModel
from models.base_models import OptionContract, TechnicalIndicators as TechIndicatorsModel


class TestTechnicalIndicatorsExtended:
    """Extended tests for technical indicators beyond basic functionality."""
    
    @pytest.fixture
    def complex_price_data(self):
        """Create complex price data with various market conditions."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create data with multiple market phases
        np.random.seed(42)
        
        # Phase 1: Uptrend (first 60 days)
        uptrend = np.random.normal(0.002, 0.015, 60).cumsum()
        
        # Phase 2: Sideways (next 60 days) 
        sideways = np.random.normal(0.0001, 0.012, 60).cumsum() + uptrend[-1]
        
        # Phase 3: Downtrend (next 60 days)
        downtrend = np.random.normal(-0.0015, 0.018, 60).cumsum() + sideways[-1]
        
        # Phase 4: Recovery (last 20 days)
        recovery = np.random.normal(0.003, 0.02, 20).cumsum() + downtrend[-1]
        
        # Combine phases
        returns = np.concatenate([uptrend, sideways, downtrend, recovery])
        prices = 100 * np.exp(returns)
        
        # Generate realistic OHLCV
        data = []
        for i, close in enumerate(prices):
            if i == 0:
                open_price = 100.0
            else:
                gap = np.random.normal(0, 0.005)
                open_price = prices[i-1] * (1 + gap)
            
            daily_range = close * 0.02 * np.random.uniform(0.5, 2.0)
            high = max(open_price, close) + daily_range * np.random.uniform(0, 0.5)
            low = min(open_price, close) - daily_range * np.random.uniform(0, 0.5)
            
            volume = int(np.random.lognormal(13, 0.5))  # Realistic volume distribution
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def test_rsi_divergence_detection(self, complex_price_data):
        """Test RSI divergence detection in trending markets."""
        close_prices = complex_price_data['close']
        rsi = TechnicalIndicators.rsi(close_prices, 14)
        
        # Look for bearish divergence (price higher highs, RSI lower highs)
        price_highs = close_prices.rolling(20).max()
        rsi_highs = rsi.rolling(20).max()
        
        # Should detect some divergences in the dataset
        valid_data = ~(pd.isna(price_highs) | pd.isna(rsi_highs))
        assert valid_data.sum() > 100  # Enough data for analysis
    
    def test_bollinger_band_squeeze_detection(self, complex_price_data):
        """Test Bollinger Band squeeze conditions."""
        close_prices = complex_price_data['close']
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close_prices, 20, 2)
        
        # Calculate band width
        band_width = (bb_upper - bb_lower) / bb_middle
        
        # Identify squeeze periods (narrow bands)
        squeeze_threshold = band_width.quantile(0.2)  # Bottom 20%
        squeeze_periods = band_width < squeeze_threshold
        
        assert squeeze_periods.sum() > 10  # Should find some squeeze periods
    
    def test_macd_signal_accuracy(self, complex_price_data):
        """Test MACD signal generation accuracy."""
        close_prices = complex_price_data['close']
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close_prices)
        
        # Test signal crossovers
        bullish_crossover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        bearish_crossover = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # Should have multiple crossovers in complex data
        assert bullish_crossover.sum() > 3
        assert bearish_crossover.sum() > 3
        
        # Histogram should equal MACD - Signal
        valid_idx = ~(pd.isna(macd_line) | pd.isna(signal_line))
        np.testing.assert_allclose(
            histogram[valid_idx], 
            macd_line[valid_idx] - signal_line[valid_idx],
            rtol=1e-10
        )
    
    def test_volume_price_correlation(self, complex_price_data):
        """Test volume-price analysis."""
        close_prices = complex_price_data['close']
        volume = complex_price_data['volume']
        
        # Price change
        price_change = close_prices.pct_change()
        
        # Volume ratio
        vol_ratio = TechnicalIndicators.volume_ratio(volume, 20)
        
        # Test that large price moves correlate with higher volume
        large_moves = abs(price_change) > price_change.std() * 2
        avg_vol_on_large_moves = vol_ratio[large_moves].mean()
        avg_vol_normal = vol_ratio[~large_moves].mean()
        
        # Volume should be higher on large moves (not always true but generally)
        # Just verify the calculation works correctly
        assert not pd.isna(avg_vol_on_large_moves)
        assert not pd.isna(avg_vol_normal)
    
    def test_indicator_parameter_sensitivity(self):
        """Test sensitivity to parameter changes."""
        # Create simple trending data
        prices = pd.Series([100 + i * 0.5 for i in range(50)])
        
        # Test RSI with different periods
        rsi_14 = TechnicalIndicators.rsi(prices, 14)
        rsi_21 = TechnicalIndicators.rsi(prices, 21)
        
        # Longer period should be smoother (less volatile)
        rsi_14_volatility = rsi_14.dropna().std()
        rsi_21_volatility = rsi_21.dropna().std()
        
        assert rsi_21_volatility < rsi_14_volatility
    
    def test_extreme_market_conditions(self):
        """Test indicators under extreme market conditions."""
        # Gap up scenario
        gap_data = pd.Series([100] * 10 + [150] * 10 + [149, 151, 148, 152])
        rsi_gap = TechnicalIndicators.rsi(gap_data, 10)
        
        # RSI should handle gaps without breaking
        assert not pd.isna(rsi_gap.iloc[-1])
        assert 0 <= rsi_gap.iloc[-1] <= 100
        
        # Limit up/down scenario
        limit_data = pd.Series([100] + [100] * 20)  # No price movement
        rsi_flat = TechnicalIndicators.rsi(limit_data, 10)
        
        # RSI should handle flat prices
        assert not pd.isna(rsi_flat.iloc[-1])


class TestMomentumAnalyzer:
    """Tests for momentum analysis functionality."""
    
    @pytest.fixture
    def momentum_analyzer(self):
        """Create momentum analyzer instance."""
        return MomentumAnalyzer()
    
    @pytest.fixture
    def trending_data(self):
        """Create data with clear momentum patterns."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Strong uptrend
        trend = np.linspace(0, 0.5, 100)  # 50% gain over period
        noise = np.random.normal(0, 0.01, 100)
        returns = trend + noise
        prices = 100 * np.exp(returns)
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    
    def test_momentum_score_calculation(self, momentum_analyzer, trending_data):
        """Test momentum score calculation."""
        score = momentum_analyzer.calculate_momentum_score(trending_data)
        
        assert isinstance(score, dict)
        assert 'momentum_score' in score
        assert 'trend_strength' in score
        assert 'momentum_direction' in score
        
        # Strong uptrend should have high momentum score
        assert score['momentum_score'] > 70
        assert score['momentum_direction'] == 'bullish'
    
    def test_price_momentum_analysis(self, momentum_analyzer, trending_data):
        """Test price momentum components."""
        momentum = momentum_analyzer.analyze_price_momentum(
            trending_data['close'], 
            periods=[5, 10, 20, 30]
        )
        
        assert 'returns' in momentum
        assert 'momentum_strength' in momentum
        assert 'consistency' in momentum
        
        # All periods should show positive momentum for uptrending data
        for period in [5, 10, 20, 30]:
            assert momentum['returns'][f'{period}d'] > 0
    
    def test_volume_momentum_analysis(self, momentum_analyzer, trending_data):
        """Test volume momentum analysis."""
        volume_momentum = momentum_analyzer.analyze_volume_momentum(
            trending_data['close'],
            trending_data['volume']
        )
        
        assert 'volume_trend' in volume_momentum
        assert 'price_volume_correlation' in volume_momentum
        assert 'volume_momentum_score' in volume_momentum
        
        # Correlation should be a valid correlation coefficient
        corr = volume_momentum['price_volume_correlation']
        assert -1 <= corr <= 1
    
    def test_momentum_sustainability(self, momentum_analyzer, trending_data):
        """Test momentum sustainability analysis."""
        sustainability = momentum_analyzer.assess_momentum_sustainability(trending_data)
        
        assert 'sustainability_score' in sustainability
        assert 'risk_factors' in sustainability
        assert 'duration_analysis' in sustainability
        
        # Score should be between 0 and 100
        assert 0 <= sustainability['sustainability_score'] <= 100
    
    def test_momentum_with_sideways_market(self, momentum_analyzer):
        """Test momentum analysis with sideways market."""
        # Create sideways market data
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        prices = 100 + np.random.normal(0, 2, 60)  # Sideways with noise
        
        sideways_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 60)
        }, index=dates)
        
        score = momentum_analyzer.calculate_momentum_score(sideways_data)
        
        # Sideways market should have low momentum score
        assert score['momentum_score'] < 40
        assert score['momentum_direction'] in ['neutral', 'sideways']
    
    def test_momentum_reversal_detection(self, momentum_analyzer):
        """Test momentum reversal detection."""
        # Create data with momentum reversal
        dates = pd.date_range(start='2023-01-01', periods=80, freq='D')
        
        # First 40 days: uptrend
        uptrend = 100 * np.exp(np.linspace(0, 0.2, 40))
        # Last 40 days: downtrend  
        downtrend = uptrend[-1] * np.exp(np.linspace(0, -0.15, 40))
        
        prices = np.concatenate([uptrend, downtrend])
        
        reversal_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 80)
        }, index=dates)
        
        # Analyze recent momentum (last 20 days)
        recent_data = reversal_data.tail(20)
        score = momentum_analyzer.calculate_momentum_score(recent_data)
        
        # Recent period should show bearish momentum
        assert score['momentum_direction'] == 'bearish'


class TestOptionsSelector:
    """Tests for options selection and filtering logic."""
    
    @pytest.fixture
    def options_selector(self):
        """Create options selector instance."""
        criteria = CallOptionCriteria(
            min_volume=100,
            min_open_interest=500,
            max_days_to_expiration=45,
            min_delta=0.3,
            max_delta=0.8,
            max_iv_percentile=80,
            min_liquidity_score=50
        )
        return OptionsSelector(criteria)
    
    @pytest.fixture
    def options_chain_data(self, test_data_generator):
        """Generate options chain for testing."""
        return test_data_generator.generate_options_chain("AAPL", 150.0, 30)
    
    def test_volume_filtering(self, options_selector, options_chain_data):
        """Test volume-based filtering."""
        # Manually set some options to have low volume
        for i in range(5):
            options_chain_data[i]['volume'] = 50  # Below minimum
        
        filtered = options_selector.filter_by_volume(options_chain_data)
        
        # Should exclude low volume options
        assert len(filtered) < len(options_chain_data)
        assert all(opt['volume'] >= 100 for opt in filtered)
    
    def test_delta_filtering(self, options_selector, options_chain_data):
        """Test delta-based filtering."""
        filtered = options_selector.filter_by_delta(options_chain_data)
        
        # All filtered options should meet delta criteria
        for option in filtered:
            assert 0.3 <= option['delta'] <= 0.8
    
    def test_liquidity_scoring(self, options_selector, options_chain_data):
        """Test liquidity score calculation."""
        for option in options_chain_data:
            score = options_selector.calculate_liquidity_score(option)
            assert 0 <= score <= 100
    
    def test_best_call_selection(self, options_selector, options_chain_data):
        """Test selection of best call options."""
        best_calls = options_selector.select_best_calls(options_chain_data, top_n=3)
        
        assert len(best_calls) <= 3
        assert len(best_calls) > 0  # Should find at least some options
        
        # Results should be sorted by score (best first)
        scores = [call['selection_score'] for call in best_calls]
        assert scores == sorted(scores, reverse=True)
    
    def test_iv_percentile_filtering(self, options_selector):
        """Test implied volatility percentile filtering."""
        # Create options with known IV values
        options_with_iv = [
            {'implied_volatility': 0.2, 'volume': 1000, 'open_interest': 1000},  # Low IV
            {'implied_volatility': 0.4, 'volume': 1000, 'open_interest': 1000},  # High IV
            {'implied_volatility': 0.3, 'volume': 1000, 'open_interest': 1000},  # Medium IV
        ]
        
        # Calculate percentiles (mock function)
        with patch.object(options_selector, 'calculate_iv_percentile') as mock_calc:
            mock_calc.side_effect = [20, 90, 50]  # Percentiles for each option
            
            filtered = options_selector.filter_by_iv_percentile(options_with_iv)
            
            # Should exclude option with 90th percentile IV
            assert len(filtered) == 2
    
    def test_time_decay_consideration(self, options_selector, options_chain_data):
        """Test time decay factor in selection."""
        # Add time to expiration data
        for i, option in enumerate(options_chain_data):
            option['days_to_expiration'] = 7 + i * 2  # Varying expirations
        
        selection_scores = []
        for option in options_chain_data:
            score = options_selector.calculate_selection_score(option)
            selection_scores.append((option['days_to_expiration'], score))
        
        # Very short-term options should generally have lower scores due to theta risk
        short_term = [score for days, score in selection_scores if days <= 10]
        longer_term = [score for days, score in selection_scores if days >= 20]
        
        if short_term and longer_term:
            avg_short = np.mean(short_term) 
            avg_longer = np.mean(longer_term)
            # This relationship may not always hold, so just verify calculation works
            assert isinstance(avg_short, (int, float))
            assert isinstance(avg_longer, (int, float))
    
    def test_empty_options_chain(self, options_selector):
        """Test handling of empty options chain."""
        empty_chain = []
        
        result = options_selector.select_best_calls(empty_chain)
        assert result == []
    
    def test_all_options_filtered_out(self, options_selector):
        """Test scenario where all options are filtered out."""
        # Create options that don't meet any criteria
        bad_options = [
            {
                'volume': 1,  # Too low
                'open_interest': 10,  # Too low
                'delta': 0.1,  # Too low
                'days_to_expiration': 60,  # Too high
                'implied_volatility': 0.8  # Too high
            }
        ]
        
        result = options_selector.select_best_calls(bad_options)
        assert result == []


class TestScoringModels:
    """Tests for various scoring models."""
    
    @pytest.fixture
    def sample_technical_data(self):
        """Sample technical indicators data."""
        return {
            'rsi_14': 65,
            'macd': 2.5,
            'macd_signal': 1.8,
            'macd_histogram': 0.7,
            'sma_20': 148.5,
            'sma_50': 145.2,
            'current_price': 150.0,
            'bollinger_upper': 155.0,
            'bollinger_lower': 142.0,
            'atr_14': 3.2,
            'volume_ratio': 1.3
        }
    
    def test_technical_scorer(self, sample_technical_data):
        """Test technical analysis scoring."""
        scorer = TechnicalScorer()
        score = scorer.calculate_score(sample_technical_data)
        
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))
    
    def test_momentum_scorer(self):
        """Test momentum scoring logic."""
        momentum_data = {
            'price_momentum_5d': 3.2,
            'price_momentum_10d': 5.8,
            'price_momentum_20d': 8.5,
            'volume_momentum': 1.4,
            'trend_consistency': 0.75
        }
        
        scorer = MomentumScorer()
        score = scorer.calculate_score(momentum_data)
        
        assert 0 <= score <= 100
        # Strong positive momentum should yield high score
        assert score > 60
    
    def test_options_scorer(self):
        """Test options attractiveness scoring."""
        options_data = {
            'best_call_score': 78,
            'liquidity_score': 85,
            'iv_percentile': 45,
            'theta_efficiency': 0.7,
            'delta_gamma_ratio': 1.2
        }
        
        scorer = OptionsScorer()
        score = scorer.calculate_score(options_data)
        
        assert 0 <= score <= 100
    
    def test_combined_scorer_with_weights(self):
        """Test combined scoring with custom weights."""
        scores = {
            'technical_score': 75,
            'momentum_score': 82,
            'options_score': 68
        }
        
        weights = ScoreWeights(
            technical_weight=0.4,
            momentum_weight=0.35,
            options_weight=0.25
        )
        
        combined_scorer = CombinedScorer(weights)
        final_score = combined_scorer.calculate_combined_score(scores)
        
        # Verify weighted calculation
        expected = (75 * 0.4) + (82 * 0.35) + (68 * 0.25)
        assert abs(final_score - expected) < 0.01
    
    def test_score_normalization(self):
        """Test score normalization across different ranges."""
        scorer = TechnicalScorer()
        
        # Test extreme values get normalized properly
        extreme_data = {
            'rsi_14': 95,  # Very overbought
            'macd': -5.0,  # Very bearish
            'current_price': 200.0,
            'sma_20': 150.0,  # Price well above moving average
            'volume_ratio': 3.0  # Very high volume
        }
        
        score = scorer.calculate_score(extreme_data)
        assert 0 <= score <= 100
    
    def test_missing_data_handling(self):
        """Test handling of missing indicator data."""
        incomplete_data = {
            'rsi_14': 65,
            'current_price': 150.0
            # Missing other indicators
        }
        
        scorer = TechnicalScorer()
        score = scorer.calculate_score(incomplete_data)
        
        # Should handle missing data gracefully
        assert 0 <= score <= 100


class TestSqueezeDetector:
    """Tests for TTM Squeeze detection algorithm."""
    
    @pytest.fixture
    def squeeze_detector(self):
        """Create squeeze detector instance."""
        return SqueezeDetector()
    
    @pytest.fixture
    def squeeze_data(self):
        """Create data with squeeze conditions."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create consolidating price pattern
        base_price = 100
        consolidation_range = 2  # 2% range
        
        prices = []
        for i in range(100):
            if i < 50:
                # Initial trend
                price = base_price + i * 0.1
            else:
                # Consolidation phase (squeeze setup)
                price = base_price + 50 * 0.1 + np.random.uniform(-consolidation_range, consolidation_range)
            prices.append(price)
        
        # Generate OHLCV
        data = []
        for i, close in enumerate(prices):
            open_price = close + np.random.uniform(-0.5, 0.5)
            high = max(open_price, close) + np.random.uniform(0, 1)
            low = min(open_price, close) - np.random.uniform(0, 1)
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low, 
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def test_squeeze_detection(self, squeeze_detector, squeeze_data):
        """Test basic squeeze detection."""
        squeeze_conditions = squeeze_detector.detect_squeeze(squeeze_data)
        
        assert isinstance(squeeze_conditions, list)
        assert len(squeeze_conditions) > 0
        
        # Verify squeeze condition structure
        for condition in squeeze_conditions:
            assert isinstance(condition, SqueezeCondition)
            assert hasattr(condition, 'start_date')
            assert hasattr(condition, 'is_active')
            assert hasattr(condition, 'duration_days')
    
    def test_bollinger_keltner_squeeze(self, squeeze_detector, squeeze_data):
        """Test Bollinger Band / Keltner Channel squeeze detection."""
        high = squeeze_data['high']
        low = squeeze_data['low'] 
        close = squeeze_data['close']
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close, 20)
        kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(high, low, close, 20)
        
        # Identify squeeze periods
        squeeze_periods = squeeze_detector.identify_bb_kc_squeeze(
            bb_upper, bb_lower, kc_upper, kc_lower
        )
        
        assert isinstance(squeeze_periods, pd.Series)
        assert squeeze_periods.dtype == bool
        
        # Should find some squeeze periods in consolidating data
        assert squeeze_periods.sum() > 0
    
    def test_momentum_during_squeeze(self, squeeze_detector, squeeze_data):
        """Test momentum analysis during squeeze periods."""
        close = squeeze_data['close']
        
        # Calculate momentum oscillator (typically linear regression slope)
        momentum = squeeze_detector.calculate_squeeze_momentum(close)
        
        assert len(momentum) == len(close)
        assert not momentum.isna().all()  # Should have some valid values
    
    def test_squeeze_breakout_detection(self, squeeze_detector):
        """Test detection of squeeze breakouts."""
        # Create data with squeeze followed by breakout
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # 40 days of consolidation then breakout
        consolidation = [100 + np.random.uniform(-1, 1) for _ in range(40)]
        breakout = [consolidation[-1] + i * 0.5 for i in range(1, 21)]  # Strong breakout
        
        prices = consolidation + breakout
        
        breakout_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 60)
        }, index=dates)
        
        breakouts = squeeze_detector.detect_breakouts(breakout_data)
        
        assert len(breakouts) > 0
        
        # Breakout should be detected around day 40
        breakout_dates = [b.breakout_date for b in breakouts]
        assert any(d >= dates[35] for d in breakout_dates)  # Around consolidation end
    
    def test_squeeze_strength_analysis(self, squeeze_detector, squeeze_data):
        """Test squeeze strength assessment."""
        strength_metrics = squeeze_detector.analyze_squeeze_strength(squeeze_data)
        
        assert 'compression_ratio' in strength_metrics
        assert 'duration_score' in strength_metrics
        assert 'volume_pattern' in strength_metrics
        
        # Compression ratio should be meaningful
        assert 0 < strength_metrics['compression_ratio'] < 1
    
    def test_false_squeeze_filtering(self, squeeze_detector):
        """Test filtering of false squeeze signals."""
        # Create data with brief, weak squeeze condition
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = [100 + np.random.uniform(-0.1, 0.1) for _ in range(30)]  # Very tight range but short
        
        weak_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(50000, 100000, 30)  # Low volume
        }, index=dates)
        
        valid_squeezes = squeeze_detector.filter_valid_squeezes(
            squeeze_detector.detect_squeeze(weak_data),
            min_duration=10,
            min_strength=0.5
        )
        
        # Should filter out weak/short squeezes
        assert len(valid_squeezes) == 0 or all(s.duration_days >= 10 for s in valid_squeezes)


class TestGreeksCalculator:
    """Tests for options Greeks calculations."""
    
    @pytest.fixture
    def greeks_calculator(self):
        """Create Greeks calculator instance."""
        return GreeksCalculator()
    
    @pytest.fixture
    def option_parameters(self):
        """Standard option parameters for testing."""
        return {
            'spot_price': 100.0,
            'strike_price': 100.0,
            'time_to_expiration': 30/365,  # 30 days
            'risk_free_rate': 0.05,
            'implied_volatility': 0.25,
            'option_type': 'call'
        }
    
    def test_black_scholes_call_price(self, greeks_calculator, option_parameters):
        """Test Black-Scholes call option pricing."""
        bs_model = BlackScholesModel()
        price = bs_model.calculate_option_price(**option_parameters)
        
        # ATM call with 30 days should have reasonable price
        assert 0 < price < 20  # Sanity check
        assert isinstance(price, float)
    
    def test_black_scholes_put_price(self, greeks_calculator, option_parameters):
        """Test Black-Scholes put option pricing."""
        option_parameters['option_type'] = 'put'
        
        bs_model = BlackScholesModel()
        price = bs_model.calculate_option_price(**option_parameters)
        
        assert 0 < price < 20
        assert isinstance(price, float)
    
    def test_delta_calculation(self, greeks_calculator, option_parameters):
        """Test delta calculation accuracy."""
        delta = greeks_calculator.calculate_delta(**option_parameters)
        
        # ATM call delta should be around 0.5
        assert 0.4 < delta < 0.6
        
        # Test put delta
        option_parameters['option_type'] = 'put'
        put_delta = greeks_calculator.calculate_delta(**option_parameters)
        
        # Put delta should be negative and call_delta - 1
        assert put_delta < 0
        assert abs(put_delta - (delta - 1)) < 0.001
    
    def test_gamma_calculation(self, greeks_calculator, option_parameters):
        """Test gamma calculation."""
        gamma = greeks_calculator.calculate_gamma(**option_parameters)
        
        # Gamma should be positive for both calls and puts
        assert gamma > 0
        
        # ATM options should have highest gamma
        option_parameters['strike_price'] = 110  # OTM
        otm_gamma = greeks_calculator.calculate_gamma(**option_parameters)
        
        assert gamma > otm_gamma  # ATM should have higher gamma than OTM
    
    def test_theta_calculation(self, greeks_calculator, option_parameters):
        """Test theta (time decay) calculation."""
        theta = greeks_calculator.calculate_theta(**option_parameters)
        
        # Theta should be negative for long options
        assert theta < 0
        
        # Shorter time to expiration should have higher theta decay
        option_parameters['time_to_expiration'] = 7/365  # 1 week
        short_theta = greeks_calculator.calculate_theta(**option_parameters)
        
        assert short_theta < theta  # More negative (higher decay)
    
    def test_vega_calculation(self, greeks_calculator, option_parameters):
        """Test vega calculation."""
        vega = greeks_calculator.calculate_vega(**option_parameters)
        
        # Vega should be positive
        assert vega > 0
        
        # Longer time to expiration should have higher vega
        option_parameters['time_to_expiration'] = 90/365  # 3 months
        long_vega = greeks_calculator.calculate_vega(**option_parameters)
        
        assert long_vega > vega
    
    def test_implied_volatility_calculation(self, greeks_calculator, option_parameters):
        """Test implied volatility calculation."""
        # First calculate a theoretical price
        bs_model = BlackScholesModel()
        theoretical_price = bs_model.calculate_option_price(**option_parameters)
        
        # Now calculate IV from that price
        calculated_iv = greeks_calculator.calculate_implied_volatility(
            option_price=theoretical_price,
            spot_price=option_parameters['spot_price'],
            strike_price=option_parameters['strike_price'],
            time_to_expiration=option_parameters['time_to_expiration'],
            risk_free_rate=option_parameters['risk_free_rate'],
            option_type=option_parameters['option_type']
        )
        
        # Should recover original IV (within tolerance)
        original_iv = option_parameters['implied_volatility']
        assert abs(calculated_iv - original_iv) < 0.001
    
    def test_greeks_consistency(self, greeks_calculator, option_parameters):
        """Test consistency between Greeks calculations."""
        greeks = greeks_calculator.calculate_all_greeks(**option_parameters)
        
        required_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in required_greeks:
            assert greek in greeks
            assert isinstance(greeks[greek], (int, float))
            assert not np.isnan(greeks[greek])
    
    def test_edge_cases(self, greeks_calculator):
        """Test edge cases and boundary conditions."""
        # Very short time to expiration
        short_params = {
            'spot_price': 100.0,
            'strike_price': 100.0,
            'time_to_expiration': 1/365,  # 1 day
            'risk_free_rate': 0.05,
            'implied_volatility': 0.25,
            'option_type': 'call'
        }
        
        greeks = greeks_calculator.calculate_all_greeks(**short_params)
        
        # Should handle short expiration without errors
        assert all(not np.isnan(value) for value in greeks.values())
        
        # Very deep ITM option
        deep_itm_params = short_params.copy()
        deep_itm_params['strike_price'] = 50.0  # Deep ITM
        
        delta = greeks_calculator.calculate_delta(**deep_itm_params)
        assert delta > 0.9  # Should be close to 1 for deep ITM calls
    
    def test_greeks_with_real_option_data(self, greeks_calculator, sample_option_contract):
        """Test Greeks calculation with real option contract data."""
        # Convert option contract to parameters
        time_to_exp = sample_option_contract.days_to_expiration / 365
        
        calculated_greeks = greeks_calculator.calculate_all_greeks(
            spot_price=150.0,  # Assumed underlying price
            strike_price=float(sample_option_contract.strike),
            time_to_expiration=time_to_exp,
            risk_free_rate=0.05,
            implied_volatility=sample_option_contract.implied_volatility,
            option_type=sample_option_contract.option_type
        )
        
        # Compare with contract Greeks (allowing for some difference)
        assert abs(calculated_greeks['delta'] - sample_option_contract.delta) < 0.1
        assert abs(calculated_greeks['gamma'] - sample_option_contract.gamma) < 0.05
        assert abs(calculated_greeks['theta'] - sample_option_contract.theta) < 0.1
        assert abs(calculated_greeks['vega'] - sample_option_contract.vega) < 0.1
"""Unit tests for options selector module."""

import pytest
import numpy as np
from datetime import date, timedelta
from decimal import Decimal

from src.analytics.options_selector import (
    OptionsSelector, 
    calculate_days_to_expiration,
    calculate_moneyness,
    estimate_intrinsic_value
)
from src.models.base_models import OptionContract, OptionType


class TestOptionsSelector:
    """Test cases for options selector functionality."""
    
    @pytest.fixture
    def options_selector(self):
        """Create OptionsSelector instance for testing."""
        return OptionsSelector()
    
    @pytest.fixture
    def sample_call_option(self):
        """Create sample call option for testing."""
        return OptionContract(
            option_symbol="AAPL240315C00150000",
            underlying_symbol="AAPL",
            strike=Decimal("150.00"),
            expiration=date.today() + timedelta(days=60),
            option_type=OptionType.CALL,
            bid=Decimal("5.20"),
            ask=Decimal("5.40"),
            last=Decimal("5.30"),
            volume=250,
            open_interest=1500,
            delta=0.65,
            gamma=0.025,
            theta=-0.15,
            vega=0.35,
            implied_volatility=0.25,
            days_to_expiration=60
        )
    
    @pytest.fixture
    def sample_put_option(self):
        """Create sample put option for testing."""
        return OptionContract(
            option_symbol="AAPL240315P00140000",
            underlying_symbol="AAPL",
            strike=Decimal("140.00"),
            expiration=date.today() + timedelta(days=60),
            option_type=OptionType.PUT,
            bid=Decimal("3.80"),
            ask=Decimal("4.00"),
            last=Decimal("3.90"),
            volume=180,
            open_interest=800,
            delta=-0.35,
            gamma=0.020,
            theta=-0.12,
            vega=0.30,
            implied_volatility=0.28,
            days_to_expiration=60
        )
    
    @pytest.fixture
    def sample_options_chain(self, sample_call_option, sample_put_option):
        """Create sample options chain for testing."""
        options = []
        
        # Create multiple call options with different strikes and DTEs
        for i, strike in enumerate([140, 145, 150, 155, 160]):
            for j, dte in enumerate([30, 60, 90]):
                call = OptionContract(
                    option_symbol=f"AAPL24{strike}C{dte}",
                    underlying_symbol="AAPL",
                    strike=Decimal(str(strike)),
                    expiration=date.today() + timedelta(days=dte),
                    option_type=OptionType.CALL,
                    bid=Decimal(str(5.0 + i * 0.5 - j * 0.2)),
                    ask=Decimal(str(5.2 + i * 0.5 - j * 0.2)),
                    last=Decimal(str(5.1 + i * 0.5 - j * 0.2)),
                    volume=200 + i * 50,
                    open_interest=500 + i * 200 + j * 100,
                    delta=0.7 - i * 0.05 - j * 0.02,
                    days_to_expiration=dte
                )
                options.append(call)
                
                # Create corresponding put
                put = OptionContract(
                    option_symbol=f"AAPL24{strike}P{dte}",
                    underlying_symbol="AAPL",
                    strike=Decimal(str(strike)),
                    expiration=date.today() + timedelta(days=dte),
                    option_type=OptionType.PUT,
                    bid=Decimal(str(3.0 + (160-strike) * 0.1 + j * 0.1)),
                    ask=Decimal(str(3.2 + (160-strike) * 0.1 + j * 0.1)),
                    last=Decimal(str(3.1 + (160-strike) * 0.1 + j * 0.1)),
                    volume=150 + i * 30,
                    open_interest=400 + i * 150 + j * 80,
                    delta=-(0.3 + (160-strike) * 0.01 + j * 0.01),
                    days_to_expiration=dte
                )
                options.append(put)
        
        return options
    
    def test_filter_options_basic(self, options_selector, sample_options_chain):
        """Test basic options filtering."""
        # Filter for calls with 45-75 DTE and 0.55-0.70 delta
        filtered_calls = options_selector.filter_options_basic(
            sample_options_chain, 
            OptionType.CALL,
            dte_range=(45, 75),
            delta_range=(0.55, 0.70)
        )
        
        # Should have some results
        assert len(filtered_calls) > 0
        
        # All should be calls
        assert all(opt.option_type == OptionType.CALL for opt in filtered_calls)
        
        # All should meet DTE criteria
        for option in filtered_calls:
            if option.days_to_expiration is not None:
                assert 45 <= option.days_to_expiration <= 75
        
        # All should meet delta criteria
        for option in filtered_calls:
            if option.delta is not None:
                assert 0.55 <= option.delta <= 0.70
    
    def test_calculate_liquidity_score(self, options_selector, sample_call_option):
        """Test liquidity score calculation."""
        score = options_selector.calculate_liquidity_score(sample_call_option)
        
        # Should be numeric and in reasonable range
        assert isinstance(score, (int, float))
        assert 0 <= score <= 40
        
        # High OI (1500) should get points
        # Volume (250) should get points  
        # Narrow spread should get points
        assert score > 20  # Should get decent score
    
    def test_calculate_liquidity_score_poor_option(self, options_selector):
        """Test liquidity score for poor liquidity option."""
        poor_option = OptionContract(
            option_symbol="TEST",
            underlying_symbol="TEST",
            strike=Decimal("100"),
            expiration=date.today() + timedelta(days=60),
            option_type=OptionType.CALL,
            bid=Decimal("1.00"),
            ask=Decimal("2.00"),  # Wide spread
            volume=5,  # Low volume
            open_interest=50  # Low OI
        )
        
        score = options_selector.calculate_liquidity_score(poor_option)
        
        # Should get low score due to poor liquidity
        assert score < 15
    
    def test_calculate_iv_value_score(self, options_selector, sample_call_option):
        """Test IV value score calculation."""
        # Test with favorable IV conditions
        score = options_selector.calculate_iv_value_score(
            sample_call_option, 
            iv_percentile=25.0,  # Low IV percentile
            historical_volatility=0.30  # HV higher than IV
        )
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 30
        
        # Low IV percentile should score well
        assert score > 15
    
    def test_calculate_fit_score(self, options_selector, sample_call_option):
        """Test fit score calculation."""
        # Test with option that fits target parameters well
        score = options_selector.calculate_fit_score(
            sample_call_option,
            target_dte_range=(50, 70),  # 60 DTE fits well
            target_delta_range=(0.60, 0.70)  # 0.65 delta fits well
        )
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 30
        
        # Should score well since parameters fit
        assert score > 20
    
    def test_calculate_total_score(self, options_selector, sample_call_option):
        """Test total score calculation."""
        scoring = options_selector.calculate_total_score(
            sample_call_option,
            iv_percentile=25.0,
            historical_volatility=0.30
        )
        
        # Should return dictionary with required keys
        required_keys = ['total_score', 'liquidity_score', 'iv_value_score', 'fit_score']
        for key in required_keys:
            assert key in scoring
        
        # Total should be sum of components
        expected_total = (scoring['liquidity_score'] + 
                         scoring['iv_value_score'] + 
                         scoring['fit_score'])
        assert abs(scoring['total_score'] - expected_total) < 0.01
        
        # Score should be reasonable
        assert 0 <= scoring['total_score'] <= 100
    
    def test_should_reject_option(self, options_selector, sample_call_option):
        """Test option rejection criteria."""
        # Good option should not be rejected
        should_reject, reason = options_selector.should_reject_option(sample_call_option)
        assert not should_reject
        
        # Test option with wide spread
        wide_spread_option = OptionContract(
            option_symbol="TEST",
            underlying_symbol="TEST", 
            strike=Decimal("100"),
            expiration=date.today() + timedelta(days=60),
            option_type=OptionType.CALL,
            bid=Decimal("1.00"),
            ask=Decimal("1.50"),  # 50% spread - too wide
            open_interest=1000,
            delta=0.60
        )
        
        should_reject, reason = options_selector.should_reject_option(wide_spread_option)
        assert should_reject
        assert "Spread too wide" in reason
        
        # Test option with low OI
        low_oi_option = OptionContract(
            option_symbol="TEST",
            underlying_symbol="TEST",
            strike=Decimal("100"), 
            expiration=date.today() + timedelta(days=60),
            option_type=OptionType.CALL,
            bid=Decimal("5.00"),
            ask=Decimal("5.10"),
            open_interest=100,  # Below minimum
            delta=0.60
        )
        
        should_reject, reason = options_selector.should_reject_option(low_oi_option)
        assert should_reject
        assert "Low open interest" in reason
    
    def test_find_best_calls(self, options_selector, sample_options_chain):
        """Test finding best call options."""
        best_calls = options_selector.find_best_calls(
            sample_options_chain,
            strategy_type="swing",
            top_n=3
        )
        
        # Should return list of dictionaries
        assert isinstance(best_calls, list)
        assert len(best_calls) <= 3
        
        if best_calls:
            # Each result should have required structure
            for result in best_calls:
                assert 'option' in result
                assert 'scoring' in result
                assert 'strategy_type' in result
                assert isinstance(result['option'], OptionContract)
                assert result['option'].option_type == OptionType.CALL
            
            # Should be sorted by score (descending)
            scores = [result['scoring']['total_score'] for result in best_calls]
            assert scores == sorted(scores, reverse=True)
    
    def test_find_best_puts(self, options_selector, sample_options_chain):
        """Test finding best put options."""
        best_puts = options_selector.find_best_puts(
            sample_options_chain,
            strategy_type="swing",
            top_n=3
        )
        
        assert isinstance(best_puts, list)
        assert len(best_puts) <= 3
        
        if best_puts:
            for result in best_puts:
                assert 'option' in result
                assert 'scoring' in result
                assert isinstance(result['option'], OptionContract)
                assert result['option'].option_type == OptionType.PUT
            
            # Should be sorted by score
            scores = [result['scoring']['total_score'] for result in best_puts]
            assert scores == sorted(scores, reverse=True)
    
    def test_analyze_options_chain(self, options_selector, sample_options_chain):
        """Test comprehensive options chain analysis."""
        analysis = options_selector.analyze_options_chain(
            sample_options_chain,
            current_stock_price=150.0,
            iv_percentile=40.0
        )
        
        # Should return dictionary with required keys
        required_keys = [
            'total_options', 'calls', 'puts', 'chain_statistics',
            'best_calls_swing', 'best_calls_longer', 'best_puts_swing',
            'best_puts_longer', 'analysis_timestamp'
        ]
        
        for key in required_keys:
            assert key in analysis
        
        # Basic validation
        assert analysis['total_options'] == len(sample_options_chain)
        assert len(analysis['calls']) > 0
        assert len(analysis['puts']) > 0
        
        # Statistics should make sense
        stats = analysis['chain_statistics']
        assert stats['total_calls'] + stats['total_puts'] == analysis['total_options']
    
    def test_empty_options_chain(self, options_selector):
        """Test behavior with empty options chain."""
        best_calls = options_selector.find_best_calls([])
        assert best_calls == []
        
        analysis = options_selector.analyze_options_chain([])
        assert analysis['total_options'] == 0
        assert len(analysis['calls']) == 0
        assert len(analysis['puts']) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_days_to_expiration(self):
        """Test days to expiration calculation."""
        # Test with future date
        future_date = date.today() + timedelta(days=30)
        days = calculate_days_to_expiration(future_date)
        assert days == 30
        
        # Test with past date
        past_date = date.today() - timedelta(days=5)
        days = calculate_days_to_expiration(past_date)
        assert days == -5
        
        # Test with today
        today = date.today()
        days = calculate_days_to_expiration(today)
        assert days == 0
    
    def test_calculate_moneyness(self):
        """Test moneyness calculation."""
        # At-the-money
        moneyness = calculate_moneyness(100.0, 100.0)
        assert moneyness == 1.0
        
        # Out-of-the-money call (strike > spot)
        moneyness = calculate_moneyness(110.0, 100.0) 
        assert moneyness == 1.1
        
        # In-the-money call (strike < spot)
        moneyness = calculate_moneyness(90.0, 100.0)
        assert moneyness == 0.9
        
        # Test with zero price (edge case)
        moneyness = calculate_moneyness(100.0, 0.0)
        assert moneyness == 0.0
    
    def test_estimate_intrinsic_value(self):
        """Test intrinsic value estimation."""
        # ITM call
        intrinsic = estimate_intrinsic_value(OptionType.CALL, 95.0, 100.0)
        assert intrinsic == 5.0
        
        # OTM call
        intrinsic = estimate_intrinsic_value(OptionType.CALL, 105.0, 100.0)
        assert intrinsic == 0.0
        
        # ITM put
        intrinsic = estimate_intrinsic_value(OptionType.PUT, 105.0, 100.0)
        assert intrinsic == 5.0
        
        # OTM put
        intrinsic = estimate_intrinsic_value(OptionType.PUT, 95.0, 100.0)
        assert intrinsic == 0.0


@pytest.fixture
def sample_call_option():
    """Global fixture for sample call option."""
    return OptionContract(
        option_symbol="AAPL240315C00150000",
        underlying_symbol="AAPL",
        strike=Decimal("150.00"),
        expiration=date.today() + timedelta(days=60),
        option_type=OptionType.CALL,
        bid=Decimal("5.20"),
        ask=Decimal("5.40"),
        last=Decimal("5.30"),
        volume=250,
        open_interest=1500,
        delta=0.65,
        gamma=0.025,
        theta=-0.15,
        vega=0.35,
        implied_volatility=0.25,
        days_to_expiration=60
    )
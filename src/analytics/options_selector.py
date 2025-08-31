"""Options selection algorithms for finding the best call and put options.

This module implements quantitative algorithms for selecting optimal options
based on liquidity, implied volatility, Greeks, and fit criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from decimal import Decimal
import warnings

from src.models.base_models import OptionContract, OptionType

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class OptionsSelector:
    """Options selection and scoring algorithms."""
    
    def __init__(self):
        """Initialize options selector with default parameters."""
        self.default_dte_range = (45, 75)  # Days to expiration range
        self.default_delta_range = (0.55, 0.70)  # Delta range for calls
        self.min_open_interest = 250
        self.max_spread_percent = 3.0
        self.earnings_buffer_days = 10
    
    def filter_options_basic(self, options_chain: List[OptionContract],
                           option_type: OptionType = OptionType.CALL,
                           dte_range: Tuple[int, int] = None,
                           delta_range: Tuple[float, float] = None) -> List[OptionContract]:
        """Apply basic filtering to options chain.
        
        Args:
            options_chain: List of option contracts
            option_type: Call or put options
            dte_range: Days to expiration range (min, max)
            delta_range: Delta range (min, max)
            
        Returns:
            Filtered list of option contracts
        """
        if not options_chain:
            return []
        
        # Use default ranges if not provided
        dte_min, dte_max = dte_range or self.default_dte_range
        delta_min, delta_max = delta_range or self.default_delta_range
        
        filtered = []
        
        for option in options_chain:
            # Filter by option type
            if option.option_type != option_type:
                continue
            
            # Filter by DTE
            if option.days_to_expiration is not None:
                if not (dte_min <= option.days_to_expiration <= dte_max):
                    continue
            
            # Filter by delta
            if option.delta is not None:
                if option_type == OptionType.CALL:
                    if not (delta_min <= option.delta <= delta_max):
                        continue
                else:  # PUT
                    # For puts, delta is negative, so we use absolute value
                    if not (delta_min <= abs(option.delta) <= delta_max):
                        continue
            
            # Basic liquidity filter
            if option.open_interest is not None and option.open_interest < self.min_open_interest:
                continue
            
            # Spread filter
            if option.bid is not None and option.ask is not None and option.bid > 0:
                spread_percent = float((option.ask - option.bid) / option.bid) * 100
                if spread_percent > self.max_spread_percent:
                    continue
            
            filtered.append(option)
        
        return filtered
    
    def calculate_liquidity_score(self, option: OptionContract) -> float:
        """Calculate liquidity score for an option (0-40 points).
        
        Scoring:
        - Open Interest > 1000: 20 points
        - Volume > 100: 10 points  
        - Spread < 2%: 10 points
        
        Args:
            option: Option contract to score
            
        Returns:
            Liquidity score (0-40)
        """
        score = 0.0
        
        # Open Interest Score (20 points max)
        if option.open_interest is not None:
            if option.open_interest >= 1000:
                score += 20
            elif option.open_interest >= 500:
                score += 15
            elif option.open_interest >= self.min_open_interest:
                score += 10
        
        # Volume Score (10 points max)
        if option.volume is not None:
            if option.volume >= 100:
                score += 10
            elif option.volume >= 50:
                score += 7
            elif option.volume >= 25:
                score += 5
        
        # Spread Score (10 points max)
        if (option.bid is not None and option.ask is not None and 
            option.bid > 0 and option.ask > option.bid):
            spread_percent = float((option.ask - option.bid) / option.bid) * 100
            if spread_percent < 1.0:
                score += 10
            elif spread_percent < 2.0:
                score += 8
            elif spread_percent < 3.0:
                score += 5
        
        return score
    
    def calculate_iv_value_score(self, option: OptionContract, 
                                iv_percentile: float = None,
                                historical_volatility: float = None) -> float:
        """Calculate IV value score for an option (0-30 points).
        
        Scoring:
        - IV percentile < 30%: 15 points
        - IV/HV ratio < 1.2: 15 points
        
        Args:
            option: Option contract to score
            iv_percentile: IV percentile vs 1-year history
            historical_volatility: Historical volatility for comparison
            
        Returns:
            IV value score (0-30)
        """
        score = 0.0
        
        # IV Percentile Score (15 points max)
        if iv_percentile is not None and not np.isnan(iv_percentile):
            if iv_percentile < 20:
                score += 15
            elif iv_percentile < 30:
                score += 12
            elif iv_percentile < 50:
                score += 8
            elif iv_percentile < 70:
                score += 4
        
        # IV/HV Ratio Score (15 points max)
        if (option.implied_volatility is not None and 
            historical_volatility is not None and 
            historical_volatility > 0):
            
            iv_hv_ratio = option.implied_volatility / historical_volatility
            
            if iv_hv_ratio < 1.0:
                score += 15
            elif iv_hv_ratio < 1.2:
                score += 12
            elif iv_hv_ratio < 1.5:
                score += 8
            elif iv_hv_ratio < 2.0:
                score += 4
        
        return score
    
    def calculate_fit_score(self, option: OptionContract,
                           target_dte_range: Tuple[int, int] = None,
                           target_delta_range: Tuple[float, float] = None) -> float:
        """Calculate fit score for an option (0-30 points).
        
        Scoring:
        - Delta in target range: 15 points
        - DTE in target range: 15 points
        
        Args:
            option: Option contract to score
            target_dte_range: Target DTE range
            target_delta_range: Target delta range
            
        Returns:
            Fit score (0-30)
        """
        score = 0.0
        
        # Use default ranges if not provided
        dte_min, dte_max = target_dte_range or self.default_dte_range
        delta_min, delta_max = target_delta_range or self.default_delta_range
        
        # Delta Score (15 points max)
        if option.delta is not None:
            delta_value = option.delta if option.option_type == OptionType.CALL else abs(option.delta)
            
            # Perfect fit
            if delta_min <= delta_value <= delta_max:
                score += 15
            # Close fit (within 0.05)
            elif (delta_min - 0.05 <= delta_value <= delta_max + 0.05):
                score += 10
            # Acceptable fit (within 0.10)  
            elif (delta_min - 0.10 <= delta_value <= delta_max + 0.10):
                score += 5
        
        # DTE Score (15 points max)
        if option.days_to_expiration is not None:
            # Perfect fit
            if dte_min <= option.days_to_expiration <= dte_max:
                score += 15
            # Close fit (within 7 days)
            elif (dte_min - 7 <= option.days_to_expiration <= dte_max + 7):
                score += 10
            # Acceptable fit (within 14 days)
            elif (dte_min - 14 <= option.days_to_expiration <= dte_max + 14):
                score += 5
        
        return score
    
    def calculate_total_score(self, option: OptionContract,
                            iv_percentile: float = None,
                            historical_volatility: float = None,
                            target_dte_range: Tuple[int, int] = None,
                            target_delta_range: Tuple[float, float] = None) -> Dict[str, float]:
        """Calculate comprehensive option score.
        
        Args:
            option: Option contract to score
            iv_percentile: IV percentile vs 1-year history
            historical_volatility: Historical volatility
            target_dte_range: Target DTE range
            target_delta_range: Target delta range
            
        Returns:
            Dictionary with detailed scoring breakdown
        """
        liquidity_score = self.calculate_liquidity_score(option)
        iv_value_score = self.calculate_iv_value_score(option, iv_percentile, historical_volatility)
        fit_score = self.calculate_fit_score(option, target_dte_range, target_delta_range)
        
        total_score = liquidity_score + iv_value_score + fit_score
        
        return {
            'total_score': total_score,
            'liquidity_score': liquidity_score,
            'iv_value_score': iv_value_score,
            'fit_score': fit_score,
            'liquidity_weight': 40.0,
            'iv_value_weight': 30.0,
            'fit_weight': 30.0
        }
    
    def has_earnings_risk(self, option: OptionContract, 
                         earnings_date: Optional[date] = None) -> bool:
        """Check if option has earnings risk.
        
        Args:
            option: Option contract
            earnings_date: Known earnings date
            
        Returns:
            True if earnings risk exists
        """
        if earnings_date is None or option.expiration is None:
            return False
        
        # Check if earnings date is within buffer period of expiration
        days_to_earnings = (earnings_date - date.today()).days
        days_to_expiry = (option.expiration - date.today()).days
        
        # Risk if earnings is between now and expiration, with buffer
        return (0 <= days_to_earnings <= days_to_expiry + self.earnings_buffer_days)
    
    def should_reject_option(self, option: OptionContract,
                           earnings_date: Optional[date] = None) -> Tuple[bool, str]:
        """Determine if option should be rejected based on criteria.
        
        Args:
            option: Option contract to evaluate
            earnings_date: Optional earnings date
            
        Returns:
            Tuple of (should_reject, reason)
        """
        # Spread check
        if (option.bid is not None and option.ask is not None and 
            option.bid > 0 and option.ask > option.bid):
            spread_percent = float((option.ask - option.bid) / option.bid) * 100
            if spread_percent > self.max_spread_percent:
                return True, f"Spread too wide: {spread_percent:.1f}%"
        
        # Open interest check
        if option.open_interest is not None and option.open_interest < self.min_open_interest:
            return True, f"Low open interest: {option.open_interest}"
        
        # Delta range check
        if option.delta is not None:
            delta_value = option.delta if option.option_type == OptionType.CALL else abs(option.delta)
            if not (0.40 <= delta_value <= 0.80):
                return True, f"Delta outside acceptable range: {delta_value:.2f}"
        
        # Earnings risk check
        if self.has_earnings_risk(option, earnings_date):
            return True, "Earnings within risk window"
        
        # Basic sanity checks
        if option.bid is not None and option.bid <= 0:
            return True, "No bid price"
        
        if option.days_to_expiration is not None and option.days_to_expiration <= 0:
            return True, "Expired or expiring today"
        
        return False, ""
    
    def find_best_calls(self, options_chain: List[OptionContract],
                       strategy_type: str = "swing",
                       iv_percentile: float = None,
                       historical_volatility: float = None,
                       earnings_date: Optional[date] = None,
                       top_n: int = 5) -> List[Dict]:
        """Find the best call options based on scoring algorithm.
        
        Args:
            options_chain: List of option contracts
            strategy_type: "swing" (45-75 DTE) or "longer_swing" (75-120 DTE)
            iv_percentile: IV percentile for scoring
            historical_volatility: Historical volatility for IV/HV ratio
            earnings_date: Earnings date for risk assessment
            top_n: Number of top options to return
            
        Returns:
            List of dictionaries with option and scoring details
        """
        # Set DTE range based on strategy
        if strategy_type == "longer_swing":
            dte_range = (75, 120)
        else:  # swing
            dte_range = self.default_dte_range
        
        # Filter for calls only
        call_options = [opt for opt in options_chain if opt.option_type == OptionType.CALL]
        
        if not call_options:
            return []
        
        # Apply basic filtering
        filtered_options = self.filter_options_basic(
            call_options, OptionType.CALL, dte_range, self.default_delta_range
        )
        
        if not filtered_options:
            return []
        
        # Score and rank options
        scored_options = []
        
        for option in filtered_options:
            # Check rejection criteria
            should_reject, reject_reason = self.should_reject_option(option, earnings_date)
            if should_reject:
                continue
            
            # Calculate comprehensive score
            scoring = self.calculate_total_score(
                option, iv_percentile, historical_volatility, dte_range, self.default_delta_range
            )
            
            # Add option details
            option_data = {
                'option': option,
                'scoring': scoring,
                'strategy_type': strategy_type,
                'dte_range': dte_range,
                'delta_range': self.default_delta_range
            }
            
            # Add calculated metrics
            if option.bid is not None and option.ask is not None and option.bid > 0:
                option_data['spread_percent'] = float((option.ask - option.bid) / option.bid) * 100
                option_data['mid_price'] = float((option.bid + option.ask) / 2)
            
            if option.last is not None and option.strike is not None:
                # Simple intrinsic value calculation
                # Note: Would need current stock price for accurate calculation
                option_data['time_value_estimate'] = float(option.last) if option.last > 0 else 0
            
            scored_options.append(option_data)
        
        # Sort by total score (descending)
        scored_options.sort(key=lambda x: x['scoring']['total_score'], reverse=True)
        
        # Return top N options
        return scored_options[:top_n]
    
    def find_best_puts(self, options_chain: List[OptionContract],
                      strategy_type: str = "swing",
                      iv_percentile: float = None,
                      historical_volatility: float = None,
                      earnings_date: Optional[date] = None,
                      top_n: int = 5) -> List[Dict]:
        """Find the best put options based on scoring algorithm.
        
        Args:
            options_chain: List of option contracts
            strategy_type: "swing" (45-75 DTE) or "longer_swing" (75-120 DTE)
            iv_percentile: IV percentile for scoring
            historical_volatility: Historical volatility for IV/HV ratio
            earnings_date: Earnings date for risk assessment
            top_n: Number of top options to return
            
        Returns:
            List of dictionaries with option and scoring details
        """
        # Set DTE range based on strategy
        if strategy_type == "longer_swing":
            dte_range = (75, 120)
        else:  # swing
            dte_range = self.default_dte_range
        
        # For puts, we use the same delta range but check absolute value
        put_delta_range = self.default_delta_range
        
        # Filter for puts only
        put_options = [opt for opt in options_chain if opt.option_type == OptionType.PUT]
        
        if not put_options:
            return []
        
        # Apply basic filtering
        filtered_options = self.filter_options_basic(
            put_options, OptionType.PUT, dte_range, put_delta_range
        )
        
        if not filtered_options:
            return []
        
        # Score and rank options
        scored_options = []
        
        for option in filtered_options:
            # Check rejection criteria
            should_reject, reject_reason = self.should_reject_option(option, earnings_date)
            if should_reject:
                continue
            
            # Calculate comprehensive score
            scoring = self.calculate_total_score(
                option, iv_percentile, historical_volatility, dte_range, put_delta_range
            )
            
            # Add option details
            option_data = {
                'option': option,
                'scoring': scoring,
                'strategy_type': strategy_type,
                'dte_range': dte_range,
                'delta_range': put_delta_range
            }
            
            # Add calculated metrics
            if option.bid is not None and option.ask is not None and option.bid > 0:
                option_data['spread_percent'] = float((option.ask - option.bid) / option.bid) * 100
                option_data['mid_price'] = float((option.bid + option.ask) / 2)
            
            if option.last is not None and option.strike is not None:
                option_data['time_value_estimate'] = float(option.last) if option.last > 0 else 0
            
            scored_options.append(option_data)
        
        # Sort by total score (descending)
        scored_options.sort(key=lambda x: x['scoring']['total_score'], reverse=True)
        
        # Return top N options
        return scored_options[:top_n]
    
    def analyze_options_chain(self, options_chain: List[OptionContract],
                             current_stock_price: float = None,
                             iv_percentile: float = None,
                             historical_volatility: float = None,
                             earnings_date: Optional[date] = None) -> Dict:
        """Comprehensive analysis of entire options chain.
        
        Args:
            options_chain: List of option contracts
            current_stock_price: Current stock price for moneyness calculations
            iv_percentile: IV percentile for scoring
            historical_volatility: Historical volatility
            earnings_date: Earnings date for risk assessment
            
        Returns:
            Dictionary with comprehensive chain analysis
        """
        analysis = {
            'total_options': len(options_chain),
            'calls': [],
            'puts': [],
            'chain_statistics': {},
            'best_calls_swing': [],
            'best_calls_longer': [],
            'best_puts_swing': [],
            'best_puts_longer': [],
            'analysis_timestamp': datetime.now()
        }
        
        if not options_chain:
            return analysis
        
        # Separate calls and puts
        calls = [opt for opt in options_chain if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options_chain if opt.option_type == OptionType.PUT]
        
        analysis['calls'] = calls
        analysis['puts'] = puts
        
        # Chain statistics
        analysis['chain_statistics'] = {
            'total_calls': len(calls),
            'total_puts': len(puts),
            'total_volume': sum(opt.volume or 0 for opt in options_chain),
            'total_open_interest': sum(opt.open_interest or 0 for opt in options_chain),
            'avg_iv': np.mean([opt.implied_volatility for opt in options_chain 
                              if opt.implied_volatility is not None]) if options_chain else None
        }
        
        # Find best options for different strategies
        try:
            analysis['best_calls_swing'] = self.find_best_calls(
                options_chain, "swing", iv_percentile, historical_volatility, earnings_date
            )
            
            analysis['best_calls_longer'] = self.find_best_calls(
                options_chain, "longer_swing", iv_percentile, historical_volatility, earnings_date
            )
            
            analysis['best_puts_swing'] = self.find_best_puts(
                options_chain, "swing", iv_percentile, historical_volatility, earnings_date
            )
            
            analysis['best_puts_longer'] = self.find_best_puts(
                options_chain, "longer_swing", iv_percentile, historical_volatility, earnings_date
            )
            
        except Exception as e:
            analysis['error'] = f"Error in options analysis: {str(e)}"
        
        return analysis


def calculate_days_to_expiration(expiration_date: date) -> int:
    """Calculate days to expiration from today.
    
    Args:
        expiration_date: Option expiration date
        
    Returns:
        Number of days until expiration
    """
    today = date.today()
    return (expiration_date - today).days


def calculate_moneyness(strike_price: float, current_price: float) -> float:
    """Calculate option moneyness (strike/spot ratio).
    
    Args:
        strike_price: Option strike price
        current_price: Current stock price
        
    Returns:
        Moneyness ratio
    """
    if current_price <= 0:
        return 0.0
    return strike_price / current_price


def estimate_intrinsic_value(option_type: OptionType, strike_price: float, 
                           current_price: float) -> float:
    """Estimate intrinsic value of option.
    
    Args:
        option_type: Call or put
        strike_price: Strike price
        current_price: Current stock price
        
    Returns:
        Intrinsic value
    """
    if option_type == OptionType.CALL:
        return max(0, current_price - strike_price)
    else:  # PUT
        return max(0, strike_price - current_price)
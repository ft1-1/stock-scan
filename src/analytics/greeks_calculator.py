"""Options Greeks calculations and Black-Scholes pricing models.

This module implements mathematical models for calculating option Greeks
and theoretical option prices using the Black-Scholes-Merton model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date
from scipy.stats import norm
from decimal import Decimal
import warnings

from src.models.base_models import OptionContract, OptionType

# Suppress scipy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BlackScholesCalculator:
    """Black-Scholes option pricing and Greeks calculations."""
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula.
        
        Formula: d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            d1 value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula.
        
        Formula: d2 = d1 - σ * √T
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            d2 value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def european_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price using Black-Scholes.
        
        Formula: C = S * N(d1) - K * e^(-r*T) * N(d2)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Call option theoretical price
        """
        if T <= 0:
            return max(0, S - K)  # Intrinsic value
        
        if sigma <= 0:
            return max(0, S - K * np.exp(-r * T))
        
        try:
            d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator._d2(S, K, T, r, sigma)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return max(0, call_price)
            
        except (ValueError, FloatingPointError):
            return max(0, S - K)
    
    @staticmethod
    def european_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price using Black-Scholes.
        
        Formula: P = K * e^(-r*T) * N(-d2) - S * N(-d1)
        
        Args:
            S: Current stock price
            K: Strike price  
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Put option theoretical price
        """
        if T <= 0:
            return max(0, K - S)  # Intrinsic value
        
        if sigma <= 0:
            return max(0, K * np.exp(-r * T) - S)
        
        try:
            d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator._d2(S, K, T, r, sigma)
            
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return max(0, put_price)
            
        except (ValueError, FloatingPointError):
            return max(0, K - S)
    
    @staticmethod
    def calculate_delta(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: OptionType) -> float:
        """Calculate option delta (price sensitivity to stock price).
        
        Formula:
        Call Delta = N(d1)
        Put Delta = N(d1) - 1 = -N(-d1)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: Call or put
            
        Returns:
            Delta value
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:  # PUT
                return -1.0 if S < K else 0.0
        
        if sigma <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K * np.exp(-r * T) else 0.0
            else:  # PUT
                return -1.0 if S < K * np.exp(-r * T) else 0.0
        
        try:
            d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
            
            if option_type == OptionType.CALL:
                return norm.cdf(d1)
            else:  # PUT
                return norm.cdf(d1) - 1.0
                
        except (ValueError, FloatingPointError):
            return 0.0
    
    @staticmethod
    def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma (rate of change of delta).
        
        Formula: Gamma = φ(d1) / (S * σ * √T)
        Where φ(x) is the standard normal probability density function
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Gamma value (same for calls and puts)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return gamma
            
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_theta(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: OptionType) -> float:
        """Calculate option theta (time decay).
        
        Formula:
        Call Theta = -(S * φ(d1) * σ)/(2*√T) - r*K*e^(-r*T)*N(d2)
        Put Theta = -(S * φ(d1) * σ)/(2*√T) + r*K*e^(-r*T)*N(-d2)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: Call or put
            
        Returns:
            Theta value (typically negative, per day)
        """
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            return 0.0
        
        try:
            d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator._d2(S, K, T, r, sigma)
            
            # Common term
            common_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            
            if option_type == OptionType.CALL:
                theta = common_term - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                theta = common_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
            
            # Convert to per-day theta (divide by 365)
            return theta / 365.0
            
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (sensitivity to volatility).
        
        Formula: Vega = S * √T * φ(d1)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Vega value (same for calls and puts, per 1% volatility change)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        try:
            d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
            vega = S * np.sqrt(T) * norm.pdf(d1)
            
            # Return vega per 1% change in volatility
            return vega / 100.0
            
        except (ValueError, FloatingPointError):
            return 0.0
    
    @staticmethod
    def calculate_rho(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: OptionType) -> float:
        """Calculate option rho (sensitivity to interest rate).
        
        Formula:
        Call Rho = K * T * e^(-r*T) * N(d2)
        Put Rho = -K * T * e^(-r*T) * N(-d2)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: Call or put
            
        Returns:
            Rho value (per 1% interest rate change)
        """
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            if option_type == OptionType.CALL:
                return K * T * np.exp(-r * T) if S > K * np.exp(-r * T) else 0.0
            else:  # PUT
                return -K * T * np.exp(-r * T) if S < K * np.exp(-r * T) else 0.0
        
        try:
            d2 = BlackScholesCalculator._d2(S, K, T, r, sigma)
            
            if option_type == OptionType.CALL:
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            # Return rho per 1% change in interest rate
            return rho / 100.0
            
        except (ValueError, FloatingPointError):
            return 0.0


class ImpliedVolatilityCalculator:
    """Calculate implied volatility using Newton-Raphson method."""
    
    @staticmethod
    def calculate_implied_volatility(market_price: float, S: float, K: float, T: float,
                                   r: float, option_type: OptionType,
                                   initial_guess: float = 0.20,
                                   max_iterations: int = 100,
                                   tolerance: float = 1e-6) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Current market price of option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: Call or put
            initial_guess: Starting volatility guess
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if convergence fails
        """
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
        
        # Check bounds
        if option_type == OptionType.CALL and market_price < max(0, S - K * np.exp(-r * T)):
            return None
        elif option_type == OptionType.PUT and market_price < max(0, K * np.exp(-r * T) - S):
            return None
        
        volatility = initial_guess
        
        for _ in range(max_iterations):
            try:
                # Calculate theoretical price and vega
                if option_type == OptionType.CALL:
                    theoretical_price = BlackScholesCalculator.european_call_price(
                        S, K, T, r, volatility)
                else:
                    theoretical_price = BlackScholesCalculator.european_put_price(
                        S, K, T, r, volatility)
                
                vega = BlackScholesCalculator.calculate_vega(S, K, T, r, volatility)
                
                # Newton-Raphson update
                price_diff = theoretical_price - market_price
                
                if abs(price_diff) < tolerance:
                    return volatility
                
                if vega == 0:
                    return None
                
                volatility = volatility - price_diff / (vega * 100)  # vega is per 1% change
                
                # Keep volatility positive and reasonable
                volatility = max(0.001, min(5.0, volatility))
                
            except (ValueError, FloatingPointError, ZeroDivisionError):
                return None
        
        return volatility if 0.001 <= volatility <= 5.0 else None


class GreeksCalculator:
    """High-level Greeks calculator for option contracts."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """Initialize with default risk-free rate.
        
        Args:
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_greeks(self, option: OptionContract,
                           current_stock_price: float,
                           risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """Calculate all Greeks for an option contract.
        
        Args:
            option: Option contract
            current_stock_price: Current stock price
            risk_free_rate: Risk-free rate (uses default if None)
            
        Returns:
            Dictionary with all Greeks and theoretical price
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Convert parameters
        S = float(current_stock_price)
        K = float(option.strike)
        
        # Calculate time to expiration in years
        if option.days_to_expiration is not None:
            T = option.days_to_expiration / 365.0
        elif option.expiration is not None:
            T = (option.expiration - date.today()).days / 365.0
        else:
            return {'error': 'No expiration information available'}
        
        # Use provided IV or estimate from market price
        if option.implied_volatility is not None:
            sigma = option.implied_volatility
        elif option.last is not None and option.last > 0:
            # Try to calculate IV from market price
            sigma = ImpliedVolatilityCalculator.calculate_implied_volatility(
                float(option.last), S, K, T, r, option.option_type
            )
            if sigma is None:
                sigma = 0.20  # Default fallback
        else:
            sigma = 0.20  # Default volatility
        
        try:
            # Calculate theoretical price
            if option.option_type == OptionType.CALL:
                theoretical_price = BlackScholesCalculator.european_call_price(S, K, T, r, sigma)
            else:
                theoretical_price = BlackScholesCalculator.european_put_price(S, K, T, r, sigma)
            
            # Calculate all Greeks
            delta = BlackScholesCalculator.calculate_delta(S, K, T, r, sigma, option.option_type)
            gamma = BlackScholesCalculator.calculate_gamma(S, K, T, r, sigma)
            theta = BlackScholesCalculator.calculate_theta(S, K, T, r, sigma, option.option_type)
            vega = BlackScholesCalculator.calculate_vega(S, K, T, r, sigma)
            rho = BlackScholesCalculator.calculate_rho(S, K, T, r, sigma, option.option_type)
            
            # Calculate additional metrics
            intrinsic_value = max(0, S - K) if option.option_type == OptionType.CALL else max(0, K - S)
            time_value = max(0, theoretical_price - intrinsic_value)
            moneyness = K / S
            
            return {
                'theoretical_price': theoretical_price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'implied_volatility': sigma,
                'intrinsic_value': intrinsic_value,
                'time_value': time_value,
                'moneyness': moneyness,
                'time_to_expiration_years': T,
                'parameters': {
                    'stock_price': S,
                    'strike_price': K,
                    'time_to_expiry': T,
                    'risk_free_rate': r,
                    'volatility': sigma
                }
            }
            
        except Exception as e:
            return {'error': f'Error calculating Greeks: {str(e)}'}
    
    def validate_greeks(self, greeks: Dict[str, float], option_type: OptionType) -> Dict[str, bool]:
        """Validate calculated Greeks against expected ranges.
        
        Args:
            greeks: Dictionary with calculated Greeks
            option_type: Call or put option
            
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Delta validation
        if 'delta' in greeks:
            delta = greeks['delta']
            if option_type == OptionType.CALL:
                validation['delta_valid'] = 0 <= delta <= 1
            else:  # PUT
                validation['delta_valid'] = -1 <= delta <= 0
        
        # Gamma validation (always positive)
        if 'gamma' in greeks:
            validation['gamma_valid'] = greeks['gamma'] >= 0
        
        # Theta validation (usually negative for long options)
        if 'theta' in greeks:
            validation['theta_valid'] = greeks['theta'] <= 0
        
        # Vega validation (always positive for long options)
        if 'vega' in greeks:
            validation['vega_valid'] = greeks['vega'] >= 0
        
        # Rho validation
        if 'rho' in greeks:
            rho = greeks['rho']
            if option_type == OptionType.CALL:
                validation['rho_valid'] = rho >= 0  # Calls have positive rho
            else:  # PUT
                validation['rho_valid'] = rho <= 0  # Puts have negative rho
        
        # Overall validation
        validation['all_valid'] = all(validation.values())
        
        return validation
    
    def calculate_portfolio_greeks(self, positions: List[Tuple[OptionContract, int, float]]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks.
        
        Args:
            positions: List of (option, quantity, stock_price) tuples
            
        Returns:
            Dictionary with portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'net_exposure': 0.0
        }
        
        for option, quantity, stock_price in positions:
            option_greeks = self.calculate_all_greeks(option, stock_price)
            
            if 'error' not in option_greeks:
                # Weight by position size
                portfolio_greeks['delta'] += option_greeks.get('delta', 0) * quantity
                portfolio_greeks['gamma'] += option_greeks.get('gamma', 0) * quantity
                portfolio_greeks['theta'] += option_greeks.get('theta', 0) * quantity
                portfolio_greeks['vega'] += option_greeks.get('vega', 0) * quantity
                portfolio_greeks['rho'] += option_greeks.get('rho', 0) * quantity
                
                # Net exposure (delta * stock price * quantity)
                delta = option_greeks.get('delta', 0)
                portfolio_greeks['net_exposure'] += delta * stock_price * quantity
        
        return portfolio_greeks


def time_to_expiration_years(expiration_date: date) -> float:
    """Convert expiration date to years from today.
    
    Args:
        expiration_date: Option expiration date
        
    Returns:
        Time to expiration in years
    """
    today = date.today()
    days_to_expiry = (expiration_date - today).days
    return max(0.0, days_to_expiry / 365.0)


def annualize_volatility(daily_returns: pd.Series) -> float:
    """Convert daily return volatility to annualized volatility.
    
    Args:
        daily_returns: Series of daily returns
        
    Returns:
        Annualized volatility
    """
    if len(daily_returns) < 2:
        return 0.20  # Default fallback
    
    daily_vol = daily_returns.std()
    # Annualize using sqrt(252) trading days
    annualized_vol = daily_vol * np.sqrt(252)
    
    return max(0.01, min(5.0, annualized_vol))  # Reasonable bounds
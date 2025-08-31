"""Analytics package for quantitative stock and options analysis.

This package provides comprehensive technical analysis, options selection,
and quantitative scoring capabilities for the options screening application.

Key Components:
- TechnicalIndicators: RSI, MACD, Bollinger Bands, ADX, ATR, and more
- MomentumAnalysis: Price momentum, relative strength, and trend analysis  
- SqueezeDetector: TTM squeeze and volatility compression detection
- OptionsSelector: Best call/put selection with quantitative scoring
- GreeksCalculator: Black-Scholes options pricing and Greeks
- QuantitativeScorer: Comprehensive opportunity scoring system
"""

from .technical_indicators import TechnicalIndicators, validate_ohlcv_data
from .momentum_analysis import MomentumAnalysis, PercentileCalculator
from .squeeze_detector import SqueezeDetector, VolatilityRegimeDetector
from .options_selector import (
    OptionsSelector,
    calculate_days_to_expiration,
    calculate_moneyness,
    estimate_intrinsic_value
)
from .greeks_calculator import (
    BlackScholesCalculator,
    ImpliedVolatilityCalculator,
    GreeksCalculator,
    time_to_expiration_years,
    annualize_volatility
)
from .scoring_models import QuantitativeScorer, RiskAssessment

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Options Quant Analyst Agent"
__description__ = "Quantitative analysis tools for options screening"

# Export all main classes and functions
__all__ = [
    # Technical Analysis
    'TechnicalIndicators',
    'validate_ohlcv_data',
    
    # Momentum Analysis
    'MomentumAnalysis', 
    'PercentileCalculator',
    
    # Squeeze Detection
    'SqueezeDetector',
    'VolatilityRegimeDetector',
    
    # Options Selection
    'OptionsSelector',
    'calculate_days_to_expiration',
    'calculate_moneyness', 
    'estimate_intrinsic_value',
    
    # Greeks Calculation
    'BlackScholesCalculator',
    'ImpliedVolatilityCalculator',
    'GreeksCalculator',
    'time_to_expiration_years',
    'annualize_volatility',
    
    # Scoring Models
    'QuantitativeScorer',
    'RiskAssessment',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Configuration constants
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_OPTIONS_DTE_RANGE = (45, 75)
DEFAULT_OPTIONS_DELTA_RANGE = (0.55, 0.70)
DEFAULT_MIN_OPEN_INTEREST = 250
DEFAULT_MAX_SPREAD_PERCENT = 3.0

# Quick access functions for common workflows
def analyze_stock_technical(ohlcv_data, benchmark_data=None):
    """Quick technical analysis of stock data.
    
    Args:
        ohlcv_data: OHLCV DataFrame
        benchmark_data: Optional benchmark for relative strength
        
    Returns:
        Dictionary with technical indicators and momentum data
    """
    # Validate data
    validate_ohlcv_data(ohlcv_data)
    
    # Calculate technical indicators
    technical = TechnicalIndicators.calculate_all_indicators(ohlcv_data)
    
    # Calculate momentum analysis
    momentum = MomentumAnalysis.calculate_all_momentum_indicators(ohlcv_data, benchmark_data)
    
    return {
        'technical_indicators': technical,
        'momentum_analysis': momentum
    }

def detect_squeeze_opportunities(ohlcv_data):
    """Quick squeeze detection analysis.
    
    Args:
        ohlcv_data: OHLCV DataFrame
        
    Returns:
        Dictionary with squeeze analysis results
    """
    validate_ohlcv_data(ohlcv_data)
    return SqueezeDetector.comprehensive_squeeze_analysis(ohlcv_data)

def find_best_options(options_chain, current_stock_price=None, iv_percentile=None):
    """Quick options selection analysis.
    
    Args:
        options_chain: List of OptionContract objects
        current_stock_price: Current stock price
        iv_percentile: IV percentile for scoring
        
    Returns:
        Dictionary with best calls and puts
    """
    selector = OptionsSelector()
    return selector.analyze_options_chain(options_chain, current_stock_price, iv_percentile)

def calculate_comprehensive_score(ohlcv_data, options_chain=None, benchmark_data=None,
                                current_stock_price=None, iv_percentile=None):
    """Calculate comprehensive quantitative opportunity score.
    
    Args:
        ohlcv_data: OHLCV DataFrame
        options_chain: Optional list of OptionContract objects
        benchmark_data: Optional benchmark DataFrame
        current_stock_price: Current stock price
        iv_percentile: IV percentile
        
    Returns:
        Dictionary with comprehensive scoring results
    """
    scorer = QuantitativeScorer()
    return scorer.calculate_comprehensive_score(
        ohlcv_data, options_chain, benchmark_data, current_stock_price, iv_percentile
    )

# Helper function for setting up Greeks calculator
def create_greeks_calculator(risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """Create a GreeksCalculator with specified risk-free rate.
    
    Args:
        risk_free_rate: Risk-free interest rate (default 5%)
        
    Returns:
        Configured GreeksCalculator instance
    """
    return GreeksCalculator(risk_free_rate=risk_free_rate)

# Convenience imports for common use cases
TechnicalAnalysis = TechnicalIndicators  # Alias for backwards compatibility
OptionsAnalysis = OptionsSelector       # Alias for backwards compatibility
SqueezeFinder = SqueezeDetector        # Alias for backwards compatibility
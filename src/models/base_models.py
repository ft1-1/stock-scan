"""Core data models for the options screening application."""

from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


class OptionType(str, Enum):
    """Option contract types."""
    CALL = "call"
    PUT = "put"


class ScreeningCriteria(BaseModel):
    """Comprehensive screening criteria for stock filtering."""
    
    # Specific symbols (overrides all other filters if provided)
    specific_symbols: Optional[List[str]] = Field(None, description="Specific symbols to screen")
    
    # Market filters
    min_market_cap: Optional[float] = Field(None, gt=0, description="Minimum market capitalization")
    max_market_cap: Optional[float] = Field(None, gt=0, description="Maximum market capitalization")
    min_price: Optional[float] = Field(None, gt=0, description="Minimum stock price")
    max_price: Optional[float] = Field(None, gt=0, description="Maximum stock price")
    min_volume: Optional[int] = Field(None, gt=0, description="Minimum daily volume")
    
    # Technical filters
    min_rsi: Optional[float] = Field(None, ge=0, le=100, description="Minimum RSI value")
    max_rsi: Optional[float] = Field(None, ge=0, le=100, description="Maximum RSI value")
    price_above_sma: Optional[int] = Field(None, gt=0, description="Price above SMA period")
    
    # Fundamental filters
    min_pe_ratio: Optional[float] = Field(None, gt=0, description="Minimum P/E ratio")
    max_pe_ratio: Optional[float] = Field(None, gt=0, description="Maximum P/E ratio") 
    min_roe: Optional[float] = Field(None, description="Minimum Return on Equity")
    
    # Options-specific filters
    min_option_volume: Optional[int] = Field(None, gt=0, description="Minimum option volume")
    min_open_interest: Optional[int] = Field(None, gt=0, description="Minimum open interest")
    max_days_to_expiration: Optional[int] = Field(None, gt=0, description="Maximum days to expiration")
    
    # Exclusions
    exclude_sectors: List[str] = Field(default_factory=list, description="Sectors to exclude")
    exclude_symbols: List[str] = Field(default_factory=list, description="Symbols to exclude")
    
    # Processing limits
    max_symbols: Optional[int] = Field(default=3000, description="Maximum number of symbols to process")
    
    @validator('max_market_cap')
    def validate_market_cap_range(cls, v, values):
        if v is not None and 'min_market_cap' in values and values['min_market_cap'] is not None:
            if v <= values['min_market_cap']:
                raise ValueError('max_market_cap must be greater than min_market_cap')
        return v


class StockQuote(BaseModel):
    """Standardized stock quote data."""
    
    symbol: str = Field(..., description="Stock symbol")
    last_price: float = Field(..., gt=0, description="Last traded price")
    change: float = Field(..., description="Price change from previous close")
    change_percent: float = Field(..., description="Percentage change from previous close")
    volume: int = Field(..., ge=0, description="Trading volume")
    
    # Optional market data
    bid: Optional[float] = Field(None, gt=0, description="Bid price")
    ask: Optional[float] = Field(None, gt=0, description="Ask price")
    high: Optional[float] = Field(None, gt=0, description="Day high")
    low: Optional[float] = Field(None, gt=0, description="Day low")
    open: Optional[float] = Field(None, gt=0, description="Opening price")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Quote timestamp")
    
    @validator('ask')
    def validate_bid_ask_spread(cls, v, values):
        if v is not None and 'bid' in values and values['bid'] is not None:
            if v <= values['bid']:
                raise ValueError('ask price must be greater than bid price')
        return v


class OptionContract(BaseModel):
    """Option contract with full Greeks and market data."""
    
    option_symbol: str = Field(..., description="Option symbol identifier")
    underlying_symbol: str = Field(..., description="Underlying stock symbol")
    strike: Decimal = Field(..., gt=0, description="Strike price")
    expiration: date = Field(..., description="Expiration date")
    option_type: OptionType = Field(..., description="Call or put option")
    
    # Market data
    bid: Optional[Decimal] = Field(None, ge=0, description="Bid price")
    ask: Optional[Decimal] = Field(None, ge=0, description="Ask price")
    last: Optional[Decimal] = Field(None, ge=0, description="Last traded price")
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    open_interest: Optional[int] = Field(None, ge=0, description="Open interest")
    
    # Greeks
    delta: Optional[float] = Field(None, ge=-1, le=1, description="Delta")
    gamma: Optional[float] = Field(None, ge=0, description="Gamma")
    theta: Optional[float] = Field(None, le=0, description="Theta (time decay)")
    vega: Optional[float] = Field(None, ge=0, description="Vega")
    rho: Optional[float] = Field(None, description="Rho")
    implied_volatility: Optional[float] = Field(None, ge=0, description="Implied volatility")
    
    # Calculated fields
    days_to_expiration: Optional[int] = Field(None, ge=0, description="Days until expiration")
    intrinsic_value: Optional[Decimal] = Field(None, ge=0, description="Intrinsic value")
    time_value: Optional[Decimal] = Field(None, ge=0, description="Time value")
    moneyness: Optional[float] = Field(None, gt=0, description="Strike/Spot ratio")
    
    @validator('ask')
    def validate_option_bid_ask(cls, v, values):
        if v is not None and 'bid' in values and values['bid'] is not None:
            # Allow bid = ask for illiquid options, only error if ask < bid
            if v < values['bid']:
                raise ValueError('ask price must not be less than bid price')
        return v


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    
    symbol: str = Field(..., description="Stock symbol")
    
    # Moving averages
    sma_20: Optional[float] = Field(None, gt=0, description="20-period Simple Moving Average")
    sma_50: Optional[float] = Field(None, gt=0, description="50-period Simple Moving Average")
    sma_200: Optional[float] = Field(None, gt=0, description="200-period Simple Moving Average")
    ema_12: Optional[float] = Field(None, gt=0, description="12-period Exponential Moving Average")
    ema_26: Optional[float] = Field(None, gt=0, description="26-period Exponential Moving Average")
    
    # Momentum indicators
    rsi_14: Optional[float] = Field(None, ge=0, le=100, description="14-period RSI")
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    
    # Volatility indicators
    bollinger_upper: Optional[float] = Field(None, gt=0, description="Bollinger Band upper")
    bollinger_middle: Optional[float] = Field(None, gt=0, description="Bollinger Band middle")
    bollinger_lower: Optional[float] = Field(None, gt=0, description="Bollinger Band lower")
    atr_14: Optional[float] = Field(None, ge=0, description="14-period Average True Range")
    
    # Volume indicators
    volume_sma_20: Optional[float] = Field(None, ge=0, description="20-period Volume SMA")
    volume_ratio: Optional[float] = Field(None, ge=0, description="Current volume / Average volume")
    
    calculation_timestamp: datetime = Field(default_factory=datetime.now, description="Calculation timestamp")


class FundamentalData(BaseModel):
    """Fundamental financial data."""
    
    symbol: str = Field(..., description="Stock symbol")
    
    # Valuation metrics
    market_cap: Optional[float] = Field(None, gt=0, description="Market capitalization")
    pe_ratio: Optional[float] = Field(None, description="Price-to-Earnings ratio")
    pb_ratio: Optional[float] = Field(None, description="Price-to-Book ratio")
    ps_ratio: Optional[float] = Field(None, description="Price-to-Sales ratio")
    
    # Profitability metrics
    roe: Optional[float] = Field(None, description="Return on Equity")
    roa: Optional[float] = Field(None, description="Return on Assets")
    profit_margin: Optional[float] = Field(None, description="Net profit margin")
    
    # Growth metrics
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")
    earnings_growth: Optional[float] = Field(None, description="Earnings growth rate")
    
    # Financial health
    debt_to_equity: Optional[float] = Field(None, ge=0, description="Debt-to-Equity ratio")
    current_ratio: Optional[float] = Field(None, ge=0, description="Current ratio")
    
    # Dividend information
    dividend_yield: Optional[float] = Field(None, ge=0, description="Dividend yield")
    
    data_timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")


class AIAnalysisResult(BaseModel):
    """AI analysis result with scoring and reasoning."""
    
    symbol: str = Field(..., description="Stock symbol")
    overall_score: int = Field(..., ge=0, le=100, description="Overall opportunity score (0-100)")
    
    # Component scores
    technical_score: Optional[int] = Field(None, ge=0, le=100, description="Technical analysis score")
    fundamental_score: Optional[int] = Field(None, ge=0, le=100, description="Fundamental analysis score")
    options_score: Optional[int] = Field(None, ge=0, le=100, description="Options setup score")
    risk_score: Optional[int] = Field(None, ge=0, le=100, description="Risk assessment score")
    
    # Analysis details
    reasoning: str = Field(..., description="Detailed reasoning for the score")
    key_factors: List[str] = Field(default_factory=list, description="Key factors influencing the score")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    opportunities: List[str] = Field(default_factory=list, description="Identified opportunities")
    
    # Recommendations
    recommended_strategy: Optional[str] = Field(None, description="Recommended options strategy")
    position_size: Optional[str] = Field(None, description="Recommended position sizing")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    tokens_used: Optional[int] = Field(None, ge=0, description="AI tokens consumed")
    analysis_cost: Optional[float] = Field(None, ge=0, description="Analysis cost in USD")


class ScreeningResult(BaseModel):
    """Complete result of options screening analysis."""
    
    symbol: str = Field(..., description="Stock symbol")
    stock_quote: StockQuote = Field(..., description="Current stock quote")
    technical_indicators: TechnicalIndicators = Field(..., description="Technical analysis")
    fundamental_data: Optional[FundamentalData] = Field(None, description="Fundamental data")
    
    # Selected options
    selected_calls: List[OptionContract] = Field(default_factory=list, description="Selected call options")
    selected_puts: List[OptionContract] = Field(default_factory=list, description="Selected put options")
    
    # Quantitative scores
    liquidity_score: float = Field(..., ge=0, le=100, description="Liquidity assessment score")
    volatility_score: float = Field(..., ge=0, le=100, description="Volatility assessment score")
    momentum_score: float = Field(..., ge=0, le=100, description="Momentum assessment score")
    overall_score: float = Field(..., ge=0, le=100, description="Overall quantitative score")
    
    # AI analysis (if available)
    ai_analysis: Optional[AIAnalysisResult] = Field(None, description="AI analysis result")
    
    screening_timestamp: datetime = Field(default_factory=datetime.now, description="Screening timestamp")


class WorkflowResult(BaseModel):
    """Final result of the complete screening workflow."""
    
    # Metadata
    workflow_id: str = Field(..., description="Unique workflow identifier")
    started_at: datetime = Field(..., description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion time")
    
    # Input parameters
    screening_criteria: ScreeningCriteria = Field(..., description="Original screening criteria")
    
    # Results
    total_stocks_screened: int = Field(..., ge=0, description="Total stocks evaluated")
    qualifying_results: List[ScreeningResult] = Field(default_factory=list, description="Qualifying opportunities")
    
    # Performance metrics
    execution_time_seconds: Optional[float] = Field(None, ge=0, description="Total execution time")
    api_calls_made: Optional[int] = Field(None, ge=0, description="Total API calls")
    total_cost: Optional[float] = Field(None, ge=0, description="Total execution cost")
    
    # Quality metrics
    success_rate: Optional[float] = Field(None, ge=0, le=1, description="Success rate of operations")
    data_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Overall data quality")
    
    # Error information
    errors_encountered: List[str] = Field(default_factory=list, description="Errors during execution")
    warnings: List[str] = Field(default_factory=list, description="Warnings during execution")
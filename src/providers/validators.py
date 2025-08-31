"""
Data validation and quality assurance utilities for provider responses.

This module provides comprehensive validation for:
- Stock quotes and market data integrity
- Options chains and contract validation
- Fundamental data consistency checks
- Technical indicators validation
- Provider response completeness
- Data quality scoring and metrics
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from enum import Enum

from src.models import (
    StockQuote,
    OptionContract,
    FundamentalData,
    TechnicalIndicators,
    ProviderResponse,
    ProviderType,
    OptionType
)
from src.providers.exceptions import DataQualityError

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None
    
    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.field} - {self.message}"


@dataclass
class ValidationResult:
    """Complete validation result with scoring."""
    is_valid: bool
    quality_score: float  # 0-100
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    critical_count: int = 0
    
    def __post_init__(self):
        self.warnings_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
        self.errors_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)
        self.critical_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL)
    
    @property
    def has_warnings(self) -> bool:
        return self.warnings_count > 0
    
    @property
    def has_errors(self) -> bool:
        return self.errors_count > 0
    
    @property
    def has_critical_issues(self) -> bool:
        return self.critical_count > 0


class DataValidator:
    """
    Comprehensive data validation and quality assessment.
    
    Provides validation for all data types returned by providers with
    configurable quality thresholds and detailed issue reporting.
    """
    
    def __init__(self, min_quality_score: float = 80.0):
        self.min_quality_score = min_quality_score
        
        # Market hours for validation (NYSE/NASDAQ)
        self.market_open_hour = 9   # 9:30 AM ET (simplified)
        self.market_close_hour = 16  # 4:00 PM ET
        
        # Reasonable bounds for validation
        self.bounds = {
            'stock_price': (0.01, 10000.0),
            'volume': (0, 1_000_000_000),
            'market_cap': (1_000_000, 10_000_000_000_000),  # $1M to $10T
            'pe_ratio': (-1000, 1000),
            'delta': (-1.0, 1.0),
            'gamma': (0.0, 10.0),
            'theta': (-100.0, 0.0),
            'vega': (0.0, 100.0),
            'iv': (0.0, 5.0),  # 0% to 500%
            'rsi': (0.0, 100.0),
            'days_to_expiration': (0, 2555)  # ~7 years max
        }
    
    def validate_stock_quote(self, quote: StockQuote, symbol: str = None) -> ValidationResult:
        """Validate stock quote data comprehensively."""
        issues = []
        
        # Symbol validation
        if symbol and quote.symbol != symbol:
            issues.append(ValidationIssue(
                field="symbol",
                message=f"Symbol mismatch: expected {symbol}, got {quote.symbol}",
                severity=ValidationSeverity.ERROR,
                expected=symbol,
                value=quote.symbol
            ))
        
        # Price validation
        issues.extend(self._validate_price_fields(quote))
        
        # Volume validation
        if quote.volume < 0:
            issues.append(ValidationIssue(
                field="volume",
                message="Volume cannot be negative",
                severity=ValidationSeverity.ERROR,
                value=quote.volume
            ))
        elif quote.volume == 0:
            issues.append(ValidationIssue(
                field="volume",
                message="Zero volume may indicate stale or invalid data",
                severity=ValidationSeverity.WARNING,
                value=quote.volume
            ))
        elif quote.volume > self.bounds['volume'][1]:
            issues.append(ValidationIssue(
                field="volume",
                message="Volume exceeds reasonable bounds",
                severity=ValidationSeverity.WARNING,
                value=quote.volume,
                expected=f"<= {self.bounds['volume'][1]}"
            ))
        
        # Change validation
        if quote.last_price > 0:
            implied_previous = quote.last_price - quote.change
            if implied_previous <= 0:
                issues.append(ValidationIssue(
                    field="change",
                    message="Change calculation results in negative previous price",
                    severity=ValidationSeverity.ERROR,
                    value=quote.change
                ))
        
        # Change percent consistency
        if quote.last_price > 0 and quote.change != 0:
            expected_change_pct = (quote.change / (quote.last_price - quote.change)) * 100
            if abs(expected_change_pct - quote.change_percent) > 0.1:
                issues.append(ValidationIssue(
                    field="change_percent",
                    message="Change percent inconsistent with price and change values",
                    severity=ValidationSeverity.WARNING,
                    value=quote.change_percent,
                    expected=expected_change_pct
                ))
        
        # Bid/Ask validation
        if quote.bid is not None and quote.ask is not None:
            if quote.bid >= quote.ask:
                issues.append(ValidationIssue(
                    field="bid_ask_spread",
                    message="Bid price should be less than ask price",
                    severity=ValidationSeverity.ERROR,
                    value=f"bid: {quote.bid}, ask: {quote.ask}"
                ))
            
            # Check if last price is within bid/ask
            if not (quote.bid <= quote.last_price <= quote.ask):
                if abs(quote.last_price - quote.bid) > 0.1 and abs(quote.last_price - quote.ask) > 0.1:
                    issues.append(ValidationIssue(
                        field="last_price",
                        message="Last price significantly outside bid/ask spread",
                        severity=ValidationSeverity.WARNING,
                        value=quote.last_price,
                        expected=f"between {quote.bid} and {quote.ask}"
                    ))
        
        # Timestamp validation
        if quote.timestamp:
            now = datetime.now()
            age_hours = (now - quote.timestamp).total_seconds() / 3600
            
            if age_hours > 24:
                issues.append(ValidationIssue(
                    field="timestamp",
                    message="Quote data is more than 24 hours old",
                    severity=ValidationSeverity.WARNING,
                    value=quote.timestamp,
                    expected="within 24 hours"
                ))
        
        return self._calculate_validation_result(issues)
    
    def validate_options_chain(self, contracts: List[OptionContract], underlying_symbol: str = None) -> ValidationResult:
        """Validate options chain data comprehensively."""
        issues = []
        
        if not contracts:
            issues.append(ValidationIssue(
                field="contracts",
                message="Empty options chain",
                severity=ValidationSeverity.WARNING,
                value=0
            ))
            return ValidationResult(is_valid=False, quality_score=0.0, issues=issues)
        
        # Validate individual contracts
        for i, contract in enumerate(contracts):
            contract_issues = self.validate_option_contract(contract, underlying_symbol)
            
            # Add contract index to issues
            for issue in contract_issues.issues:
                issue.field = f"contract[{i}].{issue.field}"
                issues.append(issue)
        
        # Chain-level validation
        issues.extend(self._validate_chain_consistency(contracts))
        
        return self._calculate_validation_result(issues)
    
    def validate_option_contract(self, contract: OptionContract, underlying_symbol: str = None) -> ValidationResult:
        """Validate individual option contract data."""
        issues = []
        
        # Underlying symbol validation
        if underlying_symbol and contract.underlying_symbol != underlying_symbol:
            issues.append(ValidationIssue(
                field="underlying_symbol",
                message=f"Underlying symbol mismatch: expected {underlying_symbol}, got {contract.underlying_symbol}",
                severity=ValidationSeverity.ERROR,
                expected=underlying_symbol,
                value=contract.underlying_symbol
            ))
        
        # Strike price validation
        if contract.strike <= 0:
            issues.append(ValidationIssue(
                field="strike",
                message="Strike price must be positive",
                severity=ValidationSeverity.ERROR,
                value=contract.strike
            ))
        elif contract.strike > 10000:
            issues.append(ValidationIssue(
                field="strike",
                message="Strike price seems unusually high",
                severity=ValidationSeverity.WARNING,
                value=contract.strike
            ))
        
        # Expiration validation
        if contract.expiration:
            if contract.expiration < date.today():
                issues.append(ValidationIssue(
                    field="expiration",
                    message="Option has expired",
                    severity=ValidationSeverity.ERROR,
                    value=contract.expiration
                ))
            elif contract.expiration > date.today() + timedelta(days=2555):  # ~7 years
                issues.append(ValidationIssue(
                    field="expiration",
                    message="Expiration date is unusually far in the future",
                    severity=ValidationSeverity.WARNING,
                    value=contract.expiration
                ))
        
        # Days to expiration consistency
        if contract.days_to_expiration is not None and contract.expiration:
            expected_dte = (contract.expiration - date.today()).days
            if abs(expected_dte - contract.days_to_expiration) > 1:
                issues.append(ValidationIssue(
                    field="days_to_expiration",
                    message="DTE inconsistent with expiration date",
                    severity=ValidationSeverity.WARNING,
                    value=contract.days_to_expiration,
                    expected=expected_dte
                ))
        
        # Pricing validation
        issues.extend(self._validate_option_pricing(contract))
        
        # Greeks validation
        issues.extend(self._validate_greeks(contract))
        
        # Volume and open interest
        if contract.volume is not None and contract.volume < 0:
            issues.append(ValidationIssue(
                field="volume",
                message="Volume cannot be negative",
                severity=ValidationSeverity.ERROR,
                value=contract.volume
            ))
        
        if contract.open_interest is not None and contract.open_interest < 0:
            issues.append(ValidationIssue(
                field="open_interest",
                message="Open interest cannot be negative",
                severity=ValidationSeverity.ERROR,
                value=contract.open_interest
            ))
        
        return self._calculate_validation_result(issues)
    
    def validate_fundamental_data(self, data: FundamentalData, symbol: str = None) -> ValidationResult:
        """Validate fundamental data comprehensively."""
        issues = []
        
        # Symbol validation
        if symbol and data.symbol != symbol:
            issues.append(ValidationIssue(
                field="symbol",
                message=f"Symbol mismatch: expected {symbol}, got {data.symbol}",
                severity=ValidationSeverity.ERROR,
                expected=symbol,
                value=data.symbol
            ))
        
        # Market cap validation
        if data.market_cap is not None:
            if data.market_cap <= 0:
                issues.append(ValidationIssue(
                    field="market_cap",
                    message="Market cap must be positive",
                    severity=ValidationSeverity.ERROR,
                    value=data.market_cap
                ))
            elif not (self.bounds['market_cap'][0] <= data.market_cap <= self.bounds['market_cap'][1]):
                issues.append(ValidationIssue(
                    field="market_cap",
                    message="Market cap outside reasonable bounds",
                    severity=ValidationSeverity.WARNING,
                    value=data.market_cap,
                    expected=f"between {self.bounds['market_cap'][0]} and {self.bounds['market_cap'][1]}"
                ))
        
        # P/E ratio validation
        if data.pe_ratio is not None:
            if not (self.bounds['pe_ratio'][0] <= data.pe_ratio <= self.bounds['pe_ratio'][1]):
                issues.append(ValidationIssue(
                    field="pe_ratio",
                    message="P/E ratio outside reasonable bounds",
                    severity=ValidationSeverity.WARNING,
                    value=data.pe_ratio,
                    expected=f"between {self.bounds['pe_ratio'][0]} and {self.bounds['pe_ratio'][1]}"
                ))
        
        # Profitability metrics validation
        for field_name in ['roe', 'roa', 'profit_margin']:
            value = getattr(data, field_name, None)
            if value is not None:
                if value < -1.0 or value > 2.0:  # -100% to 200%
                    issues.append(ValidationIssue(
                        field=field_name,
                        message=f"{field_name.upper()} outside typical range",
                        severity=ValidationSeverity.WARNING,
                        value=value,
                        expected="between -1.0 and 2.0"
                    ))
        
        # Growth rates validation
        for field_name in ['revenue_growth', 'earnings_growth']:
            value = getattr(data, field_name, None)
            if value is not None:
                if value < -1.0 or value > 5.0:  # -100% to 500%
                    issues.append(ValidationIssue(
                        field=field_name,
                        message=f"{field_name.replace('_', ' ').title()} outside typical range",
                        severity=ValidationSeverity.WARNING,
                        value=value,
                        expected="between -1.0 and 5.0"
                    ))
        
        # Debt to equity validation
        if data.debt_to_equity is not None:
            if data.debt_to_equity < 0:
                issues.append(ValidationIssue(
                    field="debt_to_equity",
                    message="Debt to equity cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=data.debt_to_equity
                ))
            elif data.debt_to_equity > 10:
                issues.append(ValidationIssue(
                    field="debt_to_equity",
                    message="Debt to equity ratio seems unusually high",
                    severity=ValidationSeverity.WARNING,
                    value=data.debt_to_equity
                ))
        
        # Current ratio validation
        if data.current_ratio is not None:
            if data.current_ratio < 0:
                issues.append(ValidationIssue(
                    field="current_ratio",
                    message="Current ratio cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=data.current_ratio
                ))
        
        # Dividend yield validation
        if data.dividend_yield is not None:
            if data.dividend_yield < 0 or data.dividend_yield > 0.15:  # 0% to 15%
                issues.append(ValidationIssue(
                    field="dividend_yield",
                    message="Dividend yield outside typical range",
                    severity=ValidationSeverity.WARNING,
                    value=data.dividend_yield,
                    expected="between 0 and 0.15"
                ))
        
        return self._calculate_validation_result(issues)
    
    def validate_technical_indicators(self, indicators: TechnicalIndicators, symbol: str = None) -> ValidationResult:
        """Validate technical indicators data."""
        issues = []
        
        # Symbol validation
        if symbol and indicators.symbol != symbol:
            issues.append(ValidationIssue(
                field="symbol",
                message=f"Symbol mismatch: expected {symbol}, got {indicators.symbol}",
                severity=ValidationSeverity.ERROR,
                expected=symbol,
                value=indicators.symbol
            ))
        
        # RSI validation
        if indicators.rsi_14 is not None:
            if not (0 <= indicators.rsi_14 <= 100):
                issues.append(ValidationIssue(
                    field="rsi_14",
                    message="RSI must be between 0 and 100",
                    severity=ValidationSeverity.ERROR,
                    value=indicators.rsi_14,
                    expected="0-100"
                ))
        
        # Moving average validation
        ma_fields = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']
        ma_values = []
        
        for field in ma_fields:
            value = getattr(indicators, field, None)
            if value is not None:
                if value <= 0:
                    issues.append(ValidationIssue(
                        field=field,
                        message=f"{field.upper()} must be positive",
                        severity=ValidationSeverity.ERROR,
                        value=value
                    ))
                else:
                    ma_values.append((field, value))
        
        # Check moving average order (longer periods should generally be smoother)
        if indicators.sma_20 and indicators.sma_50:
            if abs(indicators.sma_20 - indicators.sma_50) / indicators.sma_50 > 0.5:
                issues.append(ValidationIssue(
                    field="sma_consistency",
                    message="Large divergence between SMA20 and SMA50",
                    severity=ValidationSeverity.WARNING,
                    value=f"SMA20: {indicators.sma_20}, SMA50: {indicators.sma_50}"
                ))
        
        # Bollinger Bands validation
        if all([indicators.bollinger_upper, indicators.bollinger_middle, indicators.bollinger_lower]):
            if not (indicators.bollinger_lower < indicators.bollinger_middle < indicators.bollinger_upper):
                issues.append(ValidationIssue(
                    field="bollinger_bands",
                    message="Bollinger Bands ordering invalid (lower < middle < upper)",
                    severity=ValidationSeverity.ERROR,
                    value=f"L:{indicators.bollinger_lower}, M:{indicators.bollinger_middle}, U:{indicators.bollinger_upper}"
                ))
        
        # ATR validation
        if indicators.atr_14 is not None:
            if indicators.atr_14 < 0:
                issues.append(ValidationIssue(
                    field="atr_14",
                    message="ATR cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=indicators.atr_14
                ))
        
        # MACD validation
        if all([indicators.macd, indicators.macd_signal, indicators.macd_histogram]):
            expected_histogram = indicators.macd - indicators.macd_signal
            if abs(expected_histogram - indicators.macd_histogram) > 0.001:
                issues.append(ValidationIssue(
                    field="macd_histogram",
                    message="MACD histogram inconsistent with MACD and signal lines",
                    severity=ValidationSeverity.WARNING,
                    value=indicators.macd_histogram,
                    expected=expected_histogram
                ))
        
        return self._calculate_validation_result(issues)
    
    def validate_provider_response(self, response: ProviderResponse) -> ValidationResult:
        """Validate provider response structure and metadata."""
        issues = []
        
        # Basic response validation
        if response.success and response.error:
            issues.append(ValidationIssue(
                field="success_error_consistency",
                message="Response marked as successful but contains error message",
                severity=ValidationSeverity.WARNING,
                value=f"success: {response.success}, error: {response.error}"
            ))
        
        if not response.success and not response.error:
            issues.append(ValidationIssue(
                field="error_message",
                message="Failed response should include error message",
                severity=ValidationSeverity.WARNING,
                value="missing error message"
            ))
        
        # Response time validation
        if response.response_time_ms is not None:
            if response.response_time_ms < 0:
                issues.append(ValidationIssue(
                    field="response_time_ms",
                    message="Response time cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=response.response_time_ms
                ))
            elif response.response_time_ms > 30000:  # 30 seconds
                issues.append(ValidationIssue(
                    field="response_time_ms",
                    message="Response time unusually high",
                    severity=ValidationSeverity.WARNING,
                    value=response.response_time_ms,
                    expected="< 30000ms"
                ))
        
        # Timestamp validation
        if response.timestamp:
            age = (datetime.now() - response.timestamp).total_seconds()
            if age > 300:  # 5 minutes
                issues.append(ValidationIssue(
                    field="timestamp",
                    message="Response timestamp is more than 5 minutes old",
                    severity=ValidationSeverity.WARNING,
                    value=response.timestamp,
                    expected="within 5 minutes"
                ))
        
        return self._calculate_validation_result(issues)
    
    def _validate_price_fields(self, quote: StockQuote) -> List[ValidationIssue]:
        """Validate price-related fields in stock quote."""
        issues = []
        
        # Last price validation
        if quote.last_price <= 0:
            issues.append(ValidationIssue(
                field="last_price",
                message="Last price must be positive",
                severity=ValidationSeverity.ERROR,
                value=quote.last_price
            ))
        elif not (self.bounds['stock_price'][0] <= quote.last_price <= self.bounds['stock_price'][1]):
            issues.append(ValidationIssue(
                field="last_price",
                message="Last price outside reasonable bounds",
                severity=ValidationSeverity.WARNING,
                value=quote.last_price,
                expected=f"between {self.bounds['stock_price'][0]} and {self.bounds['stock_price'][1]}"
            ))
        
        # OHLC validation
        if all([quote.open, quote.high, quote.low, quote.last_price]):
            if not (quote.low <= quote.open <= quote.high):
                issues.append(ValidationIssue(
                    field="ohlc_consistency",
                    message="Open price not between high and low",
                    severity=ValidationSeverity.ERROR,
                    value=f"O:{quote.open}, H:{quote.high}, L:{quote.low}"
                ))
            
            if not (quote.low <= quote.last_price <= quote.high):
                issues.append(ValidationIssue(
                    field="ohlc_consistency",
                    message="Last price not between high and low",
                    severity=ValidationSeverity.WARNING,
                    value=f"Last:{quote.last_price}, H:{quote.high}, L:{quote.low}"
                ))
        
        return issues
    
    def _validate_option_pricing(self, contract: OptionContract) -> List[ValidationIssue]:
        """Validate option pricing data."""
        issues = []
        
        # Bid/Ask validation
        if contract.bid is not None and contract.ask is not None:
            # Allow bid = ask for illiquid options (common in far OTM)
            # Only error if bid > ask which is impossible
            if contract.bid > contract.ask:
                issues.append(ValidationIssue(
                    field="bid_ask_spread",
                    message="Bid cannot be greater than ask",
                    severity=ValidationSeverity.ERROR,
                    value=f"bid: {contract.bid}, ask: {contract.ask}"
                ))
            elif contract.bid < 0 or contract.ask < 0:
                issues.append(ValidationIssue(
                    field="bid_ask_prices",
                    message="Bid and ask prices cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=f"bid: {contract.bid}, ask: {contract.ask}"
                ))
        
        # Last price validation
        if contract.last is not None and contract.last < 0:
            issues.append(ValidationIssue(
                field="last_price",
                message="Last price cannot be negative",
                severity=ValidationSeverity.ERROR,
                value=contract.last
            ))
        
        # Price reasonableness for options
        prices = [p for p in [contract.bid, contract.ask, contract.last] if p is not None]
        for price_field, price in zip(['bid', 'ask', 'last'], [contract.bid, contract.ask, contract.last]):
            if price is not None and price > 1000:
                issues.append(ValidationIssue(
                    field=price_field,
                    message="Option price seems unusually high",
                    severity=ValidationSeverity.WARNING,
                    value=price
                ))
        
        return issues
    
    def _validate_greeks(self, contract: OptionContract) -> List[ValidationIssue]:
        """Validate option Greeks."""
        issues = []
        
        # Delta validation
        if contract.delta is not None:
            if not (-1.0 <= contract.delta <= 1.0):
                issues.append(ValidationIssue(
                    field="delta",
                    message="Delta must be between -1.0 and 1.0",
                    severity=ValidationSeverity.ERROR,
                    value=contract.delta,
                    expected="-1.0 to 1.0"
                ))
            else:
                # Delta sign should match option type for ITM options
                if contract.option_type == OptionType.CALL and contract.delta < 0:
                    issues.append(ValidationIssue(
                        field="delta",
                        message="Call option delta should be positive",
                        severity=ValidationSeverity.WARNING,
                        value=contract.delta
                    ))
                elif contract.option_type == OptionType.PUT and contract.delta > 0:
                    issues.append(ValidationIssue(
                        field="delta",
                        message="Put option delta should be negative",
                        severity=ValidationSeverity.WARNING,
                        value=contract.delta
                    ))
        
        # Gamma validation
        if contract.gamma is not None:
            if contract.gamma < 0:
                issues.append(ValidationIssue(
                    field="gamma",
                    message="Gamma cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=contract.gamma
                ))
            elif contract.gamma > self.bounds['gamma'][1]:
                issues.append(ValidationIssue(
                    field="gamma",
                    message="Gamma value unusually high",
                    severity=ValidationSeverity.WARNING,
                    value=contract.gamma,
                    expected=f"<= {self.bounds['gamma'][1]}"
                ))
        
        # Theta validation
        if contract.theta is not None:
            if contract.theta > 0:
                issues.append(ValidationIssue(
                    field="theta",
                    message="Theta should be negative (time decay)",
                    severity=ValidationSeverity.WARNING,
                    value=contract.theta
                ))
            elif contract.theta < self.bounds['theta'][0]:
                issues.append(ValidationIssue(
                    field="theta",
                    message="Theta value unusually negative",
                    severity=ValidationSeverity.WARNING,
                    value=contract.theta,
                    expected=f">= {self.bounds['theta'][0]}"
                ))
        
        # Vega validation
        if contract.vega is not None:
            if contract.vega < 0:
                issues.append(ValidationIssue(
                    field="vega",
                    message="Vega cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    value=contract.vega
                ))
            elif contract.vega > self.bounds['vega'][1]:
                issues.append(ValidationIssue(
                    field="vega",
                    message="Vega value unusually high",
                    severity=ValidationSeverity.WARNING,
                    value=contract.vega,
                    expected=f"<= {self.bounds['vega'][1]}"
                ))
        
        # Implied volatility validation
        if contract.implied_volatility is not None:
            if not (0 <= contract.implied_volatility <= self.bounds['iv'][1]):
                issues.append(ValidationIssue(
                    field="implied_volatility",
                    message="Implied volatility outside reasonable bounds",
                    severity=ValidationSeverity.WARNING,
                    value=contract.implied_volatility,
                    expected=f"0 to {self.bounds['iv'][1]}"
                ))
        
        return issues
    
    def _validate_chain_consistency(self, contracts: List[OptionContract]) -> List[ValidationIssue]:
        """Validate consistency across option chain."""
        issues = []
        
        if not contracts:
            return issues
        
        # Check for duplicate contracts
        seen_contracts = set()
        for contract in contracts:
            contract_key = (
                contract.underlying_symbol,
                contract.option_type,
                contract.strike,
                contract.expiration
            )
            
            if contract_key in seen_contracts:
                issues.append(ValidationIssue(
                    field="duplicate_contracts",
                    message=f"Duplicate option contract found",
                    severity=ValidationSeverity.WARNING,
                    value=contract_key
                ))
            else:
                seen_contracts.add(contract_key)
        
        # Check underlying symbol consistency
        underlying_symbols = {contract.underlying_symbol for contract in contracts}
        if len(underlying_symbols) > 1:
            issues.append(ValidationIssue(
                field="underlying_consistency",
                message="Multiple underlying symbols in single chain",
                severity=ValidationSeverity.WARNING,
                value=list(underlying_symbols)
            ))
        
        # Check for reasonable strike distribution
        strikes = [float(contract.strike) for contract in contracts]
        if strikes:
            strike_range = max(strikes) - min(strikes)
            if strike_range == 0 and len(strikes) > 1:
                issues.append(ValidationIssue(
                    field="strike_distribution",
                    message="All contracts have same strike price",
                    severity=ValidationSeverity.WARNING,
                    value=strikes[0]
                ))
        
        return issues
    
    def _calculate_validation_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Calculate validation result and quality score."""
        if not issues:
            return ValidationResult(is_valid=True, quality_score=100.0, issues=[])
        
        # Count issues by severity
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.INFO)
        
        # Calculate quality score (0-100)
        # Critical issues: -50 points each
        # Error issues: -20 points each
        # Warning issues: -5 points each
        # Info issues: -1 point each
        
        quality_score = 100.0
        quality_score -= critical_count * 50
        quality_score -= error_count * 20
        quality_score -= warning_count * 5
        quality_score -= info_count * 1
        
        quality_score = max(0.0, quality_score)
        
        # Data is invalid if there are critical issues or errors
        is_valid = critical_count == 0 and error_count == 0
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues
        )


# Global validator instance
_global_validator: Optional[DataValidator] = None


def get_validator() -> DataValidator:
    """Get or create global validator instance."""
    global _global_validator
    
    if _global_validator is None:
        _global_validator = DataValidator()
    
    return _global_validator


def initialize_validator(min_quality_score: float = 80.0) -> DataValidator:
    """Initialize global validator with configuration."""
    global _global_validator
    
    _global_validator = DataValidator(min_quality_score=min_quality_score)
    return _global_validator
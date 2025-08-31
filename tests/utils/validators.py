"""Test validation utilities for financial data testing."""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
from decimal import Decimal
from datetime import date, datetime


class TestValidators:
    """Validation utilities for testing financial data integrity."""
    
    @staticmethod
    def validate_stock_quote(quote_data: Dict) -> bool:
        """Validate stock quote data structure and values."""
        required_fields = ["symbol", "last_price", "volume", "bid", "ask", "high", "low", "open"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in quote_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate price relationships
        if quote_data["ask"] < quote_data["bid"]:
            raise ValueError(f"Ask ({quote_data['ask']}) cannot be less than bid ({quote_data['bid']})")
        
        if quote_data["high"] < quote_data["low"]:
            raise ValueError(f"High ({quote_data['high']}) cannot be less than low ({quote_data['low']})")
        
        # Validate price ranges
        for price_field in ["last_price", "bid", "ask", "high", "low", "open"]:
            if quote_data[price_field] <= 0:
                raise ValueError(f"{price_field} must be positive, got {quote_data[price_field]}")
        
        # Validate OHLC relationships
        if not (quote_data["low"] <= quote_data["open"] <= quote_data["high"]):
            raise ValueError("Open price must be between high and low")
        
        if not (quote_data["low"] <= quote_data["last_price"] <= quote_data["high"]):
            raise ValueError("Last price must be between high and low")
        
        # Validate volume
        if quote_data["volume"] < 0:
            raise ValueError(f"Volume cannot be negative: {quote_data['volume']}")
        
        return True
    
    @staticmethod
    def validate_option_contract(option_data: Dict) -> bool:
        """Validate option contract data structure and values."""
        required_fields = [
            "option_symbol", "underlying_symbol", "strike", "expiration", 
            "option_type", "bid", "ask", "delta", "gamma", "theta", "vega"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in option_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate option type
        if option_data["option_type"] not in ["call", "put"]:
            raise ValueError(f"Invalid option type: {option_data['option_type']}")
        
        # Validate pricing
        bid = float(option_data["bid"]) if isinstance(option_data["bid"], Decimal) else option_data["bid"]
        ask = float(option_data["ask"]) if isinstance(option_data["ask"], Decimal) else option_data["ask"]
        
        if ask < bid:
            raise ValueError(f"Ask ({ask}) cannot be less than bid ({bid})")
        
        if bid < 0 or ask < 0:
            raise ValueError("Option prices cannot be negative")
        
        # Validate Greeks
        delta = option_data["delta"]
        if option_data["option_type"] == "call":
            if not (0 <= delta <= 1):
                raise ValueError(f"Call delta must be between 0 and 1, got {delta}")
        else:  # put
            if not (-1 <= delta <= 0):
                raise ValueError(f"Put delta must be between -1 and 0, got {delta}")
        
        if option_data["gamma"] < 0:
            raise ValueError(f"Gamma cannot be negative: {option_data['gamma']}")
        
        if option_data["vega"] < 0:
            raise ValueError(f"Vega cannot be negative: {option_data['vega']}")
        
        # Theta should be negative for long options
        if option_data["theta"] > 0:
            raise ValueError(f"Theta should be negative for long options: {option_data['theta']}")
        
        # Validate strike price
        strike = float(option_data["strike"]) if isinstance(option_data["strike"], Decimal) else option_data["strike"]
        if strike <= 0:
            raise ValueError(f"Strike price must be positive: {strike}")
        
        return True
    
    @staticmethod
    def validate_technical_indicators(indicators: Dict) -> bool:
        """Validate technical indicators are within expected ranges."""
        # RSI validation
        if "rsi_14" in indicators:
            rsi = indicators["rsi_14"]
            if not (0 <= rsi <= 100):
                raise ValueError(f"RSI must be between 0 and 100, got {rsi}")
        
        # ATR validation
        if "atr_14" in indicators:
            if indicators["atr_14"] <= 0:
                raise ValueError(f"ATR must be positive: {indicators['atr_14']}")
        
        # Bollinger Bands validation
        bb_fields = ["bollinger_upper", "bollinger_middle", "bollinger_lower"]
        if all(field in indicators for field in bb_fields):
            upper = indicators["bollinger_upper"]
            middle = indicators["bollinger_middle"]
            lower = indicators["bollinger_lower"]
            
            if not (upper > middle > lower):
                raise ValueError(f"Bollinger Bands must be ordered: upper({upper}) > middle({middle}) > lower({lower})")
        
        # Volume validation
        volume_fields = ["volume_sma_20", "volume_ratio"]
        for field in volume_fields:
            if field in indicators and indicators[field] <= 0:
                raise ValueError(f"{field} must be positive: {indicators[field]}")
        
        # MACD validation
        if "macd_histogram" in indicators and "macd" in indicators and "macd_signal" in indicators:
            expected_histogram = indicators["macd"] - indicators["macd_signal"]
            actual_histogram = indicators["macd_histogram"]
            
            # Allow small floating point differences
            if abs(expected_histogram - actual_histogram) > 1e-6:
                raise ValueError(f"MACD histogram mismatch: expected {expected_histogram}, got {actual_histogram}")
        
        return True
    
    @staticmethod
    def validate_ohlcv_dataframe(df: pd.DataFrame) -> bool:
        """Validate OHLCV DataFrame structure and data integrity."""
        required_columns = ["open", "high", "low", "close", "volume"]
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")
        
        # Validate data types
        for col in ["open", "high", "low", "close"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        # Validate positive values
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if (df[col] <= 0).any():
                raise ValueError(f"Column {col} contains non-positive values")
        
        if (df["volume"] < 0).any():
            raise ValueError("Volume contains negative values")
        
        # Validate OHLC relationships
        if not (df["high"] >= df["open"]).all():
            raise ValueError("High must be >= Open for all rows")
        
        if not (df["high"] >= df["close"]).all():
            raise ValueError("High must be >= Close for all rows")
        
        if not (df["low"] <= df["open"]).all():
            raise ValueError("Low must be <= Open for all rows")
        
        if not (df["low"] <= df["close"]).all():
            raise ValueError("Low must be <= Close for all rows")
        
        return True
    
    @staticmethod
    def validate_options_chain(chain: List[Dict]) -> bool:
        """Validate complete options chain data."""
        if not chain:
            raise ValueError("Options chain cannot be empty")
        
        # Validate each option in the chain
        for i, option in enumerate(chain):
            try:
                TestValidators.validate_option_contract(option)
            except ValueError as e:
                raise ValueError(f"Option {i} validation failed: {e}")
        
        # Check for consistent underlying symbol
        underlying_symbols = {opt["underlying_symbol"] for opt in chain}
        if len(underlying_symbols) > 1:
            raise ValueError(f"Options chain contains multiple underlying symbols: {underlying_symbols}")
        
        # Validate strike price ordering for same expiration
        expirations = {}
        for option in chain:
            exp = option["expiration"]
            if exp not in expirations:
                expirations[exp] = []
            expirations[exp].append(float(option["strike"]))
        
        # Each expiration should have ordered strikes
        for exp, strikes in expirations.items():
            sorted_strikes = sorted(strikes)
            if strikes != sorted_strikes:
                raise ValueError(f"Strikes not properly ordered for expiration {exp}")
        
        return True
    
    @staticmethod
    def validate_portfolio_metrics(metrics: Dict) -> bool:
        """Validate portfolio/position metrics."""
        # Score validations
        score_fields = ["momentum_score", "technical_score", "options_attractiveness", "overall_rating"]
        for field in score_fields:
            if field in metrics:
                score = metrics[field]
                if not (0 <= score <= 100):
                    raise ValueError(f"{field} must be between 0 and 100, got {score}")
        
        # Confidence validation
        if "confidence" in metrics:
            confidence = metrics["confidence"]
            if not (0 <= confidence <= 100):
                raise ValueError(f"Confidence must be between 0 and 100, got {confidence}")
        
        # Risk assessment validation
        if "risk_assessment" in metrics:
            valid_risks = ["low", "moderate", "high", "very_high"]
            if metrics["risk_assessment"] not in valid_risks:
                raise ValueError(f"Invalid risk assessment: {metrics['risk_assessment']}")
        
        return True
    
    @staticmethod
    def validate_api_response_structure(response: Dict, expected_fields: List[str]) -> bool:
        """Validate API response has expected structure."""
        # Check required fields
        missing_fields = [field for field in expected_fields if field not in response]
        if missing_fields:
            raise ValueError(f"API response missing required fields: {missing_fields}")
        
        # Check for error fields
        if "error" in response:
            raise ValueError(f"API response contains error: {response['error']}")
        
        if "status" in response and response["status"] != "ok":
            raise ValueError(f"API response status not ok: {response['status']}")
        
        return True
    
    @staticmethod
    def validate_financial_ratios(ratios: Dict) -> bool:
        """Validate financial ratios are within reasonable ranges.""" 
        ratio_ranges = {
            "pe_ratio": (0, 1000),  # P/E can be very high for growth stocks
            "debt_to_equity": (0, 10),  # D/E ratio
            "roe": (-1, 1),  # Return on Equity as decimal
            "profit_margin": (-1, 1),  # Profit margin as decimal
            "dividend_yield": (0, 0.15),  # Dividend yield as decimal
            "book_value": (0, 10000),  # Book value per share
        }
        
        for ratio_name, (min_val, max_val) in ratio_ranges.items():
            if ratio_name in ratios:
                value = ratios[ratio_name]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{ratio_name} value {value} outside reasonable range [{min_val}, {max_val}]")
        
        return True
    
    @staticmethod
    def validate_market_data_timestamps(data: Union[Dict, List[Dict]]) -> bool:
        """Validate timestamps in market data are reasonable."""
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data
        
        for item in data_list:
            # Check for timestamp fields
            timestamp_fields = ["timestamp", "updated", "last_trade_time", "date"]
            
            for field in timestamp_fields:
                if field in item:
                    timestamp = item[field]
                    
                    # Convert to datetime if needed
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, str):
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                dt = pd.to_datetime(timestamp)
                            except:
                                raise ValueError(f"Invalid timestamp format: {timestamp}")
                    elif isinstance(timestamp, datetime):
                        dt = timestamp
                    else:
                        raise ValueError(f"Invalid timestamp type: {type(timestamp)}")
                    
                    # Validate timestamp is reasonable (not too old or in future)
                    now = datetime.now()
                    if dt > now:
                        raise ValueError(f"Timestamp {dt} is in the future")
                    
                    # Check if timestamp is not too old (e.g., more than 10 years)
                    ten_years_ago = datetime(now.year - 10, now.month, now.day)
                    if dt < ten_years_ago:
                        raise ValueError(f"Timestamp {dt} is too old (more than 10 years)")
        
        return True


class DataQualityChecker:
    """Check data quality metrics for testing."""
    
    @staticmethod
    def calculate_data_completeness(data: Union[Dict, pd.DataFrame]) -> float:
        """Calculate data completeness percentage."""
        if isinstance(data, dict):
            total_fields = len(data)
            non_null_fields = sum(1 for v in data.values() if v is not None and v != "")
            return (non_null_fields / total_fields) * 100 if total_fields > 0 else 0
        
        elif isinstance(data, pd.DataFrame):
            total_cells = data.size
            non_null_cells = data.count().sum()
            return (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        return 0.0
    
    @staticmethod
    def check_data_freshness(timestamp: Union[str, int, float, datetime], max_age_hours: int = 24) -> bool:
        """Check if data is fresh enough for testing."""
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            raise ValueError(f"Invalid timestamp type: {type(timestamp)}")
        
        age = datetime.now() - dt
        return age.total_seconds() / 3600 <= max_age_hours
    
    @staticmethod
    def validate_data_consistency(data_list: List[Dict], key_field: str) -> bool:
        """Validate data consistency across multiple records."""
        if not data_list:
            return True
        
        # Check that key field exists in all records
        for i, record in enumerate(data_list):
            if key_field not in record:
                raise ValueError(f"Record {i} missing key field: {key_field}")
        
        # Check for duplicate keys
        keys = [record[key_field] for record in data_list]
        if len(keys) != len(set(keys)):
            duplicates = [k for k in keys if keys.count(k) > 1]
            raise ValueError(f"Duplicate keys found: {set(duplicates)}")
        
        return True
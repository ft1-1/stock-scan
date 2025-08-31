"""Mock data generators for comprehensive testing."""

import random
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class MockDataFactory:
    """Factory for generating realistic mock financial data."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducible tests."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_stock_quote(
        self, 
        symbol: str = "AAPL", 
        base_price: float = 150.0,
        volatility: float = 0.02
    ) -> Dict:
        """Create realistic stock quote data."""
        change = base_price * random.uniform(-volatility, volatility)
        last_price = base_price + change
        change_percent = (change / base_price) * 100
        
        # Generate bid/ask spread (typically 0.01-0.05% of price)
        spread = last_price * random.uniform(0.0001, 0.0005)
        bid = last_price - spread / 2
        ask = last_price + spread / 2
        
        # Generate daily range
        high = last_price * random.uniform(1.001, 1.02)
        low = last_price * random.uniform(0.98, 0.999)
        open_price = random.uniform(low, high)
        
        return {
            "symbol": symbol,
            "last_price": round(last_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": random.randint(1_000_000, 100_000_000),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "open": round(open_price, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    def create_options_chain_data(
        self,
        underlying_symbol: str,
        spot_price: float,
        days_to_expiration: int = 30,
        num_strikes: int = 11
    ) -> List[Dict]:
        """Create realistic options chain with proper Greeks."""
        chain = []
        exp_date = date.today() + timedelta(days=days_to_expiration)
        time_to_exp = days_to_expiration / 365.0
        
        # Generate strikes around ATM
        strike_spacing = spot_price * 0.025  # 2.5% strike spacing
        strikes = [
            round(spot_price + (i - num_strikes//2) * strike_spacing, 2)
            for i in range(num_strikes)
        ]
        
        for strike in strikes:
            if strike <= 0:
                continue
                
            # Calculate basic Greeks using simplified Black-Scholes
            moneyness = spot_price / strike
            iv = self._generate_realistic_iv(moneyness, time_to_exp)
            
            option_data = self._calculate_option_metrics(
                spot_price, strike, time_to_exp, iv, "call"
            )
            
            option_data.update({
                "option_symbol": f"{underlying_symbol}{exp_date.strftime('%y%m%d')}C{int(strike*1000):08d}",
                "underlying_symbol": underlying_symbol,
                "strike": Decimal(str(strike)),
                "expiration": exp_date,
                "option_type": "call",
                "days_to_expiration": days_to_expiration,
                "volume": random.randint(0, 10000),
                "open_interest": random.randint(100, 50000)
            })
            
            chain.append(option_data)
        
        return chain
    
    def _generate_realistic_iv(self, moneyness: float, time_to_exp: float) -> float:
        """Generate realistic implied volatility with volatility smile."""
        # Base IV around 20-30%
        base_iv = random.uniform(0.18, 0.32)
        
        # Volatility smile effect - higher IV for OTM options
        if moneyness < 0.95:  # OTM calls
            smile_adj = (0.95 - moneyness) * 0.3
        elif moneyness > 1.05:  # ITM calls  
            smile_adj = (moneyness - 1.05) * 0.15
        else:
            smile_adj = 0
            
        # Time decay effect on IV
        time_adj = max(0, (0.25 - time_to_exp) * 0.1)
        
        iv = base_iv + smile_adj + time_adj
        return max(0.05, min(1.0, iv))  # Clamp between 5% and 100%
    
    def _calculate_option_metrics(
        self,
        spot: float,
        strike: float, 
        time_to_exp: float,
        iv: float,
        option_type: str = "call"
    ) -> Dict:
        """Calculate option price and Greeks using simplified formulas."""
        # Simplified Black-Scholes for testing
        risk_free_rate = 0.05
        
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * iv**2) * time_to_exp) / (iv * np.sqrt(time_to_exp))
        d2 = d1 - iv * np.sqrt(time_to_exp)
        
        from scipy.stats import norm
        
        if option_type == "call":
            # Call option calculations
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_exp) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            # Put option calculations  
            price = strike * np.exp(-risk_free_rate * time_to_exp) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            
        gamma = norm.pdf(d1) / (spot * iv * np.sqrt(time_to_exp))
        theta = -(spot * norm.pdf(d1) * iv / (2 * np.sqrt(time_to_exp)) + 
                 risk_free_rate * strike * np.exp(-risk_free_rate * time_to_exp) * norm.cdf(d2))
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_exp)
        
        # Convert to daily theta
        theta = theta / 365
        
        # Add bid-ask spread
        spread = max(0.01, price * 0.005)  # 0.5% spread minimum $0.01
        bid = max(0.01, price - spread/2)
        ask = price + spread/2
        
        return {
            "bid": Decimal(str(round(bid, 2))),
            "ask": Decimal(str(round(ask, 2))),
            "last": Decimal(str(round(price, 2))),
            "delta": round(delta, 3),
            "gamma": round(gamma, 4),
            "theta": round(theta, 3),
            "vega": round(vega, 3),
            "implied_volatility": round(iv, 3)
        }
    
    def create_historical_data(
        self,
        symbol: str,
        days: int = 252,
        base_price: float = 100.0,
        trend: float = 0.0,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """Create realistic historical OHLCV data."""
        dates = pd.date_range(end=date.today(), periods=days, freq='D')
        
        # Generate price series with trend and volatility
        returns = np.random.normal(trend/252, volatility, days)
        returns[0] = 0  # First day no return
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(price_series):
            if i == 0:
                open_price = base_price
            else:
                # Open price with some gap from previous close
                gap = np.random.normal(0, volatility * 0.5)
                open_price = price_series[i-1] * (1 + gap)
            
            # Intraday range
            daily_range = close * volatility * random.uniform(0.5, 2.0)
            high = max(open_price, close) + daily_range * random.uniform(0, 0.7)
            low = min(open_price, close) - daily_range * random.uniform(0, 0.7)
            
            # Volume with some correlation to price movement
            base_volume = random.randint(500_000, 5_000_000)
            volatility_mult = 1 + abs(returns[i]) * 10  # Higher volume on big moves
            volume = int(base_volume * volatility_mult)
            
            data.append({
                "open": round(open_price, 2),
                "high": round(high, 2), 
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def create_fundamental_data(self, symbol: str) -> Dict:
        """Create mock fundamental data for a stock."""
        return {
            "symbol": symbol,
            "market_cap": random.randint(1_000_000_000, 500_000_000_000),
            "pe_ratio": random.uniform(10, 50),
            "eps": random.uniform(1, 20),
            "dividend_yield": random.uniform(0, 0.06),
            "book_value": random.uniform(10, 100),
            "debt_to_equity": random.uniform(0.1, 2.0),
            "roe": random.uniform(0.05, 0.25),
            "revenue_growth": random.uniform(-0.1, 0.3),
            "profit_margin": random.uniform(0.02, 0.25),
            "sector": random.choice([
                "Technology", "Healthcare", "Financial", "Consumer", 
                "Industrial", "Energy", "Materials", "Utilities"
            ])
        }


class OptionsChainGenerator:
    """Specialized generator for options chain data."""
    
    @staticmethod
    def generate_full_chain(
        symbol: str,
        spot_price: float,
        expirations: Optional[List[int]] = None
    ) -> Dict[str, List[Dict]]:
        """Generate complete options chain for multiple expirations."""
        if expirations is None:
            expirations = [7, 14, 21, 30, 45, 60, 90]
        
        factory = MockDataFactory()
        chain_data = {}
        
        for days in expirations:
            exp_date = (date.today() + timedelta(days=days)).strftime('%Y-%m-%d')
            chain_data[exp_date] = factory.create_options_chain_data(
                symbol, spot_price, days
            )
        
        return chain_data
    
    @staticmethod
    def generate_liquid_options(
        symbol: str,
        spot_price: float,
        min_volume: int = 100,
        min_open_interest: int = 500
    ) -> List[Dict]:
        """Generate options with specified liquidity requirements."""
        factory = MockDataFactory()
        options = factory.create_options_chain_data(symbol, spot_price)
        
        # Filter for liquidity and adjust volumes
        liquid_options = []
        for option in options:
            option["volume"] = random.randint(min_volume, min_volume * 50)
            option["open_interest"] = random.randint(min_open_interest, min_open_interest * 20)
            liquid_options.append(option)
        
        return liquid_options
    
    @staticmethod
    def generate_options_with_conditions(
        symbol: str,
        spot_price: float,
        conditions: Dict
    ) -> List[Dict]:
        """Generate options meeting specific test conditions."""
        factory = MockDataFactory()
        base_options = factory.create_options_chain_data(symbol, spot_price)
        
        modified_options = []
        for option in base_options:
            # Apply test conditions
            for field, value in conditions.items():
                if field in option:
                    if isinstance(value, dict) and "range" in value:
                        min_val, max_val = value["range"]
                        option[field] = random.uniform(min_val, max_val)
                    else:
                        option[field] = value
            
            modified_options.append(option)
        
        return modified_options


class MarketDataGenerator:
    """Generator for various market data scenarios."""
    
    @staticmethod
    def generate_market_hours_data() -> Dict:
        """Generate market hours and trading calendar data."""
        return {
            "market_open": "09:30:00",
            "market_close": "16:00:00",
            "timezone": "America/New_York",
            "is_open": True,
            "next_open": "2024-01-02 09:30:00",
            "next_close": "2024-01-01 16:00:00"
        }
    
    @staticmethod
    def generate_sector_data(num_sectors: int = 11) -> List[Dict]:
        """Generate sector performance data."""
        sectors = [
            "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
            "Communication Services", "Industrial", "Consumer Defensive", 
            "Energy", "Utilities", "Real Estate", "Basic Materials"
        ]
        
        sector_data = []
        for sector in sectors[:num_sectors]:
            sector_data.append({
                "sector": sector,
                "performance_1d": random.uniform(-3, 3),
                "performance_1w": random.uniform(-8, 8),
                "performance_1m": random.uniform(-15, 15),
                "market_cap": random.randint(100_000_000, 10_000_000_000),
                "avg_volume": random.randint(10_000_000, 1_000_000_000)
            })
        
        return sector_data
    
    @staticmethod
    def generate_earnings_calendar(days_ahead: int = 30) -> List[Dict]:
        """Generate earnings calendar data."""
        earnings = []
        symbols = [f"TEST{i:03d}" for i in range(50)]
        
        for _ in range(random.randint(10, 30)):
            earnings_date = date.today() + timedelta(days=random.randint(1, days_ahead))
            earnings.append({
                "symbol": random.choice(symbols),
                "earnings_date": earnings_date.isoformat(),
                "estimate": random.uniform(0.5, 5.0),
                "period": "Q4",
                "time": random.choice(["BMO", "AMC", "DMT"])  # Before/After/During market
            })
        
        return earnings
    
    @staticmethod
    def generate_news_sentiment(symbol: str, num_articles: int = 10) -> List[Dict]:
        """Generate news sentiment data for testing."""
        sentiments = ["positive", "negative", "neutral"]
        sources = ["Reuters", "Bloomberg", "MarketWatch", "Yahoo Finance", "CNBC"]
        
        articles = []
        for i in range(num_articles):
            articles.append({
                "headline": f"Test news article {i} for {symbol}",
                "source": random.choice(sources),
                "sentiment": random.choice(sentiments),
                "sentiment_score": random.uniform(-1, 1),
                "published_at": (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
                "relevance_score": random.uniform(0.1, 1.0)
            })
        
        return articles


class APIResponseMocker:
    """Mock API responses for external services."""
    
    @staticmethod
    def mock_eodhd_response(symbol: str, data_type: str = "quote") -> Dict:
        """Mock EODHD API responses."""
        factory = MockDataFactory()
        
        if data_type == "quote":
            quote = factory.create_stock_quote(symbol)
            return {
                "code": f"{symbol}.US",
                "timestamp": int(datetime.now().timestamp()),
                "gmtoffset": 0,
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["last_price"],
                "volume": quote["volume"],
                "previousClose": quote["last_price"] - quote["change"],
                "change": quote["change"],
                "change_p": quote["change_percent"]
            }
        
        elif data_type == "historical":
            hist_data = factory.create_historical_data(symbol, days=30)
            return [{
                "date": row.name.strftime('%Y-%m-%d'),
                "open": row["open"],
                "high": row["high"], 
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            } for _, row in hist_data.iterrows()]
        
        return {"error": "Unknown data type"}
    
    @staticmethod
    def mock_marketdata_response(symbol: str, data_type: str = "quote") -> Dict:
        """Mock MarketData.app API responses."""
        factory = MockDataFactory()
        
        if data_type == "quote":
            quote = factory.create_stock_quote(symbol)
            return {
                "s": "ok",
                "symbol": [symbol],
                "last": [quote["last_price"]],
                "change": [quote["change"]],
                "changepct": [quote["change_percent"]],
                "volume": [quote["volume"]],
                "updated": [int(datetime.now().timestamp())]
            }
        
        elif data_type == "options":
            options = factory.create_options_chain_data(symbol, 150.0)
            return {
                "s": "ok",
                "symbol": symbol,
                "options": options
            }
        
        return {"s": "error", "errmsg": "Unknown data type"}
    
    @staticmethod
    def mock_claude_response(analysis_type: str = "comprehensive") -> Dict:
        """Mock Claude API responses."""
        return {
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": MockDataFactory().create_analysis_response(analysis_type)
            }],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": random.randint(1000, 3000),
                "output_tokens": random.randint(200, 800)
            }
        }


# Extend MockDataFactory with analysis responses
def create_analysis_response(self, analysis_type: str = "comprehensive") -> str:
    """Create mock Claude analysis response."""
    responses = {
        "comprehensive": """
        Based on my analysis of the provided data:
        
        **Technical Analysis Score: 78/100**
        - RSI indicates moderate momentum
        - MACD shows bullish crossover
        - Price above key moving averages
        
        **Options Analysis Score: 72/100**  
        - Good liquidity in near-term contracts
        - Implied volatility at reasonable levels
        - Delta/Gamma profile favorable for calls
        
        **Overall Rating: 75/100**
        Recommendation: WATCHLIST - Monitor for entry opportunity
        """,
        
        "momentum": """
        **Momentum Analysis Score: 68/100**
        - Price momentum strengthening over past 5 days
        - Volume supporting the move
        - RSI not yet overbought
        """,
        
        "technical": """
        **Technical Score: 82/100**
        - Clear uptrend pattern established
        - Support holding at $145 level
        - Bollinger Bands indicating expansion
        """
    }
    
    return responses.get(analysis_type, responses["comprehensive"])

# Monkey patch the method
MockDataFactory.create_analysis_response = create_analysis_response
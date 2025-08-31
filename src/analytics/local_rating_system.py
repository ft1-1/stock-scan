"""
Comprehensive Local Rating System for Options Screening.

This module implements the complete local rating logic including:
- Eligibility gates (fail-fast rejection)
- Feature computation with normalization
- Sub-scores calculation (A-F components)
- Best call selection with contract scoring
- Penalties and red flags
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EligibilityResult:
    """Result of eligibility check."""
    passed: bool
    reason: Optional[str] = None
    details: Dict[str, Any] = None


@dataclass
class RatingResult:
    """Complete rating result for a stock."""
    symbol: str
    pre_score: float
    final_score: float
    sub_scores: Dict[str, float]
    selected_call: Optional[Dict[str, Any]]
    red_flags: List[str]
    penalties_applied: Dict[str, float]
    eligibility: EligibilityResult
    feature_values: Dict[str, Any]


class LocalRatingSystem:
    """
    Implements the complete local rating system for options screening.
    """
    
    def __init__(self):
        """Initialize the rating system."""
        settings = get_settings()
        
        # Eligibility gates from settings
        self.min_price = settings.rating_min_price
        self.min_market_cap = settings.rating_min_market_cap
        self.min_adv_shares = settings.rating_min_adv_shares
        self.min_adv_dollars = settings.rating_min_adv_dollars
        self.min_earnings_days = settings.rating_min_earnings_days
        self.max_data_missing_pct = settings.rating_max_data_missing_pct
        
        # Options requirements from settings
        self.min_dte = settings.rating_min_dte
        self.max_dte = settings.rating_max_dte
        self.min_oi = settings.rating_min_oi
        self.max_spread_pct = settings.rating_max_spread_pct
        
    def check_eligibility(
        self,
        symbol: str,
        current_price: float,
        market_cap: Optional[float],
        adv_shares: Optional[float],
        adv_dollars: Optional[float],
        days_to_earnings: Optional[int],
        options_chain: List[Dict],
        data_completeness: float,
        is_earnings_strategy: bool = False
    ) -> EligibilityResult:
        """
        Check eligibility gates (fail-fast rejection).
        
        Returns True only if ALL gates pass.
        """
        # Price gate
        if current_price < self.min_price:
            return EligibilityResult(
                passed=False,
                reason=f"Price ${current_price:.2f} below minimum ${self.min_price}"
            )
        
        # Market cap gate
        if market_cap and market_cap < self.min_market_cap:
            return EligibilityResult(
                passed=False,
                reason=f"Market cap ${market_cap/1e9:.1f}B below minimum ${self.min_market_cap/1e9:.1f}B"
            )
        
        # ADV shares gate
        if adv_shares and adv_shares < self.min_adv_shares:
            return EligibilityResult(
                passed=False,
                reason=f"ADV {adv_shares/1e6:.1f}M shares below minimum {self.min_adv_shares/1e6:.1f}M"
            )
        
        # ADV dollars gate
        if adv_dollars and adv_dollars < self.min_adv_dollars:
            return EligibilityResult(
                passed=False,
                reason=f"ADV ${adv_dollars/1e6:.1f}M below minimum ${self.min_adv_dollars/1e6:.1f}M"
            )
        
        # Earnings gate (except for earnings strategy)
        if not is_earnings_strategy and days_to_earnings is not None:
            if days_to_earnings < self.min_earnings_days:
                return EligibilityResult(
                    passed=False,
                    reason=f"Earnings in {days_to_earnings} days (minimum {self.min_earnings_days})"
                )
        
        # Options availability gate
        valid_options = self._find_valid_options(options_chain)
        if not valid_options:
            return EligibilityResult(
                passed=False,
                reason=f"No valid options ({self.min_dte}-{self.max_dte} DTE, OI≥{self.min_oi}, spread≤{self.max_spread_pct}%)"
            )
        
        # Data completeness gate
        if data_completeness < (1 - self.max_data_missing_pct):
            return EligibilityResult(
                passed=False,
                reason=f"Data completeness {data_completeness:.1%} below minimum {(1 - self.max_data_missing_pct):.0%}"
            )
        
        return EligibilityResult(
            passed=True,
            details={
                'price': current_price,
                'market_cap': market_cap,
                'adv_shares': adv_shares,
                'adv_dollars': adv_dollars,
                'days_to_earnings': days_to_earnings,
                'valid_options_count': len(valid_options),
                'data_completeness': data_completeness
            }
        )
    
    def _find_valid_options(self, options_chain: List[Dict]) -> List[Dict]:
        """Find options meeting basic criteria."""
        valid = []
        for opt in options_chain:
            if opt.get('option_type') != 'call':
                continue
            
            dte = opt.get('days_to_expiration', 0)
            if not (self.min_dte <= dte <= self.max_dte):
                continue
            
            oi = int(opt.get('open_interest') or 0)
            if oi < self.min_oi:
                continue
            
            bid = float(opt.get('bid', 0))
            ask = float(opt.get('ask', 0))
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_pct = ((ask - bid) / mid) * 100
                if spread_pct > self.max_spread_pct:
                    continue
            else:
                continue
            
            valid.append(opt)
        
        return valid
    
    def compute_features(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        enhanced_data: Dict[str, Any],
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute all features for rating calculation.
        
        Returns normalized feature values.
        """
        features = {}
        
        # Extract data
        close = ohlcv_df['close']
        volume = ohlcv_df['volume']
        
        # 1. Trend & Momentum features
        features.update(self._compute_trend_momentum(ohlcv_df, sector))
        
        # 2. Squeeze/Volatility features
        features.update(self._compute_squeeze_volatility(ohlcv_df))
        
        # 3. Options features (computed later with best call)
        # Will be added after best call selection
        
        # 4. IV features
        features.update(self._compute_iv_features(enhanced_data))
        
        # 5. Fundamentals features
        features.update(self._compute_fundamentals(enhanced_data, sector))
        
        # 6. News/Event features
        features.update(self._compute_news_events(enhanced_data, ohlcv_df))
        
        return features
    
    def _compute_trend_momentum(self, df: pd.DataFrame, sector: Optional[str]) -> Dict[str, float]:
        """Compute trend and momentum features."""
        features = {}
        close = df['close']
        
        # Returns
        if len(df) >= 21:
            features['ret_21d'] = (close.iloc[-1] / close.iloc[-22] - 1) * 100
        else:
            features['ret_21d'] = 0
        
        if len(df) >= 63:
            features['ret_63d'] = (close.iloc[-1] / close.iloc[-64] - 1) * 100
        else:
            features['ret_63d'] = 0
        
        # Moving averages
        if len(df) >= 50:
            ma50 = close.rolling(50).mean().iloc[-1]
            features['pct_above_ma50'] = ((close.iloc[-1] - ma50) / ma50) * 100
        else:
            features['pct_above_ma50'] = 0
        
        if len(df) >= 200:
            ma200 = close.rolling(200).mean().iloc[-1]
            features['pct_above_ma200'] = ((close.iloc[-1] - ma200) / ma200) * 100
            features['ma50_above_ma200'] = 1 if ma50 > ma200 else 0
        else:
            features['pct_above_ma200'] = 0
            features['ma50_above_ma200'] = 0
        
        # RSI
        features['rsi14'] = self._calculate_rsi(close, 14)
        
        # ADX
        features['adx14'] = self._calculate_adx(df, 14)
        
        # Breakouts
        if len(df) >= 55:
            high_55d = df['high'].iloc[-55:].max()
            features['breakout_55d'] = 1 if close.iloc[-1] > high_55d else 0
        else:
            features['breakout_55d'] = 0
        
        # Weekly breakout
        features['weekly_breakout'] = self._check_weekly_breakout(df)
        
        # RS vs sector (placeholder - needs sector data)
        features['rs_vs_sector_21d'] = features['ret_21d']  # TODO: Compare to sector
        features['rs_vs_sector_63d'] = features['ret_63d']  # TODO: Compare to sector
        
        return features
    
    def _compute_squeeze_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute squeeze and volatility features."""
        features = {}
        close = df['close']
        
        # ATR%
        atr20 = self._calculate_atr(df, 20)
        features['atr20_pct'] = (atr20 / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0
        
        # ATR percentile (1-year)
        if len(df) >= 252:
            atr_series = pd.Series(index=df.index)
            for i in range(19, len(df)):
                window_df = df.iloc[max(0, i-19):i+1]
                atr_series.iloc[i] = self._calculate_atr(window_df, 20)
            
            atr_pct_series = (atr_series / df['close']) * 100
            current_atr_pct = features['atr20_pct']
            features['atr_percentile'] = (atr_pct_series < current_atr_pct).sum() / len(atr_pct_series.dropna())
        else:
            features['atr_percentile'] = 0.5
        
        # Bollinger Band width
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
        features['bb_width'] = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
        
        # BB width percentile
        if len(df) >= 252:
            bb_width_series = pd.Series(index=df.index)
            for i in range(19, len(df)):
                window_close = close.iloc[max(0, i-19):i+1]
                upper, middle, lower = self._calculate_bollinger_bands(window_close, 20, 2)
                bb_width_series.iloc[i] = ((upper - lower) / middle) * 100 if middle > 0 else 0
            
            current_bb_width = features['bb_width']
            features['bb_width_percentile'] = (bb_width_series < current_bb_width).sum() / len(bb_width_series.dropna())
        else:
            features['bb_width_percentile'] = 0.5
        
        # Keltner width
        kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(
            df['high'], df['low'], df['close'], 20, 2
        )
        features['keltner_width'] = ((kc_upper - kc_lower) / kc_middle) * 100 if kc_middle > 0 else 0
        
        # Expansion tick
        if len(df) >= 5 and features['bb_width_percentile'] < 0.3:
            bb_width_5d_ago = self._calculate_bollinger_band_width(close.iloc[-25:-5], 20, 2)
            bb_width_change = features['bb_width'] - bb_width_5d_ago
            features['expansion_tick'] = np.clip(bb_width_change / 2, -1, 1)  # Normalize to [-1, 1]
        else:
            features['expansion_tick'] = 0
        
        return features
    
    def _compute_iv_features(self, enhanced_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute IV-related features."""
        features = {}
        
        # IV percentile (from enhanced data or options)
        iv_percentile = enhanced_data.get('iv_percentile')
        
        # If not provided, try to calculate from options chain
        if iv_percentile is None:
            options_chain = enhanced_data.get('options_chain', [])
            if options_chain:
                # Get ATM option IV
                current_price = enhanced_data.get('quote', {}).get('last_price', 0)
                if current_price > 0:
                    atm_iv = self._get_atm_implied_volatility(options_chain, current_price)
                    # Since we don't have historical IV data, use a rough estimate
                    # Assume typical IV range is 15-50% for most stocks
                    if atm_iv:
                        # Map IV to percentile: 15% = 0th, 32.5% = 50th, 50% = 100th
                        iv_percentile = min(100, max(0, (atm_iv - 0.15) / 0.35 * 100))
        
        if iv_percentile is not None:
            features['iv_percentile_atm'] = iv_percentile / 100  # Normalize to [0, 1]
        else:
            features['iv_percentile_atm'] = 0.5  # Default to middle
        
        # Term structure (placeholder)
        features['iv_term_structure'] = 0  # TODO: Implement if near/next month IV available
        
        return features
    
    def _get_atm_implied_volatility(self, options_chain: List[Dict], current_price: float) -> Optional[float]:
        """Get implied volatility of ATM option."""
        best_option = None
        min_distance = float('inf')
        
        for option in options_chain:
            if option.get('type') == 'call':
                strike = float(option.get('strike', 0))
                distance = abs(strike - current_price)
                if distance < min_distance:
                    min_distance = distance
                    best_option = option
        
        if best_option:
            return float(best_option.get('implied_volatility', 0))
        return None
    
    def _compute_fundamentals(self, enhanced_data: Dict[str, Any], sector: Optional[str]) -> Dict[str, float]:
        """Compute fundamental features."""
        features = {}
        
        fundamentals = enhanced_data.get('fundamentals', {})
        
        # Margins - check both possible data structures
        if isinstance(fundamentals, dict):
            # Try the filtered structure first (from _filter_fundamental_data)
            financial_health = fundamentals.get('financial_health', {})
            if financial_health:
                # Profit margin as proxy for gross margin
                profit_margin = financial_health.get('profit_margin')
                features['gross_margin_ttm'] = profit_margin / 100 if profit_margin else 0.5
                
                # Operating margin
                op_margin = financial_health.get('operating_margin')
                features['operating_margin_ttm'] = op_margin / 100 if op_margin else 0.5
                
                # Revenue growth
                revenue_growth = financial_health.get('revenue_growth_yoy')
                features['revenue_growth_yoy'] = revenue_growth / 100 if revenue_growth else 0
            else:
                # Fallback to raw EODHD structure
                highlights = fundamentals.get('Highlights', {})
                if highlights:
                    features['gross_margin_ttm'] = (highlights.get('ProfitMargin', 50)) / 100
                    features['operating_margin_ttm'] = (highlights.get('OperatingMarginTTM', 50)) / 100
                    features['revenue_growth_yoy'] = (highlights.get('QuarterlyRevenueGrowthYOY', 0)) / 100
                else:
                    features['gross_margin_ttm'] = 0.5
                    features['operating_margin_ttm'] = 0.5
                    features['revenue_growth_yoy'] = 0
            
            # Short interest - check multiple locations
            shares_stats = fundamentals.get('SharesStats', {})
            if shares_stats:
                short_float = shares_stats.get('ShortPercent', shares_stats.get('ShortPercentFloat'))
                features['short_pct_float'] = short_float / 100 if short_float else 0
            else:
                features['short_pct_float'] = 0
        else:
            features['gross_margin_ttm'] = 0.5
            features['operating_margin_ttm'] = 0.5
            features['revenue_growth_yoy'] = 0
            features['short_pct_float'] = 0
        
        # TODO: Convert to sector-relative percentiles
        
        return features
    
    def _compute_news_events(self, enhanced_data: Dict[str, Any], df: pd.DataFrame) -> Dict[str, float]:
        """Compute news and event features."""
        features = {}
        
        # News count
        news = enhanced_data.get('news', [])
        if isinstance(news, list):
            features['news_count_10d'] = min(len(news), 10) / 10  # Normalize to [0, 1]
        else:
            features['news_count_10d'] = 0
        
        # Headline drift (simplified)
        if len(df) >= 5:
            features['headline_drift_5d'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
        else:
            features['headline_drift_5d'] = 0
        
        # Days to earnings
        earnings = enhanced_data.get('earnings', {})
        if isinstance(earnings, dict):
            next_date = earnings.get('next_earnings_date')
            if next_date:
                try:
                    if isinstance(next_date, str):
                        next_date = datetime.strptime(next_date, '%Y-%m-%d').date()
                    days_to = (next_date - date.today()).days
                    features['days_to_earnings'] = days_to
                except:
                    features['days_to_earnings'] = 30
            else:
                features['days_to_earnings'] = 30
        else:
            features['days_to_earnings'] = 30
        
        # Corporate action quiet (placeholder)
        features['corporate_action_quiet'] = 1  # TODO: Check for splits/divs
        
        return features
    
    def calculate_sub_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate sub-scores (A-E) from features.
        News removed, points redistributed to other components.
        
        Returns dict with component scores.
        """
        scores = {}
        
        # A. Trend & Momentum (40 pts - increased from 35)
        trend_score = 0
        
        # ret_21d: -10% → 0, +15% → 1 (10 pts)
        ret_21d_norm = np.clip((features.get('ret_21d', 0) + 10) / 25, 0, 1)
        trend_score += ret_21d_norm * 10
        
        # ret_63d: -15% → 0, +30% → 1 (8 pts)
        ret_63d_norm = np.clip((features.get('ret_63d', 0) + 15) / 45, 0, 1)
        trend_score += ret_63d_norm * 8
        
        # MA stack bonus
        if features.get('ma50_above_ma200', 0):
            trend_score += 0.5
        
        # %above_MA50: -5% → 0, +15% → 1 (6 pts)
        ma50_norm = np.clip((features.get('pct_above_ma50', 0) + 5) / 20, 0, 1)
        trend_score += ma50_norm * 6
        
        # %above_MA200: 0% → 0, +25% → 1 (6 pts)
        ma200_norm = np.clip(features.get('pct_above_ma200', 0) / 25, 0, 1)
        trend_score += ma200_norm * 6
        
        # RSI quality: bell curve, peak at 60 (3 pts)
        rsi = features.get('rsi14', 50)
        if 45 <= rsi <= 75:
            rsi_score = 1 - abs(rsi - 60) / 15
        else:
            rsi_score = 0
        trend_score += rsi_score * 3
        
        # ADX: 15 → 0, 30+ → 1 (2 pts)
        adx_norm = np.clip((features.get('adx14', 15) - 15) / 15, 0, 1)
        trend_score += adx_norm * 2
        
        # Breakouts
        if features.get('breakout_55d', 0):
            trend_score += 1
        
        weekly_breakout = features.get('weekly_breakout', 0)
        trend_score += weekly_breakout  # Can be -1 to +1
        
        scores['trend_momentum'] = min(trend_score, 40)  # Increased from 35
        
        # B. Squeeze/Breakout Context (12 pts - increased from 10)
        squeeze_score = 0
        
        # ATR percentile: lower is better (5 pts)
        atr_pctl = features.get('atr_percentile', 0.5)
        squeeze_score += (1 - atr_pctl) * 5
        
        # BB width percentile: lower is better (3 pts)
        bb_pctl = features.get('bb_width_percentile', 0.5)
        squeeze_score += (1 - bb_pctl) * 3
        
        # Expansion tick (2 pts)
        expansion = features.get('expansion_tick', 0)
        squeeze_score += max(0, expansion) * 2
        
        scores['squeeze_breakout'] = min(squeeze_score, 12)  # Increased from 10
        
        # C. Options Quality & Fit (23 pts - increased from 20) - Will be set after best call selection
        scores['options_quality'] = 0  # Placeholder
        
        # D. IV Value (15 pts)
        iv_score = 0
        
        # IV percentile: lower is better (12 pts)
        iv_pctl = features.get('iv_percentile_atm', 0.5)
        iv_score += (1 - iv_pctl) * 12
        
        # Term structure bonus (3 pts)
        if features.get('iv_term_structure', 0) < 0:
            iv_score += 3
        
        scores['iv_value'] = min(iv_score, 15)
        
        # E. Fundamentals (10 pts)
        fund_score = 0
        
        # Gross margin (3 pts) - using raw value as proxy for percentile
        gross_margin = features.get('gross_margin_ttm', 0.3)  # Default to 30% if missing
        fund_score += min(gross_margin, 1) * 3  # Cap at 100%
        
        # Operating margin (3 pts)
        op_margin = features.get('operating_margin_ttm', 0.15)  # Default to 15% if missing
        fund_score += min(op_margin, 1) * 3  # Cap at 100%
        
        # Revenue growth (2 pts)
        revenue_growth = features.get('revenue_growth_yoy', 0.1)  # Default to 10% if missing
        fund_score += min(max(revenue_growth, 0), 1) * 2
        
        # Short interest: lower is better (2 pts)
        short_pct = features.get('short_pct_float', 0.05)  # Default to 5% if missing
        fund_score += (1 - min(short_pct / 0.25, 1)) * 2
        
        scores['fundamentals'] = min(fund_score, 10)
        
        # F. News/Events - REMOVED from local calculation
        # News will only be considered in AI analysis
        scores['news_events'] = 0
        
        return scores
    
    def select_best_call(
        self,
        options_chain: List[Dict],
        current_price: float,
        iv_percentile: Optional[float] = None
    ) -> Tuple[Optional[Dict], float]:
        """
        Select best call option using contract scoring.
        
        Returns (best_contract, options_quality_score).
        """
        valid_options = self._find_valid_options(options_chain)
        
        if not valid_options:
            return None, 0
        
        best_contract = None
        best_score = -1
        
        for opt in valid_options:
            # Hard rejects
            spread_pct = self._calculate_spread_pct(opt)
            if spread_pct > 4:
                continue
            
            oi = int(opt.get('open_interest') or 0)
            if oi < 100:
                continue
            
            # Calculate contract score
            contract_score = self._calculate_contract_score(opt, iv_percentile)
            
            if contract_score > best_score:
                best_score = contract_score
                best_contract = opt
        
        # Calculate options quality score for component C
        if best_contract:
            quality_score = self._calculate_options_quality_score(best_contract)
        else:
            quality_score = 0
        
        return best_contract, quality_score
    
    def _calculate_contract_score(self, contract: Dict, iv_percentile: Optional[float]) -> float:
        """
        Calculate per-contract score.
        
        Score = 0.4·Liquidity + 0.3·IVValue + 0.3·Fit
        """
        score = 0
        
        # Liquidity (40%)
        spread_pct = self._calculate_spread_pct(contract)
        spread_score = max(0, 1 - spread_pct / 4)  # 0% → 1, 4% → 0
        
        oi = float(contract.get('open_interest') or 0)
        volume = float(contract.get('volume') or 0)
        oi_vol_score = min(oi / 1000, 1) * 0.6 + min(volume / 500, 1) * 0.4
        
        liquidity_score = (spread_score * 0.6 + oi_vol_score * 0.4)
        score += liquidity_score * 0.4
        
        # IV Value (30%)
        if iv_percentile is not None:
            iv_score = 1 - iv_percentile / 100
        else:
            iv_score = 0.5
        score += iv_score * 0.3
        
        # Fit (30%)
        delta = float(contract.get('delta', 0))
        delta_score = max(0, 1 - abs(delta - 0.60) / 0.15)
        
        dte = int(contract.get('days_to_expiration') or 0)
        if 45 <= dte <= 90:
            if dte <= 60:
                dte_score = (dte - 45) / 15
            else:
                dte_score = (90 - dte) / 30
        else:
            dte_score = 0
        
        fit_score = (delta_score * 0.6 + dte_score * 0.4)
        score += fit_score * 0.3
        
        return score
    
    def _calculate_options_quality_score(self, contract: Dict) -> float:
        """
        Calculate options quality score for component C (23 pts - increased from 20).
        """
        score = 0
        
        # Spread% (7 pts): 1.5% → 1, 4% → 0
        spread_pct = self._calculate_spread_pct(contract)
        spread_score = max(0, 1 - (spread_pct - 1.5) / 2.5)
        score += spread_score * 7
        
        # OI & Volume (6 pts)
        oi = float(contract.get('open_interest') or 0)
        volume = float(contract.get('volume') or 0)
        oi_vol_score = min(oi / 1000, 1) * 0.6 + min(volume / 500, 1) * 0.4
        score += oi_vol_score * 6
        
        # Delta closeness (5 pts)
        delta = float(contract.get('delta', 0.6))
        delta_score = max(0, 1 - abs(delta - 0.60) / 0.15)
        score += delta_score * 5
        
        # DTE closeness (2 pts)
        dte = int(contract.get('days_to_expiration') or 60)
        if 45 <= dte <= 90:
            if dte <= 60:
                dte_score = (dte - 45) / 15
            else:
                dte_score = (90 - dte) / 30
        else:
            dte_score = 0
        score += dte_score * 2
        
        # Earnings gap (3 pts) - placeholder
        score += 3  # Assume safe for now
        
        return min(score, 23)
    
    def apply_penalties(
        self,
        pre_score: float,
        features: Dict[str, Any],
        selected_call: Optional[Dict]
    ) -> Tuple[float, List[str], Dict[str, float]]:
        """
        Apply penalties and identify red flags.
        
        Returns (final_score, red_flags, penalties_applied).
        """
        final_score = pre_score
        red_flags = []
        penalties = {}
        
        # Earnings within 7 days (BLOCK for non-earnings strategy)
        days_to_earn = features.get('days_to_earnings', 30)
        if days_to_earn < 7:
            red_flags.append(f"Earnings in {days_to_earn} days")
            # This should have been caught in eligibility
        
        # IV percentile > 85%
        iv_pctl = features.get('iv_percentile_atm', 0.5) * 100
        if iv_pctl > 85:
            penalty = 8
            final_score -= penalty
            penalties['high_iv'] = penalty
            red_flags.append(f"IV percentile {iv_pctl:.0f}% > 85%")
        
        # Spread% > 3.5%
        if selected_call:
            spread_pct = self._calculate_spread_pct(selected_call)
            if spread_pct > 3.5:
                penalty = 5
                final_score -= penalty
                penalties['wide_spread'] = penalty
                red_flags.append(f"Spread {spread_pct:.1f}% > 3.5%")
        
        # Short % > 20%
        short_pct = features.get('short_pct_float', 0) * 100
        if short_pct > 20:
            penalty = 4
            final_score -= penalty
            penalties['high_short'] = penalty
            red_flags.append(f"Short interest {short_pct:.1f}% > 20%")
        
        # Price gap ≥ +10% in last 2 days
        if len(features) > 0:  # Check if we have price data
            ret_2d = features.get('ret_2d', 0)  # Need to add this feature
            if ret_2d >= 10:
                penalty = 3
                final_score -= penalty
                penalties['price_gap'] = penalty
                red_flags.append(f"Price gap +{ret_2d:.1f}% in 2 days")
        
        # Failed breakout (negative weekly close)
        weekly_breakout = features.get('weekly_breakout', 0)
        if weekly_breakout < 0:
            penalty = 5
            final_score -= penalty
            penalties['failed_breakout'] = penalty
            red_flags.append("Failed weekly breakout")
        
        return max(0, final_score), red_flags, penalties
    
    def rate_opportunity(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        enhanced_data: Dict[str, Any],
        options_chain: List[Dict],
        current_price: float,
        is_earnings_strategy: bool = False
    ) -> RatingResult:
        """
        Complete rating process for an opportunity.
        
        This is the main entry point for rating.
        """
        # Extract required data for eligibility
        fundamentals = enhanced_data.get('fundamentals', {})
        market_cap = None
        if isinstance(fundamentals, dict):
            market_cap = fundamentals.get('MarketCapitalization')
        
        # Calculate ADV
        if len(ohlcv_df) >= 20:
            adv_shares = float(ohlcv_df['volume'].iloc[-20:].mean())
            avg_price = float(ohlcv_df['close'].iloc[-20:].mean())
            adv_dollars = adv_shares * avg_price
        else:
            adv_shares = 0
            adv_dollars = 0
        
        # Days to earnings
        days_to_earnings = self._get_days_to_earnings(enhanced_data)
        
        # Data completeness
        data_completeness = self._calculate_data_completeness(enhanced_data)
        
        # Check eligibility
        eligibility = self.check_eligibility(
            symbol=symbol,
            current_price=current_price,
            market_cap=market_cap,
            adv_shares=adv_shares,
            adv_dollars=adv_dollars,
            days_to_earnings=days_to_earnings,
            options_chain=options_chain,
            data_completeness=data_completeness,
            is_earnings_strategy=is_earnings_strategy
        )
        
        if not eligibility.passed:
            return RatingResult(
                symbol=symbol,
                pre_score=0,
                final_score=0,
                sub_scores={},
                selected_call=None,
                red_flags=[eligibility.reason],
                penalties_applied={},
                eligibility=eligibility,
                feature_values={}
            )
        
        # Compute features
        sector = fundamentals.get('Sector') if isinstance(fundamentals, dict) else None
        features = self.compute_features(symbol, ohlcv_df, enhanced_data, sector)
        
        # Add ADV to features
        features['adv_shares'] = adv_shares
        features['adv_dollars'] = adv_dollars
        
        # Select best call option
        iv_percentile = enhanced_data.get('iv_percentile')
        selected_call, options_quality_score = self.select_best_call(
            options_chain, current_price, iv_percentile
        )
        
        # Calculate sub-scores
        sub_scores = self.calculate_sub_scores(features)
        
        # Add options quality score
        sub_scores['options_quality'] = options_quality_score
        
        # Calculate pre-score
        pre_score = sum(sub_scores.values())
        pre_score = min(pre_score, 100)
        
        # Apply penalties
        final_score, red_flags, penalties = self.apply_penalties(
            pre_score, features, selected_call
        )
        
        return RatingResult(
            symbol=symbol,
            pre_score=pre_score,
            final_score=final_score,
            sub_scores=sub_scores,
            selected_call=selected_call,
            red_flags=red_flags,
            penalties_applied=penalties,
            eligibility=eligibility,
            feature_values=features
        )
    
    # Helper methods
    def _calculate_spread_pct(self, contract: Dict) -> float:
        """Calculate spread percentage."""
        bid = float(contract.get('bid', 0))
        ask = float(contract.get('ask', 0))
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            return ((ask - bid) / mid) * 100
        return 100  # Max if no bid/ask
    
    def _get_days_to_earnings(self, enhanced_data: Dict[str, Any]) -> Optional[int]:
        """Extract days to earnings."""
        earnings = enhanced_data.get('earnings', {})
        if isinstance(earnings, dict):
            next_date = earnings.get('next_earnings_date')
            if next_date:
                try:
                    if isinstance(next_date, str):
                        next_date = datetime.strptime(next_date, '%Y-%m-%d').date()
                    return (next_date - date.today()).days
                except:
                    pass
        return None
    
    def _calculate_data_completeness(self, enhanced_data: Dict[str, Any]) -> float:
        """Calculate data completeness score."""
        # Check if we're in minimal data mode (no news/earnings present)
        has_news = 'news' in enhanced_data and enhanced_data.get('news')
        has_earnings = 'earnings' in enhanced_data and enhanced_data.get('earnings')
        
        # If news and earnings are not present, we're in local rating mode
        if not has_news and not has_earnings:
            # For local rating mode, check essential technical fields
            required_fields = ['quote', 'historical_prices', 'options_chain', 'technicals']
        else:
            # For full AI analysis mode, check all fields
            required_fields = [
                'quote', 'fundamentals', 'news', 'earnings',
                'historical_prices', 'options_chain', 'technicals'
            ]
        
        present = sum(1 for field in required_fields if enhanced_data.get(field))
        return present / len(required_fields)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX."""
        if len(df) < period * 2:
            return 0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate ATR."""
        if len(df) < period:
            return 0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
    
    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            current = prices.iloc[-1] if len(prices) > 0 else 0
            return current, current, current
        
        middle = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def _calculate_bollinger_band_width(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> float:
        """Calculate BB width."""
        upper, middle, lower = self._calculate_bollinger_bands(prices, period, std_dev)
        return ((upper - lower) / middle) * 100 if middle > 0 else 0
    
    def _calculate_keltner_channels(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20, multiplier: float = 2
    ) -> Tuple[float, float, float]:
        """Calculate Keltner Channels."""
        if len(close) < period:
            current = close.iloc[-1] if len(close) > 0 else 0
            return current, current, current
        
        middle = close.rolling(period).mean().iloc[-1]
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean().iloc[-1]
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower
    
    def _check_weekly_breakout(self, df: pd.DataFrame) -> float:
        """Check for weekly breakout."""
        if len(df) < 10:
            return 0
        
        # Get weekly high from prior week
        prior_week_high = df['high'].iloc[-10:-5].max()
        current_close = df['close'].iloc[-1]
        
        if current_close > prior_week_high * 1.02:  # 2% above
            return 1
        elif current_close < prior_week_high * 0.98:  # 2% below
            return -1
        else:
            return 0
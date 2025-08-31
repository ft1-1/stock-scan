"""
Enhanced Technical Analysis Module
Implements all required indicators for options screening as per requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EnhancedTechnicalAnalysis:
    """Enhanced technical analysis with all required indicators."""
    
    @staticmethod
    def calculate_trend_momentum(
        df: pd.DataFrame,
        current_price: float,
        sector_etf_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate Trend & Momentum indicators.
        
        Required:
        - 21-day and 63-day returns
        - % above 50/200 DMA
        - New 55-day high flag
        - RS vs sector (excess return over sector ETF, 21-63d)
        - RSI in [50-70] and ADX >= 20 as quality checks
        """
        indicators = {}
        
        try:
            # Ensure numeric types (handle string values from API)
            close = pd.to_numeric(df['close'], errors='coerce')
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            
            # Drop any NaN values that resulted from conversion
            close = close.dropna()
            high = high.dropna()
            low = low.dropna()
            
            # 1. Period returns
            if len(close) >= 21:
                indicators['return_21d'] = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100
            if len(close) >= 63:
                indicators['return_63d'] = ((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]) * 100
            
            # 2. % above moving averages
            if len(close) >= 50:
                sma_50 = close.rolling(50).mean().iloc[-1]
                indicators['pct_above_sma50'] = ((current_price - sma_50) / sma_50) * 100
                indicators['sma_50'] = sma_50
            
            if len(close) >= 200:
                sma_200 = close.rolling(200).mean().iloc[-1]
                indicators['pct_above_sma200'] = ((current_price - sma_200) / sma_200) * 100
                indicators['sma_200'] = sma_200
            
            # 3. New 55-day high flag
            if len(high) >= 55:
                high_55d = high.iloc[-55:].max()
                indicators['new_55d_high'] = current_price >= high_55d
                indicators['high_55d'] = high_55d
            
            # 4. Relative Strength vs Sector ETF
            if sector_etf_data is not None and len(sector_etf_data) >= 63:
                sector_close = pd.to_numeric(sector_etf_data['close'], errors='coerce').dropna()
                
                # 21-day RS
                if len(close) >= 21 and len(sector_close) >= 21:
                    stock_return_21d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
                    sector_return_21d = (sector_close.iloc[-1] - sector_close.iloc[-21]) / sector_close.iloc[-21]
                    indicators['rs_vs_sector_21d'] = (stock_return_21d - sector_return_21d) * 100
                
                # 63-day RS
                if len(close) >= 63 and len(sector_close) >= 63:
                    stock_return_63d = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
                    sector_return_63d = (sector_close.iloc[-1] - sector_close.iloc[-63]) / sector_close.iloc[-63]
                    indicators['rs_vs_sector_63d'] = (stock_return_63d - sector_return_63d) * 100
            
            # 5. RSI (14-period)
            if len(close) >= 15:
                rsi = EnhancedTechnicalAnalysis._calculate_rsi(close, 14)
                indicators['rsi'] = rsi
                indicators['rsi_in_range'] = 50 <= rsi <= 70  # Quality check
            
            # 6. ADX (14-period)
            if len(df) >= 15:
                adx = EnhancedTechnicalAnalysis._calculate_adx(high, low, close, 14)
                indicators['adx'] = adx
                indicators['adx_above_20'] = adx >= 20  # Quality check
            
            # 7. Quality score
            quality_checks = []
            if 'rsi_in_range' in indicators:
                quality_checks.append(indicators['rsi_in_range'])
            if 'adx_above_20' in indicators:
                quality_checks.append(indicators['adx_above_20'])
            
            if quality_checks:
                indicators['trend_quality_score'] = sum(quality_checks) / len(quality_checks)
            
        except Exception as e:
            logger.error(f"Error calculating trend momentum: {e}")
        
        return indicators
    
    @staticmethod
    def calculate_squeeze_breakout(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Squeeze & Breakout indicators.
        
        Required:
        - ATR% = ATR(20) / close and its 1-year percentile
        - BB width = (BBU-BBL)/BBM and/or Keltner width; rank as 1-year percentile
        - Weekly breakout check: current weekly close above prior multi-week range or 55-day high
        """
        indicators = {}
        
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # 1. ATR% and percentile
            if len(df) >= 20:
                atr = EnhancedTechnicalAnalysis._calculate_atr(high, low, close, 20)
                atr_pct = (atr / close.iloc[-1]) * 100
                indicators['atr_pct'] = atr_pct
                
                # Calculate 1-year percentile
                if len(df) >= 252:
                    atr_series = pd.Series(index=df.index)
                    for i in range(19, len(df)):
                        atr_val = EnhancedTechnicalAnalysis._calculate_atr(
                            high.iloc[:i+1], low.iloc[:i+1], close.iloc[:i+1], 20
                        )
                        atr_series.iloc[i] = (atr_val / close.iloc[i]) * 100
                    
                    atr_1y = atr_series.iloc[-252:]
                    indicators['atr_pct_percentile'] = (atr_1y < atr_pct).sum() / len(atr_1y) * 100
            
            # 2. Bollinger Band width and percentile
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = EnhancedTechnicalAnalysis._calculate_bollinger_bands(close, 20, 2)
                bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
                indicators['bb_width'] = bb_width
                
                # Calculate 1-year percentile
                if len(close) >= 252:
                    bb_width_series = pd.Series(index=df.index)
                    for i in range(19, len(close)):
                        if i >= 19:
                            upper, middle, lower = EnhancedTechnicalAnalysis._calculate_bollinger_bands(
                                close.iloc[:i+1], 20, 2
                            )
                            bb_width_series.iloc[i] = ((upper - lower) / middle) * 100
                    
                    bb_width_1y = bb_width_series.iloc[-252:]
                    indicators['bb_width_percentile'] = (bb_width_1y < bb_width).sum() / len(bb_width_1y) * 100
            
            # 3. Keltner Channel width
            if len(df) >= 20:
                kc_upper, kc_middle, kc_lower = EnhancedTechnicalAnalysis._calculate_keltner_channels(
                    high, low, close, 20, 2
                )
                kc_width = ((kc_upper - kc_lower) / kc_middle) * 100
                indicators['keltner_width'] = kc_width
            
            # 4. Weekly breakout check
            if len(df) >= 55:
                # Convert to weekly data
                df_weekly = df.copy()
                df_weekly['week'] = pd.to_datetime(df_weekly.index).to_period('W')
                weekly_data = df_weekly.groupby('week').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                
                if len(weekly_data) >= 8:  # At least 8 weeks of data
                    current_weekly_close = weekly_data['close'].iloc[-1]
                    prior_4w_high = weekly_data['high'].iloc[-8:-1].max()  # Prior 7 weeks
                    indicators['weekly_breakout'] = current_weekly_close > prior_4w_high
                
                # Also check 55-day high breakout
                high_55d = high.iloc[-55:].max()
                indicators['above_55d_high'] = close.iloc[-1] > high_55d
            
            # 5. Squeeze detection (BB inside KC)
            if 'bb_width' in indicators and 'keltner_width' in indicators:
                indicators['in_squeeze'] = indicators['bb_width'] < indicators['keltner_width']
            
        except Exception as e:
            logger.error(f"Error calculating squeeze/breakout: {e}")
        
        return indicators
    
    @staticmethod
    def calculate_liquidity_risk(
        df: pd.DataFrame,
        shares_float: Optional[float] = None,
        shares_short: Optional[float] = None,
        news_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate Liquidity & Risk metrics.
        
        Required:
        - ADV in $ (Average Daily Volume in dollars)
        - Short % of float
        - News count last 10 days
        """
        indicators = {}
        
        try:
            close = df['close']
            volume = df['volume']
            
            # 1. Average Daily Volume in dollars (20-day)
            if len(df) >= 20:
                adv_shares = volume.iloc[-20:].mean()
                avg_price = close.iloc[-20:].mean()
                adv_dollars = adv_shares * avg_price
                indicators['adv_dollars'] = adv_dollars
                indicators['adv_shares'] = adv_shares
                
                # Liquidity tier
                if adv_dollars > 100_000_000:
                    indicators['liquidity_tier'] = 'very_high'
                elif adv_dollars > 50_000_000:
                    indicators['liquidity_tier'] = 'high'
                elif adv_dollars > 10_000_000:
                    indicators['liquidity_tier'] = 'medium'
                elif adv_dollars > 1_000_000:
                    indicators['liquidity_tier'] = 'low'
                else:
                    indicators['liquidity_tier'] = 'very_low'
            
            # 2. Short interest metrics
            if shares_float and shares_short:
                short_pct_float = (shares_short / shares_float) * 100
                indicators['short_pct_float'] = short_pct_float
                
                # Short squeeze potential
                if short_pct_float > 20:
                    indicators['short_squeeze_risk'] = 'very_high'
                elif short_pct_float > 15:
                    indicators['short_squeeze_risk'] = 'high'
                elif short_pct_float > 10:
                    indicators['short_squeeze_risk'] = 'medium'
                elif short_pct_float > 5:
                    indicators['short_squeeze_risk'] = 'low'
                else:
                    indicators['short_squeeze_risk'] = 'very_low'
            
            # 3. News activity
            if news_count is not None:
                indicators['news_count_10d'] = news_count
                
                # News activity level
                if news_count > 20:
                    indicators['news_activity'] = 'very_high'
                elif news_count > 10:
                    indicators['news_activity'] = 'high'
                elif news_count > 5:
                    indicators['news_activity'] = 'medium'
                elif news_count > 2:
                    indicators['news_activity'] = 'low'
                else:
                    indicators['news_activity'] = 'very_low'
            
        except Exception as e:
            logger.error(f"Error calculating liquidity/risk: {e}")
        
        return indicators
    
    @staticmethod
    def select_best_call(
        options_chain: List[Dict],
        current_price: float,
        target_dte_min: int = 45,
        target_dte_max: int = 75,
        target_delta_min: float = 0.55,
        target_delta_max: float = 0.70,
        min_open_interest: int = 50,  # Reduced from 250 for testing
        max_spread_pct: float = 5.0,  # Increased from 3.0 for more flexibility
        iv_percentile_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best call option based on scoring logic.
        
        Scoring:
        - Liquidity (40%): low spread%, higher OI/volume
        - IV Value (30%): lower IV percentile preferred
        - Fit (30%): delta nearest 0.60, earnings safely beyond 7-10 days
        """
        if not options_chain:
            logger.debug("No options chain provided for best call selection")
            return None
        
        logger.info(f"Selecting best call from {len(options_chain)} options with filters: "
                   f"DTE {target_dte_min}-{target_dte_max}, Delta {target_delta_min}-{target_delta_max}, "
                   f"Min OI {min_open_interest}")
        
        candidates = []
        calls_found = 0
        dte_filtered = 0
        delta_filtered = 0
        oi_filtered = 0
        
        for contract in options_chain:
            try:
                # Skip if not a call option
                if contract.get('option_type') != 'call':
                    continue
                calls_found += 1
                
                # Filter by DTE window
                dte = contract.get('days_to_expiration') or contract.get('dte', 0)
                if dte < target_dte_min or dte > target_dte_max:
                    dte_filtered += 1
                    continue
                
                # Filter by delta range
                delta = contract.get('delta', 0)
                if delta < target_delta_min or delta > target_delta_max:
                    delta_filtered += 1
                    continue
                
                # Filter by minimum open interest
                oi = contract.get('open_interest', 0)
                if oi < min_open_interest:
                    oi_filtered += 1
                    continue
                
                # Calculate spread %
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                    spread_pct = ((ask - bid) / mid_price) * 100
                    
                    # Filter by maximum spread
                    if spread_pct > max_spread_pct:
                        continue
                else:
                    continue  # Skip if no bid/ask
                
                # Score the candidate
                scores = {}
                
                # 1. Liquidity Score (40%)
                volume = contract.get('volume', 0)
                liquidity_score = 0
                
                # Spread component (0-50 points)
                if spread_pct <= 1:
                    liquidity_score += 50
                elif spread_pct <= 2:
                    liquidity_score += 40
                elif spread_pct <= 3:
                    liquidity_score += 30
                else:
                    liquidity_score += 20
                
                # OI component (0-30 points)
                if oi >= 1000:
                    liquidity_score += 30
                elif oi >= 500:
                    liquidity_score += 20
                elif oi >= 250:
                    liquidity_score += 10
                
                # Volume component (0-20 points)
                if volume >= 500:
                    liquidity_score += 20
                elif volume >= 100:
                    liquidity_score += 10
                elif volume >= 50:
                    liquidity_score += 5
                
                scores['liquidity'] = liquidity_score * 0.4
                
                # 2. IV Value Score (30%)
                iv = contract.get('implied_volatility', 0)
                iv_score = 0
                
                if iv_percentile_data and iv > 0:
                    # Use IV percentile if available
                    iv_percentile = iv_percentile_data.get('percentile', 50)
                    if iv_percentile <= 30:
                        iv_score = 100
                    elif iv_percentile <= 50:
                        iv_score = 80
                    elif iv_percentile <= 70:
                        iv_score = 60
                    elif iv_percentile <= 80:
                        iv_score = 40
                    else:
                        iv_score = 20  # High IV, less attractive
                else:
                    # Fallback to absolute IV levels
                    if iv <= 0.25:
                        iv_score = 100
                    elif iv <= 0.35:
                        iv_score = 80
                    elif iv <= 0.45:
                        iv_score = 60
                    elif iv <= 0.60:
                        iv_score = 40
                    else:
                        iv_score = 20
                
                scores['iv_value'] = iv_score * 0.3
                
                # 3. Fit Score (30%)
                fit_score = 0
                
                # Delta fit (0-60 points) - prefer 0.60
                delta_distance = abs(delta - 0.60)
                if delta_distance <= 0.02:
                    fit_score += 60
                elif delta_distance <= 0.05:
                    fit_score += 45
                elif delta_distance <= 0.10:
                    fit_score += 30
                else:
                    fit_score += 15
                
                # DTE fit (0-40 points) - prefer middle of range
                target_dte_mid = (target_dte_min + target_dte_max) / 2
                dte_distance = abs(dte - target_dte_mid)
                if dte_distance <= 5:
                    fit_score += 40
                elif dte_distance <= 10:
                    fit_score += 30
                elif dte_distance <= 15:
                    fit_score += 20
                else:
                    fit_score += 10
                
                scores['fit'] = fit_score * 0.3
                
                # Calculate total score
                total_score = scores['liquidity'] + scores['iv_value'] + scores['fit']
                
                # Add candidate
                candidates.append({
                    'contract': contract,
                    'scores': scores,
                    'total_score': total_score,
                    'spread_pct': spread_pct,
                    'metrics': {
                        'dte': dte,
                        'delta': delta,
                        'iv': iv,
                        'open_interest': oi,
                        'volume': volume,
                        'bid': bid,
                        'ask': ask,
                        'mid_price': mid_price,
                        'spread_pct': spread_pct
                    }
                })
                
            except Exception as e:
                logger.debug(f"Error scoring contract: {e}")
                continue
        
        # Select best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['total_score'])
            
            logger.info(f"Found {len(candidates)} candidates, selected: "
                       f"Strike {best['contract'].get('strike')}, "
                       f"Exp {best['contract'].get('expiration')}, "
                       f"DTE {best['metrics']['dte']}, "
                       f"Delta {best['metrics']['delta']:.2f}, "
                       f"Score {best['total_score']:.1f}")
            
            return {
                'strike': best['contract'].get('strike'),
                'expiration': best['contract'].get('expiration'),
                'dte': best['metrics']['dte'],
                'delta': best['metrics']['delta'],
                'bid': best['metrics']['bid'],
                'ask': best['metrics']['ask'],
                'mid_price': best['metrics']['mid_price'],
                'spread_percent': best['metrics']['spread_pct'],
                'open_interest': best['metrics']['open_interest'],
                'volume': best['metrics']['volume'],
                'iv': best['metrics']['iv'],
                'total_score': best['total_score'],
                'scoring': best['scores'],
                'selection_reason': f"Best call option (DTE: {best['metrics']['dte']}, Delta: {best['metrics']['delta']:.2f}, Score: {best['total_score']:.1f})",
                'candidates_evaluated': len(candidates)
            }
        
        logger.warning(f"No candidates found after filtering {len(options_chain)} options. "
                      f"Calls: {calls_found}, DTE filtered: {dte_filtered}, "
                      f"Delta filtered: {delta_filtered}, OI filtered: {oi_filtered}")
        return None
    
    @staticmethod
    def _get_selection_reason(best_candidate: Dict) -> str:
        """Generate selection reason based on scores."""
        scores = best_candidate['scores']
        metrics = best_candidate['metrics']
        
        reasons = []
        
        # Liquidity
        if scores['liquidity'] >= 32:  # 80% of max 40
            reasons.append(f"excellent liquidity (spread {metrics['spread_pct']:.1f}%)")
        elif scores['liquidity'] >= 24:  # 60% of max 40
            reasons.append(f"good liquidity (OI {metrics['open_interest']})")
        
        # IV Value
        if scores['iv_value'] >= 24:  # 80% of max 30
            reasons.append(f"attractive IV ({metrics['iv']:.1%})")
        
        # Fit
        if scores['fit'] >= 24:  # 80% of max 30
            reasons.append(f"optimal fit (delta {metrics['delta']:.2f})")
        
        if not reasons:
            reasons.append(f"balanced profile (score {best_candidate['total_score']:.1f})")
        
        return "Selected for " + ", ".join(reasons)
    
    # Helper methods
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate ADX."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not adx.empty else 0
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """Calculate ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1]
    
    @staticmethod
    def _calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        middle = close.rolling(window=period).mean().iloc[-1]
        std = close.rolling(window=period).std().iloc[-1]
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def _calculate_keltner_channels(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 20, 
        multiplier: float = 2
    ) -> Tuple[float, float, float]:
        """Calculate Keltner Channels."""
        typical_price = (high + low + close) / 3
        middle = typical_price.rolling(window=period).mean().iloc[-1]
        atr = EnhancedTechnicalAnalysis._calculate_atr(high, low, close, period)
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        return upper, middle, lower
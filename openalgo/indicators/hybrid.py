# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Hybrid and Advanced Indicators
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .utils import true_range, ema_wilder


class ADX(BaseIndicator):
    """
    Average Directional Index

    ADX measures the strength of a trend, regardless of direction.

    Components: +DI, -DI, ADX
    """

    def __init__(self):
        super().__init__("ADX")

    @staticmethod
    @jit(nopython=True)
    def _compute_dm(high: np.ndarray, low: np.ndarray, close: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute True Range and Directional Movement (jitted)."""
        n = len(high)
        tr = np.empty(n)
        dm_plus = np.empty(n)
        dm_minus = np.empty(n)

        tr[0] = high[0] - low[0]
        dm_plus[0] = 0.0
        dm_minus[0] = 0.0

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0.0

            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0.0

        return tr, dm_plus, dm_minus

    @staticmethod
    @jit(nopython=True)
    def _compute_di_dx(sm_atr: np.ndarray, sm_dm_plus: np.ndarray,
                       sm_dm_minus: np.ndarray, period: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute DI+, DI-, DX from smoothed arrays (jitted)."""
        n = len(sm_atr)
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)

        for i in range(period - 1, n):
            if not np.isnan(sm_atr[i]) and sm_atr[i] > 0:
                di_plus[i] = (sm_dm_plus[i] / sm_atr[i]) * 100
                di_minus[i] = (sm_dm_minus[i] / sm_atr[i]) * 100

                di_sum = di_plus[i] + di_minus[i]
                if di_sum > 0:
                    dx[i] = abs(di_plus[i] - di_minus[i]) / di_sum * 100

        return di_plus, di_minus, dx

    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Average Directional Index

        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Period for ADX calculation

        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (+DI, -DI, ADX) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)

        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))

        # Stage 1 (jitted): compute raw TR, DM+, DM-
        tr, dm_plus, dm_minus = self._compute_dm(high_data, low_data, close_data)

        # Stage 2: Wilder's smoothing (each call is jitted)
        sm_atr = ema_wilder(tr, period)
        sm_dm_plus = ema_wilder(dm_plus, period)
        sm_dm_minus = ema_wilder(dm_minus, period)

        # Stage 3 (jitted): compute DI+, DI-, DX
        di_plus, di_minus, dx = self._compute_di_dx(sm_atr, sm_dm_plus, sm_dm_minus, period)

        # Stage 4: ADX = Wilder's smoothing of DX
        adx = ema_wilder(dx, period)

        return self.format_multiple_outputs((di_plus, di_minus, adx), input_type, index)


class Aroon(BaseIndicator):
    """
    Aroon Indicator
    
    Aroon indicators measure the time since the highest high and lowest low.
    
    Formula: 
    Aroon Up = ((period - periods since highest high) / period) × 100
    Aroon Down = ((period - periods since lowest low) / period) × 100
    """
    
    def __init__(self):
        super().__init__("Aroon")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_aroon(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aroon calculation matching TradingView logic

        TradingView formula:
        upper = 100 * (ta.highestbars(high, length + 1) + length)/length
        lower = 100 * (ta.lowestbars(low, length + 1) + length)/length
        """
        n = len(high)
        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        lookback = period + 1

        for i in range(lookback - 1, n):
            window_start = i - lookback + 1

            highest_pos = 0
            lowest_pos = 0

            for j in range(lookback):
                idx = window_start + j
                if high[idx] > high[window_start + highest_pos]:
                    highest_pos = j
                if low[idx] < low[window_start + lowest_pos]:
                    lowest_pos = j

            bars_since_high = lookback - 1 - highest_pos
            bars_since_low = lookback - 1 - lowest_pos

            aroon_up[i] = 100.0 * (period - bars_since_high) / period
            aroon_down[i] = 100.0 * (period - bars_since_low) / period

        return aroon_up, aroon_down
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Aroon Indicator
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        period : int, default=14
            Period for Aroon calculation
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (aroon_up, aroon_down) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        self.validate_period(period, len(high_data))
        
        results = self._calculate_aroon(high_data, low_data, period)
        return self.format_multiple_outputs(results, input_type, index)


class PivotPoints(BaseIndicator):
    """
    Pivot Points
    
    Traditional pivot points used for support and resistance levels.
    
    Formula:
    Pivot = (High + Low + Close) / 3
    R1 = 2 * Pivot - Low
    S1 = 2 * Pivot - High
    R2 = Pivot + (High - Low)
    S2 = Pivot - (High - Low)
    R3 = High + 2 * (Pivot - Low)
    S3 = Low - 2 * (High - Pivot)
    """
    
    def __init__(self):
        super().__init__("Pivot Points")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_pivot_points(high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Numba optimized Pivot Points calculation"""
        n = len(high)
        
        pivot = np.empty(n)
        r1 = np.empty(n)
        s1 = np.empty(n)
        r2 = np.empty(n)
        s2 = np.empty(n)
        r3 = np.empty(n)
        s3 = np.empty(n)
        
        for i in range(n):
            # Calculate pivot point
            pivot[i] = (high[i] + low[i] + close[i]) / 3
            
            # Calculate resistance and support levels
            r1[i] = 2 * pivot[i] - low[i]
            s1[i] = 2 * pivot[i] - high[i]
            r2[i] = pivot[i] + (high[i] - low[i])
            s2[i] = pivot[i] - (high[i] - low[i])
            r3[i] = high[i] + 2 * (pivot[i] - low[i])
            s3[i] = low[i] - 2 * (high[i] - pivot[i])
        
        return pivot, r1, s1, r2, s2, r3, s3
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list]) -> Union[Tuple[np.ndarray, ...], Tuple[pd.Series, ...]]:
        """
        Calculate Pivot Points
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
            
        Returns:
        --------
        Union[Tuple[np.ndarray, ...], Tuple[pd.Series, ...]]
            (pivot, r1, s1, r2, s2, r3, s3) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        results = self._calculate_pivot_points(high_data, low_data, close_data)
        return self.format_multiple_outputs(results, input_type, index)


class SAR(BaseIndicator):
    """
    Parabolic SAR (Stop and Reverse)
    
    SAR is a trend-following indicator that provides potential reversal points.
    
    Formula: SAR = SAR[prev] + AF × (EP - SAR[prev])
    Where: AF = Acceleration Factor, EP = Extreme Point
    """
    
    def __init__(self):
        super().__init__("Parabolic SAR")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sar(high: np.ndarray, low: np.ndarray, 
                      acceleration: float, maximum: float) -> Tuple[np.ndarray, np.ndarray]:
        """Numba optimized SAR calculation"""
        n = len(high)
        sar = np.empty(n)
        trend = np.empty(n)  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        af = acceleration
        ep = high[0]  # Extreme point
        
        for i in range(1, n):
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            
            # Calculate new SAR
            sar[i] = prev_sar + af * (ep - prev_sar)
            
            # Determine trend
            if prev_trend == 1:  # Uptrend
                if low[i] <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = ep  # Set SAR to previous EP
                    ep = low[i]  # New EP for downtrend
                    af = acceleration  # Reset AF
                else:
                    # Continue uptrend
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
                    
                    # SAR should not exceed previous two lows
                    if i >= 2:
                        sar[i] = min(sar[i], low[i-1], low[i-2])
                    elif i >= 1:
                        sar[i] = min(sar[i], low[i-1])
            else:  # Downtrend
                if high[i] >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = ep  # Set SAR to previous EP
                    ep = high[i]  # New EP for uptrend
                    af = acceleration  # Reset AF
                else:
                    # Continue downtrend
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
                    
                    # SAR should not fall below previous two highs
                    if i >= 2:
                        sar[i] = max(sar[i], high[i-1], high[i-2])
                    elif i >= 1:
                        sar[i] = max(sar[i], high[i-1])
        
        return sar, trend
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 acceleration: float = 0.02, maximum: float = 0.2) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Parabolic SAR
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        acceleration : float, default=0.02
            Acceleration factor
        maximum : float, default=0.2
            Maximum acceleration factor
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (sar_values, trend_direction) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        
        if acceleration <= 0 or maximum <= 0:
            raise ValueError("Acceleration and maximum must be positive")
        if acceleration > maximum:
            raise ValueError("Acceleration cannot be greater than maximum")
        
        results = self._calculate_sar(high_data, low_data, acceleration, maximum)
        return self.format_multiple_outputs(results, input_type, index)


class DMI(BaseIndicator):
    """
    Directional Movement Index
    
    DMI is the same as ADX system but focuses on the directional indicators.
    """
    
    def __init__(self):
        super().__init__("DMI")
        self._adx = ADX()
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Directional Movement Index
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Period for DMI calculation
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (+DI, -DI) in the same format as input
        """
        results = self._adx.calculate(high, low, close, period)
        # Return only the first two components (+DI, -DI), excluding ADX
        if isinstance(results[0], np.ndarray):
            return results[0], results[1]
        else:
            return results[0], results[1]


class ZigZag(BaseIndicator):
    """
    Zig Zag
    
    Zig Zag connects significant price swings and filters out smaller price movements.
    
    Formula: Connect swing highs and lows that exceed the deviation threshold.
    """
    
    def __init__(self):
        super().__init__("ZigZag")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_zigzag(high: np.ndarray, low: np.ndarray, close: np.ndarray, deviation: float) -> np.ndarray:
        """Numba optimized Zig Zag calculation"""
        n = len(close)
        result = np.full(n, np.nan)
        
        if n < 3:
            return result
        
        # Find first significant point
        last_pivot_idx = 0
        last_pivot_price = close[0]
        last_pivot_type = 0  # 0 = unknown, 1 = high, -1 = low
        result[0] = close[0]
        
        current_high = high[0]
        current_low = low[0]
        current_high_idx = 0
        current_low_idx = 0
        
        for i in range(1, n):
            # Update current extremes
            if high[i] > current_high:
                current_high = high[i]
                current_high_idx = i
            if low[i] < current_low:
                current_low = low[i]
                current_low_idx = i
            
            # Check for reversal from high
            if last_pivot_type != -1:  # Not currently looking for high
                change_from_high = (current_high - low[i]) / current_high * 100
                if change_from_high >= deviation:
                    # Reversal from high - mark the high
                    result[current_high_idx] = current_high
                    last_pivot_idx = current_high_idx
                    last_pivot_price = current_high
                    last_pivot_type = 1
                    current_low = low[i]
                    current_low_idx = i
            
            # Check for reversal from low  
            if last_pivot_type != 1:  # Not currently looking for low
                if current_low > 0:
                    change_from_low = (high[i] - current_low) / current_low * 100
                    if change_from_low >= deviation:
                        # Reversal from low - mark the low
                        result[current_low_idx] = current_low
                        last_pivot_idx = current_low_idx
                        last_pivot_price = current_low
                        last_pivot_type = -1
                        current_high = high[i]
                        current_high_idx = i
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 deviation: float = 5.0) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Zig Zag
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        deviation : float, default=5.0
            Minimum percentage deviation for zig zag line
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Zig Zag values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        result = self._calculate_zigzag(high_data, low_data, close_data, deviation)
        return self.format_output(result, input_type, index)


class WilliamsFractals(BaseIndicator):
    """
    Williams Fractals - matches TradingView exactly
    
    Identifies turning points (fractals) in price action using the exact TradingView logic.
    The algorithm checks for peaks and troughs over a configurable number of periods (n).
    
    TradingView Logic:
    - Checks current position against n periods before and after
    - Handles edge cases with multiple frontier checks for future periods
    - Default period = 2 (minimum value)
    """
    
    def __init__(self):
        super().__init__("Williams Fractals")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_fractals_tv(high: np.ndarray, low: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Williams Fractals using exact TradingView logic
        
        This implements the complex TradingView algorithm that handles:
        1. Down frontier checks (past periods)
        2. Multiple up frontier checks (future periods with different patterns)
        """
        length = len(high)
        fractal_up = np.full(length, False)
        fractal_down = np.full(length, False)
        
        # Process each potential fractal position
        for center in range(n, length - n):
            # =================== UP FRACTAL LOGIC ===================
            # Down frontier check: all past periods must be lower
            upflag_down_frontier = True
            for i in range(1, n + 1):
                if high[center - i] >= high[center]:
                    upflag_down_frontier = False
                    break
            
            if upflag_down_frontier:
                # Up frontier checks: multiple patterns for future periods
                upflag_up_frontier0 = True  # All future periods strictly lower
                upflag_up_frontier1 = True  # Next period equal, rest strictly lower
                upflag_up_frontier2 = True  # Next 2 periods equal, rest strictly lower
                upflag_up_frontier3 = True  # Next 3 periods equal, rest strictly lower
                upflag_up_frontier4 = True  # Next 4 periods equal, rest strictly lower
                
                for i in range(1, n + 1):
                    # Pattern 0: all strictly lower
                    if center + i < length and high[center + i] >= high[center]:
                        upflag_up_frontier0 = False
                    
                    # Pattern 1: high[n+1] <= high[n] and rest strictly lower
                    if center + 1 < length and high[center + 1] > high[center]:
                        upflag_up_frontier1 = False
                    if center + i + 1 < length and high[center + i + 1] >= high[center]:
                        upflag_up_frontier1 = False
                    
                    # Pattern 2: high[n+1] <= high[n] and high[n+2] <= high[n] and rest strictly lower
                    if center + 1 < length and high[center + 1] > high[center]:
                        upflag_up_frontier2 = False
                    if center + 2 < length and high[center + 2] > high[center]:
                        upflag_up_frontier2 = False
                    if center + i + 2 < length and high[center + i + 2] >= high[center]:
                        upflag_up_frontier2 = False
                    
                    # Pattern 3: first 3 periods <= high[n] and rest strictly lower
                    if center + 1 < length and high[center + 1] > high[center]:
                        upflag_up_frontier3 = False
                    if center + 2 < length and high[center + 2] > high[center]:
                        upflag_up_frontier3 = False
                    if center + 3 < length and high[center + 3] > high[center]:
                        upflag_up_frontier3 = False
                    if center + i + 3 < length and high[center + i + 3] >= high[center]:
                        upflag_up_frontier3 = False
                    
                    # Pattern 4: first 4 periods <= high[n] and rest strictly lower
                    if center + 1 < length and high[center + 1] > high[center]:
                        upflag_up_frontier4 = False
                    if center + 2 < length and high[center + 2] > high[center]:
                        upflag_up_frontier4 = False
                    if center + 3 < length and high[center + 3] > high[center]:
                        upflag_up_frontier4 = False
                    if center + 4 < length and high[center + 4] > high[center]:
                        upflag_up_frontier4 = False
                    if center + i + 4 < length and high[center + i + 4] >= high[center]:
                        upflag_up_frontier4 = False
                
                # Combine all up frontier patterns
                flag_up_frontier = (upflag_up_frontier0 or upflag_up_frontier1 or 
                                  upflag_up_frontier2 or upflag_up_frontier3 or upflag_up_frontier4)
                
                fractal_up[center] = flag_up_frontier
            
            # =================== DOWN FRACTAL LOGIC ===================
            # Down frontier check: all past periods must be higher
            downflag_down_frontier = True
            for i in range(1, n + 1):
                if low[center - i] <= low[center]:
                    downflag_down_frontier = False
                    break
            
            if downflag_down_frontier:
                # Up frontier checks: multiple patterns for future periods
                downflag_up_frontier0 = True  # All future periods strictly higher
                downflag_up_frontier1 = True  # Next period equal, rest strictly higher
                downflag_up_frontier2 = True  # Next 2 periods equal, rest strictly higher
                downflag_up_frontier3 = True  # Next 3 periods equal, rest strictly higher
                downflag_up_frontier4 = True  # Next 4 periods equal, rest strictly higher
                
                for i in range(1, n + 1):
                    # Pattern 0: all strictly higher
                    if center + i < length and low[center + i] <= low[center]:
                        downflag_up_frontier0 = False
                    
                    # Pattern 1: low[n+1] >= low[n] and rest strictly higher
                    if center + 1 < length and low[center + 1] < low[center]:
                        downflag_up_frontier1 = False
                    if center + i + 1 < length and low[center + i + 1] <= low[center]:
                        downflag_up_frontier1 = False
                    
                    # Pattern 2: low[n+1] >= low[n] and low[n+2] >= low[n] and rest strictly higher
                    if center + 1 < length and low[center + 1] < low[center]:
                        downflag_up_frontier2 = False
                    if center + 2 < length and low[center + 2] < low[center]:
                        downflag_up_frontier2 = False
                    if center + i + 2 < length and low[center + i + 2] <= low[center]:
                        downflag_up_frontier2 = False
                    
                    # Pattern 3: first 3 periods >= low[n] and rest strictly higher
                    if center + 1 < length and low[center + 1] < low[center]:
                        downflag_up_frontier3 = False
                    if center + 2 < length and low[center + 2] < low[center]:
                        downflag_up_frontier3 = False
                    if center + 3 < length and low[center + 3] < low[center]:
                        downflag_up_frontier3 = False
                    if center + i + 3 < length and low[center + i + 3] <= low[center]:
                        downflag_up_frontier3 = False
                    
                    # Pattern 4: first 4 periods >= low[n] and rest strictly higher
                    if center + 1 < length and low[center + 1] < low[center]:
                        downflag_up_frontier4 = False
                    if center + 2 < length and low[center + 2] < low[center]:
                        downflag_up_frontier4 = False
                    if center + 3 < length and low[center + 3] < low[center]:
                        downflag_up_frontier4 = False
                    if center + 4 < length and low[center + 4] < low[center]:
                        downflag_up_frontier4 = False
                    if center + i + 4 < length and low[center + i + 4] <= low[center]:
                        downflag_up_frontier4 = False
                
                # Combine all down frontier patterns
                flag_down_frontier = (downflag_up_frontier0 or downflag_up_frontier1 or 
                                    downflag_up_frontier2 or downflag_up_frontier3 or downflag_up_frontier4)
                
                fractal_down[center] = flag_down_frontier
        
        return fractal_up, fractal_down
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 periods: int = 2) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Williams Fractals - matches TradingView exactly
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        periods : int, default=2
            Number of periods to check (minimum 2, matches TradingView default)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (fractal_up, fractal_down) boolean arrays in the same format as input
            True indicates a fractal point, False otherwise
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        
        if periods < 2:
            raise ValueError(f"Periods must be at least 2, got {periods}")
        
        fractal_up, fractal_down = self._calculate_fractals_tv(high_data, low_data, periods)
        
        results = (fractal_up, fractal_down)
        return self.format_multiple_outputs(results, input_type, index)


class RWI(BaseIndicator):
    """
    Random Walk Index (TradingView Pine Script Implementation)
    
    Measures how much a security's price movement differs from a random walk.
    
    Formula (TradingView Pine Script):
    rwiHigh = (high - nz(low[length])) / (atr(length) * sqrt(length))
    rwiLow = (nz(high[length]) - low) / (atr(length) * sqrt(length))
    """
    
    def __init__(self):
        super().__init__("RWI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR"""
        n = len(close)
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i - 1]), 
                       abs(low[i] - close[i - 1]))
        
        # Calculate ATR using simple moving average
        for i in range(period - 1, n):
            atr[i] = np.mean(tr[i - period + 1:i + 1])
        
        return atr
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr_for_rwi(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR for RWI"""
        n = len(close)
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i - 1]), 
                       abs(low[i] - close[i - 1]))
        
        # Calculate ATR using simple moving average
        for i in range(period - 1, n):
            atr[i] = np.mean(tr[i - period + 1:i + 1])
        
        return atr
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rwi(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba optimized RWI calculation"""
        n = len(close)
        rwi_high = np.full(n, np.nan)
        rwi_low = np.full(n, np.nan)
        
        # Calculate ATR inline
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i - 1]), 
                       abs(low[i] - close[i - 1]))
        
        # Calculate ATR using simple moving average
        for i in range(period - 1, n):
            atr[i] = np.mean(tr[i - period + 1:i + 1])
        
        # Calculate RWI using TradingView Pine Script formula
        for i in range(period, n):  # Start from period to have valid lookback
            if atr[i] > 0:
                # TradingView formula: rwiHigh = (high - nz(low[length])) / (atr(length) * sqrt(length))
                # nz(low[length]) means low[i-length] with zero fill for missing values
                low_lookback = low[i - period] if i >= period else 0.0
                rwi_high[i] = (high[i] - low_lookback) / (atr[i] * np.sqrt(period))
                
                # TradingView formula: rwiLow = (nz(high[length]) - low) / (atr(length) * sqrt(length))
                # nz(high[length]) means high[i-length] with zero fill for missing values
                high_lookback = high[i - period] if i >= period else 0.0
                rwi_low[i] = (high_lookback - low[i]) / (atr[i] * np.sqrt(period))
            else:
                rwi_high[i] = 0.0
                rwi_low[i] = 0.0
        
        return rwi_high, rwi_low
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Random Walk Index (TradingView Pine Script Implementation)
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Period for RWI calculation
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (rwi_high, rwi_low) in the same format as input
            Formula: rwiHigh = (high - nz(low[length])) / (atr(length) * sqrt(length))
                    rwiLow = (nz(high[length]) - low) / (atr(length) * sqrt(length))
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        rwi_high, rwi_low = self._calculate_rwi(high_data, low_data, close_data, period)
        
        results = (rwi_high, rwi_low)
        return self.format_multiple_outputs(results, input_type, index)
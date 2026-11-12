# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Trend Indicators
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .utils import sma, ema, highest, lowest, vwma_optimized, kama_optimized, atr_wilder
from . import _backend


class SMA(BaseIndicator):
    """
    Simple Moving Average
    
    The SMA is calculated by adding the closing prices of a security for a period 
    and then dividing this total by the number of time periods.
    
    Formula: SMA = (P1 + P2 + ... + Pn) / n
    """
    
    def __init__(self):
        super().__init__("SMA")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized SMA calculation"""
        result = np.empty_like(data)
        result[:period-1] = np.nan
        
        # Calculate first SMA value
        sum_val = 0.0
        for i in range(period):
            sum_val += data[i]
        result[period-1] = sum_val / period
        
        # Calculate remaining values using rolling window
        for i in range(period, len(data)):
            sum_val = sum_val - data[i-period] + data[i]
            result[i] = sum_val / period
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Simple Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            SMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.sma(validated_data, period)
        return self.format_output(result, input_type, index)


class EMA(BaseIndicator):
    """
    Exponential Moving Average
    
    The EMA gives more weight to recent prices, making it more responsive to new information.
    
    Formula: EMA = (Close - Previous EMA) × Multiplier + Previous EMA
    Where: Multiplier = 2 / (Period + 1)
    """
    
    def __init__(self):
        super().__init__("EMA")
    
    # Removed redundant EMA calculation - using consolidated utility
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Exponential Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            EMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.ema(validated_data, period)
        return self.format_output(result, input_type, index)


class WMA(BaseIndicator):
    """
    Weighted Moving Average
    
    The WMA assigns greater weight to recent data points.
    
    Formula: WMA = (P1×1 + P2×2 + ... + Pn×n) / (1 + 2 + ... + n)
    """
    
    def __init__(self):
        super().__init__("WMA")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_wma(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized WMA calculation"""
        result = np.empty_like(data)
        result[:period-1] = np.nan
        
        # Calculate weight sum: 1 + 2 + ... + period
        weight_sum = period * (period + 1) // 2
        
        for i in range(period-1, len(data)):
            weighted_sum = 0.0
            for j in range(period):
                weight = j + 1
                weighted_sum += data[i - period + 1 + j] * weight
            result[i] = weighted_sum / weight_sum
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Weighted Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            WMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.wma(validated_data, period)
        return self.format_output(result, input_type, index)


class DEMA(BaseIndicator):
    """
    Double Exponential Moving Average
    
    DEMA attempts to reduce the lag associated with traditional moving averages.
    
    Formula: DEMA = 2 × EMA(n) - EMA(EMA(n))
    """
    
    def __init__(self):
        super().__init__("DEMA")
        self._ema = EMA()
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Double Exponential Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            DEMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.dema(validated_data, period)
        return self.format_output(result, input_type, index)


class TEMA(BaseIndicator):
    """
    Triple Exponential Moving Average
    
    TEMA further reduces lag compared to DEMA.
    
    Formula: TEMA = 3×EMA(n) - 3×EMA(EMA(n)) + EMA(EMA(EMA(n)))
    """
    
    def __init__(self):
        super().__init__("TEMA")
        self._ema = EMA()
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Triple Exponential Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            TEMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.tema(validated_data, period)
        return self.format_output(result, input_type, index)


@jit(nopython=True)
def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Average True Range"""
    n = len(high)
    tr = np.empty(n)
    atr = np.empty(n)
    
    # First TR value
    tr[0] = high[0] - low[0]
    
    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate ATR
    atr[:period-1] = np.nan
    
    # Initial ATR is simple average
    sum_tr = 0.0
    for i in range(period):
        sum_tr += tr[i]
    atr[period-1] = sum_tr / period
    
    # Subsequent ATR values use smoothed average
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


class Supertrend(BaseIndicator):
    """
    Supertrend Indicator
    
    The Supertrend indicator is a trend-following indicator that uses ATR (Average True Range) 
    to calculate dynamic support and resistance levels.
    
    Formula:
    Basic Upper Band = (HIGH + LOW) / 2 + Multiplier × ATR
    Basic Lower Band = (HIGH + LOW) / 2 - Multiplier × ATR
    """
    
    def __init__(self):
        super().__init__("Supertrend")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                             period: int, multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
        """Supertrend: hl2 +- multiplier * ATR bands that flip on a close through the active band"""
        n = len(close)
        
        # Calculate ATR
        atr = _calculate_atr(high, low, close, period)
        
        # Calculate basic bands around hl2
        hl_avg = (high + low) / 2.0
        upper_band = hl_avg + multiplier * atr
        lower_band = hl_avg - multiplier * atr
        
        # Initialize arrays
        final_upper = np.full(n, np.nan)
        final_lower = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        direction = np.full(n, np.nan)
        
        first_valid = period - 1
        if first_valid < 0 or first_valid >= n:
            return supertrend, direction
        
        # Initialize first valid values
        final_upper[first_valid] = upper_band[first_valid]
        final_lower[first_valid] = lower_band[first_valid]
        # The first bar with an ATR has no previous ATR: it starts as a downtrend
        direction[first_valid] = 1.0  # downtrend
        supertrend[first_valid] = final_upper[first_valid]
        
        # Subsequent bars
        for i in range(first_valid + 1, n):
            # Final lower band: lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # Final upper band: upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand  
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Direction logic
            # if the previous supertrend was the upper band:
            #     direction = -1 if close > upper band else 1
            # else:
            #     direction = 1 if close < lower band else -1
            if supertrend[i-1] == final_upper[i-1]:
                # Previous was upper band (downtrend)
                if close[i] > final_upper[i]:
                    direction[i] = -1.0  # Change to uptrend (green)
                else:
                    direction[i] = 1.0   # Continue downtrend (red)
            else:
                # Previous was lower band (uptrend)
                if close[i] < final_lower[i]:
                    direction[i] = 1.0   # Change to downtrend (red)
                else:
                    direction[i] = -1.0  # Continue uptrend (green)
            
            # Supertrend assignment: _direction == -1 ? lowerBand : upperBand
            if direction[i] == -1.0:  # uptrend (green)
                supertrend[i] = final_lower[i]
            else:  # downtrend (red)
                supertrend[i] = final_upper[i]
                
        return supertrend, direction
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 10, multiplier: float = 3.0) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Supertrend Indicator
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=10
            ATR period
        multiplier : float, default=3.0
            ATR multiplier
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (supertrend values, direction values) in the same format as input
            Direction: -1 for uptrend, 1 for downtrend
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        if multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {multiplier}")
        
        supertrend_result, direction_result = _backend.supertrend(high_data, low_data, close_data, period, multiplier)
        return self.format_multiple_outputs((supertrend_result, direction_result), input_type, index)


class Ichimoku(BaseIndicator):
    """
    Ichimoku Cloud
    
    The Ichimoku Cloud is a comprehensive indicator that defines support and resistance, 
    identifies trend direction, gauges momentum, and provides trading signals.
    
    Components:
    - Conversion Line: donchian(conversionPeriods) = avg(highest(9), lowest(9))
    - Base Line: donchian(basePeriods) = avg(highest(26), lowest(26))  
    - Leading Span A: avg(conversionLine, baseLine), offset +displacement-1
    - Leading Span B: donchian(laggingSpan2Periods) = avg(highest(52), lowest(52)), offset +displacement-1
    - Lagging Span: close, offset -displacement+1
    """
    
    def __init__(self):
        super().__init__("Ichimoku")
    
    @staticmethod
    def _calculate_ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                              conversion_periods: int = 9, base_periods: int = 26,
                              lagging_span2_periods: int = 52, displacement: int = 26) -> Tuple[np.ndarray, ...]:
        """
        Calculate Ichimoku
        Formula:
        donchian(len) = (lowest(low, len) + highest(high, len)) / 2
        conversionLine = donchian(conversionPeriods)
        baseLine = donchian(basePeriods)
        leadLine1 = (conversionLine + baseLine) / 2
        leadLine2 = donchian(laggingSpan2Periods)
        """
        n = len(close)
        
        # Donchian function: avg(highest, lowest)
        def donchian(data_high, data_low, period):
            high_values = highest(data_high, period)
            low_values = lowest(data_low, period)
            return (high_values + low_values) / 2.0
        
        # Calculate components
        conversion_line = donchian(high, low, conversion_periods)  # Tenkan-sen
        base_line = donchian(high, low, base_periods)             # Kijun-sen
        lead_line1 = (conversion_line + base_line) / 2.0          # Senkou Span A
        lead_line2 = donchian(high, low, lagging_span2_periods)   # Senkou Span B
        
        # Offsets: leading spans move forward by displacement - 1 bars,
        # the lagging span moves back by displacement - 1 bars
        
        # Leading Span A: offset = displacement - 1 (shift forward)
        leading_span_a = np.full(n, np.nan)
        offset_a = displacement - 1
        if offset_a > 0 and offset_a < n:
            leading_span_a[offset_a:] = lead_line1[:-offset_a]
        elif offset_a == 0:
            leading_span_a = lead_line1.copy()
        
        # Leading Span B: offset = displacement - 1 (shift forward)  
        leading_span_b = np.full(n, np.nan)
        offset_b = displacement - 1
        if offset_b > 0 and offset_b < n:
            leading_span_b[offset_b:] = lead_line2[:-offset_b]
        elif offset_b == 0:
            leading_span_b = lead_line2.copy()
        
        # Lagging Span: offset = -displacement + 1 (shift backward)
        lagging_span = np.full(n, np.nan)
        offset_lag = -displacement + 1
        if offset_lag < 0:
            lag_shift = abs(offset_lag)
            if lag_shift < n:
                lagging_span[:-lag_shift] = close[lag_shift:]
        elif offset_lag > 0:
            if offset_lag < n:
                lagging_span[offset_lag:] = close[:-offset_lag]
        else:  # offset_lag == 0
            lagging_span = close.copy()
        
        return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 conversion_periods: int = 9, base_periods: int = 26,
                 lagging_span2_periods: int = 52, displacement: int = 26) -> Union[Tuple[np.ndarray, ...], Tuple[pd.Series, ...]]:
        """
        Calculate Ichimoku Cloud
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        conversion_periods : int, default=9
            Conversion Line Length
        base_periods : int, default=26
            Base Line Length
        lagging_span2_periods : int, default=52
            Leading Span B Length
        displacement : int, default=26
            Lagging Span displacement
            
        Returns:
        --------
        Union[Tuple[np.ndarray, ...], Tuple[pd.Series, ...]]
            (conversion_line, base_line, leading_span_a, leading_span_b, lagging_span) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        # Validate periods
        for period, name in [(conversion_periods, "conversion_periods"), 
                            (base_periods, "base_periods"), 
                            (lagging_span2_periods, "lagging_span2_periods")]:
            if period <= 0:
                raise ValueError(f"{name} must be positive, got {period}")
        
        results = _backend.ichimoku(high_data, low_data, close_data, conversion_periods,
                                    base_periods, lagging_span2_periods, displacement)
        return self.format_multiple_outputs(results, input_type, index)


class HMA(BaseIndicator):
    """
    Hull Moving Average
    
    The Hull Moving Average (HMA) attempts to minimize lag and improve smoothing.
    
    Formula: HMA = WMA(2 × WMA(n/2) - WMA(n), sqrt(n))
    """
    
    def __init__(self):
        super().__init__("HMA")

    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Hull Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            HMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))

        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n)). Computed in the Rust core, which
        # runs the final WMA over the finite region only so the (period-1) leading
        # NaNs from WMA(n) never poison the rolling accumulators (which would blank
        # the whole output).
        result = _backend.hma(validated_data, period)

        return self.format_output(result, input_type, index)


class VWMA(BaseIndicator):
    """
    Volume Weighted Moving Average
    
    VWMA gives more weight to periods with higher volume.
    
    Formula: VWMA = Σ(Price × Volume) / Σ(Volume)
    """
    
    def __init__(self):
        super().__init__("VWMA")
    
    @staticmethod
    def _calculate_vwma(data: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """O(n) optimized VWMA calculation using the Rust backend"""
        return _backend.vwma(data, volume, period)
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Volume Weighted Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            VWMA values in the same format as input
        """
        data_validated, input_type, index = self.validate_input(data)
        volume_validated, _, _ = self.validate_input(volume)
        data_validated, volume_validated = self.align_arrays(data_validated, volume_validated)
        self.validate_period(period, len(data_validated))
        
        result = self._calculate_vwma(data_validated, volume_validated, period)
        return self.format_output(result, input_type, index)


class ALMA(BaseIndicator):
    """
    Arnaud Legoux Moving Average
    
    ALMA is a technical analysis indicator that combines the features of SMA and EMA.
    
    Formula: ALMA = Σ(w[i] × data[i]) / Σ(w[i])
    Where: w[i] = exp(-((i - m)^2) / (2 × s^2))
    """
    
    def __init__(self):
        super().__init__("ALMA")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_alma(data: np.ndarray, period: int, offset: float, sigma: float) -> np.ndarray:
        """Numba optimized ALMA calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        # Calculate weights
        m = offset * (period - 1)
        s = period / sigma
        
        weights = np.empty(period)
        for i in range(period):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
        
        # Normalize weights
        weight_sum = np.sum(weights)
        weights = weights / weight_sum
        
        # Calculate ALMA
        for i in range(period - 1, n):
            alma_sum = 0.0
            for j in range(period):
                alma_sum += weights[j] * data[i - period + 1 + j]
            result[i] = alma_sum
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 period: int = 21, offset: float = 0.85, sigma: float = 6.0) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Arnaud Legoux Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=21
            Number of periods for the moving average
        offset : float, default=0.85
            Phase offset (0 to 1)
        sigma : float, default=6.0
            Smoothing factor
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            ALMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        if not 0 <= offset <= 1:
            raise ValueError(f"Offset must be between 0 and 1, got {offset}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        
        result = _backend.alma(validated_data, period, offset, sigma)
        return self.format_output(result, input_type, index)


class KAMA(BaseIndicator):
    """
    Kaufman's Adaptive Moving Average
    
    KAMA is a moving average designed to account for market noise or volatility.
    
    Formula:
    mom = abs(src - src[length])
    volatility = sum(abs(src - src[1]), length)  
    er = volatility != 0 ? mom / volatility : 0
    fastAlpha = 2 / (fastLength + 1)
    slowAlpha = 2 / (slowLength + 1)
    alpha = pow(er * (fastAlpha - slowAlpha) + slowAlpha, 2)
    kama = alpha * src + (1 - alpha) * kama[1], with kama[1] = src when undefined
    """
    
    def __init__(self):
        super().__init__("KAMA")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_kama_full_window(data: np.ndarray, length: int, fast_length: int, slow_length: int) -> np.ndarray:
        """
        Calculate KAMA, summing the volatility over the full window at every bar
        """
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < length + 1:
            return result
        
        # EMA alphas of the fast and slow lengths
        fast_alpha = 2.0 / (fast_length + 1)
        slow_alpha = 2.0 / (slow_length + 1)
        
        # Calculate for each period
        for i in range(length, n):
            # mom = abs(data[i] - data[i-length])
            mom = abs(data[i] - data[i - length])
            
            # volatility = sum of absolute one-bar changes over length
            volatility = 0.0
            for j in range(i - length + 1, i + 1):
                if j > 0:
                    volatility += abs(data[j] - data[j - 1])
            
            # Efficiency Ratio: er = volatility != 0 ? mom / volatility : 0
            if volatility != 0:
                er = mom / volatility
            else:
                er = 0.0
            
            # alpha = pow(er * (fastAlpha - slowAlpha) + slowAlpha, 2)
            alpha = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            
            # kama = alpha * src + (1 - alpha) * kama[1]
            # For the first calculation, use the current price as the previous KAMA
            if i == length or np.isnan(result[i - 1]):
                prev_kama = data[i]
            else:
                prev_kama = result[i - 1]
            
            result[i] = alpha * data[i] + (1 - alpha) * prev_kama
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 length: int = 14, fast_length: int = 2, slow_length: int = 30) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Kaufman's Adaptive Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Source data (typically closing prices)
        length : int, default=14
            Length for efficiency ratio calculation
        fast_length : int, default=2
            Fast EMA Length
        slow_length : int, default=30
            Slow EMA Length
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            KAMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(length + 1, len(validated_data))
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        if fast_length <= 0:
            raise ValueError(f"Fast length must be positive, got {fast_length}")
        if slow_length <= 0:
            raise ValueError(f"Slow length must be positive, got {slow_length}")
        if fast_length >= slow_length:
            raise ValueError("Fast length must be less than slow length")
        
        result = _backend.kama_full_window(validated_data, length, fast_length, slow_length)
        return self.format_output(result, input_type, index)


class ZLEMA(BaseIndicator):
    """
    Zero Lag Exponential Moving Average
    
    ZLEMA is an EMA that attempts to minimize lag by using price momentum.
    
    Formula: ZLEMA = EMA(2 × Price - Price[lag])
    Where: lag = (period - 1) / 2
    """
    
    def __init__(self):
        super().__init__("ZLEMA")
        self._ema = EMA()
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Zero Lag Exponential Moving Average using O(n) algorithm
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            ZLEMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = _backend.zlema(validated_data, period)
        return self.format_output(result, input_type, index)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_zlema_optimized(data: np.ndarray, period: int) -> np.ndarray:
        """O(n) optimized ZLEMA calculation using consolidated EMA utility"""
        n = len(data)
        lag = (period - 1) // 2
        
        # Create adjusted data using vectorized operations where possible
        adjusted_data = np.empty(n)
        adjusted_data[:lag] = data[:lag]
        
        # Vectorized calculation for the main portion
        for i in range(lag, n):
            adjusted_data[i] = 2 * data[i] - data[i - lag]
        
        # Use the optimized EMA utility
        return ema(adjusted_data, period)


class T3(BaseIndicator):
    """
    T3 Moving Average
    
    T3 is a type of moving average which is the result of applying EMA three times.
    
    Formula: T3 = GD(GD(GD(data)))
    Where: GD = Generalized DEMA
    """
    
    def __init__(self):
        super().__init__("T3")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_gd(data: np.ndarray, period: int, v_factor: float) -> np.ndarray:
        """Calculate Generalized DEMA"""
        alpha = 2.0 / (period + 1)
        
        # First EMA
        ema1 = np.empty_like(data)
        ema1[0] = data[0]
        for i in range(1, len(data)):
            ema1[i] = alpha * data[i] + (1 - alpha) * ema1[i - 1]
        
        # Second EMA
        ema2 = np.empty_like(data)
        ema2[0] = ema1[0]
        for i in range(1, len(data)):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
        
        # Generalized DEMA
        gd = (1 + v_factor) * ema1 - v_factor * ema2
        return gd
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 period: int = 21, v_factor: float = 0.7) -> Union[np.ndarray, pd.Series]:
        """
        Calculate T3 Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=21
            Number of periods for the moving average
        v_factor : float, default=0.7
            Volume factor for T3 calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            T3 values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = _backend.t3(validated_data, period, v_factor)
        return self.format_output(result, input_type, index)


class FRAMA(BaseIndicator):
    """
    Fractal Adaptive Moving Average
    
    FRAMA adjusts its smoothing based on the fractal dimension of the price series.
    
    Formula: FRAMA = (2 - D) × Price + (D - 1) × FRAMA[prev]
    Where: D = fractal dimension
    """
    
    def __init__(self):
        super().__init__("FRAMA")
    
    @staticmethod
    def _calculate_frama(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """FRAMA values: the shared kernel, `_backend.frama`"""
        return _backend.frama(high, low, period)
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 period: int = 26) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Fractal Adaptive Moving Average
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices  
        period : int, default=26
            Number of periods for FRAMA calculation
            Must be even number, minimum 2
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            FRAMA values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        self.validate_period(period, len(high_data))
        
        if period < 2:
            raise ValueError("Period must be at least 2 for FRAMA calculation")
        
        if period % 2 != 0:
            raise ValueError("Period must be even for FRAMA calculation")
        
        result = _backend.frama(high_data, low_data, period)
        return self.format_output(result, input_type, index)


class ChandeKrollStop(BaseIndicator):
    """
    Chande Kroll Stop
    
    The Chande Kroll Stop is a trailing stop-loss technique that combines 
    ATR-based calculations with highest highs and lowest lows.
    
    Formula:
    Long Stop = Highest(High, period) - ATR_multiplier * ATR(period)
    Short Stop = Lowest(Low, period) + ATR_multiplier * ATR(period)
    """
    
    def __init__(self):
        super().__init__("Chande Kroll Stop")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR"""
        n = len(close)
        tr = np.empty(n)
        
        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                       abs(high[i] - close[i-1]),
                       abs(low[i] - close[i-1]))
        
        # Calculate ATR using exponential moving average
        atr = np.empty(n)
        atr[0] = tr[0]
        alpha = 1.0 / period
        
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        return atr
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_chande_kroll_stop(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                   p: int, x: float, q: int) -> Tuple[np.ndarray, np.ndarray]:
        """Chande Kroll Stop: highest/lowest of the ATR-offset extremes over q bars"""
        n = len(close)
        long_stop = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)
        
        # Calculate ATR using Wilder's smoothing
        atr = atr_wilder(high, low, close, p)
        
        # Step 1: Calculate first_high_stop and first_low_stop
        first_high_stop = np.full(n, np.nan)
        first_low_stop = np.full(n, np.nan)
        
        for i in range(p - 1, n):
            # Calculate highest high and lowest low over period p
            highest_high = np.max(high[i - p + 1:i + 1])
            lowest_low = np.min(low[i - p + 1:i + 1])
            
            # first_high_stop = highest(high, p) - x * atr(p)
            first_high_stop[i] = highest_high - x * atr[i]
            # first_low_stop = lowest(low, p) + x * atr(p)
            first_low_stop[i] = lowest_low + x * atr[i]
        
        # Step 2: Calculate final stops using highest/lowest of first stops over period q
        # stop_short = highest(first_high_stop, q)
        # stop_long = lowest(first_low_stop, q)
        start_idx = p + q - 2  # Need both p and q periods to start
        
        for i in range(start_idx, n):
            # Calculate highest of first_high_stop over period q
            if i >= q - 1:
                q_start = max(0, i - q + 1)
                # Find highest non-NaN value in the q-period window
                window_high = first_high_stop[q_start:i + 1]
                valid_high = window_high[~np.isnan(window_high)]
                if len(valid_high) > 0:
                    short_stop[i] = np.max(valid_high)
                
                # Find lowest non-NaN value in the q-period window  
                window_low = first_low_stop[q_start:i + 1]
                valid_low = window_low[~np.isnan(window_low)]
                if len(valid_low) > 0:
                    long_stop[i] = np.min(valid_low)
        
        return long_stop, short_stop
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 p: int = 10, x: float = 1.0, q: int = 9) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Chande Kroll Stop
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        p : int, default=10
            ATR Length (period for ATR and highest/lowest calculation)
        x : float, default=1.0
            ATR Coefficient (multiplier for ATR)
        q : int, default=9
            Stop Length (period for smoothing the stops)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (stop_long, stop_short) in the same format as input
            - stop_long: lowest of first_low_stop over q periods
            - stop_short: highest of first_high_stop over q periods
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(p, len(close_data))
        self.validate_period(q, len(close_data))
        
        if x <= 0:
            raise ValueError(f"ATR coefficient (x) must be positive, got {x}")
        
        long_stop, short_stop = _backend.chande_kroll_stop(high_data, low_data, close_data, p, x, q)

        results = (long_stop, short_stop)
        return self.format_multiple_outputs(results, input_type, index)


class TRIMA(BaseIndicator):
    """
    Triangular Moving Average
    
    TRIMA is a double-smoothed moving average that applies SMA twice.
    
    Formula: TRIMA = SMA(SMA(Close, n), n)
    Where n = (period + 1) / 2
    """
    
    def __init__(self):
        super().__init__("TRIMA")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_trima(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized TRIMA calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        # First, calculate n for SMA
        n1 = (period + 1) // 2
        n2 = period - n1 + 1
        
        # First SMA
        first_sma = np.full(n, np.nan)
        for i in range(n1 - 1, n):
            first_sma[i] = np.mean(data[i - n1 + 1:i + 1])
        
        # Second SMA on first SMA
        for i in range(n1 + n2 - 2, n):
            if not np.isnan(first_sma[i]):
                window_start = max(0, i - n2 + 1)
                window = first_sma[window_start:i + 1]
                valid_values = window[~np.isnan(window)]
                if len(valid_values) >= n2:
                    result[i] = np.mean(valid_values[-n2:])
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Triangular Moving Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the moving average
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            TRIMA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.trima(validated_data, period)
        return self.format_output(result, input_type, index)


class McGinley(BaseIndicator):
    """
    McGinley Dynamic
    
    A moving average that adjusts automatically for market speed changes.
    
    Formula: MD[i] = MD[i-1] + (Close[i] - MD[i-1]) / (N * (Close[i]/MD[i-1])^4)
    """
    
    def __init__(self):
        super().__init__("McGinley Dynamic")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_mcginley(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized McGinley Dynamic calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < period:
            return result
        
        # Initialize with SMA
        result[period - 1] = np.mean(data[:period])
        
        for i in range(period, n):
            if result[i - 1] != 0:
                ratio = data[i] / result[i - 1]
                factor = period * (ratio ** 4)
                result[i] = result[i - 1] + (data[i] - result[i - 1]) / factor
            else:
                result[i] = data[i]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int) -> Union[np.ndarray, pd.Series]:
        """
        Calculate McGinley Dynamic
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int
            Number of periods for the calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            McGinley Dynamic values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.mcginley(validated_data, period)
        return self.format_output(result, input_type, index)


class VIDYA(BaseIndicator):
    """
    Variable Index Dynamic Average (VIDYA)
    
    VIDYA uses the Chande Momentum Oscillator (CMO) to adjust the smoothing constant of an EMA.
    
    Formula: VIDYA[i] = VIDYA[i-1] + alpha * |CMO[i]| / 100 * (Close[i] - VIDYA[i-1])
    """
    
    def __init__(self):
        super().__init__("VIDYA")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_cmo(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Chande Momentum Oscillator"""
        n = len(data)
        result = np.full(n, np.nan)
        
        # Calculate price changes
        changes = np.diff(data)
        
        for i in range(period, n):
            sum_up = 0.0
            sum_down = 0.0
            
            for j in range(period):
                change = changes[i - period + j]
                if change > 0:
                    sum_up += change
                elif change < 0:
                    sum_down += abs(change)
            
            total_movement = sum_up + sum_down
            if total_movement > 0:
                result[i] = 100 * (sum_up - sum_down) / total_movement
            else:
                result[i] = 0.0
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_vidya(data: np.ndarray, period: int, alpha: float) -> np.ndarray:
        """Numba optimized VIDYA calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        # Calculate CMO inline
        cmo = np.full(n, np.nan)
        
        for i in range(period, n):
            gains = 0.0
            losses = 0.0
            
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    diff = data[j] - data[j - 1]
                    if diff > 0:
                        gains += diff
                    elif diff < 0:
                        losses += abs(diff)
            
            if gains + losses != 0:
                cmo[i] = 100 * (gains - losses) / (gains + losses)
            else:
                cmo[i] = 0.0
        
        # Initialize with first valid data point
        result[period] = data[period]
        
        for i in range(period + 1, n):
            if not np.isnan(cmo[i]):
                smoothing_constant = alpha * abs(cmo[i]) / 100
                result[i] = result[i - 1] + smoothing_constant * (data[i] - result[i - 1])
            else:
                result[i] = result[i - 1]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 period: int = 14, alpha: float = 0.2) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Variable Index Dynamic Average
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=14
            Period for CMO calculation
        alpha : float, default=0.2
            Alpha factor for smoothing
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            VIDYA values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period + 1, len(validated_data))
        result = _backend.vidya(validated_data, period, alpha)
        return self.format_output(result, input_type, index)


class Alligator(BaseIndicator):
    """
    Alligator (Bill Williams)
    
    The Alligator consists of three smoothed moving averages:
    - Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    - Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    - Lips (green line): 5-period SMMA, shifted 3 bars forward
    """
    
    def __init__(self):
        super().__init__("Alligator")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_smma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Smoothed Moving Average (SMMA)"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < period:
            return result
        
        # Initialize with SMA
        result[period - 1] = np.mean(data[:period])
        
        # Calculate SMMA
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _shift_series(data: np.ndarray, shift: int) -> np.ndarray:
        """Shift series forward by specified periods"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(shift, n):
            result[i] = data[i - shift]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 jaw_period: int = 13, jaw_shift: int = 8,
                 teeth_period: int = 8, teeth_shift: int = 5,
                 lips_period: int = 5, lips_shift: int = 3) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Alligator
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically (high + low) / 2)
        jaw_period : int, default=13
            Period for Jaw line
        jaw_shift : int, default=8
            Forward shift for Jaw line
        teeth_period : int, default=8
            Period for Teeth line
        teeth_shift : int, default=5
            Forward shift for Teeth line
        lips_period : int, default=5
            Period for Lips line
        lips_shift : int, default=3
            Forward shift for Lips line
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (jaw, teeth, lips) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        results = _backend.alligator(validated_data, jaw_period, jaw_shift,
                                     teeth_period, teeth_shift, lips_period, lips_shift)
        return self.format_multiple_outputs(results, input_type, index)


class MovingAverageEnvelopes(BaseIndicator):
    """
    Moving Average Envelopes
    
    Upper and lower envelopes around a moving average, calculated as percentage bands.
    
    Formula: 
    Upper = MA * (1 + percentage/100)
    Lower = MA * (1 - percentage/100)
    """
    
    def __init__(self):
        super().__init__("MA Envelopes")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            result[i] = np.mean(data[i - period + 1:i + 1])
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        n = len(data)
        result = np.empty(n)
        alpha = 2.0 / (period + 1)
        
        result[0] = data[0]
        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 period: int = 20, percentage: float = 2.5,
                 ma_type: str = "SMA") -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Moving Average Envelopes
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=20
            Period for moving average calculation
        percentage : float, default=2.5
            Percentage for envelope calculation
        ma_type : str, default="SMA"
            Type of moving average ("SMA" or "EMA")
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (upper_envelope, middle_line, lower_envelope) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        results = _backend.ma_envelopes(validated_data, period, percentage, ma_type)
        return self.format_multiple_outputs(results, input_type, index)
# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Volatility Indicators
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .utils import (ema, atr_wilder, true_range, sma, stdev, highest, lowest,
                    rolling_sum, ulcer_index_optimized)
from . import _backend


class ATR(BaseIndicator):
    """
    Average True Range
    
    ATR is a technical analysis indicator that measures market volatility by 
    decomposing the entire range of an asset price for that period.
    
    Formula:
    True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
    ATR = Moving Average of True Range over n periods
    """
    
    def __init__(self):
        super().__init__("ATR")
    
    # Removed redundant ATR calculation - using consolidated utility
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Average True Range
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Number of periods for ATR calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            ATR values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        result = _backend.atr_wilder(high_data, low_data, close_data, period)
        return self.format_output(result, input_type, index)


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands that are 
    standard deviations away from the middle band.
    
    Formula:
    Middle Band = Simple Moving Average (SMA)
    Upper Band = SMA + (Standard Deviation × multiplier)
    Lower Band = SMA - (Standard Deviation × multiplier)
    """
    
    def __init__(self):
        super().__init__("Bollinger Bands")
    
    @staticmethod
    def _calculate_bollinger_bands(data: np.ndarray, period: int, 
                                  std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """O(n) optimized Bollinger Bands calculation using the Rust backend"""
        return _backend.bbands(data, period, std_dev)
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 period: int = 20, std_dev: float = 2.0) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Bollinger Bands
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=20
            Number of periods for moving average and standard deviation
        std_dev : float, default=2.0
            Number of standard deviations for the bands
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (upper_band, middle_band, lower_band) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        if std_dev <= 0:
            raise ValueError(f"Standard deviation multiplier must be positive, got {std_dev}")
        
        results = self._calculate_bollinger_bands(validated_data, period, std_dev)
        return self.format_multiple_outputs(results, input_type, index)


# Helper function for EMA calculation (outside class for Numba)
@jit(nopython=True)
def _calculate_ema_keltner(data: np.ndarray, period: int) -> np.ndarray:
    """EMA calculation for Keltner Channel"""
    n = len(data)
    ema = np.empty(n)
    alpha = 2.0 / (period + 1)
    
    # Seed initial NaNs until period-1
    ema[:period-1] = np.nan
    
    # Initial SMA
    sum_val = 0.0
    for i in range(period):
        sum_val += data[i]
    ema[period-1] = sum_val / period
    
    # Exponential smoothing thereafter
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema


class Keltner(BaseIndicator):
    """
    Keltner Channel
    
    Keltner Channels are volatility-based envelopes set above and below an 
    exponential moving average. The channels use ATR to set channel distance.
    
    Formula:
    Middle Line = EMA of Close
    Upper Channel = EMA + (multiplier × ATR)
    Lower Channel = EMA - (multiplier × ATR)
    """
    
    def __init__(self):
        super().__init__("Keltner Channel")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                  ema_period: int, atr_period: int, 
                                  multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba optimized Keltner Channel calculation"""
        n = len(close)
        
        # Calculate EMA of close (middle line)
        middle = _calculate_ema_keltner(close, ema_period)
        
        # Calculate ATR
        tr = np.empty(n)
        atr = np.full(n, np.nan)
        
        # First TR value
        tr[0] = high[0] - low[0]
        
        # Calculate True Range
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR
        if n >= atr_period:
            # Initial ATR
            sum_tr = 0.0
            for i in range(atr_period):
                sum_tr += tr[i]
            atr[atr_period-1] = sum_tr / atr_period
            
            # Subsequent ATR values
            for i in range(atr_period, n):
                atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
        
        # Calculate upper and lower channels
        upper = np.empty(n)
        lower = np.empty(n)
        
        for i in range(n):
            if np.isnan(atr[i]):
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                upper[i] = middle[i] + multiplier * atr[i]
                lower[i] = middle[i] - multiplier * atr[i]
        
        return upper, middle, lower
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 ema_period: int = 20, atr_period: int = 10, 
                 multiplier: float = 2.0) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Keltner Channel
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        ema_period : int, default=20
            Period for the EMA calculation
        atr_period : int, default=10
            Period for the ATR calculation
        multiplier : float, default=2.0
            Multiplier for the ATR
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (upper_channel, middle_line, lower_channel) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        # Validate parameters
        self.validate_period(ema_period, len(close_data))
        self.validate_period(atr_period, len(close_data))
        if multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {multiplier}")
        
        results = _backend.keltner(high_data, low_data, close_data, ema_period, atr_period, multiplier)
        return self.format_multiple_outputs(results, input_type, index)


class Donchian(BaseIndicator):
    """
    Donchian Channel
    
    Donchian Channels are formed by taking the highest high and the lowest low 
    of the last n periods. The middle line is the average of the upper and lower lines.
    
    Formula:
    Upper Channel = Highest High over n periods
    Lower Channel = Lowest Low over n periods
    Middle Line = (Upper Channel + Lower Channel) / 2
    """
    
    def __init__(self):
        super().__init__("Donchian Channel")
    
    @staticmethod
    def _calculate_donchian_channel(high: np.ndarray, low: np.ndarray, 
                                   period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """O(n) optimized Donchian Channel calculation using utils"""
        # Use optimized O(n) deque-based utilities
        upper = highest(high, period)
        lower = lowest(low, period)
        
        # Calculate middle line
        middle = (upper + lower) / 2.0
        
        return upper, middle, lower
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 period: int = 20) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Donchian Channel
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        period : int, default=20
            Number of periods for the channel calculation
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (upper_channel, middle_line, lower_channel) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        # Align arrays
        high_data, low_data = self.align_arrays(high_data, low_data)
        self.validate_period(period, len(high_data))
        
        results = _backend.donchian(high_data, low_data, period)
        return self.format_multiple_outputs(results, input_type, index)


class Chaikin(BaseIndicator):
    """
    Chaikin Volatility
    
    Chaikin Volatility measures the rate of change of the trading range.
    
    Formula: CV = ((H-L EMA - H-L EMA[n periods ago]) / H-L EMA[n periods ago]) × 100
    """
    
    def __init__(self):
        super().__init__("Chaikin Volatility")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        n = len(data)
        result = np.full(n, np.nan)  # Initialize with NaN
        alpha = 2.0 / (period + 1)
        
        # Initial SMA seed
        sma = 0.0
        for i in range(period):
            sma += data[i]
        result[period-1] = sma / period
        
        for i in range(period, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 ema_period: int = 10, roc_period: int = 10) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Chaikin Volatility
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        ema_period : int, default=10
            Period for EMA of high-low range
        roc_period : int, default=10
            Period for rate of change calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Chaikin Volatility values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        
        cv = _backend.chaikin(high_data, low_data, ema_period, roc_period)
        return self.format_output(cv, input_type, index)


class NATR(BaseIndicator):
    """
    Normalized Average True Range
    
    NATR is ATR expressed as a percentage of closing price.
    
    Formula: NATR = (ATR / Close) × 100
    """
    
    def __init__(self):
        super().__init__("NATR")
        self._atr = ATR()
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Normalized Average True Range
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Period for ATR calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            NATR values in the same format as input
        """
        close_data, input_type, index = self.validate_input(close)

        natr = _backend.natr(high, low, close, period)
        return self.format_output(natr, input_type, index)


class RVI(BaseIndicator):
    """
    Relative Volatility Index
    
    Relative Volatility Index of Donald Dorsey (1993): the share of recent
    volatility that fell on rising bars.
    
    Formula: RVI = 100 * U / (U + D)
    Where: U, D = Wilder's smoothing of the rolling standard deviation on bars
    that closed up and on bars that closed down (see `_backend.rvi_volatility`)
    """
    
    def __init__(self):
        super().__init__("RVI")
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 stdev_period: int = 10, rsi_period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Relative Volatility Index
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        stdev_period : int, default=10
            Window, in bars, of the rolling population standard deviation
        rsi_period : int, default=14
            Bars of Wilder's smoothing (the average the RSI uses)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            RVI values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        result = _backend.rvi_volatility(validated_data, stdev_period, rsi_period)
        return self.format_output(result, input_type, index)


class ULTOSC(BaseIndicator):
    """
    Ultimate Oscillator (Volatility version)
    
    A different implementation focusing on volatility aspects.
    """
    
    def __init__(self):
        super().__init__("Ultimate Oscillator")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ultosc(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         period1: int, period2: int, period3: int) -> np.ndarray:
        """Calculate Ultimate Oscillator"""
        n = len(close)
        result = np.full(n, np.nan)
        
        # Calculate True Range and Buying Pressure
        tr = np.empty(n)
        bp = np.empty(n)
        
        tr[0] = high[0] - low[0]
        bp[0] = close[0] - min(low[0], close[0])
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
            bp[i] = close[i] - min(low[i], close[i-1])
        
        # Calculate Ultimate Oscillator
        max_period = max(period1, period2, period3)
        for i in range(max_period - 1, n):
            # Short period
            bp_sum1 = np.sum(bp[i - period1 + 1:i + 1])
            tr_sum1 = np.sum(tr[i - period1 + 1:i + 1])
            raw1 = bp_sum1 / tr_sum1 if tr_sum1 > 0 else 0
            
            # Medium period
            bp_sum2 = np.sum(bp[i - period2 + 1:i + 1])
            tr_sum2 = np.sum(tr[i - period2 + 1:i + 1])
            raw2 = bp_sum2 / tr_sum2 if tr_sum2 > 0 else 0
            
            # Long period
            bp_sum3 = np.sum(bp[i - period3 + 1:i + 1])
            tr_sum3 = np.sum(tr[i - period3 + 1:i + 1])
            raw3 = bp_sum3 / tr_sum3 if tr_sum3 > 0 else 0
            
            # Ultimate Oscillator formula
            result[i] = 100 * (4 * raw1 + 2 * raw2 + raw3) / 7
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period1: int = 7, period2: int = 14, period3: int = 28) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Ultimate Oscillator
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period1 : int, default=7
            Short period
        period2 : int, default=14
            Medium period
        period3 : int, default=28
            Long period
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Ultimate Oscillator values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        result = _backend.ultosc(high_data, low_data, close_data, period1, period2, period3)
        return self.format_output(result, input_type, index)


class STDDEV(BaseIndicator):
    """
    Standard Deviation
    
    Standard deviation is a measure of volatility.
    
    Formula: STDDEV = sqrt(Σ(Price - SMA)² / n)
    """
    
    def __init__(self):
        super().__init__("Standard Deviation")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_stddev(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized standard deviation calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            mean_val = np.mean(window)
            
            variance = 0.0
            for j in range(period):
                diff = window[j] - mean_val
                variance += diff * diff
            
            result[i] = np.sqrt(variance / period)
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 20) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Standard Deviation
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=20
            Period for standard deviation calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Standard deviation values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = self._calculate_stddev(validated_data, period)
        return self.format_output(result, input_type, index)


class TRANGE(BaseIndicator):
    """
    True Range
    
    True Range is a measure of volatility that accounts for gaps.
    
    Formula: TR = max(H-L, |H-C[prev]|, |L-C[prev]|)
    """
    
    def __init__(self):
        super().__init__("True Range")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_trange(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Numba optimized True Range calculation"""
        n = len(high)
        result = np.empty(n)
        
        # First value
        result[0] = high[0] - low[0]
        
        # Calculate True Range
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            result[i] = max(hl, hc, lc)
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate True Range
        
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
        Union[np.ndarray, pd.Series]
            True Range values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        result = _backend.true_range(high_data, low_data, close_data)
        return self.format_output(result, input_type, index)


class MASS(BaseIndicator):
    """
    Mass Index (Pine Script v6)
    
    The Mass Index uses the high-low range to identify trend reversals 
    based on range expansion.
    
    Pine Script v6 Formula:
    span = high - low
    mi = math.sum(ta.ema(span, 9) / ta.ema(ta.ema(span, 9), 9), length)
    
    Default length = 10
    """
    
    def __init__(self):
        super().__init__("Mass Index")
    
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            result[i] = np.mean(data[i - period + 1:i + 1])
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 length: int = 10) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Mass Index
        
        Pine Script v6 Formula:
        span = high - low
        mi = math.sum(ta.ema(span, 9) / ta.ema(ta.ema(span, 9), 9), length)
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        length : int, default=10
            Period for sum calculation (Pine Script default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Mass Index values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        
        result = _backend.mass(high_data, low_data, length)
        return self.format_output(result, input_type, index)


class BBPercent(BaseIndicator):
    """
    Bollinger Bands %B
    
    %B shows where price is in relation to the bands.
    %B = 1 when price is at the upper band, 0 when at the lower band.
    
    Formula: %B = (Close - Lower Band) / (Upper Band - Lower Band)
    """
    
    def __init__(self):
        super().__init__("BB %B")
    
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
    def _calculate_stddev(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            mean_val = np.mean(window)
            variance = np.mean((window - mean_val) ** 2)
            result[i] = np.sqrt(variance)
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 period: int = 20, std_dev: float = 2.0) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Bollinger Bands %B
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=20
            Period for moving average and standard deviation
        std_dev : float, default=2.0
            Number of standard deviations for the bands
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            %B values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        percent_b = _backend.bbpercent(validated_data, period, std_dev)
        return self.format_output(percent_b, input_type, index)


class BBWidth(BaseIndicator):
    """
    Bollinger Bandwidth
    
    Bollinger Bandwidth measures the width of the Bollinger Bands.
    Used to identify periods of low volatility (squeeze) and high volatility.
    
    Formula: Bandwidth = (Upper Band - Lower Band) / Middle Band
    """
    
    def __init__(self):
        super().__init__("BB Bandwidth")
    
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
    def _calculate_stddev(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            mean_val = np.mean(window)
            variance = np.mean((window - mean_val) ** 2)
            result[i] = np.sqrt(variance)
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 period: int = 20, std_dev: float = 2.0) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Bollinger Bandwidth
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=20
            Period for moving average and standard deviation
        std_dev : float, default=2.0
            Number of standard deviations for the bands
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Bandwidth values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        bandwidth = _backend.bbwidth(validated_data, period, std_dev)
        return self.format_output(bandwidth, input_type, index)


class ChandelierExit(BaseIndicator):
    """
    Chandelier Exit
    
    A trailing stop-loss technique that follows price action.
    
    Formula:
    Long Exit = Highest High(n) - ATR(n) × Multiplier
    Short Exit = Lowest Low(n) + ATR(n) × Multiplier
    """
    
    def __init__(self):
        super().__init__("Chandelier Exit")
    
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
        
        # Calculate ATR using SMA
        for i in range(period - 1, n):
            atr[i] = np.mean(tr[i - period + 1:i + 1])
        
        return atr
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_chandelier(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                             period: int, multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
        """Numba optimized Chandelier Exit calculation"""
        n = len(close)
        long_exit = np.full(n, np.nan)
        short_exit = np.full(n, np.nan)
        
        # Calculate ATR inline
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        tr[0] = high[0] - low[0]
        for j in range(1, n):
            hl = high[j] - low[j]
            hc = abs(high[j] - close[j - 1])
            lc = abs(low[j] - close[j - 1])
            tr[j] = max(hl, hc, lc)
        
        # Calculate ATR using SMA
        for j in range(period - 1, n):
            atr[j] = np.mean(tr[j - period + 1:j + 1])
        
        for i in range(period - 1, n):
            # Highest high and lowest low over period
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            
            # Calculate exits
            long_exit[i] = highest_high - atr[i] * multiplier
            short_exit[i] = lowest_low + atr[i] * multiplier
        
        return long_exit, short_exit
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 22, multiplier: float = 3.0) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Chandelier Exit
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=22
            Period for highest/lowest and ATR calculation
        multiplier : float, default=3.0
            ATR multiplier
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (long_exit, short_exit) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        long_exit, short_exit = _backend.chandelier_exit(high_data, low_data, close_data, period, multiplier)

        results = (long_exit, short_exit)
        return self.format_multiple_outputs(results, input_type, index)



@jit(nopython=True)
def _calculate_stdev_tv(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate standard deviation using TradingView's ta.stdev method
    This matches TradingView's population standard deviation calculation
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        valid_data = window[~np.isnan(window)]
        
        if len(valid_data) == period:
            # TradingView uses population standard deviation (N in denominator)
            mean_val = np.mean(valid_data)
            variance = np.sum((valid_data - mean_val) ** 2) / period
            result[i] = np.sqrt(variance)
    
    return result

class HistoricalVolatility(BaseIndicator):
    """
    Historical Volatility (HV) - matches TradingView exactly
    
    Measures the standard deviation of logarithmic returns over a specified period.
    Uses TradingView's exact formula with 365-day annualization and timeframe detection.
    
    TradingView Formula: 
    hv = 100 * ta.stdev(math.log(close / close[1]), length) * math.sqrt(annual / per)
    Where:
    - annual = 365 (TradingView default)
    - per = 1 for daily/intraday, 7 for weekly+ timeframes
    """
    
    def __init__(self):
        super().__init__("Historical Volatility")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_hv_tv(close: np.ndarray, length: int, annual: int, per: int) -> np.ndarray:
        """
        Calculate Historical Volatility using exact TradingView formula
        hv = 100 * ta.stdev(math.log(close / close[1]), length) * math.sqrt(annual / per)
        """
        n = len(close)
        
        # Calculate log returns: math.log(close / close[1])
        log_returns = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                log_returns[i] = np.log(close[i] / close[i - 1])
        
        # Calculate standard deviation of log returns
        stdev_returns = _calculate_stdev_tv(log_returns, length)
        
        # Apply TradingView formula: 100 * stdev * sqrt(annual / per)
        annualization_factor = np.sqrt(annual / per)
        result = 100.0 * stdev_returns * annualization_factor
        
        return result
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 length: int = 10, annual: int = 365, per: int = 1) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Historical Volatility - matches TradingView exactly
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        length : int, default=10
            Period for volatility calculation (TradingView default)
        annual : int, default=365
            Annual periods for scaling (TradingView uses 365)
        per : int, default=1
            Timeframe periods (1 for daily/intraday, 7 for weekly+)
            TradingView logic:
            - per = 1 if timeframe.isintraday or (timeframe.isdaily and multiplier == 1)  
            - per = 7 otherwise (weekly, monthly timeframes)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Historical volatility values in the same format as input
            Values are annualized percentages (e.g., 20.5 = 20.5% annual volatility)
        """
        close_data, input_type, index = self.validate_input(close)
        self.validate_period(length + 1, len(close_data))
        
        if length < 1:
            raise ValueError(f"Length must be at least 1, got {length}")
        
        if annual <= 0:
            raise ValueError(f"Annual periods must be positive, got {annual}")
            
        if per <= 0:
            raise ValueError(f"Per periods must be positive, got {per}")
        
        result = _backend.hv(close_data, length, annual, per)
        return self.format_output(result, input_type, index)


class UlcerIndex(BaseIndicator):
    """
    Ulcer Index - TradingView Pine Script v4 Implementation
    
    Measures downside risk by calculating the depth and duration of drawdowns.
    Based on TradingView Pine Script v4 implementation by Alex Orekhov (everget).
    
    TradingView Formula:
    highest = highest(src, length)
    drawdown = 100 * (src - highest) / highest
    ulcer = sqrt(sma(pow(drawdown, 2), smoothLength))
    signal = signalType == "SMA" ? sma(ulcer, signalLength) : ema(ulcer, signalLength)
    
    Default Parameters (TradingView):
    - length = 14
    - smoothLength = 14
    - signalLength = 52
    - signalType = "SMA"
    - breakout = 1.5
    """
    
    def __init__(self):
        super().__init__("Ulcer Index")
    
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 length: int = 14, smooth_length: int = 14, signal_length: int = 52, 
                 signal_type: str = "SMA", return_signal: bool = False) -> Union[np.ndarray, pd.Series, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate Ulcer Index using TradingView Pine Script v4 formula
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        length : int, default=14
            Period for highest calculation (TradingView length)
        smooth_length : int, default=14
            Period for smoothing the squared drawdowns (TradingView smoothLength)
        signal_length : int, default=52
            Period for signal line calculation (TradingView signalLength)
        signal_type : str, default="SMA"
            Type of signal smoothing: "SMA" or "EMA" (TradingView signalType)
        return_signal : bool, default=False
            Whether to return both ulcer and signal line
            
        Returns:
        --------
        Union[np.ndarray, pd.Series] or Tuple[np.ndarray, np.ndarray]
            Ulcer Index values (and signal if return_signal=True) in the same format as input
            
        Notes:
        ------
        This implementation matches TradingView Pine Script v4:
        highest = highest(src, length)
        drawdown = 100 * (src - highest) / highest
        ulcer = sqrt(sma(pow(drawdown, 2), smoothLength))
        signal = signalType == "SMA" ? sma(ulcer, signalLength) : ema(ulcer, signalLength)
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(length, len(validated_data))
        self.validate_period(smooth_length, len(validated_data))
        self.validate_period(signal_length, len(validated_data))
        
        if return_signal:
            ulcer, signal = _backend.ulcerindex(validated_data, length, smooth_length,
                                                signal_length, signal_type, True)
            return (self.format_output(ulcer, input_type, index),
                    self.format_output(signal, input_type, index))
        ulcer = _backend.ulcerindex(validated_data, length, smooth_length,
                                    signal_length, signal_type, False)
        return self.format_output(ulcer, input_type, index)


class STARC(BaseIndicator):
    """
    STARC Bands (Stoller Channels) - TradingView Pine Script v2 Implementation
    
    STARC Bands use an SMA and Average True Range to create bands.
    Based on TradingView Pine Script implementation with specific parameters.
    
    TradingView Formula:
    xMA = sma(close, LengthMA)
    xATR = atr(LengthATR)
    xSTARCBandUp = xMA + xATR * K
    xSTARCBandDn = xMA - xATR * K
    
    Default Parameters (TradingView):
    - LengthMA = 5
    - LengthATR = 15
    - K = 1.33
    """
    
    def __init__(self):
        super().__init__("STARC Bands")
    
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
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR"""
        n = len(close)
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        if n > 0:
            tr[0] = high[0] - low[0]
            for i in range(1, n):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i - 1])
                lc = abs(low[i] - close[i - 1])
                tr[i] = max(hl, max(hc, lc))
            
            # Calculate ATR using SMA
            for i in range(period - 1, n):
                sum_tr = 0.0
                for j in range(i - period + 1, i + 1):
                    sum_tr += tr[j]
                atr[i] = sum_tr / period
        
        return atr
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 ma_period: int = 5, atr_period: int = 15, 
                 multiplier: float = 1.33) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate STARC Bands using TradingView Pine Script formula
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        ma_period : int, default=5
            Period for SMA calculation (LengthMA in TradingView)
        atr_period : int, default=15
            Period for ATR calculation (LengthATR in TradingView)
        multiplier : float, default=1.33
            ATR multiplier (K in TradingView)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (upper_band, middle_line, lower_band) in the same format as input
            
        Notes:
        ------
        This implementation matches TradingView Pine Script v2:
        - xMA = sma(close, 5)
        - xATR = atr(15)
        - Upper Band = xMA + xATR * 1.33
        - Lower Band = xMA - xATR * 1.33
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        results = _backend.starc(high_data, low_data, close_data, ma_period, atr_period, multiplier)
        return self.format_multiple_outputs(results, input_type, index)
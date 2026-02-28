# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Volume Indicators
"""

import numpy as np
import pandas as pd
from numba import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .trend import SMA, EMA, WMA
from .volatility import BollingerBands


class OBV(BaseIndicator):
    """
    On Balance Volume

    OBV is a momentum indicator that uses volume flow to predict changes in stock price.

    Formula:
    sign = 1 if close >= close[1], -1 if close < close[1]
    obv = cumsum(sign * volume)
    """
    
    def __init__(self):
        super().__init__("OBV")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Numba optimized OBV calculation"""
        n = len(close)
        obv = np.empty(n)
        
        # First value is 0 (no previous close to compare)
        obv[0] = 0.0
        
        for i in range(1, n):
            if close[i] < close[i-1]:
                sign = -1.0
            else:
                sign = 1.0

            obv[i] = obv[i-1] + (sign * float(volume[i]))
        
        return obv
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate On Balance Volume
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            OBV values in the same format as input
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        # Align arrays
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        result = self._calculate_obv(close_data, volume_data)
        return self.format_output(result, input_type, index)


class OBVSmoothed(BaseIndicator):
    """
    On Balance Volume with Smoothing Options (TradingView Pine Script Implementation)
    
    OBV with various smoothing moving average options and Bollinger Bands support.
    Supports: None, SMA, SMA + Bollinger Bands, EMA, SMMA (RMA), WMA, VWMA
    
    Formula:
    obv = ta.cum(math.sign(ta.change(src)) * volume)
    Smoothed according to selected MA type
    """
    
    def __init__(self):
        super().__init__("OBVSmoothed")
        self._obv = OBV()
        self._sma = SMA()
        self._ema = EMA()
        self._wma = WMA()
        self._bb = BollingerBands()
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rma(values: np.ndarray, length: int) -> np.ndarray:
        """Numba optimized RMA (Running Moving Average / SMMA) calculation"""
        n = len(values)
        rma = np.full(n, np.nan)
        
        # Initialize with first valid value
        first_valid_idx = 0
        for i in range(n):
            if not np.isnan(values[i]):
                first_valid_idx = i
                break
        
        if first_valid_idx >= n:
            return rma
        
        # Calculate initial SMA for the first length period
        sum_val = 0.0
        count = 0
        for i in range(first_valid_idx, min(first_valid_idx + length, n)):
            if not np.isnan(values[i]):
                sum_val += values[i]
                count += 1
        
        if count == length:
            rma[first_valid_idx + length - 1] = sum_val / length
            
            # Calculate RMA for subsequent values
            alpha = 1.0 / length
            for i in range(first_valid_idx + length, n):
                if not np.isnan(values[i]):
                    rma[i] = alpha * values[i] + (1.0 - alpha) * rma[i-1]
        
        return rma
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_vwma(values: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        """Numba optimized Volume Weighted Moving Average calculation"""
        n = len(values)
        vwma = np.full(n, np.nan)
        
        for i in range(length - 1, n):
            sum_vw = 0.0
            sum_v = 0.0
            
            for j in range(i - length + 1, i + 1):
                if not np.isnan(values[j]) and not np.isnan(volume[j]):
                    sum_vw += values[j] * volume[j]
                    sum_v += volume[j]
            
            if sum_v > 0:
                vwma[i] = sum_vw / sum_v
        
        return vwma
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 ma_type: str = "None",
                 ma_length: int = 20,
                 bb_length: int = 20,
                 bb_mult: float = 2.0) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Calculate OBV with smoothing options
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        ma_type : str, default "None"
            Moving average type: "None", "SMA", "SMA + Bollinger Bands", "EMA", "SMMA (RMA)", "WMA", "VWMA"
        ma_length : int, default 20
            Moving average length
        bb_length : int, default 20
            Bollinger Bands length (only used for "SMA + Bollinger Bands")
        bb_mult : float, default 2.0
            Bollinger Bands multiplier (only used for "SMA + Bollinger Bands")
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            If ma_type is "SMA + Bollinger Bands": (obv_smoothed, bb_upper, bb_lower)
            Otherwise: obv_smoothed
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        # Align arrays
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        # Calculate base OBV
        obv = self._obv.calculate(close_data, volume_data)
        
        # Apply smoothing based on ma_type
        if ma_type == "None":
            result = obv
        elif ma_type == "SMA":
            result = self._sma.calculate(obv, ma_length)
        elif ma_type == "SMA + Bollinger Bands":
            bb_middle, bb_upper, bb_lower = self._bb.calculate(obv, bb_length, bb_mult)
            return (self.format_output(bb_middle, input_type, index),
                    self.format_output(bb_upper, input_type, index),
                    self.format_output(bb_lower, input_type, index))
        elif ma_type == "EMA":
            result = self._ema.calculate(obv, ma_length)
        elif ma_type == "SMMA (RMA)":
            result = self._calculate_rma(obv, ma_length)
        elif ma_type == "WMA":
            result = self._wma.calculate(obv, ma_length)
        elif ma_type == "VWMA":
            result = self._calculate_vwma(obv, volume_data, ma_length)
        else:
            raise ValueError(f"Unsupported ma_type: {ma_type}")
        
        return self.format_output(result, input_type, index)


class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price - TradingView Pine Script v6 Implementation
    
    VWAP is the average price a security has traded at throughout the day, 
    based on both volume and price. It gives more weight to prices with higher volume.
    
    TradingView Pine Script v6 Features:
    - Session-based anchoring (Session, Week, Month, Quarter, Year, etc.)
    - Standard deviation and percentage-based bands
    - Multiple band levels
    - Source selection (hlc3, hl2, ohlc4, etc.)
    - Volume validation
    
    Formula: 
    VWAP = Σ(Source × Volume) / Σ(Volume)
    Where: Source = hlc3 by default = (high + low + close) / 3
    
    Bands:
    - Standard Deviation: upper/lower = vwap ± (stdev * multiplier)
    - Percentage: upper/lower = vwap ± (vwap * 0.01 * multiplier)
    """
    
    def __init__(self):
        super().__init__("VWAP")
    
    @staticmethod
    def _calculate_session_vwap(source: np.ndarray, volume: np.ndarray, 
                               session_starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate session-based VWAP with standard deviation
        
        Parameters:
        -----------
        source : np.ndarray
            Source price data (typically hlc3)
        volume : np.ndarray
            Volume data
        session_starts : np.ndarray
            Boolean array indicating session start points
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (vwap, stdev) arrays
        """
        n = len(source)
        vwap = np.full(n, np.nan)
        stdev = np.full(n, np.nan)
        
        session_sum_pv = 0.0
        session_sum_v = 0.0
        session_sum_pv2 = 0.0  # For variance calculation
        
        for i in range(n):
            # Check if new session starts
            if session_starts[i] or i == 0:
                session_sum_pv = 0.0
                session_sum_v = 0.0
                session_sum_pv2 = 0.0
            
            # Add current bar to session
            pv = source[i] * volume[i]
            session_sum_pv += pv
            session_sum_v += volume[i]
            session_sum_pv2 += (source[i] * source[i] * volume[i])
            
            if session_sum_v > 0:
                # Calculate VWAP
                vwap[i] = session_sum_pv / session_sum_v
                
                # Calculate variance and standard deviation
                # Var = E[X²] - E[X]²
                # Where E[X] = VWAP and E[X²] = sum(price² * volume) / sum(volume)
                mean_squared_price = session_sum_pv2 / session_sum_v
                variance = mean_squared_price - (vwap[i] ** 2)
                stdev[i] = np.sqrt(max(0, variance))  # Ensure non-negative
            else:
                vwap[i] = source[i]
                stdev[i] = 0.0
        
        return vwap, stdev
    
    @staticmethod
    def _detect_session_starts(timestamps: Optional[np.ndarray] = None, 
                              session_length: int = 1440) -> np.ndarray:
        """
        Detect session start points based on timestamps or fixed intervals
        
        Parameters:
        -----------
        timestamps : Optional[np.ndarray]
            Unix timestamps (if available)
        session_length : int, default=1440
            Session length in minutes (1440 = 1 day)
            
        Returns:
        --------
        np.ndarray
            Boolean array indicating session starts
        """
        if timestamps is not None:
            n = len(timestamps)
            session_starts = np.zeros(n, dtype=bool)
            session_starts[0] = True
            
            # Detect new trading sessions (e.g., new trading day)
            for i in range(1, n):
                # Simple day change detection (this is simplified)
                # In real implementation, would use proper timezone handling
                current_day = int(timestamps[i] // 86400)  # 86400 seconds in a day
                prev_day = int(timestamps[i-1] // 86400)
                
                if current_day != prev_day:
                    session_starts[i] = True
                    
            return session_starts
        else:
            # For data without timestamps, use fixed intervals
            n = 100  # Default assumption - will be resized
            session_starts = np.zeros(n, dtype=bool)
            session_starts[0] = True
            return session_starts
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 source_type: str = "hlc3",
                 anchor: str = "Session",
                 session_starts: Optional[Union[np.ndarray, list]] = None,
                 timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Volume Weighted Average Price - TradingView Pine Script v6
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        source_type : str, default="hlc3"
            Source calculation: "hlc3", "hl2", "ohlc4", "close"
        anchor : str, default="Session"
            Anchor period: "Session", "Week", "Month", etc.
        session_starts : Optional[Union[np.ndarray, list]]
            Boolean array indicating session start points
        timestamps : Optional[Union[np.ndarray, pd.DatetimeIndex]]
            Timestamps for session detection
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            VWAP values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        # Align arrays
        high_data, low_data, close_data, volume_data = self.align_arrays(
            high_data, low_data, close_data, volume_data)
        
        # TradingView-style volume validation
        total_volume = np.sum(volume_data)
        if total_volume == 0:
            raise RuntimeError("No volume is provided by the data vendor.")
        
        # Calculate source based on type
        if source_type == "hlc3":
            source = (high_data + low_data + close_data) / 3.0
        elif source_type == "hl2":
            source = (high_data + low_data) / 2.0
        elif source_type == "ohlc4":
            # For ohlc4, we would need open prices, fallback to hlc3
            source = (high_data + low_data + close_data) / 3.0
        elif source_type == "close":
            source = close_data
        else:
            source = (high_data + low_data + close_data) / 3.0
        
        # Handle session starts
        if session_starts is not None:
            session_starts_array = np.array(session_starts, dtype=bool)
        else:
            # Detect sessions based on timestamps or use simple method
            if timestamps is not None:
                timestamp_array = np.array(timestamps)
                if hasattr(timestamps, 'values'):  # pandas DatetimeIndex
                    timestamp_array = timestamps.values.astype('datetime64[s]').astype(int)
                session_starts_array = self._detect_session_starts(timestamp_array)
            else:
                # Default: treat as single session (cumulative VWAP)
                session_starts_array = np.zeros(len(source), dtype=bool)
                session_starts_array[0] = True
        
        # Ensure session_starts array has correct length
        if len(session_starts_array) != len(source):
            session_starts_array = np.zeros(len(source), dtype=bool)
            session_starts_array[0] = True
        
        # Calculate VWAP
        vwap, _ = self._calculate_session_vwap(source, volume_data, session_starts_array)
        
        return self.format_output(vwap, input_type, index)
    
    def calculate_with_bands(self, high: Union[np.ndarray, pd.Series, list],
                           low: Union[np.ndarray, pd.Series, list],
                           close: Union[np.ndarray, pd.Series, list],
                           volume: Union[np.ndarray, pd.Series, list],
                           source_type: str = "hlc3",
                           anchor: str = "Session",
                           band_mode: str = "Standard Deviation",
                           band_multipliers: Tuple[float, ...] = (1.0, 2.0, 3.0),
                           session_starts: Optional[Union[np.ndarray, list]] = None,
                           timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Union[Tuple, Tuple]:
        """
        Calculate VWAP with bands - TradingView Pine Script v6 Implementation
        
        Parameters:
        -----------
        high, low, close, volume : Union[np.ndarray, pd.Series, list]
            OHLC and volume data
        source_type : str, default="hlc3"
            Source calculation type
        anchor : str, default="Session"
            Anchor period
        band_mode : str, default="Standard Deviation"
            "Standard Deviation" or "Percentage"
        band_multipliers : Tuple[float, ...], default=(1.0, 2.0, 3.0)
            Multipliers for each band level
        session_starts : Optional[Union[np.ndarray, list]]
            Session start indicators
        timestamps : Optional[Union[np.ndarray, pd.DatetimeIndex]]
            Timestamps for session detection
            
        Returns:
        --------
        Union[Tuple, Tuple]
            (vwap, upper_bands, lower_bands) where bands are tuples of arrays
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        # Align arrays
        high_data, low_data, close_data, volume_data = self.align_arrays(
            high_data, low_data, close_data, volume_data)
        
        # TradingView-style volume validation
        total_volume = np.sum(volume_data)
        if total_volume == 0:
            raise RuntimeError("No volume is provided by the data vendor.")
        
        # Calculate source
        if source_type == "hlc3":
            source = (high_data + low_data + close_data) / 3.0
        elif source_type == "hl2":
            source = (high_data + low_data) / 2.0
        elif source_type == "close":
            source = close_data
        else:
            source = (high_data + low_data + close_data) / 3.0
        
        # Handle session starts
        if session_starts is not None:
            session_starts_array = np.array(session_starts, dtype=bool)
        else:
            if timestamps is not None:
                timestamp_array = np.array(timestamps)
                if hasattr(timestamps, 'values'):
                    timestamp_array = timestamps.values.astype('datetime64[s]').astype(int)
                session_starts_array = self._detect_session_starts(timestamp_array)
            else:
                session_starts_array = np.zeros(len(source), dtype=bool)
                session_starts_array[0] = True
        
        # Ensure correct length
        if len(session_starts_array) != len(source):
            session_starts_array = np.zeros(len(source), dtype=bool)
            session_starts_array[0] = True
        
        # Calculate VWAP and standard deviation
        vwap, stdev = self._calculate_session_vwap(source, volume_data, session_starts_array)
        
        # Calculate bands
        upper_bands = []
        lower_bands = []
        
        for multiplier in band_multipliers:
            if band_mode == "Standard Deviation":
                band_basis = stdev * multiplier
            else:  # Percentage
                band_basis = vwap * 0.01 * multiplier
            
            upper_band = vwap + band_basis
            lower_band = vwap - band_basis
            
            upper_bands.append(self.format_output(upper_band, input_type, index))
            lower_bands.append(self.format_output(lower_band, input_type, index))
        
        vwap_formatted = self.format_output(vwap, input_type, index)
        
        return vwap_formatted, tuple(upper_bands), tuple(lower_bands)


class MFI(BaseIndicator):
    """
    Money Flow Index
    
    MFI is a momentum indicator that uses both price and volume to measure 
    buying and selling pressure. It is also known as Volume-Weighted RSI.
    
    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price × Volume
    3. Positive/Negative Money Flow based on Typical Price comparison
    4. Money Ratio = Positive Money Flow / Negative Money Flow
    5. MFI = 100 - (100 / (1 + Money Ratio))
    """
    
    def __init__(self):
        super().__init__("MFI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      volume: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized MFI aligned with TA-Lib"""
        n = len(close)
        result = np.full(n, np.nan)
        
        tp = (high + low + close) / 3.0
        rmf = tp * volume  # raw money flow
        
        # Pre-compute positive / negative flows per bar
        pos_raw = np.zeros(n)
        neg_raw = np.zeros(n)
        for i in range(1, n):
            if tp[i] > tp[i - 1]:
                pos_raw[i] = rmf[i]
            elif tp[i] < tp[i - 1]:
                neg_raw[i] = rmf[i]
        
        # Rolling window sums
        pos_sum = 0.0
        neg_sum = 0.0
        for i in range(1, n):
            pos_sum += pos_raw[i]
            neg_sum += neg_raw[i]
            
            if i >= period:
                pos_sum -= pos_raw[i - period]
                neg_sum -= neg_raw[i - period]
            
            if i >= period - 1:
                if neg_sum == 0:
                    result[i] = 100.0
                else:
                    m_ratio = pos_sum / neg_sum
                    result[i] = 100.0 - (100.0 / (1.0 + m_ratio))
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Money Flow Index
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        period : int, default=14
            Number of periods for MFI calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            MFI values (range: 0 to 100) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        # Align arrays
        high_data, low_data, close_data, volume_data = self.align_arrays(high_data, low_data, close_data, volume_data)
        self.validate_period(period, len(close_data))
        
        result = self._calculate_mfi(high_data, low_data, close_data, volume_data, period)
        return self.format_output(result, input_type, index)


class ADL(BaseIndicator):
    """
    Accumulation/Distribution Line
    
    ADL is a volume-based indicator designed to measure the cumulative flow 
    of money into and out of a security.
    
    Formula: ADL = Previous ADL + Money Flow Volume
    Where: Money Flow Volume = Volume × ((Close - Low) - (High - Close)) / (High - Low)
    """
    
    def __init__(self):
        super().__init__("ADL")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_adl(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      volume: np.ndarray) -> np.ndarray:
        """Numba optimized ADL calculation"""
        n = len(close)
        result = np.full(n, np.nan)
        
        result[0] = 0.0  # Seed baseline at 0 as per common definition
        
        for i in range(1, n):
            if high[i] != low[i]:
                mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                mfm = 0.0
            
            mfv = mfm * volume[i]
            result[i] = result[i - 1] + mfv
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Accumulation/Distribution Line
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            ADL values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        high_data, low_data, close_data, volume_data = self.align_arrays(high_data, low_data, close_data, volume_data)
        
        result = self._calculate_adl(high_data, low_data, close_data, volume_data)
        return self.format_output(result, input_type, index)


class CMF(BaseIndicator):
    """
    Chaikin Money Flow
    
    CMF is the sum of Money Flow Volume over a period divided by the sum of volume.
    
    Formula: CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)
    """
    
    def __init__(self):
        super().__init__("CMF")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      volume: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized CMF calculation"""
        n = len(close)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            sum_mfv = 0.0
            sum_volume = 0.0
            
            for j in range(period):
                idx = i - period + 1 + j
                
                if high[idx] != low[idx]:
                    mfm = ((close[idx] - low[idx]) - (high[idx] - close[idx])) / (high[idx] - low[idx])
                else:
                    mfm = 0
                
                mfv = mfm * volume[idx]
                sum_mfv += mfv
                sum_volume += volume[idx]
            
            if sum_volume > 0:
                result[i] = sum_mfv / sum_volume
            else:
                result[i] = 0
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 period: int = 20) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Chaikin Money Flow
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        period : int, default=20
            Number of periods for CMF calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            CMF values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        high_data, low_data, close_data, volume_data = self.align_arrays(high_data, low_data, close_data, volume_data)
        self.validate_period(period, len(close_data))
        
        result = self._calculate_cmf(high_data, low_data, close_data, volume_data, period)
        return self.format_output(result, input_type, index)


class EMV(BaseIndicator):
    """
    Ease of Movement - matches TradingView exactly
    
    EMV relates price change to volume and is particularly useful 
    for assessing the strength of a trend. TradingView version includes
    automatic SMA smoothing.
    
    TradingView Formula: EMV = SMA(div * change(hl2) * (high - low) / volume, length)
    Where: hl2 = (high + low) / 2
           change(hl2) = current hl2 - previous hl2
    """
    
    def __init__(self):
        super().__init__("EMV")
    
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
    def _calculate_emv_raw(high: np.ndarray, low: np.ndarray, volume: np.ndarray,
                          divisor: float) -> np.ndarray:
        """Calculate raw EMV values before smoothing - matches TradingView formula"""
        n = len(high)
        result = np.full(n, np.nan)
        
        for i in range(1, n):
            # Calculate hl2 (typical price)
            hl2_current = (high[i] + low[i]) / 2
            hl2_previous = (high[i-1] + low[i-1]) / 2
            
            # Change in hl2 (ta.change(hl2) in TradingView)
            change_hl2 = hl2_current - hl2_previous
            
            # High - Low range
            high_low_range = high[i] - low[i]
            
            # TradingView formula: div * change(hl2) * (high - low) / volume
            if volume[i] > 0 and high_low_range > 0:
                result[i] = divisor * change_hl2 * high_low_range / volume[i]
            else:
                result[i] = 0.0
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 length: int = 14, divisor: int = 10000) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Ease of Movement - matches TradingView exactly
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        length : int, default=14
            Period for SMA smoothing (TradingView default)
        divisor : int, default=10000
            Divisor for scaling EMV values (TradingView default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            EMV values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        volume_data, _, _ = self.validate_input(volume)
        
        high_data, low_data, volume_data = self.align_arrays(high_data, low_data, volume_data)
        self.validate_period(length, len(high_data))
        
        if divisor <= 0:
            raise ValueError(f"Divisor must be positive, got {divisor}")
        
        # Calculate raw EMV values
        raw_emv = self._calculate_emv_raw(high_data, low_data, volume_data, float(divisor))
        
        # Apply SMA smoothing (TradingView always smooths)
        smoothed_emv = self._calculate_sma(raw_emv, length)
        
        return self.format_output(smoothed_emv, input_type, index)


class FI(BaseIndicator):
    """
    Elder Force Index - matches TradingView exactly
    
    The Elder Force Index (EFI) combines price and volume to assess the power 
    used to move the price of an asset. TradingView version applies EMA smoothing
    to reduce noise.
    
    TradingView Formula: EFI = EMA(volume * change(close), length)
    Where: change(close) = close - close[1]
    """
    
    def __init__(self):
        super().__init__("Elder Force Index")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        n = len(data)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid (non-NaN) value
        first_valid = -1
        for i in range(n):
            if not np.isnan(data[i]):
                first_valid = i
                result[i] = data[i]
                break
        
        if first_valid == -1:
            return result
        
        # Calculate EMA
        for i in range(first_valid + 1, n):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_raw_fi(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate raw Force Index values"""
        n = len(close)
        result = np.full(n, np.nan)
        
        for i in range(1, n):
            # TradingView: ta.change(close) * volume
            price_change = close[i] - close[i-1]
            result[i] = volume[i] * price_change
        
        return result
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 length: int = 13) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Elder Force Index - matches TradingView exactly
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        length : int, default=13
            Period for EMA smoothing (TradingView default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Elder Force Index values in the same format as input
            
        Raises:
        -------
        ValueError
            If no volume is provided by the data vendor (cumulative volume is zero)
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        self.validate_period(length, len(close_data))
        
        # TradingView volume validation: check if cumulative volume is zero
        cumulative_volume = np.nansum(volume_data)
        if cumulative_volume == 0:
            raise ValueError("No volume is provided by the data vendor.")
        
        # Calculate raw Force Index: volume * change(close)
        raw_fi = self._calculate_raw_fi(close_data, volume_data)
        
        # Apply EMA smoothing (TradingView: ta.ema(raw_fi, length))
        smoothed_fi = self._calculate_ema(raw_fi, length)
        
        return self.format_output(smoothed_fi, input_type, index)


class NVI(BaseIndicator):
    """
    Negative Volume Index (Pine Script Implementation)
    
    NVI focuses on days when volume decreases from the previous day.
    Uses cumulative sum of rate of change instead of multiplication.
    
    Pine Script Formula:
    xROC = roc(close, 1)
    nRes = iff(volume < volume[1], nz(nRes[1], 0) + xROC, nz(nRes[1], 0))
    
    Where: ROC = (Close - Previous Close) / Previous Close * 100
    """
    
    def __init__(self):
        super().__init__("NVI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_nvi(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Numba optimized NVI calculation (Pine Script method)"""
        n = len(close)
        result = np.full(n, np.nan)
        
        result[0] = 0.0  # Start with 0 (Pine Script uses nz(..., 0))
        
        for i in range(1, n):
            # Calculate ROC (rate of change) * 100
            if close[i-1] != 0:
                roc = ((close[i] - close[i-1]) / close[i-1]) * 100.0
            else:
                roc = 0.0
            
            # Pine Script logic: if volume decreases, add ROC, else keep previous value
            if volume[i] < volume[i-1]:
                result[i] = result[i-1] + roc
            else:
                result[i] = result[i-1]
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA with NaN handling"""
        n = len(data)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        first_valid_idx = -1
        for i in range(n):
            if not np.isnan(data[i]):
                first_valid_idx = i
                break
        
        if first_valid_idx == -1:
            return result
            
        # Initialize with first valid value
        result[first_valid_idx] = data[first_valid_idx]
        
        # Continue EMA calculation
        for i in range(first_valid_idx + 1, n):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
        
        return result
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Negative Volume Index (Pine Script Implementation)
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            NVI values in the same format as input
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        result = self._calculate_nvi(close_data, volume_data)
        return self.format_output(result, input_type, index)
    
    def calculate_with_ema(self, close: Union[np.ndarray, pd.Series, list],
                          volume: Union[np.ndarray, pd.Series, list],
                          ema_length: int = 255) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate NVI with EMA (Complete Pine Script Implementation)
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        ema_length : int, default=255
            EMA period (Pine Script default)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (nvi, nvi_ema) in the same format as input
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        # Calculate NVI
        nvi = self._calculate_nvi(close_data, volume_data)
        
        # Calculate EMA of NVI
        nvi_ema = self._calculate_ema(nvi, ema_length)
        
        results = (nvi, nvi_ema)
        return self.format_multiple_outputs(results, input_type, index)


class PVI(BaseIndicator):
    """
    Positive Volume Index (TradingView Pine Script Implementation)
    
    PVI focuses on days when volume increases from the previous day.
    
    TradingView Pine Script Formula:
    pvi := na(pvi[1]) ? initial : (change(volume) > 0 ? pvi[1] * close / close[1] : pvi[1])
    
    Where: change(volume) > 0 means volume[i] > volume[i-1]
    """
    
    def __init__(self):
        super().__init__("PVI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_pvi(close: np.ndarray, volume: np.ndarray, initial_value: float) -> np.ndarray:
        """Numba optimized PVI calculation (TradingView Pine Script method)"""
        n = len(close)
        result = np.full(n, np.nan)
        
        result[0] = initial_value  # Start with initial value (TradingView default: 100)
        
        for i in range(1, n):
            # TradingView formula: change(volume) > 0 ? pvi[1] * close / close[1] : pvi[1]
            if volume[i] > volume[i-1]:  # change(volume) > 0
                if close[i-1] != 0:
                    result[i] = result[i-1] * (close[i] / close[i-1])
                else:
                    result[i] = result[i-1]
            else:
                result[i] = result[i-1]  # Keep previous value
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA with NaN handling"""
        n = len(data)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        first_valid_idx = -1
        for i in range(n):
            if not np.isnan(data[i]):
                first_valid_idx = i
                break
        
        if first_valid_idx == -1:
            return result
            
        # Initialize with first valid value
        result[first_valid_idx] = data[first_valid_idx]
        
        # Continue EMA calculation
        for i in range(first_valid_idx + 1, n):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA with NaN handling"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            valid_values = window[~np.isnan(window)]
            if len(valid_values) > 0:
                result[i] = np.mean(valid_values)
        
        return result
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 initial_value: float = 100.0) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Positive Volume Index (TradingView Pine Script Implementation)
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        initial_value : float, default=100.0
            Initial PVI value (TradingView default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            PVI values in the same format as input
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        result = self._calculate_pvi(close_data, volume_data, initial_value)
        return self.format_output(result, input_type, index)
    
    def calculate_with_signal(self, close: Union[np.ndarray, pd.Series, list],
                             volume: Union[np.ndarray, pd.Series, list],
                             initial_value: float = 100.0,
                             signal_type: str = "EMA",
                             signal_length: int = 255) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate PVI with Signal Line (Complete TradingView Pine Script Implementation)
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        initial_value : float, default=100.0
            Initial PVI value (TradingView default)
        signal_type : str, default="EMA"
            Signal smoothing type ("EMA" or "SMA")
        signal_length : int, default=255
            Signal line period (TradingView default)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (pvi, signal) in the same format as input
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        # Calculate PVI
        pvi = self._calculate_pvi(close_data, volume_data, initial_value)
        
        # Calculate signal line
        if signal_type.upper() == "EMA":
            signal = self._calculate_ema(pvi, signal_length)
        elif signal_type.upper() == "SMA":
            signal = self._calculate_sma(pvi, signal_length)
        else:
            raise ValueError(f"Invalid signal_type: {signal_type}. Must be 'EMA' or 'SMA'")
        
        results = (pvi, signal)
        return self.format_multiple_outputs(results, input_type, index)


class VOLOSC(BaseIndicator):
    """
    Volume Oscillator - TradingView Pine Script v6 Implementation
    
    Volume Oscillator shows the relationship between two exponential moving averages of volume.
    
    TradingView Pine Script v6 Formula:
    short = ta.ema(volume, shortlen)
    long = ta.ema(volume, longlen)
    osc = 100 * (short - long) / long
    
    Where:
    - short = EMA of volume with short length
    - long = EMA of volume with long length
    - osc = Volume Oscillator percentage
    """
    
    def __init__(self):
        super().__init__("VO")
    
    @staticmethod
    def _calculate_ema_safe(data: np.ndarray, period: int) -> np.ndarray:
        """Safe EMA calculation without Numba issues"""
        n = len(data)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        first_valid_idx = -1
        for i in range(n):
            if not np.isnan(data[i]) and data[i] >= 0:  # Volume should be non-negative
                first_valid_idx = i
                break
        
        if first_valid_idx == -1:
            return result
            
        # Initialize with first valid value
        result[first_valid_idx] = data[first_valid_idx]
        
        # Continue EMA calculation
        for i in range(first_valid_idx + 1, n):
            if not np.isnan(data[i]) and data[i] >= 0:
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
        
        return result
    
    def calculate(self, volume: Union[np.ndarray, pd.Series, list],
                 short_length: int = 5, long_length: int = 10,
                 check_volume_validity: bool = True) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Volume Oscillator - TradingView Pine Script v6 Implementation
        
        Parameters:
        -----------
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        short_length : int, default=5
            Short EMA length (TradingView default)
        long_length : int, default=10
            Long EMA length (TradingView default)
        check_volume_validity : bool, default=True
            Check for valid volume data (TradingView style validation)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Volume Oscillator values in the same format as input
            
        Notes:
        ------
        TradingView Pine Script v6 Formula:
        short = ta.ema(volume, shortlen)
        long = ta.ema(volume, longlen)
        osc = 100 * (short - long) / long
        
        The TradingView script includes volume validation that raises an error
        if no volume is provided by the data vendor.
        """
        validated_volume, input_type, index = self.validate_input(volume)
        
        # TradingView-style volume validation
        if check_volume_validity:
            total_volume = np.nansum(validated_volume)
            if total_volume == 0:
                raise RuntimeError("No volume is provided by the data vendor.")
        
        # Calculate EMA moving averages (TradingView uses EMA, not SMA)
        short_ema = self._calculate_ema_safe(validated_volume, short_length)
        long_ema = self._calculate_ema_safe(validated_volume, long_length)
        
        # Calculate Volume Oscillator using TradingView formula
        vo = np.full_like(validated_volume, np.nan, dtype=np.float64)
        for i in range(len(validated_volume)):
            if not np.isnan(long_ema[i]) and long_ema[i] != 0:
                # TradingView formula: osc = 100 * (short - long) / long
                vo[i] = 100.0 * (short_ema[i] - long_ema[i]) / long_ema[i]
            else:
                vo[i] = np.nan
        
        return self.format_output(vo, input_type, index)


class VROC(BaseIndicator):
    """
    Volume Rate of Change
    
    VROC measures the rate of change in volume.
    
    Formula: VROC = ((Volume - Volume[n periods ago]) / Volume[n periods ago]) × 100
    """
    
    def __init__(self):
        super().__init__("VROC")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_vroc(volume: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized VROC calculation"""
        n = len(volume)
        result = np.full(n, np.nan)
        
        for i in range(period, n):
            if volume[i - period] != 0:
                result[i] = ((volume[i] - volume[i - period]) / volume[i - period]) * 100
            else:
                result[i] = 0
        
        return result
    
    def calculate(self, volume: Union[np.ndarray, pd.Series, list], period: int = 25) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Volume Rate of Change
        
        Parameters:
        -----------
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        period : int, default=25
            Number of periods to look back
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            VROC values in the same format as input
        """
        validated_volume, input_type, index = self.validate_input(volume)
        self.validate_period(period, len(validated_volume))
        
        result = self._calculate_vroc(validated_volume, period)
        return self.format_output(result, input_type, index)


class KlingerVolumeOscillator(BaseIndicator):
    """
    Klinger Volume Oscillator (KVO) - matches TradingView exactly
    
    The KVO is designed to predict price reversals in a market by comparing 
    volume to price movement.
    
    TradingView Formula:
    xTrend = iff(hlc3 > hlc3[1], volume * 100, -volume * 100)
    xFast = ema(xTrend, FastX)
    xSlow = ema(xTrend, SlowX)
    xKVO = xFast - xSlow
    xTrigger = ema(xKVO, TrigLen)
    """
    
    def __init__(self):
        super().__init__("KVO")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_kvo_tv(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                         trig_len: int, fast_x: int, slow_x: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate KVO using exact TradingView logic
        TradingView formula:
        xTrend = iff(hlc3 > hlc3[1], volume * 100, -volume * 100)
        xFast = ema(xTrend, FastX)
        xSlow = ema(xTrend, SlowX)
        xKVO = xFast - xSlow
        xTrigger = ema(xKVO, TrigLen)
        """
        n = len(close)
        
        # Calculate hlc3 (typical price)
        hlc3 = (high + low + close) / 3.0
        
        # Calculate xTrend using TradingView logic
        # xTrend = iff(hlc3 > hlc3[1], volume * 100, -volume * 100)
        x_trend = np.zeros(n)  # Initialize with zeros instead of NaN
        x_trend[0] = volume[0] * 100.0  # First value assumes positive
        
        for i in range(1, n):
            if hlc3[i] > hlc3[i - 1]:
                x_trend[i] = volume[i] * 100.0
            else:
                x_trend[i] = -volume[i] * 100.0
        
        # Calculate EMAs using TradingView logic
        # xFast = ema(xTrend, FastX)
        x_fast = np.zeros(n)
        fast_alpha = 2.0 / (fast_x + 1)
        
        # Initialize EMA with first value
        x_fast[0] = x_trend[0]
        for i in range(1, n):
            x_fast[i] = fast_alpha * x_trend[i] + (1 - fast_alpha) * x_fast[i - 1]
        
        # xSlow = ema(xTrend, SlowX)
        x_slow = np.zeros(n)
        slow_alpha = 2.0 / (slow_x + 1)
        
        # Initialize EMA with first value
        x_slow[0] = x_trend[0]
        for i in range(1, n):
            x_slow[i] = slow_alpha * x_trend[i] + (1 - slow_alpha) * x_slow[i - 1]
        
        # xKVO = xFast - xSlow
        x_kvo = x_fast - x_slow
        
        
        # xTrigger = ema(xKVO, TrigLen)
        x_trigger = np.zeros(n)
        trig_alpha = 2.0 / (trig_len + 1)
        
        # Initialize trigger EMA with first KVO value
        x_trigger[0] = x_kvo[0]
        for i in range(1, n):
            x_trigger[i] = trig_alpha * x_kvo[i] + (1 - trig_alpha) * x_trigger[i - 1]
        
        return x_kvo, x_trigger
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 trig_len: int = 13, fast_x: int = 34, slow_x: int = 55) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Klinger Volume Oscillator - matches TradingView exactly
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        trig_len : int, default=13
            Trigger line EMA period (TradingView: TrigLen)
        fast_x : int, default=34
            Fast EMA period (TradingView: FastX)
        slow_x : int, default=55
            Slow EMA period (TradingView: SlowX)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (kvo, trigger) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        # Align arrays
        high_data, low_data, close_data, volume_data = self.align_arrays(high_data, low_data, close_data, volume_data)
        
        # Validate parameters
        for param, name in [(trig_len, "trig_len"), (fast_x, "fast_x"), (slow_x, "slow_x")]:
            if param <= 0:
                raise ValueError(f"{name} must be positive, got {param}")
        
        kvo, trigger = self._calculate_kvo_tv(high_data, low_data, close_data, volume_data, trig_len, fast_x, slow_x)
        
        results = (kvo, trigger)
        return self.format_multiple_outputs(results, input_type, index)


class PriceVolumeTrend(BaseIndicator):
    """
    Price Volume Trend (PVT) - TradingView Pine Script Implementation
    
    PVT combines price and volume to show the cumulative volume based on 
    price changes. Uses the TradingView Pine Script formula.
    
    Formula: vt = ta.cum(ta.change(src)/src[1]*volume)
    Which expands to: PVT = cumsum((close[i] - close[i-1]) / close[i-1] * volume[i])
    """
    
    def __init__(self):
        super().__init__("PVT")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_pvt(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Numba optimized PVT calculation (TradingView Pine Script formula)"""
        n = len(close)
        pvt = np.zeros(n)
        
        # First value is always 0 (no previous close to compare)
        pvt[0] = 0.0
        
        # Calculate PVT using TradingView formula: cum(change(close)/close[1]*volume)
        for i in range(1, n):
            if close[i-1] != 0.0:
                # TradingView formula: change(src)/src[1]*volume
                # change(src) = close[i] - close[i-1]
                # src[1] = close[i-1] (previous close)
                price_change_ratio = (close[i] - close[i-1]) / close[i-1]
                pvt_change = price_change_ratio * volume[i]
                pvt[i] = pvt[i-1] + pvt_change
            else:
                # Handle division by zero - keep previous value
                pvt[i] = pvt[i-1]
        
        return pvt
    
    def calculate(self, close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Price Volume Trend (TradingView Pine Script Implementation)
        
        Parameters:
        -----------
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            PVT values in the same format as input
            Formula: vt = ta.cum(ta.change(src)/src[1]*volume)
        """
        close_data, input_type, index = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        close_data, volume_data = self.align_arrays(close_data, volume_data)
        
        result = self._calculate_pvt(close_data, volume_data)
        return self.format_output(result, input_type, index)


class RVOL(BaseIndicator):
    """
    Relative Volume (RVOL)
    
    Compares current volume to average volume over a specified period.
    
    Formula: RVOL = Current Volume / Average Volume
    """
    
    def __init__(self):
        super().__init__("Relative Volume")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rvol(volume: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized RVOL calculation"""
        n = len(volume)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            # Calculate average volume over the period
            avg_volume = 0.0
            for j in range(i - period + 1, i + 1):
                avg_volume += volume[j]
            avg_volume = avg_volume / period
            
            # Avoid division by zero
            if avg_volume > 0:
                result[i] = volume[i] / avg_volume
            else:
                result[i] = 1.0  # Default to 1.0 when average volume is 0
        
        return result
    
    def calculate(self, volume: Union[np.ndarray, pd.Series, list], period: int = 20) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Relative Volume
        
        Parameters:
        -----------
        volume : Union[np.ndarray, pd.Series, list]
            Volume data
        period : int, default=20
            Period for average volume calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            RVOL values in the same format as input
        """
        volume_data, input_type, index = self.validate_input(volume)
        self.validate_period(period, len(volume_data))
        
        result = self._calculate_rvol(volume_data, period)
        return self.format_output(result, input_type, index)
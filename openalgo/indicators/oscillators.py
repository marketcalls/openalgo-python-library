# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Oscillators
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .utils import sma, ema, highest, lowest, rolling_sum, true_range, cmo_optimized
from . import _backend


@jit(nopython=True)
def _calculate_swma_helper(data: np.ndarray) -> np.ndarray:
    """Helper function to calculate Symmetrically Weighted Moving Average (SWMA)
    SWMA is a 4-period weighted moving average with weights [1, 2, 2, 1] / 6
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(3, n):
        # SWMA weights: [1, 2, 2, 1] / 6
        result[i] = (data[i-3] + 2*data[i-2] + 2*data[i-1] + data[i]) / 6.0
    
    return result


class ROC(BaseIndicator):
    """
    Rate of Change (Price Oscillator)
    
    ROC measures the percentage change in price from n periods ago.
    
    Formula: ROC = ((Price - Price[n periods ago]) / Price[n periods ago]) × 100
    """
    
    def __init__(self):
        super().__init__("ROC")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_roc(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized ROC calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period, n):
            if data[i - period] != 0:
                result[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
            else:
                result[i] = 0.0
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 12) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Rate of Change
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=12
            Number of periods to look back
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            ROC values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = _backend.roc_osc(validated_data, period)
        return self.format_output(result, input_type, index)


class CMO(BaseIndicator):
    """
    Chande Momentum Oscillator
    
    CMO is a momentum oscillator developed by Tushar Chande.
    
    Formula: CMO = 100 × (Sum of Up Days - Sum of Down Days) / (Sum of Up Days + Sum of Down Days)
    """
    
    def __init__(self):
        super().__init__("CMO")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_cmo(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized CMO calculation"""
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
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Chande Momentum Oscillator
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=14
            Number of periods for CMO calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            CMO values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period + 1, len(validated_data))  # +1 for diff
        result = _backend.cmo(validated_data, period)
        return self.format_output(result, input_type, index)


class TRIX(BaseIndicator):
    """
    TRIX - Triple Exponential Average - TradingView Pine Script v6 Implementation
    
    TRIX is a momentum oscillator that displays the rate of change 
    of a triple exponentially smoothed moving average of the logarithm of price.
    
    TradingView Formula (Pine Script v6):
    out = 10000 * ta.change(ta.ema(ta.ema(ta.ema(math.log(close), length), length), length))
    
    Default Parameters (TradingView):
    - length = 18
    """
    
    def __init__(self):
        super().__init__("TRIX")
    
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], length: int = 18) -> Union[np.ndarray, pd.Series]:
        """
        Calculate TRIX using TradingView Pine Script v6 formula
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        length : int, default=18
            Number of periods for EMA calculation (TradingView default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            TRIX values in the same format as input
            
        Notes:
        ------
        This implementation matches TradingView Pine Script v6:
        out = 10000 * ta.change(ta.ema(ta.ema(ta.ema(math.log(close), length), length), length))
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(length, len(validated_data))
        
        trix = _backend.trix(validated_data, length)
        return self.format_output(trix, input_type, index)


class UO(BaseIndicator):
    """
    Ultimate Oscillator
    
    The Ultimate Oscillator combines short, medium, and long-term price action 
    into one oscillator.
    
    Formula: UO = 100 × (4×AVG7 + 2×AVG14 + AVG28) / (4 + 2 + 1)
    Where: AVG = Average of (Close - TrueLow) / (TrueRange)
    """
    
    def __init__(self):
        super().__init__("UO")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_uo(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period1: int, period2: int, period3: int) -> np.ndarray:
        """Numba optimized Ultimate Oscillator calculation"""
        n = len(close)
        result = np.full(n, np.nan)
        
        # Calculate True Low and True Range
        true_low = np.empty(n)
        true_range = np.empty(n)
        buying_pressure = np.empty(n)
        
        true_low[0] = low[0]
        true_range[0] = high[0] - low[0]
        buying_pressure[0] = close[0] - true_low[0]
        
        for i in range(1, n):
            true_low[i] = min(low[i], close[i - 1])
            true_range[i] = max(high[i] - low[i], 
                               abs(high[i] - close[i - 1]), 
                               abs(low[i] - close[i - 1]))
            buying_pressure[i] = close[i] - true_low[i]
        
        # Calculate Ultimate Oscillator
        max_period = max(period1, period2, period3)
        for i in range(max_period - 1, n):
            # Calculate averages for each period
            bp1 = np.sum(buying_pressure[i - period1 + 1:i + 1])
            tr1 = np.sum(true_range[i - period1 + 1:i + 1])
            avg1 = bp1 / tr1 if tr1 > 0 else 0
            
            bp2 = np.sum(buying_pressure[i - period2 + 1:i + 1])
            tr2 = np.sum(true_range[i - period2 + 1:i + 1])
            avg2 = bp2 / tr2 if tr2 > 0 else 0
            
            bp3 = np.sum(buying_pressure[i - period3 + 1:i + 1])
            tr3 = np.sum(true_range[i - period3 + 1:i + 1])
            avg3 = bp3 / tr3 if tr3 > 0 else 0
            
            # Calculate Ultimate Oscillator
            result[i] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        
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
        
        result = _backend.uo(high_data, low_data, close_data, period1, period2, period3)
        return self.format_output(result, input_type, index)


class AO(BaseIndicator):
    """
    Awesome Oscillator
    
    The Awesome Oscillator is an indicator used to measure market momentum.
    
    Formula: AO = SMA(HL/2, 5) - SMA(HL/2, 34)
    Where: HL/2 = (High + Low) / 2
    """
    
    def __init__(self):
        super().__init__("AO")
    
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
                 fast_period: int = 5, slow_period: int = 34) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Awesome Oscillator
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        fast_period : int, default=5
            Fast SMA period
        slow_period : int, default=34
            Slow SMA period
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Awesome Oscillator values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        
        result = _backend.ao(high_data, low_data, fast_period, slow_period)
        return self.format_output(result, input_type, index)


class AC(BaseIndicator):
    """
    Accelerator Oscillator
    
    The Accelerator Oscillator measures acceleration and deceleration of momentum.
    
    Formula: AC = AO - SMA(AO, 5)
    Where: AO = Awesome Oscillator
    """
    
    def __init__(self):
        super().__init__("AC")
        self._ao = AO()
    
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
                 period: int = 5) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Accelerator Oscillator
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        period : int, default=5
            SMA period for acceleration calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Accelerator Oscillator values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        result_arr = _backend.ac(high_data, low_data, period)
        return self.format_output(result_arr, input_type, index)


class PPO(BaseIndicator):
    """
    Percentage Price Oscillator
    
    PPO is a momentum oscillator that measures the difference between two 
    moving averages as a percentage of the larger moving average.
    
    Formula: PPO = ((Fast EMA - Slow EMA) / Slow EMA) × 100
    """
    
    def __init__(self):
        super().__init__("PPO")
    
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
                 fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Percentage Price Oscillator
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        fast_period : int, default=12
            Fast EMA period
        slow_period : int, default=26
            Slow EMA period
        signal_period : int, default=9
            Signal line EMA period
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (ppo_line, signal_line, histogram) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        results = _backend.ppo(validated_data, fast_period, slow_period, signal_period)
        return self.format_multiple_outputs(results, input_type, index)


class PO(BaseIndicator):
    """
    Price Oscillator
    
    Price Oscillator shows the difference between two moving averages.
    
    Formula: PO = Fast MA - Slow MA
    """
    
    def __init__(self):
        super().__init__("PO")
    
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
                 fast_period: int = 10, slow_period: int = 20,
                 ma_type: str = "SMA") -> Union[np.ndarray, pd.Series]:
        """
        Calculate Price Oscillator
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        fast_period : int, default=10
            Fast moving average period
        slow_period : int, default=20
            Slow moving average period
        ma_type : str, default="SMA"
            Type of moving average ("SMA" or "EMA")
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Price Oscillator values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        result = _backend.price_oscillator(validated_data, fast_period, slow_period, ma_type)
        return self.format_output(result, input_type, index)


class DPO(BaseIndicator):
    """
    Detrended Price Oscillator - matches TradingView exactly
    
    DPO attempts to eliminate the trend in prices by comparing a current or past price 
    to a moving average. TradingView offers both centered and non-centered modes.
    
    Centered Mode: DPO = Close[barsback] - SMA(Close, period)
    Non-Centered Mode: DPO = Close - SMA(Close, period)[barsback]
    Where: barsback = period/2 + 1
    """
    
    def __init__(self):
        super().__init__("DPO")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            result[i] = np.mean(data[i - period + 1:i + 1])
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 period: int = 21, is_centered: bool = False) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Detrended Price Oscillator - matches TradingView exactly
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=21
            Length for SMA calculation (TradingView default)
        is_centered : bool, default=False
            When False (default): DPO = Close - SMA[barsback] (non-centered)
            When True: DPO = Close[barsback] - SMA (centered)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            DPO values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        dpo = _backend.dpo(validated_data, period, is_centered)
        return self.format_output(dpo, input_type, index)


class AROONOSC(BaseIndicator):
    """
    Aroon Oscillator
    
    The Aroon Oscillator is the difference between Aroon Up and Aroon Down.
    
    Formula: Aroon Oscillator = Aroon Up - Aroon Down
    """
    
    def __init__(self):
        super().__init__("Aroon Oscillator")
    
    @staticmethod
    # @jit(nopython=True)  # Disabled for consistency with Aroon fix
    def _calculate_aroon_osc(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """
        Aroon Oscillator calculation matching TradingView logic
        
        TradingView formula:
        upper = 100 * (highestbars(high, length+1) + length)/length
        lower = 100 * (lowestbars(low, length+1) + length)/length
        oscillator = upper - lower
        """
        n = len(high)
        result = np.full(n, np.nan)
        
        # TradingView uses period + 1 bars for lookback
        lookback = period + 1
        
        for i in range(lookback - 1, n):
            # Find highest high and lowest low positions in the window (period + 1 bars)
            window_start = i - lookback + 1
            window_end = i + 1
            high_window = high[window_start:window_end]
            low_window = low[window_start:window_end]
            
            # Find positions of highest high and lowest low
            # Use FIRST occurrence to match TradingView behavior
            highest_pos = 0
            lowest_pos = 0
            
            for j in range(len(high_window)):
                if high_window[j] > high_window[highest_pos]:  # Changed >= to > for first occurrence
                    highest_pos = j
                if low_window[j] < low_window[lowest_pos]:     # Changed <= to < for first occurrence
                    lowest_pos = j
            
            # Calculate bars since highest/lowest (0 = current bar)
            bars_since_high = len(high_window) - 1 - highest_pos
            bars_since_low = len(low_window) - 1 - lowest_pos
            
            # TradingView formula: 100 * (ta.highestbars + period) / period
            # ta.highestbars returns -bars_since_high, so:
            aroon_up = 100 * (period - bars_since_high) / period
            aroon_down = 100 * (period - bars_since_low) / period
            
            # Aroon Oscillator = Aroon Up - Aroon Down
            result[i] = aroon_up - aroon_down
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Aroon Oscillator
        
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
        Union[np.ndarray, pd.Series]
            Aroon Oscillator values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        self.validate_period(period, len(high_data))
        
        result = _backend.aroon_osc(high_data, low_data, period)
        return self.format_output(result, input_type, index)


class StochRSI(BaseIndicator):
    """
    Stochastic RSI
    
    The Stochastic RSI is an oscillator that uses RSI values instead of price values as inputs
    to the Stochastic formula.
    
    Formula: StochRSI = (RSI - Lowest(RSI, K)) / (Highest(RSI, K) - Lowest(RSI, K))
    """
    
    def __init__(self):
        super().__init__("StochRSI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < period + 1:
            return result
        
        # Calculate price changes
        deltas = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # Initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate RSI for remaining periods
        for i in range(period + 1, n):
            gain = gains[i - 1]
            loss = losses[i - 1]
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_stochrsi(data: np.ndarray, rsi_period: int, stoch_period: int, k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba optimized Stochastic RSI calculation"""
        # Calculate RSI inline
        n_data = len(data)
        rsi = np.full(n_data, np.nan)
        
        if n_data < rsi_period + 1:
            return rsi, rsi
        
        # Calculate price changes
        deltas = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:rsi_period])
        avg_loss = np.mean(losses[:rsi_period])
        
        # Calculate first RSI value
        if avg_loss == 0:
            rsi[rsi_period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[rsi_period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate subsequent RSI values using Wilder's smoothing
        for i in range(rsi_period, n_data - 1):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            
            avg_gain = (avg_gain * (rsi_period - 1) + gain) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + loss) / rsi_period
            
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        
        n = len(rsi)
        stoch_rsi = np.full(n, np.nan)
        
        # Calculate Stochastic RSI
        for i in range(stoch_period - 1, n):
            rsi_window = rsi[i - stoch_period + 1:i + 1]
            rsi_window_clean = rsi_window[~np.isnan(rsi_window)]
            
            if len(rsi_window_clean) > 0:
                rsi_high = np.max(rsi_window_clean)
                rsi_low = np.min(rsi_window_clean)
                
                if rsi_high != rsi_low:
                    stoch_rsi[i] = (rsi[i] - rsi_low) / (rsi_high - rsi_low) * 100
                else:
                    stoch_rsi[i] = 50.0
        
        # Calculate %K (SMA of StochRSI)
        k_values = np.full(n, np.nan)
        for i in range(k_period - 1, n):
            window = stoch_rsi[i - k_period + 1:i + 1]
            window_clean = window[~np.isnan(window)]
            if len(window_clean) > 0:
                k_values[i] = np.mean(window_clean)
        
        # Calculate %D (SMA of %K)
        d_values = np.full(n, np.nan)
        for i in range(d_period - 1, n):
            window = k_values[i - d_period + 1:i + 1]
            window_clean = window[~np.isnan(window)]
            if len(window_clean) > 0:
                d_values[i] = np.mean(window_clean)
        
        return k_values, d_values
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 rsi_period: int = 14, stoch_period: int = 14,
                 k_period: int = 3, d_period: int = 3) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Stochastic RSI
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        rsi_period : int, default=14
            Period for RSI calculation
        stoch_period : int, default=14
            Period for Stochastic calculation on RSI
        k_period : int, default=3
            Period for %K smoothing
        d_period : int, default=3
            Period for %D smoothing
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (%K, %D) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(rsi_period + stoch_period, len(validated_data))
        
        k_values, d_values = _backend.stochrsi(validated_data, rsi_period, stoch_period, k_period, d_period)

        results = (k_values, d_values)
        return self.format_multiple_outputs(results, input_type, index)


class RVI(BaseIndicator):
    """
    Relative Vigor Index (TradingView Pine Script Implementation)
    
    The RVI compares the closing price to the trading range and smooths the result
    using SWMA (Symmetrically Weighted Moving Average).
    
    Formula: 
    rvi = math.sum(ta.swma(close-open), len) / math.sum(ta.swma(high-low), len)
    sig = ta.swma(rvi)
    """
    
    def __init__(self):
        super().__init__("RVI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rvi(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba optimized RVI calculation (TradingView Pine Script formula)"""
        n = len(close)
        rvi = np.full(n, np.nan)
        signal = np.full(n, np.nan)
        
        # Calculate close-open and high-low differences
        close_open_diff = close - open_prices
        high_low_diff = high - low
        
        # Apply SWMA to the differences
        swma_close_open = _calculate_swma_helper(close_open_diff)
        swma_high_low = _calculate_swma_helper(high_low_diff)
        
        # Calculate RVI using TradingView formula: 
        # rvi = math.sum(ta.swma(close-open), len) / math.sum(ta.swma(high-low), len)
        for i in range(period + 2, n):  # +2 because SWMA needs 3 periods to start
            # Sum of SWMA values over the period
            numerator_sum = 0.0
            denominator_sum = 0.0
            
            for j in range(i - period + 1, i + 1):
                if not np.isnan(swma_close_open[j]):
                    numerator_sum += swma_close_open[j]
                if not np.isnan(swma_high_low[j]):
                    denominator_sum += swma_high_low[j]
            
            if denominator_sum != 0.0:
                rvi[i] = numerator_sum / denominator_sum
            else:
                rvi[i] = 0.0
        
        # Calculate signal line using SWMA of RVI (sig = ta.swma(rvi))
        signal = _calculate_swma_helper(rvi)
        
        return rvi, signal
    
    def calculate(self, open_prices: Union[np.ndarray, pd.Series, list],
                 high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 10) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Relative Vigor Index (TradingView Pine Script Implementation)
        
        Parameters:
        -----------
        open_prices : Union[np.ndarray, pd.Series, list]
            Opening prices
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=10
            Period for RVI calculation
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (rvi, signal) in the same format as input
            Formula: rvi = math.sum(ta.swma(close-open), len) / math.sum(ta.swma(high-low), len)
                    sig = ta.swma(rvi)
        """
        open_data, input_type, index = self.validate_input(open_prices)
        high_data, _, _ = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        open_data, high_data, low_data, close_data = self.align_arrays(open_data, high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        rvi, signal = self._calculate_rvi(open_data, high_data, low_data, close_data, period)
        
        results = (rvi, signal)
        return self.format_multiple_outputs(results, input_type, index)


class CHO(BaseIndicator):
    """
    Chaikin Oscillator (Chaikin A/D Oscillator)
    
    The Chaikin Oscillator is the difference between the 3-day and 10-day EMAs
    of the Accumulation Distribution Line.
    
    Formula: Chaikin Osc = EMA(A/D Line, 3) - EMA(A/D Line, 10)
    """
    
    def __init__(self):
        super().__init__("Chaikin Oscillator")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_adl(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Accumulation Distribution Line"""
        n = len(close)
        adl = np.zeros(n)
        
        for i in range(n):
            if high[i] != low[i]:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                clv = 0.0
            
            mfv = clv * volume[i]
            
            if i == 0:
                adl[i] = mfv
            else:
                adl[i] = adl[i - 1] + mfv
        
        return adl
    
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
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 volume: Union[np.ndarray, pd.Series, list],
                 fast_period: int = 3, slow_period: int = 10) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Chaikin Oscillator
        
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
        fast_period : int, default=3
            Fast EMA period
        slow_period : int, default=10
            Slow EMA period
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Chaikin Oscillator values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        volume_data, _, _ = self.validate_input(volume)
        
        high_data, low_data, close_data, volume_data = self.align_arrays(high_data, low_data, close_data, volume_data)
        
        result = _backend.cho(high_data, low_data, close_data, volume_data, fast_period, slow_period)
        return self.format_output(result, input_type, index)


class CHOP(BaseIndicator):
    """
    Choppiness Index
    
    The Choppiness Index measures whether the market is choppy (ranging) or trending.
    Values near 100 indicate a choppy market, while values near 0 indicate a trending market.
    
    Formula: CHOP = 100 * log10(sum(ATR, n) / (max(high, n) - min(low, n))) / log10(n)
    """
    
    def __init__(self):
        super().__init__("CHOP")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr_sum(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate sum of ATR over period"""
        n = len(close)
        atr_sum = np.full(n, np.nan)
        
        # Calculate True Range for each bar
        tr = np.full(n, np.nan)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i - 1]), 
                       abs(low[i] - close[i - 1]))
        
        # Calculate sum of ATR
        for i in range(period - 1, n):
            atr_sum[i] = np.sum(tr[i - period + 1:i + 1])
        
        return atr_sum
    
    @staticmethod
    def _calculate_chop(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """O(n) optimized CHOP calculation using utils"""
        # Use optimized O(n) utilities
        tr = true_range(high, low, close)
        atr_sum = rolling_sum(tr, period)
        highest_high = highest(high, period)
        lowest_low = lowest(low, period)
        
        # Calculate range
        range_val = highest_high - lowest_low
        
        # Calculate CHOP
        result = np.full(len(high), np.nan)
        valid_mask = (range_val > 0) & (atr_sum > 0) & ~np.isnan(range_val) & ~np.isnan(atr_sum)
        
        if period > 1:  # Avoid log10(1) = 0 division
            log_period = np.log10(period)
            result[valid_mask] = 100 * np.log10(atr_sum[valid_mask] / range_val[valid_mask]) / log_period
        
        # Set default middle value where calculation is invalid
        invalid_mask = ~valid_mask & ~np.isnan(atr_sum)
        result[invalid_mask] = 50.0
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Choppiness Index
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Period for CHOP calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            CHOP values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        result = _backend.chop(high_data, low_data, close_data, period)
        return self.format_output(result, input_type, index)


class KST(BaseIndicator):
    """
    Know Sure Thing (KST) - matches TradingView exactly
    
    KST is a momentum oscillator developed by Martin Pring based on the smoothed rate-of-change values.
    
    TradingView Formula:
    smaroc(roclen, smalen) => ta.sma(ta.roc(close, roclen), smalen)
    kst = smaroc(roclen1, smalen1) + 2 * smaroc(roclen2, smalen2) + 3 * smaroc(roclen3, smalen3) + 4 * smaroc(roclen4, smalen4)
    sig = ta.sma(kst, siglen)
    """
    
    def __init__(self):
        super().__init__("KST")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_roc(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rate of Change"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period, n):
            if data[i - period] != 0:
                result[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
            else:
                result[i] = 0.0
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            if not np.isnan(data[i]):
                window = data[i - period + 1:i + 1]
                valid_values = window[~np.isnan(window)]
                if len(valid_values) >= period:
                    result[i] = np.mean(valid_values[-period:])
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        n = len(data)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        first_valid = -1
        for i in range(n):
            if not np.isnan(data[i]):
                first_valid = i
                break
        
        if first_valid == -1:
            return result
        
        result[first_valid] = data[first_valid]
        
        for i in range(first_valid + 1, n):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 roclen1: int = 10, roclen2: int = 15, roclen3: int = 20, roclen4: int = 30,
                 smalen1: int = 10, smalen2: int = 10, smalen3: int = 10, smalen4: int = 15,
                 siglen: int = 9) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Know Sure Thing - matches TradingView exactly
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Close price data
        roclen1, roclen2, roclen3, roclen4 : int
            ROC Length periods (TradingView: roclen1, roclen2, roclen3, roclen4)
        smalen1, smalen2, smalen3, smalen4 : int
            SMA Length periods for smoothing ROC (TradingView: smalen1, smalen2, smalen3, smalen4)
        siglen : int, default=9
            Signal Line Length (TradingView: siglen)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (kst, signal_line) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        # Validate parameters
        for param, name in [(roclen1, "roclen1"), (roclen2, "roclen2"), (roclen3, "roclen3"), (roclen4, "roclen4"),
                           (smalen1, "smalen1"), (smalen2, "smalen2"), (smalen3, "smalen3"), (smalen4, "smalen4"),
                           (siglen, "siglen")]:
            if param <= 0:
                raise ValueError(f"{name} must be positive, got {param}")
        
        # Calculate ROCs using TradingView naming
        roc_1 = self._calculate_roc(validated_data, roclen1)
        roc_2 = self._calculate_roc(validated_data, roclen2)
        roc_3 = self._calculate_roc(validated_data, roclen3)
        roc_4 = self._calculate_roc(validated_data, roclen4)
        
        # Calculate smoothed ROCs (smaroc function)
        smaroc_1 = self._calculate_sma(roc_1, smalen1)
        smaroc_2 = self._calculate_sma(roc_2, smalen2)
        smaroc_3 = self._calculate_sma(roc_3, smalen3)
        smaroc_4 = self._calculate_sma(roc_4, smalen4)
        
        # Calculate KST using TradingView formula
        # kst = smaroc(roclen1, smalen1) + 2 * smaroc(roclen2, smalen2) + 3 * smaroc(roclen3, smalen3) + 4 * smaroc(roclen4, smalen4)
        kst = smaroc_1 * 1 + smaroc_2 * 2 + smaroc_3 * 3 + smaroc_4 * 4
        
        # Calculate signal line using SMA (not EMA) to match TradingView
        # sig = ta.sma(kst, siglen)
        signal_line = self._calculate_sma(kst, siglen)
        
        results = (kst, signal_line)
        return self.format_multiple_outputs(results, input_type, index)


class TSI(BaseIndicator):
    """
    True Strength Index (TSI)
    
    TSI is a momentum oscillator that uses moving averages of price changes.
    
    Formula: TSI = 100 * (Double Smoothed PC / Double Smoothed Absolute PC)
    Where PC = Price Change
    """
    
    def __init__(self):
        super().__init__("TSI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        n = len(data)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        first_valid = -1
        for i in range(n):
            if not np.isnan(data[i]):
                first_valid = i
                break
        
        if first_valid == -1:
            return result
        
        result[first_valid] = data[first_valid]
        
        for i in range(first_valid + 1, n):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 long: int = 25, short: int = 13, signal: int = 13) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate True Strength Index
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        long : int, default=25
            Long period for first smoothing
        short : int, default=13
            Short period for second smoothing
        signal : int, default=13
            Signal line period
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (tsi, signal_line) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        # Calculate price changes
        price_changes = np.diff(validated_data)
        price_changes = np.concatenate([np.array([0.0]), price_changes])
        
        # Calculate absolute price changes
        abs_price_changes = np.abs(price_changes)
        
        # First smoothing
        pc_smooth1 = self._calculate_ema(price_changes, long)
        apc_smooth1 = self._calculate_ema(abs_price_changes, long)
        
        # Second smoothing
        pc_smooth2 = self._calculate_ema(pc_smooth1, short)
        apc_smooth2 = self._calculate_ema(apc_smooth1, short)
        
        # Calculate TSI
        tsi = np.full_like(validated_data, np.nan)
        for i in range(len(validated_data)):
            if apc_smooth2[i] != 0:
                tsi[i] = 100 * (pc_smooth2[i] / apc_smooth2[i])
            else:
                tsi[i] = 0.0
        
        # Calculate signal line
        signal_line = self._calculate_ema(tsi, signal)
        
        results = (tsi, signal_line)
        return self.format_multiple_outputs(results, input_type, index)


class VI(BaseIndicator):
    """
    Vortex Indicator (VI+ and VI-) - TradingView Pine Script v6 Implementation
    
    The Vortex Indicator identifies the start of a new trend or the continuation of an existing trend.
    
    TradingView Pine Script v6 Formula:
    VMP = math.sum( math.abs( high - low[1]), period_ )
    VMM = math.sum( math.abs( low - high[1]), period_ )
    STR = math.sum( ta.atr(1), period_ )
    VIP = VMP / STR
    VIM = VMM / STR
    
    Where:
    - VMP (Vortex Movement Positive) = Sum of |high - low[1]|
    - VMM (Vortex Movement Minus) = Sum of |low - high[1]|
    - STR = Sum of ATR(1) over period
    - VIP = VI+ (Positive Vortex Indicator)
    - VIM = VI- (Minus Vortex Indicator)
    """
    
    def __init__(self):
        super().__init__("VI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr_single(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate single-period ATR (True Range)"""
        n = len(close)
        atr = np.full(n, np.nan)
        
        atr[0] = high[0] - low[0]  # First value
        
        for i in range(1, n):
            tr = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
            atr[i] = tr
        
        return atr
    
    @staticmethod
    def _calculate_vi_tv_optimized(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized TradingView Pine Script v6 Vortex Indicator calculation using O(N) rolling sums"""
        n = len(close)
        vi_plus = np.full(n, np.nan)
        vi_minus = np.full(n, np.nan)
        
        # Pre-calculate VMP, VMM, and ATR arrays
        vmp_values = np.full(n, np.nan)
        vmm_values = np.full(n, np.nan)
        atr_single = np.full(n, np.nan)
        
        # First period calculations
        atr_single[0] = high[0] - low[0]
        
        for i in range(1, n):
            # VMP = abs(high - low[1])
            vmp_values[i] = abs(high[i] - low[i - 1])
            
            # VMM = abs(low - high[1]) 
            vmm_values[i] = abs(low[i] - high[i - 1])
            
            # ATR(1) = True Range
            tr = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
            atr_single[i] = tr
        
        # Use O(N) rolling sums for VMP, VMM, and STR
        vmp_rolling = rolling_sum(vmp_values, period)
        vmm_rolling = rolling_sum(vmm_values, period)
        str_rolling = rolling_sum(atr_single, period)
        
        # Calculate VI+ and VI- using the rolling sums
        for i in range(period, n):
            if str_rolling[i] > 0 and not np.isnan(str_rolling[i]):
                vi_plus[i] = vmp_rolling[i] / str_rolling[i]  # VIP = VMP / STR
                vi_minus[i] = vmm_rolling[i] / str_rolling[i]  # VIM = VMM / STR
            else:
                vi_plus[i] = 0.0
                vi_minus[i] = 0.0
        
        return vi_plus, vi_minus
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Vortex Indicator - TradingView Pine Script v6 Implementation
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Period for VI calculation (TradingView default)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (vi_plus, vi_minus) in the same format as input
            
        Notes:
        ------
        TradingView Pine Script v6 Formula:
        VMP = math.sum( math.abs( high - low[1]), period_ )
        VMM = math.sum( math.abs( low - high[1]), period_ )
        STR = math.sum( ta.atr(1), period_ )
        VIP = VMP / STR
        VIM = VMM / STR
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period + 1, len(close_data))
        
        vi_plus, vi_minus = self._calculate_vi_tv_optimized(high_data, low_data, close_data, period)
        
        results = (vi_plus, vi_minus)
        return self.format_multiple_outputs(results, input_type, index)


class GatorOscillator(BaseIndicator):
    """
    Gator Oscillator (Bill Williams) - matches TradingView exactly
    
    The Gator Oscillator shows the convergence/divergence of the Alligator lines.
    Based on the Bill Williams Alligator indicator with offset.
    
    TradingView Formula:
    jaw = offset(rma(hl2, 13), 8)
    teeth = offset(rma(hl2, 8), 5)  
    lips = offset(rma(hl2, 5), 3)
    upper = abs(jaw - teeth)
    lower = -abs(teeth - lips)
    """
    
    def __init__(self):
        super().__init__("Gator Oscillator")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rma(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate RMA (Running Moving Average) - TradingView's rma() function
        This is equivalent to SMMA/SMMA/EMA with alpha = 1/period
        """
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < period:
            return result
        
        # Initialize first value as SMA
        result[period - 1] = np.mean(data[:period])
        
        # Calculate RMA: rma[i] = (rma[i-1] * (period-1) + data[i]) / period
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _apply_offset(data: np.ndarray, offset: int) -> np.ndarray:
        """
        Apply TradingView offset function - shifts data forward by offset periods
        TradingView offset(series, offset) shifts the series forward
        """
        n = len(data)
        result = np.full(n, np.nan)
        
        # Shift forward by offset periods
        for i in range(offset, n):
            result[i] = data[i - offset]
        
        return result
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 jaw_period: int = 13, teeth_period: int = 8, lips_period: int = 5) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Gator Oscillator - matches TradingView exactly
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        jaw_period : int, default=13
            Period for Jaw line (TradingView default)
        teeth_period : int, default=8
            Period for Teeth line (TradingView default)
        lips_period : int, default=5
            Period for Lips line (TradingView default)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (upper_histogram, lower_histogram) in the same format as input
            upper_histogram = abs(jaw - teeth)
            lower_histogram = -abs(teeth - lips)
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        
        # Calculate hl2 (TradingView's hl2)
        hl2 = (high_data + low_data) / 2.0
        
        # Calculate RMA for each Alligator line
        jaw_rma = self._calculate_rma(hl2, jaw_period)
        teeth_rma = self._calculate_rma(hl2, teeth_period)
        lips_rma = self._calculate_rma(hl2, lips_period)
        
        # Apply TradingView offsets
        jaw = self._apply_offset(jaw_rma, 8)      # offset(rma(hl2, 13), 8)
        teeth = self._apply_offset(teeth_rma, 5)  # offset(rma(hl2, 8), 5)
        lips = self._apply_offset(lips_rma, 3)    # offset(rma(hl2, 5), 3)
        
        # Calculate Gator Oscillator histograms
        upper_histogram = np.abs(jaw - teeth)     # abs(jaw - teeth)
        lower_histogram = -np.abs(teeth - lips)   # -abs(teeth - lips)
        
        results = (upper_histogram, lower_histogram)
        return self.format_multiple_outputs(results, input_type, index)


class STC(BaseIndicator):
    """
    Schaff Trend Cycle (STC) - TradingView Pine Script v4 Implementation
    
    STC is a cyclical oscillator that combines slow stochastics and the MACD.
    Based on TradingView Pine Script v4 implementation by Alex Orekhov (everget).
    
    TradingView Formula:
    macd = ema(src, fastLength) - ema(src, slowLength)
    k = nz(stoch(macd, macd, macd, cycleLength))
    d = ema(k, d1Length)
    kd = nz(stoch(d, d, d, cycleLength))
    stc = ema(kd, d2Length)
    stc := max(min(stc, 100), 0)
    
    Default Parameters (TradingView):
    - fastLength = 23 (MACD Fast Length)
    - slowLength = 50 (MACD Slow Length)
    - cycleLength = 10 (Cycle Length)
    - d1Length = 3 (1st %D Length)
    - d2Length = 3 (2nd %D Length)
    """
    
    def __init__(self):
        super().__init__("STC")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_stochastic(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate stochastic oscillator exactly like TradingView stoch() function"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            # Find highest and lowest values in the period
            window_start = i - period + 1
            highest = -np.inf
            lowest = np.inf
            
            # Calculate highest and lowest in the window
            for j in range(window_start, i + 1):
                if not np.isnan(data[j]):
                    if data[j] > highest:
                        highest = data[j]
                    if data[j] < lowest:
                        lowest = data[j]
            
            # Calculate stochastic value
            if highest != lowest and not np.isnan(data[i]):
                result[i] = ((data[i] - lowest) / (highest - lowest)) * 100.0
            elif highest == lowest:
                result[i] = 50.0  # When no range, default to middle
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 fast_length: int = 23, slow_length: int = 50, cycle_length: int = 10, 
                 d1_length: int = 3, d2_length: int = 3) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Schaff Trend Cycle using TradingView Pine Script v4 formula
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        fast_length : int, default=23
            MACD Fast Length (TradingView fastLength)
        slow_length : int, default=50
            MACD Slow Length (TradingView slowLength)
        cycle_length : int, default=10
            Cycle Length for stochastic calculations (TradingView cycleLength)
        d1_length : int, default=3
            1st %D Length - EMA smoothing period (TradingView d1Length)
        d2_length : int, default=3
            2nd %D Length - EMA smoothing period (TradingView d2Length)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            STC values in the same format as input (0-100 range)
            
        Notes:
        ------
        This implementation matches TradingView Pine Script v4:
        macd = ema(src, fastLength) - ema(src, slowLength)
        k = nz(stoch(macd, macd, macd, cycleLength))
        d = ema(k, d1Length)
        kd = nz(stoch(d, d, d, cycleLength))
        stc = ema(kd, d2Length)
        stc := max(min(stc, 100), 0)
        """
        validated_data, input_type, index = self.validate_input(data)
        
        # Step 1: Calculate MACD
        # macd = ema(src, fastLength) - ema(src, slowLength)
        fast_ema = ema(validated_data, fast_length)
        slow_ema = ema(validated_data, slow_length)
        macd = fast_ema - slow_ema
        
        # Step 2: First stochastic calculation on MACD
        # k = nz(stoch(macd, macd, macd, cycleLength))
        k = self._calculate_stochastic(macd, cycle_length)
        # Handle nz() - replace NaN with 0
        k = np.where(np.isnan(k), 0.0, k)
        
        # Step 3: Smooth with EMA
        # d = ema(k, d1Length)
        d = ema(k, d1_length)
        
        # Step 4: Second stochastic calculation
        # kd = nz(stoch(d, d, d, cycleLength))
        kd = self._calculate_stochastic(d, cycle_length)
        # Handle nz() - replace NaN with 0
        kd = np.where(np.isnan(kd), 0.0, kd)
        
        # Step 5: Final EMA smoothing
        # stc = ema(kd, d2Length)
        stc = ema(kd, d2_length)
        
        # Step 6: Clamp between 0 and 100
        # stc := max(min(stc, 100), 0)
        stc = np.clip(stc, 0.0, 100.0)
        
        return self.format_output(stc, input_type, index)


class Coppock(BaseIndicator):
    """
    Coppock Curve - TradingView Pine Script v6 Implementation
    
    The Coppock Curve is a long-term momentum indicator primarily used for major stock indices.
    It's calculated as a Weighted Moving Average of the sum of Rate of Change values over two periods.
    
    Formula: 
    curve = ta.wma(ta.roc(source, longRoCLength) + ta.roc(source, shortRoCLength), wmaLength)
    
    Default TradingView Parameters:
    - WMA Length: 10
    - Long RoC Length: 14  
    - Short RoC Length: 11
    """
    
    def __init__(self):
        super().__init__("Coppock")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_roc_for_coppock(data: np.ndarray, period: int) -> np.ndarray:
        """Rate of Change calculation for Coppock (Numba optimized)"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period, n):
            if data[i - period] != 0:
                result[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_wma_for_coppock(data: np.ndarray, period: int) -> np.ndarray:
        """Weighted Moving Average calculation for Coppock (Numba optimized)"""
        n = len(data)
        result = np.full(n, np.nan)
        
        # Calculate weight sum for normalization
        weight_sum = 0.0
        for i in range(1, period + 1):
            weight_sum += i
        
        for i in range(period - 1, n):
            if not np.isnan(data[i]):
                weighted_sum = 0.0
                valid_weights = 0.0
                
                # Calculate weighted sum
                for j in range(period):
                    idx = i - period + 1 + j
                    if not np.isnan(data[idx]):
                        weight = j + 1  # Weights: 1, 2, 3, ..., period
                        weighted_sum += data[idx] * weight
                        valid_weights += weight
                
                if valid_weights > 0:
                    result[i] = weighted_sum / valid_weights
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_coppock(data: np.ndarray, wma_length: int, 
                          long_roc_length: int, short_roc_length: int) -> np.ndarray:
        """Coppock Curve calculation (Numba optimized)"""
        n = len(data)
        
        # Calculate ROC for both periods - inline the ROC calculation
        long_roc = np.full(n, np.nan)
        for i in range(long_roc_length, n):
            if data[i - long_roc_length] != 0:
                long_roc[i] = ((data[i] - data[i - long_roc_length]) / data[i - long_roc_length]) * 100
        
        short_roc = np.full(n, np.nan)
        for i in range(short_roc_length, n):
            if data[i - short_roc_length] != 0:
                short_roc[i] = ((data[i] - data[i - short_roc_length]) / data[i - short_roc_length]) * 100
        
        # Sum the ROC values
        roc_sum = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(long_roc[i]) and not np.isnan(short_roc[i]):
                roc_sum[i] = long_roc[i] + short_roc[i]
        
        # Apply WMA to the sum - inline the WMA calculation
        result = np.full(n, np.nan)
        
        for i in range(wma_length - 1, n):
            if not np.isnan(roc_sum[i]):
                weighted_sum = 0.0
                valid_weights = 0.0
                
                # Calculate weighted sum
                for j in range(wma_length):
                    idx = i - wma_length + 1 + j
                    if not np.isnan(roc_sum[idx]):
                        weight = j + 1  # Weights: 1, 2, 3, ..., wma_length
                        weighted_sum += roc_sum[idx] * weight
                        valid_weights += weight
                
                if valid_weights > 0:
                    result[i] = weighted_sum / valid_weights
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                  wma_length: int = 10, long_roc_length: int = 14, 
                  short_roc_length: int = 11) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Coppock Curve
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        wma_length : int, default=10
            WMA Length for final smoothing (TradingView default)
        long_roc_length : int, default=14
            Long RoC Length (TradingView default)  
        short_roc_length : int, default=11
            Short RoC Length (TradingView default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Coppock Curve values in the same format as input
            
        Notes:
        ------
        TradingView Pine Script v6 Formula:
        curve = ta.wma(ta.roc(source, longRoCLength) + ta.roc(source, shortRoCLength), wmaLength)
        
        Interpretation:
        - Values above zero indicate bullish momentum
        - Values below zero indicate bearish momentum
        - Zero crossovers provide buy/sell signals
        - Primarily used for long-term trend analysis
        """
        validated_data, input_type, index = self.validate_input(data)
        
        result = self._calculate_coppock(validated_data, wma_length, 
                                       long_roc_length, short_roc_length)
        
        return self.format_output(result, input_type, index)
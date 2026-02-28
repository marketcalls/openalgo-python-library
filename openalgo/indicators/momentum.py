# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Momentum Indicators
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .utils import ema, highest, lowest, sma, rolling_sum


@jit(nopython=True, cache=True)
def _ema_for_macd(data: np.ndarray, period: int) -> np.ndarray:
    """Specialized EMA for MACD - Numba compatible"""
    n = len(data)
    result = np.empty(n)
    alpha = 2.0 / (period + 1)
    
    # Initialize with first value for MACD compatibility
    result[0] = data[0]
    
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result


class RSI(BaseIndicator):
    """
    Relative Strength Index
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    It oscillates between 0 and 100, with readings above 70 indicating overbought conditions
    and readings below 30 indicating oversold conditions.
    
    Formula: RSI = 100 - (100 / (1 + RS))
    Where: RS = Average Gain / Average Loss
    """
    
    def __init__(self):
        super().__init__("RSI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized RSI calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < period + 1:
            return result
        
        # Calculate price changes
        deltas = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate first RSI value
        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate subsequent RSI values using Wilder's smoothing
        for i in range(period, n - 1):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Relative Strength Index
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=14
            Number of periods for RSI calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            RSI values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        result = self._calculate_rsi(validated_data, period)
        return self.format_output(result, input_type, index)


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence
    
    MACD is a trend-following momentum indicator that shows the relationship between
    two exponential moving averages of prices.
    
    Components:
    - MACD Line: 12-day EMA - 26-day EMA
    - Signal Line: 9-day EMA of MACD Line
    - MACD Histogram: MACD Line - Signal Line
    """
    
    def __init__(self):
        super().__init__("MACD")
    
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_macd(data: np.ndarray, fast_period: int, slow_period: int, 
                       signal_period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba optimized MACD calculation"""
        # Calculate EMAs
        ema_fast = _ema_for_macd(data, fast_period)
        ema_slow = _ema_for_macd(data, slow_period)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = _ema_for_macd(macd_line, signal_period)
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate MACD
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        fast_period : int, default=12
            Period for fast EMA
        slow_period : int, default=26
            Period for slow EMA
        signal_period : int, default=9
            Period for signal line EMA
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series, pd.Series]]
            (macd_line, signal_line, histogram) in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        
        # Validate periods
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("All periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        results = self._calculate_macd(validated_data, fast_period, slow_period, signal_period)
        return self.format_multiple_outputs(results, input_type, index)


class Stochastic(BaseIndicator):
    """
    Stochastic Oscillator
    
    The Stochastic Oscillator compares a security's closing price to its price range
    over a given time period. It consists of two lines: %K and %D.
    
    Formula:
    %K = 100 × (Current Close - Lowest Low) / (Highest High - Lowest Low)
    %D = 3-period SMA of %K
    """
    
    def __init__(self):
        super().__init__("Stochastic")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_stochastic(close: np.ndarray,
                             k_period: int, smooth_k: int, d_period: int,
                             hh: np.ndarray, ll: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Numba optimized Stochastic using pre-computed highest/lowest.

        Returns (slow_k, slow_d) where:
          fast_k  = raw stochastic
          slow_k  = SMA(fast_k, smooth_k)   ← returned as %K
          slow_d  = SMA(slow_k, d_period)    ← returned as %D

        Uses index arithmetic to avoid NaN checks (fastmath-safe).
        """
        n = len(close)
        fast_k = np.full(n, np.nan)

        fk_start = k_period - 1  # first valid fast_k index

        # Fast %K from pre-computed highest high / lowest low
        for i in range(fk_start, n):
            if hh[i] != ll[i]:
                fast_k[i] = 100.0 * (close[i] - ll[i]) / (hh[i] - ll[i])
            else:
                fast_k[i] = 50.0

        # Slow %K = SMA(fast_k, smooth_k)
        slow_k = np.full(n, np.nan)
        sk_start = fk_start + smooth_k - 1  # first valid slow_k index
        if sk_start < n:
            sk_sum = 0.0
            for j in range(fk_start, fk_start + smooth_k):
                sk_sum += fast_k[j]
            slow_k[sk_start] = sk_sum / smooth_k
            for i in range(sk_start + 1, n):
                sk_sum += fast_k[i] - fast_k[i - smooth_k]
                slow_k[i] = sk_sum / smooth_k

        # Slow %D = SMA(slow_k, d_period)
        slow_d = np.full(n, np.nan)
        sd_start = sk_start + d_period - 1  # first valid slow_d index
        if sd_start < n:
            sd_sum = 0.0
            for j in range(sk_start, sk_start + d_period):
                sd_sum += slow_k[j]
            slow_d[sd_start] = sd_sum / d_period
            for i in range(sd_start + 1, n):
                sd_sum += slow_k[i] - slow_k[i - d_period]
                slow_d[i] = sd_sum / d_period

        return slow_k, slow_d

    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 k_period: int = 14, smooth_k: int = 3,
                 d_period: int = 3) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Stochastic Oscillator

        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        k_period : int, default=14
            Lookback period for raw (fast) %K
        smooth_k : int, default=3
            Smoothing period for %K (SMA of fast %K)
        d_period : int, default=3
            Period for %D calculation (SMA of smoothed %K)

        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (k_percent, d_percent) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)

        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)

        # Validate periods
        self.validate_period(k_period, len(close_data))
        if d_period <= 0:
            raise ValueError(f"d_period must be positive, got {d_period}")
        if smooth_k <= 0:
            raise ValueError(f"smooth_k must be positive, got {smooth_k}")

        # Pre-compute O(n) highest/lowest
        hh = highest(high_data, k_period)
        ll = lowest(low_data, k_period)

        results = self._calculate_stochastic(close_data, k_period, smooth_k, d_period, hh, ll)
        return self.format_multiple_outputs(results, input_type, index)


class CCI(BaseIndicator):
    """
    Commodity Channel Index
    
    CCI measures the current price level relative to an average price level over a given period.
    It is used to identify cyclical trends in commodities, equities, and currencies.
    
    Formula: CCI = (Typical Price - SMA of TP) / (0.015 × Mean Deviation)
    Where: Typical Price = (High + Low + Close) / 3
    """
    
    def __init__(self):
        super().__init__("CCI")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      period: int) -> np.ndarray:
        """Numba optimized CCI with O(n) rolling sum for SMA"""
        n = len(close)
        cci = np.full(n, np.nan)

        # Typical Price
        tp = (high + low + close) / 3.0

        # Rolling sum for SMA of TP
        rsum = 0.0
        for i in range(period):
            rsum += tp[i]

        for i in range(period - 1, n):
            if i > period - 1:
                rsum = rsum + tp[i] - tp[i - period]
            sma_tp = rsum / period

            # Mean deviation still O(period) per bar — no shortcut exists
            mean_dev = 0.0
            for j in range(period):
                mean_dev += abs(tp[i - period + 1 + j] - sma_tp)
            mean_dev /= period

            if mean_dev != 0.0:
                cci[i] = (tp[i] - sma_tp) / (0.015 * mean_dev)
            else:
                cci[i] = 0.0

        return cci
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 20) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Commodity Channel Index
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=20
            Number of periods for CCI calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            CCI values in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))
        
        result = self._calculate_cci(high_data, low_data, close_data, period)
        return self.format_output(result, input_type, index)


class WilliamsR(BaseIndicator):
    """
    Williams %R
    
    Williams %R is a momentum indicator that measures overbought and oversold levels.
    It is similar to the Stochastic Oscillator but is plotted on a negative scale from 0 to -100.
    
    Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
    """
    
    def __init__(self):
        super().__init__("Williams %R")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_williams_r(close: np.ndarray, period: int,
                             hh: np.ndarray, ll: np.ndarray) -> np.ndarray:
        """Numba optimized Williams %R using pre-computed highest/lowest"""
        n = len(close)
        williams_r = np.full(n, np.nan)

        for i in range(period - 1, n):
            if hh[i] != ll[i]:
                williams_r[i] = -100.0 * (hh[i] - close[i]) / (hh[i] - ll[i])
            else:
                williams_r[i] = -50.0

        return williams_r
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Williams %R
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=14
            Number of periods for Williams %R calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Williams %R values (range: 0 to -100) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)

        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        self.validate_period(period, len(close_data))

        # Pre-compute O(n) highest/lowest
        hh = highest(high_data, period)
        ll = lowest(low_data, period)

        result = self._calculate_williams_r(close_data, period, hh, ll)
        return self.format_output(result, input_type, index)


class BOP(BaseIndicator):
    """
    Balance of Power (BOP)
    
    Balance of Power measures the strength of buyers versus sellers by assessing
    the ability of each side to drive prices to an extreme level.
    
    Formula: BOP = (Close - Open) / (High - Low)
    """
    
    def __init__(self):
        super().__init__("BOP")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_bop(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Numba optimized BOP calculation"""
        n = len(close)
        bop = np.full(n, np.nan)
        
        for i in range(n):
            if high[i] != low[i]:
                bop[i] = (close[i] - open_prices[i]) / (high[i] - low[i])
            else:
                bop[i] = 0.0
        
        return bop
    
    def calculate(self, open_prices: Union[np.ndarray, pd.Series, list],
                 high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Balance of Power
        
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
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            BOP values in the same format as input
        """
        open_data, input_type, index = self.validate_input(open_prices)
        high_data, _, _ = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        open_data, high_data, low_data, close_data = self.align_arrays(open_data, high_data, low_data, close_data)
        
        result = self._calculate_bop(open_data, high_data, low_data, close_data)
        return self.format_output(result, input_type, index)


class ElderRay(BaseIndicator):
    """
    Elder Ray Index (Bull/Bear Power)
    
    Elder Ray Index consists of two indicators:
    - Bull Power = High - EMA
    - Bear Power = Low - EMA
    
    They measure the ability of bulls and bears to drive prices above or below an EMA.
    """
    
    def __init__(self):
        super().__init__("Elder Ray")
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 close: Union[np.ndarray, pd.Series, list],
                 period: int = 13) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Elder Ray Index

        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Closing prices
        period : int, default=13
            Period for EMA calculation

        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (bull_power, bear_power) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)

        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)

        ema_close = ema(close_data, period)

        bull_power = high_data - ema_close
        bear_power = low_data - ema_close

        results = (bull_power, bear_power)
        return self.format_multiple_outputs(results, input_type, index)


class Fisher(BaseIndicator):
    """
    Fisher Transform - matches TradingView exactly
    
    The Fisher Transform converts prices into a Gaussian normal distribution.
    TradingView version uses recursive smoothing for both value calculation 
    and the Fisher transform itself.
    
    TradingView Formula:
    value := round_(.66 * ((hl2 - low_) / (high_ - low_) - .5) + .67 * nz(value[1]))
    fish1 := .5 * math.log((1 + value) / (1 - value)) + .5 * nz(fish1[1])
    fish2 = fish1[1]
    """
    
    def __init__(self):
        super().__init__("Fisher Transform")
    
    @staticmethod
    @jit(nopython=True)
    def _round_value(val: float) -> float:
        """TradingView round_ function: constrain value to avoid log division issues"""
        if val > 0.99:
            return 0.999
        elif val < -0.99:
            return -0.999
        else:
            return val
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_fisher_tv(data: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Fisher Transform - matches TradingView exactly"""
        n = len(data)
        fish1 = np.full(n, np.nan)
        fish2 = np.full(n, np.nan)
        
        # Initialize recursive variables
        value = 0.0
        fish1_prev = 0.0
        
        for i in range(length - 1, n):
            # Calculate highest and lowest over length period
            # TradingView: high_ = ta.highest(hl2, len), low_ = ta.lowest(hl2, len)
            window_start = i - length + 1
            window = data[window_start:i + 1]
            high_ = np.max(window)
            low_ = np.min(window)
            
            if high_ != low_:
                # TradingView: value := round_(.66 * ((hl2 - low_) / (high_ - low_) - .5) + .67 * nz(value[1]))
                normalized = (data[i] - low_) / (high_ - low_) - 0.5
                new_value = 0.66 * normalized + 0.67 * value
                
                # Apply round_ function inline (TradingView constraint)
                if new_value > 0.99:
                    value = 0.999
                elif new_value < -0.99:
                    value = -0.999
                else:
                    value = new_value
                
                # TradingView: fish1 := .5 * math.log((1 + value) / (1 - value)) + .5 * nz(fish1[1])
                log_term = 0.5 * np.log((1 + value) / (1 - value))
                fish1[i] = log_term + 0.5 * fish1_prev
                
                # Update previous value for next iteration
                fish1_prev = fish1[i]
            else:
                # Handle case where high == low
                fish1[i] = fish1_prev
            
            # TradingView: fish2 = fish1[1] (previous Fisher value)
            if i > length - 1:
                fish2[i] = fish1[i - 1]
            else:
                fish2[i] = 0.0
        
        return fish1, fish2
    
    def calculate(self, high: Union[np.ndarray, pd.Series, list],
                 low: Union[np.ndarray, pd.Series, list],
                 length: int = 9) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]:
        """
        Calculate Fisher Transform - matches TradingView exactly
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices  
        length : int, default=9
            Length for highest/lowest calculation (TradingView default)
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.Series]]
            (fisher, trigger) in the same format as input
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        
        high_data, low_data = self.align_arrays(high_data, low_data)
        self.validate_period(length, len(high_data))
        
        # Calculate HL2 (typical price) as TradingView uses
        hl2 = (high_data + low_data) / 2.0
        
        fish1, fish2 = self._calculate_fisher_tv(hl2, length)
        
        results = (fish1, fish2)
        return self.format_multiple_outputs(results, input_type, index)


class CRSI(BaseIndicator):
    """
    Connors RSI (CRSI) - matches TradingView exactly
    
    Connors RSI is a composite momentum oscillator consisting of three components:
    1. RSI of price (ta.rsi(src, lenrsi))
    2. RSI of updown streak (ta.rsi(updown(src), lenupdown))
    3. Percent rank of 1-period ROC (ta.percentrank(ta.roc(src, 1), lenroc))
    
    Formula: CRSI = math.avg(rsi, updownrsi, percentrank)
    Default parameters: lenrsi=3, lenupdown=2, lenroc=100
    """
    
    def __init__(self):
        super().__init__("Connors RSI")

    @staticmethod
    @jit(nopython=True)
    def _calculate_updown_streak(data: np.ndarray) -> np.ndarray:
        """Calculate updown streak - matches TradingView Pine Script logic"""
        n = len(data)
        streak = np.zeros(n)
        
        for i in range(1, n):
            is_equal = data[i] == data[i-1]
            is_growing = data[i] > data[i-1]
            
            if is_equal:
                streak[i] = 0.0
            elif is_growing:
                # Growing: if previous was <= 0, start at 1, else increment
                if streak[i-1] <= 0:
                    streak[i] = 1.0
                else:
                    streak[i] = streak[i-1] + 1.0
            else:
                # Declining: if previous was >= 0, start at -1, else decrement
                if streak[i-1] >= 0:
                    streak[i] = -1.0
                else:
                    streak[i] = streak[i-1] - 1.0
        
        return streak
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_percent_rank(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate percent rank"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            current_val = data[i]
            
            count_below = 0
            for j in range(len(window)):
                if window[j] < current_val:
                    count_below += 1
            
            result[i] = (count_below / period) * 100
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list],
                 lenrsi: int = 3, lenupdown: int = 2, 
                 lenroc: int = 100) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Connors RSI - matches TradingView exactly
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        lenrsi : int, default=3
            RSI Length (period for price RSI)
        lenupdown : int, default=2
            UpDown Length (period for streak RSI)
        lenroc : int, default=100
            ROC Length (period for ROC percent rank)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Connors RSI values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(max(lenrsi, lenupdown, lenroc), len(validated_data))
        
        # Component 1: RSI of price (ta.rsi(src, lenrsi))
        price_rsi = RSI._calculate_rsi(validated_data, lenrsi)

        # Component 2: RSI of updown streak (ta.rsi(updown(src), lenupdown))
        updown_streak = self._calculate_updown_streak(validated_data)
        streak_rsi = RSI._calculate_rsi(updown_streak, lenupdown)
        
        # Component 3: Percent rank of 1-period ROC (ta.percentrank(ta.roc(src, 1), lenroc))
        # TradingView: ta.roc(src, 1) calculates 1-period rate of change
        roc_1period = np.full_like(validated_data, np.nan)
        for i in range(1, len(validated_data)):
            if validated_data[i - 1] != 0:
                roc_1period[i] = ((validated_data[i] - validated_data[i - 1]) / validated_data[i - 1]) * 100
        
        # Then calculate percent rank of this ROC over lenroc period
        roc_percentrank = self._calculate_percent_rank(roc_1period, lenroc)
        
        # Calculate Connors RSI: math.avg(rsi, updownrsi, percentrank)
        crsi = np.full_like(validated_data, np.nan)
        for i in range(len(validated_data)):
            if not np.isnan(price_rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(roc_percentrank[i]):
                crsi[i] = (price_rsi[i] + streak_rsi[i] + roc_percentrank[i]) / 3.0
        
        return self.format_output(crsi, input_type, index)
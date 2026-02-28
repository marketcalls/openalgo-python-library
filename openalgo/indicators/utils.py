# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Utility Functions
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit, njit, prange
from typing import Union, Optional


# ------------------------------------------------------------------
# Core helper – ensure every indicator receives a contiguous float64
# ------------------------------------------------------------------

def validate_input(arr: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Return C-contiguous float64 numpy array (zero-copy when possible)."""
    arr = np.asarray(arr, dtype=np.float64)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr


@njit(cache=True, nogil=True)
def crossover(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Check if series1 crosses over series2
    
    Parameters:
    -----------
    series1 : np.ndarray
        First series
    series2 : np.ndarray
        Second series
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating crossover points
    """
    n = len(series1)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(1, n):
        if (not np.isnan(series1[i]) and not np.isnan(series2[i]) and
            not np.isnan(series1[i-1]) and not np.isnan(series2[i-1]) and
            series1[i] > series2[i] and series1[i-1] <= series2[i-1]):
            result[i] = True

    return result


@njit(cache=True, nogil=True)
def crossunder(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Check if series1 crosses under series2
    
    Parameters:
    -----------
    series1 : np.ndarray
        First series
    series2 : np.ndarray
        Second series
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating crossunder points
    """
    n = len(series1)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(1, n):
        if (not np.isnan(series1[i]) and not np.isnan(series2[i]) and
            not np.isnan(series1[i-1]) and not np.isnan(series2[i-1]) and
            series1[i] < series2[i] and series1[i-1] >= series2[i-1]):
            result[i] = True
    
    return result


@njit(cache=True, nogil=True)
def highest(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate the highest value over a rolling window using O(n) deque algorithm
    with circular buffer for O(1) front-pop.

    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        Window size

    Returns:
    --------
    np.ndarray
        Array of highest values
    """
    n = len(data)
    result = np.full(n, np.nan)
    cap = n + 1
    dq_val = np.empty(cap, dtype=np.float64)
    dq_idx = np.empty(cap, dtype=np.int64)
    head = 0
    tail = 0

    for i in range(n):
        # Pop front if outside window
        while head < tail and dq_idx[head % cap] <= i - period:
            head += 1

        # Pop back while smaller than current
        while head < tail and dq_val[(tail - 1) % cap] <= data[i]:
            tail -= 1

        # Push current
        dq_val[tail % cap] = data[i]
        dq_idx[tail % cap] = i
        tail += 1

        if i >= period - 1:
            result[i] = dq_val[head % cap]

    return result


@njit(cache=True, nogil=True)
def lowest(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate the lowest value over a rolling window using O(n) deque algorithm
    with circular buffer for O(1) front-pop.

    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        Window size

    Returns:
    --------
    np.ndarray
        Array of lowest values
    """
    n = len(data)
    result = np.full(n, np.nan)
    cap = n + 1
    dq_val = np.empty(cap, dtype=np.float64)
    dq_idx = np.empty(cap, dtype=np.int64)
    head = 0
    tail = 0

    for i in range(n):
        # Pop front if outside window
        while head < tail and dq_idx[head % cap] <= i - period:
            head += 1

        # Pop back while larger than current
        while head < tail and dq_val[(tail - 1) % cap] >= data[i]:
            tail -= 1

        # Push current
        dq_val[tail % cap] = data[i]
        dq_idx[tail % cap] = i
        tail += 1

        if i >= period - 1:
            result[i] = dq_val[head % cap]

    return result


@njit(cache=True, nogil=True)
def change(data: np.ndarray, length: int = 1) -> np.ndarray:
    """
    Calculate the change in value over a specified number of periods
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    length : int, default=1
        Number of periods to look back
        
    Returns:
    --------
    np.ndarray
        Array of change values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(length, n):
        result[i] = data[i] - data[i - length]
    
    return result


@njit(cache=True, nogil=True)
def roc(data: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate Rate of Change (ROC)
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    length : int
        Number of periods to look back
        
    Returns:
    --------
    np.ndarray
        Array of ROC values as percentages
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(length, n):
        if data[i - length] != 0:
            result[i] = ((data[i] - data[i - length]) / data[i - length]) * 100
    
    return result


@njit(cache=True, nogil=True)
def sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average using O(n) rolling sum algorithm
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        Moving average period
        
    Returns:
    --------
    np.ndarray
        Array of SMA values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result
    
    # Calculate initial sum for first window
    rolling_sum = 0.0
    for i in range(period):
        rolling_sum += data[i]
    result[period - 1] = rolling_sum / period
    
    # Use rolling sum for subsequent values
    for i in range(period, n):
        rolling_sum = rolling_sum + data[i] - data[i - period]
        result[i] = rolling_sum / period
    
    return result


@njit(cache=True, nogil=True)
def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average utility function

    Uses first-value seed and alpha = 2 / (period + 1), adjust=False.

    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        EMA period

    Returns:
    --------
    np.ndarray
        Array of EMA values
    """
    n = len(data)
    result = np.empty(n)
    alpha = 2.0 / (period + 1)

    # First-value seed
    result[0] = data[0]

    # EMA recursion
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


@njit(cache=True, nogil=True)
def stdev(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate rolling standard deviation using O(n) rolling sums algorithm
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        Window size for standard deviation calculation
        
    Returns:
    --------
    np.ndarray
        Array of standard deviation values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result
    
    # Initialize rolling sums
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    
    # Calculate initial sums for first window
    for i in range(period):
        rolling_sum += data[i]
        rolling_sum_sq += data[i] * data[i]
    
    mean = rolling_sum / period
    variance = (rolling_sum_sq / period) - (mean * mean)
    result[period - 1] = np.sqrt(max(0.0, variance))  # Ensure non-negative
    
    # Use rolling sums for subsequent values
    for i in range(period, n):
        old_val = data[i - period]
        new_val = data[i]
        
        rolling_sum = rolling_sum + new_val - old_val
        rolling_sum_sq = rolling_sum_sq + (new_val * new_val) - (old_val * old_val)
        
        mean = rolling_sum / period
        variance = (rolling_sum_sq / period) - (mean * mean)
        result[i] = np.sqrt(max(0.0, variance))  # Ensure non-negative
    
    return result


@njit(cache=True, nogil=True)
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calculate True Range (Numba optimized)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Closing prices
        
    Returns:
    --------
    np.ndarray
        Array of True Range values
    """
    n = len(high)
    tr = np.empty(n)
    
    # First TR value
    tr[0] = high[0] - low[0]
    
    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    return tr


@njit(cache=True, nogil=True)
def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range using Wilder's smoothing method (consolidated kernel)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Closing prices
    period : int
        ATR period
        
    Returns:
    --------
    np.ndarray
        Array of ATR values
    """
    n = len(high)
    tr = true_range(high, low, close)
    atr = np.full(n, np.nan)
    
    if n >= period:
        # Initial ATR is simple average
        sum_tr = 0.0
        for i in range(period):
            sum_tr += tr[i]
        atr[period-1] = sum_tr / period
        
        # Subsequent ATR values use Wilder's smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


@njit(cache=True, nogil=True)
def atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range using Simple Moving Average (consolidated kernel)
    
    Parameters:
    -----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Closing prices
    period : int
        ATR period
        
    Returns:
    --------
    np.ndarray
        Array of ATR values
    """
    tr = true_range(high, low, close)
    return sma(tr, period)


@njit(cache=True, nogil=True)
def ema_wilder(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average using Wilder's smoothing (alpha = 1/period)

    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        EMA period

    Returns:
    --------
    np.ndarray
        Array of EMA values
    """
    n = len(data)
    result = np.full(n, np.nan)

    # Find first valid (non-NaN) index
    first_valid = 0
    while first_valid < n and np.isnan(data[first_valid]):
        first_valid += 1

    if first_valid + period > n:
        return result

    # SMA seed over first 'period' values from first_valid
    sum_val = 0.0
    for i in range(first_valid, first_valid + period):
        if np.isnan(data[i]):
            return result
        sum_val += data[i]

    start = first_valid + period - 1
    result[start] = sum_val / period

    # Wilder's smoothing: result[i] = (prev * (period-1) + data[i]) / period
    for i in range(start + 1, n):
        if np.isnan(data[i]):
            result[i] = result[i - 1]
        else:
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period

    return result




@njit(cache=True, nogil=True)
def rolling_variance(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate rolling variance using O(n) rolling sums algorithm
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        Window size for variance calculation
        
    Returns:
    --------
    np.ndarray
        Array of variance values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result
    
    # Initialize rolling sums
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    
    # Calculate initial sums for first window
    for i in range(period):
        rolling_sum += data[i]
        rolling_sum_sq += data[i] * data[i]
    
    mean = rolling_sum / period
    result[period - 1] = (rolling_sum_sq / period) - (mean * mean)
    
    # Use rolling sums for subsequent values
    for i in range(period, n):
        old_val = data[i - period]
        new_val = data[i]
        
        rolling_sum = rolling_sum + new_val - old_val
        rolling_sum_sq = rolling_sum_sq + (new_val * new_val) - (old_val * old_val)
        
        mean = rolling_sum / period
        result[i] = (rolling_sum_sq / period) - (mean * mean)
    
    return result


@njit(cache=True, nogil=True)
def rolling_sum(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate rolling sum using O(n) algorithm
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        Window size
        
    Returns:
    --------
    np.ndarray
        Array of rolling sum values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result
    
    # Calculate initial sum for first window
    rolling_sum = 0.0
    for i in range(period):
        rolling_sum += data[i]
    result[period - 1] = rolling_sum
    
    # Use rolling sum for subsequent values
    for i in range(period, n):
        rolling_sum = rolling_sum + data[i] - data[i - period]
        result[i] = rolling_sum
    
    return result


@njit(cache=True, nogil=True)
def vwma_optimized(data: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """
    Volume Weighted Moving Average using O(n) rolling sums algorithm
    
    Parameters:
    -----------
    data : np.ndarray
        Price data
    volume : np.ndarray
        Volume data
    period : int
        Window size
        
    Returns:
    --------
    np.ndarray
        Array of VWMA values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result
    
    # Initialize rolling sums
    rolling_sum_pv = 0.0  # price * volume
    rolling_sum_v = 0.0   # volume
    
    # Calculate initial sums for first window
    for i in range(period):
        rolling_sum_pv += data[i] * volume[i]
        rolling_sum_v += volume[i]
    
    # Set first result
    if rolling_sum_v > 0:
        result[period - 1] = rolling_sum_pv / rolling_sum_v
    else:
        result[period - 1] = data[period - 1]
    
    # Use rolling sums for subsequent values
    for i in range(period, n):
        old_pv = data[i - period] * volume[i - period]
        new_pv = data[i] * volume[i]
        old_v = volume[i - period]
        new_v = volume[i]
        
        rolling_sum_pv = rolling_sum_pv + new_pv - old_pv
        rolling_sum_v = rolling_sum_v + new_v - old_v
        
        if rolling_sum_v > 0:
            result[i] = rolling_sum_pv / rolling_sum_v
        else:
            result[i] = data[i]
    
    return result


@njit(cache=True, nogil=True)
def cmo_optimized(data: np.ndarray, period: int) -> np.ndarray:
    """
    Chande Momentum Oscillator using O(n) rolling sums algorithm
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int
        CMO period
        
    Returns:
    --------
    np.ndarray
        Array of CMO values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period + 1:
        return result
    
    # Calculate price changes
    changes = np.empty(n - 1)
    for i in range(1, n):
        changes[i - 1] = data[i] - data[i - 1]
    
    # Initialize rolling sums
    rolling_sum_up = 0.0
    rolling_sum_down = 0.0
    
    # Calculate initial sums for first window
    for i in range(period):
        change = changes[i]
        if change > 0:
            rolling_sum_up += change
        elif change < 0:
            rolling_sum_down += abs(change)
    
    # Set first result
    total_movement = rolling_sum_up + rolling_sum_down
    if total_movement > 0:
        result[period] = ((rolling_sum_up - rolling_sum_down) / total_movement) * 100
    else:
        result[period] = 0.0
    
    # Use rolling sums for subsequent values
    for i in range(period + 1, n):
        # Remove old change
        old_change = changes[i - period - 1]
        if old_change > 0:
            rolling_sum_up -= old_change
        elif old_change < 0:
            rolling_sum_down -= abs(old_change)
        
        # Add new change
        new_change = changes[i - 1]
        if new_change > 0:
            rolling_sum_up += new_change
        elif new_change < 0:
            rolling_sum_down += abs(new_change)
        
        # Calculate CMO
        total_movement = rolling_sum_up + rolling_sum_down
        if total_movement > 0:
            result[i] = ((rolling_sum_up - rolling_sum_down) / total_movement) * 100
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True, nogil=True)
def kama_optimized(data: np.ndarray, period: int = 10, fast_sc: float = 2.0, slow_sc: float = 30.0) -> np.ndarray:
    """
    Kaufman's Adaptive Moving Average using O(n) rolling volatility calculation
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    period : int, default=10
        Period for efficiency ratio calculation
    fast_sc : float, default=2.0
        Fast smoothing constant
    slow_sc : float, default=30.0
        Slow smoothing constant
        
    Returns:
    --------
    np.ndarray
        Array of KAMA values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period + 1:
        return result
    
    # Convert smoothing constants to alpha values
    fast_alpha = 2.0 / (fast_sc + 1.0)
    slow_alpha = 2.0 / (slow_sc + 1.0)
    
    # Initialize first value
    result[period] = data[period]
    
    # Initialize rolling volatility sum
    rolling_volatility = 0.0
    
    # Calculate initial volatility sum
    for i in range(1, period + 1):
        rolling_volatility += abs(data[i] - data[i - 1])
    
    for i in range(period + 1, n):
        # Calculate direction (net change over period)
        direction = abs(data[i] - data[i - period])
        
        # Update rolling volatility using O(n) approach
        # Remove old volatility component
        old_volatility = abs(data[i - period] - data[i - period - 1])
        rolling_volatility -= old_volatility
        
        # Add new volatility component
        new_volatility = abs(data[i] - data[i - 1])
        rolling_volatility += new_volatility
        
        # Calculate efficiency ratio
        if rolling_volatility > 0:
            er = direction / rolling_volatility
        else:
            er = 0.0
        
        # Calculate smoothing constant
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # Calculate KAMA
        result[i] = result[i - 1] + sc * (data[i] - result[i - 1])
    
    return result


@njit(cache=True, nogil=True)
def ulcer_index_optimized(data: np.ndarray, period: int) -> np.ndarray:
    """
    Improved O(n×period) Ulcer Index using running peak calculation
    (True O(n) is complex for Ulcer Index due to the nature of drawdown calculation)
    
    Parameters:
    -----------
    data : np.ndarray
        Price data (typically closing prices)
    period : int
        Period for Ulcer Index calculation
        
    Returns:
    --------
    np.ndarray
        Array of Ulcer Index values
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result
    
    for i in range(period - 1, n):
        sum_squared_drawdown = 0.0
        running_peak = data[i - period + 1]
        
        # Use running peak approach - O(period) per point
        for j in range(period):
            idx = i - period + 1 + j
            running_peak = max(running_peak, data[idx])
            
            if running_peak > 0:
                drawdown = (data[idx] - running_peak) / running_peak
                sum_squared_drawdown += drawdown * drawdown
        
        result[i] = np.sqrt(sum_squared_drawdown / period) * 100
    
    return result


@njit(cache=True, nogil=True)
def exrem(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """
    Excess Removal function - eliminates excessive signals
    
    Parameters:
    -----------
    primary : np.ndarray
        Primary signal array (boolean-like)
    secondary : np.ndarray
        Secondary signal array (boolean-like)
        
    Returns:
    --------
    np.ndarray
        Boolean array with excess signals removed
    """
    n = len(primary)
    result = np.zeros(n, dtype=np.bool_)
    active = False
    
    for i in range(n):
        if not active and primary[i]:
            result[i] = True
            active = True
        elif secondary[i]:
            active = False
    
    return result


@njit(cache=True, nogil=True)
def flip(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """
    Flip function - creates a toggle state based on two signals
    
    Parameters:
    -----------
    primary : np.ndarray
        Primary signal array (boolean-like)
    secondary : np.ndarray
        Secondary signal array (boolean-like)
        
    Returns:
    --------
    np.ndarray
        Boolean array representing flip state
    """
    n = len(primary)
    result = np.zeros(n, dtype=np.bool_)
    active = False
    
    for i in range(n):
        if primary[i]:
            active = True
        elif secondary[i]:
            active = False
        result[i] = active
    
    return result


@njit(cache=True, nogil=True)
def valuewhen(expr: np.ndarray, array: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Returns the value of array when expr was true for the nth most recent time
    
    Parameters:
    -----------
    expr : np.ndarray
        Expression array (boolean-like)
    array : np.ndarray
        Value array to sample from
    n : int, default=1
        Which occurrence to get (1 = most recent, 2 = second most recent, etc.)
        
    Returns:
    --------
    np.ndarray
        Array of values when condition was true
    """
    length = len(expr)
    result = np.full(length, np.nan)
    
    # Pre-allocate arrays for true indices tracking
    max_lookback = min(1000, length)  # Limit memory usage
    true_indices = np.empty(max_lookback, dtype=np.int64)
    true_count = 0
    
    for i in range(length):
        # Add current index if expr is true
        if expr[i]:
            # Shift existing indices if we're at capacity
            if true_count >= max_lookback:
                for j in range(max_lookback - 1):
                    true_indices[j] = true_indices[j + 1]
                true_indices[max_lookback - 1] = i
            else:
                true_indices[true_count] = i
                true_count += 1
        
        # Find the nth most recent true occurrence
        if true_count >= n:
            target_idx = true_indices[true_count - n]
            result[i] = array[target_idx]
    
    return result


@njit(cache=True, nogil=True)
def rising(data: np.ndarray, length: int) -> np.ndarray:
    """
    Check if data is rising (current value > value n periods ago)
    
    Parameters:
    -----------
    data : np.ndarray
        Input data series
    length : int
        Number of periods to look back
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating rising periods
    """
    n = len(data)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(length, n):
        if not np.isnan(data[i]) and not np.isnan(data[i - length]):
            result[i] = data[i] > data[i - length]

    return result


@njit(cache=True, nogil=True)
def falling(data: np.ndarray, length: int) -> np.ndarray:
    """
    Check if data is falling (current value < value n periods ago)
    
    Parameters:
    -----------
    data : np.ndarray
        Input data series
    length : int
        Number of periods to look back
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating falling periods
    """
    n = len(data)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(length, n):
        if not np.isnan(data[i]) and not np.isnan(data[i - length]):
            result[i] = data[i] < data[i - length]

    return result


@njit(cache=True, nogil=True)
def cross(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Check if series1 crosses series2 (either direction)
    Combines crossover and crossunder functionality
    
    Parameters:
    -----------
    series1 : np.ndarray
        First series
    series2 : np.ndarray
        Second series
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating cross points (both over and under)
    """
    n = len(series1)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(1, n):
        if (not np.isnan(series1[i]) and not np.isnan(series2[i]) and
            not np.isnan(series1[i-1]) and not np.isnan(series2[i-1])):

            crossover_condition = (series1[i] > series2[i] and series1[i-1] <= series2[i-1])
            crossunder_condition = (series1[i] < series2[i] and series1[i-1] >= series2[i-1])

            result[i] = crossover_condition or crossunder_condition
    
    return result
# -*- coding: utf-8 -*-
"""
OpenAlgo Technical Indicators - Statistical Indicators
"""

import numpy as np
import pandas as pd
from openalgo.numba_shim import jit
from typing import Union, Tuple, Optional
from .base import BaseIndicator
from .utils import sma, ema, stdev


class LINREG(BaseIndicator):
    """
    Linear Regression
    
    Linear Regression calculates the linear regression line for the given period.
    
    Formula: y = mx + b (least squares method)
    """
    
    def __init__(self):
        super().__init__("Linear Regression")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_linearreg(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized Linear Regression calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            # Extract window
            y = data[i - period + 1:i + 1]
            x = np.arange(period)
            
            # Calculate linear regression
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            # Calculate slope and intercept
            denominator = period * sum_x2 - sum_x * sum_x
            if denominator != 0:
                slope = (period * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / period
                
                # Calculate value at the end of the period
                result[i] = slope * (period - 1) + intercept
            else:
                result[i] = y[-1]  # Fallback to last value
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Linear Regression
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=14
            Period for linear regression calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Linear Regression values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = self._calculate_linearreg(validated_data, period)
        return self.format_output(result, input_type, index)


class LRSLOPE(BaseIndicator):
    """
    Linear Regression Slope - matches TradingView exactly
    
    TradingView calculates slope as the difference between consecutive 
    linear regression values divided by interval:
    linear_reg = linreg(close_price, len, 0)
    linear_reg_prev = linreg(close[1], len, 0)  
    slope = ((linear_reg - linear_reg_prev) / interval)
    """
    
    def __init__(self):
        super().__init__("Linear Regression Slope")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_linreg_value(data: np.ndarray, period: int, offset: int) -> float:
        """Calculate linear regression value at given offset"""
        y = data
        x = np.arange(len(y))
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        denominator = period * sum_x2 - sum_x * sum_x
        if denominator != 0:
            slope = (period * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / period
            
            # Calculate value at offset position (TradingView uses offset 0 for current)
            return slope * (period - 1 - offset) + intercept
        else:
            return y[-1 - offset] if offset < len(y) else y[-1]
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 period: int = 100, interval: int = 1) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Linear Regression Slope - matches TradingView exactly
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (typically closing prices)
        period : int, default=100
            Period for linear regression calculation (TradingView default)
        interval : int, default=1
            Interval divisor (TradingView uses timeframe interval)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Slope values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period + 1, len(validated_data))  # Need period+1 for TradingView method
        
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        
        result = _calculate_slope_tv(validated_data, period, interval)
        return self.format_output(result, input_type, index)


@jit(nopython=True)
def _calculate_linreg_value_standalone(data: np.ndarray, period: int, offset: int) -> float:
    """Calculate linear regression value at given offset (standalone function)"""
    y = data
    x = np.arange(len(y))
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = period * sum_x2 - sum_x * sum_x
    if denominator != 0:
        slope = (period * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / period
        
        # Calculate value at offset position (TradingView uses offset 0 for current)
        return slope * (period - 1 - offset) + intercept
    else:
        return y[-1 - offset] if offset < len(y) else y[-1]


@jit(nopython=True) 
def _calculate_slope_tv(data: np.ndarray, period: int, interval: int = 1) -> np.ndarray:
    """
    Calculate slope using TradingView method:
    slope = ((linreg(close, len, 0) - linreg(close[1], len, 0)) / interval)
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period, n):  # Start from period (not period-1) to have previous value
        # Current window
        current_window = data[i - period + 1:i + 1]
        # Previous window (shifted by 1)
        prev_window = data[i - period:i] 
        
        # Calculate linear regression values
        linear_reg = _calculate_linreg_value_standalone(current_window, period, 0)
        linear_reg_prev = _calculate_linreg_value_standalone(prev_window, period, 0)
        
        # Calculate slope as TradingView does
        result[i] = (linear_reg - linear_reg_prev) / interval
    
    return result


class CORREL(BaseIndicator):
    """
    Pearson Correlation Coefficient
    
    Measures the correlation between two data series.
    
    Formula: r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)
    """
    
    def __init__(self):
        super().__init__("Correlation")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_correl(data1: np.ndarray, data2: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized correlation calculation"""
        n = len(data1)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            x = data1[i - period + 1:i + 1]
            y = data2[i - period + 1:i + 1]
            
            # Calculate means
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            # Calculate correlation coefficient
            numerator = np.sum((x - mean_x) * (y - mean_y))
            sum_sq_x = np.sum((x - mean_x) ** 2)
            sum_sq_y = np.sum((y - mean_y) ** 2)
            
            denominator = np.sqrt(sum_sq_x * sum_sq_y)
            
            if denominator > 0:
                result[i] = numerator / denominator
            else:
                result[i] = 0
        
        return result
    
    def calculate(self, data1: Union[np.ndarray, pd.Series, list],
                 data2: Union[np.ndarray, pd.Series, list],
                 period: int = 20) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Pearson Correlation Coefficient
        
        Parameters:
        -----------
        data1 : Union[np.ndarray, pd.Series, list]
            First data series
        data2 : Union[np.ndarray, pd.Series, list]
            Second data series
        period : int, default=20
            Period for correlation calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Correlation values in the same format as input
        """
        data1_validated, input_type, index = self.validate_input(data1)
        data2_validated, _, _ = self.validate_input(data2)
        
        data1_validated, data2_validated = self.align_arrays(data1_validated, data2_validated)
        self.validate_period(period, len(data1_validated))
        
        result = self._calculate_correl(data1_validated, data2_validated, period)
        return self.format_output(result, input_type, index)


class BETA(BaseIndicator):
    """
    Beta Coefficient
    
    Measures the volatility of a security relative to the market.
    
    Formula: β = Cov(asset, market) / Var(market)
    """
    
    def __init__(self):
        super().__init__("Beta")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_beta_optimized(asset: np.ndarray, market: np.ndarray, period: int) -> np.ndarray:
        """Optimized Beta calculation with pre-computed returns"""
        n = len(asset)
        result = np.full(n, np.nan)
        
        # Pre-calculate returns once (optimization from audit suggestion)
        asset_returns = np.full(n, np.nan)
        market_returns = np.full(n, np.nan)
        
        for i in range(1, n):
            asset_returns[i] = asset[i] - asset[i - 1]
            market_returns[i] = market[i] - market[i - 1]
        
        # O(N) rolling covariance and variance using incremental updates
        for i in range(period, n):
            # Extract window of returns
            asset_window = asset_returns[i - period + 1:i + 1]
            market_window = market_returns[i - period + 1:i + 1]
            
            # Calculate means
            mean_asset = np.mean(asset_window)
            mean_market = np.mean(market_window)
            
            # Calculate covariance and variance
            covariance = 0.0
            market_variance = 0.0
            
            for j in range(period):
                asset_dev = asset_window[j] - mean_asset
                market_dev = market_window[j] - mean_market
                covariance += asset_dev * market_dev
                market_variance += market_dev * market_dev
            
            covariance /= period
            market_variance /= period
            
            if market_variance > 0:
                result[i] = covariance / market_variance
            else:
                result[i] = 0
        
        return result
    
    def calculate(self, asset: Union[np.ndarray, pd.Series, list],
                 market: Union[np.ndarray, pd.Series, list],
                 period: int = 252) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Beta Coefficient
        
        Parameters:
        -----------
        asset : Union[np.ndarray, pd.Series, list]
            Asset price data
        market : Union[np.ndarray, pd.Series, list]
            Market price data
        period : int, default=252
            Period for beta calculation (typically 1 year = 252 trading days)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Beta values in the same format as input
        """
        asset_data, input_type, index = self.validate_input(asset)
        market_data, _, _ = self.validate_input(market)
        
        asset_data, market_data = self.align_arrays(asset_data, market_data)
        self.validate_period(period + 1, len(asset_data))  # +1 for diff
        
        result = self._calculate_beta_optimized(asset_data, market_data, period)
        return self.format_output(result, input_type, index)


class VAR(BaseIndicator):
    """
    Variance - TradingView Pine Script v4 Implementation
    
    Calculates variance with support for both logarithmic returns and price modes,
    including EMA smoothing and z-score analysis with signal generation.
    
    TradingView Pine Script v4 Implementation by moot-al-cabal:
    
    Variance modes:
    - "LR" (Logarithmic Returns): variance of log(close/close[1])*100
    - "PR" (Price): variance of price values
    
    Formula:
    1. source = if mode == "LR" then log(close/close[1])*100 else close
    2. mean = sma(source, lookback)
    3. For each i in lookback: array.push(s, pow(source[i]-mean, 2))
    4. variance = array.sum(s) / (lookback-1)
    5. stdev = sqrt(variance)
    6. ema_variance = ema(variance, ema_period)
    7. zscore = (variance - sma(variance, filter_lookback)) / stdev(variance, filter_lookback)
    8. ema_zscore = ema(zscore, ema_length)
    """
    
    def __init__(self):
        super().__init__("Variance")
    
    
    @staticmethod
    @jit(nopython=True) 
    def _calculate_variance_tv_optimized(data: np.ndarray, lookback: int, use_log_returns: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized TradingView Variance calculation with O(N) rolling statistics"""
        n = len(data)
        
        # Step 1: Calculate source based on mode
        source = np.full(n, np.nan)
        
        if use_log_returns:  # Logarithmic Returns
            for i in range(1, n):
                if data[i] > 0 and data[i-1] > 0:
                    source[i] = np.log(data[i] / data[i-1]) * 100
        else:  # Price mode
            source = data.copy()
        
        # Step 2: Calculate variance using O(N) rolling statistics
        variance = np.full(n, np.nan)
        stdev = np.full(n, np.nan)
        
        if n < lookback:
            return source, variance, stdev
        
        # O(N) rolling mean calculation using cumulative sums
        rolling_sum = 0.0
        rolling_sum_sq = 0.0
        
        # Initialize first window
        for i in range(lookback):
            if not np.isnan(source[i]):
                rolling_sum += source[i]
                rolling_sum_sq += source[i] * source[i]
        
        # Calculate first variance
        if lookback > 1:
            mean_val = rolling_sum / lookback
            variance_val = (rolling_sum_sq - lookback * mean_val * mean_val) / (lookback - 1)
            if variance_val >= 0:  # Ensure non-negative due to floating point precision
                variance[lookback - 1] = variance_val
                stdev[lookback - 1] = np.sqrt(variance_val)
        
        # Rolling window updates for O(N) complexity
        for i in range(lookback, n):
            # Remove old value, add new value
            old_val = source[i - lookback]
            new_val = source[i]
            
            if not np.isnan(old_val):
                rolling_sum -= old_val
                rolling_sum_sq -= old_val * old_val
            
            if not np.isnan(new_val):
                rolling_sum += new_val
                rolling_sum_sq += new_val * new_val
            
            # Calculate variance for current window
            if lookback > 1:
                mean_val = rolling_sum / lookback
                variance_val = (rolling_sum_sq - lookback * mean_val * mean_val) / (lookback - 1)
                if variance_val >= 0:  # Ensure non-negative due to floating point precision
                    variance[i] = variance_val
                    stdev[i] = np.sqrt(variance_val)
        
        return source, variance, stdev
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 lookback: int = 20, mode: str = "PR", ema_period: int = 20,
                 filter_lookback: int = 20, ema_length: int = 14,
                 return_components: bool = False) -> Union[np.ndarray, pd.Series, Tuple]:
        """
        Calculate Variance - TradingView Pine Script v4 Implementation
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (close prices)
        lookback : int, default=20
            Variance lookback period
        mode : str, default="PR"
            Variance mode: "LR" for Logarithmic Returns, "PR" for Price
        ema_period : int, default=20
            EMA period for variance smoothing
        filter_lookback : int, default=20
            Lookback period for variance filter (z-score calculation)
        ema_length : int, default=14
            EMA length for z-score smoothing
        return_components : bool, default=False
            If True, returns (variance, ema_variance, zscore, ema_zscore, stdev)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series, Tuple]
            Variance values or tuple of all components
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(lookback, len(validated_data))
        
        # Validate mode
        if mode not in ["LR", "PR"]:
            raise ValueError("Mode must be 'LR' (Logarithmic Returns) or 'PR' (Price)")
        
        # Calculate base variance components using optimized O(N) method
        use_log_returns = (mode == "LR")
        source, variance, stdev_values = self._calculate_variance_tv_optimized(validated_data, lookback, use_log_returns)
        
        # Calculate EMA of variance using optimized utility
        ema_variance = ema(variance, ema_period)
        
        # Calculate z-score components using optimized utilities
        variance_sma = sma(variance, filter_lookback)
        variance_stdev = stdev(variance, filter_lookback)
        
        # Calculate z-score: (variance - sma(variance)) / stdev(variance) - vectorized
        zscore = np.where((~np.isnan(variance)) & (~np.isnan(variance_sma)) & 
                         (~np.isnan(variance_stdev)) & (variance_stdev > 0),
                         (variance - variance_sma) / variance_stdev, np.nan)
        
        # Calculate EMA of z-score using optimized utility
        ema_zscore = ema(zscore, ema_length)
        
        if return_components:
            results = (variance, ema_variance, zscore, ema_zscore, stdev_values)
            return self.format_multiple_outputs(results, input_type, index)
        else:
            return self.format_output(variance, input_type, index)


class TSF(BaseIndicator):
    """
    Time Series Forecast
    
    Forecasts the next value using linear regression.
    """
    
    def __init__(self):
        super().__init__("Time Series Forecast")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_tsf(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized TSF calculation"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            y = data[i - period + 1:i + 1]
            x = np.arange(period)
            
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            denominator = period * sum_x2 - sum_x * sum_x
            if denominator != 0:
                slope = (period * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / period
                
                # Forecast next value
                result[i] = slope * period + intercept
            else:
                result[i] = y[-1]
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 14) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Time Series Forecast
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data
        period : int, default=14
            Period for forecast calculation
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Time Series Forecast values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = self._calculate_tsf(validated_data, period)
        return self.format_output(result, input_type, index)


class MEDIAN(BaseIndicator):
    """
    Rolling Median (Pine Script v6)
    
    Calculates the median value over a rolling window using percentile_nearest_rank.
    Pine Script v6 Formula:
    median = ta.percentile_nearest_rank(source, length, 50)
    
    This is equivalent to the 50th percentile of the data.
    """
    
    def __init__(self):
        super().__init__("Median")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_median(data: np.ndarray, period: int) -> np.ndarray:
        """Numba optimized median calculation (percentile_nearest_rank with 50th percentile)"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1].copy()
            
            # Sort the window
            for j in range(period):
                for k in range(period - 1 - j):
                    if window[k] > window[k + 1]:
                        window[k], window[k + 1] = window[k + 1], window[k]
            
            # Get median (50th percentile) - nearest rank method
            # For percentile_nearest_rank, we take the actual value at the position
            if period % 2 == 1:
                result[i] = window[period // 2]
            else:
                # For even length, take the average of the two middle values
                result[i] = (window[period // 2 - 1] + window[period // 2]) / 2.0
        
        return result
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], period: int = 3) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Rolling Median (Pine Script v6)
        
        Pine Script v6 Formula:
        median = ta.percentile_nearest_rank(source, length, 50)
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data (default hl2 in Pine Script)
        period : int, default=3
            Period for median calculation (Pine Script default)
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Median values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = self._calculate_median(validated_data, period)
        return self.format_output(result, input_type, index)


class MedianBands(BaseIndicator):
    """
    Median Bands (Pine Script v6 Complete Implementation)
    
    Implements the complete Pine Script v6 Median indicator with:
    - Median line using percentile_nearest_rank
    - ATR-based upper and lower bands
    - EMA of the median
    
    Pine Script v6 Formula:
    median = ta.percentile_nearest_rank(source, length, 50)
    atr_ = atr_mult * ta.atr(atr_length)
    upper = median + atr_
    lower = median - atr_
    median_ema = ta.ema(median, length)
    """
    
    def __init__(self):
        super().__init__("Median Bands")
    
    def calculate(self, *args, **kwargs):
        """Wrapper for calculate_with_bands to satisfy BaseIndicator abstract method"""
        return self.calculate_with_bands(*args, **kwargs)
        
    @staticmethod
    @jit(nopython=True)
    def _calculate_median_percentile(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate median using percentile_nearest_rank method"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1].copy()
            
            # Sort the window
            for j in range(period):
                for k in range(period - 1 - j):
                    if window[k] > window[k + 1]:
                        window[k], window[k + 1] = window[k + 1], window[k]
            
            # percentile_nearest_rank for 50th percentile
            if period % 2 == 1:
                result[i] = window[period // 2]
            else:
                # For even length, take the lower middle value
                result[i] = window[period // 2 - 1]
        
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
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR"""
        n = len(close)
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR (Wilder's smoothing)
        # Initial ATR
        if n >= period:
            sum_tr = 0.0
            for i in range(period):
                sum_tr += tr[i]
            atr[period-1] = sum_tr / period
            
            # Subsequent ATR values
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def calculate_with_bands(self, high: Union[np.ndarray, pd.Series, list],
                            low: Union[np.ndarray, pd.Series, list],
                            close: Union[np.ndarray, pd.Series, list],
                            source: Optional[Union[np.ndarray, pd.Series, list]] = None,
                            median_length: int = 3,
                            atr_length: int = 14,
                            atr_mult: float = 2.0) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                                                           Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Median with Bands and EMA (Pine Script v6 Complete)
        
        Parameters:
        -----------
        high : Union[np.ndarray, pd.Series, list]
            High prices
        low : Union[np.ndarray, pd.Series, list]
            Low prices
        close : Union[np.ndarray, pd.Series, list]
            Close prices
        source : Optional[Union[np.ndarray, pd.Series, list]]
            Source data for median (default: hl2 = (high + low) / 2)
        median_length : int, default=3
            Period for median calculation
        atr_length : int, default=14
            Period for ATR calculation
        atr_mult : float, default=2.0
            ATR multiplier for bands
            
        Returns:
        --------
        Tuple of (median, upper_band, lower_band, median_ema)
        """
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        close_data, _, _ = self.validate_input(close)
        
        # Align arrays
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        
        # Calculate source (hl2 if not provided)
        if source is None:
            source_data = (high_data + low_data) / 2.0
        else:
            source_data, _, _ = self.validate_input(source)
        
        # Calculate median
        median = self._calculate_median_percentile(source_data, median_length)
        
        # Calculate ATR
        atr = self._calculate_atr(high_data, low_data, close_data, atr_length)
        atr_scaled = atr * atr_mult
        
        # Calculate bands
        upper_band = median + atr_scaled
        lower_band = median - atr_scaled
        
        # Calculate EMA of median
        median_ema = self._calculate_ema(median, median_length)
        
        results = (median, upper_band, lower_band, median_ema)
        return self.format_multiple_outputs(results, input_type, index)


class MODE(BaseIndicator):
    """
    Rolling Mode
    
    Calculates the most frequent value over a rolling window.
    """
    
    def __init__(self):
        super().__init__("Mode")
    
    def calculate(self, data: Union[np.ndarray, pd.Series, list], 
                 period: int = 20, bins: int = 10) -> Union[np.ndarray, pd.Series]:
        """
        Calculate Rolling Mode using optimized algorithm
        
        Parameters:
        -----------
        data : Union[np.ndarray, pd.Series, list]
            Price data
        period : int, default=20
            Period for mode calculation
        bins : int, default=10
            Number of bins for discretization
            
        Returns:
        --------
        Union[np.ndarray, pd.Series]
            Mode values in the same format as input
        """
        validated_data, input_type, index = self.validate_input(data)
        self.validate_period(period, len(validated_data))
        
        result = self._calculate_mode_optimized(validated_data, period, bins)
        return self.format_output(result, input_type, index)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_mode_optimized(data: np.ndarray, period: int, bins: int) -> np.ndarray:
        """Optimized rolling mode calculation using vectorized binning"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            
            # Fast min/max calculation
            min_val = np.min(window)
            max_val = np.max(window)
            
            if max_val > min_val:
                bin_width = (max_val - min_val) / bins
                
                # Vectorized binning
                bin_indices = ((window - min_val) / bin_width).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, bins - 1)
                
                # Fast histogram using numpy bincount
                counts = np.bincount(bin_indices, minlength=bins)
                
                # Find mode bin
                mode_bin = np.argmax(counts)
                result[i] = min_val + (mode_bin + 0.5) * bin_width
            else:
                result[i] = window[0]
        
        return result
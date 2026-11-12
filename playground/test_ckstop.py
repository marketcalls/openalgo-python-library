#!/usr/bin/env python
"""Test Chande Kroll Stop with its default parameters"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import ta
import numpy as np

def test_ckstop_basic():
    """Test CKStop with basic functionality"""
    
    print("Testing Chande Kroll Stop (CKStop)")
    print("=" * 40)
    
    # Create test data with trend
    np.random.seed(42)
    n = 100
    
    # Create uptrend data
    trend = np.linspace(0, 0.2, n)
    noise = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(trend + noise))
    
    high = prices * (1 + np.random.uniform(0.001, 0.02, n))
    low = prices * (1 - np.random.uniform(0.001, 0.02, n))
    close = prices
    
    print(f"Generated {n} test data points")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test with the default parameters: p=10, x=1, q=9
    p, x, q = 10, 1, 9
    
    # Calculate CKStop
    stop_long, stop_short = ta.ckstop(high, low, close, p=p, x=x, q=q)
    
    print(f"\nUsing default parameters: p={p}, x={x}, q={q}")
    
    # Check for valid data
    valid_long = ~np.isnan(stop_long)
    valid_short = ~np.isnan(stop_short)
    
    long_count = valid_long.sum()
    short_count = valid_short.sum()
    
    print(f"Valid stop_long values: {long_count}/{n}")
    print(f"Valid stop_short values: {short_count}/{n}")
    
    if long_count > 0 and short_count > 0:
        first_valid_idx = max(np.where(valid_long)[0][0], np.where(valid_short)[0][0])
        expected_start = p + q - 2  # Need p periods for ATR + q periods for smoothing
        print(f"First valid index: {first_valid_idx}, Expected: {expected_start}")
        
        # Show some values
        print(f"\nSample values (last 10):")
        for i in range(max(0, n-10), n):
            if valid_long[i] and valid_short[i]:
                print(f"Index {i}: High={high[i]:.2f}, Low={low[i]:.2f}, Close={close[i]:.2f}")
                print(f"  Stop Long={stop_long[i]:.2f}, Stop Short={stop_short[i]:.2f}")
        
        # Basic validation
        valid_indices = np.where(valid_long & valid_short)[0]
        if len(valid_indices) > 0:
            # stop_long should be below prices (support)
            # stop_short should be above prices (resistance)
            long_stops = stop_long[valid_indices]
            short_stops = stop_short[valid_indices]
            lows = low[valid_indices]
            highs = high[valid_indices]
            
            long_below_low = np.all(long_stops <= lows)
            short_above_high = np.all(short_stops >= highs)
            
            print(f"\nValidation:")
            print(f"Stop Long <= Low: {long_below_low}")
            print(f"Stop Short >= High: {short_above_high}")
            
            # Statistics
            print(f"\nStatistics:")
            print(f"Stop Long range: {long_stops.min():.2f} - {long_stops.max():.2f}")
            print(f"Stop Short range: {short_stops.min():.2f} - {short_stops.max():.2f}")
            
            # Test different parameters
            print(f"\nTesting different parameters:")
            
            # Test with different ATR coefficient
            stop_long2, stop_short2 = ta.ckstop(high, low, close, p=10, x=2.0, q=9)
            valid_idx = np.where(~np.isnan(stop_long2) & ~np.isnan(stop_short2))[0]
            if len(valid_idx) > 0:
                last_idx = valid_idx[-1]
                print(f"x=2.0: Stop Long={stop_long2[last_idx]:.2f}, Stop Short={stop_short2[last_idx]:.2f}")
                print(f"x=1.0: Stop Long={stop_long[last_idx]:.2f}, Stop Short={stop_short[last_idx]:.2f}")
                print(f"Higher x makes wider stops: {stop_long2[last_idx] < stop_long[last_idx] and stop_short2[last_idx] > stop_short[last_idx]}")
            
            return True
    
    return False

def test_ckstop_real_pattern():
    """Test with a more realistic pattern"""
    
    print(f"\nTesting with realistic price pattern:")
    print("-" * 40)
    
    # Create realistic OHLC data
    np.random.seed(123)
    n = 50
    
    # Base price movement
    returns = np.random.normal(0.001, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    
    # Create realistic OHLC
    high = np.zeros(n)
    low = np.zeros(n) 
    close = prices
    
    for i in range(n):
        daily_range = abs(np.random.normal(0, 0.01)) * prices[i]
        high[i] = prices[i] + daily_range * np.random.uniform(0.3, 0.7)
        low[i] = prices[i] - daily_range * np.random.uniform(0.3, 0.7)
    
    # Calculate with defaults
    stop_long, stop_short = ta.ckstop(high, low, close)
    
    valid_mask = ~np.isnan(stop_long) & ~np.isnan(stop_short)
    valid_count = valid_mask.sum()
    
    print(f"Generated {n} realistic bars")
    print(f"Valid CKStop values: {valid_count}/{n}")
    
    if valid_count > 5:
        print(f"\nLast 5 valid values:")
        valid_indices = np.where(valid_mask)[0][-5:]
        for idx in valid_indices:
            print(f"Bar {idx}: H={high[idx]:.2f} L={low[idx]:.2f} C={close[idx]:.2f}")
            print(f"  Stops: Long={stop_long[idx]:.2f} Short={stop_short[idx]:.2f}")
        
        return True
    
    return False

if __name__ == "__main__":
    print("Chande Kroll Stop Test")
    print("=" * 55)
    
    success1 = test_ckstop_basic()
    success2 = test_ckstop_real_pattern()
    
    print(f"\nOverall Test: {'PASSED' if success1 and success2 else 'FAILED'}")
    print(f"\nCKStop Implementation Summary:")
    print(f"- Uses 3 parameters: p (ATR length), x (ATR coefficient), q (stop length)")
    print(f"- Default values: p=10, x=1.0, q=9")
    print(f"- Two-step calculation: first_stops -> smoothed_stops")
    print(f"- Returns (stop_long, stop_short)")
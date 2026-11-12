#!/usr/bin/env python
"""Comprehensive test of both Aroon and Aroon Oscillator"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import ta
import numpy as np

def test_aroon_comprehensive():
    """Test both Aroon and Aroon Oscillator implementations"""
    
    print("Comprehensive Aroon & Aroon Oscillator Test")
    print("=" * 50)
    
    # Create test data with clear trend patterns
    np.random.seed(42)
    n = 100
    
    # Create segments: uptrend, sideways, downtrend
    uptrend = np.linspace(0, 0.15, 30)
    sideways = np.random.normal(0, 0.005, 30) 
    downtrend = np.linspace(0, -0.1, 40)
    
    trends = np.concatenate([uptrend, sideways, downtrend])
    noise = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(trends + noise))
    
    high = prices * (1 + np.random.uniform(0.001, 0.015, n))
    low = prices * (1 - np.random.uniform(0.001, 0.015, n))
    
    print(f"Generated {n} data points with trend patterns")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test with the default period
    period = 14
    
    # Calculate indicators
    aroon_up, aroon_down = ta.aroon(high, low, period)
    aroon_osc = ta.aroon_oscillator(high, low, period)
    
    # Verify consistency
    manual_osc = aroon_up - aroon_down
    max_diff = np.nanmax(np.abs(aroon_osc - manual_osc))
    
    print(f"\nCalculation Results (period={period}):")
    print(f"Aroon Oscillator consistency check: {max_diff:.10f}")
    print(f"Consistency: {'PERFECT' if max_diff < 1e-10 else 'ISSUE'}")
    
    # Analyze segments
    valid_mask = ~np.isnan(aroon_osc)
    if valid_mask.sum() > 0:
        valid_start = np.where(valid_mask)[0][0]
        
        # Segment analysis (accounting for warm-up period)
        uptrend_start = max(valid_start, 0)
        uptrend_end = min(30, n)
        sideways_start = max(valid_start, 30) 
        sideways_end = min(60, n)
        downtrend_start = max(valid_start, 60)
        downtrend_end = n
        
        if uptrend_end > uptrend_start:
            uptrend_osc = aroon_osc[uptrend_start:uptrend_end]
            uptrend_avg = np.nanmean(uptrend_osc)
        else:
            uptrend_avg = np.nan
            
        if sideways_end > sideways_start:
            sideways_osc = aroon_osc[sideways_start:sideways_end]
            sideways_avg = np.nanmean(sideways_osc)
        else:
            sideways_avg = np.nan
            
        if downtrend_end > downtrend_start:
            downtrend_osc = aroon_osc[downtrend_start:downtrend_end]
            downtrend_avg = np.nanmean(downtrend_osc)
        else:
            downtrend_avg = np.nan
        
        print(f"\nTrend Analysis:")
        print(f"Uptrend avg oscillator: {uptrend_avg:.1f}")
        print(f"Sideways avg oscillator: {sideways_avg:.1f}")
        print(f"Downtrend avg oscillator: {downtrend_avg:.1f}")
        
        # Expected behavior
        expected_behavior = True
        if not np.isnan(uptrend_avg) and uptrend_avg <= 0:
            expected_behavior = False
            print("  WARNING: Uptrend should have positive oscillator")
        if not np.isnan(downtrend_avg) and downtrend_avg >= 0:
            expected_behavior = False
            print("  WARNING: Downtrend should have negative oscillator")
        
        if expected_behavior:
            print("  Trend behavior: EXPECTED")
        
        # Value ranges
        valid_up = aroon_up[valid_mask]
        valid_down = aroon_down[valid_mask]
        valid_osc = aroon_osc[valid_mask]
        
        print(f"\nValue Ranges:")
        print(f"Aroon Up: {valid_up.min():.1f} to {valid_up.max():.1f}")
        print(f"Aroon Down: {valid_down.min():.1f} to {valid_down.max():.1f}")
        print(f"Aroon Oscillator: {valid_osc.min():.1f} to {valid_osc.max():.1f}")
        
        # Range validation
        range_ok = (
            valid_up.min() >= 0 and valid_up.max() <= 100 and
            valid_down.min() >= 0 and valid_down.max() <= 100 and
            valid_osc.min() >= -100 and valid_osc.max() <= 100
        )
        
        print(f"Range validation: {'PASS' if range_ok else 'FAIL'}")
        
        # Recent values
        print(f"\nRecent values (last 5):")
        for i in range(max(0, n-5), n):
            if valid_mask[i]:
                print(f"  Index {i}: Up={aroon_up[i]:.1f}, Down={aroon_down[i]:.1f}, Osc={aroon_osc[i]:.1f}")
        
        return max_diff < 1e-10 and range_ok and expected_behavior
    
    return False

if __name__ == "__main__":
    success = test_aroon_comprehensive()
    print(f"\nOverall Test: {'PASSED' if success else 'FAILED'}")
    print(f"\nSummary:")
    print(f"- Uses period + 1 lookback")
    print(f"- Uses first occurrence for ties")
    print(f"- Default period changed from 25 to 14")
    print(f"- Aroon Oscillator = Aroon Up - Aroon Down")
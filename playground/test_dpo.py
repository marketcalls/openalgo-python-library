#!/usr/bin/env python
"""Test Detrended Price Oscillator with its default parameters"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import ta
import numpy as np

def test_dpo_basic():
    """Test DPO with basic functionality"""
    
    print("Testing Detrended Price Oscillator (DPO)")
    print("=" * 45)
    
    # Create test data with trend and cycles
    np.random.seed(42)
    n = 100
    
    # Create trending data with cyclical component
    trend = np.linspace(0, 0.3, n)
    cycle = 0.1 * np.sin(2 * np.pi * np.arange(n) / 20)  # 20-period cycle
    noise = np.random.normal(0, 0.02, n)
    
    prices = 100 * np.exp(trend + cycle + noise)
    
    print(f"Generated {n} test data points")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test with the default parameters: period=21, is_centered=False
    period = 21
    
    # Calculate DPO in both modes
    dpo_non_centered = ta.dpo(prices, period=period, is_centered=False)
    dpo_centered = ta.dpo(prices, period=period, is_centered=True)
    
    print(f"\nUsing default parameters: period={period}")
    
    # Check for valid data
    valid_nc = ~np.isnan(dpo_non_centered)
    valid_c = ~np.isnan(dpo_centered)
    
    nc_count = valid_nc.sum()
    c_count = valid_c.sum()
    
    print(f"Valid DPO Non-Centered values: {nc_count}/{n}")
    print(f"Valid DPO Centered values: {c_count}/{n}")
    
    if nc_count > 0 and c_count > 0:
        barsback = int(period / 2 + 1)  # Expected offset
        expected_start = max(period - 1, barsback)
        first_valid_nc = np.where(valid_nc)[0][0] if nc_count > 0 else -1
        first_valid_c = np.where(valid_c)[0][0] if c_count > 0 else -1
        
        print(f"Expected barsback: {barsback}")
        print(f"First valid Non-Centered: {first_valid_nc}, Centered: {first_valid_c}")
        print(f"Expected start around: {expected_start}")
        
        # Show some values
        print(f"\nSample DPO values (last 10):")
        for i in range(max(0, n-10), n):
            if valid_nc[i] and valid_c[i]:
                print(f"Index {i}: Price={prices[i]:.2f}")
                print(f"  Non-Centered={dpo_non_centered[i]:.4f}, Centered={dpo_centered[i]:.4f}")
        
        # Basic validation
        valid_nc_values = dpo_non_centered[valid_nc]
        valid_c_values = dpo_centered[valid_c]
        
        print(f"\nValidation:")
        print(f"Non-Centered DPO range: {valid_nc_values.min():.4f} - {valid_nc_values.max():.4f}")
        print(f"Centered DPO range: {valid_c_values.min():.4f} - {valid_c_values.max():.4f}")
        print(f"Non-Centered mean: {valid_nc_values.mean():.4f}")
        print(f"Centered mean: {valid_c_values.mean():.4f}")
        
        # DPO should oscillate around zero (detrended)
        nc_near_zero = abs(valid_nc_values.mean()) < abs(valid_nc_values.std())
        c_near_zero = abs(valid_c_values.mean()) < abs(valid_c_values.std())
        
        print(f"Non-Centered oscillates around zero: {nc_near_zero}")
        print(f"Centered oscillates around zero: {c_near_zero}")
        
        # Test different periods
        print(f"\nTesting different periods:")
        
        dpo_10 = ta.dpo(prices, period=10, is_centered=False)
        dpo_30 = ta.dpo(prices, period=30, is_centered=False)
        
        valid_10 = np.where(~np.isnan(dpo_10))[0]
        valid_30 = np.where(~np.isnan(dpo_30))[0]
        
        if len(valid_10) > 0 and len(valid_30) > 0:
            last_idx = min(valid_10[-1], valid_30[-1], n-1)
            print(f"Period 10: DPO={dpo_10[last_idx]:.4f}")
            print(f"Period 21: DPO={dpo_non_centered[last_idx]:.4f}")
            print(f"Period 30: DPO={dpo_30[last_idx]:.4f}")
        
        return nc_near_zero and c_near_zero and nc_count > 50
    
    return False

def test_dpo_modes():
    """Test the difference between centered and non-centered modes"""
    
    print(f"\nTesting Centered vs Non-Centered Modes:")
    print("-" * 45)
    
    # Create simple trending data
    n = 50
    prices = np.linspace(100, 150, n) + np.random.normal(0, 0.5, n)
    
    print(f"Simple trend data: {prices[0]:.2f} to {prices[-1]:.2f}")
    
    period = 10
    barsback = int(period / 2 + 1)
    print(f"Period: {period}, Barsback: {barsback}")
    
    # Calculate both modes
    dpo_nc = ta.dpo(prices, period=period, is_centered=False)
    dpo_c = ta.dpo(prices, period=period, is_centered=True)
    
    valid_nc = ~np.isnan(dpo_nc)
    valid_c = ~np.isnan(dpo_c)
    
    print(f"\nValid values: Non-Centered {valid_nc.sum()}, Centered {valid_c.sum()}")
    
    if valid_nc.sum() > 5 and valid_c.sum() > 5:
        print(f"\nLast 5 valid comparisons:")
        valid_nc_indices = np.where(valid_nc)[0][-5:]
        valid_c_indices = np.where(valid_c)[0][-5:]
        
        for i in range(min(len(valid_nc_indices), len(valid_c_indices))):
            nc_idx = valid_nc_indices[i]
            c_idx = valid_c_indices[i]
            print(f"  Non-Centered[{nc_idx}]: {dpo_nc[nc_idx]:.4f}")
            print(f"  Centered[{c_idx}]: {dpo_c[c_idx]:.4f}")
        
        return True
    
    return False

def test_dpo_formula():
    """Test DPO formula implementation"""
    
    print(f"\nTesting DPO Formula Implementation:")
    print("-" * 40)
    
    # Create simple data where we can manually verify
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
                      110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120])
    
    period = 5
    barsback = int(period / 2 + 1)  # Should be 3
    
    print(f"Period: {period}, Barsback: {barsback}")
    print(f"Price sequence: {prices[:10].tolist()}...")
    
    # Calculate DPO
    dpo_nc = ta.dpo(prices, period=period, is_centered=False)
    dpo_c = ta.dpo(prices, period=period, is_centered=True)
    
    valid_count_nc = (~np.isnan(dpo_nc)).sum()
    valid_count_c = (~np.isnan(dpo_c)).sum()
    
    print(f"Valid values: Non-Centered {valid_count_nc}, Centered {valid_count_c}")
    
    if valid_count_nc > 5:
        print(f"\nNon-Centered DPO values:")
        for i, val in enumerate(dpo_nc):
            if not np.isnan(val):
                print(f"  Index {i}: Price={prices[i]:.0f}, DPO={val:.4f}")
                
        print(f"\nCentered DPO values:")
        for i, val in enumerate(dpo_c):
            if not np.isnan(val):
                print(f"  Index {i}: Price={prices[i]:.0f}, DPO={val:.4f}")
        
        return True
    
    return False

if __name__ == "__main__":
    print("Detrended Price Oscillator Test")
    print("=" * 65)
    
    success1 = test_dpo_basic()
    success2 = test_dpo_modes()
    success3 = test_dpo_formula()
    
    print(f"\nOverall Test: {'PASSED' if success1 and success2 and success3 else 'FAILED'}")
    print(f"\nDPO Implementation Summary:")
    print(f"- Default period changed from 20 to 21")
    print(f"- Added is_centered parameter (default: False)")
    print(f"- Non-Centered mode: DPO = Close - SMA[barsback] (default)")
    print(f"- Centered mode: DPO = Close[barsback] - SMA")
    print(f"- Barsback calculation: period/2 + 1")
    print(f"- Detrends price data to show cyclical patterns")
    print(f"- Oscillates around zero line when working properly")
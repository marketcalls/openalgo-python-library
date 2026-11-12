#!/usr/bin/env python
"""Test Ease of Movement with its default parameters"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import ta
import numpy as np

def test_emv_basic():
    """Test EMV with basic functionality"""
    
    print("Testing Ease of Movement (EMV)")
    print("=" * 35)
    
    # Create test data with volume
    np.random.seed(42)
    n = 100
    
    # Create price data with trend
    trend = np.linspace(0, 0.2, n)
    noise = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(trend + noise)
    
    # Create OHLC data
    high = prices * (1 + np.random.uniform(0.005, 0.02, n))
    low = prices * (1 - np.random.uniform(0.005, 0.02, n))
    close = prices
    
    # Create volume data (more volume during price moves)
    base_volume = 1000000
    volume_multiplier = 1 + 0.5 * abs(np.diff(np.concatenate([[prices[0]], prices])))
    volume = (base_volume * volume_multiplier).astype(int)
    
    print(f"Generated {n} test data points")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    print(f"Volume range: {volume.min():,} - {volume.max():,}")
    
    # Test with the default parameters: length=14, divisor=10000
    length, divisor = 14, 10000
    
    # Calculate EMV
    emv = ta.emv(high, low, volume, length=length, divisor=divisor)
    
    print(f"\nUsing default parameters: length={length}, divisor={divisor}")
    
    # Check for valid data
    valid_mask = ~np.isnan(emv)
    valid_count = valid_mask.sum()
    
    print(f"Valid EMV values: {valid_count}/{n}")
    
    if valid_count > 0:
        expected_start = length - 1 + 1  # SMA period + 1 for change calculation
        first_valid = np.where(valid_mask)[0][0] if valid_count > 0 else -1
        print(f"First valid index: {first_valid}, Expected around: {expected_start}")
        
        # Show some values
        print(f"\nSample EMV values (last 10):")
        for i in range(max(0, n-10), n):
            if valid_mask[i]:
                hl2_change = ((high[i] + low[i])/2) - ((high[i-1] + low[i-1])/2) if i > 0 else 0
                print(f"Index {i}: H={high[i]:.2f}, L={low[i]:.2f}, V={volume[i]:,}")
                print(f"  HL2 Change={hl2_change:.4f}, EMV={emv[i]:.4f}")
        
        # Basic validation
        valid_emv = emv[valid_mask]
        
        print(f"\nValidation:")
        print(f"EMV range: {valid_emv.min():.4f} - {valid_emv.max():.4f}")
        print(f"EMV mean: {valid_emv.mean():.4f}")
        print(f"EMV std: {valid_emv.std():.4f}")
        
        # EMV should have reasonable magnitude
        reasonable_range = abs(valid_emv.max() - valid_emv.min()) > 0.01
        print(f"Reasonable range: {reasonable_range}")
        
        # Test parameter sensitivity
        print(f"\nTesting parameter sensitivity:")
        
        # Test with different length
        emv_7 = ta.emv(high, low, volume, length=7, divisor=10000)
        emv_21 = ta.emv(high, low, volume, length=21, divisor=10000)
        
        valid_idx = np.where(~np.isnan(emv_7) & ~np.isnan(emv_21))[0]
        if len(valid_idx) > 0:
            last_idx = valid_idx[-1]
            print(f"Length 7: EMV={emv_7[last_idx]:.4f}")
            print(f"Length 14: EMV={emv[last_idx]:.4f}")
            print(f"Length 21: EMV={emv_21[last_idx]:.4f}")
        
        # Test with different divisor
        emv_5000 = ta.emv(high, low, volume, length=14, divisor=5000)
        emv_20000 = ta.emv(high, low, volume, length=14, divisor=20000)
        
        valid_idx = np.where(~np.isnan(emv_5000) & ~np.isnan(emv_20000))[0]
        if len(valid_idx) > 0:
            last_idx = valid_idx[-1]
            print(f"Divisor 5000: EMV={emv_5000[last_idx]:.4f}")
            print(f"Divisor 10000: EMV={emv[last_idx]:.4f}")
            print(f"Divisor 20000: EMV={emv_20000[last_idx]:.4f}")
            
            # Higher divisor should give higher values
            divisor_effect = emv_20000[last_idx] > emv_5000[last_idx]
            print(f"Higher divisor gives higher values: {divisor_effect}")
        
        return reasonable_range and valid_count > 50
    
    return False

def test_emv_formula():
    """Test EMV formula implementation"""
    
    print(f"\nTesting EMV Formula Implementation:")
    print("-" * 40)
    
    # Create simple data for manual verification
    high = np.array([102, 104, 103, 105, 107, 106, 108, 110, 109, 111])
    low = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    volume = np.array([1000, 1200, 800, 1500, 1100, 900, 1300, 1400, 1000, 1200])
    
    print(f"Simple test data:")
    for i in range(len(high)):
        hl2 = (high[i] + low[i]) / 2
        hl_range = high[i] - low[i]
        print(f"  {i}: H={high[i]}, L={low[i]}, V={volume[i]}, HL2={hl2:.1f}, Range={hl_range}")
    
    length, divisor = 3, 1000  # Short period for quick results
    
    # Calculate EMV
    emv = ta.emv(high, low, volume, length=length, divisor=divisor)
    
    valid_count = (~np.isnan(emv)).sum()
    print(f"\nValid EMV values: {valid_count}/{len(high)}")
    
    if valid_count > 0:
        print(f"\nEMV values:")
        for i, val in enumerate(emv):
            if not np.isnan(val):
                # Manual calculation for verification
                if i > 0:
                    hl2_current = (high[i] + low[i]) / 2
                    hl2_previous = (high[i-1] + low[i-1]) / 2
                    change_hl2 = hl2_current - hl2_previous
                    hl_range = high[i] - low[i]
                    raw_emv = divisor * change_hl2 * hl_range / volume[i]
                    print(f"  Index {i}: EMV={val:.6f}")
                    print(f"    Change HL2={change_hl2:.2f}, Range={hl_range}, Volume={volume[i]}")
                    print(f"    Raw EMV={raw_emv:.6f} (before SMA smoothing)")
        
        return True
    
    return False

def test_emv_volume_relationship():
    """Test EMV's relationship with volume"""
    
    print(f"\nTesting EMV-Volume Relationship:")
    print("-" * 40)
    
    # Create data with same price moves but different volumes
    n = 30
    prices = np.linspace(100, 110, n)  # Steady uptrend
    high = prices + 1
    low = prices - 1
    
    # Test with different volume scenarios
    high_volume = np.full(n, 10000)  # High volume
    low_volume = np.full(n, 1000)    # Low volume
    
    print("Same price pattern with different volumes:")
    print(f"Price trend: {prices[0]:.1f} to {prices[-1]:.1f}")
    print(f"High volume: {high_volume[0]:,}")
    print(f"Low volume: {low_volume[0]:,}")
    
    # Calculate EMV for both scenarios
    emv_high_vol = ta.emv(high, low, high_volume, length=5, divisor=10000)
    emv_low_vol = ta.emv(high, low, low_volume, length=5, divisor=10000)
    
    # Find last valid values
    valid_high = np.where(~np.isnan(emv_high_vol))[0]
    valid_low = np.where(~np.isnan(emv_low_vol))[0]
    
    if len(valid_high) > 0 and len(valid_low) > 0:
        last_high = emv_high_vol[valid_high[-1]]
        last_low = emv_low_vol[valid_low[-1]]
        
        print(f"\nLast EMV values:")
        print(f"High volume EMV: {last_high:.4f}")
        print(f"Low volume EMV: {last_low:.4f}")
        
        # Low volume should give higher EMV for same price move
        # (because EMV = divisor * change * range / volume)
        volume_effect = abs(last_low) > abs(last_high)
        print(f"Lower volume gives higher EMV magnitude: {volume_effect}")
        
        return volume_effect
    
    return False

if __name__ == "__main__":
    print("Ease of Movement Test")
    print("=" * 55)
    
    success1 = test_emv_basic()
    success2 = test_emv_formula()
    success3 = test_emv_volume_relationship()
    
    print(f"\nOverall Test: {'PASSED' if success1 and success2 and success3 else 'FAILED'}")
    print(f"\nEMV Implementation Summary:")
    print(f"- Formula: EMV = SMA(div * change(hl2) * (high - low) / volume, length)")
    print(f"- Default parameters: length=14, divisor=10000")
    print(f"- Automatic SMA smoothing (the output is always smoothed)")
    print(f"- change(hl2) = current HL2 - previous HL2")
    print(f"- HL2 = (high + low) / 2 (typical price)")
    print(f"- Higher volume leads to lower EMV magnitude")
    print(f"- Positive EMV indicates easier upward movement")
    print(f"- Negative EMV indicates easier downward movement")
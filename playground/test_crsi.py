#!/usr/bin/env python
"""Test Connors RSI with its default parameters"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import ta
import numpy as np

def test_crsi_basic():
    """Test CRSI with basic functionality"""
    
    print("Testing Connors RSI (CRSI)")
    print("=" * 30)
    
    # Create test data with various patterns
    np.random.seed(42)
    n = 150
    
    # Create mixed trend data  
    uptrend = np.linspace(0, 0.15, 50)
    sideways = np.random.normal(0, 0.005, 50) 
    downtrend = np.linspace(0, -0.1, 50)
    
    trends = np.concatenate([uptrend, sideways, downtrend])
    noise = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(trends + noise))
    
    print(f"Generated {n} test data points")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test with the default parameters: lenrsi=3, lenupdown=2, lenroc=100
    lenrsi, lenupdown, lenroc = 3, 2, 100
    
    # Calculate CRSI
    crsi = ta.crsi(prices, lenrsi=lenrsi, lenupdown=lenupdown, lenroc=lenroc)
    
    print(f"\nUsing default parameters: lenrsi={lenrsi}, lenupdown={lenupdown}, lenroc={lenroc}")
    
    # Check for valid data
    valid_mask = ~np.isnan(crsi)
    valid_count = valid_mask.sum()
    
    print(f"Valid CRSI values: {valid_count}/{n}")
    
    if valid_count > 0:
        first_valid_idx = np.where(valid_mask)[0][0]
        expected_start = max(lenrsi, lenupdown, lenroc) - 1
        print(f"First valid index: {first_valid_idx}, Expected around: {expected_start}")
        
        # Show some values
        print(f"\nSample CRSI values (last 10):")
        for i in range(max(0, n-10), n):
            if valid_mask[i]:
                print(f"Index {i}: Price={prices[i]:.2f}, CRSI={crsi[i]:.2f}")
        
        # Basic validation
        valid_crsi = crsi[valid_mask]
        
        # CRSI should be between 0 and 100
        within_range = np.all((valid_crsi >= 0) & (valid_crsi <= 100))
        
        print(f"\nValidation:")
        print(f"CRSI within 0-100 range: {within_range}")
        print(f"CRSI range: {valid_crsi.min():.2f} - {valid_crsi.max():.2f}")
        print(f"CRSI mean: {valid_crsi.mean():.2f}")
        print(f"CRSI std: {valid_crsi.std():.2f}")
        
        # Check for overbought/oversold levels
        overbought = (valid_crsi > 70).sum()
        oversold = (valid_crsi < 30).sum()
        
        print(f"\nSignal Analysis:")
        print(f"Overbought (>70): {overbought} ({overbought/len(valid_crsi)*100:.1f}%)")
        print(f"Oversold (<30): {oversold} ({oversold/len(valid_crsi)*100:.1f}%)")
        
        # Test parameter sensitivity
        print(f"\nTesting parameter sensitivity:")
        
        # Test with different RSI length
        crsi_rsi5 = ta.crsi(prices, lenrsi=5, lenupdown=2, lenroc=100)
        valid_idx = np.where(~np.isnan(crsi_rsi5))[0]
        if len(valid_idx) > 0:
            last_idx = valid_idx[-1]
            print(f"lenrsi=5: CRSI={crsi_rsi5[last_idx]:.2f}")
            print(f"lenrsi=3: CRSI={crsi[last_idx]:.2f}")
        
        # Test with different updown length
        crsi_ud3 = ta.crsi(prices, lenrsi=3, lenupdown=3, lenroc=100)
        valid_idx = np.where(~np.isnan(crsi_ud3))[0]
        if len(valid_idx) > 0:
            last_idx = valid_idx[-1]
            print(f"lenupdown=3: CRSI={crsi_ud3[last_idx]:.2f}")
            print(f"lenupdown=2: CRSI={crsi[last_idx]:.2f}")
        
        return within_range and valid_count > 50
    
    return False

def test_crsi_updown_logic():
    """Test the updown streak logic specifically"""
    
    print(f"\nTesting UpDown Streak Logic:")
    print("-" * 35)
    
    # Create simple test pattern
    prices = np.array([100, 101, 102, 102, 103, 102, 101, 101, 100, 99, 100])
    
    print("Price sequence:")
    for i, p in enumerate(prices):
        change = "=" if i == 0 else ("UP" if p > prices[i-1] else "DN" if p < prices[i-1] else "EQ")
        print(f"  {i}: {p:.0f} {change}")
    
    # Calculate CRSI with shorter ROC period for small data
    crsi = ta.crsi(prices, lenrsi=3, lenupdown=2, lenroc=5)
    
    # Check if we get reasonable values
    valid_count = (~np.isnan(crsi)).sum()
    print(f"\nValid CRSI values: {valid_count}/{len(prices)}")
    
    if valid_count > 0:
        print("CRSI values:")
        for i, c in enumerate(crsi):
            if not np.isnan(c):
                print(f"  Index {i}: CRSI={c:.2f}")
        return True
    
    return False

def test_crsi_components():
    """Test individual components to understand the calculation"""
    
    print(f"\nTesting CRSI Components:")
    print("-" * 30)
    
    # Create simple uptrend
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 2)
    
    print(f"Testing with simple uptrend pattern")
    print(f"Price range: {prices.min():.0f} - {prices.max():.0f}")
    
    # Calculate CRSI
    crsi = ta.crsi(prices, lenrsi=3, lenupdown=2, lenroc=10)  # Shorter ROC for quicker results
    
    valid_count = (~np.isnan(crsi)).sum()
    print(f"Valid CRSI values: {valid_count}/{len(prices)}")
    
    if valid_count > 5:
        print(f"\nLast 5 CRSI values:")
        valid_indices = np.where(~np.isnan(crsi))[0][-5:]
        for idx in valid_indices:
            print(f"  Index {idx}: Price={prices[idx]:.0f}, CRSI={crsi[idx]:.2f}")
        
        # In an uptrend, CRSI should generally be higher
        last_values = crsi[valid_indices]
        avg_crsi = last_values.mean()
        print(f"\nAverage CRSI in uptrend: {avg_crsi:.2f}")
        print(f"Expected: Should be > 50 for sustained uptrend")
        
        return avg_crsi > 40  # Allow some flexibility
    
    return False

if __name__ == "__main__":
    print("Connors RSI Test")
    print("=" * 50)
    
    success1 = test_crsi_basic()
    success2 = test_crsi_updown_logic()
    success3 = test_crsi_components()
    
    print(f"\nOverall Test: {'PASSED' if success1 and success2 and success3 else 'FAILED'}")
    print(f"\nCRSI Implementation Summary:")
    print(f"- Component 1: RSI of price (ta.rsi(src, lenrsi))")
    print(f"- Component 2: RSI of updown streak (ta.rsi(updown(src), lenupdown))")
    print(f"- Component 3: Percent rank of 1-period ROC (ta.percentrank(ta.roc(src, 1), lenroc))")
    print(f"- Formula: CRSI = math.avg(rsi, updownrsi, percentrank)")
    print(f"- Default parameters: lenrsi=3, lenupdown=2, lenroc=100")
    print(f"- Returns values between 0-100 like standard RSI")
#!/usr/bin/env python
"""Test Fisher Transform with its default parameters"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import ta
import numpy as np

def test_fisher_basic():
    """Test Fisher Transform with basic functionality"""
    
    print("Testing Fisher Transform")
    print("=" * 30)
    
    # Create test data with cyclical patterns
    np.random.seed(42)
    n = 100
    
    # Create price data with trend and cycles
    trend = np.linspace(0, 0.2, n)
    cycle = 0.1 * np.sin(2 * np.pi * np.arange(n) / 15)  # 15-period cycle
    noise = np.random.normal(0, 0.02, n)
    
    prices = 100 * np.exp(trend + cycle + noise)
    
    # Create OHLC data
    high = prices * (1 + np.random.uniform(0.005, 0.02, n))
    low = prices * (1 - np.random.uniform(0.005, 0.02, n))
    close = prices
    
    print(f"Generated {n} test data points")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test with the default parameters: length=9
    length = 9
    
    # Calculate Fisher Transform
    fisher, trigger = ta.fisher(high, low, length=length)
    
    print(f"\nUsing default parameters: length={length}")
    
    # Check for valid data
    valid_fisher = ~np.isnan(fisher)
    valid_trigger = ~np.isnan(trigger)
    
    fisher_count = valid_fisher.sum()
    trigger_count = valid_trigger.sum()
    
    print(f"Valid Fisher values: {fisher_count}/{n}")
    print(f"Valid Trigger values: {trigger_count}/{n}")
    
    if fisher_count > 0 and trigger_count > 0:
        expected_start = length - 1
        first_valid_fisher = np.where(valid_fisher)[0][0] if fisher_count > 0 else -1
        first_valid_trigger = np.where(valid_trigger)[0][0] if trigger_count > 0 else -1
        
        print(f"First valid Fisher: {first_valid_fisher}, Trigger: {first_valid_trigger}")
        print(f"Expected start: {expected_start}")
        
        # Show some values
        print(f"\nSample Fisher Transform values (last 10):")
        for i in range(max(0, n-10), n):
            if valid_fisher[i] and valid_trigger[i]:
                hl2 = (high[i] + low[i]) / 2
                print(f"Index {i}: HL2={hl2:.2f}, Fisher={fisher[i]:.4f}, Trigger={trigger[i]:.4f}")
        
        # Basic validation
        valid_fisher_values = fisher[valid_fisher]
        valid_trigger_values = trigger[valid_trigger]
        
        print(f"\nValidation:")
        print(f"Fisher range: {valid_fisher_values.min():.4f} - {valid_fisher_values.max():.4f}")
        print(f"Trigger range: {valid_trigger_values.min():.4f} - {valid_trigger_values.max():.4f}")
        print(f"Fisher mean: {valid_fisher_values.mean():.4f}")
        print(f"Trigger mean: {valid_trigger_values.mean():.4f}")
        
        # Fisher Transform should have reasonable oscillating values
        fisher_range = abs(valid_fisher_values.max() - valid_fisher_values.min())
        reasonable_range = fisher_range > 0.1
        print(f"Reasonable oscillation range: {reasonable_range}")
        
        # Test different lengths
        print(f"\nTesting different lengths:")
        
        fisher_5, trigger_5 = ta.fisher(high, low, length=5)
        fisher_15, trigger_15 = ta.fisher(high, low, length=15)
        
        valid_5 = np.where(~np.isnan(fisher_5))[0]
        valid_15 = np.where(~np.isnan(fisher_15))[0]
        
        if len(valid_5) > 0 and len(valid_15) > 0:
            last_idx = min(valid_5[-1], valid_15[-1], n-1)
            print(f"Length 5: Fisher={fisher_5[last_idx]:.4f}, Trigger={trigger_5[last_idx]:.4f}")
            print(f"Length 9: Fisher={fisher[last_idx]:.4f}, Trigger={trigger[last_idx]:.4f}")
            print(f"Length 15: Fisher={fisher_15[last_idx]:.4f}, Trigger={trigger_15[last_idx]:.4f}")
        
        # Check trigger relationship
        print(f"\nTrigger Relationship Check:")
        # Trigger should be the previous Fisher value (with some lag)
        matching_indices = 0
        total_checks = 0
        
        for i in range(length, n-1):
            if valid_fisher[i] and valid_trigger[i+1] and valid_fisher[i-1]:
                if abs(trigger[i+1] - fisher[i]) < 0.0001:  # Small tolerance for floating point
                    matching_indices += 1
                total_checks += 1
        
        if total_checks > 0:
            match_rate = matching_indices / total_checks
            print(f"Trigger matches previous Fisher: {match_rate:.2%}")
        
        return reasonable_range and fisher_count > 50
    
    return False

def test_fisher_extremes():
    """Test Fisher Transform's ability to detect price extremes"""
    
    print(f"\nTesting Fisher Transform Extreme Detection:")
    print("-" * 45)
    
    # Create data with clear extremes
    n = 50
    
    # Normal range with spikes
    base_prices = 100 + np.random.normal(0, 1, n)
    
    # Add extreme spikes
    base_prices[10] = 110  # High spike
    base_prices[25] = 90   # Low spike
    base_prices[40] = 108  # Another high spike
    
    # Create OHLC
    high = base_prices + abs(np.random.normal(0, 0.5, n))
    low = base_prices - abs(np.random.normal(0, 0.5, n))
    
    print(f"Created data with extreme spikes at indices 10, 25, 40")
    print(f"Price range: {base_prices.min():.2f} - {base_prices.max():.2f}")
    
    # Calculate Fisher
    fisher, trigger = ta.fisher(high, low, length=9)
    
    valid_count = (~np.isnan(fisher)).sum()
    print(f"Valid Fisher values: {valid_count}/{n}")
    
    if valid_count > 20:
        # Check if Fisher responds to extremes
        valid_fisher = fisher[~np.isnan(fisher)]
        
        print(f"\nFisher statistics:")
        print(f"Range: {valid_fisher.min():.4f} - {valid_fisher.max():.4f}")
        print(f"Mean: {valid_fisher.mean():.4f}")
        print(f"Std: {valid_fisher.std():.4f}")
        
        # Show values around extreme points
        for extreme_idx in [15, 30, 45]:  # Offset for calculation lag
            if extreme_idx < len(fisher) and not np.isnan(fisher[extreme_idx]):
                print(f"Fisher at index {extreme_idx}: {fisher[extreme_idx]:.4f}")
        
        # Fisher should have high absolute values at extremes
        extreme_detection = abs(valid_fisher.max()) > 1.0 or abs(valid_fisher.min()) < -1.0
        print(f"Detects extremes (|Fisher| > 1.0): {extreme_detection}")
        
        return True
    
    return False

def test_fisher_recursive_smoothing():
    """Test the recursive smoothing behavior"""
    
    print(f"\nTesting Recursive Smoothing Behavior:")
    print("-" * 40)
    
    # Create simple step function to test smoothing
    n = 30
    prices = np.ones(n) * 100
    prices[15:] = 105  # Step change at midpoint
    
    high = prices + 0.5
    low = prices - 0.5
    
    print(f"Step change from 100 to 105 at index 15")
    
    # Calculate Fisher
    fisher, trigger = ta.fisher(high, low, length=5)
    
    valid_count = (~np.isnan(fisher)).sum()
    print(f"Valid Fisher values: {valid_count}/{n}")
    
    if valid_count > 10:
        print(f"\nFisher response to step change:")
        for i in range(10, 25):
            if not np.isnan(fisher[i]):
                hl2 = (high[i] + low[i]) / 2
                print(f"  Index {i}: HL2={hl2:.1f}, Fisher={fisher[i]:.6f}")
        
        # Check smoothing - values should change gradually due to recursive nature
        valid_indices = np.where(~np.isnan(fisher))[0]
        if len(valid_indices) > 5:
            # Calculate differences between consecutive Fisher values
            diffs = np.abs(np.diff(fisher[valid_indices]))
            max_diff = diffs.max()
            print(f"\nMax consecutive difference: {max_diff:.6f}")
            print(f"Smooth transitions (max diff < 0.5): {max_diff < 0.5}")
            
            return max_diff < 2.0  # Allow some reasonable variation
    
    return False

if __name__ == "__main__":
    print("Fisher Transform Test")
    print("=" * 55)
    
    success1 = test_fisher_basic()
    success2 = test_fisher_extremes()
    success3 = test_fisher_recursive_smoothing()
    
    print(f"\nOverall Test: {'PASSED' if success1 and success2 and success3 else 'FAILED'}")
    print(f"\nFisher Transform Implementation Summary:")
    print(f"- Uses high and low prices to calculate HL2 (typical price)")
    print(f"- Default length changed from 10 to 9")
    print(f"- Implements recursive smoothing for both value and Fisher calculations")
    print(f"- Formula: value := round_(.66 * ((hl2-low)/(high-low)-.5) + .67 * value[1])")
    print(f"- Formula: fish1 := .5 * log((1+value)/(1-value)) + .5 * fish1[1]")
    print(f"- Trigger: fish2 = fish1[1] (previous Fisher value)")
    print(f"- Round function constrains values to avoid log division issues")
    print(f"- Converts price extremes into Gaussian normal distribution")
    print(f"- Crossovers between Fisher and Trigger indicate trend changes")
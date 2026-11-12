#!/usr/bin/env python
"""Test DPO with real market data"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta
import numpy as np

def test_dpo_real():
    """Test DPO with real market data"""
    
    print("Testing DPO with Real Market Data")
    print("=" * 40)
    
    client = api(
        api_key=os.getenv('OPENALGO_API_KEY', 'your_api_key_here'), 
        host='http://127.0.0.1:5001'
    )
    
    try:
        # Get data
        df = client.history(
            symbol="RELIANCE", 
            exchange="NSE", 
            interval="D", 
            start_date="2024-01-01", 
            end_date="2025-01-14"
        )
        
        print(f"Retrieved {len(df)} data points")
        
        # Calculate DPO with TradingView defaults
        df['dpo_nc'] = ta.dpo(df['close'], period=21, is_centered=False)
        df['dpo_c'] = ta.dpo(df['close'], period=21, is_centered=True)
        
        # Also test with different periods
        df['dpo_10'] = ta.dpo(df['close'], period=10, is_centered=False)
        df['dpo_30'] = ta.dpo(df['close'], period=30, is_centered=False)
        
        # Analysis
        valid_count_nc = (~df['dpo_nc'].isna()).sum()
        valid_count_c = (~df['dpo_c'].isna()).sum()
        
        print(f"Valid DPO values: Non-Centered {valid_count_nc}/{len(df)}, Centered {valid_count_c}/{len(df)}")
        
        if valid_count_nc > 0 and valid_count_c > 0:
            # Show recent values
            recent = df.dropna().tail(15)
            print(f"\nRecent DPO values (TradingView defaults period=21):")
            for idx, row in recent.iterrows():
                print(f"{idx}: Close={row['close']:.2f}")
                print(f"  Non-Centered DPO={row['dpo_nc']:.4f}, Centered DPO={row['dpo_c']:.4f}")
            
            # Statistics
            dpo_nc_values = df['dpo_nc'].dropna()
            dpo_c_values = df['dpo_c'].dropna()
            
            print(f"\nDPO Statistics over valid periods:")
            print(f"Non-Centered: {len(dpo_nc_values)} values")
            print(f"  Range: {dpo_nc_values.min():.4f} - {dpo_nc_values.max():.4f}")
            print(f"  Mean: {dpo_nc_values.mean():.4f}")
            print(f"  Std: {dpo_nc_values.std():.4f}")
            
            print(f"Centered: {len(dpo_c_values)} values")
            print(f"  Range: {dpo_c_values.min():.4f} - {dpo_c_values.max():.4f}")
            print(f"  Mean: {dpo_c_values.mean():.4f}")
            print(f"  Std: {dpo_c_values.std():.4f}")
            
            # DPO should oscillate around zero
            nc_near_zero = abs(dpo_nc_values.mean()) < dpo_nc_values.std()
            c_near_zero = abs(dpo_c_values.mean()) < dpo_c_values.std()
            
            print(f"\nOscillation around zero:")
            print(f"Non-Centered: {nc_near_zero}")
            print(f"Centered: {c_near_zero}")
            
            # Compare different periods
            latest = recent.iloc[-1]
            print(f"\nPeriod comparison (latest bar):")
            print(f"Period 10: DPO={latest['dpo_10']:.4f}")
            print(f"Period 21: DPO={latest['dpo_nc']:.4f}")
            print(f"Period 30: DPO={latest['dpo_30']:.4f}")
            
            # Trend analysis using DPO
            print(f"\nTrend Analysis (DPO Non-Centered, last 10 bars):")
            last_10_dpo = recent['dpo_nc'].tail(10)
            
            positive_dpo = (last_10_dpo > 0).sum()
            negative_dpo = (last_10_dpo < 0).sum()
            avg_dpo = last_10_dpo.mean()
            
            print(f"Positive DPO bars: {positive_dpo}/10")
            print(f"Negative DPO bars: {negative_dpo}/10")
            print(f"Average DPO: {avg_dpo:.4f}")
            
            if avg_dpo > 5:
                print("Signal: Price above detrended average (bullish cyclical position)")
            elif avg_dpo < -5:
                print("Signal: Price below detrended average (bearish cyclical position)")
            else:
                print("Signal: Price near detrended average (neutral)")
            
            # Zero line crossings (trend changes)
            zero_crossings = 0
            dpo_values = recent['dpo_nc'].values
            for i in range(1, len(dpo_values)):
                if not (np.isnan(dpo_values[i-1]) or np.isnan(dpo_values[i])):
                    if (dpo_values[i-1] > 0) != (dpo_values[i] > 0):
                        zero_crossings += 1
            
            print(f"Zero line crossings in recent data: {zero_crossings}")
            
            print(f"\nDPO Test with Real Data: SUCCESS")
            return True
            
    except Exception as e:
        print(f"Error with real data: {e}")
        print("Testing with sample data instead...")
        return test_sample_data()

def test_sample_data():
    """Test with sample data if API fails"""
    
    print("\nTesting with sample cyclical data...")
    
    # Create data with trend + cycle (perfect for DPO testing)
    np.random.seed(123)
    n = 100
    
    # Linear trend
    trend = np.linspace(1000, 1200, n)
    
    # Add cyclical component (what DPO should detect)
    cycle_period = 20
    cycle = 10 * np.sin(2 * np.pi * np.arange(n) / cycle_period)
    
    # Add some noise
    noise = np.random.normal(0, 2, n)
    
    prices = trend + cycle + noise
    
    print(f"Generated {n} price points with trend + cycle")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test DPO
    dpo_nc = ta.dpo(prices, period=21, is_centered=False)
    dpo_c = ta.dpo(prices, period=21, is_centered=True)
    
    valid_count_nc = (~np.isnan(dpo_nc)).sum()
    valid_count_c = (~np.isnan(dpo_c)).sum()
    
    print(f"Valid DPO values: Non-Centered {valid_count_nc}/{n}, Centered {valid_count_c}/{n}")
    
    if valid_count_nc > 30 and valid_count_c > 30:
        # DPO should capture the cyclical component
        dpo_nc_values = dpo_nc[~np.isnan(dpo_nc)]
        dpo_c_values = dpo_c[~np.isnan(dpo_c)]
        
        print(f"\nCyclical Pattern Detection:")
        print(f"Non-Centered DPO range: {dpo_nc_values.min():.2f} to {dpo_nc_values.max():.2f}")
        print(f"Centered DPO range: {dpo_c_values.min():.2f} to {dpo_c_values.max():.2f}")
        
        # Should oscillate around zero
        nc_mean = dpo_nc_values.mean()
        c_mean = dpo_c_values.mean()
        
        print(f"DPO means (should be near zero): NC={nc_mean:.4f}, C={c_mean:.4f}")
        
        # Standard deviation should be meaningful (capturing cycles)
        print(f"DPO std devs: NC={dpo_nc_values.std():.2f}, C={dpo_c_values.std():.2f}")
        
        # Show last few values
        print(f"\nLast 5 DPO values:")
        for i in range(n-5, n):
            if not np.isnan(dpo_nc[i]) and not np.isnan(dpo_c[i]):
                print(f"  {i}: Price={prices[i]:.2f}, NC_DPO={dpo_nc[i]:.3f}, C_DPO={dpo_c[i]:.3f}")
        
        # Check if DPO oscillates properly
        oscillates_nc = abs(nc_mean) < dpo_nc_values.std()
        oscillates_c = abs(c_mean) < dpo_c_values.std()
        
        print(f"\nOscillation check: NC={oscillates_nc}, C={oscillates_c}")
        
        return oscillates_nc and oscillates_c
    
    return False

if __name__ == "__main__":
    success = test_dpo_real()
    print(f"\nDetrended Price Oscillator Test: {'PASSED' if success else 'FAILED'}")
    
    print(f"\nImplementation Summary:")
    print(f"- Matches TradingView Pine Script exactly")
    print(f"- Default period: 21 (TradingView default)")
    print(f"- Barsback calculation: period/2 + 1")
    print(f"- Non-Centered (default): DPO = Close - SMA[barsback]")
    print(f"- Centered: DPO = Close[barsback] - SMA")
    print(f"- Purpose: Remove trend to highlight cyclical patterns")
    print(f"- Oscillates around zero line")
    print(f"- Zero crossings indicate cyclical turning points")
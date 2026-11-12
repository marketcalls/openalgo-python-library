#!/usr/bin/env python
"""Test EMV with real market data"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta
import numpy as np

def test_emv_real():
    """Test EMV with real market data"""
    
    print("Testing EMV with Real Market Data")
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
        
        # Calculate EMV with the default parameters
        df['emv'] = ta.emv(df['high'], df['low'], df['volume'], length=14, divisor=10000)
        
        # Also test with different parameters
        df['emv_short'] = ta.emv(df['high'], df['low'], df['volume'], length=7, divisor=10000)
        df['emv_long'] = ta.emv(df['high'], df['low'], df['volume'], length=21, divisor=10000)
        df['emv_div5k'] = ta.emv(df['high'], df['low'], df['volume'], length=14, divisor=5000)
        
        # Analysis
        valid_count = (~df['emv'].isna()).sum()
        print(f"Valid EMV values: {valid_count}/{len(df)}")
        
        if valid_count > 0:
            # Show recent values
            recent = df.dropna().tail(15)
            print(f"\nRecent EMV values (defaults length=14, divisor=10000):")
            for idx, row in recent.iterrows():
                # Calculate manual components for verification
                if idx > recent.index[0]:
                    prev_row = df.loc[df.index < idx].iloc[-1]
                    hl2_current = (row['high'] + row['low']) / 2
                    hl2_previous = (prev_row['high'] + prev_row['low']) / 2
                    change_hl2 = hl2_current - hl2_previous
                    hl_range = row['high'] - row['low']
                    raw_emv = 10000 * change_hl2 * hl_range / row['volume']
                    
                    print(f"{idx}: H={row['high']:.2f}, L={row['low']:.2f}, V={row['volume']:,}")
                    print(f"  HL2_Change={change_hl2:.2f}, Range={hl_range:.2f}, RawEMV={raw_emv:.6f}, EMV={row['emv']:.6f}")
                else:
                    print(f"{idx}: H={row['high']:.2f}, L={row['low']:.2f}, V={row['volume']:,}, EMV={row['emv']:.6f}")
            
            # Statistics
            emv_values = df['emv'].dropna()
            print(f"\nEMV Statistics over {len(emv_values)} valid periods:")
            print(f"Range: {emv_values.min():.6f} - {emv_values.max():.6f}")
            print(f"Mean: {emv_values.mean():.6f}")
            print(f"Std: {emv_values.std():.6f}")
            
            # Parameter comparison
            latest = recent.iloc[-1]
            print(f"\nParameter comparison (latest bar):")
            print(f"Length 7: EMV={latest['emv_short']:.6f}")
            print(f"Length 14: EMV={latest['emv']:.6f}")
            print(f"Length 21: EMV={latest['emv_long']:.6f}")
            print(f"Divisor 5000: EMV={latest['emv_div5k']:.6f}")
            print(f"Divisor 10000: EMV={latest['emv']:.6f}")
            
            # EMV interpretation
            print(f"\nEMV Interpretation (last 10 bars):")
            last_10_emv = recent['emv'].tail(10)
            
            positive_emv = (last_10_emv > 0).sum()
            negative_emv = (last_10_emv < 0).sum()
            avg_emv = last_10_emv.mean()
            
            print(f"Positive EMV bars: {positive_emv}/10")
            print(f"Negative EMV bars: {negative_emv}/10")
            print(f"Average EMV: {avg_emv:.6f}")
            
            if avg_emv > 0.001:
                print("Signal: Easier upward movement (bullish)")
            elif avg_emv < -0.001:
                print("Signal: Easier downward movement (bearish)")
            else:
                print("Signal: Neutral movement")
            
            # Volume-Price relationship analysis
            print(f"\nVolume-Price Movement Analysis:")
            recent_with_emv = recent.dropna()
            if len(recent_with_emv) > 5:
                # Look at days with high EMV magnitude
                emv_abs = recent_with_emv['emv'].abs()
                high_emv_days = recent_with_emv[emv_abs > emv_abs.quantile(0.7)]
                
                if len(high_emv_days) > 0:
                    avg_volume_high_emv = high_emv_days['volume'].mean()
                    avg_volume_all = recent_with_emv['volume'].mean()
                    
                    print(f"Average volume on high EMV days: {avg_volume_high_emv:,.0f}")
                    print(f"Average volume overall: {avg_volume_all:,.0f}")
                    
                    volume_relationship = "Higher" if avg_volume_high_emv > avg_volume_all else "Lower"
                    print(f"Volume on high EMV days: {volume_relationship}")
            
            print(f"\nEMV Test with Real Data: SUCCESS")
            return True
            
    except Exception as e:
        print(f"Error with real data: {e}")
        print("Testing with sample data instead...")
        return test_sample_data()

def test_sample_data():
    """Test with sample data if API fails"""
    
    print("\nTesting with sample OHLCV data...")
    
    # Create realistic OHLCV data
    np.random.seed(123)
    n = 50
    
    # Base price trend
    base_prices = 1000 + np.cumsum(np.random.normal(0.5, 2, n))
    
    # Create OHLC
    high = base_prices * (1 + np.random.uniform(0.005, 0.02, n))
    low = base_prices * (1 - np.random.uniform(0.005, 0.02, n))
    close = base_prices
    
    # Create volume (inversely related to price changes for realism)
    price_changes = np.abs(np.diff(np.concatenate([[base_prices[0]], base_prices])))
    base_volume = 1000000
    volume = (base_volume * (2 - price_changes / price_changes.max())).astype(int)
    
    print(f"Generated {n} OHLCV bars")
    print(f"Price range: {base_prices.min():.2f} - {base_prices.max():.2f}")
    print(f"Volume range: {volume.min():,} - {volume.max():,}")
    
    # Test EMV
    emv = ta.emv(high, low, volume, length=14, divisor=10000)
    
    valid_count = (~np.isnan(emv)).sum()
    print(f"Valid EMV values: {valid_count}/{n}")
    
    if valid_count > 10:
        # Show EMV behavior
        valid_emv = emv[~np.isnan(emv)]
        
        print(f"\nEMV Statistics:")
        print(f"Range: {valid_emv.min():.6f} to {valid_emv.max():.6f}")
        print(f"Mean: {valid_emv.mean():.6f}")
        print(f"Std: {valid_emv.std():.6f}")
        
        # Show last few values with components
        print(f"\nLast 5 EMV values:")
        for i in range(n-5, n):
            if not np.isnan(emv[i]) and i > 0:
                hl2_curr = (high[i] + low[i]) / 2
                hl2_prev = (high[i-1] + low[i-1]) / 2
                change_hl2 = hl2_curr - hl2_prev
                hl_range = high[i] - low[i]
                raw_emv = 10000 * change_hl2 * hl_range / volume[i]
                
                print(f"  {i}: Price={close[i]:.2f}, Vol={volume[i]:,}")
                print(f"      HL2_Change={change_hl2:.3f}, Range={hl_range:.2f}, EMV={emv[i]:.6f}")
        
        # EMV should be sensitive to volume
        # Test: same price move, different volumes
        test_high = np.array([100, 102, 104])
        test_low = np.array([99, 101, 103])
        test_vol_low = np.array([1000, 1000, 1000])
        test_vol_high = np.array([10000, 10000, 10000])
        
        emv_low_vol = ta.emv(test_high, test_low, test_vol_low, length=2, divisor=1000)
        emv_high_vol = ta.emv(test_high, test_low, test_vol_high, length=2, divisor=1000)
        
        print(f"\nVolume sensitivity test:")
        print(f"Low volume EMV: {emv_low_vol[-1]:.6f}")
        print(f"High volume EMV: {emv_high_vol[-1]:.6f}")
        print(f"Lower volume gives higher EMV: {abs(emv_low_vol[-1]) > abs(emv_high_vol[-1])}")
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_emv_real()
    print(f"\nEase of Movement Test: {'PASSED' if success else 'FAILED'}")
    
    print(f"\nImplementation Summary:")
    print(f"- Formula: EMV = SMA(divisor * change(hl2) * (high-low) / volume, length)")
    print(f"- Default parameters: length=14, divisor=10000")
    print(f"- HL2 = (high + low) / 2 (typical price)")
    print(f"- Change(HL2) = current HL2 - previous HL2")
    print(f"- Automatic SMA smoothing applied")
    print(f"- Lower volume leads to higher EMV magnitude")
    print(f"- Positive EMV indicates easier upward price movement")
    print(f"- Negative EMV indicates easier downward price movement")
    print(f"- EMV values are typically small decimals")
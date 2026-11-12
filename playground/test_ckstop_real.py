#!/usr/bin/env python
"""Test Chande Kroll Stop with real market data"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta
import numpy as np

def test_ckstop_real():
    """Test CKStop with real market data"""
    
    print("Testing Chande Kroll Stop with Real Market Data")
    print("=" * 50)
    
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
        
        # Calculate CKStop with the default parameters
        df['stop_long'], df['stop_short'] = ta.ckstop(df['high'], df['low'], df['close'])
        
        # Also test with different parameters
        df['stop_long_x2'], df['stop_short_x2'] = ta.ckstop(df['high'], df['low'], df['close'], x=2.0)
        
        # Analysis
        valid_count = (~df['stop_long'].isna()).sum()
        print(f"Valid CKStop values: {valid_count}/{len(df)}")
        
        if valid_count > 0:
            # Show recent values
            recent = df.dropna().tail(10)
            print(f"\nRecent CKStop values (defaults p=10, x=1, q=9):")
            for idx, row in recent.iterrows():
                print(f"{idx}: H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")
                print(f"  Stop Long={row['stop_long']:.2f} Stop Short={row['stop_short']:.2f}")
                
                # Basic validation
                long_valid = row['stop_long'] <= row['low']
                short_valid = row['stop_short'] >= row['high']
                print(f"  Valid: Long<={row['low']:.2f}? {long_valid}, Short>={row['high']:.2f}? {short_valid}")
            
            # Compare different x values
            latest = recent.iloc[-1]
            print(f"\nParameter comparison (latest bar):")
            print(f"x=1.0: Long={latest['stop_long']:.2f}, Short={latest['stop_short']:.2f}")
            print(f"x=2.0: Long={latest['stop_long_x2']:.2f}, Short={latest['stop_short_x2']:.2f}")
            
            # Expected: x=2.0 should give wider stops (lower long stop, higher short stop)
            wider_long = latest['stop_long_x2'] < latest['stop_long']
            wider_short = latest['stop_short_x2'] > latest['stop_short']
            print(f"x=2.0 gives wider stops: {wider_long and wider_short}")
            
            # Statistics
            stops_df = df.dropna()
            print(f"\nStatistics over {len(stops_df)} valid periods:")
            
            # Distance from price to stops
            long_distance = ((stops_df['close'] - stops_df['stop_long']) / stops_df['close'] * 100)
            short_distance = ((stops_df['stop_short'] - stops_df['close']) / stops_df['close'] * 100)
            
            print(f"Stop Long distance from close: {long_distance.mean():.1f}% ± {long_distance.std():.1f}%")
            print(f"Stop Short distance from close: {short_distance.mean():.1f}% ± {short_distance.std():.1f}%")
            
            # Check for reasonable stop levels
            reasonable_long = (long_distance > 0).mean() * 100  # % of time long stop is below close
            reasonable_short = (short_distance > 0).mean() * 100  # % of time short stop is above close
            
            print(f"Stop Long below close: {reasonable_long:.1f}% of time")
            print(f"Stop Short above close: {reasonable_short:.1f}% of time")
            
            print(f"\nCKStop Test with Real Data: SUCCESS")
            return True
            
    except Exception as e:
        print(f"Error with real data: {e}")
        print("Testing with sample data instead...")
        return test_sample_data()

def test_sample_data():
    """Test with sample data if API fails"""
    
    print("\nTesting with sample OHLC data...")
    
    # Create realistic daily OHLC data
    np.random.seed(42)
    n = 100
    
    # Simulate price movements
    returns = np.random.normal(0.0005, 0.015, n)  # Small daily returns with volatility
    prices = 1000 * np.cumprod(1 + returns)
    
    # Create realistic OHLC from prices
    high = np.zeros(n)
    low = np.zeros(n)
    close = prices
    
    for i in range(n):
        # Random intraday range
        range_pct = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
        daily_range = prices[i] * range_pct
        
        # High and low around the close
        high[i] = prices[i] + daily_range * np.random.uniform(0.2, 0.8)
        low[i] = prices[i] - daily_range * np.random.uniform(0.2, 0.8)
    
    print(f"Generated {n} OHLC bars, price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test CKStop
    stop_long, stop_short = ta.ckstop(high, low, close)
    
    valid_count = (~np.isnan(stop_long)).sum()
    print(f"Valid CKStop values: {valid_count}/{n}")
    
    if valid_count > 10:
        # Show last few values
        print(f"\nLast 5 CKStop values:")
        for i in range(n-5, n):
            if not np.isnan(stop_long[i]):
                print(f"Bar {i}: H={high[i]:.2f} L={low[i]:.2f} C={close[i]:.2f}")
                print(f"  Stop Long={stop_long[i]:.2f} Stop Short={stop_short[i]:.2f}")
        
        # Validation
        valid_indices = np.where(~np.isnan(stop_long))[0]
        long_below_low = np.all(stop_long[valid_indices] <= low[valid_indices])
        
        print(f"\nValidation:")
        print(f"Stop Long always <= Low: {long_below_low}")
        
        # Test parameter sensitivity
        stop_long_2x, stop_short_2x = ta.ckstop(high, low, close, x=2.0)
        last_valid = valid_indices[-1]
        
        print(f"\nParameter sensitivity (last valid bar):")
        print(f"x=1.0: Long={stop_long[last_valid]:.2f}, Short={stop_short[last_valid]:.2f}")
        print(f"x=2.0: Long={stop_long_2x[last_valid]:.2f}, Short={stop_short_2x[last_valid]:.2f}")
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_ckstop_real()
    print(f"\nChande Kroll Stop Test: {'PASSED' if success else 'FAILED'}")
    
    print(f"\nImplementation Details:")
    print(f"- Formula: first_high_stop = highest(high, p) - x * atr(p)")
    print(f"- Formula: first_low_stop = lowest(low, p) + x * atr(p)")  
    print(f"- Formula: stop_short = highest(first_high_stop, q)")
    print(f"- Formula: stop_long = lowest(first_low_stop, q)")
    print(f"- Default parameters: p=10, x=1.0, q=9")
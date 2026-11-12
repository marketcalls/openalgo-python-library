#!/usr/bin/env python
"""Test fixed Aroon with real data"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta
import numpy as np

def test_aroon_real_data():
    """Test Aroon with real market data"""
    
    print("Testing Fixed Aroon with Real Data")
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
        
        # Calculate Aroon with default period (14)
        df['aroon_up'], df['aroon_down'] = ta.aroon(df['high'], df['low'])
        
        # Analysis
        valid_count = (~df['aroon_up'].isna()).sum()
        print(f"Valid Aroon values: {valid_count}/{len(df)}")
        
        if valid_count > 0:
            # Show recent values
            recent = df.dropna().tail(10)
            print(f"\nRecent Aroon values:")
            print(recent[['high', 'low', 'close', 'aroon_up', 'aroon_down']].round(2))
            
            # Trend analysis
            latest = recent.iloc[-1]
            print(f"\nLatest Aroon Analysis:")
            print(f"Aroon Up: {latest['aroon_up']:.1f}%")
            print(f"Aroon Down: {latest['aroon_down']:.1f}%")
            
            if latest['aroon_up'] > latest['aroon_down']:
                strength = "Strong" if latest['aroon_up'] > 70 else "Weak"
                print(f"Trend: {strength} Uptrend")
            else:
                strength = "Strong" if latest['aroon_down'] > 70 else "Weak" 
                print(f"Trend: {strength} Downtrend")
            
            # Find strong trend periods
            strong_up = df[df['aroon_up'] > 70]
            strong_down = df[df['aroon_down'] > 70]
            
            print(f"\nTrend Statistics:")
            print(f"Strong uptrend periods: {len(strong_up)} ({len(strong_up)/len(df)*100:.1f}%)")
            print(f"Strong downtrend periods: {len(strong_down)} ({len(strong_down)/len(df)*100:.1f}%)")
            
            print("\nAroon test successful!")
            
        else:
            print("No valid Aroon values calculated!")
            
    except Exception as e:
        print(f"Error: {e}")
        test_sample_data()

def test_sample_data():
    """Test with sample data if API fails"""
    
    print("\nTesting with sample data...")
    
    # Create trending data
    np.random.seed(42)
    n = 50
    
    # Uptrend
    trend = np.linspace(0, 0.2, n)
    noise = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(trend + noise))
    
    high = prices * (1 + np.random.uniform(0.001, 0.015, n))
    low = prices * (1 - np.random.uniform(0.001, 0.015, n))
    
    # Calculate Aroon
    aroon_up, aroon_down = ta.aroon(high, low, period=14)
    
    valid_count = (~np.isnan(aroon_up)).sum()
    print(f"Valid Aroon values: {valid_count}/{n}")
    
    if valid_count > 0:
        print(f"Sample results (last 5):")
        for i in range(-5, 0):
            if not np.isnan(aroon_up[i]):
                print(f"  Up: {aroon_up[i]:.1f}%, Down: {aroon_down[i]:.1f}%")
        
        avg_up = np.nanmean(aroon_up)
        avg_down = np.nanmean(aroon_down)
        print(f"Average - Up: {avg_up:.1f}%, Down: {avg_down:.1f}%")
        print(f"Expected: Up > Down in uptrend - {'OK' if avg_up > avg_down else 'FAIL'}")

if __name__ == "__main__":
    test_aroon_real_data()
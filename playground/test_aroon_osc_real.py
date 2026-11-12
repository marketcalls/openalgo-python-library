#!/usr/bin/env python
"""Test Aroon Oscillator with real market data"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta
import numpy as np

def test_aroon_oscillator_real():
    """Test Aroon Oscillator with real data"""
    
    print("Testing Aroon Oscillator with Real Data")
    print("=" * 45)
    
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
        
        # Calculate Aroon Oscillator with default period (14)
        df['aroon_osc'] = ta.aroon_oscillator(df['high'], df['low'])
        
        # Also calculate individual components for comparison
        df['aroon_up'], df['aroon_down'] = ta.aroon(df['high'], df['low'])
        
        # Verify oscillator = up - down
        df['manual_osc'] = df['aroon_up'] - df['aroon_down']
        
        # Analysis
        valid_count = (~df['aroon_osc'].isna()).sum()
        print(f"Valid Aroon Oscillator values: {valid_count}/{len(df)}")
        
        if valid_count > 0:
            # Verify consistency
            diff = np.abs(df['aroon_osc'] - df['manual_osc']).max()
            print(f"Max difference between direct and manual calculation: {diff:.10f}")
            
            # Show recent values
            recent = df.dropna().tail(10)
            print(f"\nRecent Aroon Oscillator values:")
            print(recent[['high', 'low', 'close', 'aroon_up', 'aroon_down', 'aroon_osc']].round(2))
            
            # Analysis
            latest = recent.iloc[-1]
            print(f"\nLatest Analysis:")
            print(f"Aroon Up: {latest['aroon_up']:.1f}%")
            print(f"Aroon Down: {latest['aroon_down']:.1f}%")
            print(f"Aroon Oscillator: {latest['aroon_osc']:.1f}")
            
            # Interpretation
            if latest['aroon_osc'] > 20:
                print("Signal: Strong Uptrend")
            elif latest['aroon_osc'] < -20:
                print("Signal: Strong Downtrend") 
            else:
                print("Signal: Sideways/Weak Trend")
            
            # Statistics
            osc_values = df['aroon_osc'].dropna()
            print(f"\nOscillator Statistics:")
            print(f"Min: {osc_values.min():.1f}")
            print(f"Max: {osc_values.max():.1f}")
            print(f"Mean: {osc_values.mean():.1f}")
            print(f"Std: {osc_values.std():.1f}")
            
            # Signal analysis
            strong_up = (osc_values > 50).sum()
            strong_down = (osc_values < -50).sum()
            neutral = ((osc_values >= -20) & (osc_values <= 20)).sum()
            
            print(f"\nSignal Distribution:")
            print(f"Strong uptrend (>50): {strong_up} ({strong_up/len(osc_values)*100:.1f}%)")
            print(f"Strong downtrend (<-50): {strong_down} ({strong_down/len(osc_values)*100:.1f}%)")
            print(f"Neutral (-20 to 20): {neutral} ({neutral/len(osc_values)*100:.1f}%)")
            
            print("\nAroon Oscillator test successful!")
            
        else:
            print("No valid Aroon Oscillator values!")
            
    except Exception as e:
        print(f"Error: {e}")
        test_sample_data()

def test_sample_data():
    """Test with sample data if API fails"""
    
    print("\nTesting with sample data...")
    
    # Create alternating trend data
    np.random.seed(42)
    n = 100
    
    # Create data with clear trend changes
    uptrend = np.linspace(0, 0.15, 40)  # 40 days uptrend
    sideways = np.random.normal(0, 0.005, 20)  # 20 days sideways
    downtrend = np.linspace(0, -0.1, 40)  # 40 days downtrend
    
    trends = np.concatenate([uptrend, sideways, downtrend])
    noise = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(trends + noise))
    
    high = prices * (1 + np.random.uniform(0.001, 0.015, n))
    low = prices * (1 - np.random.uniform(0.001, 0.015, n))
    
    # Calculate Aroon Oscillator
    aroon_osc = ta.aroon_oscillator(high, low, period=14)
    
    valid_count = (~np.isnan(aroon_osc)).sum()
    print(f"Valid values: {valid_count}/{n}")
    
    if valid_count > 0:
        print(f"\nPeriod analysis:")
        valid_osc = aroon_osc[~np.isnan(aroon_osc)]
        
        # Expected: positive during uptrend, negative during downtrend
        uptrend_period = valid_osc[0:20]  # First 20 valid values (uptrend)
        downtrend_period = valid_osc[-20:]  # Last 20 values (downtrend)
        
        up_avg = np.mean(uptrend_period)
        down_avg = np.mean(downtrend_period)
        
        print(f"Uptrend period average: {up_avg:.1f}")
        print(f"Downtrend period average: {down_avg:.1f}")
        print(f"Expected behavior: {'OK' if up_avg > 0 and down_avg < 0 else 'UNEXPECTED'}")
        
        print(f"Sample oscillator range: {valid_osc.min():.1f} to {valid_osc.max():.1f}")

if __name__ == "__main__":
    test_aroon_oscillator_real()
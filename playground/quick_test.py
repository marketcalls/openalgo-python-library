#!/usr/bin/env python
"""
Quick ADX test - minimal script for fast testing
"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta

def quick_adx_test():
    """Quick test of ADX with minimal output"""
    
    client = api(
        api_key=os.getenv('OPENALGO_API_KEY', 'your_api_key_here'), 
        host='http://127.0.0.1:5001'
    )
    
    try:
        # Get data (extended range to ensure enough data points)
        df = client.history(symbol="RELIANCE", exchange="NSE", interval="D", 
                          start_date="2024-01-01", end_date="2025-01-14")
        
        print(f"Retrieved {len(df)} data points")
        
        if len(df) < 28:  # Need at least 2*period for ADX
            print(f"Warning: Only {len(df)} data points, need at least 28 for reliable ADX")
            period = max(5, len(df) // 3)  # Use smaller period
            print(f"Using period={period} instead")
        else:
            period = 14
            
        # Calculate ADX  
        df['plus_di'], df['minus_di'], df['adx'] = ta.adx(df['high'], df['low'], df['close'], period=period)
        
        # Quick check
        valid_adx = (~df['adx'].isna()).sum()
        total = len(df)
        
        print(f"ADX Test Results:")
        print(f"Data points: {total}")
        print(f"Valid ADX: {valid_adx}")
        print(f"Success rate: {valid_adx/total*100:.1f}%")
        
        if valid_adx > 0:
            latest = df.dropna().tail(1)
            print(f"Latest ADX: {latest['adx'].iloc[0]:.2f}")
            print("ADX working!")
        else:
            print("ADX failed!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_adx_test()
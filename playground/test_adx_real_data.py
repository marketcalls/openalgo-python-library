#!/usr/bin/env python
"""
Test ADX with real data from OpenAlgo API
"""

import sys
import os

# Add the parent directory to path to use our current testing version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from openalgo import ta
import pandas as pd

def test_adx_with_real_data():
    """Test ADX indicator with real market data"""
    
    print("Testing ADX with Real Market Data")
    print("=" * 50)
    
    # Initialize API client
    # Replace with your actual API key and host
    client = api(
        api_key=os.getenv('OPENALGO_API_KEY', 'your_api_key_here'), 
        host='http://127.0.0.1:5001'
    )
    
    try:
        # Fetch historical data
        print("Fetching RELIANCE data from NSE...")
        df = client.history(
            symbol="RELIANCE", 
            exchange="NSE", 
            interval="D", 
            start_date="2024-01-01", 
            end_date="2025-01-14"
        )
        
        print(f"Retrieved {len(df)} data points")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nFirst few rows of raw data:")
        print(df.head().round(2))
        
        # Calculate ADX using our fixed implementation
        print(f"\nCalculating ADX with period=14...")
        df['plus_di'], df['minus_di'], df['adx'] = ta.adx(
            df['high'], 
            df['low'], 
            df['close'], 
            period=14
        )
        
        # Analysis
        adx_valid = ~df['adx'].isna()
        valid_count = adx_valid.sum()
        total_count = len(df)
        
        print(f"\nADX Analysis:")
        print(f"Total data points: {total_count}")
        print(f"Valid ADX values: {valid_count}")
        print(f"NaN values: {total_count - valid_count}")
        print(f"Success rate: {valid_count/total_count*100:.1f}%")
        
        if valid_count > 0:
            print(f"\nADX Statistics:")
            print(f"Min ADX: {df['adx'].min():.2f}")
            print(f"Max ADX: {df['adx'].max():.2f}")
            print(f"Mean ADX: {df['adx'].mean():.2f}")
            
            # Show first valid ADX values
            first_valid_idx = df[adx_valid].index[0]
            print(f"\nFirst valid ADX on: {first_valid_idx}")
            
            print(f"\nFirst 10 valid ADX calculations:")
            valid_data = df[adx_valid][['high', 'low', 'close', 'plus_di', 'minus_di', 'adx']].head(10)
            print(valid_data.round(2))
            
            print(f"\nLast 5 ADX calculations:")
            recent_data = df[adx_valid][['high', 'low', 'close', 'plus_di', 'minus_di', 'adx']].tail(5)
            print(recent_data.round(2))
            
            # Check for any unusual values
            high_adx = df[df['adx'] > 50]
            if len(high_adx) > 0:
                print(f"\nHigh ADX periods (>50):")
                print(high_adx[['high', 'low', 'close', 'plus_di', 'minus_di', 'adx']].round(2))
            
            print(f"\n✓ ADX calculation successful with real market data!")
            
            # Save results for further analysis
            output_file = "playground/adx_results.csv"
            df.round(4).to_csv(output_file)
            print(f"Results saved to: {output_file}")
            
        else:
            print(f"\n✗ No valid ADX values calculated!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your API key and host are correct, and the OpenAlgo server is running.")
        
        # Fallback: Test with sample data
        print(f"\nFalling back to sample data test...")
        test_with_sample_data()

def test_with_sample_data():
    """Fallback test with generated sample data"""
    import numpy as np
    
    print("Testing with sample data...")
    
    # Generate realistic price data
    np.random.seed(42)
    n = 100
    base_price = 2500  # Similar to RELIANCE price range
    
    # Generate realistic OHLC data
    returns = np.random.normal(0, 0.02, n)  # 2% daily volatility
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate High/Low with realistic spreads
    spreads = np.random.uniform(0.005, 0.03, n)  # 0.5% to 3% daily range
    high_prices = close_prices * (1 + spreads/2)
    low_prices = close_prices * (1 - spreads/2)
    
    # Create DataFrame
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'high': high_prices,
        'low': low_prices, 
        'close': close_prices
    }, index=dates)
    
    print(f"Generated {len(df)} sample data points")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Calculate ADX
    df['plus_di'], df['minus_di'], df['adx'] = ta.adx(
        df['high'], 
        df['low'], 
        df['close'], 
        period=14
    )
    
    valid_count = (~df['adx'].isna()).sum()
    print(f"Valid ADX values: {valid_count}/{len(df)}")
    
    if valid_count > 0:
        print(f"Sample ADX results:")
        print(df[['high', 'low', 'close', 'plus_di', 'minus_di', 'adx']].tail(10).round(2))
        print("✓ Sample data test successful!")
    else:
        print("✗ Sample data test failed!")

if __name__ == "__main__":
    test_adx_with_real_data()
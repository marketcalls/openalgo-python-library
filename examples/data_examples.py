"""
OpenAlgo Market Data Examples
"""

import os
from openalgo import api
import json
from datetime import datetime, timedelta
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Format floats to 2 decimal places

def print_response(title, response):
    """Helper function to print responses in a readable format"""
    print(f"\n{title}:")
    if isinstance(response, dict):
        if response.get('status') == 'error':
            print(f"Error Type: {response.get('error_type', 'unknown')}")
            print("Message:", response.get('message', 'Unknown error'))
            if 'raw_data' in response:
                print("Raw Data:", response['raw_data'])
        else:
            print(json.dumps(response, indent=2))
    elif isinstance(response, pd.DataFrame):
        if response.empty:
            print("No data available")
        else:
            print("\nFirst few rows:")
            print(response.head().to_string())
            print("\nDataFrame Info:")
            print(response.info())
    else:
        print(response)

# Initialize the API client
client = api(
    api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),
    host="http://127.0.0.1:5000"
)

def quotes_example():
    """Example of getting real-time quotes"""
    try:
        # Get quotes for RELIANCE
        response = client.quotes(
            symbol="RELIANCE",
            exchange="NSE"
        )
        print_response("Quotes for RELIANCE", response)

        # Get quotes for SBIN
        response = client.quotes(
            symbol="SBIN",
            exchange="NSE"
        )
        print_response("Quotes for SBIN", response)
    except Exception as e:
        print(f"Error in quotes_example: {e}")

def depth_example():
    """Example of getting market depth data"""
    try:
        # Get depth for RELIANCE
        response = client.depth(
            symbol="RELIANCE",
            exchange="NSE"
        )
        print_response("Market Depth for RELIANCE", response)

        # Get depth for SBIN
        response = client.depth(
            symbol="SBIN",
            exchange="NSE"
        )
        print_response("Market Depth for SBIN", response)
    except Exception as e:
        print(f"Error in depth_example: {e}")

def intervals_example():
    """Example of getting supported intervals"""
    try:
        response = client.intervals()
        print_response("Supported Intervals", response)
        
        if response.get('status') == 'success':
            print("\nAvailable intervals by category:")
            for category, intervals in response['data'].items():
                print(f"\n{category.capitalize()}:")
                print(", ".join(intervals))
    except Exception as e:
        print(f"Error in intervals_example: {e}")

def interval_example():
    """Legacy example of getting supported intervals"""
    try:
        print("\nNOTE: interval() is a legacy method. It's recommended to use intervals() instead.")
        response = client.interval()
        print_response("Supported Intervals (Legacy Method)", response)
    except Exception as e:
        print(f"Error in interval_example: {e}")

def symbol_example():
    """Example of getting symbol details"""
    try:
        # Get symbol details for RELIANCE
        response = client.symbol(
            symbol="RELIANCE",
            exchange="NSE"
        )
        print_response("Symbol Details for RELIANCE", response)
        
        # Get symbol details for a futures contract
        response = client.symbol(
            symbol="NIFTY29MAY25FUT",
            exchange="NFO"
        )
        print_response("Symbol Details for NIFTY29MAY25FUT", response)
    except Exception as e:
        print(f"Error in symbol_example: {e}")

def history_example():
    """Example of getting historical data"""
    try:
        # Get current date and date 7 days ago
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        print(f"\nFetching historical data for SBIN from {start_date} to {end_date}")
        
        # Get 1-minute data
        minute_data = client.history(
            symbol="SBIN",
            exchange="NSE",
            interval="1m",
            start_date=start_date,
            end_date=end_date
        )
        print_response("1-Minute Historical Data", minute_data)

        if isinstance(minute_data, pd.DataFrame):
            print("\nBasic Statistics:")
            print(minute_data.describe().to_string())

        # Get daily data
        daily_data = client.history(
            symbol="SBIN",
            exchange="NSE",
            interval="D",
            start_date=start_date,
            end_date=end_date
        )
        print_response("Daily Historical Data", daily_data)

        if isinstance(daily_data, pd.DataFrame):
            # Calculate additional indicators
            daily_data['SMA_5'] = daily_data['close'].rolling(window=5).mean()
            daily_data['SMA_20'] = daily_data['close'].rolling(window=20).mean()
            daily_data['Volume_MA'] = daily_data['volume'].rolling(window=5).mean()
            
            print("\nDaily Data with Indicators:")
            print(daily_data.tail().to_string())

    except Exception as e:
        print(f"Error in history_example: {e}")

if __name__ == "__main__":
    print("Running Market Data Examples...")
    print("=" * 50)
    
    print("\nTesting Individual APIs:")
    quotes_example()
    depth_example()
    intervals_example()  # New method
    interval_example()   # Legacy method
    symbol_example()     # New method
    history_example()
    
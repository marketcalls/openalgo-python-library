"""
OpenAlgo Error Handling Test
Tests the improved error handling for connection timeouts and other network errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openalgo import api
import json
import time

def print_response(title, response):
    """Helper function to print responses in a readable format"""
    print(f"\n{title}:")
    print("=" * 50)
    if isinstance(response, dict):
        if response.get('status') == 'error':
            print(f"✓ Error Type: {response.get('error_type', 'unknown')}")
            print(f"✓ Message: {response.get('message', 'Unknown error')}")
            if 'code' in response:
                print(f"✓ HTTP Code: {response['code']}")
            if 'raw_response' in response:
                print(f"✓ Raw Response: {response['raw_response'][:200]}...")
        else:
            print("✓ Success Response:")
            print(json.dumps(response, indent=2))
    else:
        print(response)

def test_connection_error():
    """Test connection error handling with invalid host"""
    print("\n🔍 TESTING CONNECTION ERROR HANDLING")
    print("=" * 60)
    
    # Test with invalid host to trigger connection error
    client = api(
        api_key="dummy_api_key",  # not an OpenAlgo server: never send a real key here
        host="http://invalid-host-that-does-not-exist.com:5000"
    )
    
    # Test various API methods
    print("Testing quotes() with invalid host...")
    response = client.quotes(symbol="RELIANCE", exchange="NSE")
    print_response("Connection Error Test - Quotes", response)
    
    print("\nTesting search() with invalid host...")
    response = client.search(query="RELIANCE")
    print_response("Connection Error Test - Search", response)

def test_timeout_error():
    """Test timeout error handling"""
    print("\n🔍 TESTING TIMEOUT ERROR HANDLING")
    print("=" * 60)
    
    # Test with a host that might timeout (using httpbin.org delay endpoint)
    # This simulates a slow server response
    client = api(
        api_key="dummy_api_key",  # not an OpenAlgo server: never send a real key here
        host="https://httpbin.org/delay/10"  # 10 second delay - should timeout
    )
    
    print("Testing quotes() with timeout-prone host...")
    response = client.quotes(symbol="RELIANCE", exchange="NSE")
    print_response("Timeout Error Test - Quotes", response)

def test_valid_connection():
    """Test with valid connection and provided API key"""
    print("\n🔍 TESTING VALID CONNECTION")
    print("=" * 60)
    
    client = api(
        api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),
        host="http://127.0.0.1:5000"
    )
    
    # Test the new search function
    print("Testing search() function...")
    response = client.search(query="RELIANCE")
    print_response("Search Test - RELIANCE", response)
    
    # Test search with exchange filter
    print("\nTesting search() with exchange filter...")
    response = client.search(query="NIFTY", exchange="NFO")
    print_response("Search Test - NIFTY on NFO", response)
    
    # Test quotes
    print("\nTesting quotes() function...")
    response = client.quotes(symbol="RELIANCE", exchange="NSE")
    print_response("Quotes Test - RELIANCE", response)
    
    # Test intervals
    print("\nTesting intervals() function...")
    response = client.intervals()
    print_response("Intervals Test", response)

def test_api_error_responses():
    """Test API error responses with valid connection but invalid data"""
    print("\n🔍 TESTING API ERROR RESPONSES")
    print("=" * 60)
    
    client = api(
        api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),
        host="http://127.0.0.1:5000"
    )
    
    # Test with invalid symbol
    print("Testing with invalid symbol...")
    response = client.quotes(symbol="INVALID_SYMBOL_XYZ", exchange="NSE")
    print_response("Invalid Symbol Test", response)
    
    # Test with invalid exchange
    print("\nTesting with invalid exchange...")
    response = client.quotes(symbol="RELIANCE", exchange="INVALID_EXCHANGE")
    print_response("Invalid Exchange Test", response)
    
    # Test search with empty query
    print("\nTesting search with empty query...")
    response = client.search(query="")
    print_response("Empty Query Test", response)

def test_different_error_types():
    """Test different error types across all modules"""
    print("\n🔍 TESTING ERROR TYPES ACROSS MODULES")
    print("=" * 60)
    
    # Test with connection error
    invalid_client = api(
        api_key="dummy_api_key",  # not an OpenAlgo server: never send a real key here
        host="http://non-existent-host.com:5000"
    )
    
    print("Testing DataAPI error handling...")
    response = invalid_client.quotes(symbol="RELIANCE", exchange="NSE")
    print_response("DataAPI Connection Error", response)
    
    print("\nTesting OrderAPI error handling...")
    response = invalid_client.placeorder(
        symbol="RELIANCE",
        action="BUY",
        exchange="NSE",
        quantity=1
    )
    print_response("OrderAPI Connection Error", response)
    
    print("\nTesting AccountAPI error handling...")
    response = invalid_client.funds()
    print_response("AccountAPI Connection Error", response)

if __name__ == "__main__":
    print("🚀 OPENALGO ERROR HANDLING TEST")
    print("=" * 60)
    print("Testing improved error handling for:")
    print("- Connection timeouts")
    print("- Connection errors")
    print("- API errors")
    print("- HTTP errors")
    print("- JSON parsing errors")
    print("=" * 60)
    
    # Test connection errors
    test_connection_error()
    
    # Test timeout errors (commented out as it takes time)
    # test_timeout_error()
    
    # Test valid connection
    test_valid_connection()
    
    # Test API error responses
    test_api_error_responses()
    
    # Test different error types
    test_different_error_types()
    
    print("\n✅ ERROR HANDLING TEST COMPLETED!")
    print("=" * 60)
    print("All error types should now be properly handled with:")
    print("- Clear error messages")
    print("- Specific error types")
    print("- Consistent error format")
    print("- No retry logic (as requested)")
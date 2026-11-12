#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test OpenAlgo API with live connection
Testing margin function and NIFTY expiry dates
"""

import os
from openalgo import api
from datetime import datetime
import json

# Initialize the API client with provided key
API_KEY = os.getenv("OPENALGO_API_KEY", "your_api_key_here")
client = api(api_key=API_KEY, host="http://127.0.0.1:5000")

def test_connection():
    """Test basic API connection"""
    print("\n" + "="*60)
    print("Testing OpenAlgo API Connection")
    print("="*60)

    try:
        # Test with funds endpoint
        result = client.funds()
        if result.get('status') == 'success':
            print("✅ API Connection Successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ API Connection Failed: {result.get('message', 'Unknown error')}")
        return result
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        return None

def check_nifty_expiry():
    """Check NIFTY expiry dates to verify Tuesday expiry"""
    print("\n" + "="*60)
    print("Checking NIFTY Expiry Dates")
    print("="*60)

    try:
        # Get NIFTY options expiry dates
        result = client.expiry(
            symbol="NIFTY",
            exchange="NFO",
            instrumenttype="options"
        )

        if result.get('status') == 'success':
            expiry_dates = result.get('data', [])
            print(f"✅ Found {len(expiry_dates)} expiry dates for NIFTY options")
            print("\nNext 5 expiry dates:")

            # Parse and check day of week for first 5 expiries
            for i, expiry_str in enumerate(expiry_dates[:5], 1):
                try:
                    # Parse the date (format: DD-MMM-YY)
                    expiry_date = datetime.strptime(expiry_str, "%d-%b-%y")
                    day_name = expiry_date.strftime("%A")
                    print(f"  {i}. {expiry_str} - {day_name}")

                    # Check if it's Tuesday (weekday 1)
                    if expiry_date.weekday() == 1:
                        print(f"     ✅ This is a TUESDAY expiry!")
                except Exception as e:
                    print(f"  {i}. {expiry_str} - Could not parse date: {e}")

            # Check weekly expiries pattern
            print("\n📊 Weekly Expiry Analysis:")
            tuesday_count = 0
            thursday_count = 0
            other_count = 0

            for expiry_str in expiry_dates[:20]:  # Check first 20 expiries
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d-%b-%y")
                    weekday = expiry_date.weekday()
                    if weekday == 1:  # Tuesday
                        tuesday_count += 1
                    elif weekday == 3:  # Thursday
                        thursday_count += 1
                    else:
                        other_count += 1
                except:
                    pass

            print(f"  Tuesday expiries: {tuesday_count}")
            print(f"  Thursday expiries: {thursday_count}")
            print(f"  Other day expiries: {other_count}")

            if tuesday_count > thursday_count:
                print("\n✅ CONFIRMED: NIFTY now expires on TUESDAYS primarily!")

        else:
            print(f"❌ Failed to get expiry dates: {result.get('message', 'Unknown error')}")

        return result
    except Exception as e:
        print(f"❌ Error checking expiry: {str(e)}")
        return None

def test_margin_function():
    """Test the new margin function with various scenarios"""
    print("\n" + "="*60)
    print("Testing Margin Calculator Function")
    print("="*60)

    # Test 1: Single equity position
    print("\n1. Testing Single Equity Position (SBIN)")
    try:
        result = client.margin(positions=[{
            "symbol": "SBIN",
            "exchange": "NSE",
            "action": "BUY",
            "product": "MIS",
            "pricetype": "MARKET",
            "quantity": "10"
        }])

        if result.get('status') == 'success':
            print(f"✅ Margin calculation successful!")
            margin_data = result.get('data', {})
            print(f"   Total Margin Required: ₹{margin_data.get('total_margin_required', 'N/A')}")
            if 'span_margin' in margin_data:
                print(f"   SPAN Margin: ₹{margin_data.get('span_margin')}")
            if 'exposure_margin' in margin_data:
                print(f"   Exposure Margin: ₹{margin_data.get('exposure_margin')}")
        else:
            print(f"❌ Margin calculation failed: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 2: NIFTY Tuesday expiry option
    print("\n2. Testing NIFTY Tuesday Expiry Option")
    try:
        # First get the nearest Tuesday expiry
        expiry_result = client.expiry(
            symbol="NIFTY",
            exchange="NFO",
            instrumenttype="options"
        )

        if expiry_result.get('status') == 'success':
            expiry_dates = expiry_result.get('data', [])
            # Find first Tuesday expiry
            tuesday_expiry = None
            for expiry_str in expiry_dates[:5]:
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d-%b-%y")
                    if expiry_date.weekday() == 1:  # Tuesday
                        tuesday_expiry = expiry_str.replace("-", "").upper()
                        print(f"   Using Tuesday expiry: {expiry_str}")
                        break
                except:
                    pass

            if tuesday_expiry:
                # Create option symbol (format: NIFTY12NOV2425000CE)
                symbol = f"NIFTY{tuesday_expiry}25000CE"
                print(f"   Testing with symbol: {symbol}")

                result = client.margin(positions=[{
                    "symbol": symbol,
                    "exchange": "NFO",
                    "action": "SELL",
                    "product": "NRML",
                    "pricetype": "MARKET",
                    "quantity": "75"  # 1 lot of NIFTY
                }])

                if result.get('status') == 'success':
                    print(f"✅ Margin calculation for Tuesday expiry successful!")
                    margin_data = result.get('data', {})
                    print(f"   Total Margin Required: ₹{margin_data.get('total_margin_required', 'N/A')}")
                    if 'span_margin' in margin_data:
                        print(f"   SPAN Margin: ₹{margin_data.get('span_margin')}")
                    if 'exposure_margin' in margin_data:
                        print(f"   Exposure Margin: ₹{margin_data.get('exposure_margin')}")
                else:
                    print(f"❌ Margin calculation failed: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 3: Multiple positions (basket)
    print("\n3. Testing Basket Margin (Multiple Positions)")
    try:
        result = client.margin(positions=[
            {
                "symbol": "RELIANCE",
                "exchange": "NSE",
                "action": "BUY",
                "product": "MIS",
                "pricetype": "MARKET",
                "quantity": "5"
            },
            {
                "symbol": "TCS",
                "exchange": "NSE",
                "action": "BUY",
                "product": "MIS",
                "pricetype": "MARKET",
                "quantity": "3"
            }
        ])

        if result.get('status') == 'success':
            print(f"✅ Basket margin calculation successful!")
            margin_data = result.get('data', {})
            print(f"   Total Margin Required: ₹{margin_data.get('total_margin_required', 'N/A')}")
        else:
            print(f"❌ Basket margin calculation failed: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("OpenAlgo API Test Suite v1.0.36")
    print("Testing with API Key: " + API_KEY[:10] + "..." + API_KEY[-10:])
    print("="*60)

    # Test connection first
    connection_result = test_connection()

    if connection_result and connection_result.get('status') == 'success':
        # Check NIFTY expiry dates
        check_nifty_expiry()

        # Test margin function
        test_margin_function()
    else:
        print("\n❌ Cannot proceed with other tests due to connection failure")
        print("Please ensure:")
        print("1. OpenAlgo server is running at http://127.0.0.1:5000")
        print("2. The API key is valid")
        print("3. The broker is connected in OpenAlgo")

    print("\n" + "="*60)
    print("Test Suite Completed")
    print("="*60)

if __name__ == "__main__":
    main()
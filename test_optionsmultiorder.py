# -*- coding: utf-8 -*-
"""
Test script for optionsmultiorder function
"""

import os
import time
from openalgo import api

# Initialize with API key
client = api(
    api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),
    host="http://127.0.0.1:5000"
)

print("=" * 60)
print("Testing optionsmultiorder function")
print("=" * 60)

# Test 1: Long Straddle (2 legs - BUY only)
print("\n1. Testing Long Straddle (2 BUY legs)")
print("-" * 40)
result = client.optionsmultiorder(
    strategy="Long Straddle Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="25NOV25",
    legs=[
        {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": 75},
        {"offset": "ATM", "option_type": "PE", "action": "BUY", "quantity": 75}
    ]
)
print(f"Status: {result.get('status')}")
if result.get('status') == 'success':
    print(f"Underlying LTP: {result.get('underlying_ltp')}")
    if 'mode' in result:
        print(f"Mode: {result.get('mode')}")
    print("Results:")
    for leg_result in result.get('results', []):
        print(f"  Leg {leg_result.get('leg')}: {leg_result.get('symbol')} - "
              f"{leg_result.get('action')} {leg_result.get('option_type')} - "
              f"Status: {leg_result.get('status')} - Order: {leg_result.get('orderid')}")
else:
    print(f"Error: {result.get('message')}")

time.sleep(3)

# Test 2: Iron Condor (4 legs - mixed BUY/SELL)
print("\n2. Testing Iron Condor (4 legs - BUY first, then SELL)")
print("-" * 40)
result = client.optionsmultiorder(
    strategy="Iron Condor Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="25NOV25",
    legs=[
        {"offset": "OTM10", "option_type": "CE", "action": "BUY", "quantity": 75},
        {"offset": "OTM10", "option_type": "PE", "action": "BUY", "quantity": 75},
        {"offset": "OTM5", "option_type": "CE", "action": "SELL", "quantity": 75},
        {"offset": "OTM5", "option_type": "PE", "action": "SELL", "quantity": 75}
    ]
)
print(f"Status: {result.get('status')}")
if result.get('status') == 'success':
    print(f"Underlying LTP: {result.get('underlying_ltp')}")
    if 'mode' in result:
        print(f"Mode: {result.get('mode')}")
    print("Results:")
    for leg_result in result.get('results', []):
        print(f"  Leg {leg_result.get('leg')}: {leg_result.get('symbol')} - "
              f"{leg_result.get('action')} {leg_result.get('option_type')} - "
              f"Status: {leg_result.get('status')} - Order: {leg_result.get('orderid')}")
else:
    print(f"Error: {result.get('message')}")

time.sleep(3)

# Test 3: Bull Call Spread
print("\n3. Testing Bull Call Spread (2 legs)")
print("-" * 40)
result = client.optionsmultiorder(
    strategy="Bull Call Spread Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="25NOV25",
    legs=[
        {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": 75},
        {"offset": "OTM3", "option_type": "CE", "action": "SELL", "quantity": 75}
    ]
)
print(f"Status: {result.get('status')}")
if result.get('status') == 'success':
    print(f"Underlying LTP: {result.get('underlying_ltp')}")
    if 'mode' in result:
        print(f"Mode: {result.get('mode')}")
    print("Results:")
    for leg_result in result.get('results', []):
        print(f"  Leg {leg_result.get('leg')}: {leg_result.get('symbol')} - "
              f"{leg_result.get('action')} {leg_result.get('option_type')} - "
              f"Status: {leg_result.get('status')} - Order: {leg_result.get('orderid')}")
else:
    print(f"Error: {result.get('message')}")

time.sleep(3)

# Test 4: Calendar Spread (same strike, different expiries)
print("\n4. Testing Calendar Spread (per-leg expiry dates)")
print("-" * 40)
result = client.optionsmultiorder(
    strategy="Calendar Spread Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    legs=[
        {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": 75, "expiry_date": "30DEC25"},
        {"offset": "ATM", "option_type": "CE", "action": "SELL", "quantity": 75, "expiry_date": "25NOV25"}
    ]
)
print(f"Status: {result.get('status')}")
if result.get('status') == 'success':
    print(f"Underlying LTP: {result.get('underlying_ltp')}")
    if 'mode' in result:
        print(f"Mode: {result.get('mode')}")
    print("Results:")
    for leg_result in result.get('results', []):
        print(f"  Leg {leg_result.get('leg')}: {leg_result.get('symbol')} - "
              f"{leg_result.get('action')} {leg_result.get('option_type')} - "
              f"Status: {leg_result.get('status')} - Order: {leg_result.get('orderid')}")
else:
    print(f"Error: {result.get('message')}")

time.sleep(3)

# Test 5: Diagonal Spread (different strikes and expiries)
print("\n5. Testing Diagonal Spread (per-leg expiry dates)")
print("-" * 40)
result = client.optionsmultiorder(
    strategy="Diagonal Spread Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    legs=[
        {"offset": "ITM2", "option_type": "CE", "action": "BUY", "quantity": 75, "expiry_date": "30DEC25"},
        {"offset": "OTM2", "option_type": "CE", "action": "SELL", "quantity": 75, "expiry_date": "25NOV25"}
    ]
)
print(f"Status: {result.get('status')}")
if result.get('status') == 'success':
    print(f"Underlying LTP: {result.get('underlying_ltp')}")
    if 'mode' in result:
        print(f"Mode: {result.get('mode')}")
    print("Results:")
    for leg_result in result.get('results', []):
        print(f"  Leg {leg_result.get('leg')}: {leg_result.get('symbol')} - "
              f"{leg_result.get('action')} {leg_result.get('option_type')} - "
              f"Status: {leg_result.get('status')} - Order: {leg_result.get('orderid')}")
else:
    print(f"Error: {result.get('message')}")

print("\n" + "=" * 60)
print("Tests completed")
print("=" * 60)

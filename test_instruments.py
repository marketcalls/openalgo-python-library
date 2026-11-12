#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for instruments() function
"""

import os
from openalgo import api

# API credentials
API_KEY = os.getenv("OPENALGO_API_KEY", "your_api_key_here")
HOST = "http://127.0.0.1:5000"

# Initialize client
client = api(api_key=API_KEY, host=HOST)

print("="*80)
print("Testing instruments() function")
print("="*80)

# Test 1: Get all instruments
print("\nTest 1: Get all instruments (no exchange filter)")
print("-"*80)
result = client.instruments()

if isinstance(result, dict) and result.get('status') == 'error':
    print(f"[ERROR] {result.get('message')}")
    print(f"   Error Type: {result.get('error_type')}")
    print(f"   Code: {result.get('code', 'N/A')}")
else:
    import pandas as pd
    if isinstance(result, pd.DataFrame):
        print(f"[SUCCESS] Retrieved {len(result)} instruments")
        print(f"\nFirst 5 instruments:")
        print(result.head())
        print(f"\nColumns: {list(result.columns)}")
    else:
        print(f"[ERROR] Unexpected result type: {type(result)}")

# Test 2: Get NSE instruments only
print("\n\nTest 2: Get NSE instruments only")
print("-"*80)
result = client.instruments(exchange="NSE")

if isinstance(result, dict) and result.get('status') == 'error':
    print(f"[ERROR] {result.get('message')}")
else:
    import pandas as pd
    if isinstance(result, pd.DataFrame):
        print(f"[SUCCESS] Retrieved {len(result)} NSE instruments")
        print(f"\nFirst 5 NSE instruments:")
        print(result.head())
    else:
        print(f"❌ Unexpected result type: {type(result)}")

# Test 3: Get NFO instruments only
print("\n\nTest 3: Get NFO instruments only")
print("-"*80)
result = client.instruments(exchange="NFO")

if isinstance(result, dict) and result.get('status') == 'error':
    print(f"[ERROR] {result.get('message')}")
else:
    import pandas as pd
    if isinstance(result, pd.DataFrame):
        print(f"[SUCCESS] Retrieved {len(result)} NFO instruments")
        print(f"\nInstrument types in NFO:")
        if 'instrumenttype' in result.columns:
            print(result['instrumenttype'].value_counts())
    else:
        print(f"❌ Unexpected result type: {type(result)}")

print("\n" + "="*80)
print("Test completed!")
print("="*80)

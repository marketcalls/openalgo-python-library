#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for instruments() function (fixed version)
"""

import os
from openalgo import api
import pandas as pd

# API credentials
API_KEY = os.getenv("OPENALGO_API_KEY", "your_api_key_here")
HOST = "http://127.0.0.1:5000"

# Initialize client
client = api(api_key=API_KEY, host=HOST)

print("="*80)
print("Testing instruments() function - Fixed Version")
print("="*80)

# Test 1: Get NSE instruments
print("\nTest 1: Get NSE instruments")
print("-"*80)
result = client.instruments(exchange="NSE")

if isinstance(result, pd.DataFrame):
    print(f"[SUCCESS] Retrieved {len(result)} NSE instruments")
    print(f"\nColumns: {list(result.columns)}")
    print(f"\nFirst 5 instruments:")
    print(result.head())

    # Show instrument type distribution
    if 'instrumenttype' in result.columns:
        print(f"\nInstrument types:")
        print(result['instrumenttype'].value_counts())
else:
    print(f"[ERROR] {result.get('message', 'Unknown error')}")

# Test 2: Get NFO instruments
print("\n\nTest 2: Get NFO instruments")
print("-"*80)
result = client.instruments(exchange="NFO")

if isinstance(result, pd.DataFrame):
    print(f"[SUCCESS] Retrieved {len(result)} NFO instruments")

    # Show instrument type distribution
    if 'instrumenttype' in result.columns:
        print(f"\nInstrument types:")
        print(result['instrumenttype'].value_counts())

    # Show some NIFTY options
    if 'symbol' in result.columns:
        nifty_options = result[result['symbol'].str.contains('NIFTY25NOV25', na=False)]
        print(f"\nNIFTY 25NOV25 options: {len(nifty_options)} contracts")
        if not nifty_options.empty:
            print(nifty_options.head())
else:
    print(f"[ERROR] {result.get('message', 'Unknown error')}")

# Test 3: Filter and analyze
print("\n\nTest 3: Filter and analyze NSE instruments")
print("-"*80)
df = client.instruments(exchange="NSE")

if isinstance(df, pd.DataFrame):
    # Filter equities
    equities = df[df['instrumenttype'] == 'EQ']
    print(f"Total equities: {len(equities)}")

    # Search for RELIANCE
    if 'symbol' in df.columns:
        reliance = df[df['symbol'] == 'RELIANCE']
        if not reliance.empty:
            print(f"\nRELIANCE details:")
            print(reliance.to_dict('records')[0])

    # Export to CSV
    df.to_csv('nse_instruments_test.csv', index=False)
    print(f"\nExported to nse_instruments_test.csv")

print("\n" + "="*80)
print("All tests completed successfully!")
print("="*80)

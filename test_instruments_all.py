#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for instruments() function - Test ALL exchanges download
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
print("Testing instruments() - ALL Exchanges vs Specific Exchange")
print("="*80)

# Test 1: Download ALL exchanges (no parameter)
print("\n" + "="*80)
print("Test 1: Download ALL exchanges - client.instruments()")
print("="*80)
result_all = client.instruments()

if isinstance(result_all, pd.DataFrame):
    print(f"[SUCCESS] Retrieved {len(result_all)} total instruments from ALL exchanges")

    # Show exchange breakdown
    if 'exchange' in result_all.columns:
        print(f"\nExchange breakdown:")
        exchange_counts = result_all['exchange'].value_counts()
        for exchange, count in exchange_counts.items():
            print(f"  {exchange}: {count:,} instruments")

        print(f"\nTotal exchanges: {len(exchange_counts)}")

    # Show instrument types
    if 'instrumenttype' in result_all.columns:
        print(f"\nInstrument types across all exchanges:")
        type_counts = result_all['instrumenttype'].value_counts()
        for itype, count in type_counts.head(10).items():
            print(f"  {itype}: {count:,}")

    print(f"\nFirst 5 instruments:")
    print(result_all.head())

else:
    print(f"[ERROR] {result_all.get('message', 'Unknown error')}")
    print(f"Error type: {result_all.get('error_type')}")

# Test 2: Download specific exchange (NSE)
print("\n\n" + "="*80)
print("Test 2: Download NSE only - client.instruments(exchange='NSE')")
print("="*80)
result_nse = client.instruments(exchange="NSE")

if isinstance(result_nse, pd.DataFrame):
    print(f"[SUCCESS] Retrieved {len(result_nse)} NSE instruments")

    if 'instrumenttype' in result_nse.columns:
        print(f"\nNSE Instrument types:")
        print(result_nse['instrumenttype'].value_counts())
else:
    print(f"[ERROR] {result_nse.get('message', 'Unknown error')}")

# Test 3: Download specific exchange (NFO)
print("\n\n" + "="*80)
print("Test 3: Download NFO only - client.instruments(exchange='NFO')")
print("="*80)
result_nfo = client.instruments(exchange="NFO")

if isinstance(result_nfo, pd.DataFrame):
    print(f"[SUCCESS] Retrieved {len(result_nfo)} NFO instruments")

    if 'instrumenttype' in result_nfo.columns:
        print(f"\nNFO Instrument types:")
        print(result_nfo['instrumenttype'].value_counts())
else:
    print(f"[ERROR] {result_nfo.get('message', 'Unknown error')}")

# Test 4: Verify filtering works
if isinstance(result_all, pd.DataFrame) and isinstance(result_nse, pd.DataFrame):
    print("\n\n" + "="*80)
    print("Test 4: Verify filtering - Compare ALL vs NSE-only")
    print("="*80)

    # Filter NSE from ALL
    nse_from_all = result_all[result_all['exchange'] == 'NSE']

    print(f"NSE from client.instruments(): {len(result_nse)}")
    print(f"NSE from client.instruments() filtered: {len(nse_from_all)}")

    if len(nse_from_all) == len(result_nse):
        print(f"[SUCCESS] Counts match! Filtering works correctly.")
    else:
        print(f"[WARNING] Counts don't match. Difference: {abs(len(nse_from_all) - len(result_nse))}")

# Summary
print("\n\n" + "="*80)
print("SUMMARY")
print("="*80)

if isinstance(result_all, pd.DataFrame):
    print(f"[PASS] client.instruments() - Downloads ALL exchanges: {len(result_all):,} instruments")
else:
    print(f"[FAIL] client.instruments() - Error occurred")

if isinstance(result_nse, pd.DataFrame):
    print(f"[PASS] client.instruments(exchange='NSE') - Downloads NSE only: {len(result_nse):,} instruments")
else:
    print(f"[FAIL] client.instruments(exchange='NSE') - Error occurred")

if isinstance(result_nfo, pd.DataFrame):
    print(f"[PASS] client.instruments(exchange='NFO') - Downloads NFO only: {len(result_nfo):,} instruments")
else:
    print(f"[FAIL] client.instruments(exchange='NFO') - Error occurred")

print("\n" + "="*80)
print("Test completed!")
print("="*80)

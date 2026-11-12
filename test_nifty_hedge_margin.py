#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIFTY Futures and Hedged Options Margin Analysis
This script tests margin requirements for various NIFTY futures and options strategies
"""

import os
from openalgo import api
from datetime import datetime
import json

# Initialize the API client
API_KEY = os.getenv("OPENALGO_API_KEY", "your_api_key_here")
client = api(api_key=API_KEY, host="http://127.0.0.1:5000")

def format_margin(amount):
    """Format margin amount in Indian style"""
    if amount is None:
        return "N/A"
    return f"₹{amount:,.2f}"

def test_nifty_futures_standalone():
    """Test margin for standalone NIFTY futures"""
    print("\n" + "="*80)
    print("1. STANDALONE NIFTY FUTURES (NIFTY25NOV25FUT)")
    print("="*80)

    try:
        # Long futures position
        print("\n📈 LONG FUTURES (Buy 1 lot = 75 qty)")
        result = client.margin(positions=[{
            "symbol": "NIFTY25NOV25FUT",
            "exchange": "NFO",
            "action": "BUY",
            "product": "NRML",
            "pricetype": "MARKET",
            "quantity": "75"
        }])

        if result.get('status') == 'success':
            data = result.get('data', {})
            print(f"  Total Margin Required: {format_margin(data.get('total_margin_required'))}")
            print(f"  SPAN Margin: {format_margin(data.get('span_margin'))}")
            print(f"  Exposure Margin: {format_margin(data.get('exposure_margin'))}")
            long_futures_margin = data.get('total_margin_required', 0)
        else:
            print(f"  ❌ Error: {result.get('message')}")
            long_futures_margin = 0

        # Short futures position
        print("\n📉 SHORT FUTURES (Sell 1 lot = 75 qty)")
        result = client.margin(positions=[{
            "symbol": "NIFTY25NOV25FUT",
            "exchange": "NFO",
            "action": "SELL",
            "product": "NRML",
            "pricetype": "MARKET",
            "quantity": "75"
        }])

        if result.get('status') == 'success':
            data = result.get('data', {})
            print(f"  Total Margin Required: {format_margin(data.get('total_margin_required'))}")
            print(f"  SPAN Margin: {format_margin(data.get('span_margin'))}")
            print(f"  Exposure Margin: {format_margin(data.get('exposure_margin'))}")
            short_futures_margin = data.get('total_margin_required', 0)
        else:
            print(f"  ❌ Error: {result.get('message')}")
            short_futures_margin = 0

        return long_futures_margin, short_futures_margin

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0, 0

def test_long_futures_with_put_hedge():
    """Test margin for long futures hedged with long put (protective put on futures)"""
    print("\n" + "="*80)
    print("2. LONG FUTURES + LONG PUT (Protective Put Strategy)")
    print("="*80)
    print("Strategy: Buy NIFTY Futures + Buy ATM Put for downside protection")

    try:
        # Assuming NIFTY spot around 25000, using 25000 strike put
        positions = [
            {
                "symbol": "NIFTY25NOV25FUT",
                "exchange": "NFO",
                "action": "BUY",  # Long futures
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            },
            {
                "symbol": "NIFTY25NOV2525000PE",  # ATM Put
                "exchange": "NFO",
                "action": "BUY",  # Long Put for protection
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            }
        ]

        result = client.margin(positions=positions)

        if result.get('status') == 'success':
            data = result.get('data', {})
            print(f"\n  Combined Margin Required: {format_margin(data.get('total_margin_required'))}")
            print(f"  SPAN Margin: {format_margin(data.get('span_margin'))}")
            print(f"  Exposure Margin: {format_margin(data.get('exposure_margin'))}")

            print("\n  📊 Strategy Benefits:")
            print("  • Limited downside risk (max loss = futures entry - put strike + put premium)")
            print("  • Unlimited upside potential")
            print("  • Lower margin due to hedging benefit")

            return data.get('total_margin_required', 0)
        else:
            print(f"  ❌ Error: {result.get('message')}")
            return 0

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0

def test_short_futures_with_call_hedge():
    """Test margin for short futures hedged with long call (protective call on short futures)"""
    print("\n" + "="*80)
    print("3. SHORT FUTURES + LONG CALL (Protective Call Strategy)")
    print("="*80)
    print("Strategy: Sell NIFTY Futures + Buy ATM Call for upside protection")

    try:
        positions = [
            {
                "symbol": "NIFTY25NOV25FUT",
                "exchange": "NFO",
                "action": "SELL",  # Short futures
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            },
            {
                "symbol": "NIFTY25NOV2525000CE",  # ATM Call
                "exchange": "NFO",
                "action": "BUY",  # Long Call for protection
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            }
        ]

        result = client.margin(positions=positions)

        if result.get('status') == 'success':
            data = result.get('data', {})
            print(f"\n  Combined Margin Required: {format_margin(data.get('total_margin_required'))}")
            print(f"  SPAN Margin: {format_margin(data.get('span_margin'))}")
            print(f"  Exposure Margin: {format_margin(data.get('exposure_margin'))}")

            print("\n  📊 Strategy Benefits:")
            print("  • Limited upside risk (max loss = call strike - futures entry + call premium)")
            print("  • Profit from downward movement")
            print("  • Lower margin due to hedging benefit")

            return data.get('total_margin_required', 0)
        else:
            print(f"  ❌ Error: {result.get('message')}")
            return 0

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0

def test_synthetic_long():
    """Test margin for synthetic long (Long Call + Short Put at same strike)"""
    print("\n" + "="*80)
    print("4. SYNTHETIC LONG (Long Call + Short Put at 25000)")
    print("="*80)
    print("Strategy: Replicates long futures using options")

    try:
        positions = [
            {
                "symbol": "NIFTY25NOV2525000CE",
                "exchange": "NFO",
                "action": "BUY",  # Long Call
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            },
            {
                "symbol": "NIFTY25NOV2525000PE",
                "exchange": "NFO",
                "action": "SELL",  # Short Put
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            }
        ]

        result = client.margin(positions=positions)

        if result.get('status') == 'success':
            data = result.get('data', {})
            print(f"\n  Combined Margin Required: {format_margin(data.get('total_margin_required'))}")
            print(f"  SPAN Margin: {format_margin(data.get('span_margin'))}")
            print(f"  Exposure Margin: {format_margin(data.get('exposure_margin'))}")

            print("\n  📊 Strategy Characteristics:")
            print("  • Behaves exactly like long futures")
            print("  • P&L profile identical to futures")
            print("  • May have different margin requirements than futures")

            return data.get('total_margin_required', 0)
        else:
            print(f"  ❌ Error: {result.get('message')}")
            return 0

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0

def test_collar_strategy():
    """Test margin for collar strategy (Long Futures + Long OTM Put + Short OTM Call)"""
    print("\n" + "="*80)
    print("5. COLLAR STRATEGY (Long Futures + Long Put + Short Call)")
    print("="*80)
    print("Strategy: Long Futures + Buy 24500 Put + Sell 25500 Call")

    try:
        positions = [
            {
                "symbol": "NIFTY25NOV25FUT",
                "exchange": "NFO",
                "action": "BUY",  # Long futures
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            },
            {
                "symbol": "NIFTY25NOV2524500PE",  # OTM Put
                "exchange": "NFO",
                "action": "BUY",  # Long Put for protection
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            },
            {
                "symbol": "NIFTY25NOV2525500CE",  # OTM Call
                "exchange": "NFO",
                "action": "SELL",  # Short Call to finance put
                "product": "NRML",
                "pricetype": "MARKET",
                "quantity": "75"
            }
        ]

        result = client.margin(positions=positions)

        if result.get('status') == 'success':
            data = result.get('data', {})
            print(f"\n  Combined Margin Required: {format_margin(data.get('total_margin_required'))}")
            print(f"  SPAN Margin: {format_margin(data.get('span_margin'))}")
            print(f"  Exposure Margin: {format_margin(data.get('exposure_margin'))}")

            print("\n  📊 Strategy Benefits:")
            print("  • Limited downside (protected by put at 24500)")
            print("  • Limited upside (capped by call at 25500)")
            print("  • Very low margin due to full hedge")
            print("  • Often used for low-cost protection")

            return data.get('total_margin_required', 0)
        else:
            print(f"  ❌ Error: {result.get('message')}")
            return 0

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0

def main():
    """Run all margin tests and compare results"""
    print("\n" + "="*80)
    print("NIFTY FUTURES AND OPTIONS MARGIN ANALYSIS")
    print("Testing Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)

    # Store results for comparison
    results = {}

    # Test standalone futures
    long_fut, short_fut = test_nifty_futures_standalone()
    results['Long Futures'] = long_fut
    results['Short Futures'] = short_fut

    # Test hedged strategies
    results['Long Futures + Put'] = test_long_futures_with_put_hedge()
    results['Short Futures + Call'] = test_short_futures_with_call_hedge()
    results['Synthetic Long'] = test_synthetic_long()
    results['Collar Strategy'] = test_collar_strategy()

    # Summary comparison
    print("\n" + "="*80)
    print("MARGIN COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 Margin Requirements Comparison:")
    print("-" * 60)

    for strategy, margin in results.items():
        if margin > 0:
            print(f"  {strategy:<25}: {format_margin(margin):>20}")

            # Calculate savings compared to naked futures
            if strategy == 'Long Futures + Put' and long_fut > 0:
                savings = ((long_fut - margin) / long_fut) * 100
                print(f"  {'':25}  Savings vs naked long: {savings:.1f}%")
            elif strategy == 'Short Futures + Call' and short_fut > 0:
                savings = ((short_fut - margin) / short_fut) * 100
                print(f"  {'':25}  Savings vs naked short: {savings:.1f}%")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)

    print("""
    1. NAKED FUTURES:
       • Highest margin requirement
       • Unlimited risk in adverse direction
       • Simple to execute

    2. HEDGED FUTURES (with protective options):
       • Significantly lower margin (often 30-50% less)
       • Limited risk due to option protection
       • Small premium cost for protection

    3. SYNTHETIC POSITIONS:
       • Can replicate futures payoff using options
       • May have different margin requirements
       • Useful when futures liquidity is low

    4. COLLAR STRATEGY:
       • Lowest margin due to full hedge
       • Both upside and downside are limited
       • Cost-effective protection strategy

    💡 IMPORTANT NOTES:
    • Margin benefits vary by broker (Angel One, Zerodha, etc.)
    • SPAN margin calculation considers portfolio risk
    • Hedged positions get margin benefit due to lower risk
    • Always check real-time margin before placing orders
    """)

    print("="*80)
    print("Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
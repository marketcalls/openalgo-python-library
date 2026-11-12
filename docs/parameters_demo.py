#!/usr/bin/env python3
"""
OpenAlgo Technical Indicators - Parameters Demo

This script demonstrates the difference between mandatory and optional parameters
in the OpenAlgo technical indicators library.
"""

import numpy as np
from openalgo import ta

# Generate sample data
np.random.seed(42)
n = 100
close = 100 + np.cumsum(np.random.randn(n) * 0.01)
high = close + np.random.uniform(0, 2, n)
low = close - np.random.uniform(0, 2, n)
volume = np.random.randint(10000, 100000, n)

print("*** OpenAlgo Technical Indicators - Parameters Demo ***")
print("=" * 60)

print("\nMANDATORY vs OPTIONAL Parameters Examples")
print("-" * 50)

# 1. Indicators with ALL mandatory parameters
print("\nIndicators with ALL MANDATORY parameters:")
print("   (Must provide all parameters)")

try:
    sma_20 = ta.sma(close, 20)
    print(f"[OK] ta.sma(close, 20) - SMA: {sma_20[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    obv_values = ta.obv(close, volume)
    print(f"[OK] ta.obv(close, volume) - OBV: {obv_values[-1]:,.0f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    true_range = ta.true_range(high, low, close)
    print(f"[OK] ta.true_range(high, low, close) - TR: {true_range[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

# 2. Indicators with optional parameters - using defaults
print("\nIndicators with OPTIONAL parameters (using defaults):")
print("   (Can omit optional parameters)")

try:
    rsi_default = ta.rsi(close)  # Uses default period=14
    print(f"[OK] ta.rsi(close) - RSI with default period=14: {rsi_default[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    macd_line, signal_line, histogram = ta.macd(close)  # Uses defaults 12,26,9
    print(f"[OK] ta.macd(close) - MACD with defaults (12,26,9): {macd_line[-1]:.4f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    atr_default = ta.atr(high, low, close)  # Uses default period=14
    print(f"[OK] ta.atr(high, low, close) - ATR with default period=14: {atr_default[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    bb_upper, bb_middle, bb_lower = ta.bbands(close)  # Uses defaults 20, 2.0
    print(f"[OK] ta.bbands(close) - BB with defaults (20, 2.0): {bb_upper[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    supertrend_vals, direction = ta.supertrend(high, low, close)  # Uses defaults 10, 3.0
    print(f"[OK] ta.supertrend(high, low, close) - ST with defaults (10, 3.0): {supertrend_vals[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    alma_default = ta.alma(close)  # Uses defaults 21, 0.85, 6.0
    print(f"[OK] ta.alma(close) - ALMA with defaults (21, 0.85, 6.0): {alma_default[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

# 3. Same indicators with custom parameters
print("\nSame indicators with CUSTOM parameters:")
print("   (Overriding defaults with custom values)")

try:
    rsi_custom = ta.rsi(close, 21)  # Custom period
    print(f"[OK] ta.rsi(close, 21) - RSI with custom period=21: {rsi_custom[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    macd_line, signal_line, histogram = ta.macd(close, 10, 20, 7)  # Custom periods
    print(f"[OK] ta.macd(close, 10, 20, 7) - MACD with custom (10,20,7): {macd_line[-1]:.4f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    atr_custom = ta.atr(high, low, close, 20)  # Custom period
    print(f"[OK] ta.atr(high, low, close, 20) - ATR with custom period=20: {atr_custom[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    bb_upper, bb_middle, bb_lower = ta.bbands(close, 14, 1.5)  # Custom parameters
    print(f"[OK] ta.bbands(close, 14, 1.5) - BB with custom (14, 1.5): {bb_upper[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    supertrend_vals, direction = ta.supertrend(high, low, close, 7, 2.0)  # Custom parameters
    print(f"[OK] ta.supertrend(high, low, close, 7, 2.0) - ST with custom (7, 2.0): {supertrend_vals[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    alma_custom = ta.alma(close, 30, 0.9, 7.0)  # Custom parameters
    print(f"[OK] ta.alma(close, 30, 0.9, 7.0) - ALMA with custom (30, 0.9, 7.0): {alma_custom[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

# 4. Partial parameter customization
print("\nPARTIAL parameter customization:")
print("   (Mix of defaults and custom values)")

try:
    # ALMA with custom period but default offset and sigma
    alma_partial = ta.alma(close, period=30)  # period=30, offset=0.85, sigma=6.0
    print(f"[OK] ta.alma(close, period=30) - ALMA: period=30, offset=0.85 (default), sigma=6.0 (default)")
    print(f"   Result: {alma_partial[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # Keltner Channel with custom EMA period but default ATR and multiplier
    kc_upper, kc_middle, kc_lower = ta.keltner(high, low, close, ema_period=14)
    print(f"[OK] ta.keltner(high, low, close, ema_period=14)")
    print(f"   ema_period=14 (custom), atr_period=10 (default), multiplier=2.0 (default)")
    print(f"   Result: {kc_upper[-1]:.2f}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

# 5. Common mistakes
# 4. NEW UTILITY FUNCTIONS Examples
print("\nNEW UTILITY FUNCTIONS:")
print("   (Signal cleanup, state management, and historical reference)")

# Generate signal data for utility function examples
ema_fast = ta.ema(close, 10)
ema_slow = ta.ema(close, 20)
rsi = ta.rsi(close, 14)

# Create some sample signals
price_above_ma = close > ta.sma(close, 20)
price_below_ma = close < ta.sma(close, 20)
buy_signals = ta.crossover(ema_fast, ema_slow)
sell_signals = ta.crossunder(ema_fast, ema_slow)
oversold_signals = rsi < 30

try:
    # EXREM - Remove excessive signals
    clean_signals = ta.exrem(price_above_ma, price_below_ma)
    print(f"[OK] ta.exrem(primary, secondary) - Signal cleanup: {np.sum(clean_signals)} clean signals")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # FLIP - Toggle state creation
    uptrend_state = ta.flip(buy_signals, sell_signals)
    print(f"[OK] ta.flip(primary, secondary) - Trend state: {np.sum(uptrend_state)} uptrend periods")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # VALUEWHEN - Historical value reference (default n=1)
    price_at_oversold = ta.valuewhen(oversold_signals, close)
    valid_references = np.sum(~np.isnan(price_at_oversold))
    print(f"[OK] ta.valuewhen(expr, array) - Historical ref: {valid_references} valid references")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # VALUEWHEN with custom n parameter
    price_2nd_oversold = ta.valuewhen(oversold_signals, close, 2)
    valid_2nd_refs = np.sum(~np.isnan(price_2nd_oversold))
    print(f"[OK] ta.valuewhen(expr, array, 2) - 2nd occurrence: {valid_2nd_refs} valid references")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # RISING - value above its value n bars ago
    price_rising = ta.rising(close, 5)
    rising_periods = np.sum(price_rising)
    print(f"[OK] ta.rising(close, 5) - Rising trend: {rising_periods} periods")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # FALLING - value below its value n bars ago
    price_falling = ta.falling(close, 5)
    falling_periods = np.sum(price_falling)
    print(f"[OK] ta.falling(close, 5) - Falling trend: {falling_periods} periods")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # CROSS - Bidirectional crossover detection
    any_cross = ta.cross(ema_fast, ema_slow)
    cross_count = np.sum(any_cross)
    print(f"[OK] ta.cross(fast_ma, slow_ma) - Any direction crosses: {cross_count}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # COPPOCK - Long-term momentum indicator (uses defaults)
    coppock_default = ta.coppock(close)
    valid_coppock = np.sum(~np.isnan(coppock_default))
    print(f"[OK] ta.coppock(close) - Default params (10,14,11): {valid_coppock} valid values")
except Exception as e:
    print(f"[ERROR] Error: {e}")

try:
    # COPPOCK with custom parameters
    coppock_custom = ta.coppock(close, 8, 10, 8)
    valid_custom_coppock = np.sum(~np.isnan(coppock_custom))
    print(f"[OK] ta.coppock(close, 8, 10, 8) - Custom params: {valid_custom_coppock} valid values")
except Exception as e:
    print(f"[ERROR] Error: {e}")

print("\n[ERROR] COMMON MISTAKES:")
print("   (What happens when you forget mandatory parameters)")

print("\nTrying to call indicators without mandatory parameters:")

try:
    # This will fail - missing required parameters
    wrong_sma = ta.sma(close)  # Missing period parameter
    print(f"ta.sma(close) - This should fail!")
except Exception as e:
    print(f"[ERROR] ta.sma(close) failed as expected: {type(e).__name__}")

try:
    # This will fail - missing required parameters  
    wrong_obv = ta.obv(close)  # Missing volume parameter
    print(f"ta.obv(close) - This should fail!")
except Exception as e:
    print(f"[ERROR] ta.obv(close) failed as expected: {type(e).__name__}")

try:
    # This will fail - missing required parameters
    wrong_stoch = ta.stochastic(high, low)  # Missing close parameter
    print(f"ta.stochastic(high, low) - This should fail!")
except Exception as e:
    print(f"[ERROR] ta.stochastic(high, low) failed as expected: {type(e).__name__}")

print("\n[SUMMARY] SUMMARY:")
print("=" * 60)
print("[REQUIRED] MANDATORY parameters: Must always be provided")
print("[OPTIONAL] OPTIONAL parameters: Have sensible defaults, can be omitted")
print("[CUSTOM] CUSTOM parameters: Can override defaults for fine-tuning")
print("PARTIAL customization: Mix defaults with custom values")
print("\n[TIP] TIP: Start with defaults, then customize as needed!")
print("\n[DOCS] See PARAMETERS_REFERENCE.md for complete parameter details")

if __name__ == "__main__":
    print("\n[OK] Demo completed successfully!")
    print("[LINK] For more details, check the documentation:")
    print("   - PARAMETERS_REFERENCE.md - Complete parameter specifications")
    print("   - INDICATORS_QUICK_REFERENCE.md - Quick lookup tables")
    print("   - TECHNICAL_INDICATORS_GUIDE.md - Detailed explanations")
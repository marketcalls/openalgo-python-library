# -*- coding: utf-8 -*-
"""
Parity gate: Rust core (openalgo._oaindicators) vs the Python reference kernels
(openalgo.indicators.utils, run with NUMBA_DISABLE_JIT=1 = identical algorithm).
The helpers whose utils function now calls the shared kernel itself (highest, lowest,
change, cmo, the crossings, rising, falling, valuewhen) are compared with plain
pandas and NumPy expressions written in this script, or with the plain-Python
reference in test/indicator_references.py, so that each gate can fail.

HARD GATE  = Rust must match the Python reference (this is the backend we replace):
             identical length, identical NaN mask, and finite values within
             atol=1e-12 / rtol=1e-9 (EMA-family) or exact (min/max/diff/bool).

Exit code 0 only if every HARD gate passes on every dataset.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import sys
from pathlib import Path
# Ensure the LOCAL openalgo package (repo root) wins over any pip-installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import numpy as np
import pandas as pd

import indicator_references as references
import openalgo._oaindicators as rs
from openalgo.indicators import utils as ref
from openalgo.indicators.momentum import RSI as _RSI, Stochastic as _STO, CCI as _CCI, WilliamsR as _WR
from openalgo.indicators.trend import KAMA as _KAMA

DATA_DIR = Path(__file__).resolve().parent / "data"
ATOL, RTOL = 1e-12, 1e-9


def load(symbol, interval):
    df = pd.read_csv(DATA_DIR / f"{symbol}_{interval}.csv")
    return {
        "open": df["open"].to_numpy(np.float64),
        "high": df["high"].to_numpy(np.float64),
        "low": df["low"].to_numpy(np.float64),
        "close": df["close"].to_numpy(np.float64),
        "volume": df["volume"].to_numpy(np.float64),
    }


def cmp_float(a, b):
    """Return (ok, max_abs, max_rel, nan_ok) comparing two float arrays."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return False, np.inf, np.inf, False
    na, nb = np.isnan(a), np.isnan(b)
    nan_ok = np.array_equal(na, nb)
    mask = ~(na | nb)
    if not mask.any():
        return nan_ok, 0.0, 0.0, nan_ok
    da = np.abs(a[mask] - b[mask])
    denom = np.maximum(np.abs(b[mask]), 1e-300)
    dr = da / denom
    max_abs, max_rel = float(da.max()), float(dr.max())
    ok = nan_ok and (max_abs <= ATOL or max_rel <= RTOL)
    return ok, max_abs, max_rel, nan_ok


def cmp_bool(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        return False, 1, 1, False
    diff = int(np.sum(a != b))
    return diff == 0, float(diff), float(diff), True


# Each entry: (label, rust_fn, ref_fn, kind)
#   kind: "f" float gate, "b" bool gate
def build_cases(P):
    h, l, c, v = P["high"], P["low"], P["close"], P["volume"]
    per = 14
    cases = [
        ("sma(14)", lambda: rs.sma(c, per), lambda: ref.sma(c, per), "f"),
        ("wma(14)", lambda: rs.wma(c, per), lambda: _ref_wma(c, per), "f"),
        ("ema(14)", lambda: rs.ema(c, per), lambda: ref.ema(c, per), "f"),
        ("ema_wilder(14)", lambda: rs.ema_wilder(c, per), lambda: ref.ema_wilder(c, per), "f"),
        ("stdev(14)", lambda: rs.stdev(c, per), lambda: ref.stdev(c, per), "f"),
        ("rolling_sum(14)", lambda: rs.rolling_sum(c, per), lambda: ref.rolling_sum(c, per), "f"),
        ("rolling_variance(14)", lambda: rs.rolling_variance(c, per),
         lambda: ref.rolling_variance(c, per), "f"),
        ("change(1)", lambda: rs.change(c, 1), lambda: _ref_change(c, 1), "f"),
        ("roc(10)", lambda: rs.roc(c, 10), lambda: ref.roc(c, 10), "f"),
        ("highest(14)", lambda: rs.highest(c, per),
         lambda: pd.Series(c).rolling(per).max().to_numpy(), "f"),
        ("lowest(14)", lambda: rs.lowest(c, per),
         lambda: pd.Series(c).rolling(per).min().to_numpy(), "f"),
        ("true_range", lambda: rs.true_range(h, l, c), lambda: ref.true_range(h, l, c), "f"),
        ("atr_wilder(14)", lambda: rs.atr_wilder(h, l, c, per),
         lambda: ref.atr_wilder(h, l, c, per), "f"),
        ("atr_sma(14)", lambda: rs.atr_sma(h, l, c, per),
         lambda: ref.atr_sma(h, l, c, per), "f"),
        ("vwma(14)", lambda: rs.vwma(c, v, per), lambda: ref.vwma_optimized(c, v, per), "f"),
        ("cmo(14)", lambda: rs.cmo(c, per), lambda: references.cmo(c, per), "f"),
        ("kama(10,2,30)", lambda: rs.kama(c, 10, 2.0, 30.0),
         lambda: ref.kama_optimized(c, 10, 2.0, 30.0), "f"),
        ("ulcer_index(14)", lambda: rs.ulcer_index(c, per),
         lambda: ref.ulcer_index_optimized(c, per), "f"),
        # boolean kernels
        ("crossover", lambda: rs.crossover(c, ref.sma(c, per)),
         lambda: _ref_crossings(c, ref.sma(c, per))[0], "b"),
        ("crossunder", lambda: rs.crossunder(c, ref.sma(c, per)),
         lambda: _ref_crossings(c, ref.sma(c, per))[1], "b"),
        ("cross", lambda: rs.cross(c, ref.sma(c, per)),
         lambda: np.logical_or(*_ref_crossings(c, ref.sma(c, per))), "b"),
        ("rising(5)", lambda: rs.rising(c, 5), lambda: _ref_change(c, 5) > 0, "b"),
        ("falling(5)", lambda: rs.falling(c, 5), lambda: _ref_change(c, 5) < 0, "b"),
        ("valuewhen", lambda: rs.valuewhen(_expr(c, per), c, 1),
         lambda: _ref_valuewhen(_expr(c, per), c), "f"),
        # Phase-1 indicators (reference = original numba staticmethods, pure-python)
        ("rsi(14)", lambda: rs.rsi(c, per), lambda: _RSI._calculate_rsi(c, per), "f"),
        ("cci(14)", lambda: rs.cci(h, l, c, per),
         lambda: _CCI._calculate_cci(h, l, c, per), "f"),
        ("williams_r(14)", lambda: rs.williams_r(h, l, c, per),
         lambda: _WR._calculate_williams_r(c, per, ref.highest(h, per), ref.lowest(l, per)), "f"),
        ("stoch_k", lambda: rs.stochastic(h, l, c, per, 3, 3)[0],
         lambda: _ref_stoch(h, l, c, per, 3, 3)[0], "f"),
        ("stoch_d", lambda: rs.stochastic(h, l, c, per, 3, 3)[1],
         lambda: _ref_stoch(h, l, c, per, 3, 3)[1], "f"),
        ("kama_full_window(14,2,30)", lambda: rs.kama_full_window(c, 14, 2, 30),
         lambda: _KAMA._calculate_kama_full_window(c, 14, 2, 30), "f"),
    ]
    return cases


def _ref_stoch(h, l, c, k, sk, d):
    hh = ref.highest(h, k)
    ll = ref.lowest(l, k)
    return _STO._calculate_stochastic(c, k, sk, d, hh, ll)


def _ref_wma(c, period):
    n = len(c)
    out = np.full(n, np.nan)
    w = np.arange(1, period + 1, dtype=np.float64)
    ws = w.sum()
    for i in range(period - 1, n):
        out[i] = float(np.dot(c[i - period + 1:i + 1], w) / ws)
    return out


def _ref_change(c, length):
    """c[i] - c[i - length], NaN for the first `length` bars."""
    return (pd.Series(c) - pd.Series(c).shift(length)).to_numpy()


def _ref_crossings(a, b):
    """(crossover, crossunder): a moves from at-or-below b to above it, or the reverse.
    A NaN on the bar or the bar before compares False, so it never fires."""
    now, before = a - b, np.concatenate(([np.nan], (a - b)[:-1]))
    return (before <= 0) & (now > 0), (before >= 0) & (now < 0)


def _ref_valuewhen(expr, values):
    """`values` at the latest bar where `expr` is nonzero, NaN before the first one."""
    out = np.full(len(values), np.nan)
    latest = np.nan
    for i, (flag, value) in enumerate(zip(expr, values)):
        if flag != 0:
            latest = value
        out[i] = latest
    return out


def _expr(c, period):
    """Boolean-like float array: close above its SMA (drives valuewhen)."""
    return (c > np.nan_to_num(ref.sma(c, period), nan=np.inf)).astype(np.float64)


def main():
    datasets = [("RELIANCE", "D"), ("RELIANCE", "1m"), ("SBIN", "D"), ("SBIN", "1m")]
    all_pass = True
    hard_fail = []
    print(f"Parity gate  (atol={ATOL}, rtol={RTOL})")
    for symbol, interval in datasets:
        P = load(symbol, interval)
        n = len(P["close"])
        print(f"\n=== {symbol} {interval}  ({n} bars) ===")
        print(f"{'kernel':22} {'gate':6} {'max_abs':>12} {'max_rel':>12}")
        for label, rfn, reffn, kind in build_cases(P):
            rv = rfn()
            ev = reffn()
            if kind == "f":
                ok, ma, mr, nan_ok = cmp_float(rv, ev)
            else:
                ok, ma, mr, nan_ok = cmp_bool(rv, ev)
            gate = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
                hard_fail.append(f"{symbol}/{interval}/{label} (nan_ok={nan_ok})")
            print(f"{label:22} {gate:6} {ma:>12.3e} {mr:>12.3e}")
    print("\n" + ("=" * 60))
    if all_pass:
        print("RESULT: ALL HARD GATES PASS (Rust == Python reference).")
        return 0
    print("RESULT: HARD GATE FAILURES:")
    for f in hard_fail:
        print("  -", f)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

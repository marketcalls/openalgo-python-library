# -*- coding: utf-8 -*-
"""
Wrapper parity: prove the public `from openalgo import ta` API is unchanged after
swapping a class to the Rust backend.

Deliberately does NOT set NUMBA_DISABLE_JIT -- importing openalgo must now succeed
on its own (warm-up is defensive; swapped indicators are numba-free). Each migrated
indicator is checked against an INDEPENDENT reference (pandas / manual),
and for type/index preservation (pd.Series in -> pd.Series out with same index;
np.ndarray/list in -> np.ndarray out).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import numpy as np
import pandas as pd

import indicator_references as references
import openalgo
from openalgo import ta
import openalgo._oaindicators as _rs  # noqa: F401  (proves extension present)

DATA_DIR = Path(__file__).resolve().parent / "data"
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def close_series():
    df = pd.read_csv(DATA_DIR / "RELIANCE_D.csv", index_col=0)
    return df["close"]


def main():
    print(f"openalgo {openalgo.__version__} imported WITHOUT NUMBA_DISABLE_JIT "
          f"(rust ext: {'yes' if getattr(_rs, '__version__', None) else 'no'})")
    s = close_series()
    arr = s.to_numpy(np.float64)
    per = 14

    # ---- SMA vs pandas rolling mean ----
    out = ta.sma(s, per)
    ref = s.rolling(per).mean()
    check("sma type/index", isinstance(out, pd.Series) and out.index.equals(s.index))
    m = ~(np.isnan(out.to_numpy()) | np.isnan(ref.to_numpy()))
    check("sma vs pandas", float(np.abs(out.to_numpy()[m] - ref.to_numpy()[m]).max()) < 1e-9)

    # ---- WMA vs the plain-Python weighted average (weights 1..per, newest heaviest) ----
    outw = ta.wma(arr, per)
    check("wma returns ndarray", isinstance(outw, np.ndarray))
    refw = references.wma(arr, per)
    mw = ~(np.isnan(outw) | np.isnan(refw))
    check("wma vs manual", np.array_equal(np.isnan(outw), np.isnan(refw))
          and float(np.abs(outw[mw] - refw[mw]).max()) < 1e-9,
          f"maxdiff={np.abs(outw[mw] - refw[mw]).max():.2e}")

    # ---- EMA vs manual recursion (first-value seed, alpha=2/(p+1)) ----
    oute = ta.ema(arr, per)
    alpha = 2.0 / (per + 1.0)
    refe = np.empty_like(arr)
    refe[0] = arr[0]
    for i in range(1, len(arr)):
        refe[i] = alpha * arr[i] + (1 - alpha) * refe[i - 1]
    check("ema vs manual", float(np.abs(oute - refe).max()) < 1e-9,
          f"maxdiff={np.abs(oute-refe).max():.2e}")

    h = pd.read_csv(DATA_DIR / "RELIANCE_D.csv", index_col=0)["high"].to_numpy(np.float64)
    lo_ = pd.read_csv(DATA_DIR / "RELIANCE_D.csv", index_col=0)["low"].to_numpy(np.float64)

    # tuple-returning wrappers preserve Series typing
    k, d = ta.stochastic(pd.Series(h), pd.Series(lo_), pd.Series(arr))
    check("stochastic returns Series tuple", isinstance(k, pd.Series) and isinstance(d, pd.Series))

    # ---- list input -> ndarray out ----
    outl = ta.sma([1.0, 2.0, 3.0, 4.0, 5.0], 2)
    check("sma list->ndarray", isinstance(outl, np.ndarray) and outl[1] == 1.5)

    print("\nRESULT:", "ALL WRAPPER GATES PASS" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

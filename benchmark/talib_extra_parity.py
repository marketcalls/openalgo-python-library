# -*- coding: utf-8 -*-
"""Parity for the new TA-Lib-compatible indicators: ta.* must match TA-Lib exactly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import talib
from openalgo import ta

DATA = Path(__file__).resolve().parent / "data"
FAILS = []


def cmp(name, got, ref, tol=1e-9):
    got = np.asarray(getattr(got, "values", got), np.float64)
    ref = np.asarray(ref, np.float64)
    n = min(len(got), len(ref)); got, ref = got[-n:], ref[-n:]
    m = ~(np.isnan(got) | np.isnan(ref))
    nan_ok = np.array_equal(np.isnan(got), np.isnan(ref))
    d = float(np.abs(got[m] - ref[m]).max()) if m.any() else 0.0
    ok = nan_ok and d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:12} maxdiff={d:.2e} nan_ok={nan_ok}")
    if not ok:
        FAILS.append(name)


def main():
    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    o, h, l, c = (df[x].to_numpy(np.float64) for x in ["open", "high", "low", "close"])
    cmp("avgprice", ta.avgprice(o, h, l, c), talib.AVGPRICE(o, h, l, c))
    cmp("medprice", ta.medprice(h, l), talib.MEDPRICE(h, l))
    cmp("typprice", ta.typprice(h, l, c), talib.TYPPRICE(h, l, c))
    cmp("wclprice", ta.wclprice(h, l, c), talib.WCLPRICE(h, l, c))
    cmp("midpoint", ta.midpoint(c, 14), talib.MIDPOINT(c, 14))
    cmp("midprice", ta.midprice(h, l, 14), talib.MIDPRICE(h, l, 14))
    cmp("mom", ta.mom(c, 10), talib.MOM(c, 10))
    cmp("rocp", ta.rocp(c, 10), talib.ROCP(c, 10))
    cmp("rocr", ta.rocr(c, 10), talib.ROCR(c, 10))
    cmp("rocr100", ta.rocr100(c, 10), talib.ROCR100(c, 10))
    cmp("apo", ta.apo(c, 12, 26, "SMA"), talib.APO(c, 12, 26, 0))
    # Directional-movement family + linreg variants (TA-Lib-faithful)
    cmp("plus_dm", ta.plus_dm(h, l, 14), talib.PLUS_DM(h, l, 14))
    cmp("minus_dm", ta.minus_dm(h, l, 14), talib.MINUS_DM(h, l, 14))
    cmp("dx", ta.dx(h, l, c, 14), talib.DX(h, l, c, 14), tol=1e-7)
    cmp("adxr", ta.adxr(h, l, c, 14), talib.ADXR(h, l, c, 14), tol=1e-7)
    fk, fd = ta.stochf(h, l, c, 5, 3)
    tk, td = talib.STOCHF(h, l, c, 5, 3, 0)
    cmp("stochf_k", fk, tk, tol=1e-7)
    cmp("stochf_d", fd, td, tol=1e-7)
    cmp("linregangle", ta.linregangle(c, 14), talib.LINEARREG_ANGLE(c, 14), tol=1e-7)
    cmp("linregintercept", ta.linregintercept(c, 14), talib.LINEARREG_INTERCEPT(c, 14), tol=1e-7)
    print("\nRESULT:", "ALL TALIB-EXTRA PARITY PASS" if not FAILS else f"FAIL: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

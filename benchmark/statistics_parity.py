# -*- coding: utf-8 -*-
"""
Statistics parity: migrated _backend vs ORIGINAL numba kernels (NUMBA_DISABLE_JIT=1)
on real RELIANCE data. LINREG, LRSLOPE, CORREL, BETA, VAR, TSF, MEDIAN, MedianBands, MODE.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from openalgo.indicators import _backend as b
from openalgo.indicators import statistics as st
from openalgo.indicators.statistics import (LINREG, CORREL, BETA, TSF, MEDIAN, MedianBands, MODE,
                                            _calculate_slope_tv)

DATA = Path(__file__).resolve().parent / "data"
FAILS = []


def cmp(name, got, ref, tol=0.0):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    na, nb = np.isnan(got), np.isnan(ref)
    nan_ok = np.array_equal(na, nb)
    m = ~(na | nb)
    d = float(np.abs(got[m] - ref[m]).max()) if m.any() else 0.0
    ok = got.shape == ref.shape and nan_ok and d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:16} maxdiff={d:.3e} nan_ok={nan_ok}")
    if not ok:
        FAILS.append(name)


def main():
    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    df2 = pd.read_csv(DATA / "SBIN_D.csv", index_col=0)
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    m2 = df2["close"].to_numpy(np.float64)[:len(c)]

    # linreg/correl/beta/tsf: Rust naive summation vs numpy pairwise summation differs
    # by ~1e-14 abs (~1e-16 rel) - within the 1e-12 rel target, not bit-exact cross-impl.
    cmp("linreg", b.linreg(c, 14), LINREG._calculate_linearreg(c, 14), tol=1e-9)
    cmp("lrslope", b.lrslope(c, 100, 1), _calculate_slope_tv(c, 100, 1), tol=1e-9)
    cmp("correl", b.correl(c, m2, 20), CORREL._calculate_correl(c, m2, 20), tol=1e-9)
    cmp("beta", b.beta(c, m2, 60), BETA._calculate_beta_optimized(c, m2, 60), tol=1e-9)
    cmp("tsf", b.tsf(c, 14), TSF._calculate_tsf(c, 14), tol=1e-9)
    cmp("median", b.median(c, 5), MEDIAN._calculate_median(c, 5))
    cmp("mode", b.mode(c, 20, 10), MODE._calculate_mode_optimized(c, 20, 10))

    # VAR (PR mode, variance only)
    sV, vV, sdV = st.VAR._calculate_variance_tv_optimized(c, 20, False)
    cmp("var.pr", b.variance(c, 20, "PR", 20, 20, 14, False), vV)
    sV2, vV2, _ = st.VAR._calculate_variance_tv_optimized(c, 20, True)
    cmp("var.lr", b.variance(c, 20, "LR", 20, 20, 14, False), vV2)

    # MedianBands
    med, up, low_, mema = b.median_bands(h, lo, c, (h + lo) / 2, 3, 14, 2.0)
    cmp("medbands.med", med, MedianBands._calculate_median_percentile((h + lo) / 2, 3))

    print("\nRESULT:", "ALL STATISTICS PARITY PASS" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

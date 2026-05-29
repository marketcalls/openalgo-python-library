# -*- coding: utf-8 -*-
"""
Hybrid parity: migrated _backend vs ORIGINAL numba kernels (NUMBA_DISABLE_JIT=1)
on real RELIANCE data. ADX, Aroon, PivotPoints, SAR, DMI, ZigZag, WilliamsFractals, RWI.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from openalgo.indicators import _backend as b
from openalgo.indicators.hybrid import (ADX, Aroon, PivotPoints, SAR, ZigZag,
                                        WilliamsFractals, RWI)

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


def cmpb(name, got, ref):
    got = np.asarray(got).astype(bool)
    ref = np.asarray(ref).astype(bool)
    ok = got.shape == ref.shape and int(np.sum(got != ref)) == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:16} diff={int(np.sum(got != ref))}")
    if not ok:
        FAILS.append(name)


def main():
    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)

    # ADX reference (compose original staticmethods)
    tr, dmp, dmm = ADX._compute_dm(h, lo, c)
    from openalgo.indicators import utils as u
    sa, sp, sm = u.ema_wilder(tr, 14), u.ema_wilder(dmp, 14), u.ema_wilder(dmm, 14)
    rdip, rdim, rdx = ADX._compute_di_dx(sa, sp, sm, 14)
    radx = u.ema_wilder(rdx, 14)
    dip, dim, adxv = b.adx(h, lo, c, 14)
    cmp("adx.di_plus", dip, rdip)
    cmp("adx.di_minus", dim, rdim)
    cmp("adx.adx", adxv, radx)

    au, ad = b.aroon(h, lo, 14)
    rau, rad = Aroon._calculate_aroon(h, lo, 14)
    cmp("aroon.up", au, rau)
    cmp("aroon.down", ad, rad)

    pp = b.pivot_points(h, lo, c)
    rpp = PivotPoints._calculate_pivot_points(h, lo, c)
    for nm, g, r in zip(["pivot", "r1", "s1", "r2", "s2", "r3", "s3"], pp, rpp):
        cmp(f"pp.{nm}", g, r)

    sv, st_ = b.sar(h, lo, 0.02, 0.2)
    rsv, rst = SAR._calculate_sar(h, lo, 0.02, 0.2)
    cmp("sar.val", sv, rsv)
    cmp("sar.trend", st_, rst)

    cmp("zigzag", b.zigzag(h, lo, c, 5.0), ZigZag._calculate_zigzag(h, lo, c, 5.0))

    fu, fd = b.williams_fractals(h, lo, 2)
    rfu, rfd = WilliamsFractals._calculate_fractals_tv(h, lo, 2)
    cmpb("fractals.up", fu, rfu)
    cmpb("fractals.down", fd, rfd)

    rh, rl = b.rwi(h, lo, c, 14)
    rrh, rrl = RWI._calculate_rwi(h, lo, c, 14)
    cmp("rwi.high", rh, rrh, tol=1e-9)  # win_mean naive vs numpy pairwise ~1e-15
    cmp("rwi.low", rl, rrl, tol=1e-9)

    print("\nRESULT:", "ALL HYBRID PARITY PASS" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

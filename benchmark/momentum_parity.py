# -*- coding: utf-8 -*-
"""
Momentum parity: migrated _backend implementations vs the ORIGINAL numba kernels
(NUMBA_DISABLE_JIT=1) on real RELIANCE data. Covers BOP, ElderRay, Fisher, CRSI
(RSI/MACD/Stochastic/CCI/WilliamsR are gated in benchmark/parity.py).
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from openalgo.indicators import _backend as b
from openalgo.indicators import utils as u
from openalgo.indicators.momentum import BOP, Fisher, CRSI, RSI

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
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)

    cmp("bop", b.bop(o, h, lo, c), BOP._calculate_bop(o, h, lo, c))

    e = u.ema(c, 13)
    bull, bear = b.elderray(h, lo, c, 13)
    cmp("elderray.bull", bull, h - e)
    cmp("elderray.bear", bear, lo - e)

    hl2 = (h + lo) / 2.0
    f1, f2 = b.fisher(hl2, 9)
    rf1, rf2 = Fisher._calculate_fisher_tv(hl2, 9)
    cmp("fisher.fish", f1, rf1, tol=1e-9)
    cmp("fisher.trig", f2, rf2, tol=1e-9)

    # CRSI reference from original staticmethods
    pr_rsi = RSI._calculate_rsi(c, 3)
    streak = CRSI._calculate_updown_streak(c)
    st_rsi = RSI._calculate_rsi(streak, 2)
    roc1 = np.full_like(c, np.nan)
    for i in range(1, len(c)):
        if c[i - 1] != 0:
            roc1[i] = (c[i] - c[i - 1]) / c[i - 1] * 100
    prank = CRSI._calculate_percent_rank(roc1, 100)
    ref_crsi = np.full_like(c, np.nan)
    msk = ~(np.isnan(pr_rsi) | np.isnan(st_rsi) | np.isnan(prank))
    ref_crsi[msk] = (pr_rsi[msk] + st_rsi[msk] + prank[msk]) / 3.0
    cmp("crsi", b.crsi(c, 3, 2, 100), ref_crsi)

    print("\nRESULT:", "ALL MOMENTUM PARITY PASS" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

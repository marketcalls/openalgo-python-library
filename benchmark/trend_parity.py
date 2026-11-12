# -*- coding: utf-8 -*-
"""
Trend moving-average parity: the migrated _backend implementations vs the ORIGINAL
numba kernels (run pure-python via NUMBA_DISABLE_JIT) on real RELIANCE data.

Covers the composite/standalone trend MAs whose wrappers were swapped to Rust:
vwma, kama (full window), zlema, t3, trima, plus dema/tema/hma which compose the already
bit-exact ema/wma kernels (checked for finiteness + identical NaN mask vs a
utils.ema reconstruction). FRAMA has no kernel of its own left in its class, so it is
checked against the plain-Python reference in test/indicator_references.py.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import numpy as np
import pandas as pd

import indicator_references as references
from openalgo.indicators import _backend as b
from openalgo.indicators import utils as u
from openalgo.indicators.trend import (KAMA, ZLEMA, T3, TRIMA, ALMA, McGinley, VIDYA,
                                        Alligator, MovingAverageEnvelopes,
                                        Supertrend, Ichimoku, ChandeKrollStop)
from openalgo import ta

DATA = Path(__file__).resolve().parent / "data"
FAILS = []


def cmp(name, got, ref, tol=0.0):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    ok_shape = got.shape == ref.shape
    na, nb = np.isnan(got), np.isnan(ref)
    nan_ok = np.array_equal(na, nb)
    m = ~(na | nb)
    d = float(np.abs(got[m] - ref[m]).max()) if m.any() else 0.0
    ok = ok_shape and nan_ok and d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:18} maxdiff={d:.3e} nan_ok={nan_ok}")
    if not ok:
        FAILS.append(name)


def main():
    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)

    # Direct ports vs original numba kernels (bit-exact target)
    cmp("vwma(20)", b.vwma(c, v, 20), u.vwma_optimized(c, v, 20))
    # KAMA is recursive EMA-family (target 1e-12 rel); ~1e-13 abs drift is expected.
    cmp("kama_full_window(14)", b.kama_full_window(c, 14, 2, 30), KAMA._calculate_kama_full_window(c, 14, 2, 30), tol=1e-9)
    cmp("zlema(14)", b.zlema(c, 14), ZLEMA._calculate_zlema_optimized(c, 14))
    gd1 = T3._calculate_gd(c, 21, 0.7)
    gd2 = T3._calculate_gd(gd1, 21, 0.7)
    t3ref = T3._calculate_gd(gd2, 21, 0.7)
    cmp("t3(21,0.7)", b.t3(c, 21, 0.7), t3ref)
    cmp("trima(20)", b.trima(c, 20), TRIMA._calculate_trima(c, 20), tol=1e-9)  # rust naive vs numpy mean ~1e-13

    # ALMA/McGinley use exp/pow -> recursive/transcendental tolerance
    cmp("alma(21)", b.alma(c, 21, 0.85, 6.0), ALMA._calculate_alma(c, 21, 0.85, 6.0), tol=1e-9)
    cmp("mcginley(14)", b.mcginley(c, 14), McGinley._calculate_mcginley(c, 14), tol=1e-9)
    cmp("vidya(14)", b.vidya(c, 14, 0.2), VIDYA._calculate_vidya(c, 14, 0.2), tol=1e-9)
    # Alligator/MA-Envelopes compose bit-exact kernels
    jr, tr, lr = (Alligator._shift_series(Alligator._calculate_smma(c, p), s)
                  for p, s in [(13, 8), (8, 5), (5, 3)])
    aj, at_, al = b.alligator(c, 13, 8, 8, 5, 5, 3)
    cmp("alligator.jaw", aj, jr)
    cmp("alligator.teeth", at_, tr)
    cmp("alligator.lips", al, lr)
    me = MovingAverageEnvelopes()
    up_r = me._calculate_sma(c, 20) * (1 + 2.5 / 100)
    mu, mm, ml = b.ma_envelopes(c, 20, 2.5, "SMA")
    cmp("ma_env.upper", mu, up_r)
    cmp("ma_env.mid", mm, me._calculate_sma(c, 20))

    # Compositions of bit-exact ema/wma: reference rebuilt with utils.ema/wma logic
    cmp("dema(20)", ta.dema(c, 20), _dema_ref(c, 20))
    cmp("tema(20)", ta.tema(c, 20), _tema_ref(c, 20))
    cmp("hma(20)", ta.hma(c, 20), _hma_ref(c, 20))

    # Complex stateful / multi-output
    st, dr = b.supertrend(h, lo, c, 10, 3.0)
    rst, rdr = Supertrend._calculate_supertrend(h, lo, c, 10, 3.0)
    cmp("supertrend.line", st, rst)
    cmp("supertrend.dir", dr, rdr)
    ls, ss = b.chande_kroll_stop(h, lo, c, 10, 1.0, 9)
    rls, rss = ChandeKrollStop._calculate_chande_kroll_stop(h, lo, c, 10, 1.0, 9)
    cmp("ckstop.long", ls, rls)
    cmp("ckstop.short", ss, rss)
    ic = b.ichimoku(h, lo, c, 9, 26, 52, 26)
    ric = Ichimoku._calculate_ichimoku(h, lo, c, 9, 26, 52, 26)
    for nm, g, r in zip(["conv", "base", "spanA", "spanB", "lag"], ic, ric):
        cmp(f"ichimoku.{nm}", g, r)
    cmp("frama(26)", b.frama(h, lo, 26), references.frama(h, lo, 26), tol=1e-8)

    print("\nRESULT:", "ALL TREND PARITY PASS" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


def _dema_ref(data, period):
    e1 = u.ema(data, period)
    fv = period - 1
    e2 = np.full_like(data, np.nan)
    p2 = u.ema(e1[fv:], period)
    s = fv + period - 1
    e2[s:] = p2[period - 1:period - 1 + (len(data) - s)]
    return 2 * e1 - e2


def _tema_ref(data, period):
    e1 = u.ema(data, period)
    fv = period - 1
    e2 = np.full_like(data, np.nan)
    p2 = u.ema(e1[fv:], period)
    s = fv + period - 1
    e2[s:] = p2[period - 1:period - 1 + (len(data) - s)]
    e3 = np.full_like(data, np.nan)
    ve2 = e2[s:]
    clean = ve2[~np.isnan(ve2)]
    p3 = u.ema(clean, period)
    s3 = s + period - 1
    e3[s3:] = p3[period - 1:period - 1 + (len(data) - s3)]
    return 3 * e1 - 3 * e2 + e3


def _hma_ref(data, period):
    from openalgo.indicators.trend import WMA
    w = WMA()
    wh = w.calculate(data, period // 2)
    wf = w.calculate(data, period)
    diff = 2 * wh - wf
    return w.calculate(diff, int(np.sqrt(period)))


if __name__ == "__main__":
    raise SystemExit(main())

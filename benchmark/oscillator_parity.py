# -*- coding: utf-8 -*-
"""
Oscillator parity: migrated _backend vs ORIGINAL numba kernels (NUMBA_DISABLE_JIT=1)
on real RELIANCE data. Batch 1: ROC, CMO, TRIX, AO, AC, PPO, PO, DPO, AROONOSC.
CMO and RVI have no kernel of their own left in their classes, so they are checked
against the plain-Python references in test/indicator_references.py.
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
from openalgo.indicators.oscillators import (ROC, TRIX, AO, AC, PPO, PO, DPO, AROONOSC,
                                             UO, StochRSI, CHO, CHOP, KST, TSI,
                                             VI, GatorOscillator, STC, Coppock)

# Where the kernel and the reference add or smooth in a different order (a fresh sum per
# window against a running sum, for example), their values agree only to the last few bits.
SUM_ORDER_TOL = 1e-9

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
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:14} maxdiff={d:.3e} nan_ok={nan_ok}")
    if not ok:
        FAILS.append(name)


def _trix_ref(c, length):
    e1 = ema_ref(np.log(c), length)
    e2 = ema_ref(e1, length)
    e3 = ema_ref(e2, length)
    out = np.full_like(c, np.nan)
    out[1:] = (e3[1:] - e3[:-1]) * 10000.0
    return out


def ema_ref(d, p):
    from openalgo.indicators.oscillators import PPO
    return PPO._calculate_ema(d, p)


def main():
    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)

    cmp("roc", b.roc_osc(c, 12), ROC._calculate_roc(c, 12))
    cmp("cmo", b.cmo(c, 14), references.cmo(c, 14), tol=1e-9)  # rolling vs per-window
    cmp("trix", b.trix(c, 18), _trix_ref(c, 18))
    cmp("ao", b.ao(h, lo, 5, 34), AO._calculate_sma((h + lo) / 2, 5) - AO._calculate_sma((h + lo) / 2, 34),
        tol=SUM_ORDER_TOL)
    cmp("ac", b.ac(h, lo, 5), _ac_ref(h, lo, 5), tol=SUM_ORDER_TOL)
    pl, sl, hi = b.ppo(c, 12, 26, 9)
    rpl, rsl, rhi = _ppo_ref(c, 12, 26, 9)
    cmp("ppo.line", pl, rpl)
    cmp("ppo.signal", sl, rsl)
    cmp("po.sma", b.price_oscillator(c, 10, 20, "SMA"),
        PO._calculate_sma(c, 10) - PO._calculate_sma(c, 20), tol=1e-9)
    cmp("po.ema", b.price_oscillator(c, 10, 20, "EMA"),
        PO._calculate_ema(c, 10) - PO._calculate_ema(c, 20))
    cmp("dpo", b.dpo(c, 21, False), _dpo_ref(c, 21), tol=SUM_ORDER_TOL)
    cmp("aroonosc", b.aroon_osc(h, lo, 14), AROONOSC._calculate_aroon_osc(h, lo, 14))
    cmp("uo", b.uo(h, lo, c, 7, 14, 28), UO._calculate_uo(h, lo, c, 7, 14, 28), tol=1e-9)
    o = df["open"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    uk, ud = b.stochrsi(c, 14, 14, 3, 3)
    ruk, rud = StochRSI._calculate_stochrsi(c, 14, 14, 3, 3)
    cmp("stochrsi.k", uk, ruk)
    cmp("stochrsi.d", ud, rud)
    cmp("cho", b.cho(h, lo, c, v, 3, 10), _cho_ref(h, lo, c, v))
    cmp("chop", b.chop(h, lo, c, 14), CHOP._calculate_chop(h, lo, c, 14))
    rk, rks = b.rvi_vigor(o, h, lo, c, 10)
    rrk, rrks = references.rvi(o, h, lo, c, 10)
    cmp("rvi.vigor", rk, rrk, tol=SUM_ORDER_TOL)
    cmp("rvi.signal", rks, rrks, tol=SUM_ORDER_TOL)
    kk, ks = b.kst(c, 10, 15, 20, 30, 10, 10, 10, 15, 9)
    cmp("kst", kk, _kst_ref(c), tol=1e-9)
    tk, ts = b.tsi(c, 25, 13, 13)
    cmp("tsi", tk, _tsi_ref(c))
    vp, vm = b.vi(h, lo, c, 14)
    rvp, rvm = VI._calculate_vi_rolling(h, lo, c, 14)
    cmp("vi.plus", vp, rvp)
    cmp("vi.minus", vm, rvm)
    gu, gl = b.gator(h, lo, 13, 8, 5)
    rgu, rgl = _gator_ref(h, lo)
    cmp("gator.up", gu, rgu, tol=SUM_ORDER_TOL)
    cmp("gator.low", gl, rgl, tol=SUM_ORDER_TOL)
    cmp("stc", b.stc(c, 23, 50, 10, 3, 3), _stc_ref(c))
    cmp("coppock", b.coppock(c, 10, 14, 11), Coppock._calculate_coppock(c, 10, 14, 11))

    print("\nRESULT:", "ALL OSCILLATOR PARITY PASS" if not FAILS else f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


def _gator_ref(h, lo):
    hl2 = (h + lo) / 2.0
    jaw = GatorOscillator._apply_offset(GatorOscillator._calculate_rma(hl2, 13), 8)
    teeth = GatorOscillator._apply_offset(GatorOscillator._calculate_rma(hl2, 8), 5)
    lips = GatorOscillator._apply_offset(GatorOscillator._calculate_rma(hl2, 5), 3)
    return np.abs(jaw - teeth), -np.abs(teeth - lips)


def _stc_ref(c):
    from openalgo.indicators.trend import EMA
    from openalgo.indicators import utils as u
    macd = u.ema(c, 23) - u.ema(c, 50)
    k = STC._calculate_stochastic(macd, 10)
    k = np.where(np.isnan(k), 0.0, k)
    d = u.ema(k, 3)
    kd = STC._calculate_stochastic(d, 10)
    kd = np.where(np.isnan(kd), 0.0, kd)
    return np.clip(u.ema(kd, 3), 0.0, 100.0)


def _kst_ref(c):
    r1 = KST._calculate_roc(c, 10); r2 = KST._calculate_roc(c, 15)
    r3 = KST._calculate_roc(c, 20); r4 = KST._calculate_roc(c, 30)
    s1 = KST._calculate_sma(r1, 10); s2 = KST._calculate_sma(r2, 10)
    s3 = KST._calculate_sma(r3, 10); s4 = KST._calculate_sma(r4, 15)
    return s1 * 1 + s2 * 2 + s3 * 3 + s4 * 4


def _tsi_ref(c):
    pc = np.concatenate([[0.0], np.diff(c)])
    apc = np.abs(pc)
    pcs2 = TSI._calculate_ema(TSI._calculate_ema(pc, 25), 13)
    apcs2 = TSI._calculate_ema(TSI._calculate_ema(apc, 25), 13)
    out = np.full_like(c, np.nan)
    for i in range(len(c)):
        out[i] = 100 * (pcs2[i] / apcs2[i]) if apcs2[i] != 0 else 0.0
    return out


def _cho_ref(h, lo, c, v):
    adl = CHO._calculate_adl(h, lo, c, v)
    return CHO._calculate_ema(adl, 3) - CHO._calculate_ema(adl, 10)


def _ac_ref(h, lo, period):
    mp = (h + lo) / 2
    a = AO._calculate_sma(mp, 5) - AO._calculate_sma(mp, 34)
    return a - AC._calculate_sma(a, period)


def _ppo_ref(c, f, s, sig):
    fe = PPO._calculate_ema(c, f)
    se = PPO._calculate_ema(c, s)
    pl = np.empty_like(c)
    for i in range(len(c)):
        pl[i] = (fe[i] - se[i]) / se[i] * 100 if se[i] != 0 else 0
    sl = PPO._calculate_ema(pl, sig)
    return pl, sl, pl - sl


def _dpo_ref(c, period):
    sma = DPO._calculate_sma(c, period)
    bb = int(period / 2 + 1)
    out = np.full_like(c, np.nan)
    for i in range(bb, len(c)):
        if not np.isnan(sma[i - bb]):
            out[i] = c[i] - sma[i - bb]
    return out


if __name__ == "__main__":
    raise SystemExit(main())

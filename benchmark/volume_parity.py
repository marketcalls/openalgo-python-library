# -*- coding: utf-8 -*-
"""
Volume parity: migrated _backend implementations vs ORIGINAL numba kernels
(NUMBA_DISABLE_JIT=1) on real RELIANCE data. Batch 1: OBV, ADL, CMF, MFI, EMV, FI.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from openalgo.indicators import _backend as b
from openalgo.indicators.volume import (OBV, ADL, CMF, MFI, EMV, FI, NVI, PVI, VROC,
                                        VOLOSC, KlingerVolumeOscillator as KVO,
                                        PriceVolumeTrend as PVT, RVOL)

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


def main():
    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)

    cmp("obv", b.obv(c, v), OBV._calculate_obv(c, v))
    cmp("adl", b.adl(h, lo, c, v), ADL._calculate_adl(h, lo, c, v))
    cmp("cmf", b.cmf(h, lo, c, v, 20), CMF._calculate_cmf(h, lo, c, v, 20))
    cmp("mfi", b.mfi(h, lo, c, v, 14), MFI._calculate_mfi(h, lo, c, v, 14))

    raw_emv = EMV._calculate_emv_raw(h, lo, v, 10000.0)
    cmp("emv", b.emv(h, lo, v, 14, 10000), EMV._calculate_sma(raw_emv, 14))

    raw_fi = FI._calculate_raw_fi(c, v)
    cmp("force_index", b.force_index(c, v, 13), FI._calculate_ema(raw_fi, 13))

    cmp("nvi", b.nvi(c, v), NVI._calculate_nvi(c, v))
    cmp("pvi", b.pvi(c, v, 100.0), PVI._calculate_pvi(c, v, 100.0))
    cmp("pvt", b.pvt(c, v), PVT._calculate_pvt(c, v))
    cmp("vroc", b.vroc(v, 25), VROC._calculate_vroc(v, 25))
    cmp("volosc", b.volosc(v, 5, 10), _volosc_ref(v, 5, 10))
    kk, kt = b.kvo(h, lo, c, v, 13, 34, 55)
    rkk, rkt = KVO._calculate_kvo_tv(h, lo, c, v, 13, 34, 55)
    cmp("kvo", kk, rkk)
    cmp("kvo.trig", kt, rkt)
    cmp("rvol", b.rvol(v, 20), RVOL._calculate_rvol(v, 20))

    # OBVSmoothed branches + VWAP (public API level, vs independent references)
    from openalgo import ta
    from openalgo.indicators.volume import OBVSmoothed, VWAP
    obv = OBV._calculate_obv(c, v)
    cmp("obvsm.smma", np.asarray(ta.obv_smoothed(c, v, "SMMA (RMA)", 20)),
        OBVSmoothed._calculate_rma(obv, 20))
    cmp("obvsm.sma", np.asarray(ta.obv_smoothed(c, v, "SMA", 20)),
        np.asarray(ta.sma(OBV._calculate_obv(c, v), 20)))
    src = (h + lo + c) / 3.0
    starts = np.zeros(len(c)); starts[0] = 1.0
    rv, _ = VWAP._calculate_session_vwap(src, v, starts.astype(bool))
    cmp("vwap", np.asarray(ta.vwap(h, lo, c, v)), rv)

    print("\nRESULT:", "ALL VOLUME PARITY PASS" if not FAILS else f"FAILURES: {FAILS}")


def _ema_wilder_ref(data, period):
    from openalgo.indicators.volume import OBVSmoothed
    return OBVSmoothed._calculate_rma(data, period)


def _volosc_ref(v, s, l):
    se = VOLOSC._calculate_ema_safe(v, s)
    le = VOLOSC._calculate_ema_safe(v, l)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        if not np.isnan(le[i]) and le[i] != 0:
            out[i] = 100.0 * (se[i] - le[i]) / le[i]
    return out
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

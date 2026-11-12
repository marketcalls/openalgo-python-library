# -*- coding: utf-8 -*-
"""
Speed benchmark: Rust-backed indicators (current ta.*) vs the legacy pure-Python
reference (the original _calculate_* / utils kernels, now interpreted since numba is
gone). Writes benchmark/SPEED.md.

Datasets: real RELIANCE daily (benchmark/data/RELIANCE_D.csv) and a deterministic
synthetic 100k OHLCV series.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from openalgo.indicators import _backend as B
from openalgo.indicators import utils as U
from openalgo.indicators.trend import WMA, SMA
from openalgo.indicators.momentum import RSI, MACD, Stochastic
from openalgo.indicators.volatility import BollingerBands
from openalgo.indicators.hybrid import ADX, SAR
from openalgo.indicators.statistics import LINREG

DATA = Path(__file__).resolve().parent / "data"


def best(fn, repeats=5):
    t = float("inf")
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t * 1000.0  # ms


def synth(n, seed=11):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n))
    h = c + np.abs(rng.standard_normal(n))
    lo = c - np.abs(rng.standard_normal(n))
    return h, lo, c


def bench_dataset(label, h, lo, c, lines):
    n = len(c)
    lines.append(f"\n### {label} ({n} bars)\n")
    lines.append("| indicator | Rust (ms) | Py-ref (ms) | speedup |")
    lines.append("|-----------|----------:|------------:|--------:|")

    # (name, rust_call, pyref_call)
    cases = [
        ("sma(20)", lambda: B.sma(c, 20), lambda: SMA._calculate_sma(c, 20)),
        ("ema(20)", lambda: B.ema(c, 20), lambda: U.ema(c, 20)),
        ("wma(20)", lambda: B.wma(c, 20), lambda: WMA._calculate_wma(c, 20)),
        ("rsi(14)", lambda: B.rsi(c, 14), lambda: RSI._calculate_rsi(c, 14)),
        ("macd", lambda: B.macd(c, 12, 26, 9), lambda: MACD._calculate_macd(c, 12, 26, 9)),
        ("bbands(20)", lambda: B.bbands(c, 20, 2.0),
         lambda: BollingerBands._calculate_bollinger_bands(c, 20, 2.0)),
        ("atr(14)", lambda: B.atr_wilder(h, lo, c, 14), lambda: U.atr_wilder(h, lo, c, 14)),
        ("stochastic", lambda: B.stochastic(h, lo, c, 14, 3, 3),
         lambda: _stoch_ref(h, lo, c)),
        ("supertrend", lambda: B.supertrend(h, lo, c, 10, 3.0),
         lambda: _supertrend_ref(h, lo, c)),
        ("adx(14)", lambda: B.adx(h, lo, c, 14), lambda: _adx_ref(h, lo, c)),
        ("sar", lambda: B.sar(h, lo, 0.02, 0.2), lambda: SAR._calculate_sar(h, lo, 0.02, 0.2)),
        ("linreg(14)", lambda: B.linreg(c, 14), lambda: LINREG._calculate_linearreg(c, 14)),
    ]
    for name, rfn, pfn in cases:
        rt = best(rfn)
        pt = best(pfn, repeats=2)  # interpreted ref is slow; fewer repeats
        sp = pt / rt if rt > 0 else float("inf")
        lines.append(f"| {name} | {rt:.3f} | {pt:.3f} | {sp:.1f}x |")


def _stoch_ref(h, lo, c):
    hh = U.highest(h, 14); ll = U.lowest(lo, 14)
    return Stochastic._calculate_stochastic(c, 14, 3, 3, hh, ll)


def _supertrend_ref(h, lo, c):
    from openalgo.indicators.trend import Supertrend
    return Supertrend._calculate_supertrend(h, lo, c, 10, 3.0)


def _adx_ref(h, lo, c):
    tr, dmp, dmm = ADX._compute_dm(h, lo, c)
    sa, sp, sm = U.ema_wilder(tr, 14), U.ema_wilder(dmp, 14), U.ema_wilder(dmm, 14)
    dip, dim, dx = ADX._compute_di_dx(sa, sp, sm, 14)
    return dip, dim, U.ema_wilder(dx, 14)


def main():
    lines = ["# OpenAlgo Indicators - Rust Core Speed Benchmark", ""]
    lines.append("Rust = current `ta.*` backend (openalgo._oaindicators). "
                 "Py-ref = the original kernels run interpreted (numba removed). "
                 "Speedup = Py-ref / Rust.")

    df = pd.read_csv(DATA / "RELIANCE_D.csv", index_col=0)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    bench_dataset("RELIANCE daily (real)", h, lo, c, lines)

    h2, lo2, c2 = synth(100_000)
    bench_dataset("Synthetic 100k", h2, lo2, c2, lines)

    out = Path(__file__).resolve().parent / "SPEED.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

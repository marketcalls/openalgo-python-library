# -*- coding: utf-8 -*-
"""
Full benchmark on a large real dataset (NIFTY 50.csv, ~925k 1-min bars).

For every public indicator it times three implementations and writes
benchmark/FULL_BENCHMARK.md:
  - New (Rust)      : current `ta.*` backend (openalgo._oaindicators)
  - Old (OpenAlgo)  : the original kernels run INTERPRETED. numba is uninstallable on
                      this Python 3.14 / numpy 2.x env (the reason for the migration),
                      so this is a conservative lower bound on the old JIT speed.
  - TA-Lib          : where a comparable function exists.
Also reports max|delta| of New vs Old as an accuracy check.

Volume in the index feed is 0, so a positive synthetic volume is used for volume
indicators (timing only; parity of volume indicators is covered elsewhere on yfinance).
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from openalgo import ta
from openalgo.indicators import utils as U
from openalgo.indicators import trend as T, momentum as M, volatility as V
from openalgo.indicators import volume as VO, oscillators as O, statistics as S, hybrid as H

try:
    import talib
    TL = True
except Exception:
    TL = False

CSV = Path(__file__).resolve().parents[1] / "NIFTY 50.csv"


def best(fn, repeats):
    t = float("inf")
    last = None
    for _ in range(repeats):
        s = time.perf_counter()
        last = fn()
        t = min(t, time.perf_counter() - s)
    return t * 1000.0, last


def _arr(x):
    if isinstance(x, tuple):
        x = x[0]
    return np.asarray(getattr(x, "values", x), dtype=np.float64)


def maxdelta(a, b):
    try:
        a, b = _arr(a), _arr(b)
        n = min(len(a), len(b))
        a, b = a[-n:], b[-n:]
        m = ~(np.isnan(a) | np.isnan(b))
        return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0
    except Exception:
        return float("nan")


def main():
    _nrows = os.environ.get("OABENCH_NROWS")
    df = pd.read_csv(CSV, nrows=int(_nrows) if _nrows else None)
    o = df["OPEN"].to_numpy(np.float64)
    h = df["HIGH"].to_numpy(np.float64)
    lo = df["LOW"].to_numpy(np.float64)
    c = df["CLOSE"].to_numpy(np.float64)
    n = len(c)
    # synthetic positive volume (index feed volume is 0)
    rng = np.random.default_rng(3)
    v = rng.integers(1_000, 50_000, n).astype(np.float64)

    def stoch_old(): hh = U.highest(h, 14); ll = U.lowest(lo, 14); return M.Stochastic._calculate_stochastic(c, 14, 3, 3, hh, ll)
    def adx_old():
        tr, dp, dm = H.ADX._compute_dm(h, lo, c)
        a, p, m = U.ema_wilder(tr, 14), U.ema_wilder(dp, 14), U.ema_wilder(dm, 14)
        dip, dim, dx = H.ADX._compute_di_dx(a, p, m, 14)
        return dip, dim, U.ema_wilder(dx, 14)

    # label, new_fn, old_fn(or None), talib_fn(or None)
    cases = [
        ("[Trend]", None, None, None),
        ("sma(20)", lambda: ta.sma(c, 20), lambda: T.SMA._calculate_sma(c, 20), (lambda: talib.SMA(c, 20)) if TL else None),
        ("ema(20)", lambda: ta.ema(c, 20), lambda: U.ema(c, 20), (lambda: talib.EMA(c, 20)) if TL else None),
        ("wma(20)", lambda: ta.wma(c, 20), lambda: T.WMA._calculate_wma(c, 20), (lambda: talib.WMA(c, 20)) if TL else None),
        ("dema(20)", lambda: ta.dema(c, 20), None, (lambda: talib.DEMA(c, 20)) if TL else None),
        ("tema(20)", lambda: ta.tema(c, 20), None, (lambda: talib.TEMA(c, 20)) if TL else None),
        ("trima(20)", lambda: ta.trima(c, 20), lambda: T.TRIMA._calculate_trima(c, 20), (lambda: talib.TRIMA(c, 20)) if TL else None),
        ("hma(20)", lambda: ta.hma(c, 20), None, None),
        ("vwma(20)", lambda: ta.vwma(c, v, 20), lambda: U.vwma_optimized(c, v, 20), None),
        ("alma(21)", lambda: ta.alma(c, 21), lambda: T.ALMA._calculate_alma(c, 21, 0.85, 6.0), None),
        ("kama(14)", lambda: ta.kama(c, 14), lambda: T.KAMA._calculate_kama_tv(c, 14, 2, 30), (lambda: talib.KAMA(c, 30)) if TL else None),
        ("zlema(20)", lambda: ta.zlema(c, 20), lambda: T.ZLEMA._calculate_zlema_optimized(c, 20), None),
        ("t3(21)", lambda: ta.t3(c, 21), None, (lambda: talib.T3(c, 21)) if TL else None),
        ("mcginley(14)", lambda: ta.mcginley(c, 14), lambda: T.McGinley._calculate_mcginley(c, 14), None),
        ("vidya(14)", lambda: ta.vidya(c, 14), lambda: T.VIDYA._calculate_vidya(c, 14, 0.2), None),
        ("supertrend(10,3)", lambda: ta.supertrend(h, lo, c, 10, 3.0), lambda: T.Supertrend._calculate_supertrend(h, lo, c, 10, 3.0), None),
        ("ichimoku", lambda: ta.ichimoku(h, lo, c), lambda: T.Ichimoku._calculate_ichimoku_tv(h, lo, c, 9, 26, 52, 26), None),
        ("frama(26)", lambda: ta.frama(h, lo, 26), lambda: T.FRAMA._calculate_frama_tv(h, lo, 26), None),
        ("ckstop", lambda: ta.ckstop(h, lo, c), lambda: T.ChandeKrollStop._calculate_chande_kroll_stop(h, lo, c, 10, 1.0, 9), None),

        ("[Momentum]", None, None, None),
        ("rsi(14)", lambda: ta.rsi(c, 14), lambda: M.RSI._calculate_rsi(c, 14), (lambda: talib.RSI(c, 14)) if TL else None),
        ("macd", lambda: ta.macd(c), lambda: M.MACD._calculate_macd(c, 12, 26, 9), (lambda: talib.MACD(c)) if TL else None),
        ("stochastic", lambda: ta.stochastic(h, lo, c), stoch_old, (lambda: talib.STOCH(h, lo, c)) if TL else None),
        ("cci(20)", lambda: ta.cci(h, lo, c, 20), lambda: M.CCI._calculate_cci(h, lo, c, 20), (lambda: talib.CCI(h, lo, c, 20)) if TL else None),
        ("williams_r(14)", lambda: ta.williams_r(h, lo, c, 14), None, (lambda: talib.WILLR(h, lo, c, 14)) if TL else None),
        ("bop", lambda: ta.bop(o, h, lo, c), lambda: M.BOP._calculate_bop(o, h, lo, c), (lambda: talib.BOP(o, h, lo, c)) if TL else None),
        ("elderray(13)", lambda: ta.elderray(h, lo, c, 13), None, None),
        ("fisher(9)", lambda: ta.fisher(h, lo, 9), None, None),
        ("crsi", lambda: ta.crsi(c), None, None),

        ("[Volatility]", None, None, None),
        ("atr(14)", lambda: ta.atr(h, lo, c, 14), lambda: U.atr_wilder(h, lo, c, 14), (lambda: talib.ATR(h, lo, c, 14)) if TL else None),
        ("natr(14)", lambda: ta.natr(h, lo, c, 14), None, (lambda: talib.NATR(h, lo, c, 14)) if TL else None),
        ("bbands(20)", lambda: ta.bbands(c, 20), lambda: V.BollingerBands._calculate_bollinger_bands(c, 20, 2.0), (lambda: talib.BBANDS(c, 20)) if TL else None),
        ("keltner", lambda: ta.keltner(h, lo, c), lambda: V.Keltner._calculate_keltner_channel(h, lo, c, 20, 10, 2.0), None),
        ("donchian(20)", lambda: ta.donchian(h, lo, 20), None, None),
        ("chaikin", lambda: ta.chaikin(h, lo), None, None),
        ("trange", lambda: ta.true_range(h, lo, c), lambda: V.TRANGE._calculate_trange(h, lo, c), (lambda: talib.TRANGE(h, lo, c)) if TL else None),
        ("ultosc", lambda: ta.ultimate_oscillator(h, lo, c), lambda: V.ULTOSC._calculate_ultosc(h, lo, c, 7, 14, 28), (lambda: talib.ULTOSC(h, lo, c)) if TL else None),
        ("massindex", lambda: ta.massindex(h, lo), None, None),
        ("bbpercent(20)", lambda: ta.bbpercent(c, 20), None, None),
        ("bbwidth(20)", lambda: ta.bbwidth(c, 20), None, None),
        ("chandelier_exit", lambda: ta.chandelier_exit(h, lo, c), None, None),
        ("hv", lambda: ta.hv(c), None, None),
        ("ulcerindex", lambda: ta.ulcerindex(c), None, None),
        ("starc", lambda: ta.starc(h, lo, c), None, None),

        ("[Volume]", None, None, None),
        ("obv", lambda: ta.obv(c, v), lambda: VO.OBV._calculate_obv(c, v), (lambda: talib.OBV(c, v)) if TL else None),
        ("vwap", lambda: ta.vwap(h, lo, c, v), None, None),
        ("mfi(14)", lambda: ta.mfi(h, lo, c, v, 14), lambda: VO.MFI._calculate_mfi(h, lo, c, v, 14), (lambda: talib.MFI(h, lo, c, v, 14)) if TL else None),
        ("adl", lambda: ta.adl(h, lo, c, v), lambda: VO.ADL._calculate_adl(h, lo, c, v), (lambda: talib.AD(h, lo, c, v)) if TL else None),
        ("cmf(20)", lambda: ta.cmf(h, lo, c, v, 20), lambda: VO.CMF._calculate_cmf(h, lo, c, v, 20), None),
        ("emv", lambda: ta.emv(h, lo, v), None, None),
        ("force_index(13)", lambda: ta.force_index(c, v, 13), None, None),
        ("nvi", lambda: ta.nvi(c, v), None, None),
        ("pvi", lambda: ta.pvi(c, v), None, None),
        ("volosc", lambda: ta.volosc(v), None, None),
        ("vroc(25)", lambda: ta.vroc(v, 25), lambda: VO.VROC._calculate_vroc(v, 25), None),
        ("kvo", lambda: ta.kvo(h, lo, c, v), None, None),
        ("pvt", lambda: ta.pvt(c, v), lambda: VO.PriceVolumeTrend._calculate_pvt(c, v), None),
        ("rvol(20)", lambda: ta.rvol(v, 20), lambda: VO.RVOL._calculate_rvol(v, 20), None),

        ("[Oscillators]", None, None, None),
        ("roc(12)", lambda: ta.roc(c, 12), lambda: O.ROC._calculate_roc(c, 12), (lambda: talib.ROC(c, 12)) if TL else None),
        ("cmo(14)", lambda: ta.cmo(c, 14), lambda: O.CMO._calculate_cmo(c, 14), (lambda: talib.CMO(c, 14)) if TL else None),
        ("trix(18)", lambda: ta.trix(c, 18), None, (lambda: talib.TRIX(c, 18)) if TL else None),
        ("awesome_osc", lambda: ta.awesome_oscillator(h, lo), None, None),
        ("accel_osc", lambda: ta.accelerator_oscillator(h, lo), None, None),
        ("ppo", lambda: ta.ppo(c), None, (lambda: talib.PPO(c)) if TL else None),
        ("po", lambda: ta.po(c), None, None),
        ("dpo(21)", lambda: ta.dpo(c, 21), None, None),
        ("aroonosc(14)", lambda: ta.aroon_oscillator(h, lo, 14), lambda: O.AROONOSC._calculate_aroon_osc(h, lo, 14), (lambda: talib.AROONOSC(h, lo, 14)) if TL else None),
        ("stochrsi", lambda: ta.stochrsi(c), None, None),
        ("rvi_osc", lambda: ta.rvi(o, h, lo, c), None, None),
        ("cho", lambda: ta.cho(h, lo, c, v), None, (lambda: talib.ADOSC(h, lo, c, v)) if TL else None),
        ("chop(14)", lambda: ta.chop(h, lo, c, 14), None, None),
        ("kst", lambda: ta.kst(c), None, None),
        ("tsi", lambda: ta.tsi(c), None, None),
        ("vi(14)", lambda: ta.vi(h, lo, c, 14), None, None),
        ("gator", lambda: ta.gator_oscillator(h, lo), None, None),
        ("stc", lambda: ta.stc(c), None, None),
        ("coppock", lambda: ta.coppock(c), None, None),

        ("[Statistics]", None, None, None),
        ("linreg(14)", lambda: ta.linreg(c, 14), lambda: S.LINREG._calculate_linearreg(c, 14), (lambda: talib.LINEARREG(c, 14)) if TL else None),
        ("lrslope(100)", lambda: ta.lrslope(c, 100), None, (lambda: talib.LINEARREG_SLOPE(c, 100)) if TL else None),
        ("correlation(20)", lambda: ta.correlation(c, o, 20), lambda: S.CORREL._calculate_correl(c, o, 20), (lambda: talib.CORREL(c, o, 20)) if TL else None),
        ("beta(60)", lambda: ta.beta(c, o, 60), lambda: S.BETA._calculate_beta_optimized(c, o, 60), (lambda: talib.BETA(c, o, 60)) if TL else None),
        ("variance(20)", lambda: ta.variance(c, 20), None, (lambda: talib.VAR(c, 20)) if TL else None),
        ("tsf(14)", lambda: ta.tsf(c, 14), lambda: S.TSF._calculate_tsf(c, 14), (lambda: talib.TSF(c, 14)) if TL else None),
        ("median(3)", lambda: ta.median(c, 3), lambda: S.MEDIAN._calculate_median(c, 3), None),
        ("mode(20)", lambda: ta.mode(c, 20), None, None),

        ("[Hybrid]", None, None, None),
        ("adx(14)", lambda: ta.adx(h, lo, c, 14), adx_old, (lambda: talib.ADX(h, lo, c, 14)) if TL else None),
        ("aroon(14)", lambda: ta.aroon(h, lo, 14), lambda: H.Aroon._calculate_aroon(h, lo, 14), (lambda: talib.AROON(h, lo, 14)) if TL else None),
        ("pivot_points", lambda: ta.pivot_points(h, lo, c), lambda: H.PivotPoints._calculate_pivot_points(h, lo, c), None),
        ("psar", lambda: ta.psar(h, lo), lambda: H.SAR._calculate_sar(h, lo, 0.02, 0.2), (lambda: talib.SAR(h, lo)) if TL else None),
        ("dmi(14)", lambda: ta.dmi(h, lo, c, 14), None, None),
        ("fractals", lambda: ta.fractals(h, lo), None, None),
        ("rwi(14)", lambda: ta.rwi(h, lo, c, 14), lambda: H.RWI._calculate_rwi(h, lo, c, 14), None),
    ]

    lines = [
        "# OpenAlgo Indicators - Full Benchmark (NIFTY 50, 1-min)", "",
        f"- Dataset: `NIFTY 50.csv` - **{n:,} bars** (OHLC real; volume synthesized, index feed has volume=0).",
        "- **New (Rust)**: current `ta.*` backend (openalgo._oaindicators).",
        "- **Old (OpenAlgo)**: original kernels run INTERPRETED. numba does not import on this "
        "Python 3.14 / numpy 2.x env (the reason for the migration), so this is a conservative "
        "lower bound on the old JIT speed - real numba would be faster, so true speedups are smaller.",
        "- **TA-Lib**: shown where a comparable function exists (C implementation).",
        "- Times are best-of-3 (New, TA-Lib) / single-run (Old, since interpreted is slow), in ms.",
        "- `Speedup` = Old / New. `max|d|` = max abs difference New vs Old (accuracy).", "",
        "| Indicator | New Rust (ms) | Old OpenAlgo (ms) | TA-Lib (ms) | Speedup (Old/New) | max&#124;d&#124; New-vs-Old |",
        "|-----------|--------------:|------------------:|------------:|------------------:|---------------------:|",
    ]

    for label, newf, oldf, tlf in cases:
        if newf is None:
            lines.append(f"| **{label}** | | | | | |")
            continue
        try:
            nt, nv = best(newf, 3)
            ntxt = f"{nt:.2f}"
        except Exception as e:
            nt, nv, ntxt = None, None, f"err"
        if oldf is not None:
            try:
                ot, ov = best(oldf, 1)
                otxt = f"{ot:.1f}"
                sp = f"{ot/nt:.1f}x" if (nt and ot) else "-"
                md = maxdelta(nv, ov)
                mdtxt = f"{md:.1e}"
            except Exception:
                otxt, sp, mdtxt = "err", "-", "-"
        else:
            otxt, sp, mdtxt = "-", "-", "-"
        if tlf is not None:
            try:
                tt, _ = best(tlf, 3)
                ttxt = f"{tt:.2f}"
            except Exception:
                ttxt = "err"
        else:
            ttxt = "-"
        lines.append(f"| {label} | {ntxt} | {otxt} | {ttxt} | {sp} | {mdtxt} |")
        print(f"{label:20} new={ntxt:>8}  old={otxt:>9}  talib={ttxt:>8}  sp={sp:>7}  d={mdtxt}")

    out = Path(__file__).resolve().parent / "FULL_BENCHMARK.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

# Indicator Conventions

OpenAlgo's indicators run on a Rust core (`openalgo._oaindicators`), with a pure NumPy
fallback that returns the same values. Several classic indicators have more than one
published form; this page states which seed, smoothing and window each `ta` function
uses in this release, so its values can be reproduced by hand. A convention can change
between releases, for example to fix a defect: every such change is listed in
`CHANGELOG.md` with the values it moves (2.1.0 changes `ta.vwap`, `ta.psar` and the
`volatility.RVI` class on clean data, and several indicators around NaN input).

## Seeds, smoothing and windows

| Indicator(s) | Convention |
|---|---|
| EMA, DEMA, TEMA, T3, MACD, PPO | EMA with alpha = 2 / (period + 1), seeded with the first value |
| ATR, NATR, ADX, +DI/-DI | Wilder smoothing (alpha = 1 / period), seeded with the simple average of the first `period` values |
| WMA | Weights 1, 2, ..., `period`, the newest bar weighted most; NaN for the first `period - 1` bars |
| KAMA | Efficiency ratio over `length`; smoothing constant (er * (fast_alpha - slow_alpha) + slow_alpha)^2 from the fast and slow EMA lengths; the volatility sum is recomputed over the full window at every bar; seeded with the source |
| FRAMA | Midpoint (high + low) / 2. The filter is the midpoint for the first `period` bars; then D = (ln(N1 + N2) - ln(N3)) / ln 2, where N3 is the high-low range of the last `period` bars divided by `period` and N1, N2 the ranges of its newer and older halves divided by `period / 2` (D = 1 when a range is not positive), and filter = w * midpoint + (1 - w) * previous filter with w = exp(-4.6 * (D - 1)) held between 0.01 and 1. The output is the midpoint for the first `period + 1` bars and the mean of the last five filter values after that. `period` is even |
| Fisher Transform | Midpoint (high + low) / 2 and its position (price - lowest) / (highest - lowest) in the last `length` bars. value = 0.66 * (position - 0.5) + 0.67 * value[1], held at +-0.999 once it passes +-0.99; fisher = 0.5 * ln((1 + value) / (1 - value)) + 0.5 * fisher[1]; both start at 0 and stay unchanged on a window without range. The trigger is the previous bar's fisher (0 on the first bar that has one). NaN for the first `length - 1` bars |
| TRIX | 10000 * one-bar change of the triple EMA of log(close) |
| CMO | Plain sums of the rises and the falls over the window, no smoothing; 0 when nothing moved |
| CMF | sum(money flow volume) / sum(volume) over `period` bars, money flow volume = ((close - low) - (high - close)) / (high - low) * volume and 0 on a bar without range; 0 when the window's volume is not positive |
| MFI | Typical price (high + low + close) / 3 times volume, positive when the typical price rose from the bar before and negative when it fell; summed over `period` bars; 100 when the window has no negative flow; NaN on bar 0 |
| RVI (vigor, `ta.rvi`) | sum(swma(close - open)) / sum(swma(high - low)) over `period` bars, swma the 1-2-2-1 weighted mean of the last four bars; signal = swma(rvi) |
| RWI | The true range averaged with a simple mean over `period` bars, not Wilder smoothing; NaN for the first `period` bars and where that mean is 0 |
| Parabolic SAR | Starts long on the first bar (stop = low, extreme point = high). Each bar: stop += AF * (EP - stop); a reversal when the stop crosses the bar (strictly), restarting from max(high, EP) or min(low, EP) with AF reset; otherwise a new extreme raises AF by the step up to the maximum; last, the stop is held beyond the two previous bars' lows (long) or highs (short) |
| VWAP | Typical price (high + low + close) / 3 weighted by volume, restarted at each `anchor` period (`anchor=None`: one period from the first bar) |
| VAR | Sample variance, divided by (n - 1) |
| LRSLOPE | Change of the regression end value between two consecutive windows, divided by `interval` |
| AROON / AROONOSC | `length + 1` bar lookback; a tied extreme counts at its oldest bar |
| OBV | Starts at 0; a flat close (close == previous close) contributes 0 |
| ADL | Starts at 0 on the first bar; money-flow volume accumulates from the second bar |

## NaN input

Rolling-window kernels are window-local in NaN: a value is NaN while its window holds a
NaN and recovers once the NaN slides out. Recursive kernels (the EMA family) skip leading
NaNs, seed at the first finite value and carry their state across an interior NaN, so
only the NaN bar itself is NaN. FRAMA, the Fisher Transform, the Parabolic SAR and the
VWAP keep their state across a NaN bar and are NaN while it is in their window (the SAR
and the VWAP on that bar only).

## Single-formula indicators

| OpenAlgo | Formula |
|---|---|
| `ta.avgprice(o,h,l,c)` | (open+high+low+close)/4 |
| `ta.medprice(h,l)` | (high+low)/2 |
| `ta.typprice(h,l,c)` | (high+low+close)/3 |
| `ta.wclprice(h,l,c)` | (high+low+2*close)/4 |
| `ta.midpoint(data,n)` | (max+min)/2 of source over n |
| `ta.midprice(h,l,n)` | (highest(high,n)+lowest(low,n))/2 |
| `ta.mom(data,n)` | data - data[n] |
| `ta.rocp(data,n)` | (price-prev)/prev |
| `ta.rocr(data,n)` | price/prev |
| `ta.rocr100(data,n)` | price/prev*100 |
| `ta.apo(data,fast,slow,ma_type)` | MA(fast) - MA(slow), SMA or EMA |
| `ta.plus_dm(h,l,n)` | Wilder running sum of +DM |
| `ta.minus_dm(h,l,n)` | Wilder running sum of -DM |
| `ta.dx(h,l,c,n)` | 100*&#124;+DI - -DI&#124;/(+DI + -DI) |
| `ta.adxr(h,l,c,n)` | (ADX + ADX[n-1]) / 2 |
| `ta.stochf(h,l,c,fk,fd)` | fast %K, %D=SMA(%K,fd) -> (fastk, fastd) |
| `ta.linregangle(data,n)` | degrees(atan(OLS slope)) |
| `ta.linregintercept(data,n)` | OLS intercept b |

The directional-movement functions in this table (`plus_dm`, `minus_dm`, `dx`, `adxr`)
use Wilder running sums: each sum is seeded with the plain sum of the first `period-1`
one-bar movements, then updated as `dm = dm - dm/period + todayDM`. The first DX is at
index `period` and the first ADX (for ADXR) at `2*period-1`, the mean of `period` DX
values. This seed and warm-up differ from `ta.adx` and `ta.atr`, which average the first
`period` bars, so the two families are kept as separate functions.

## How the conventions are checked

`test/indicator_references.py` implements WMA, FRAMA, the Fisher Transform, CMO, CMF,
MFI, RVI, VWAP and the Parabolic SAR from the rows above with plain Python loops, and
`test/test_indicators_fixes.py` compares each with `ta` on both backends in CI, within
1e-9. The same file checks NaN handling and other indicators against values worked out
by hand.

## Performance

Most `ta` functions take time proportional to the number of bars, through running sums
or recursions. Functions that recompute a window at every bar take time proportional to
the number of bars times the window length: among them KAMA (its volatility sum over the
full window) and `ta.correlation`, which keeps a numerically stable two-pass computation
per window so that its output always stays within [-1, 1]. Timings on 924,782 NIFTY
1-minute bars are in `benchmark/FULL_BENCHMARK.md`.

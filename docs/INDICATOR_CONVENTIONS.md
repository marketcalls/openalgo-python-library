# Indicator Conventions

OpenAlgo's indicators run on a Rust core (`openalgo._oaindicators`), with a pure NumPy
fallback that returns the same values. Several classic indicators have more than one
published form; this page states which seed, smoothing and window each `ta` function
uses, so its values can be reproduced by hand. These conventions are kept fixed so that
existing users get identical output from one release to the next.

## Seeds, smoothing and windows

| Indicator(s) | Convention |
|---|---|
| EMA, DEMA, TEMA, T3, MACD, PPO | EMA with alpha = 2 / (period + 1), seeded with the first value |
| ATR, NATR, ADX, +DI/-DI | Wilder smoothing (alpha = 1 / period), seeded with the simple average of the first `period` values |
| KAMA | Efficiency ratio over `length`; smoothing constant (er * (fast_alpha - slow_alpha) + slow_alpha)^2 from the fast and slow EMA lengths; the volatility sum is recomputed over the full window at every bar; seeded with the source |
| TRIX | 10000 * one-bar change of the triple EMA of log(close) |
| CMO | Plain sums of the rises and the falls over the window, no smoothing |
| VAR | Sample variance, divided by (n - 1) |
| LRSLOPE | Change of the regression end value between two consecutive windows, divided by `interval` |
| AROON / AROONOSC | `length + 1` bar lookback; a tied extreme counts at its oldest bar |
| OBV | Starts at 0; a flat close (close == previous close) contributes 0 |
| ADL | Starts at 0 on the first bar; money-flow volume accumulates from the second bar |

Rolling-window kernels are window-local in NaN: a value is NaN while its window holds a
NaN and recovers once the NaN slides out. Recursive kernels (the EMA family) skip leading
NaNs, seed at the first finite value and carry their state across an interior NaN.

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

## Performance

All `ta.*` functions are O(n). Timings on 924,782 NIFTY 1-minute bars are in
`benchmark/FULL_BENCHMARK.md`. `ta.correlation` keeps a numerically stable two-pass
computation so its output always stays within [-1, 1].

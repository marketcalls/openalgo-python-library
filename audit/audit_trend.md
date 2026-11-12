# Trend Indicators Audit (openalgo.indicators.trend)

_Date_: 2025-06-11 19:30 IST_. The "Verified by" column, the EMA, WMA and KAMA rows,
the methodology and the performance notes were updated for release 2.1.0.

## Indicators Reviewed
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double EMA)
- TEMA (Triple EMA)
- HMA (Hull MA)
- VWMA (Volume-Weighted MA)
- ALMA (Arnaud Legoux MA)
- KAMA (Kaufman Adaptive MA)
- ZLEMA (Zero-Lag EMA)
- T3 (Tillson T3)
- FRAMA (Fractal Adaptive MA)
- Supertrend
- Ichimoku Cloud

## Findings & Actions
| Indicator | Status | Notes | Verified by (tolerance) |
|-----------|--------|-------|-------------------------|
| SMA | OK | Rolling window sum. | `benchmark/ci_smoke.py` (CI): equals the pandas rolling mean (atol 1e-9), with the same NaN positions on a series with NaN bars. `benchmark/wrapper_parity.py`: the same check (1e-9). |
| EMA | OK | alpha = 2 / (period + 1), seeded with the first value (`docs/INDICATOR_CONVENTIONS.md`). The 2025 note of an SMA seed no longer describes the code. | `benchmark/wrapper_parity.py`: equals the recursion written in the script from the first value (1e-9). `test/test_indicators_fixes.py` (CI): DEMA and TEMA values worked out by hand from that EMA (1e-9). |
| WMA | OK | Weights 1 to n, denominator n(n+1)/2, computed with running sums in the Rust core. | `test/test_indicators_fixes.py` (CI): the plain-Python reference `indicator_references.wma` on 200 bars (1e-9). `benchmark/parity.py` row `wma(14)`: a weighted average written in the script (atol 1e-12 or rtol 1e-9). `benchmark/wrapper_parity.py` row `wma vs manual`: the same reference as CI (1e-9). |
| DEMA | OK | `2 x EMA - EMA(EMA)`. | `test/test_indicators_fixes.py` (CI): a value worked out by hand after a NaN bar (1e-9). `benchmark/trend_parity.py` row `dema(20)`: equals a rebuild from `utils.ema` exactly. |
| TEMA | OK | `3 x EMA - 3 x EMA(EMA) + EMA(EMA(EMA))`. | `test/test_indicators_fixes.py` (CI): a value worked out by hand after a NaN bar (1e-9). `benchmark/trend_parity.py` row `tema(20)`: equals a rebuild from `utils.ema` exactly. |
| HMA | OK | `WMA(2 x WMA(n/2) - WMA(n), sqrt(n))`. | `benchmark/trend_parity.py` row `hma(20)`: equals that composition of the `WMA` class exactly. |
| VWMA | OK | Sum(P x V) / Sum(V); handles zero volume. | `benchmark/parity.py` row `vwma(14)` and `benchmark/trend_parity.py` row `vwma(20)`: equal `utils.vwma_optimized` (atol 1e-12 or rtol 1e-9; exact). |
| ALMA | OK | Gaussian weights centred at `offset x (n-1)`, sigma = n / sigma parameter. | `benchmark/trend_parity.py` row `alma(21)`: equals `ALMA._calculate_alma` (1e-9). |
| KAMA | OK | Efficiency ratio, smoothing constant and squared SC as in Kaufman (1998), with the volatility summed over the full window at every bar. | `benchmark/parity.py` row `kama_full_window(14,2,30)` and `benchmark/trend_parity.py` row `kama_full_window(14)`: equal `KAMA._calculate_kama_full_window` (atol 1e-12 or rtol 1e-9; 1e-9). |
| ZLEMA | OK | Lag = floor((n - 1) / 2); `EMA(2P - P_lag)`. | `benchmark/trend_parity.py` row `zlema(14)`: equals `ZLEMA._calculate_zlema_optimized` exactly. |
| T3 | OK | Three generalised DEMA passes, v-factor 0.7 by default. | `benchmark/trend_parity.py` row `t3(21,0.7)`: equals three passes of `T3._calculate_gd` exactly. |
| FRAMA | OK | Fractal dimension and adaptive weight per Ehlers; the full convention is in `docs/INDICATOR_CONVENTIONS.md`. | `test/test_indicators_fixes.py` (CI): a straight line worked out by hand, NaN handling, and the plain-Python reference `indicator_references.frama` on 200 bars shaped to reach every branch (1e-9). `benchmark/trend_parity.py` row `frama(26)`: the same reference (1e-8). |
| Supertrend | Fixed (2025) | Seed index shifted to the first bar with an ATR; arrays initialised to avoid NaN propagation. | `benchmark/trend_parity.py` rows `supertrend.line` and `supertrend.dir`: equal `Supertrend._calculate_supertrend` exactly. `benchmark/ci_smoke.py` (CI) checks only that the direction is +1 or -1. |
| Ichimoku | OK | Tenkan, Kijun, Senkou and Chikou lines and their displacements. | `benchmark/trend_parity.py` rows `ichimoku.*`: equal `Ichimoku._calculate_ichimoku` exactly. `test/test_indicators_causality.py` (CI): the lagging span is the close `displacement - 1` bars later, exactly. |

No formula discrepancies were detected in 2025; **no code changes were required**
then beyond the Supertrend fix.

## How the verdicts are checked
1. The 2025 review used 2 000 synthetic bars and daily BTC-USD data, compared the
   indicators with other indicator implementations and with spreadsheet
   calculations, accepted a mean absolute error below 1e-5 (ignoring the warm-up),
   and checked constant series, zero volume and series shorter than the period.
   Those comparisons are not kept in this repository, so they cannot be rerun and no
   longer count as evidence.
2. Each row now names the check in this repository that verifies it and its
   tolerance. The parity scripts in `benchmark/` are run by hand on data written by
   `benchmark/fetch_data.py` (from an OpenAlgo server); `benchmark/ci_smoke.py` and
   the tests under `test/` named above run in CI. "Exact" means equal to the last
   bit.

## Performance Notes (non-blocking)
- `WMA` now runs on running sums in the Rust core, O(n); the 2025 note of O(n x p)
  nested loops no longer applies.
- `ALMA` recomputes its weight vector on every call; it could be cached for repeated
  periods.

---
_The rows above say which check verifies each trend indicator._

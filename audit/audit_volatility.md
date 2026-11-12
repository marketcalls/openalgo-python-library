# Volatility Indicators Audit (openalgo.indicators.volatility)

_Date_: 2025-06-11 19:42 IST. The "Verified by" column, the RVI and MASS rows and the
methodology were updated for release 2.1.0.

## Indicators Reviewed
- ATR
- Bollinger Bands (BBANDS)
- Keltner Channel (KC)
- Donchian Channel (DC)
- Chaikin Volatility (CV)
- NATR
- RVI (Relative Volatility Index)
- ULTOSC (Ultimate Oscillator)
- STDDEV (rolling σ)
- TRANGE (True Range)
- MASS (Mass Index)

## Findings & Fixes
| Indicator | Status | Notes | Verified by (tolerance) |
|-----------|--------|-------|-------------------------|
| ATR | OK | Wilder smoothing, seeded with the mean of the first `period` true ranges. | `benchmark/parity.py` row `atr_wilder(14)`: equals `utils.atr_wilder` (atol 1e-12 or rtol 1e-9). |
| BBANDS | Partly verified | SMA +/- k x the population standard deviation (divisor *n*). | Not checked as a whole. Its middle band is `ta.sma` (`benchmark/ci_smoke.py`, CI: equals the pandas rolling mean within 1e-9) and its deviation is the `stdev` kernel (`benchmark/parity.py` row `stdev(14)`: equals `utils.stdev`, atol 1e-12 or rtol 1e-9); `ci_smoke.py` checks only that upper >= middle >= lower. %B and bandwidth are checked in `benchmark/volatility_parity.py` (1e-9). |
| Keltner Channel | Fixed (2025) | Middle line: an EMA seeded with the SMA of the first `period` values, NaN before; bands at +/- multiplier x Wilder ATR. | `benchmark/volatility_parity.py` rows `keltner.up`, `keltner.mid` and `keltner.low`: equal `Keltner`'s own kernels exactly. |
| Donchian Channel | OK | Highest high, lowest low and their average. | `benchmark/volatility_parity.py` rows `donchian.up` and `donchian.mid`: equal `Donchian._calculate_donchian_channel` exactly. |
| Chaikin Volatility | Fixed (2025) | ROC of an EMA of the high-low range, that EMA seeded with an SMA and NaN before. | `benchmark/volatility_parity.py` row `chaikin`: equals that formula written in the script exactly. |
| NATR | OK | ATR / close x 100. | `benchmark/volatility_parity.py` row `natr`: equals `utils.atr_wilder` / close x 100 exactly. |
| RVI | Fixed (2.1.0) | The published Relative Volatility Index: the population standard deviation of the last `stdev_period` closes, put on the up side when the close rose and on the down side when it fell, each side under Wilder's smoothing over `rsi_period`; RVI = 100 x U / (U + D). The 2025 verdict described the earlier formula (an RSI of the standard deviation), which 2.1.0 replaced (see `CHANGELOG.md`). | `test/test_indicators_fixes.py` (CI): series worked out by hand, the all-up, all-down and flat cases and NaN handling (1e-9). |
| ULTOSC | OK | BP/TR sums and weighted formula (7/14/28 default). | `benchmark/volatility_parity.py` row `ultosc`: equals `ULTOSC._calculate_ultosc` (1e-9). |
| STDDEV | OK | Rolling standard deviation (population). | `benchmark/parity.py` row `stdev(14)`: equals `utils.stdev` (atol 1e-12 or rtol 1e-9). `benchmark/ci_smoke.py` (CI) checks its NaN positions against the pandas rolling deviation (ddof 0). |
| TRANGE | OK | Largest of high - low, abs(high - previous close) and abs(low - previous close). | `benchmark/volatility_parity.py` row `trange`: equals `TRANGE._calculate_trange` exactly; `benchmark/parity.py` row `true_range`: equals `utils.true_range`. |
| MASS | OK | Sum over `length` of EMA(9) of the high-low range divided by EMA(9) of that EMA. The EMAs are seeded with their first value, as every `ta` EMA (`docs/INDICATOR_CONVENTIONS.md`); the 2025 note that they were reseeded with an SMA no longer describes the code. | `benchmark/volatility_parity.py` row `mass`: equals that formula written in the script exactly. |

## How the verdicts are checked
1. The 2025 review used 3 000 synthetic bars and daily BTC-USD data, compared the
   indicators with other indicator implementations and with spreadsheet calculations,
   and accepted a mean absolute difference below 1e-5. Those comparisons are not kept
   in this repository, so they cannot be rerun and no longer count as evidence.
2. Each row now names the check in this repository that verifies it and its
   tolerance. The parity scripts in `benchmark/` are run by hand on data written by
   `benchmark/fetch_data.py` (from an OpenAlgo server); `benchmark/ci_smoke.py` and
   `test/test_indicators_fixes.py` run in CI. "Exact" means equal to the last bit.

---
The 2025 EMA seeding corrections apply today to the Keltner Channel and Chaikin
Volatility; the Mass Index uses first-value-seeded EMAs.

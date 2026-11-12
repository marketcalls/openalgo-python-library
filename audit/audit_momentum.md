# Momentum Indicators Audit (openalgo.indicators.momentum)

_Date_: 2025-06-11 19:19 IST. The "Verified by" column and the methodology were
updated for release 2.1.0.

## Indicators Reviewed
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator (%K, %D)
- CCI (Commodity Channel Index)
- Williams %R

## Findings & Actions
| Indicator | Status | Notes | Verified by (tolerance) |
|-----------|--------|-------|-------------------------|
| RSI | OK | Wilder's RSI: averages seeded with the mean of the first `period` changes, then Wilder smoothing. | `benchmark/parity.py` row `rsi(14)`: equals the Python kernel `RSI._calculate_rsi` (atol 1e-12 or rtol 1e-9). `benchmark/ci_smoke.py` (CI) checks only that it stays within 0 to 100. |
| MACD | Unverified | Fast and slow EMA, signal EMA, histogram = MACD - signal (default 12-26-9). | No check compares MACD values. Its EMA is verified (see EMA in `audit_trend.md`); `benchmark/full_bench.py` reports its difference from `MACD._calculate_macd` without a pass or fail tolerance. |
| Stochastic | OK | %K over the window (bounds inclusive), %D = SMA(%K). | `benchmark/parity.py` rows `stoch_k` and `stoch_d`: equal `Stochastic._calculate_stochastic` fed the shared highest and lowest (atol 1e-12 or rtol 1e-9). |
| CCI | OK | Typical price, its SMA, the mean deviation and the constant 0.015. | `benchmark/parity.py` row `cci(14)`: equals `CCI._calculate_cci` (atol 1e-12 or rtol 1e-9). |
| Williams %R | OK | -100 x (HH - close) / (HH - LL). | `benchmark/parity.py` row `williams_r(14)`: equals `WilliamsR._calculate_williams_r` (atol 1e-12 or rtol 1e-9). |

No defects were discovered; **no code changes required**.

## How the verdicts are checked
1. The 2025 review computed each indicator on 2 000 synthetic bars (geometric Brownian
   motion) and on daily BTC-USD data, and compared the outputs with other indicator
   implementations. Those comparisons are not kept in this repository, so they cannot
   be rerun and no longer count as evidence.
2. Each row now names the check in this repository that verifies it. The parity
   scripts in `benchmark/` are run by hand on data written by
   `benchmark/fetch_data.py` (from an OpenAlgo server); `benchmark/ci_smoke.py` and
   `test/test_indicators_fixes.py` run in CI.
3. A row with no check that compares its values is marked Unverified.

## Conclusion
RSI, Stochastic, CCI and Williams %R are verified as stated; MACD is not checked as a
whole.

# Utility Functions Audit (openalgo.indicators.utils)

_Date_: 2025-06-11 19:38 IST_. The "Verified by" column, the `ema` row and the
methodology were updated for release 2.1.0.

## Functions Reviewed
- `crossover`
- `crossunder`
- `highest`
- `lowest`
- `change`
- `roc`
- `sma`
- `ema` (Exponential MA helper)
- `stdev`
- `validate_input`
- `true_range`

## Findings & Actions
| Function | Status | Notes | Verified by (tolerance) |
|----------|--------|-------|-------------------------|
| crossover / crossunder | OK | Fires when the first series moves from at or below the second to above it (crossover) or the reverse; a NaN on the bar or the bar before blocks it. Both now call the shared kernel. | `test/test_indicators_fixes.py` (CI): signals worked out by hand, including NaN bars, exactly. `benchmark/parity.py` rows `crossover`, `crossunder` and `cross`: comparisons written in the script, exactly. |
| highest / lowest | OK | Rolling maximum and minimum, NaN while the window holds a NaN. Both now call the shared kernel. | `test/test_indicators_fixes.py` (CI): windows with a NaN and periods longer than the series, worked out by hand (1e-9). `benchmark/parity.py` rows `highest(14)` and `lowest(14)`: equal the pandas rolling max and min exactly. |
| change | OK | data[i] - data[i - length]. Now calls the shared kernel. | `test/test_indicators_fixes.py` (CI): worked out by hand (1e-9). `benchmark/parity.py` row `change(1)`: equals the pandas shift difference exactly. |
| roc | OK | Percent change times 100; NaN where the earlier value is 0, never an error. | `benchmark/parity.py` row `roc(10)`: the Rust kernel equals `utils.roc` (atol 1e-12 or rtol 1e-9). |
| sma | OK | Rolling mean; NaN until `period - 1`. | `benchmark/parity.py` row `sma(14)`: the Rust kernel equals `utils.sma` (atol 1e-12 or rtol 1e-9), and `benchmark/ci_smoke.py` (CI) checks that kernel against the pandas rolling mean (atol 1e-9). |
| ema | OK (convention changed) | First-value seed, alpha = 2 / (period + 1), no NaN warm-up: the convention every `ta` EMA follows (`docs/INDICATOR_CONVENTIONS.md`). The 2025 note that it was changed to an SMA seed with NaN until `period - 1` no longer describes the code; the SMA-seeded EMA used by the Keltner Channel and Chaikin Volatility is a separate kernel. | `benchmark/parity.py` row `ema(14)`: the Rust kernel equals `utils.ema` (atol 1e-12 or rtol 1e-9); `benchmark/wrapper_parity.py`: `ta.ema` equals the first-value recursion written in the script (1e-9). |
| stdev | OK | Population standard deviation (ddof = 0) over the window. | `benchmark/parity.py` row `stdev(14)`: the Rust kernel equals `utils.stdev` (atol 1e-12 or rtol 1e-9); `benchmark/ci_smoke.py` (CI) checks its NaN positions against the pandas rolling deviation (ddof 0). |
| validate_input | Unverified | Accepts ndarray, Series or list and casts to float64. | No test calls it directly. |
| true_range | OK | Wilder's true range. | `benchmark/parity.py` row `true_range`: the Rust kernel equals `utils.true_range` (atol 1e-12 or rtol 1e-9); `benchmark/volatility_parity.py` row `trange`: equals `TRANGE._calculate_trange` exactly. |

## How the verdicts are checked
1. The 2025 review used a synthetic 10 000-point price series and OHLC sets, compared
   the outputs with pandas (`rolling`, `pct_change` and the like) and with another
   indicator implementation, accepted a mean absolute error below 1e-8 (exact for the
   boolean outputs), and checked constant series, zeros, leading NaNs and series
   shorter than the period. The comparisons with the other implementation are not
   kept in this repository, so they no longer count as evidence.
2. Each row now names the check in this repository that verifies it and its
   tolerance. The parity scripts in `benchmark/` are run by hand on data written by
   `benchmark/fetch_data.py` (from an OpenAlgo server); `benchmark/ci_smoke.py` and
   `test/test_indicators_fixes.py` run in CI. "Exact" means equal to the last bit.

---
The 2025 note on `ema` does not describe the code today (see its row); every other
utility function is verified as the table states, except `validate_input`.

# Utility Functions Audit (openalgo.indicators.utils)

_Date_: 2025-06-11 19:38 IST_

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
| Function | Status | Notes |
|----------|--------|-------|
| crossover / crossunder | OK | Boolean crossover detection matches an independent reference implementation; NaN-guarded. |
| highest / lowest | OK | Sliding window max/min using `.max()` / `.min()`; error-free. |
| change | OK | Simple diff over `length`; aligns with pandas `diff`. |
| roc | OK | % change times 100; ignores divide-by-zero. |
| sma | OK | Rolling mean via `np.mean` on window; produces NaNs until `period-1`. |
| ema | **Fixed** | Previously seeded EMA at index 0 → produced values before enough data accumulated and mismatched library EMA. Now returns NaNs until `period-1` and seeds with initial SMA, then smooths with α (standard). |
| stdev | OK | Population σ (ddof=0) over window; matches numpy rolling. |
| validate_input | OK | Accepts ndarray/Series/list, type-casts to float64, empties raise; robust. |
| true_range | OK | Wilder TR formula. |

## Test Methodology
1. Synthetic 10 000-point random price series plus OHLC sets.
2. Compared outputs to pandas equivalents (`rolling`, `pct_change`, etc.).
3. Error tolerance: MAE < 1 × 10⁻⁸ for numerical outputs; exact match for boolean arrays.
4. Edge cases: constant series, zeros, NaNs at start, short series (< period).

---
All utility functions validated; **only `ema` required adjustment**, which has been implemented.

# Statistical Indicators Audit (openalgo.indicators.statistics)

_Date_: 2025-06-11 19:27 IST_

## Indicators Reviewed
- LINEARREG (Linear Regression Value)
- LINEARREG_SLOPE
- CORREL (Pearson Correlation)
- BETA
- VAR (Rolling Variance)
- TSF (Time-Series Forecast)
- MEDIAN (Rolling Median)
- MODE (Rolling Mode)

## Findings & Actions
| Indicator | Status | Notes |
|-----------|--------|-------|
| LINEARREG | OK | Least-squares slope/intercept formula validated. |
| LINEARREG_SLOPE | OK | Slope component verified. |
| CORREL | OK | Pearson r computed correctly; sanitation when stdev=0 returns 0. |
| BETA | OK | Uses diff-returns; beta = covariance / variance. Off-by-one handled with `period+1` validator. |
| VAR | OK | Population variance over rolling window calculated; matches pandas `rolling(var, ddof=0)`. |
| TSF | OK | Forecast value `slope*period + intercept`. |
| MEDIAN | OK (perf note) | Bubble sort correct for periods ≤100; mean abs diff=0 vs `np.median` roll. Could be optimised but left unchanged. |
| MODE | OK | Discretised histogram approach returns modal bin centre; bins default 10. Edge cases (flat window) handled. |

No defects detected; **no code changes required**.

## Test Methodology
1. Generated 2 000 random‐walk prices and an independent market series.
2. Compared each indicator versus pandas equivalents (where available).
3. Criteria: `mean(|diff|) < 1e-5`, ignoring leading NaNs.
4. Manual sanity checks on edge cases (constant series, zero variance, flat windows).

## Performance Suggestions (not implemented)
• `MEDIAN` currently uses O(p²) bubble sort. For larger `period` consider in-place selection (Numba `np.partition`) to reduce complexity.

---
No code modifications committed for statistical indicators.

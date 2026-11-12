# Hybrid Indicators Audit (openalgo.indicators.hybrid)

_Date_: 2025-06-11 19:14 IST. The "Verified by" column and the methodology were
updated for release 2.1.0.

## Scope
Validated the following indicators:

| Indicator | Status | Notes | Verified by (tolerance) |
|-----------|--------|-------|-------------------------|
| ADX (+DI, -DI, ADX) | **Fixed** (2025) | ADX is the Wilder average of DX, seeded with the mean of the first `period` DX values: with period 14 the first +DI and -DI are at bar 13 and the first ADX at bar 26 (counting from 0). Before the 2025 fix ADX was seeded at `period`, which gave values too large and shifted left by `period - 1`. | `benchmark/hybrid_parity.py` rows `adx.di_plus`, `adx.di_minus` and `adx.adx`: equal `ADX._compute_dm` and `ADX._compute_di_dx` composed with `utils.ema_wilder`, exactly. |
| Aroon | OK | Formula and window logic correct. | `benchmark/hybrid_parity.py` rows `aroon.up` and `aroon.down`: equal `Aroon._calculate_aroon` exactly. |
| PivotPoints | OK | Traditional floor trader levels. | `test/test_indicators_fixes.py` (CI): levels worked out by hand, with and without an anchor (1e-9). `benchmark/hybrid_parity.py` rows `pp.*`: equal `PivotPoints._calculate_pivot_points` exactly. |
| Parabolic SAR (SAR/PSAR) | OK | Trend flips and the AF cap; 2.1.0 changed the stop on a reversal bar (see `CHANGELOG.md`). | `test/test_indicators_fixes.py` (CI): paths worked out by hand through reversals and the clamp, and the plain-Python reference `indicator_references.sar` on 200 bars (1e-9). `benchmark/hybrid_parity.py` rows `sar.val` and `sar.trend`: the same reference, exactly. |
| DMI | OK | The +DI and -DI of the ADX class. | As ADX: `benchmark/hybrid_parity.py` rows `adx.di_plus` and `adx.di_minus`. |
| HT_TRENDLINE | Removed | No `ta` function or class computes it in this release, so the 2025 verdict no longer applies. | None. |

## How the verdicts are checked
1. The 2025 review computed each indicator on 2 000 synthetic bars (geometric Brownian
   motion), compared ADX, Aroon and SAR with another indicator implementation
   (accepting a mean absolute difference below 1e-5), checked the pivot points by hand
   and inspected the first and last 10 rows. Those comparisons are not kept in this
   repository, so they cannot be rerun and no longer count as evidence.
2. Each row now names the check in this repository that verifies it and its
   tolerance. The parity scripts in `benchmark/` are run by hand on data written by
   `benchmark/fetch_data.py` (from an OpenAlgo server); `test/test_indicators_fixes.py`
   runs in CI. "Exact" means equal to the last bit.

## Outcome
Only **ADX** required changes in 2025. The ADX code has since moved to the Rust core
(`rust/oa_core/src/lib.rs`, `adx`); the row above states what it computes today.

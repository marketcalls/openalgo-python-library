# Oscillators Indicators Audit (openalgo.indicators.oscillators)

_Date_: 2025-06-11 19:25 IST. The "Verified by" column, the TRIX row and the
methodology were updated for release 2.1.0.

## Indicators Reviewed
- ROC (Rate of Change)
- CMO (Chande Momentum Oscillator)
- TRIX (Triple Exponential Average)
- UO (Ultimate Oscillator)
- AO (Awesome Oscillator)
- AC (Accelerator Oscillator)
- PPO (Percentage Price Oscillator)
- PO (Price Oscillator)
- DPO (Detrended Price Oscillator)
- AROONOSC (Aroon Oscillator)

## Findings & Actions
| Indicator | Status | Notes | Verified by (tolerance) |
|-----------|--------|-------|-------------------------|
| ROC | OK | Off-by-one handled. | `benchmark/oscillator_parity.py` row `roc`: equals `ROC._calculate_roc` exactly. |
| CMO | OK | Plain sums of the rises and the falls over the window. | `test/test_indicators_fixes.py` (CI): a hand-worked window, NaN handling, and the plain-Python reference `indicator_references.cmo` on 200 bars (1e-9). `benchmark/oscillator_parity.py` row `cmo`: the same reference (1e-9). |
| TRIX | OK (convention changed) | 10000 x the one-bar change of the triple EMA of log(close), as `docs/INDICATOR_CONVENTIONS.md` states. The 2025 note that the factor became x100 no longer describes the code. | `benchmark/oscillator_parity.py` row `trix`: equals that formula written in the script exactly. |
| UO | OK | Buying pressure and true range sums over the three windows. | `benchmark/oscillator_parity.py` row `uo`: equals `UO._calculate_uo` (1e-9); `benchmark/volatility_parity.py` row `ultosc`: equals `ULTOSC._calculate_ultosc` (1e-9). |
| AO | OK | SMA(5) - SMA(34) of the midpoint. | `benchmark/oscillator_parity.py` row `ao`: equals that difference of `AO._calculate_sma` (1e-9). |
| AC | **Fixed** (2025) | AC = AO - SMA(AO). The 2025 fix converted the AO output to NumPy before its SMA, which had raised a dtype error. | `benchmark/oscillator_parity.py` row `ac`: equals that formula written in the script (1e-9). |
| PPO | OK | Percentage difference of the fast and slow EMA, its signal and histogram. | `benchmark/oscillator_parity.py` rows `ppo.line` and `ppo.signal`: equal that formula written in the script exactly. |
| PO | OK | SMA and EMA branches. | `benchmark/oscillator_parity.py` rows `po.sma` (1e-9) and `po.ema` (exact): equal the difference of the class's own averages. |
| DPO | OK | Price shift n/2 + 1 against the SMA of n. | `benchmark/oscillator_parity.py` row `dpo`: equals that formula written in the script (1e-9). |
| AROONOSC | OK | Aroon up - Aroon down. | `benchmark/oscillator_parity.py` row `aroonosc`: equals `AROONOSC._calculate_aroon_osc` exactly. |

## How the verdicts are checked
1. The 2025 review computed each indicator on 2 000 synthetic bars (geometric Brownian
   motion), compared most of them with other indicator implementations and checked AC
   and PO in a spreadsheet, accepting a mean absolute difference below 1e-5. Those
   comparisons are not kept in this repository, so they cannot be rerun and no longer
   count as evidence.
2. Each row now names the check in this repository that verifies it and its
   tolerance. The parity scripts in `benchmark/` are run by hand on data written by
   `benchmark/fetch_data.py` (from an OpenAlgo server); `test/test_indicators_fixes.py`
   runs in CI. "Exact" means equal to the last bit.

## Code Changes (2025)
- `oscillators.py`
  - **AC.calculate**: convert the AO pandas Series to NumPy; use NumPy arrays for the
    SMA and the subtraction.

No breaking API changes.

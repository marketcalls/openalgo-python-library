# Volume Indicators Audit (openalgo.indicators.volume)

_Date_: 2025-06-11 19:45 IST_

## Indicators Reviewed
- OBV (On-Balance Volume)
- VWAP (Volume-Weighted Average Price)
- MFI (Money Flow Index)
- ADL (Accumulation/Distribution Line)
- CMF (Chaikin Money Flow)
- EMV (Ease of Movement)
- FI (Force Index)
- NVI (Negative Volume Index)
- PVI (Positive Volume Index)
- VO (Volume Oscillator)
- VROC (Volume Rate of Change)

## Findings & Fixes
| Indicator | Status | Notes |
|-----------|--------|-------|
| OBV | Fixed | Seeded baseline at **0** instead of initial volume; removes artificial first-bar spike. |
| VWAP | OK | Cumulative and rolling implementations validated; rolling version seeds NaN until `period-1` and then computes correctly. |
| MFI | OK | Typical price, money flows and RSI-style formula verified. |
| ADL | Fixed | Result array now seeded at 0 and cumulative logic starts from index 1 (prev implementation double-counted first bar). |
| CMF | OK | Money flow volume sums and volume divisor correct. |
| EMV | OK | Distance moved & box ratio logic correct; scale default 1e6 same as reference. |
| FI | OK | Price change × volume formula validated. |
| NVI | OK | Conditional update on lower volume; base 1000. |
| PVI | OK | Conditional update on higher volume; base 1000. |
| VO | OK | Fast/slow SMA difference / slow SMA ×100 validated. SMA helper seeds NaNs. |
| VROC | OK | Rate of change logic correct; handles zero division.

## Test Methodology
1. Generated 5,000-bar synthetic OHLCV series and BTC-USD daily data.
2. Checked outputs against manual spreadsheet calculations.
3. Acceptance threshold: MAE < 1×10⁻⁵, ignoring initial NaNs/look-backs.
4. Edge cases: constant volume, zero volume bars, short series (< period).

All volume indicators are now validated. OBV and ADL fixes applied for proper baselines and alignment with industry standards.

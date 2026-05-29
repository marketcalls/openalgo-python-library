# OpenAlgo Indicators - Full Benchmark (NIFTY 50, 1-min)

- Dataset: `NIFTY 50.csv` - **924,782 bars** (OHLC real; volume synthesized, index feed has volume=0).
- **New (Rust)**: current `ta.*` backend (openalgo._oaindicators).
- **Old (OpenAlgo)**: original kernels run INTERPRETED. numba does not import on this Python 3.14 / numpy 2.x env (the reason for the migration), so this is a conservative lower bound on the old JIT speed - real numba would be faster, so true speedups are smaller.
- **TA-Lib**: shown where a comparable function exists (C implementation).
- Times are best-of-3 (New, TA-Lib) / single-run (Old, since interpreted is slow), in ms.
- `Speedup` = Old / New. `max|d|` = max abs difference New vs Old (accuracy).

| Indicator | New Rust (ms) | Old OpenAlgo (ms) | TA-Lib (ms) | Speedup (Old/New) | max&#124;d&#124; New-vs-Old |
|-----------|--------------:|------------------:|------------:|------------------:|---------------------:|
| **[Trend]** | | | | | |
| sma(20) | 3.38 | 190.5 | 1.61 | 56.3x | 6.4e-10 |
| ema(20) | 2.24 | 195.8 | 1.93 | 87.6x | 0.0e+00 |
| wma(20) | 13.10 | 2581.0 | 1.52 | 197.0x | 0.0e+00 |
| dema(20) | 7.06 | - | 4.96 | - | - |
| tema(20) | 14.81 | - | 6.70 | - | - |
| trima(20) | 5775.08 | 5764.9 | 1.93 | 1.0x | 0.0e+00 |
| hma(20) | 29.03 | - | - | - | - |
| vwma(20) | 4.03 | 455.4 | - | 113.0x | 0.0e+00 |
| alma(21) | 9.02 | 3078.2 | - | 341.4x | 1.8e-11 |
| kama(14) | 7.16 | 3428.3 | 1.99 | 478.7x | 7.3e-12 |
| zlema(20) | 5.47 | 355.6 | - | 65.1x | 0.0e+00 |
| t3(21) | 19.65 | - | 2.33 | - | - |
| mcginley(14) | 9.18 | 438.9 | - | 47.8x | 3.6e-12 |
| vidya(14) | 52.70 | 3643.0 | - | 69.1x | 0.0e+00 |
| supertrend(10,3) | 21.71 | 1327.7 | - | 61.1x | 0.0e+00 |
| ichimoku | 68.40 | 2824.0 | - | 41.3x | 0.0e+00 |
| frama(26) | 45.89 | 11978.6 | - | 261.0x | 0.0e+00 |
| ckstop | 23.53 | 8644.4 | - | 367.4x | 0.0e+00 |
| **[Momentum]** | | | | | |
| rsi(14) | 5.40 | 481.7 | 4.75 | 89.2x | 0.0e+00 |
| macd | 7.11 | 580.3 | 8.24 | 81.6x | 0.0e+00 |
| stochastic | 26.34 | 1626.8 | 11.40 | 61.8x | 0.0e+00 |
| cci(20) | 13.86 | 3140.8 | 17.62 | 226.6x | 0.0e+00 |
| williams_r(14) | 23.07 | - | 7.65 | - | - |
| bop | 5.62 | 302.3 | 1.35 | 53.8x | 0.0e+00 |
| elderray(13) | 6.27 | - | - | - | - |
| fisher(9) | 15.80 | - | - | - | - |
| crsi | 37.70 | - | - | - | - |
| **[Volatility]** | | | | | |
| atr(14) | 8.61 | 572.9 | 5.41 | 66.5x | 0.0e+00 |
| natr(14) | 10.57 | - | 5.29 | - | - |
| bbands(20) | 10.69 | 9.4 | 5.66 | 0.9x | 0.0e+00 |
| keltner | 15.29 | 1602.2 | - | 104.8x | 0.0e+00 |
| donchian(20) | 22.17 | - | - | - | - |
| chaikin | 6.65 | - | - | - | - |
| trange | 3.73 | 375.3 | 1.12 | 100.6x | 0.0e+00 |
| ultosc | 35.66 | 9922.1 | 7.08 | 278.2x | 0.0e+00 |
| massindex | 13.80 | - | - | - | - |
| bbpercent(20) | 32.21 | - | - | - | - |
| bbwidth(20) | 30.02 | - | - | - | - |
| chandelier_exit | 33.85 | - | - | - | - |
| hv | 22.05 | - | - | - | - |
| ulcerindex | 21.33 | - | - | - | - |
| starc | 15.17 | - | - | - | - |
| **[Volume]** | | | | | |
| obv | 4.65 | 232.6 | 3.45 | 50.0x | 0.0e+00 |
| vwap | 14.20 | - | - | - | - |
| mfi(14) | 12.46 | 655.3 | 6.99 | 52.6x | 0.0e+00 |
| adl | 5.64 | 549.1 | 1.41 | 97.3x | 0.0e+00 |
| cmf(20) | 23.64 | 11828.5 | - | 500.3x | 0.0e+00 |
| emv | 8.85 | - | - | - | - |
| force_index(13) | 11.48 | - | - | - | - |
| nvi | 10.87 | - | - | - | - |
| pvi | 10.24 | - | - | - | - |
| volosc | 14.90 | - | - | - | - |
| vroc(25) | 7.27 | 268.7 | - | 37.0x | 0.0e+00 |
| kvo | 18.43 | - | - | - | - |
| pvt | 11.19 | 368.8 | - | 33.0x | 0.0e+00 |
| rvol(20) | 10.04 | 1555.8 | - | 154.9x | 0.0e+00 |
| **[Oscillators]** | | | | | |
| roc(12) | 269.32 | 270.0 | 1.30 | 1.0x | 0.0e+00 |
| cmo(14) | 10.56 | 1886.6 | 4.88 | 178.7x | 1.4e-14 |
| trix(18) | 11.89 | - | 6.60 | - | - |
| awesome_osc | 18.25 | - | - | - | - |
| accel_osc | 21.02 | - | - | - | - |
| ppo | 12.15 | - | 4.07 | - | - |
| po | 11.40 | - | - | - | - |
| dpo(21) | 11.02 | - | - | - | - |
| aroonosc(14) | 50.83 | 2777.6 | 7.39 | 54.6x | 0.0e+00 |
| stochrsi | 8044.46 | - | - | - | - |
| rvi_osc | 18.11 | - | - | - | - |
| cho | 20.27 | - | 1.82 | - | - |
| chop(14) | 37.00 | - | - | - | - |
| kst | 55.70 | - | - | - | - |
| tsi | 15.96 | - | - | - | - |
| vi(14) | 27.91 | - | - | - | - |
| gator | 32.52 | - | - | - | - |
| stc | 22.75 | - | - | - | - |
| coppock | 16.62 | - | - | - | - |
| **[Statistics]** | | | | | |
| linreg(14) | 10.13 | 7343.2 | 9.06 | 725.0x | 3.3e-11 |
| lrslope(100) | 136.74 | - | 52.10 | - | - |
| correlation(20) | 23.15 | 11305.2 | 5.50 | 488.3x | 1.0e+00 |
| beta(60) | 65.88 | 20089.0 | 4.65 | 304.9x | 0.0e+00 |
| variance(20) | 7.75 | - | 2.16 | - | - |
| tsf(14) | 10.93 | 7291.6 | 8.95 | 667.0x | 3.6e-11 |
| median(3) | 10.47 | 1016.6 | - | 97.1x | 0.0e+00 |
| mode(20) | 102.25 | - | - | - | - |
| **[Hybrid]** | | | | | |
| adx(14) | 36.92 | 4845.3 | 7.43 | 131.2x | 0.0e+00 |
| aroon(14) | 48.28 | 3016.9 | 8.22 | 62.5x | 0.0e+00 |
| pivot_points | 20.69 | 1316.0 | - | 63.6x | 0.0e+00 |
| psar | 10.78 | 570.0 | 4.25 | 52.9x | 0.0e+00 |
| dmi(14) | 37.77 | - | - | - | - |
| fractals | 16.98 | - | - | - | - |
| rwi(14) | 13.73 | 4335.9 | - | 315.8x | 0.0e+00 |

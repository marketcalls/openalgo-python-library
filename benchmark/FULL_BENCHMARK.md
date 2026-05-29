# OpenAlgo Indicators - Full Benchmark (NIFTY 50, 1-min)

- Dataset: `NIFTY 50.csv` - **5,000 bars** (OHLC real; volume synthesized, index feed has volume=0).
- **New (Rust)**: current `ta.*` backend (openalgo._oaindicators).
- **Old (OpenAlgo)**: original kernels run INTERPRETED. numba does not import on this Python 3.14 / numpy 2.x env (the reason for the migration), so this is a conservative lower bound on the old JIT speed - real numba would be faster, so true speedups are smaller.
- **TA-Lib**: shown where a comparable function exists (C implementation).
- Times are best-of-3 (New, TA-Lib) / single-run (Old, since interpreted is slow), in ms.
- `Speedup` = Old / New. `max|d|` = max abs difference New vs Old (accuracy).

| Indicator | New Rust (ms) | Old OpenAlgo (ms) | TA-Lib (ms) | Speedup (Old/New) | max&#124;d&#124; New-vs-Old |
|-----------|--------------:|------------------:|------------:|------------------:|---------------------:|
| **[Trend]** | | | | | |
| sma(20) | 0.01 | 1.0 | 0.01 | 101.6x | 0.0e+00 |
| ema(20) | 0.01 | 1.0 | 0.01 | 100.8x | 0.0e+00 |
| wma(20) | 0.06 | 14.0 | 0.01 | 223.4x | 0.0e+00 |
| dema(20) | 0.02 | - | 0.02 | - | - |
| tema(20) | 0.04 | - | 0.03 | - | - |
| trima(20) | 30.78 | 31.9 | 0.01 | 1.0x | 0.0e+00 |
| hma(20) | 0.11 | - | - | - | - |
| vwma(20) | 0.01 | 2.5 | - | 221.9x | 0.0e+00 |
| alma(21) | 0.04 | 16.4 | - | 380.3x | 5.5e-12 |
| kama(14) | 0.03 | 18.7 | 0.01 | 553.1x | 0.0e+00 |
| zlema(20) | 0.01 | 1.9 | - | 138.6x | 0.0e+00 |
| t3(21) | 0.05 | - | 0.01 | - | - |
| mcginley(14) | 0.04 | 2.6 | - | 63.9x | 0.0e+00 |
| vidya(14) | 0.25 | 22.2 | - | 88.9x | 0.0e+00 |
| supertrend(10,3) | 0.05 | 7.0 | - | 136.4x | 0.0e+00 |
| ichimoku | 0.26 | 15.4 | - | 59.6x | 0.0e+00 |
| frama(26) | 0.23 | 64.7 | - | 284.1x | 0.0e+00 |
| ckstop | 0.08 | 46.2 | - | 559.3x | 0.0e+00 |
| **[Momentum]** | | | | | |
| rsi(14) | 0.02 | 3.4 | 0.02 | 150.5x | 0.0e+00 |
| macd | 0.03 | 3.9 | 0.03 | 140.6x | 0.0e+00 |
| stochastic | 0.09 | 8.9 | 0.03 | 103.2x | 0.0e+00 |
| cci(20) | 0.06 | 17.9 | 0.09 | 320.2x | 0.0e+00 |
| williams_r(14) | 0.07 | - | 0.03 | - | - |
| bop | 0.01 | 1.7 | 0.01 | 137.1x | 0.0e+00 |
| elderray(13) | 0.01 | - | - | - | - |
| fisher(9) | 0.06 | - | - | - | - |
| crsi | 0.14 | - | - | - | - |
| **[Volatility]** | | | | | |
| atr(14) | 0.03 | 3.0 | 0.02 | 117.2x | 0.0e+00 |
| natr(14) | 0.03 | - | 0.02 | - | - |
| bbands(20) | 0.03 | 0.0 | 0.02 | 0.9x | 0.0e+00 |
| keltner | 0.04 | 8.6 | - | 217.3x | 0.0e+00 |
| donchian(20) | 0.08 | - | - | - | - |
| chaikin | 0.02 | - | - | - | - |
| trange | 0.01 | 2.1 | 0.00 | 307.5x | 0.0e+00 |
| ultosc | 0.09 | 53.0 | 0.03 | 564.6x | 0.0e+00 |
| massindex | 0.04 | - | - | - | - |
| bbpercent(20) | 0.13 | - | - | - | - |
| bbwidth(20) | 0.12 | - | - | - | - |
| chandelier_exit | 0.12 | - | - | - | - |
| hv | 0.08 | - | - | - | - |
| ulcerindex | 0.05 | - | - | - | - |
| starc | 0.04 | - | - | - | - |
| **[Volume]** | | | | | |
| obv | 0.02 | 1.3 | 0.01 | 81.5x | 0.0e+00 |
| vwap | 0.04 | - | - | - | - |
| mfi(14) | 0.03 | 3.5 | 0.01 | 129.3x | 0.0e+00 |
| adl | 0.01 | 3.1 | 0.01 | 266.0x | 0.0e+00 |
| cmf(20) | 0.11 | 63.7 | - | 556.4x | 0.0e+00 |
| emv | 0.03 | - | - | - | - |
| force_index(13) | 0.03 | - | - | - | - |
| nvi | 0.04 | - | - | - | - |
| pvi | 0.04 | - | - | - | - |
| volosc | 0.05 | - | - | - | - |
| vroc(25) | 0.02 | 1.5 | - | 79.3x | 0.0e+00 |
| kvo | 0.04 | - | - | - | - |
| pvt | 0.03 | 2.1 | - | 63.7x | 0.0e+00 |
| rvol(20) | 0.04 | 8.4 | - | 194.9x | 0.0e+00 |
| **[Oscillators]** | | | | | |
| roc(12) | 1.46 | 1.5 | 0.01 | 1.0x | 0.0e+00 |
| cmo(14) | 0.02 | 10.8 | 0.02 | 528.9x | 1.4e-14 |
| trix(18) | 0.04 | - | 0.03 | - | - |
| awesome_osc | 0.07 | - | - | - | - |
| accel_osc | 0.08 | - | - | - | - |
| ppo | 0.04 | - | 0.02 | - | - |
| po | 0.05 | - | - | - | - |
| dpo(21) | 0.04 | - | - | - | - |
| aroonosc(14) | 0.24 | 15.1 | 0.03 | 62.3x | 0.0e+00 |
| stochrsi | 42.96 | - | - | - | - |
| rvi_osc | 0.05 | - | - | - | - |
| cho | 0.05 | - | 0.01 | - | - |
| chop(14) | 0.13 | - | - | - | - |
| kst | 0.15 | - | - | - | - |
| tsi | 0.05 | - | - | - | - |
| vi(14) | 0.06 | - | - | - | - |
| gator | 0.10 | - | - | - | - |
| stc | 0.11 | - | - | - | - |
| coppock | 0.07 | - | - | - | - |
| **[Statistics]** | | | | | |
| linreg(14) | 0.05 | 41.5 | 0.04 | 903.3x | 1.3e-11 |
| lrslope(100) | 0.68 | - | 0.27 | - | - |
| correlation(20) | 0.12 | 60.9 | 0.03 | 524.9x | 1.2e-15 |
| beta(60) | 0.32 | 106.3 | 0.02 | 334.7x | 0.0e+00 |
| variance(20) | 0.03 | - | 0.01 | - | - |
| tsf(14) | 0.05 | 39.6 | 0.05 | 800.8x | 1.5e-11 |
| median(3) | 0.05 | 5.5 | - | 120.9x | 0.0e+00 |
| mode(20) | 0.54 | - | - | - | - |
| **[Hybrid]** | | | | | |
| adx(14) | 0.14 | 26.2 | 0.03 | 187.8x | 0.0e+00 |
| aroon(14) | 0.24 | 16.5 | 0.02 | 67.7x | 0.0e+00 |
| pivot_points | 0.03 | 7.2 | - | 224.6x | 0.0e+00 |
| psar | 0.03 | 3.6 | 0.01 | 108.7x | 0.0e+00 |
| dmi(14) | 0.15 | - | - | - | - |
| fractals | 0.11 | - | - | - | - |
| rwi(14) | 0.07 | 24.1 | - | 356.2x | 0.0e+00 |

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
| sma(20) | 3.64 | 193.5 | 1.93 | 53.1x | 6.4e-10 |
| ema(20) | 3.91 | 197.3 | 2.05 | 50.5x | 0.0e+00 |
| wma(20) | 4.12 | 2893.0 | 2.80 | 701.5x | 4.1e-06 |
| dema(20) | 11.04 | - | 5.85 | - | - |
| tema(20) | 21.81 | - | 7.69 | - | - |
| trima(20) | 5.79 | 6858.2 | 2.50 | 1184.3x | 2.4e-09 |
| hma(20) | 19.01 | - | - | - | - |
| vwma(20) | 6.95 | 598.2 | - | 86.0x | 0.0e+00 |
| alma(21) | 12.52 | 3650.7 | - | 291.6x | 1.8e-11 |
| kama(14) | 7.83 | 3513.4 | 2.07 | 448.8x | 7.3e-12 |
| zlema(20) | 5.55 | 356.3 | - | 64.2x | 0.0e+00 |
| t3(21) | 14.00 | - | 2.34 | - | - |
| mcginley(14) | 8.92 | 452.3 | - | 50.7x | 3.6e-12 |
| vidya(14) | 56.75 | 3897.8 | - | 68.7x | 0.0e+00 |
| supertrend(10,3) | 23.91 | 1327.3 | - | 55.5x | 0.0e+00 |
| ichimoku | 66.90 | 2837.9 | - | 42.4x | 0.0e+00 |
| frama(26) | 45.49 | 12245.8 | - | 269.2x | 0.0e+00 |
| ckstop | 25.71 | 8985.3 | - | 349.5x | 0.0e+00 |
| **[Momentum]** | | | | | |
| rsi(14) | 5.37 | 494.8 | 4.78 | 92.1x | 0.0e+00 |
| macd | 7.33 | 586.8 | 8.59 | 80.0x | 0.0e+00 |
| stochastic | 26.52 | 1649.8 | 11.49 | 62.2x | 0.0e+00 |
| cci(20) | 15.13 | 3340.7 | 17.29 | 220.9x | 0.0e+00 |
| williams_r(14) | 20.90 | - | 7.62 | - | - |
| bop | 5.99 | 311.1 | 1.42 | 51.9x | 0.0e+00 |
| elderray(13) | 6.57 | - | - | - | - |
| fisher(9) | 16.89 | - | - | - | - |
| crsi | 41.49 | - | - | - | - |
| **[Volatility]** | | | | | |
| atr(14) | 9.56 | 582.0 | 5.13 | 60.9x | 0.0e+00 |
| natr(14) | 10.41 | - | 5.29 | - | - |
| bbands(20) | 11.38 | 9.9 | 5.54 | 0.9x | 0.0e+00 |
| keltner | 17.74 | 1622.8 | - | 91.5x | 0.0e+00 |
| donchian(20) | 20.76 | - | - | - | - |
| chaikin | 7.43 | - | - | - | - |
| trange | 4.13 | 382.6 | 1.26 | 92.7x | 0.0e+00 |
| ultosc | 9.56 | 10034.5 | 6.72 | 1049.1x | 0.0e+00 |
| massindex | 14.87 | - | - | - | - |
| bbpercent(20) | 34.56 | - | - | - | - |
| bbwidth(20) | 32.97 | - | - | - | - |
| chandelier_exit | 34.78 | - | - | - | - |
| hv | 24.79 | - | - | - | - |
| ulcerindex | 20.00 | - | - | - | - |
| starc | 16.85 | - | - | - | - |
| **[Volume]** | | | | | |
| obv | 4.78 | 279.6 | 3.46 | 58.5x | 0.0e+00 |
| vwap | 16.23 | - | - | - | - |
| mfi(14) | 13.60 | 664.9 | 7.01 | 48.9x | 0.0e+00 |
| adl | 5.99 | 560.2 | 1.61 | 93.5x | 0.0e+00 |
| cmf(20) | 26.50 | 12623.3 | - | 476.3x | 0.0e+00 |
| emv | 9.26 | - | - | - | - |
| force_index(13) | 11.40 | - | - | - | - |
| nvi | 12.42 | - | - | - | - |
| pvi | 12.20 | - | - | - | - |
| volosc | 17.13 | - | - | - | - |
| vroc(25) | 8.94 | 298.8 | - | 33.4x | 0.0e+00 |
| kvo | 25.93 | - | - | - | - |
| pvt | 12.79 | 439.9 | - | 34.4x | 0.0e+00 |
| rvol(20) | 11.05 | 1628.6 | - | 147.4x | 0.0e+00 |
| **[Oscillators]** | | | | | |
| roc(12) | 1.66 | 299.6 | 1.36 | 180.6x | 0.0e+00 |
| cmo(14) | 10.73 | 2146.2 | 4.92 | 200.0x | 1.4e-14 |
| trix(18) | 11.94 | - | 6.47 | - | - |
| awesome_osc | 17.93 | - | - | - | - |
| accel_osc | 21.22 | - | - | - | - |
| ppo | 12.45 | - | 3.88 | - | - |
| po | 11.13 | - | - | - | - |
| dpo(21) | 11.20 | - | - | - | - |
| aroonosc(14) | 19.63 | 2901.9 | 7.58 | 147.9x | 0.0e+00 |
| stochrsi | 20.39 | - | - | - | - |
| rvi_osc | 17.88 | - | - | - | - |
| cho | 9.77 | - | 1.88 | - | - |
| chop(14) | 36.23 | - | - | - | - |
| kst | 55.25 | - | - | - | - |
| tsi | 16.45 | - | - | - | - |
| vi(14) | 27.34 | - | - | - | - |
| gator | 31.80 | - | - | - | - |
| stc | 28.59 | - | - | - | - |
| coppock | 16.96 | - | - | - | - |
| **[Statistics]** | | | | | |
| linreg(14) | 6.23 | 7835.0 | 9.03 | 1258.0x | 6.6e-05 |
| lrslope(100) | 6.31 | - | 52.15 | - | - |
| correlation(20) | 23.72 | 11775.6 | 5.46 | 496.4x | 1.0e+00 |
| beta(60) | 8.29 | 21104.0 | 4.62 | 2545.6x | 7.9e-12 |
| variance(20) | 7.22 | - | 2.03 | - | - |
| tsf(14) | 6.37 | 7589.8 | 9.13 | 1191.1x | 7.6e-05 |
| median(3) | 10.59 | 1051.6 | - | 99.3x | 0.0e+00 |
| mode(20) | 103.66 | - | - | - | - |
| **[Hybrid]** | | | | | |
| adx(14) | 16.57 | 5042.9 | 7.46 | 304.4x | 0.0e+00 |
| aroon(14) | 17.70 | 3117.2 | 8.21 | 176.1x | 0.0e+00 |
| pivot_points | 22.58 | 1394.3 | - | 61.7x | 0.0e+00 |
| psar | 11.09 | 557.0 | 4.19 | 50.2x | 0.0e+00 |
| dmi(14) | 16.97 | - | - | - | - |
| fractals | 17.54 | - | - | - | - |
| rwi(14) | 12.66 | 4602.4 | - | 363.5x | 0.0e+00 |

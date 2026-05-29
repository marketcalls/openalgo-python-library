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
| sma(20) | 0.01 | 1.0 | 0.01 | 106.4x | 0.0e+00 |
| ema(20) | 0.01 | 1.0 | 0.01 | 117.6x | 0.0e+00 |
| wma(20) | 0.06 | 14.3 | 0.01 | 231.2x | 0.0e+00 |
| dema(20) | 0.02 | - | 0.02 | - | - |
| tema(20) | 0.04 | - | 0.03 | - | - |
| trima(20) | 30.73 | 30.9 | 0.01 | 1.0x | 0.0e+00 |
| hma(20) | 0.11 | - | - | - | - |
| vwma(20) | 0.01 | 2.7 | - | 239.4x | 0.0e+00 |
| alma(21) | 0.03 | 19.5 | - | 675.6x | 5.5e-12 |
| kama(14) | 0.03 | 19.2 | 0.01 | 560.9x | 0.0e+00 |
| zlema(20) | 0.01 | 2.2 | - | 162.0x | 0.0e+00 |
| t3(21) | 0.05 | - | 0.01 | - | - |
| mcginley(14) | 0.04 | 2.8 | - | 69.5x | 0.0e+00 |
| vidya(14) | 0.25 | 21.9 | - | 87.4x | 0.0e+00 |
| supertrend(10,3) | 0.05 | 7.6 | - | 165.4x | 0.0e+00 |
| ichimoku | 0.24 | 15.3 | - | 62.5x | 0.0e+00 |
| frama(26) | 0.23 | 65.7 | - | 283.1x | 0.0e+00 |
| ckstop | 0.09 | 47.0 | - | 536.0x | 0.0e+00 |
| **[Momentum]** | | | | | |
| rsi(14) | 0.02 | 2.8 | 0.02 | 123.5x | 0.0e+00 |
| macd | 0.02 | 3.5 | 0.03 | 146.8x | 0.0e+00 |
| stochastic | 0.09 | 9.6 | 0.03 | 112.3x | 0.0e+00 |
| cci(20) | 0.06 | 19.7 | 0.09 | 357.1x | 0.0e+00 |
| williams_r(14) | 0.07 | - | 0.02 | - | - |
| bop | 0.01 | 2.0 | 0.01 | 173.7x | 0.0e+00 |
| elderray(13) | 0.01 | - | - | - | - |
| fisher(9) | 0.06 | - | - | - | - |
| crsi | 0.14 | - | - | - | - |
| **[Volatility]** | | | | | |
| atr(14) | 0.03 | 3.6 | 0.02 | 140.8x | 0.0e+00 |
| natr(14) | 0.03 | - | 0.02 | - | - |
| bbands(20) | 0.03 | 0.0 | 0.02 | 1.0x | 0.0e+00 |
| keltner | 0.04 | 9.5 | - | 245.1x | 0.0e+00 |
| donchian(20) | 0.10 | - | - | - | - |
| chaikin | 0.03 | - | - | - | - |
| trange | 0.01 | 2.5 | 0.00 | 225.6x | 0.0e+00 |
| ultosc | 0.10 | 52.7 | 0.03 | 547.4x | 0.0e+00 |
| massindex | 0.04 | - | - | - | - |
| bbpercent(20) | 42.50 | - | - | - | - |
| bbwidth(20) | 42.73 | - | - | - | - |
| chandelier_exit | 11.83 | - | - | - | - |
| hv | 31.29 | - | - | - | - |
| ulcerindex | 0.06 | - | - | - | - |
| starc | 22.89 | - | - | - | - |
| **[Volume]** | | | | | |
| obv | 0.02 | 1.3 | 0.01 | 81.4x | 0.0e+00 |
| vwap | 0.03 | - | - | - | - |
| mfi(14) | 0.03 | 3.8 | 0.01 | 135.6x | 0.0e+00 |
| adl | 0.01 | 3.9 | 0.01 | 326.7x | 0.0e+00 |
| cmf(20) | 0.11 | 70.7 | - | 663.6x | 0.0e+00 |
| emv | 11.50 | - | - | - | - |
| force_index(13) | 0.03 | - | - | - | - |
| nvi | 0.03 | - | - | - | - |
| pvi | 0.03 | - | - | - | - |
| volosc | 0.05 | - | - | - | - |
| vroc(25) | 0.02 | 1.5 | - | 82.6x | 0.0e+00 |
| kvo | 0.04 | - | - | - | - |
| pvt | 0.03 | 2.7 | - | 82.6x | 0.0e+00 |
| rvol(20) | 11.49 | 10.1 | - | 0.9x | 0.0e+00 |
| **[Oscillators]** | | | | | |
| roc(12) | 1.71 | 1.7 | 0.01 | 1.0x | 0.0e+00 |
| cmo(14) | 0.02 | 11.3 | 0.02 | 528.7x | 1.4e-14 |
| trix(18) | 0.04 | - | 0.03 | - | - |
| awesome_osc | 22.96 | - | - | - | - |
| accel_osc | 34.56 | - | - | - | - |
| ppo | 0.04 | - | 0.02 | - | - |
| po | 22.90 | - | - | - | - |
| dpo(21) | 15.07 | - | - | - | - |
| aroonosc(14) | 17.05 | 18.1 | 0.02 | 1.1x | 0.0e+00 |
| stochrsi | 42.80 | - | - | - | - |
| rvi_osc | 63.62 | - | - | - | - |
| cho | 0.06 | - | 0.01 | - | - |
| chop(14) | 0.15 | - | - | - | - |
| kst | 57.54 | - | - | - | - |
| tsi | 0.06 | - | - | - | - |
| vi(14) | 0.07 | - | - | - | - |
| gator | 0.10 | - | - | - | - |
| stc | 35.16 | - | - | - | - |
| coppock | 40.49 | - | - | - | - |
| **[Statistics]** | | | | | |
| linreg(14) | 0.05 | 39.6 | 0.04 | 809.1x | 1.3e-11 |
| lrslope(100) | 0.69 | - | 0.27 | - | - |
| correlation(20) | 0.11 | 60.3 | 0.03 | 529.4x | 1.2e-15 |
| beta(60) | 0.35 | 127.1 | 0.02 | 359.3x | 0.0e+00 |
| variance(20) | 8.51 | - | 0.01 | - | - |
| tsf(14) | 0.05 | 39.8 | 0.04 | 813.3x | 1.5e-11 |
| median(3) | 37.26 | 5.5 | - | 0.1x | 0.0e+00 |
| mode(20) | 50.31 | - | - | - | - |
| **[Hybrid]** | | | | | |
| adx(14) | 7.19 | 27.4 | 0.03 | 3.8x | 0.0e+00 |
| aroon(14) | 19.31 | 21.3 | 0.03 | 1.1x | 0.0e+00 |
| pivot_points | 0.03 | 8.0 | - | 273.6x | 0.0e+00 |
| psar | 0.03 | 3.5 | 0.01 | 103.1x | 0.0e+00 |
| dmi(14) | 7.08 | - | - | - | - |
| fractals | 19.92 | - | - | - | - |
| rwi(14) | 14.17 | 27.2 | - | 1.9x | 0.0e+00 |

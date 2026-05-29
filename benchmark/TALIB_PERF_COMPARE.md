# OpenAlgo vs TA-Lib - Performance Comparison

- Dataset: `NIFTY 50.csv` - **924,782 bars**. Best-of-5 timings.
- **New (ms)** = OpenAlgo `ta.*` (Rust core). **TA-Lib (ms)** = C implementation.
- **Speed (TA-Lib/New)**: >1.0 = OpenAlgo faster; <1.0 = TA-Lib faster (target).
- **max|d|** = max abs difference OpenAlgo vs TA-Lib (intentional TradingView-
  convention differences inflate this for ema/adx/atr/macd/stoch/cci etc.; see
  TALIB_COMPATIBILITY.md).
- Summary: **7 at/faster than TA-Lib**, 49 slower (sorted slowest-first).

| Indicator | New (ms) | TA-Lib (ms) | Speed (TA-Lib/New) | max&#124;d&#124; |
|-----------|---------:|------------:|-------------------:|---------:|
| t3(21) | 23.01 | 2.48 | 0.11x | 3.38e-02 |
| wclprice | 7.45 | 1.11 | 0.15x | 0.00e+00 |
| avgprice | 11.10 | 1.77 | 0.16x | 7.28e-12 |
| adosc | 11.59 | 2.38 | 0.21x | 0.00e+00 |
| midprice(14) | 20.30 | 4.42 | 0.22x | 0.00e+00 |
| typprice | 6.12 | 1.34 | 0.22x | 0.00e+00 |
| medprice | 4.20 | 0.97 | 0.23x | 0.00e+00 |
| correl(20) | 25.49 | 5.89 | 0.23x | 2.72e+00 |
| adl | 6.47 | 1.64 | 0.25x | 1.46e+03 |
| apo | 14.51 | 3.98 | 0.27x | 8.84e-10 |
| variance(20) | 8.72 | 2.42 | 0.28x | 1.11e+04 |
| ppo | 16.88 | 4.68 | 0.28x | 2.42e+00 |
| rocp(10) | 4.83 | 1.35 | 0.28x | 0.00e+00 |
| bop | 6.60 | 1.88 | 0.29x | 0.00e+00 |
| trange | 4.17 | 1.25 | 0.30x | 0.00e+00 |
| kama(14) | 7.31 | 2.28 | 0.31x | 1.64e+00 |
| tema(20) | 18.58 | 6.69 | 0.36x | 5.01e-02 |
| mom(10) | 3.63 | 1.32 | 0.36x | 0.00e+00 |
| williams_r(14) | 23.83 | 8.88 | 0.37x | 5.00e+01 |
| psar | 11.51 | 4.32 | 0.38x | 2.73e+02 |
| trima(20) | 5.28 | 1.99 | 0.38x | 1.15e-09 |
| trima(100) | 4.82 | 1.93 | 0.40x | 2.78e-09 |
| aroonosc(14) | 19.62 | 8.19 | 0.42x | 1.00e+02 |
| stochf | 25.83 | 11.03 | 0.43x | 2.84e-14 |
| trix(18) | 16.13 | 6.91 | 0.43x | 3.12e+01 |
| adx(14) | 17.39 | 7.68 | 0.44x | 9.81e+01 |
| aroon(14) | 19.41 | 8.65 | 0.45x | 1.00e+02 |
| rocr(10) | 3.41 | 1.57 | 0.46x | 0.00e+00 |
| cmo(14) | 10.84 | 5.02 | 0.46x | 1.47e+02 |
| stochastic | 26.14 | 12.15 | 0.46x | 1.00e+02 |
| mfi(14) | 14.48 | 6.79 | 0.47x | 3.75e+02 |
| sma(20) | 3.00 | 1.51 | 0.50x | 6.44e-10 |
| bbands(20) | 12.30 | 6.37 | 0.52x | 4.88e-04 |
| wma(20) | 3.02 | 1.63 | 0.54x | 5.14e-09 |
| natr(14) | 10.22 | 5.60 | 0.55x | 3.16e-03 |
| beta(60) | 8.84 | 4.97 | 0.56x | 2.31e+00 |
| atr(14) | 9.22 | 5.41 | 0.59x | 2.73e-01 |
| adxr(14) | 13.92 | 8.59 | 0.62x | 4.26e-14 |
| ultosc | 11.50 | 7.44 | 0.65x | 1.42e-14 |
| midpoint(14) | 20.26 | 13.58 | 0.67x | 0.00e+00 |
| dema(20) | 7.40 | 5.01 | 0.68x | 8.58e-01 |
| plus_dm(14) | 6.45 | 4.41 | 0.68x | 0.00e+00 |
| stochrsi | 22.89 | 16.05 | 0.70x | 1.00e+02 |
| minus_dm(14) | 6.05 | 4.41 | 0.73x | 0.00e+00 |
| dx(14) | 9.64 | 7.34 | 0.76x | 5.68e-14 |
| obv | 4.50 | 3.50 | 0.78x | 4.08e+04 |
| roc(12) | 1.98 | 1.59 | 0.80x | 1.15e-14 |
| ema(20) | 2.28 | 1.99 | 0.87x | 1.67e+00 |
| rsi(14) | 5.55 | 4.85 | 0.87x | 2.84e-14 |
| macd | 9.39 | 10.16 | 1.08x | 1.18e+00 |
| stddev(20) | 3.43 | 3.82 | 1.12x | 2.44e-04 |
| cci(20) | 15.39 | 18.25 | 1.19x | 1.33e+02 |
| linreg(14) | 7.22 | 9.25 | 1.28x | 6.61e-05 |
| linregintercept(14) | 7.28 | 9.38 | 1.29x | 6.61e-05 |
| linregangle(14) | 11.68 | 16.48 | 1.41x | 5.82e-04 |
| tsf(14) | 6.46 | 9.89 | 1.53x | 7.62e-05 |

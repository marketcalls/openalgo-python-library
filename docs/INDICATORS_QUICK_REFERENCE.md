# OpenAlgo Technical Indicators - Quick Reference

## Installation & Import

```python
pip install openalgo
```

```python
from openalgo import ta
import numpy as np
```

## Indicators by Category

### Trend Indicators (19) - 100% Working ✅

**Legend:** Required | Optional (with default)

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| Simple Moving Average | `ta.sma(data, period)` | data, period | Array | `sma_20 = ta.sma(close, 20)` |
| Exponential Moving Average | `ta.ema(data, period)` | data, period | Array | `ema_50 = ta.ema(close, 50)` |
| Weighted Moving Average | `ta.wma(data, period)` | data, period | Array | `wma_10 = ta.wma(close, 10)` |
| Double EMA | `ta.dema(data, period)` | data, period | Array | `dema_20 = ta.dema(close, 20)` |
| Triple EMA | `ta.tema(data, period)` | data, period | Array | `tema_20 = ta.tema(close, 20)` |
| Hull Moving Average | `ta.hma(data, period)` | data, period | Array | `hma_20 = ta.hma(close, 20)` |
| Volume Weighted MA | `ta.vwma(data, volume, period)` | data, volume, period | Array | `vwma_20 = ta.vwma(close, volume, 20)` |
| Arnaud Legoux MA | `ta.alma(data, period=21, offset=0.85, sigma=6.0)` | data, period=21, offset=0.85, sigma=6.0 | Array | `alma = ta.alma(close)` |
| Kaufman Adaptive MA | `ta.kama(data, period=10, fast=2, slow=30)` | data, period=10, fast=2, slow=30 | Array | `kama = ta.kama(close)` |
| Zero Lag EMA | `ta.zlema(data, period)` | data, period | Array | `zlema_20 = ta.zlema(close, 20)` |
| T3 Moving Average | `ta.t3(data, period=21, v_factor=0.7)` | data, period=21, v_factor=0.7 | Array | `t3 = ta.t3(close)` |
| Fractal Adaptive MA | `ta.frama(high, low, period=26)` | high, low, period=26 | Array | `frama = ta.frama(h, l, 26)` |
| Supertrend | `ta.supertrend(high, low, close, period=10, mult=3.0)` | high, low, close, period=10, mult=3.0 | (values, direction) | `st, dir = ta.supertrend(h, l, c)` |
| Ichimoku Cloud | `ta.ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52, disp=26)` | high, low, close, tenkan=9, kijun=26, senkou=52, disp=26 | (tenkan, kijun, span_a, span_b, chikou) | `t, k, sa, sb, c = ta.ichimoku(h, l, c)` |
| Chande Kroll Stop | `ta.ckstop(high, low, close, p=10, x=1.0, q=9)` | high, low, close, p=10, x=1.0, q=9 | (long_stop, short_stop) | `ls, ss = ta.ckstop(h, l, c)` |
| Triangular MA | `ta.trima(data, period)` | data, period | Array | `trima_20 = ta.trima(close, 20)` |
| McGinley Dynamic | `ta.mcginley(data, period)` | data, period | Array | `md_14 = ta.mcginley(close, 14)` |
| VIDYA | `ta.vidya(data, period=14, alpha=0.2)` | data, period=14, alpha=0.2 | Array | `vidya = ta.vidya(close)` |
| Alligator | `ta.alligator(data, jaw=13, teeth=8, lips=5)` | data, jaw=13, teeth=8, lips=5 | (jaw, teeth, lips) | `j, t, l = ta.alligator(hl2)` |
| MA Envelopes | `ta.ma_envelopes(data, period=20, pct=2.5, ma_type="SMA")` | data, period=20, pct=2.5, ma_type | (upper, middle, lower) | `u, m, l = ta.ma_envelopes(close)` |

### Momentum Indicators (9) - 100% Working ✅

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| RSI | `ta.rsi(data, period=14)` | data, period=14 | Array (0-100) | `rsi = ta.rsi(close)` |
| MACD | `ta.macd(data, fast=12, slow=26, signal=9)` | data, fast=12, slow=26, signal=9 | (macd, signal, histogram) | `m, s, h = ta.macd(close)` |
| Stochastic | `ta.stochastic(high, low, close, k_period=14, smooth_k=3, d_period=3)` | high, low, close, k_period=14, smooth_k=3, d_period=3 | (k%, d%) | `k, d = ta.stochastic(h, l, c)` |
| CCI | `ta.cci(high, low, close, period=20)` | high, low, close, period=20 | Array | `cci = ta.cci(high, low, close)` |
| Williams %R | `ta.williams_r(high, low, close, period=14)` | high, low, close, period=14 | Array (-100 to 0) | `wr = ta.williams_r(h, l, c)` |
| Balance of Power | `ta.bop(open, high, low, close)` | open, high, low, close | Array (-1 to 1) | `bop = ta.bop(o, h, l, c)` |
| Elder Ray Index | `ta.elderray(high, low, close, period=13)` | high, low, close, period=13 | (bull_power, bear_power) | `bull, bear = ta.elderray(h, l, c)` |
| Fisher Transform | `ta.fisher(high, low, length=9)` | high, low, length=9 | (fisher, trigger) | `fish, trig = ta.fisher(h, l, 9)` |
| Connors RSI | `ta.crsi(data, lenrsi=3, lenupdown=2, lenroc=100)` | data, lenrsi=3, lenupdown=2, lenroc=100 | Array (0-100) | `crsi = ta.crsi(close)` |

### Volatility Indicators (15) - 100% Working ✅

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| ATR | `ta.atr(high, low, close, period)` | high, low, close, period=14 | Array | `atr_14 = ta.atr(h, l, c, 14)` |
| Bollinger Bands | `ta.bbands(data, period, std)` | data, period=20, std_dev=2 | (upper, middle, lower) | `u, m, l = ta.bbands(close, 20, 2)` |
| Keltner Channel | `ta.keltner(h, l, c, ema, atr, mult)` | high, low, close, ema=20, atr=10, mult=2 | (upper, middle, lower) | `u, m, l = ta.keltner(h, l, c)` |
| Donchian Channel | `ta.donchian(high, low, period)` | high, low, period=20 | (upper, middle, lower) | `u, m, l = ta.donchian(h, l, 20)` |
| Chaikin Volatility | `ta.chaikin(h, l, ema, roc)` | high, low, ema=10, roc=10 | Array | `cv = ta.chaikin(h, l, 10, 10)` |
| NATR | `ta.natr(high, low, close, period)` | high, low, close, period=14 | Array (%) | `natr = ta.natr(h, l, c, 14)` |
| RVI | `ta.rvol(data, std, rsi)` | data, stdev_period=10, rsi_period=14 | Array | `rvi = ta.rvol(close, 10, 14)` |
| Ultimate Oscillator | `ta.ultimate_oscillator(h, l, c, p1, p2, p3)` | high, low, close, p1=7, p2=14, p3=28 | Array | `uo = ta.ultimate_oscillator(h, l, c)` |
| Standard Deviation | `ta.stddev(data, period)` | data, period=20 | Array | `std = ta.stddev(close, 20)` |
| True Range | `ta.true_range(high, low, close)` | high, low, close | Array | `tr = ta.true_range(h, l, c)` |
| Mass Index | `ta.massindex(high, low, length)` | high, low, length=10 | Array | `mi = ta.massindex(h, l, 10)` |
| Bollinger %B | `ta.bbpercent(data, period=20, std_dev=2.0)` | data, period=20, std_dev=2.0 | Array | `bb_pct_b = ta.bbpercent(close)` |
| Bollinger Bandwidth | `ta.bbwidth(data, period=20, std_dev=2.0)` | data, period=20, std_dev=2.0 | Array | `bb_bw = ta.bbwidth(close)` |
| Chandelier Exit | `ta.chandelier_exit(h, l, c, period=22, mult=3.0)` | high, low, close, period=22, mult=3.0 | (long_exit, short_exit) | `le, se = ta.chandelier_exit(h, l, c)` |
| Historical Volatility | `ta.hv(close, length=10, annual=365, per=1)` | close, length=10, annual=365, per=1 | Array (%) | `hv = ta.hv(close)` |
| Ulcer Index | `ta.ulcerindex(data, length=14, smooth_length=14, signal_length=52, signal_type="SMA", return_signal=False)` | data, length=14, smooth_length=14, signal_length=52, signal_type="SMA", return_signal=False | Array or Tuple | `ui = ta.ulcerindex(close)` |
| STARC Bands | `ta.starc(h, l, c, ma=5, atr=15, mult=1.33)` | high, low, close, ma=5, atr=15, mult=1.33 | (upper, middle, lower) | `u, m, l = ta.starc(h, l, c)` |

### Volume Indicators (15) - 100% Working ✅

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| OBV | `ta.obv(close, volume)` | close, volume | Array | `obv = ta.obv(close, volume)` |
| OBV Smoothed | `ta.obv_smoothed(close, volume, ma_type, ma_length, bb_length, bb_mult)` | close, volume, ma_type="None", ma_length=20, bb_length=20, bb_mult=2.0 | Array or Tuple | `obv_smooth = ta.obv_smoothed(close, volume, "EMA", 20)` |
| VWAP | `ta.vwap(h, l, c, v, anchor, source, stdev_mult_1, stdev_mult_2, stdev_mult_3, percent_mult_1, percent_mult_2, percent_mult_3)` | high, low, close, volume, anchor="Session", source="hlc3", stdev_mult_1=1.0, stdev_mult_2=2.0, stdev_mult_3=3.0, percent_mult_1=0.236, percent_mult_2=0.382, percent_mult_3=0.618 | Array | `vwap = ta.vwap(h, l, c, v, "Session", "hlc3")` |
| MFI | `ta.mfi(h, l, c, v, period)` | high, low, close, volume, period=14 | Array (0-100) | `mfi = ta.mfi(h, l, c, v, 14)` |
| A/D Line | `ta.adl(high, low, close, volume)` | high, low, close, volume | Array | `adl = ta.adl(h, l, c, v)` |
| Chaikin Money Flow | `ta.cmf(h, l, c, v, period)` | high, low, close, volume, period=20 | Array | `cmf = ta.cmf(h, l, c, v, 20)` |
| Ease of Movement | `ta.emv(high, low, volume, length, divisor)` | high, low, volume, length=14, divisor=10000 | Array | `emv = ta.emv(h, l, v, 14, 10000)` |
| Force Index | `ta.force_index(close, volume, length)` | close, volume, length=13 | Array | `fi = ta.force_index(close, volume, 13)` |
| NVI | `ta.nvi(close, volume)` | close, volume | Array | `nvi = ta.nvi(close, volume)` |
| NVI with EMA | `ta.nvi_with_ema(close, volume, ema_length)` | close, volume, ema_length=255 | (nvi, ema) | `nvi, ema = ta.nvi_with_ema(c, v)` |
| PVI | `ta.pvi(close, volume, initial_value)` | close, volume, initial_value=100.0 | Array | `pvi = ta.pvi(close, volume)` |
| PVI with Signal | `ta.pvi_with_signal(c, v, init, sig_type, sig_len)` | close, volume, initial_value=100.0, signal_type="EMA", signal_length=255 | (pvi, signal) | `pvi, sig = ta.pvi_with_signal(c, v)` |
| Volume Oscillator (TV v6) | `ta.volosc(v, short_length, long_length, check_volume_validity)` | volume, short_length=5, long_length=10, check_volume_validity=True | Array (%) | `vo = ta.volosc(v, 5, 10)` |
| Volume ROC | `ta.vroc(volume, period)` | volume, period=25 | Array (%) | `vroc = ta.vroc(volume, 25)` |
| Klinger Volume Osc | `ta.kvo(h, l, c, v, fast=34, slow=55)` | high, low, close, volume, fast=34, slow=55 | Array | `kvo = ta.kvo(h, l, c, v)` |
| Price Volume Trend | `ta.pvt(close, volume)` | close, volume | Array | `pvt = ta.pvt(close, volume)` # TradingView formula |

### Oscillators (18) - 100% Working ✅

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| ROC | `ta.roc(data, length)` | data, length | Array (%) | `roc = ta.roc(close, 12)` |
| CMO | `ta.cmo(data, period)` | data, period=14 | Array (-100 to 100) | `cmo = ta.cmo(close, 14)` |
| TRIX | `ta.trix(data, length=18)` | data, length=18 | Array | `trix = ta.trix(close, 18)` |
| Ultimate Oscillator | `ta.uo_oscillator(h, l, c, p1, p2, p3)` | high, low, close, p1=7, p2=14, p3=28 | Array | `uo = ta.uo_oscillator(h, l, c)` |
| Awesome Oscillator | `ta.awesome_oscillator(h, l, fast, slow)` | high, low, fast=5, slow=34 | Array | `ao = ta.awesome_oscillator(h, l)` |
| Accelerator Osc | `ta.accelerator_oscillator(h, l, period)` | high, low, period=5 | Array | `ac = ta.accelerator_oscillator(h, l)` |
| PPO | `ta.ppo(data, fast, slow, signal)` | data, fast=12, slow=26, signal=9 | (ppo, signal, hist) | `p, s, h = ta.ppo(close)` |
| Price Oscillator | `ta.po(data, fast, slow, type)` | data, fast=10, slow=20, ma_type="SMA" | Array | `po = ta.po(close)` |
| DPO | `ta.dpo(data, period, is_centered)` | data, period=21, is_centered=False | Array | `dpo = ta.dpo(close, 21, False)` |
| Aroon Oscillator | `ta.aroon_oscillator(high, low, period)` | high, low, period=25 | Array (-100 to 100) | `ao = ta.aroon_oscillator(h, l, 25)` |
| Stochastic RSI | `ta.stochrsi(data, rsi=14, stoch=14, k=3, d=3)` | data, rsi=14, stoch=14, k=3, d=3 | (k%, d%) | `k, d = ta.stochrsi(close)` |
| Relative Vigor Index | `ta.rvi(o, h, l, c, period=10)` | open, high, low, close, period=10 | (rvi, signal) | `rvi, sig = ta.rvi(o, h, l, c)` |
| Chaikin Oscillator | `ta.cho(h, l, c, v, fast=3, slow=10)` | high, low, close, volume, fast=3, slow=10 | Array | `co = ta.cho(h, l, c, v)` |
| Choppiness Index | `ta.chop(high, low, close, period=14)` | high, low, close, period=14 | Array (0-100) | `chop = ta.chop(h, l, c)` |
| Know Sure Thing | `ta.kst(data, roclen1=10, roclen2=15, roclen3=20, roclen4=30, smalen1=10, smalen2=10, smalen3=10, smalen4=15, siglen=9)` | data, ROC/SMA periods, siglen=9 | (kst, signal) | `kst, sig = ta.kst(close)` |
| Schaff Trend Cycle | `ta.stc(data, fast_length=23, slow_length=50, cycle_length=10, d1_length=3, d2_length=3)` | data, fast_length=23, slow_length=50, cycle_length=10, d1_length=3, d2_length=3 | Array (0-100) | `stc = ta.stc(close)` |
| True Strength Index | `ta.tsi(data, long=25, short=13, signal=13)` | data, long=25, short=13, signal=13 | (tsi, signal) | `tsi, sig = ta.tsi(close)` |
| Vortex Indicator (TV v6) | `ta.vi(high, low, close, period=14)` | high, low, close, period=14 | (vi_plus, vi_minus) | `vip, vim = ta.vi(h, l, c)` |
| Gator Oscillator | `ta.gator_oscillator(high, low, jaw=13, teeth=8, lips=5)` | high, low, jaw=13, teeth=8, lips=5 | (upper, lower) | `u, l = ta.gator_oscillator(h, l)` |
| **Coppock** | `ta.coppock(data, wma=10, long_roc=14, short_roc=11)` | data, wma=10, long_roc=14, short_roc=11 | Array | `coppock = ta.coppock(close)` |

### Statistical Indicators (8) - 100% Working ✅

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| Linear Regression | `ta.linreg(data, period)` | data, period=14 | Array | `lr = ta.linreg(close, 14)` |
| LR Slope | `ta.lrslope(data, period, interval)` | data, period=100, interval=1 | Array | `slope = ta.lrslope(close, 100, 1)` |
| Correlation | `ta.correlation(data1, data2, period)` | data1, data2, period=20 | Array (-1 to 1) | `corr = ta.correlation(stock1, stock2, 20)` |
| Beta | `ta.beta(asset, market, period)` | asset, market, period=252 | Array | `beta = ta.beta(stock, market, 252)` |
| Variance (TV v4) | `ta.variance(data, lookback, mode, ema_period, filter_lookback, ema_length, return_components)` | data, lookback=20, mode="PR", ema_period=20, filter_lookback=20, ema_length=14, return_components=False | Array/Tuple | `var = ta.variance(close, lookback=20, mode="LR")` |
| TSF | `ta.tsf(data, period)` | data, period=14 | Array | `tsf = ta.tsf(close, 14)` |
| Median | `ta.median(data, period)` | data, period=3 | Array | `med = ta.median(close, 3)` |
| Median Bands | `ta.median_bands(h, l, c, src, m_len, a_len, a_mult)` | high, low, close, source=None, median_length=3, atr_length=14, atr_mult=2.0 | (median, upper, lower, ema) | `m, u, l, e = ta.median_bands(h, l, c)` |
| Mode | `ta.mode(data, period, bins)` | data, period=20, bins=10 | Array | `mode = ta.mode(close, 20, 10)` |

### Hybrid Indicators (7) - 100% Working ✅

| Indicator | Function | Parameters | Returns | Example |
|-----------|----------|------------|---------|---------|
| ADX System | `ta.adx(h, l, c, period)` | high, low, close, period=14 | (+DI, -DI, ADX) | `p, m, adx = ta.adx(h, l, c, 14)` |
| Aroon System | `ta.aroon(high, low, period)` | high, low, period=25 | (up, down) | `up, down = ta.aroon(h, l, 25)` |
| Pivot Points | `ta.pivot_points(high, low, close)` | high, low, close | (P, R1, S1, R2, S2, R3, S3) | `p, r1, s1, r2, s2, r3, s3 = ta.pivot_points(h, l, c)` |
| DMI | `ta.dmi(h, l, c, period)` | high, low, close, period=14 | (+DI, -DI) | `plus, minus = ta.dmi(h, l, c)` |
| Parabolic SAR | `ta.psar(high, low, accel, max)` | high, low, accel=0.02, max=0.2 | Array | `psar = ta.psar(h, l, 0.02, 0.2)` |
| Zig Zag | `ta.zigzag(high, low, close, deviation=5.0)` | high, low, close, deviation=5.0 | Array | `zz = ta.zigzag(h, l, c)` |
| Williams Fractals | `ta.fractals(high, low, periods=2)` | high, low, periods=2 | (fractal_up, fractal_down) | `fu, fd = ta.fractals(h, l, 2)` |
| Random Walk Index | `ta.rwi(high, low, close, period=14)` | high, low, close, period=14 | (rwi_high, rwi_low) | `rwh, rwl = ta.rwi(h, l, c)` |

### Utility Functions (11) - 100% Working ✅

| Function | Syntax | Parameters | Returns | Example |
|----------|--------|------------|---------|---------|
| Crossover | `ta.crossover(series1, series2)` | series1, series2 | Boolean array | `buy = ta.crossover(fast_ma, slow_ma)` |
| Crossunder | `ta.crossunder(series1, series2)` | series1, series2 | Boolean array | `sell = ta.crossunder(fast_ma, slow_ma)` |
| Highest | `ta.highest(data, period)` | data, period | Array | `hh = ta.highest(high, 20)` |
| Lowest | `ta.lowest(data, period)` | data, period | Array | `ll = ta.lowest(low, 20)` |
| Change | `ta.change(data, length)` | data, length=1 | Array | `chg = ta.change(close, 1)` |
| ROC | `ta.roc(data, length)` | data, length | Array (%) | `roc = ta.roc(close, 10)` |
| StdDev | `ta.stdev(data, period)` | data, period | Array | `std = ta.stdev(close, 20)` |
| **EXREM** 🆕 | `ta.exrem(primary, secondary)` | primary, secondary | Boolean array | ✅ | `clean_buy = ta.exrem(buy_signals, sell_signals)` |
| **FLIP** 🆕 | `ta.flip(primary, secondary)` | primary, secondary | Boolean array | ✅ | `trend_state = ta.flip(uptrend_start, downtrend_start)` |
| **VALUEWHEN** 🆕 | `ta.valuewhen(expr, array, n=1)` | expr, array, n=1 | Array | ✅ | `price_at_signal = ta.valuewhen(buy_signal, close, 1)` |
| **RISING** 🆕 | `ta.rising(data, length)` | data, length | Boolean array | ✅ | `price_rising = ta.rising(close, 5)` |
| **FALLING** 🆕 | `ta.falling(data, length)` | data, length | Boolean array | ✅ | `price_falling = ta.falling(close, 5)` |
| **CROSS** 🆕 | `ta.cross(series1, series2)` | series1, series2 | Boolean array | ✅ | `any_cross = ta.cross(fast_ma, slow_ma)` |

## Common Trading Patterns

### Moving Average Crossover
```python
# Golden Cross / Death Cross
ema_fast = ta.ema(close, 50)
ema_slow = ta.ema(close, 200)
golden_cross = ta.crossover(ema_fast, ema_slow)
death_cross = ta.crossunder(ema_fast, ema_slow)
```

### RSI Divergence
```python
# Bullish divergence: Price makes lower low, RSI makes higher low
rsi = ta.rsi(close, 14)
price_ll = close == ta.lowest(close, 20)
rsi_hl = rsi > ta.lowest(rsi, 20)
bullish_divergence = price_ll & rsi_hl
```

### Signal Cleanup with EXREM
```python
# Remove excessive buy signals until sell signal occurs
price_above_ma = close > ta.sma(close, 20)
price_below_ma = close < ta.sma(close, 20)
clean_buy_signals = ta.exrem(price_above_ma, price_below_ma)

# RSI oversold/overbought cleanup
rsi = ta.rsi(close, 14)
oversold = rsi < 30
overbought = rsi > 70
clean_oversold = ta.exrem(oversold, overbought)  # Remove excess oversold signals
```

### Trend State with FLIP
```python
# Create persistent trend state indicator
fast_ma = ta.ema(close, 10)
slow_ma = ta.ema(close, 20)
uptrend_start = ta.crossover(fast_ma, slow_ma)
downtrend_start = ta.crossunder(fast_ma, slow_ma)
in_uptrend = ta.flip(uptrend_start, downtrend_start)

# Breakout state tracking
breakout_up = close > ta.highest(close, 20)
breakout_down = close < ta.lowest(close, 20)
breakout_active = ta.flip(breakout_up, breakout_down)
```

### Historical Reference with VALUEWHEN
```python
# Get price when specific conditions occurred
rsi = ta.rsi(close, 14)
oversold_signal = rsi < 30
price_at_oversold = ta.valuewhen(oversold_signal, close, 1)  # Most recent
prev_oversold_price = ta.valuewhen(oversold_signal, close, 2)  # 2nd most recent

# Breakout reference levels
new_high = close > ta.highest(close, 50)
breakout_price = ta.valuewhen(new_high, close, 1)

# Stop-loss using signal points
buy_signal = ta.crossover(ta.ema(close, 10), ta.ema(close, 20))
stop_loss = ta.valuewhen(buy_signal, ta.lowest(low, 10), 1)
```

### Rising/Falling Trend Detection
```python
# Pine Script-style trend detection
price_rising_5 = ta.rising(close, 5)  # Price rising over 5 periods
volume_rising = ta.rising(volume, 3)  # Volume increasing over 3 periods

# Momentum confirmation
rsi = ta.rsi(close, 14)
rsi_rising = ta.rising(rsi, 2)  # RSI trending up
momentum_up = price_rising_5 & rsi_rising & (rsi > 50)

# Declining trends  
price_falling_5 = ta.falling(close, 5)
volatility_falling = ta.falling(ta.atr(high, low, close, 14), 10)
```

### Cross Detection (Any Direction)
```python
# Detect any cross between series (up or down)
fast_ma = ta.ema(close, 10)
slow_ma = ta.ema(close, 20)
any_ma_cross = ta.cross(fast_ma, slow_ma)

# RSI crosses 50 line (either direction)
rsi = ta.rsi(close, 14)
rsi_cross_midline = ta.cross(rsi, np.full(len(rsi), 50))

# Price crosses key moving average
price_cross_200ma = ta.cross(close, ta.sma(close, 200))
```

### Coppock Curve for Long-term Analysis
```python
# Long-term momentum indicator for indices
coppock = ta.coppock(close)  # Uses TradingView defaults (10, 14, 11)

# Zero line crossovers for major signals
coppock_bullish = (coppock > 0) & (ta.change(coppock, 1) > 0)
coppock_bearish = (coppock < 0) & (ta.change(coppock, 1) < 0)

# Custom parameters for different sensitivities
coppock_fast = ta.coppock(close, 8, 10, 8)   # More sensitive
coppock_slow = ta.coppock(close, 15, 20, 15)  # More reliable
```

### Bollinger Band Squeeze
```python
# Volatility contraction
upper, middle, lower = ta.bbands(close, 20, 2)
bandwidth = (upper - lower) / middle
squeeze = bandwidth == ta.lowest(bandwidth, 20)
```

### MACD Signal
```python
# MACD crossover with trend filter
macd_line, signal_line, histogram = ta.macd(close, 12, 26, 9)
sma_200 = ta.sma(close, 200)

bullish_macd = ta.crossover(macd_line, signal_line) & (close > sma_200)
bearish_macd = ta.crossunder(macd_line, signal_line) & (close < sma_200)
```

### Supertrend Strategy
```python
# Trend following with Supertrend
supertrend, direction = ta.supertrend(high, low, close, 10, 3)
buy_signal = direction == -1  # Bullish
sell_signal = direction == 1   # Bearish
stop_loss = supertrend  # Use as trailing stop
```

### Volume Confirmation
```python
# Price breakout with volume
high_20 = ta.highest(high, 20)
breakout = high >= high_20
volume_surge = volume > ta.sma(volume, 20) * 1.5
confirmed_breakout = breakout & volume_surge
```

## Position Sizing

### ATR-Based Position Sizing
```python
def position_size_atr(capital, risk_percent, atr_value, atr_multiplier=2):
    risk_amount = capital * (risk_percent / 100)
    stop_distance = atr_value * atr_multiplier
    shares = risk_amount / stop_distance
    return int(shares)

# Example
atr = ta.atr(high, low, close, 14)
shares = position_size_atr(100000, 1, atr[-1], 2)
```

### Volatility-Adjusted Sizing
```python
def position_size_volatility(capital, base_risk, current_vol, avg_vol):
    adjusted_risk = base_risk * (avg_vol / current_vol)
    adjusted_risk = max(0.5, min(2.0, adjusted_risk))  # Cap between 0.5% and 2%
    return capital * (adjusted_risk / 100)

# Example
current_atr = ta.atr(high, low, close, 14)[-1]
avg_atr = ta.sma(ta.atr(high, low, close, 14), 50)[-1]
risk_amount = position_size_volatility(100000, 1, current_atr, avg_atr)
```

## Performance Tips

1. **Use NumPy arrays** for best performance
   ```python
   close_np = np.array(price_list)  # Convert once
   sma = ta.sma(close_np, 20)       # Fast calculation
   ```

2. **Batch calculations** to minimize overhead
   ```python
   # Calculate all at once
   indicators = {
       'sma_20': ta.sma(close, 20),
       'ema_20': ta.ema(close, 20),
       'rsi_14': ta.rsi(close, 14),
       'atr_14': ta.atr(high, low, close, 14)
   }
   ```

3. **Reuse calculations** when possible
   ```python
   # Calculate once, use multiple times
   atr_14 = ta.atr(high, low, close, 14)
   position_size = calculate_size(atr_14[-1])
   stop_loss = close[-1] - (2 * atr_14[-1])
   take_profit = close[-1] + (3 * atr_14[-1])
   ```

## Complete Strategy Template

```python
from openalgo import ta
import numpy as np

class TradingStrategy:
    def __init__(self, capital=100000, risk_percent=1):
        self.capital = capital
        self.risk_percent = risk_percent
        
    def analyze(self, high, low, close, volume):
        # Calculate all indicators
        indicators = self._calculate_indicators(high, low, close, volume)
        
        # Generate signal
        signal = self._generate_signal(indicators, close)
        
        # Risk management
        risk_params = self._calculate_risk(indicators, close)
        
        return {
            'signal': signal,
            'indicators': indicators,
            'risk': risk_params
        }
    
    def _calculate_indicators(self, high, low, close, volume):
        return {
            # Trend
            'sma_50': ta.sma(close, 50),
            'sma_200': ta.sma(close, 200),
            'ema_20': ta.ema(close, 20),
            'supertrend': ta.supertrend(high, low, close, 10, 3),
            
            # Momentum
            'rsi': ta.rsi(close, 14),
            'macd': ta.macd(close, 12, 26, 9),
            'stoch': ta.stochastic(high, low, close, k_period=14, smooth_k=3, d_period=3),
            
            # Volatility
            'atr': ta.atr(high, low, close, 14),
            'bbands': ta.bbands(close, 20, 2),
            
            # Volume
            'obv': ta.obv(close, volume),
            'cmf': ta.cmf(high, low, close, volume, 20),
            'mfi': ta.mfi(high, low, close, volume, 14)
        }
    
    def _generate_signal(self, indicators, close):
        # Extract latest values
        i = -1  # Latest index
        
        # Trend conditions
        trend_up = (
            close[i] > indicators['sma_200'][i] and
            indicators['sma_50'][i] > indicators['sma_200'][i]
        )
        
        # Momentum conditions
        rsi_ok = 30 < indicators['rsi'][i] < 70
        macd_bullish = indicators['macd'][0][i] > indicators['macd'][1][i]
        
        # Volume confirmation
        volume_positive = indicators['cmf'][i] > 0
        
        # Generate signal
        if trend_up and rsi_ok and macd_bullish and volume_positive:
            return "BUY"
        elif not trend_up and rsi_ok and not macd_bullish and not volume_positive:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_risk(self, indicators, close):
        atr = indicators['atr'][-1]
        
        # Position sizing
        risk_amount = self.capital * (self.risk_percent / 100)
        stop_distance = atr * 2
        position_size = int(risk_amount / stop_distance)
        
        # Stop loss and take profit
        if indicators['supertrend'][1][-1] == -1:  # Bullish
            stop_loss = close[-1] - stop_distance
            take_profit = close[-1] + (stop_distance * 2)
        else:  # Bearish
            stop_loss = close[-1] + stop_distance
            take_profit = close[-1] - (stop_distance * 2)
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount
        }

# Usage
strategy = TradingStrategy(capital=100000, risk_percent=1)
result = strategy.analyze(high, low, close, volume)
print(f"Signal: {result['signal']}")
print(f"Position: {result['risk']['position_size']} shares")
print(f"Stop: ${result['risk']['stop_loss']:.2f}")
print(f"Target: ${result['risk']['take_profit']:.2f}")
```

## New Indicators Added

### Advanced Momentum & Oscillators

```python
# Stochastic RSI - combines RSI with stochastic calculation
k, d = ta.stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3)

# Relative Vigor Index - measures price vigor relative to trading range
rvi, signal = ta.rvi(open_prices, high, low, close, period=10)

# Chaikin Oscillator - momentum of Accumulation/Distribution Line
chaikin_osc = ta.chaikin_osc(high, low, close, volume, fast_period=3, slow_period=10)

# Balance of Power - measures buyer vs seller strength
bop = ta.bop(open_prices, high, low, close)

# Elder Ray Index - bull and bear power
bull_power, bear_power = ta.elder_ray(high, low, close, period=13)

# Fisher Transform - converts prices to normal distribution
fisher, trigger = ta.fisher_transform((high + low) / 2, period=10)

# Connors RSI - composite momentum oscillator
connors_rsi = ta.connors_rsi(close, rsi_period=3, streak_period=2, roc_period=100)
```

### Advanced Volume Indicators

```python
# Klinger Volume Oscillator - predicts price reversals using volume
kvo = ta.kvo(high, low, close, volume, fast_period=34, slow_period=55)

# Price Volume Trend - cumulative volume based on price changes
pvt = ta.pvt(close, volume)
```

### Advanced Trend Indicators

```python
# Chande Kroll Stop - trailing stop-loss using ATR
long_stop, short_stop = ta.chande_kroll_stop(high, low, close, period=10, mult=1.0)

# Triangular Moving Average - double-smoothed average
trima = ta.trima(close, period=20)

# McGinley Dynamic - automatically adjusts for market speed
mcginley = ta.mcginley(close, period=14)

# VIDYA - uses CMO to adjust EMA smoothing
vidya = ta.vidya(close, period=14, alpha=0.2)

# Alligator - Bill Williams indicator with three SMMA lines
jaw, teeth, lips = ta.alligator((high + low) / 2)

# Moving Average Envelopes - percentage bands around MA
upper, middle, lower = ta.ma_envelopes(close, period=20, percentage=2.5, ma_type="SMA")
```

### Advanced Oscillators

```python
# Choppiness Index - measures trending vs ranging markets
chop = ta.chop(high, low, close, period=14)

# Know Sure Thing - smoothed rate-of-change momentum oscillator
kst, signal = ta.kst(close)

# Schaff Trend Cycle - TradingView Pine Script v4 (combines MACD with stochastics)
stc = ta.stc(close, fast_length=23, slow_length=50, cycle_length=10)

# True Strength Index - double-smoothed momentum oscillator
tsi, signal = ta.tsi(close, long=25, short=13)

# Vortex Indicator - identifies trend changes
vi_plus, vi_minus = ta.vi(high, low, close, period=14)

# Gator Oscillator - shows Alligator convergence/divergence
upper, lower = ta.gator_oscillator((high + low) / 2)
```

### Advanced Volatility Indicators

```python
# Bollinger Bands %B - position relative to bands
bb_percent_b = ta.bbpercent(close, period=20, std_dev=2.0)

# Bollinger Bandwidth - measures band width
bb_bandwidth = ta.bbwidth(close, period=20, std_dev=2.0)

# Chandelier Exit - trailing stop based on highest/lowest
long_exit, short_exit = ta.chandelier_exit(high, low, close, period=22, multiplier=3.0)

# Historical Volatility - realized volatility measure
hv = ta.hv(close, period=20, annualize=True)

# Ulcer Index - TradingView Pine Script v4 (downside risk measure with signal line)
ui = ta.ulcerindex(close, length=14, smooth_length=14)
# With signal line
ui, signal = ta.ulcerindex(close, length=14, smooth_length=14, signal_length=52, signal_type="SMA", return_signal=True)

# STARC Bands - TradingView Pine Script v2 (SMA with ATR-based bands)
upper, middle, lower = ta.starc(high, low, close, ma_period=5, atr_period=15, multiplier=1.33)
```

### Advanced Pattern Recognition

```python
# Zig Zag - connects significant price swings
zigzag = ta.zigzag(high, low, close, deviation=5.0)

# Williams Fractals - identifies reversal points
fractal_up, fractal_down = ta.fractals(high, low)

# Random Walk Index - measures trend strength
rwi_high, rwi_low = ta.rwi(high, low, close, period=14)
```

### Example Usage in Trading Strategy

```python
import numpy as np
from openalgo import ta

# Sample data
high = np.random.randn(100).cumsum() + 100
low = high - np.random.rand(100) * 2
close = low + (high - low) * np.random.rand(100)
open_prices = np.roll(close, 1)
volume = np.random.randint(1000, 10000, 100)

# Advanced momentum analysis
stoch_k, stoch_d = ta.stoch_rsi(close)
rvi_main, rvi_signal = ta.rvi(open_prices, high, low, close)
connors = ta.connors_rsi(close)

# Advanced volume analysis
kvo = ta.kvo(high, low, close, volume)
pvt = ta.pvt(close, volume)

# Risk management with Chande Kroll Stop
long_stop, short_stop = ta.chande_kroll_stop(high, low, close)

# Trading signals
buy_signal = (stoch_k[-1] > stoch_d[-1]) & (rvi_main[-1] > rvi_signal[-1]) & (kvo[-1] > 0)
sell_signal = (stoch_k[-1] < stoch_d[-1]) & (rvi_main[-1] < rvi_signal[-1]) & (kvo[-1] < 0)

print(f"Latest signals:")
print(f"Stochastic RSI: K={stoch_k[-1]:.2f}, D={stoch_d[-1]:.2f}")
print(f"RVI: {rvi_main[-1]:.4f}, Signal: {rvi_signal[-1]:.4f}")
print(f"Connors RSI: {connors[-1]:.2f}")
print(f"KVO: {kvo[-1]:.2f}")
print(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")
print(f"Stop Levels: Long={long_stop[-1]:.2f}, Short={short_stop[-1]:.2f}")
```
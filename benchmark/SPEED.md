# OpenAlgo Indicators - Rust Core Speed Benchmark

Rust = current `ta.*` backend (openalgo._oaindicators). Py-ref = the original kernels run interpreted (numba removed). Speedup = Py-ref / Rust.

### RELIANCE daily (real) (7629 bars)

| indicator | Rust (ms) | Py-ref (ms) | speedup |
|-----------|----------:|------------:|--------:|
| sma(20) | 0.010 | 1.581 | 158.1x |
| ema(20) | 0.010 | 1.718 | 179.0x |
| wma(20) | 0.093 | 22.468 | 240.6x |
| rsi(14) | 0.031 | 4.092 | 130.7x |
| macd | 0.032 | 5.091 | 157.6x |
| bbands(20) | 0.037 | 0.038 | 1.0x |
| atr(14) | 0.033 | 5.245 | 160.4x |
| stochastic | 0.108 | 15.091 | 140.1x |
| supertrend | 0.061 | 12.584 | 206.6x |
| adx(14) | 0.199 | 41.259 | 207.4x |
| sar | 0.040 | 5.154 | 128.2x |
| linreg(14) | 0.068 | 59.965 | 887.1x |

### Synthetic 100k (100000 bars)

| indicator | Rust (ms) | Py-ref (ms) | speedup |
|-----------|----------:|------------:|--------:|
| sma(20) | 0.123 | 20.925 | 170.0x |
| ema(20) | 0.121 | 22.447 | 185.1x |
| wma(20) | 1.194 | 299.451 | 250.8x |
| rsi(14) | 0.406 | 53.783 | 132.4x |
| macd | 0.782 | 68.969 | 88.2x |
| bbands(20) | 0.939 | 0.980 | 1.0x |
| atr(14) | 0.643 | 70.942 | 110.2x |
| stochastic | 2.421 | 199.702 | 82.5x |
| supertrend | 1.744 | 168.702 | 96.7x |
| adx(14) | 3.733 | 544.574 | 145.9x |
| sar | 0.991 | 66.380 | 67.0x |
| linreg(14) | 0.874 | 787.907 | 901.7x |

## Live session loop: a tick from its arrival at the loop to the risk rules

`python benchmark/live_loop_speed.py --check` (`--slack` for a slower machine). A live
session on the in-process fake OpenAlgo with its own loop and the fake's direct feed: two
strategies with leg rules (a stop, a trailing stop and a target) hold SBIN and INFY while
both trade every 100 ms of the fake's clock from 14:40:05 to 14:58 IST, rising a paisa a
second, so the trail ratchets and its modifies go through the order worker. Each tick is
timed on the wall clock from the start of the loop's step A.1 for it to the return of the
kernel's `push_tick`. Target: the 99th percentile under 500 microseconds (15-phase2-plan
section 4 and 16-session-design 2.9: "well under a millisecond").

Measured 2026-11-18 on the development desktop (Windows 11, Python 3.12), three runs:

| ticks measured | trailing modifies | median (us) | 99th percentile (us) | worst (us) |
|---:|---:|---:|---:|---:|
| 21,298 | 12 | 6.7 | 18.3 | 76.0 |
| 21,298 | 12 | 6.8 | 18.5 | 53.7 |
| 21,298 | 12 | 6.8 | 19.6 | 78.5 |

The worst tick is shown, not checked: a pause of the interpreter's garbage collector or
of the machine lands there.

# -*- coding: utf-8 -*-
"""
Indicator compute backend.

Single seam between the Python indicator wrappers and the compiled Rust core
(``openalgo._oaindicators``). When the extension is present every kernel runs in
Rust; when it is absent (a source checkout without a built wheel) a pure-NumPy
fallback returns the same values. Neither path depends on numba / llvmlite.

That equivalence includes NaN placement, and the two backends once diverged there
(issue #2029), so the contract is stated explicitly: rolling-window kernels are
*window-local* - ``out[i]`` is NaN iff ``i < period-1`` or the window contains a
NaN, recovering as soon as it slides out, the same rule as pandas
``.rolling(period)``. Recursive kernels (the EMA family) instead skip leading NaNs,
seed at the first finite value, and carry state across an interior NaN. Both paths
are checked against each other in ``benchmark/ci_smoke.py``.

Wrappers should call these functions instead of the legacy numba kernels.
"""
import numpy as np

try:
    from openalgo import _oaindicators as _rs
    HAVE_RUST = True
except Exception:  # noqa: BLE001 - extension optional; fall back to numpy
    _rs = None
    HAVE_RUST = False


def _f(a):
    """Contiguous float64 view (Rust as_slice requires C-contiguous float64)."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _first_clean_window(data, period):
    """Start index of the first NaN-free window of `period` values, else -1.

    Recursive kernels seed here instead of blindly at index 0, so leading warm-up
    NaNs from an upstream indicator only delay the start of the output.
    """
    if period <= 0 or data.size < period:
        return -1
    run = 0
    for i in range(data.size):
        run = 0 if np.isnan(data[i]) else run + 1
        if run == period:
            return i + 1 - period
    return -1


def _roll_nan(data, period, squares=False):
    """Rolling window sums with NaN treated as absent, plus a per-window NaN count.

    Returns ``(sums, nan_counts)`` (or ``(sums, sum_of_squares, nan_counts)`` when
    ``squares``), each aligned so element 0 is the window ending at ``period-1``.

    Callers publish a value only where the count is 0. That is the window-local
    NaN contract the Rust kernels implement (see rust/oa_core "Rolling
    reductions"): a NaN blanks only the windows containing it and the series
    recovers once it slides out, the same rule as pandas ``.rolling(period)``.
    A plain ``np.cumsum`` cannot do this - one NaN poisons every later element.
    """
    m = np.isnan(data)
    clean = np.where(m, 0.0, data)
    c = np.concatenate(([0.0], np.cumsum(clean)))
    cm = np.concatenate(([0], np.cumsum(m.astype(np.int64))))
    sums = c[period:] - c[:-period]
    counts = cm[period:] - cm[:-period]
    if not squares:
        return sums, counts
    cq = np.concatenate(([0.0], np.cumsum(clean * clean)))
    return sums, cq[period:] - cq[:-period], counts


def _require_same_length(**series):
    """Raise ValueError unless every named series has the same number of bars."""
    sizes = {name: values.size for name, values in series.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"All series must have the same length, got {sizes}")


def _unknown_in_window(unknown, period):
    """Number of True flags in each `period`-bar window; element k is bars [k, k+period)."""
    before = np.concatenate(([0], np.cumsum(unknown)))
    return before[period:] - before[:-period]


def sma(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.sma(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    s, nans = _roll_nan(data, period)
    out[period - 1:] = np.where(nans == 0, s / period, np.nan)
    return out


def wma(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.wma(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    weights = np.arange(1, period + 1, dtype=np.float64)
    wsum = weights.sum()
    valid = np.convolve(data, weights[::-1], mode="valid") / wsum
    out[period - 1:] = valid
    return out


def hma(data, period):
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n)).

    The intermediate 2*WMA(n/2)-WMA(n) is finite only from index period-1; the
    final WMA is run over that valid region only so leading NaNs never poison the
    rolling backend. First valid output at (period-1)+(sqrt(period)-1).
    """
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.hma(data, period)
    n = data.size
    out = np.full(n, np.nan)
    half = period // 2
    sqrt_p = int(np.sqrt(period)) if period > 0 else 0
    if period <= 0 or n < period or half == 0 or sqrt_p == 0:
        return out
    # `wma` is window-local in NaN, so the outer pass can run over the whole
    # series: the first valid output still lands at (period-1)+(sqrt_p-1), and a
    # NaN in `data` only pushes that out locally instead of blanking the tail.
    return wma(2.0 * wma(data, half) - wma(data, period), sqrt_p)


def ema(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.ema(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = np.nan
    for i in range(n):
        x = data[i]
        if np.isnan(x):
            continue
        prev = x if np.isnan(prev) else alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def stdev(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.stdev(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    s, sq, nans = _roll_nan(data, period, squares=True)
    mean = s / period
    var = sq / period - mean * mean
    out[period - 1:] = np.where(nans == 0, np.sqrt(np.maximum(0.0, var)), np.nan)
    return out


def true_range(high, low, close):
    high, low, close = _f(high), _f(low), _f(close)
    if HAVE_RUST:
        return _rs.true_range(high, low, close)
    n = high.size
    tr = np.empty(n)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    # An unknown bar must stay unknown: `np.maximum` propagates NaN, but be
    # explicit so this keeps matching the Rust kernel's guard.
    tr[1:] = np.where(
        np.isnan(hl) | np.isnan(hc) | np.isnan(lc),
        np.nan,
        np.maximum(np.maximum(hl, hc), lc),
    )
    return tr


def atr_wilder(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.atr_wilder(high, low, close, period)
    n = high.size
    if period <= 0 or n < period:
        return np.full(n, np.nan)
    # Wilder smoothing of the true range: `ema_wilder` is the same recursion with
    # the leading-NaN handling already in it, so ATR inherits that rather than
    # re-seeding blindly at index 0 and poisoning `atr[i - 1]` for the rest of the
    # series the moment a NaN reaches the true range.
    return ema_wilder(true_range(high, low, close), period)


def rsi(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.rsi(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period + 1:
        return out
    # Seed on the first NaN-free run of period+1 values, matching the Rust kernel.
    # A NaN delta is neither > 0 nor < 0, so seeding blindly at index 0 folds an
    # upstream indicator's warm-up into avg_gain == avg_loss == 0 and publishes the
    # avg_loss == 0 branch: a confident, wrong RSI of 100 over the whole prefix.
    fv = _first_clean_window(data, period + 1)
    if fv < 0:
        return out
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[fv:fv + period].mean()
    avg_loss = losses[fv:fv + period].mean()
    start = fv + period
    out[start] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(start, n - 1):
        if np.isnan(deltas[i]):
            # Bar is unknown, not flat: leave NaN and keep the averages intact.
            continue
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def macd(data, fast_period, slow_period, signal_period):
    macd_line = ema(data, fast_period) - ema(data, slow_period)
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _ema_from(series, period, start):
    """EMA of `series` from bar `start` on; the bars before it are NaN."""
    out = np.full(series.size, np.nan)
    if start < series.size:
        out[start:] = ema(series[start:], period)
    return out


def _first_value(data):
    """Index of the first bar that is not NaN, or the series length when there is none."""
    known = ~np.isnan(data)
    return int(np.argmax(known)) if known.any() else data.size


def dema(data, period):
    """Double EMA: 2 * EMA - EMA(EMA).

    Counting from the first bar that has a value, the second EMA is seeded
    period-1 bars in and the result starts period-1 bars after that. A NaN bar
    is NaN in the result; the EMAs keep their state across it.
    """
    data = _f(data)
    period = int(period)
    warmup = period - 1
    first = _first_value(data)
    ema1 = ema(data, period)
    ema2 = _ema_from(ema1, period, first + warmup)
    out = 2 * ema1 - ema2
    out[:first + 2 * warmup] = np.nan
    return out


def tema(data, period):
    """Triple EMA: 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA)).

    Counting from the first bar that has a value, each further EMA is seeded
    period-1 bars after the one before it and the result starts period-1 bars
    after the third. A NaN bar is NaN in the result; the EMAs keep their state
    across it.
    """
    data = _f(data)
    period = int(period)
    warmup = period - 1
    first = _first_value(data)
    ema1 = ema(data, period)
    ema2 = _ema_from(ema1, period, first + warmup)
    ema3 = _ema_from(ema2, period, first + 2 * warmup)
    out = 3 * ema1 - 3 * ema2 + ema3
    out[:first + 3 * warmup] = np.nan
    return out


def bbands(data, period, std_dev):
    middle = sma(data, period)
    sd = stdev(data, period)
    upper = middle + std_dev * sd
    lower = middle - std_dev * sd
    return upper, middle, lower


def stochastic(high, low, close, k_period, smooth_k, d_period):
    high, low, close = _f(high), _f(low), _f(close)
    if HAVE_RUST:
        return _rs.stochastic(high, low, close, int(k_period), int(smooth_k), int(d_period))
    hh = _roll(high, int(k_period), np.max)
    ll = _roll(low, int(k_period), np.min)
    n = close.size
    fast_k = np.full(n, np.nan)
    fk = k_period - 1
    for i in range(fk, n):
        fast_k[i] = 100.0 * (close[i] - ll[i]) / (hh[i] - ll[i]) if hh[i] != ll[i] else 50.0
    # Each warm-up is capped at the series length, so both lines have one value
    # per bar however short the input is.
    warm_k = min(n, fk)
    slow_k = np.concatenate([np.full(warm_k, np.nan), sma(fast_k[warm_k:], smooth_k)])
    warm_d = min(n, fk + smooth_k - 1)
    slow_d = np.concatenate([np.full(warm_d, np.nan), sma(slow_k[warm_d:], d_period)])
    return slow_k, slow_d


def cci(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.cci(high, low, close, period)
    n = close.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    tp = (high + low + close) / 3.0
    p = float(period)
    # Replicate the Rust kernel's exact float operation order: an incremental
    # running-sum SMA (not np.cumsum) and a sequential mean-absolute-deviation
    # (not np.mean). np.cumsum/np.mean round differently, which in exactly-flat
    # windows flips the residual sign and yields +-66.67 instead of matching Rust
    # (a fallback-vs-Rust divergence of up to +-133). This keeps the fallback
    # bit-identical to Rust. Shipped (Rust) output is unchanged.
    rsum = 0.0
    for k in range(period):
        rsum += tp[k]
    for i in range(period - 1, n):
        if i > period - 1:
            rsum = rsum + tp[i] - tp[i - period]
        sma_tp = rsum / p
        md = 0.0
        for j in range(period):
            md += abs(tp[i + 1 - period + j] - sma_tp)
        md /= p
        out[i] = (tp[i] - sma_tp) / (0.015 * md) if md != 0.0 else 0.0
    return out


def williams_r(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.williams_r(high, low, close, period)
    hh = _roll(high, period, np.max)
    ll = _roll(low, period, np.min)
    n = close.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = -100.0 * (hh[i] - close[i]) / (hh[i] - ll[i]) if hh[i] != ll[i] else -50.0
    return out


def vwma(data, volume, period):
    data, volume = _f(data), _f(volume)
    period = int(period)
    if HAVE_RUST:
        return _rs.vwma(data, volume, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    # A NaN in either leg blanks only the windows containing it. Masking both
    # legs (rather than relying on NaN arithmetic) matters because a NaN volume
    # would make `sv > 0` false and silently take the price fallback branch.
    bad = np.isnan(data) | np.isnan(volume)
    ones = np.ones(period)
    spv = np.convolve(np.where(bad, 0.0, data * volume), ones, "valid")
    sv = np.convolve(np.where(bad, 0.0, volume), ones, "valid")
    nans = np.convolve(bad.astype(np.float64), ones, "valid")
    vals = np.where(sv > 0, spv / np.where(sv == 0, 1, sv), data[period - 1:])
    out[period - 1:] = np.where(nans == 0, vals, np.nan)
    return out


def zlema(data, period):
    data = _f(data)
    period = int(period)
    n = data.size
    lag = (period - 1) // 2
    adjusted = data.copy()
    if lag > 0:
        adjusted[lag:] = 2.0 * data[lag:] - data[:n - lag]
    return ema(adjusted, period)


def t3(data, period, v_factor):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.t3(data, period, float(v_factor))

    def gd(x):
        e1 = ema(x, period)
        e2 = ema(e1, period)
        return (1.0 + v_factor) * e1 - v_factor * e2

    return gd(gd(gd(data)))


def kama_tv(data, length, fast_length, slow_length):
    data = _f(data)
    length, fast_length, slow_length = int(length), int(fast_length), int(slow_length)
    if HAVE_RUST:
        return _rs.kama_tv(data, length, fast_length, slow_length)
    n = data.size
    out = np.full(n, np.nan)
    if length <= 0 or n < length + 1:
        return out
    fa = 2.0 / (fast_length + 1)
    sa = 2.0 / (slow_length + 1)
    for i in range(length, n):
        mom = abs(data[i] - data[i - length])
        vol = 0.0
        for j in range(i - length + 1, i + 1):
            if j > 0:
                vol += abs(data[j] - data[j - 1])
        er = mom / vol if vol != 0 else 0.0
        alpha = (er * (fa - sa) + sa) ** 2
        prev = data[i] if (i == length or np.isnan(out[i - 1])) else out[i - 1]
        out[i] = alpha * data[i] + (1.0 - alpha) * prev
    return out


def trima(data, period):
    """Triangular MA = SMA(SMA(.)) via per-window np.mean (matches reference)."""
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.trima(data, period)
    n = data.size
    out = np.full(n, np.nan)
    n1 = (period + 1) // 2
    n2 = period - n1 + 1
    first = np.full(n, np.nan)
    for i in range(n1 - 1, n):
        first[i] = np.mean(data[i - n1 + 1:i + 1])
    for i in range(n1 + n2 - 2, n):
        if not np.isnan(first[i]):
            window = first[max(0, i - n2 + 1):i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= n2:
                out[i] = np.mean(valid[-n2:])
    return out


def ema_wilder(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.ema_wilder(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0:
        return out
    # Skip leading NaNs and seed from the first `period` valid values, matching the
    # Rust kernel (rust/oa_core ema_wilder). Without this, a NaN warm-up prefix
    # (e.g. the dx series inside adx) poisons the seed and the whole output is NaN.
    first_valid = _first_clean_window(data, period)
    if first_valid < 0:
        return out
    seed = data[first_valid:first_valid + period]
    start = first_valid + period - 1
    out[start] = seed.sum() / period
    for i in range(start + 1, n):
        v = data[i]
        out[i] = out[i - 1] if np.isnan(v) else (out[i - 1] * (period - 1) + v) / period
    return out


def alma(data, period, offset, sigma):
    data = _f(data)
    if HAVE_RUST:
        return _rs.alma(data, int(period), float(offset), float(sigma))
    n = data.size
    out = np.full(n, np.nan)
    period = int(period)
    if period <= 0 or n < period:
        return out
    m = offset * (period - 1)
    s = period / sigma
    w = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    w = w / w.sum()
    for i in range(period - 1, n):
        out[i] = float(np.dot(w, data[i - period + 1:i + 1]))
    return out


def mcginley(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.mcginley(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    out[period - 1] = data[:period].sum() / period
    for i in range(period, n):
        if out[i - 1] != 0:
            ratio = data[i] / out[i - 1]
            out[i] = out[i - 1] + (data[i] - out[i - 1]) / (period * ratio ** 4)
        else:
            out[i] = data[i]
    return out


def vidya(data, period, alpha):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.vidya(data, period, float(alpha))
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period + 1:
        return out
    cmo = np.full(n, np.nan)
    for i in range(period, n):
        g = ls = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                diff = data[j] - data[j - 1]
                if diff > 0:
                    g += diff
                elif diff < 0:
                    ls += -diff
        cmo[i] = 100 * (g - ls) / (g + ls) if (g + ls) != 0 else 0.0
    out[period] = data[period]
    for i in range(period + 1, n):
        sc = alpha * abs(cmo[i]) / 100
        out[i] = out[i - 1] + sc * (data[i] - out[i - 1])
    return out


def bop(open_, high, low, close):
    open_, high, low, close = _f(open_), _f(high), _f(low), _f(close)
    if HAVE_RUST:
        return _rs.bop(open_, high, low, close)
    n = close.size
    out = np.full(n, np.nan)
    rng = high - low
    nz = rng != 0
    out[nz] = (close[nz] - open_[nz]) / rng[nz]
    out[~nz] = 0.0
    return out


def elderray(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    e = ema(close, int(period))
    return high - e, low - e


# Fisher Transform as TradingView publishes it:
#   value = 0.66 * ((price - lowest) / (highest - lowest) - 0.5) + 0.67 * value[1]
#   fisher = 0.5 * ln((1 + value) / (1 - value)) + 0.5 * fisher[1]
# with value held at +-0.999 once it passes +-0.99, so the logarithm stays finite.
_FISHER_PRICE_WEIGHT = 0.66
_FISHER_CARRY_WEIGHT = 0.67
_FISHER_LIMIT = 0.99
_FISHER_HELD_AT = 0.999


def fisher(data, length):
    """Fisher Transform of `data` over `length` bars: returns (fisher, trigger).

    The trigger is the Fisher value before the bar, 0.0 at the start. Both are
    NaN for the first length-1 bars and while a NaN is in the window; the
    recursion keeps its state across such bars.
    """
    data = _f(data)
    length = int(length)
    if HAVE_RUST:
        return _rs.fisher(data, length)
    n = data.size
    fish = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    if length <= 0 or n < length:
        return fish, trigger
    # np.max and np.min are NaN for a window that holds a NaN.
    windows = np.lib.stride_tricks.sliding_window_view(data, length)
    highest_ = windows.max(axis=1).tolist()
    lowest_ = windows.min(axis=1).tolist()
    value = 0.0
    prev = 0.0
    first = length - 1
    for i in range(first, n):
        hi, lo = highest_[i - first], lowest_[i - first]
        if hi != hi:
            continue
        trigger[i] = prev
        if hi != lo:
            value = (_FISHER_PRICE_WEIGHT * ((data[i] - lo) / (hi - lo) - 0.5)
                     + _FISHER_CARRY_WEIGHT * value)
            if value > _FISHER_LIMIT:
                value = _FISHER_HELD_AT
            elif value < -_FISHER_LIMIT:
                value = -_FISHER_HELD_AT
            prev = 0.5 * np.log((1.0 + value) / (1.0 - value)) + 0.5 * prev
        fish[i] = prev
    return fish, trigger


def updown_streak(data):
    data = _f(data)
    if HAVE_RUST:
        return _rs.updown_streak(data)
    n = data.size
    s = np.zeros(n)
    for i in range(1, n):
        if data[i] == data[i - 1]:
            s[i] = 0.0
        elif data[i] > data[i - 1]:
            s[i] = 1.0 if s[i - 1] <= 0 else s[i - 1] + 1.0
        else:
            s[i] = -1.0 if s[i - 1] >= 0 else s[i - 1] - 1.0
    return s


def percent_rank(data, period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.percent_rank(data, int(period))
    return _roll(data, int(period), lambda w: (np.sum(w[:-1] < w[-1]) / period) * 100.0)


def crsi(data, lenrsi, lenupdown, lenroc):
    data = _f(data)
    price_rsi = rsi(data, int(lenrsi))
    streak_rsi = rsi(updown_streak(data), int(lenupdown))
    roc1 = roc(data, 1)
    pr = percent_rank(roc1, int(lenroc))
    out = np.full(data.size, np.nan)
    m = ~(np.isnan(price_rsi) | np.isnan(streak_rsi) | np.isnan(pr))
    out[m] = (price_rsi[m] + streak_rsi[m] + pr[m]) / 3.0
    return out


def roc(data, length):
    data = _f(data)
    length = int(length)
    if HAVE_RUST:
        return _rs.roc(data, length)
    n = data.size
    out = np.full(n, np.nan)
    prev = data[:-length] if length < n else np.array([])
    if length < n:
        nz = prev != 0
        idx = np.arange(length, n)
        out[idx[nz]] = (data[length:][nz] - prev[nz]) / prev[nz] * 100.0
    return out


def _shift(arr, k):
    out = np.full(len(arr), np.nan)
    if 0 <= k < len(arr):
        out[k:] = arr[:len(arr) - k]
    return out


def alligator(data, jaw_p, jaw_s, teeth_p, teeth_s, lips_p, lips_s):
    data = _f(data)
    jaw = _shift(ema_wilder(data, jaw_p), int(jaw_s))
    teeth = _shift(ema_wilder(data, teeth_p), int(teeth_s))
    lips = _shift(ema_wilder(data, lips_p), int(lips_s))
    return jaw, teeth, lips


def ma_envelopes(data, period, percentage, ma_type):
    data = _f(data)
    period = int(period)
    if ma_type.upper() == "SMA":
        ma = _roll(data, period, np.mean)
    elif ma_type.upper() == "EMA":
        ma = ema(data, period)
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")
    mult = percentage / 100
    return ma * (1 + mult), ma, ma * (1 - mult)


def ema_first_valid(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.ema_first_valid(data, period)
    n = data.size
    out = np.full(n, np.nan)
    alpha = 2.0 / (period + 1.0)
    fv = -1
    for i in range(n):
        if not np.isnan(data[i]):
            out[i] = data[i]
            fv = i
            break
    if fv == -1:
        return out
    # Carry the recursion in `prev`, not in `out[i - 1]`: an interior NaN leaves a
    # NaN in `out` and reading it back would poison every later bar.
    prev = out[fv]
    for i in range(fv + 1, n):
        if not np.isnan(data[i]):
            prev = alpha * data[i] + (1.0 - alpha) * prev
            out[i] = prev
    return out


def obv(close, volume):
    close, volume = _f(close), _f(volume)
    if HAVE_RUST:
        return _rs.obv(close, volume)
    n = close.size
    out = np.zeros(n)
    sign = np.where(close[1:] > close[:-1], 1.0, np.where(close[1:] < close[:-1], -1.0, 0.0))
    out[1:] = np.cumsum(sign * volume[1:])
    return out


def adl(high, low, close, volume):
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    if HAVE_RUST:
        return _rs.adl(high, low, close, volume)
    n = close.size
    out = np.full(n, np.nan)
    rng = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = np.where(rng != 0, ((close - low) - (high - close)) / np.where(rng == 0, 1, rng), 0.0)
    out[0] = 0.0
    out[1:] = np.cumsum((mfm * volume)[1:])
    return out


def cmf(high, low, close, volume, period):
    """Chaikin Money Flow: sum(money flow volume, period) / sum(volume, period).

    Money flow volume is ((close - low) - (high - close)) / (high - low) * volume,
    zero on a bar with no range. 0.0 when the window's volume is not positive.
    NaN for the first period-1 bars and while a NaN high, low, close or volume
    is in the window.
    """
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    _require_same_length(high=high, low=low, close=close, volume=volume)
    period = int(period)
    if HAVE_RUST:
        return _rs.cmf(high, low, close, volume, period)
    n = close.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    unknown = np.isnan(high) | np.isnan(low) | np.isnan(close) | np.isnan(volume)
    bar_range = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        multiplier = np.where(
            high != low,
            ((close - low) - (high - close)) / np.where(bar_range == 0, 1.0, bar_range),
            0.0,
        )
        flow_volume = multiplier * volume
        # Add the window one offset at a time, oldest bar first. That is the
        # order the Rust kernel sums each window in, so the two agree to the
        # last bit, which a difference of cumulative sums would not.
        windows = n - period + 1
        flow_sum = np.zeros(windows)
        volume_sum = np.zeros(windows)
        for offset in range(period):
            flow_sum += flow_volume[offset:offset + windows]
            volume_sum += volume[offset:offset + windows]
        ratio = np.where(
            volume_sum > 0, flow_sum / np.where(volume_sum > 0, volume_sum, 1.0), 0.0)
    out[period - 1:] = np.where(_unknown_in_window(unknown, period) > 0, np.nan, ratio)
    return out


def mfi(high, low, close, volume, period):
    """Money Flow Index, 0 to 100: 100 - 100 / (1 + positive flow / negative flow).

    A bar's flow is typical price times volume: positive when the typical price
    rose from the bar before, negative when it fell, zero when it did not move
    or there is no bar before to compare with (bar 0, or the bar after a NaN
    one). Flows are summed over `period` bars; 100 when the window has no
    negative flow. NaN on bar 0, for the first period-1 bars, and while a bar
    with a NaN price or volume is in the window.
    """
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    _require_same_length(high=high, low=low, close=close, volume=volume)
    period = int(period)
    if HAVE_RUST:
        return _rs.mfi(high, low, close, volume, period)
    n = close.size
    out = np.full(n, np.nan)
    if period <= 0 or n < 2:
        return out
    typical = (high + low + close) / 3.0
    unknown = np.isnan(typical) | np.isnan(volume)
    flow = typical * volume
    comparable = ~unknown[1:] & ~np.isnan(typical[:-1])
    positive = np.zeros(n)
    negative = np.zeros(n)
    positive[1:] = np.where(comparable & (typical[1:] > typical[:-1]), flow[1:], 0.0)
    negative[1:] = np.where(comparable & (typical[1:] < typical[:-1]), flow[1:], 0.0)
    # Running sums, updated add-then-drop like the Rust kernel so both backends
    # accumulate the same rounding. Python floats are IEEE doubles too, and
    # stepping through lists of them is far quicker than through an array.
    positive, negative = positive.tolist(), negative.tolist()
    positive_sum = [0.0] * n
    negative_sum = [0.0] * n
    pos = neg = 0.0
    for i in range(1, n):
        pos += positive[i]
        neg += negative[i]
        if i >= period:
            pos -= positive[i - period]
            neg -= negative[i - period]
        positive_sum[i] = pos
        negative_sum[i] = neg
    first = max(1, period - 1)
    pos_sum = np.asarray(positive_sum[first:])
    neg_sum = np.asarray(negative_sum[first:])
    # Unknown bars among the last `period` (fewer while the series is that short).
    seen = np.cumsum(unknown)
    in_window = seen - np.concatenate((np.zeros(min(period, n), dtype=seen.dtype), seen[:-period]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = pos_sum / np.where(neg_sum == 0.0, 1.0, neg_sum)
        index = np.where(neg_sum == 0.0, 100.0, 100.0 - 100.0 / (1.0 + ratio))
    out[first:] = np.where(in_window[first:] > 0, np.nan, index)
    return out


def emv(high, low, volume, length, divisor):
    high, low, volume = _f(high), _f(low), _f(volume)
    if HAVE_RUST:
        raw = _rs.emv_raw(high, low, volume, float(divisor))
    else:
        n = high.size
        raw = np.full(n, np.nan)
        for i in range(1, n):
            chl2 = (high[i] + low[i]) / 2 - (high[i - 1] + low[i - 1]) / 2
            hlr = high[i] - low[i]
            raw[i] = divisor * chl2 * hlr / volume[i] if (volume[i] > 0 and hlr > 0) else 0.0
    return _win_mean(raw, int(length))


def force_index(close, volume, length):
    close, volume = _f(close), _f(volume)
    n = close.size
    raw = np.full(n, np.nan)
    raw[1:] = volume[1:] * (close[1:] - close[:-1])
    return ema_first_valid(raw, int(length))


def rma_smma(data, length):
    """RMA/SMMA in the alpha=1/length, alpha*v+(1-alpha)*prev form (OBVSmoothed)."""
    data = _f(data)
    length = int(length)
    n = data.size
    out = np.full(n, np.nan)
    fv = _first_clean_window(data, length)
    if fv < 0:
        return out
    prev = data[fv:fv + length].sum() / length
    out[fv + length - 1] = prev
    alpha = 1.0 / length
    for i in range(fv + length, n):
        if not np.isnan(data[i]):
            prev = alpha * data[i] + (1.0 - alpha) * prev
            out[i] = prev
    return out


def session_vwap(source, volume, starts):
    """Anchored VWAP: returns (vwap, stdev), restarted where `starts` is nonzero.

    vwap = sum(price * volume) / sum(volume) over the period so far, stdev the
    volume-weighted standard deviation of the price around it. Both are NaN on
    a bar whose price or volume is NaN (the sums keep their state) and while
    the period has no volume yet.
    """
    source, volume = _f(source), _f(volume)
    starts = np.ascontiguousarray(starts, dtype=np.float64)
    _require_same_length(source=source, volume=volume, starts=starts)
    if HAVE_RUST:
        return _rs.session_vwap(source, volume, starts)
    n = source.size
    vwap = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    spv = sv = spv2 = 0.0
    for i in range(n):
        if starts[i] != 0.0 or i == 0:
            spv = sv = spv2 = 0.0
        if np.isnan(source[i]) or np.isnan(volume[i]):
            continue
        spv += source[i] * volume[i]
        sv += volume[i]
        spv2 += source[i] * source[i] * volume[i]
        if sv > 0:
            vwap[i] = spv / sv
            sd[i] = np.sqrt(max(0.0, spv2 / sv - vwap[i] ** 2))
    return vwap, sd


def adx(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.adx(high, low, close, period)
    n = close.size
    tr = true_range(high, low, close)
    dmp = np.zeros(n)
    dmm = np.zeros(n)
    if n > 1:
        up = high[1:] - high[:-1]
        dn = low[:-1] - low[1:]
        dmp[1:] = np.where((up > dn) & (up > 0), up, 0.0)
        dmm[1:] = np.where((dn > up) & (dn > 0), dn, 0.0)
    sm_atr = ema_wilder(tr, period)
    sm_dmp = ema_wilder(dmp, period)
    sm_dmm = ema_wilder(dmm, period)
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(sm_atr[i]) and sm_atr[i] > 0:
            di_plus[i] = (sm_dmp[i] / sm_atr[i]) * 100
            di_minus[i] = (sm_dmm[i] / sm_atr[i]) * 100
            dsum = di_plus[i] + di_minus[i]
            if dsum > 0:
                dx[i] = abs(di_plus[i] - di_minus[i]) / dsum * 100
    adx_ = ema_wilder(dx, period)
    return di_plus, di_minus, adx_


def aroon(high, low, period):
    high, low = _f(high), _f(low)
    period = int(period)
    if HAVE_RUST:
        return _rs.aroon(high, low, period)
    n = high.size
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    lookback = period + 1
    for i in range(lookback - 1, n):
        ws = i - lookback + 1
        hp = lp = 0
        for j in range(lookback):
            if high[ws + j] > high[ws + hp]:
                hp = j
            if low[ws + j] < low[ws + lp]:
                lp = j
        up[i] = 100.0 * (period - (lookback - 1 - hp)) / period
        down[i] = 100.0 * (period - (lookback - 1 - lp)) / period
    return up, down


def pivot_points(high, low, close):
    high, low, close = _f(high), _f(low), _f(close)
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3


_SAR_LONG = 1.0
_SAR_SHORT = -1.0


def sar(high, low, acceleration, maximum):
    """Parabolic SAR: returns (sar, trend), trend +1.0 while long and -1.0 while short.

    acceleration: starting value and step of the acceleration factor AF.
    maximum:      largest AF.

    The first bar that has a high and a low starts a long trend with stop = low
    and extreme point EP = high. No close series is taken, so the first trend
    is not read off the first two closes: it is always long, and a falling
    market reverses it on the next bar. Every later bar, in the order of
    TradingView's `ta.sar` reference script:

    1. stop = stop + AF * (EP - stop).
    2. Reversal, tested against that unclamped stop with strict comparisons.
       Long and stop > low: go short, stop = max(high, EP), EP = low, AF reset.
       Short and stop < high: go long, stop = min(low, EP), EP = high, AF reset.
       A bar that only touches the stop does not reverse.
    3. Without a reversal, a new extreme becomes EP and raises AF by
       `acceleration`, up to `maximum`.
    4. Clamp, on reversal bars too: when long the stop is at most the lows of
       the two previous bars, when short at least their highs.

    A bar with a NaN high or low is skipped: its outputs are NaN, the state is
    kept, and it does not count as one of the two previous bars.
    """
    high, low = _f(high), _f(low)
    _require_same_length(high=high, low=low)
    acceleration, maximum = float(acceleration), float(maximum)
    if HAVE_RUST:
        return _rs.sar(high, low, acceleration, maximum)
    n = high.size
    stops = [np.nan] * n
    trends = [np.nan] * n
    long = True
    stop = extreme = np.nan
    factor = acceleration
    previous = before_previous = None  # (high, low) of the last two bars that had both
    for i, (bar_high, bar_low) in enumerate(zip(high.tolist(), low.tolist(), strict=True)):
        if bar_high != bar_high or bar_low != bar_low:
            continue
        if previous is None:
            stop, extreme = bar_low, bar_high
        else:
            stop += factor * (extreme - stop)
            if long and stop > bar_low:
                long = False
                stop, extreme, factor = max(bar_high, extreme), bar_low, acceleration
            elif not long and stop < bar_high:
                long = True
                stop, extreme, factor = min(bar_low, extreme), bar_high, acceleration
            else:
                new_extreme = bar_high > extreme if long else bar_low < extreme
                if new_extreme:
                    extreme = bar_high if long else bar_low
                    factor = min(factor + acceleration, maximum)
            earlier = [bar for bar in (previous, before_previous) if bar is not None]
            if long:
                stop = min([stop] + [earlier_low for _, earlier_low in earlier])
            else:
                stop = max([stop] + [earlier_high for earlier_high, _ in earlier])
        stops[i] = stop
        trends[i] = _SAR_LONG if long else _SAR_SHORT
        previous, before_previous = (bar_high, bar_low), previous
    return np.asarray(stops, dtype=np.float64), np.asarray(trends, dtype=np.float64)


def rwi(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.rwi(high, low, close, period)
    n = close.size
    atr = _win_mean(true_range(high, low, close), period)
    rh = np.full(n, np.nan)
    rl = np.full(n, np.nan)
    sq = np.sqrt(period)
    for i in range(period, n):
        if atr[i] > 0:
            rh[i] = (high[i] - low[i - period]) / (atr[i] * sq)
            rl[i] = (high[i - period] - low[i]) / (atr[i] * sq)
    return rh, rl


def zigzag(high, low, close, deviation):
    high, low, close = _f(high), _f(low), _f(close)
    n = close.size
    out = np.full(n, np.nan)
    if n < 3:
        return out
    last_type = 0
    out[0] = close[0]
    ch, cl = high[0], low[0]
    chi, cli = 0, 0
    for i in range(1, n):
        if high[i] > ch:
            ch, chi = high[i], i
        if low[i] < cl:
            cl, cli = low[i], i
        if last_type != -1:
            if (ch - low[i]) / ch * 100 >= deviation:
                out[chi] = ch
                last_type = 1
                cl, cli = low[i], i
        if last_type != 1:
            if cl > 0 and (high[i] - cl) / cl * 100 >= deviation:
                out[cli] = cl
                last_type = -1
                ch, chi = high[i], i
    return out


def williams_fractals(high, low, periods):
    high, low = _f(high), _f(low)
    if HAVE_RUST:
        return _rs.williams_fractals(high, low, int(periods))
    n = int(periods)
    length = high.size
    fu = np.zeros(length, dtype=bool)
    fd = np.zeros(length, dtype=bool)
    for center in range(n, length - n):
        df = True
        for i in range(1, n + 1):
            if high[center - i] >= high[center]:
                df = False
                break
        if df:
            f = [True] * 5
            for i in range(1, n + 1):
                if center + i < length and high[center + i] >= high[center]:
                    f[0] = False
                for p in range(1, 5):
                    for q in range(1, p + 1):
                        if center + q < length and high[center + q] > high[center]:
                            f[p] = False
                    if center + i + p < length and high[center + i + p] >= high[center]:
                        f[p] = False
            fu[center] = any(f)
        df = True
        for i in range(1, n + 1):
            if low[center - i] <= low[center]:
                df = False
                break
        if df:
            f = [True] * 5
            for i in range(1, n + 1):
                if center + i < length and low[center + i] <= low[center]:
                    f[0] = False
                for p in range(1, 5):
                    for q in range(1, p + 1):
                        if center + q < length and low[center + q] < low[center]:
                            f[p] = False
                    if center + i + p < length and low[center + i + p] <= low[center]:
                        f[p] = False
            fd[center] = any(f)
    return fu, fd


def _linreg_end(y, period):
    x = np.arange(period)
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sx2 = np.sum(x * x)
    den = period * sx2 - sx * sx
    if den != 0:
        slope = (period * sxy - sx * sy) / den
        intercept = (sy - slope * sx) / period
        return slope, intercept
    return None, None


def linreg(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.linreg(data, period)
    n = data.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = data[i - period + 1:i + 1]
        slope, intercept = _linreg_end(y, period)
        out[i] = slope * (period - 1) + intercept if slope is not None else y[-1]
    return out


def lrslope(data, period, interval):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.lrslope(data, period, float(interval))
    n = data.size
    out = np.full(n, np.nan)

    def endval(y):
        slope, intercept = _linreg_end(y, period)
        return slope * (period - 1) + intercept if slope is not None else y[-1]

    for i in range(period, n):
        out[i] = (endval(data[i - period + 1:i + 1]) - endval(data[i - period:i])) / interval
    return out


def tsf(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.tsf(data, period)
    n = data.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = data[i - period + 1:i + 1]
        slope, intercept = _linreg_end(y, period)
        out[i] = slope * period + intercept if slope is not None else y[-1]
    return out


def correl(data1, data2, period):
    data1, data2 = _f(data1), _f(data2)
    period = int(period)
    if HAVE_RUST:
        return _rs.correl(data1, data2, period)
    n = data1.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        x = data1[i - period + 1:i + 1]
        y = data2[i - period + 1:i + 1]
        mx, my = np.mean(x), np.mean(y)
        num = np.sum((x - mx) * (y - my))
        den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
        out[i] = num / den if den > 0 else 0.0
    return out


def beta(asset, market, period):
    asset, market = _f(asset), _f(market)
    period = int(period)
    if HAVE_RUST:
        return _rs.beta(asset, market, period)
    n = asset.size
    out = np.full(n, np.nan)
    ar = np.full(n, np.nan)
    mr = np.full(n, np.nan)
    ar[1:] = asset[1:] - asset[:-1]
    mr[1:] = market[1:] - market[:-1]
    for i in range(period, n):
        aw = ar[i - period + 1:i + 1]
        mw = mr[i - period + 1:i + 1]
        ma, mm = np.mean(aw), np.mean(mw)
        cov = mvar = 0.0
        for j in range(period):
            ad = aw[j] - ma
            md = mw[j] - mm
            cov += ad * md
            mvar += md * md
        cov /= period
        mvar /= period
        out[i] = cov / mvar if mvar > 0 else 0.0
    return out


def variance(data, lookback, mode, ema_period, filter_lookback, ema_length, return_components):
    data = _f(data)
    lookback = int(lookback)
    n = data.size
    if HAVE_RUST:
        var = _rs.variance(data, lookback, mode == "LR")
        with np.errstate(invalid="ignore"):
            sd = np.sqrt(var)
    else:
        source = np.full(n, np.nan)
        if mode == "LR":
            with np.errstate(invalid="ignore", divide="ignore"):
                valid = (data[1:] > 0) & (data[:-1] > 0)
            ratio = np.full(n - 1, np.nan)
            ratio[valid] = np.log(data[1:][valid] / data[:-1][valid]) * 100
            source[1:] = ratio
        else:
            source = data.copy()
        var = np.full(n, np.nan)
        sd = np.full(n, np.nan)
        if n >= lookback and lookback > 1:
            rs = rsq = 0.0
            for i in range(lookback):
                if not np.isnan(source[i]):
                    rs += source[i]
                    rsq += source[i] * source[i]
            mean = rs / lookback
            v = (rsq - lookback * mean * mean) / (lookback - 1)
            if v >= 0:
                var[lookback - 1] = v
                sd[lookback - 1] = np.sqrt(v)
            for i in range(lookback, n):
                old, new = source[i - lookback], source[i]
                if not np.isnan(old):
                    rs -= old
                    rsq -= old * old
                if not np.isnan(new):
                    rs += new
                    rsq += new * new
                mean = rs / lookback
                v = (rsq - lookback * mean * mean) / (lookback - 1)
                if v >= 0:
                    var[i] = v
                    sd[i] = np.sqrt(v)
    if not return_components:
        return var
    ema_var = ema(var, int(ema_period))
    var_sma = sma(var, int(filter_lookback))
    var_sd = stdev(var, int(filter_lookback))
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where((~np.isnan(var)) & (~np.isnan(var_sma)) & (~np.isnan(var_sd)) & (var_sd > 0),
                          (var - var_sma) / np.where(var_sd == 0, 1.0, var_sd), np.nan)
    ema_z = ema(zscore, int(ema_length))
    return var, ema_var, zscore, ema_z, sd


def median(data, period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.median(data, int(period))
    return _roll(data, int(period), np.median)


def _median_pnr(data, period):
    period = int(period)
    n = data.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = np.sort(data[i - period + 1:i + 1])
        out[i] = w[period // 2] if period % 2 == 1 else w[period // 2 - 1]
    return out


def median_bands(high, low, close, source, median_length, atr_length, atr_mult):
    high, low, close = _f(high), _f(low), _f(close)
    src = (high + low) / 2.0 if source is None else _f(source)
    med = _median_pnr(src, int(median_length))
    atr = atr_wilder(high, low, close, int(atr_length)) * atr_mult
    return med, med + atr, med - atr, ema_first_valid(med, int(median_length))


def mode(data, period, bins):
    data = _f(data)
    period, bins = int(period), int(bins)
    if HAVE_RUST:
        return _rs.mode(data, period, bins)
    n = data.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = data[i - period + 1:i + 1]
        mn, mx = np.min(w), np.max(w)
        if mx > mn:
            bw = (mx - mn) / bins
            bi = ((w - mn) / bw).astype(np.int32)
            bi = np.clip(bi, 0, bins - 1)
            counts = np.bincount(bi, minlength=bins)
            mode_bin = np.argmax(counts)
            out[i] = mn + (mode_bin + 0.5) * bw
        else:
            out[i] = w[0]
    return out


def vi(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    n = close.size
    vmp = np.full(n, np.nan)
    vmm = np.full(n, np.nan)
    if n > 1:
        vmp[1:] = np.abs(high[1:] - low[:-1])
        vmm[1:] = np.abs(low[1:] - high[:-1])
    atr1 = true_range(high, low, close)  # tr[0]=high-low, matches atr_single
    vmpr = rolling_sum(vmp, period)
    vmmr = rolling_sum(vmm, period)
    strr = rolling_sum(atr1, period)
    vip = np.full(n, np.nan)
    vim = np.full(n, np.nan)
    sl = slice(period, n)
    sr = strr[sl]
    ok = (sr > 0) & ~np.isnan(sr)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.where(sr == 0, 1.0, sr)
        vip[sl] = np.where(ok, vmpr[sl] / den, 0.0)
        vim[sl] = np.where(ok, vmmr[sl] / den, 0.0)
    return vip, vim


def gator(high, low, jaw_period, teeth_period, lips_period):
    high, low = _f(high), _f(low)
    hl2 = (high + low) / 2.0
    jaw = _shift(ema_wilder(hl2, int(jaw_period)), 8)
    teeth = _shift(ema_wilder(hl2, int(teeth_period)), 5)
    lips = _shift(ema_wilder(hl2, int(lips_period)), 3)
    return np.abs(jaw - teeth), -np.abs(teeth - lips)


def _stoch_single(data, period):
    if HAVE_RUST:
        return _rs.stoch_single(_f(data), int(period))
    n = data.size
    out = np.full(n, np.nan)
    for i in range(int(period) - 1, n):
        w = data[i - period + 1:i + 1]
        wc = w[~np.isnan(w)]
        if len(wc):
            hi, lo = wc.max(), wc.min()
            if hi != lo and not np.isnan(data[i]):
                out[i] = (data[i] - lo) / (hi - lo) * 100.0
            elif hi == lo:
                out[i] = 50.0
    return out


def stc(data, fast_length, slow_length, cycle_length, d1_length, d2_length):
    data = _f(data)
    macd_ = ema(data, int(fast_length)) - ema(data, int(slow_length))
    k = _stoch_single(macd_, int(cycle_length))
    k = np.where(np.isnan(k), 0.0, k)
    d = ema(k, int(d1_length))
    kd = _stoch_single(d, int(cycle_length))
    kd = np.where(np.isnan(kd), 0.0, kd)
    s = ema(kd, int(d2_length))
    return np.clip(s, 0.0, 100.0)


def _wma_nan(data, period):
    if HAVE_RUST:
        return _rs.wma_nan(_f(data), int(period))
    n = data.size
    out = np.full(n, np.nan)
    period = int(period)
    for i in range(period - 1, n):
        if not np.isnan(data[i]):
            ws = vw = 0.0
            for j in range(period):
                idx = i - period + 1 + j
                if not np.isnan(data[idx]):
                    w = j + 1
                    ws += data[idx] * w
                    vw += w
            if vw > 0:
                out[i] = ws / vw
    return out


def coppock(data, wma_length, long_roc_length, short_roc_length):
    data = _f(data)
    rs = roc(data, int(long_roc_length)) + roc(data, int(short_roc_length))
    return _wma_nan(rs, int(wma_length))


def swma(data):
    """Symmetrically Weighted MA: weights [1,2,2,1]/6 over the last 4 bars."""
    data = _f(data)
    n = data.size
    out = np.full(n, np.nan)
    if n >= 4:
        out[3:] = (data[:-3] + 2 * data[1:-2] + 2 * data[2:-1] + data[3:]) / 6.0
    return out


_SWMA_WARMUP = 3  # bars before the 4-bar swma has its first value


def rvi_vigor(open_, high, low, close, period):
    """Relative Vigor Index: returns (rvi, signal).

    rvi = sum(swma(close - open), period) / sum(swma(high - low), period) and
    signal = swma(rvi) (John Ehlers, "Relative Vigor Index", Stocks &
    Commodities, 2002; the formula TradingView publishes). The first value
    is at bar period + 2, the first signal three bars later. A bar is NaN while
    any bar feeding its window is NaN.
    """
    open_, high, low, close = _f(open_), _f(high), _f(low), _f(close)
    _require_same_length(open=open_, high=high, low=low, close=close)
    period = int(period)
    n = close.size
    first = period + _SWMA_WARMUP - 1
    if period < 0 or n <= first:
        return np.full(n, np.nan), np.full(n, np.nan)
    if HAVE_RUST:
        return _rs.rvi_vigor(open_, high, low, close, period)
    body = swma(close - open_)
    bar_range = swma(high - low)
    # Sum each window oldest bar first (the Rust kernel's order). A NaN bar
    # makes the sums it enters NaN, and such a bar stays NaN.
    windows = n - first
    body_sum = np.zeros(windows)
    range_sum = np.zeros(windows)
    for offset in range(_SWMA_WARMUP, _SWMA_WARMUP + period):
        body_sum += body[offset:offset + windows]
        range_sum += bar_range[offset:offset + windows]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            range_sum != 0.0, body_sum / np.where(range_sum != 0.0, range_sum, 1.0), 0.0)
    rvi = np.full(n, np.nan)
    rvi[first:] = np.where(np.isnan(body_sum) | np.isnan(range_sum), np.nan, ratio)
    return rvi, swma(rvi)


def kst(data, roclen1, roclen2, roclen3, roclen4, smalen1, smalen2, smalen3, smalen4, siglen):
    data = _f(data)
    sm1 = _win_mean(roc_osc(data, roclen1), int(smalen1))
    sm2 = _win_mean(roc_osc(data, roclen2), int(smalen2))
    sm3 = _win_mean(roc_osc(data, roclen3), int(smalen3))
    sm4 = _win_mean(roc_osc(data, roclen4), int(smalen4))
    k = sm1 * 1 + sm2 * 2 + sm3 * 3 + sm4 * 4
    return k, _win_mean(k, int(siglen))


def tsi(data, long_period, short_period, signal_period):
    data = _f(data)
    pc = np.concatenate([[0.0], np.diff(data)])
    apc = np.abs(pc)
    pcs2 = ema(ema(pc, int(long_period)), int(short_period))
    apcs2 = ema(ema(apc, int(long_period)), int(short_period))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(apcs2 != 0, 100 * (pcs2 / np.where(apcs2 == 0, 1.0, apcs2)), 0.0)
    return t, ema(t, int(signal_period))


def uo(high, low, close, period1, period2, period3):
    high, low, close = _f(high), _f(low), _f(close)
    n = close.size
    tl = np.empty(n)
    if n:
        tl[0] = low[0]
        tl[1:] = np.minimum(low[1:], close[:-1])
    tr = true_range(high, low, close)
    bp = close - tl

    def avg(p):
        bps = _roll(bp, int(p), np.sum)
        trs = _roll(tr, int(p), np.sum)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(trs > 0, bps / np.where(trs == 0, 1.0, trs), 0.0)

    out = 100.0 * (4 * avg(period1) + 2 * avg(period2) + avg(period3)) / 7.0
    mp = max(int(period1), int(period2), int(period3))
    out[:mp - 1] = np.nan
    return out


def stochrsi(data, rsi_period, stoch_period, k_period, d_period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.stochrsi(data, int(rsi_period), int(stoch_period), int(k_period), int(d_period))
    r = rsi(data, int(rsi_period))
    n = r.size
    sr = np.full(n, np.nan)
    for i in range(int(stoch_period) - 1, n):
        w = r[i - stoch_period + 1:i + 1]
        wc = w[~np.isnan(w)]
        if len(wc):
            hi, lo = wc.max(), wc.min()
            sr[i] = (r[i] - lo) / (hi - lo) * 100 if hi != lo else 50.0

    def smooth(src, p):
        out = np.full(n, np.nan)
        for i in range(int(p) - 1, n):
            w = src[i - p + 1:i + 1]
            wc = w[~np.isnan(w)]
            if len(wc):
                out[i] = wc.mean()
        return out

    k = smooth(sr, k_period)
    d = smooth(k, d_period)
    return k, d


def cho(high, low, close, volume, fast_period, slow_period):
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    if HAVE_RUST:
        return _rs.adosc(high, low, close, volume, int(fast_period), int(slow_period))
    rng = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = np.where(rng != 0, ((close - low) - (high - close)) / np.where(rng == 0, 1.0, rng), 0.0)
    adl_ = np.cumsum(clv * volume)
    return ema(adl_, int(fast_period)) - ema(adl_, int(slow_period))


def chop(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    tr = true_range(high, low, close)
    atr_sum = rolling_sum(tr, period)
    rng = highest(high, period) - lowest(low, period)
    out = np.full(high.size, np.nan)
    valid = (rng > 0) & (atr_sum > 0) & ~np.isnan(rng) & ~np.isnan(atr_sum)
    if period > 1:
        lp = np.log10(period)
        out[valid] = 100 * np.log10(atr_sum[valid] / rng[valid]) / lp
    invalid = (~valid) & ~np.isnan(atr_sum)
    out[invalid] = 50.0
    return out


def cmo(data, period):
    """Chande Momentum Oscillator, -100 to 100: 100 * (up - down) / (up + down).

    `up` and `down` are the sums of the rises and of the falls among the last
    `period` bar-to-bar changes of `data`. 0.0 when nothing moved. NaN for the
    first `period` bars and while a NaN is among the period+1 bars those
    changes span.
    """
    data = _f(data)
    period = int(period)
    n = data.size
    out = np.full(n, np.nan)
    if period < 0 or period >= n:
        return out
    if HAVE_RUST:
        return _rs.cmo(data, period)
    changes = np.diff(data)
    # A change next to a NaN bar is NaN: neither a rise nor a fall.
    unknown = np.isnan(changes)
    rises = np.where(changes > 0.0, changes, 0.0).tolist()
    falls = np.where(changes < 0.0, -changes, 0.0).tolist()

    def oscillator(up, down):
        total = up + down
        return ((up - down) / total) * 100.0 if total > 0.0 else 0.0

    # Rolling sums, updated drop-then-add like the Rust kernel so both backends
    # accumulate the same rounding. Bar i ends the window of changes i-period..i-1.
    # (A plain loop, not sum(): the built-in compensates float rounding.)
    up = down = 0.0
    for rise, fall in zip(rises[:period], falls[:period], strict=True):
        up += rise
        down += fall
    values = [oscillator(up, down)]
    for i in range(period + 1, n):
        up = up - rises[i - period - 1] + rises[i - 1]
        down = down - falls[i - period - 1] + falls[i - 1]
        values.append(oscillator(up, down))
    in_window = _unknown_in_window(unknown, period) if period > 0 else np.zeros(n, dtype=int)
    out[period:] = np.where(in_window > 0, np.nan, values)
    return out


def roc_osc(data, period):
    """ROC with else-0 (oscillators.ROC): 0 when divisor is 0, NaN warm-up."""
    data = _f(data)
    period = int(period)
    n = data.size
    out = np.full(n, np.nan)
    if period >= n:
        return out
    prev = data[:-period]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[period:] = np.where(prev != 0, (data[period:] - prev) / np.where(prev == 0, 1.0, prev) * 100.0, 0.0)
    return out


def trix(data, length):
    data = _f(data)
    length = int(length)
    e = ema(ema(ema(np.log(data), length), length), length)
    out = np.full(data.size, np.nan)
    out[1:] = (e[1:] - e[:-1]) * 10000.0
    return out


def ao(high, low, fast_period, slow_period):
    high, low = _f(high), _f(low)
    mp = (high + low) / 2.0
    return _win_mean(mp, int(fast_period)) - _win_mean(mp, int(slow_period))


def ac(high, low, period):
    a = ao(high, low, 5, 34)
    return a - _win_mean(a, int(period))


def ppo(data, fast_period, slow_period, signal_period):
    data = _f(data)
    fe = ema(data, int(fast_period))
    se = ema(data, int(slow_period))
    with np.errstate(invalid="ignore", divide="ignore"):
        ppo_line = np.where(se != 0, (fe - se) / np.where(se == 0, 1.0, se) * 100.0, 0.0)
    signal_line = ema(ppo_line, int(signal_period))
    return ppo_line, signal_line, ppo_line - signal_line


def price_oscillator(data, fast_period, slow_period, ma_type):
    data = _f(data)
    if ma_type.upper() == "SMA":
        fast_ma = _win_mean(data, int(fast_period))
        slow_ma = _win_mean(data, int(slow_period))
    elif ma_type.upper() == "EMA":
        fast_ma = ema(data, int(fast_period))
        slow_ma = ema(data, int(slow_period))
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")
    return fast_ma - slow_ma


def dpo(data, period, is_centered):
    data = _f(data)
    period = int(period)
    n = data.size
    sma_ = _win_mean(data, period)
    barsback = int(period / 2 + 1)
    out = np.full(n, np.nan)
    if barsback >= n:
        return out
    if is_centered:
        sh = sma_[barsback:]
        out[barsback:] = np.where(~np.isnan(sh), data[:n - barsback] - sh, np.nan)
    else:
        sh = sma_[:n - barsback]
        out[barsback:] = np.where(~np.isnan(sh), data[barsback:] - sh, np.nan)
    return out


def aroon_osc(high, low, period):
    high, low = _f(high), _f(low)
    period = int(period)
    if HAVE_RUST:
        up, down = _rs.aroon(high, low, period)
        return up - down
    n = high.size
    out = np.full(n, np.nan)
    lookback = period + 1
    for i in range(lookback - 1, n):
        hw = high[i - lookback + 1:i + 1]
        lw = low[i - lookback + 1:i + 1]
        hp = lp = 0
        for j in range(len(hw)):
            if hw[j] > hw[hp]:
                hp = j
            if lw[j] < lw[lp]:
                lp = j
        bsh = len(hw) - 1 - hp
        bsl = len(lw) - 1 - lp
        out[i] = 100 * (period - bsh) / period - 100 * (period - bsl) / period
    return out


def vwma_strict(values, volume, length):
    """Per-window VWMA that skips NaN and yields NaN when window volume <= 0
    (matches OBVSmoothed._calculate_vwma)."""
    values, volume = _f(values), _f(volume)
    length = int(length)
    n = values.size
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        sw = sv = 0.0
        for j in range(i - length + 1, i + 1):
            if not np.isnan(values[j]) and not np.isnan(volume[j]):
                sw += values[j] * volume[j]
                sv += volume[j]
        if sv > 0:
            out[i] = sw / sv
    return out


# ---------------------------------------------------------------------------
# TA-Lib-compatible additions (new indicators; match TA-Lib exactly by definition)
# ---------------------------------------------------------------------------

def avgprice(open_, high, low, close):
    return (_f(open_) + _f(high) + _f(low) + _f(close)) / 4.0


def medprice(high, low):
    return (_f(high) + _f(low)) / 2.0


def typprice(high, low, close):
    return (_f(high) + _f(low) + _f(close)) / 3.0


def wclprice(high, low, close):
    return (_f(high) + _f(low) + 2.0 * _f(close)) / 4.0


def midpoint(data, period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.midpoint(data, int(period))
    return (highest(data, int(period)) + lowest(data, int(period))) / 2.0


def midprice(high, low, period):
    high, low = _f(high), _f(low)
    if HAVE_RUST:
        return _rs.midprice(high, low, int(period))
    return (highest(high, int(period)) + lowest(low, int(period))) / 2.0


def mom(data, period):
    """Momentum = data - data[period] (TA-Lib MOM)."""
    data = _f(data)
    period = int(period)
    n = data.size
    out = np.full(n, np.nan)
    if period < n:
        out[period:] = data[period:] - data[:-period]
    return out


def rocp(data, period):
    data = _f(data)
    period = int(period)
    n = data.size
    out = np.full(n, np.nan)
    if period < n:
        prev = data[:-period]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[period:] = (data[period:] - prev) / prev
    return out


def rocr(data, period):
    data = _f(data)
    period = int(period)
    n = data.size
    out = np.full(n, np.nan)
    if period < n:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[period:] = data[period:] / data[:-period]
    return out


def rocr100(data, period):
    return rocr(data, period) * 100.0


def apo(data, fast_period, slow_period, ma_type):
    """Absolute Price Oscillator = MA(fast) - MA(slow) (TA-Lib APO; matype SMA/EMA)."""
    data = _f(data)
    if ma_type.upper() == "EMA":
        return ema(data, int(fast_period)) - ema(data, int(slow_period))
    return _win_mean(data, int(fast_period)) - _win_mean(data, int(slow_period))


# --- TA-Lib-faithful directional-movement family (matches TA-Lib seeding) ---

def plus_dm(high, low, period):
    high, low = _f(high), _f(low)
    period = int(period)
    if HAVE_RUST:
        return _rs.plus_dm_talib(high, low, period)
    n = high.size
    out = np.full(n, np.nan)
    if period == 0 or n < period:
        return out
    prev = 0.0
    for i in range(1, period):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        if dp > 0.0 and dp > dm:
            prev += dp
    out[period - 1] = prev
    for i in range(period, n):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        add = dp if (dp > 0.0 and dp > dm) else 0.0
        prev = prev - prev / period + add
        out[i] = prev
    return out


def minus_dm(high, low, period):
    high, low = _f(high), _f(low)
    period = int(period)
    if HAVE_RUST:
        return _rs.minus_dm_talib(high, low, period)
    n = high.size
    out = np.full(n, np.nan)
    if period == 0 or n < period:
        return out
    prev = 0.0
    for i in range(1, period):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        if dm > 0.0 and dm > dp:
            prev += dm
    out[period - 1] = prev
    for i in range(period, n):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        add = dm if (dm > 0.0 and dm > dp) else 0.0
        prev = prev - prev / period + add
        out[i] = prev
    return out


def dx(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.dx_talib(high, low, close, period)
    n = close.size
    out = np.full(n, np.nan)
    if period == 0 or n < period + 1:
        return out
    tr = true_range(high, low, close)
    s_tr = s_p = s_m = 0.0
    for i in range(1, period):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        if dp > 0.0 and dp > dm:
            s_p += dp
        elif dm > 0.0 and dm > dp:
            s_m += dm
        s_tr += tr[i]
    for i in range(period, n):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        s_tr = s_tr - s_tr / period + tr[i]
        s_p = s_p - s_p / period + (dp if (dp > 0.0 and dp > dm) else 0.0)
        s_m = s_m - s_m / period + (dm if (dm > 0.0 and dm > dp) else 0.0)
        if s_tr != 0.0:
            pdi = 100.0 * s_p / s_tr
            mdi = 100.0 * s_m / s_tr
            den = pdi + mdi
            out[i] = (100.0 * abs(pdi - mdi) / den) if den != 0.0 else 0.0
        else:
            out[i] = 0.0
    return out


def adx_talib(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.adx_talib(high, low, close, period)
    n = close.size
    out = np.full(n, np.nan)
    if period == 0 or n < 2 * period:
        return out
    tr = true_range(high, low, close)
    s_tr = s_p = s_m = 0.0
    for i in range(1, period):
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        if dp > 0.0 and dp > dm:
            s_p += dp
        elif dm > 0.0 and dm > dp:
            s_m += dm
        s_tr += tr[i]

    def _step(i):
        nonlocal s_tr, s_p, s_m
        dp = high[i] - high[i - 1]
        dm = low[i - 1] - low[i]
        s_tr = s_tr - s_tr / period + tr[i]
        s_p = s_p - s_p / period + (dp if (dp > 0.0 and dp > dm) else 0.0)
        s_m = s_m - s_m / period + (dm if (dm > 0.0 and dm > dp) else 0.0)
        if s_tr != 0.0:
            pdi = 100.0 * s_p / s_tr
            mdi = 100.0 * s_m / s_tr
            den = pdi + mdi
            if den != 0.0:
                return 100.0 * abs(pdi - mdi) / den
        return 0.0

    sumdx = 0.0
    for i in range(period, 2 * period):
        sumdx += _step(i)
    prevadx = sumdx / period
    out[2 * period - 1] = prevadx
    for i in range(2 * period, n):
        d = _step(i)
        prevadx = (prevadx * (period - 1) + d) / period
        out[i] = prevadx
    return out


def adxr(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.adxr_talib(high, low, close, period)
    a = adx_talib(high, low, close, period)
    n = a.size
    out = np.full(n, np.nan)
    if period == 0:
        return out
    off = period - 1
    start = 2 * period - 1 + off
    for i in range(start, n):
        if not (np.isnan(a[i]) or np.isnan(a[i - off])):
            out[i] = (a[i] + a[i - off]) / 2.0
    return out


def stochf(high, low, close, fastk_period, fastd_period):
    high, low, close = _f(high), _f(low), _f(close)
    fastk_period, fastd_period = int(fastk_period), int(fastd_period)
    if HAVE_RUST:
        return _rs.stochf(high, low, close, fastk_period, fastd_period)
    n = close.size
    k = np.full(n, np.nan)
    d = np.full(n, np.nan)
    if fastk_period == 0 or fastd_period == 0 or n < fastk_period:
        return k, d
    hh = highest(high, fastk_period)
    ll = lowest(low, fastk_period)
    raw = np.full(n, np.nan)
    for i in range(fastk_period - 1, n):
        rng = hh[i] - ll[i]
        raw[i] = (100.0 * (close[i] - ll[i]) / rng) if rng != 0.0 else 0.0
    start = fastk_period - 1 + fastd_period - 1
    for i in range(start, n):
        d[i] = raw[i + 1 - fastd_period:i + 1].sum() / fastd_period
        k[i] = raw[i]
    return k, d


def linreg_angle(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.linreg_angle(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period == 0 or n < period:
        return out
    x = np.arange(period, dtype=np.float64)
    sx = x.sum()
    den = period * (x * x).sum() - sx * sx
    for i in range(period - 1, n):
        y = data[i + 1 - period:i + 1]
        slope = (period * (x * y).sum() - sx * y.sum()) / den if den != 0.0 else 0.0
        out[i] = np.degrees(np.arctan(slope))
    return out


def linreg_intercept(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.linreg_intercept(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period == 0 or n < period:
        return out
    x = np.arange(period, dtype=np.float64)
    sx = x.sum()
    den = period * (x * x).sum() - sx * sx
    for i in range(period - 1, n):
        y = data[i + 1 - period:i + 1]
        sy = y.sum()
        slope = (period * (x * y).sum() - sx * sy) / den if den != 0.0 else 0.0
        out[i] = (sy - slope * sx) / period
    return out


def nvi(close, volume):
    close, volume = _f(close), _f(volume)
    n = close.size
    out = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        roc_ = np.where(close[:-1] != 0, (close[1:] - close[:-1]) / close[:-1] * 100.0, 0.0)
    add = np.where(volume[1:] < volume[:-1], roc_, 0.0)
    out[1:] = np.cumsum(add)
    return out


def pvi(close, volume, initial_value):
    close, volume = _f(close), _f(volume)
    n = close.size
    out = np.full(n, np.nan)
    out[0] = initial_value
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((volume[1:] > volume[:-1]) & (close[:-1] != 0),
                         close[1:] / np.where(close[:-1] == 0, 1.0, close[:-1]), 1.0)
    # Thread the initial value through cumprod so the left-to-right multiply order
    # matches the reference's sequential prev*ratio exactly (bit-for-bit).
    factors = np.empty(n)
    factors[0] = initial_value
    factors[1:] = ratio
    out[:] = np.cumprod(factors)
    return out


def pvt(close, volume):
    close, volume = _f(close), _f(volume)
    n = close.size
    out = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        term = np.where(close[:-1] != 0,
                        (close[1:] - close[:-1]) / np.where(close[:-1] == 0, 1.0, close[:-1]) * volume[1:],
                        0.0)
    out[1:] = np.cumsum(term)
    return out


def vroc(volume, period):
    volume = _f(volume)
    period = int(period)
    n = volume.size
    out = np.full(n, np.nan)
    if period >= n:
        return out
    prev = volume[:-period]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[period:] = np.where(prev != 0, (volume[period:] - prev) / np.where(prev == 0, 1.0, prev) * 100.0, 0.0)
    return out


def volosc(volume, short_length, long_length):
    volume = _f(volume)
    se = ema_first_valid(volume, int(short_length))
    le = ema_first_valid(volume, int(long_length))
    with np.errstate(invalid="ignore", divide="ignore"):
        vo = np.where((~np.isnan(le)) & (le != 0), 100.0 * (se - le) / le, np.nan)
    return vo


def kvo(high, low, close, volume, trig_len, fast_x, slow_x):
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    n = close.size
    hlc3 = (high + low + close) / 3.0
    xt = np.empty(n)
    if n:
        xt[0] = volume[0] * 100.0
        up = hlc3[1:] > hlc3[:-1]
        xt[1:] = np.where(up, volume[1:] * 100.0, -volume[1:] * 100.0)
    xfast = ema(xt, int(fast_x))
    xslow = ema(xt, int(slow_x))
    xkvo = xfast - xslow
    xtrig = ema(xkvo, int(trig_len))
    return xkvo, xtrig


def rvol(volume, period):
    volume = _f(volume)
    period = int(period)
    avg = _win_mean(volume, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(avg > 0, volume / np.where(avg == 0, 1.0, avg), 1.0)
    out[np.isnan(avg)] = np.nan
    return out


def _win_mean(data, period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.win_mean(data, int(period))
    return _roll(data, int(period), np.mean)


def _win_std(data, period):
    """Per-window population std; NaN if the window contains any NaN (TV ta.stdev)."""
    data = _f(data)
    if HAVE_RUST:
        return _rs.win_std(data, int(period))

    def f(w):
        if np.isnan(w).any():
            return np.nan
        m = w.mean()
        return np.sqrt(np.mean((w - m) ** 2))
    return _roll(data, int(period), f)


def mass(high, low, length):
    high, low = _f(high), _f(low)
    span = high - low
    e1 = ema(span, 9)
    e2 = ema(e1, 9)
    cond = (e2 != 0) & ~np.isnan(e1) & ~np.isnan(e2)
    ratio = np.where(cond, e1 / np.where(e2 == 0, 1.0, e2), np.nan)
    return rolling_sum(ratio, int(length))


def _bb_bands(data, period, std_dev):
    m = _win_mean(data, period)
    sd = _win_std(data, period)
    return m + sd * std_dev, m, m - sd * std_dev


def bbpercent(data, period, std_dev):
    data = _f(data)
    up, m, lo = _bb_bands(data, int(period), std_dev)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (data - lo) / (up - lo)
    out[(~np.isnan(up)) & (up == lo)] = 0.5
    return out


def bbwidth(data, period, std_dev):
    data = _f(data)
    up, m, lo = _bb_bands(data, int(period), std_dev)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (up - lo) / m
    out[(~np.isnan(m)) & (m == 0)] = 0.0
    return out


def chandelier_exit(high, low, close, period, multiplier):
    high, low, close = _f(high), _f(low), _f(close)
    atr = _win_mean(true_range(high, low, close), int(period))
    hh = highest(high, int(period))
    ll = lowest(low, int(period))
    return hh - atr * multiplier, ll + atr * multiplier


def hv(close, length, annual, per):
    close = _f(close)
    n = close.size
    lr = np.full(n, np.nan)
    if n > 1:
        valid = (close[:-1] > 0) & (close[1:] > 0)
        idx = np.arange(1, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = close[1:] / close[:-1]
        lr[idx[valid]] = np.log(ratios[valid])
    sd = _win_std(lr, int(length))
    return 100.0 * sd * np.sqrt(annual / per)


def ulcerindex(data, length, smooth_length, signal_length, signal_type, return_signal):
    data = _f(data)
    hh = highest(data, int(length))
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where((~np.isnan(hh)) & (hh != 0),
                      100.0 * (data - hh) / np.where(hh == 0, 1.0, hh), np.nan)
    # A rolling mean of squares can land a hair below zero from rounding in the
    # running sum; that is zero, not NaN. np.maximum keeps NaN where the mean is NaN.
    ulcer = np.sqrt(np.maximum(sma(dd ** 2, int(smooth_length)), 0.0))
    if return_signal:
        sig = sma(ulcer, int(signal_length)) if signal_type.upper() == "SMA" \
            else ema(ulcer, int(signal_length))
        return ulcer, sig
    return ulcer


def starc(high, low, close, ma_period, atr_period, multiplier):
    high, low, close = _f(high), _f(low), _f(close)
    m = _win_mean(close, int(ma_period))
    atr = _win_mean(true_range(high, low, close), int(atr_period))
    return m + atr * multiplier, m, m - atr * multiplier


def _roll(data, period, fn):
    """Rolling reduction for numpy fallbacks (NaN warm-up)."""
    n = data.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = fn(data[i - period + 1:i + 1])
    return out


def ema_sma(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.ema_sma(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    fv = _first_clean_window(data, period)
    if fv < 0:
        return out
    alpha = 2.0 / (period + 1.0)
    start = fv + period - 1
    prev = data[fv:fv + period].sum() / period
    out[start] = prev
    for i in range(start + 1, n):
        if np.isnan(data[i]):
            continue
        prev = alpha * data[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def rolling_sum(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.rolling_sum(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    s, nans = _roll_nan(data, period)
    out[period - 1:] = np.where(nans == 0, s, np.nan)
    return out


def keltner(high, low, close, ema_period, atr_period, multiplier):
    high, low, close = _f(high), _f(low), _f(close)
    middle = ema_sma(close, int(ema_period))
    atr = atr_wilder(high, low, close, int(atr_period))
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return upper, middle, lower


def donchian(high, low, period):
    high, low = _f(high), _f(low)
    upper = highest(high, int(period))
    lower = lowest(low, int(period))
    return upper, (upper + lower) / 2.0, lower


def chaikin(high, low, ema_period, roc_period):
    high, low = _f(high), _f(low)
    er = ema_sma(high - low, int(ema_period))
    return roc(er, int(roc_period))


def natr(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    atr = atr_wilder(high, low, close, int(period))
    return np.where(close != 0, (atr / close) * 100.0, 0.0)


def rvi_volatility(data, stdev_period, rsi_period):
    """Relative Volatility Index (Donald Dorsey, Stocks & Commodities, 1993), 0 to 100.

    s = population standard deviation of the last `stdev_period` values.
    A bar that closes above the bar before puts s on the up side, a bar that
    closes below puts it on the down side, an unchanged bar on neither.
    U and D are the up and down sides under Wilder's smoothing over
    `rsi_period` bars (the average the RSI uses: seeded with the mean of the
    first `rsi_period` values, then (previous * (n - 1) + new) / n).
    RVI = 100 * U / (U + D), and 50 when there is no volatility on either side.

    The first value is at bar stdev_period + rsi_period - 2. A bar is NaN while
    a NaN is in its deviation window or the bar before it is NaN; the smoothing
    keeps its state across such bars.
    """
    data = _f(data)
    deviation = stdev(data, int(stdev_period))
    move = np.full(data.size, np.nan)
    move[1:] = data[1:] - data[:-1]
    known = ~np.isnan(deviation) & ~np.isnan(move)
    up_side = ema_wilder(np.where(known, np.where(move > 0, deviation, 0.0), np.nan),
                         int(rsi_period))
    down_side = ema_wilder(np.where(known, np.where(move < 0, deviation, 0.0), np.nan),
                           int(rsi_period))
    total = up_side + down_side
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, 100.0 * up_side / np.where(total > 0, total, 1.0), 50.0)
    out[~known | np.isnan(total)] = np.nan
    return out


def ultosc(high, low, close, period1, period2, period3):
    high, low, close = _f(high), _f(low), _f(close)
    if HAVE_RUST:
        return _rs.ultosc(high, low, close, int(period1), int(period2), int(period3))
    n = close.size
    tr = true_range(high, low, close)
    bp = np.empty(n)
    if n:
        bp[0] = close[0] - min(low[0], close[0])
        bp[1:] = close[1:] - np.minimum(low[1:], close[:-1])

    def raw(p):
        bps = rolling_sum(bp, p)
        trs = rolling_sum(tr, p)
        return np.where(trs > 0, bps / np.where(trs == 0, 1.0, trs), 0.0)

    r1, r2, r3 = raw(int(period1)), raw(int(period2)), raw(int(period3))
    out = 100.0 * (4 * r1 + 2 * r2 + r3) / 7.0
    mp = max(int(period1), int(period2), int(period3))
    out[:mp - 1] = np.nan
    return out


def _rolling_extreme(data, period, want_max):
    """Rolling maximum or minimum of `data` over `period` bars.

    NaN for the first period-1 bars and where the window holds a NaN. A period
    of 0, or longer than the series, has no window: every bar is NaN.
    """
    data = _f(data)
    period = _lookback(period, "period")
    out = np.full(data.size, np.nan)
    if period == 0 or period > data.size:
        return out
    if HAVE_RUST:
        return (_rs.highest if want_max else _rs.lowest)(data, period)
    windows = np.lib.stride_tricks.sliding_window_view(data, period)
    out[period - 1:] = windows.max(axis=1) if want_max else windows.min(axis=1)
    return out


def highest(data, period):
    """Rolling maximum (see `_rolling_extreme`)."""
    return _rolling_extreme(data, period, want_max=True)


def lowest(data, period):
    """Rolling minimum (see `_rolling_extreme`)."""
    return _rolling_extreme(data, period, want_max=False)


def supertrend(high, low, close, period, multiplier):
    high, low, close = _f(high), _f(low), _f(close)
    if HAVE_RUST:
        return _rs.supertrend(high, low, close, int(period), float(multiplier))
    # numpy fallback (stateful loop)
    n = close.size
    st = np.full(n, np.nan)
    dr = np.full(n, np.nan)
    atr = atr_wilder(high, low, close, int(period))
    hl = (high + low) / 2.0
    ub = hl + multiplier * atr
    lb = hl - multiplier * atr
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    fv = period - 1
    if fv >= n:
        return st, dr
    fu[fv], fl[fv], dr[fv], st[fv] = ub[fv], lb[fv], 1.0, ub[fv]
    for i in range(fv + 1, n):
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        if st[i - 1] == fu[i - 1]:
            dr[i] = -1.0 if close[i] > fu[i] else 1.0
        else:
            dr[i] = 1.0 if close[i] < fl[i] else -1.0
        st[i] = fl[i] if dr[i] == -1.0 else fu[i]
    return st, dr


def chande_kroll_stop(high, low, close, p, x, q):
    high, low, close = _f(high), _f(low), _f(close)
    if HAVE_RUST:
        return _rs.chande_kroll_stop(high, low, close, int(p), float(x), int(q))
    n = close.size
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    atr = atr_wilder(high, low, close, int(p))
    fhs = np.full(n, np.nan)
    fls = np.full(n, np.nan)
    for i in range(p - 1, n):
        fhs[i] = np.max(high[i - p + 1:i + 1]) - x * atr[i]
        fls[i] = np.min(low[i - p + 1:i + 1]) + x * atr[i]
    for i in range(p + q - 2, n):
        wh = fhs[i - q + 1:i + 1]
        wl = fls[i - q + 1:i + 1]
        vh, vl = wh[~np.isnan(wh)], wl[~np.isnan(wl)]
        if len(vh):
            short_stop[i] = np.max(vh)
        if len(vl):
            long_stop[i] = np.min(vl)
    return long_stop, short_stop


# FRAMA constants (John Ehlers, Fractal Adaptive Moving Average, 2005):
# the EMA weight is exp(_FRAMA_DECAY * (dimension - 1)), never below
# _FRAMA_MIN_ALPHA. The TradingView script this follows then takes a trailing
# mean of the filter over _FRAMA_SMOOTHING bars.
_FRAMA_DECAY = -4.6
_FRAMA_MIN_ALPHA = 0.01
_FRAMA_SMOOTHING = 5


def _window_range(high, low, width):
    """Highest high minus lowest low of each `width`-bar window; element k is bars [k, k+width)."""
    return (np.lib.stride_tricks.sliding_window_view(high, width).max(axis=1)
            - np.lib.stride_tricks.sliding_window_view(low, width).min(axis=1))


def frama(high, low, period):
    """Fractal Adaptive Moving Average of the bar midpoint (high + low) / 2.

    Bars are counted from the first one that has a midpoint. For the first
    `period` bars the filter is the midpoint. After that the fractal dimension
    compares the high-low range of the last `period` bars with the ranges of
    its newer and older halves: D = (ln(N1 + N2) - ln(N3)) / ln(2), each N a
    range divided by its bar count, and D = 1 when a range is not positive.
    The filter is an EMA of the midpoint with weight exp(-4.6 * (D - 1)) held
    between 0.01 and 1. The result is the midpoint for the first period + 1
    bars and the 5-bar mean of the filter from then on. `period` is in bars
    and is meant to be even.

    The filter is NaN while a NaN bar is among the `period` bars of its ranges
    and resumes from its last value afterwards; the result is NaN while a NaN
    filter value is among the five it averages.
    """
    high, low = _f(high), _f(low)
    _require_same_length(high=high, low=low)
    # A period beyond the series never leaves the first-bars rule, like the
    # series length itself.
    period = min(int(period), high.size)
    if HAVE_RUST:
        return _rs.frama(high, low, period)
    out = np.full(high.size, np.nan)
    first = _first_value(high + low)
    high, low = high[first:], low[first:]
    price = (high + low) / 2.0
    bars = price.size
    half = period // 2
    filt = price.copy()
    if half > 0 and bars > period:
        # Element k of a window array covers the bars from k on, so the window
        # ending at bar i with `width` bars is element i - width + 1.
        adaptive = np.arange(period, bars)
        whole = _window_range(high, low, period)[adaptive - period + 1] / period
        newer = _window_range(high, low, half)[adaptive - half + 1] / half
        older = _window_range(high, low, period - half)[adaptive - period + 1] / half
        with np.errstate(invalid="ignore", divide="ignore"):
            defined = (newer > 0) & (older > 0) & (whole > 0)
            dimension = np.where(
                defined,
                (np.log(np.where(defined, newer + older, 1.0))
                 - np.log(np.where(defined, whole, 1.0))) / np.log(2.0),
                1.0,
            )
            alpha = np.fmax(
                np.fmin(np.exp(_FRAMA_DECAY * (dimension - 1.0)), 1.0), _FRAMA_MIN_ALPHA)
        usable = _unknown_in_window(np.isnan(price), period)[adaptive - period + 1] == 0
        weights = alpha.tolist()
        values = price.tolist()
        last = np.nan
        for i in range(bars):
            if i >= period:
                if usable[i - period]:
                    weight = weights[i - period]
                    values[i] = weight * values[i] + (1.0 - weight) * last
                else:
                    values[i] = np.nan
            if values[i] == values[i]:
                last = values[i]
        filt[:] = values
    smoothed = price.copy()
    lag = _FRAMA_SMOOTHING - 1
    start = max(period + 1, lag)
    smoothed[period + 1:start] = filt[period + 1:start]
    if bars > start:
        # Summed oldest bar first (the order the Rust kernel uses). A NaN filter
        # value among the five makes the sum, and so the bar, NaN.
        total = np.zeros(bars - start)
        for offset in range(_FRAMA_SMOOTHING):
            total += filt[start - lag + offset:bars - lag + offset]
        smoothed[start:] = total / _FRAMA_SMOOTHING
    out[first:] = smoothed
    return out


# ---------------------------------------------------------------------------
# Signal helpers behind the public `ta` utilities (crossover, exrem, ...).
#
# Each runs in the Rust core and has a vectorised NumPy fallback that returns
# the same values, so neither backend loops over bars in Python. All of them
# take array-likes and return a NumPy array with one value per bar.
# ---------------------------------------------------------------------------

# Largest n for which valuewhen resolves the n-th most recent true bar.
_VALUEWHEN_LOOKBACK_CAP = 1000


def _lookback(length, name="length"):
    """A bar count as a non-negative int. Raises ValueError for a negative one."""
    length = int(length)
    if length < 0:
        raise ValueError(f"{name} must be zero or positive, got {length}")
    return length


def _aligned(first, second, first_name, second_name):
    """Two series as float64 arrays of the length of the first.

    The first series sets the number of bars. A longer second series is read
    only that far; a shorter one is an IndexError.
    """
    a, b = _f(first), _f(second)
    if b.size < a.size:
        raise IndexError(
            f"{second_name} has {b.size} values but {first_name} has {a.size}: "
            f"{second_name} must be at least as long as {first_name}"
        )
    return a, (b if b.size == a.size else np.ascontiguousarray(b[:a.size]))


def _cross_masks(a, b):
    """(crossed above, crossed below) as boolean arrays, both False on bar 0."""
    over = np.zeros(a.size, dtype=bool)
    under = np.zeros(a.size, dtype=bool)
    # A comparison against NaN is False, so a NaN on the bar or on the bar
    # before blocks both signals without an explicit mask.
    over[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    under[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return over, under


def crossover(series1, series2):
    """Bars where series1 crosses above series2.

    True on bar i when series1[i] > series2[i] and series1[i-1] <= series2[i-1].
    False on bar 0 and wherever one of those four values is NaN.
    """
    a, b = _aligned(series1, series2, "series1", "series2")
    if HAVE_RUST:
        return _rs.crossover(a, b)
    return _cross_masks(a, b)[0]


def crossunder(series1, series2):
    """Bars where series1 crosses below series2.

    True on bar i when series1[i] < series2[i] and series1[i-1] >= series2[i-1].
    False on bar 0 and wherever one of those four values is NaN.
    """
    a, b = _aligned(series1, series2, "series1", "series2")
    if HAVE_RUST:
        return _rs.crossunder(a, b)
    return _cross_masks(a, b)[1]


def cross(series1, series2):
    """Bars where series1 crosses series2 in either direction (crossover or crossunder)."""
    a, b = _aligned(series1, series2, "series1", "series2")
    if HAVE_RUST:
        return _rs.cross(a, b)
    over, under = _cross_masks(a, b)
    return over | under


def rising(data, length):
    """Bars where data[i] > data[i-length]; `length` in bars.

    False for the first `length` bars and wherever either value is NaN.
    """
    data = _f(data)
    length = _lookback(length)
    if HAVE_RUST:
        return _rs.rising(data, length)
    out = np.zeros(data.size, dtype=bool)
    if 0 < length < data.size:
        out[length:] = data[length:] > data[:-length]
    return out


def falling(data, length):
    """Bars where data[i] < data[i-length]; `length` in bars.

    False for the first `length` bars and wherever either value is NaN.
    """
    data = _f(data)
    length = _lookback(length)
    if HAVE_RUST:
        return _rs.falling(data, length)
    out = np.zeros(data.size, dtype=bool)
    if 0 < length < data.size:
        out[length:] = data[length:] < data[:-length]
    return out


def change(data, length=1):
    """data[i] - data[i-length], as float64; NaN for the first `length` bars."""
    data = _f(data)
    length = _lookback(length)
    if HAVE_RUST:
        return _rs.change(data, length)
    if length == 0:
        return data - data
    out = np.full(data.size, np.nan)
    if length < data.size:
        out[length:] = data[length:] - data[:-length]
    return out


def _last_true(mask):
    """Index of the latest True at or before each bar, -1 before the first one."""
    return np.maximum.accumulate(np.where(mask, np.arange(mask.size), -1))


def exrem(primary, secondary):
    """Excess removal: keep the first `primary` signal, drop the repeats.

    primary, secondary: boolean-like, one value per bar; nonzero (NaN included)
    counts as True. Returns a boolean array: True on a `primary` bar while
    armed. A kept signal disarms, a `secondary` bar re-arms. On a bar where
    both are set the signal is kept if armed, otherwise the bar re-arms.
    """
    p, s = _aligned(primary, secondary, "primary", "secondary")
    if HAVE_RUST:
        return _rs.exrem(p, s)
    p, s = p != 0.0, s != 0.0
    # Excess removal is a two-state machine (armed or latched), and a bar
    # applies one of four maps to the state: primary only -> latched, secondary
    # only -> armed, both -> the other state, neither -> unchanged. So the
    # state after a bar is the one set by the latest one-sided bar, switched
    # once for every two-sided bar since then.
    one_sided = p ^ s
    two_sided = p & s
    setter = _last_true(one_sided)
    set_latched = np.where(setter >= 0, p[np.maximum(setter, 0)], False)
    switches = np.cumsum(two_sided)
    switches_since = switches - np.where(setter >= 0, switches[np.maximum(setter, 0)], 0)
    latched_after = set_latched ^ (switches_since % 2 == 1)
    latched_before = np.zeros(p.size, dtype=bool)
    latched_before[1:] = latched_after[:-1]
    return p & ~latched_before


def flip(primary, secondary):
    """Latch: True from a `primary` bar until the next `secondary` bar.

    primary, secondary: boolean-like, one value per bar; nonzero (NaN included)
    counts as True. Returns a boolean array. A bar where both are set counts
    as `primary`.
    """
    p, s = _aligned(primary, secondary, "primary", "secondary")
    if HAVE_RUST:
        return _rs.flip(p, s)
    p, s = p != 0.0, s != 0.0
    latest = _last_true(p | s)
    return np.where(latest >= 0, p[np.maximum(latest, 0)], False)


def valuewhen(expr, array, n=1):
    """`array` at the n-th most recent bar where `expr` was true (n=1: the latest).

    expr: boolean-like condition, one value per bar (nonzero is True).
    Returns float64, NaN until `expr` has been true n times. An n below 1, above
    the series length or above 1000 never resolves: every bar is NaN.
    """
    condition, values = _aligned(expr, array, "expr", "array")
    n = int(n)
    out = np.full(condition.size, np.nan)
    if n < 1 or n > min(_VALUEWHEN_LOOKBACK_CAP, condition.size):
        return out
    if HAVE_RUST:
        return _rs.valuewhen(condition, values, n)
    true_bar = condition != 0.0
    true_bars = np.flatnonzero(true_bar)
    seen = np.cumsum(true_bar)
    resolved = seen >= n
    out[resolved] = values[true_bars[seen[resolved] - n]]
    return out


def ichimoku(high, low, close, conv, base, span2, disp):
    high, low, close = _f(high), _f(low), _f(close)
    n = close.size

    def don(p):
        return (highest(high, p) + lowest(low, p)) / 2.0

    conversion = don(int(conv))
    base_line = don(int(base))
    lead1 = (conversion + base_line) / 2.0
    lead2 = don(int(span2))
    off = disp - 1
    la = np.full(n, np.nan)
    lb = np.full(n, np.nan)
    if 0 < off < n:
        la[off:] = lead1[:-off]
        lb[off:] = lead2[:-off]
    elif off == 0:
        la, lb = lead1.copy(), lead2.copy()
    lag = np.full(n, np.nan)
    offlag = -disp + 1
    if offlag < 0:
        sh = abs(offlag)
        if sh < n:
            lag[:-sh] = close[sh:]
    elif offlag > 0:
        if offlag < n:
            lag[offlag:] = close[:-offlag]
    else:
        lag = close.copy()
    return conversion, base_line, la, lb, lag

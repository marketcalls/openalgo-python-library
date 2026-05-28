# -*- coding: utf-8 -*-
"""
Indicator compute backend.

Single seam between the Python indicator wrappers and the compiled Rust core
(``openalgo._oaindicators``). When the extension is present every kernel runs in
Rust; when it is absent (a source checkout without a built wheel) a pure-NumPy
fallback returns the same values. Neither path depends on numba / llvmlite.

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


def sma(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.sma(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    c = np.cumsum(data)
    out[period - 1] = c[period - 1] / period
    if n > period:
        out[period:] = (c[period:] - c[:-period]) / period
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


def ema(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.ema(data, period)
    n = data.size
    out = np.empty(n)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
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
    c = np.cumsum(data)
    csq = np.cumsum(data * data)
    s = np.empty(n)
    sq = np.empty(n)
    s[period - 1] = c[period - 1]
    sq[period - 1] = csq[period - 1]
    if n > period:
        s[period:] = c[period:] - c[:-period]
        sq[period:] = csq[period:] - csq[:-period]
    mean = s[period - 1:] / period
    var = sq[period - 1:] / period - mean * mean
    out[period - 1:] = np.sqrt(np.maximum(0.0, var))
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
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return tr


def atr_wilder(high, low, close, period):
    high, low, close = _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.atr_wilder(high, low, close, period)
    n = high.size
    atr = np.full(n, np.nan)
    if period <= 0 or n < period:
        return atr
    tr = true_range(high, low, close)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def rsi(data, period):
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.rsi(data, period)
    n = data.size
    out = np.full(n, np.nan)
    if period <= 0 or n < period + 1:
        return out
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def macd(data, fast_period, slow_period, signal_period):
    macd_line = ema(data, fast_period) - ema(data, slow_period)
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


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
    slow_k = sma(fast_k[fk:], smooth_k)
    slow_k = np.concatenate([np.full(fk, np.nan), slow_k])
    slow_d = sma(slow_k[fk + smooth_k - 1:], d_period)
    slow_d = np.concatenate([np.full(fk + smooth_k - 1, np.nan), slow_d])
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
    sma_tp = sma(tp, period)
    for i in range(period - 1, n):
        md = np.abs(tp[i - period + 1:i + 1] - sma_tp[i]).mean()
        out[i] = (tp[i] - sma_tp[i]) / (0.015 * md) if md != 0 else 0.0
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
    pv = data * volume
    spv = np.convolve(pv, np.ones(period), "valid")
    sv = np.convolve(volume, np.ones(period), "valid")
    vals = np.where(sv > 0, spv / np.where(sv == 0, 1, sv), data[period - 1:])
    out[period - 1:] = vals
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
    if period <= 0 or n < period:
        return out
    out[period - 1] = data[:period].sum() / period
    for i in range(period, n):
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


def fisher(data, length):
    data = _f(data)
    if HAVE_RUST:
        return _rs.fisher(data, int(length))
    raise RuntimeError("fisher requires the _oaindicators extension")


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


def _roll(data, period, fn):
    """Rolling reduction for numpy fallbacks (NaN warm-up)."""
    n = data.size
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = fn(data[i - period + 1:i + 1])
    return out


def highest(data, period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.highest(data, int(period))
    return _roll(data, int(period), np.max)


def lowest(data, period):
    data = _f(data)
    if HAVE_RUST:
        return _rs.lowest(data, int(period))
    return _roll(data, int(period), np.min)


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


def frama(high, low, period):
    high, low = _f(high), _f(low)
    if HAVE_RUST:
        return _rs.frama(high, low, int(period))
    # numpy fallback omitted for brevity; rust path is the supported one.
    raise RuntimeError("frama requires the _oaindicators extension")


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

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
    for i in range(fv + 1, n):
        if not np.isnan(data[i]):
            out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def obv(close, volume):
    close, volume = _f(close), _f(volume)
    if HAVE_RUST:
        return _rs.obv(close, volume)
    n = close.size
    out = np.zeros(n)
    sign = np.where(close[1:] < close[:-1], -1.0, 1.0)
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
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    if HAVE_RUST:
        return _rs.cmf(high, low, close, volume, int(period))
    raise RuntimeError("cmf requires the _oaindicators extension")


def mfi(high, low, close, volume, period):
    high, low, close, volume = _f(high), _f(low), _f(close), _f(volume)
    if HAVE_RUST:
        return _rs.mfi(high, low, close, volume, int(period))
    raise RuntimeError("mfi requires the _oaindicators extension")


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
    fv = 0
    for i in range(n):
        if not np.isnan(data[i]):
            fv = i
            break
    s = 0.0
    cnt = 0
    for i in range(fv, min(fv + length, n)):
        if not np.isnan(data[i]):
            s += data[i]
            cnt += 1
    if cnt == length:
        out[fv + length - 1] = s / length
        alpha = 1.0 / length
        for i in range(fv + length, n):
            if not np.isnan(data[i]):
                out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def session_vwap(source, volume, starts):
    source, volume = _f(source), _f(volume)
    starts = np.ascontiguousarray(starts, dtype=np.float64)
    if HAVE_RUST:
        return _rs.session_vwap(source, volume, starts)
    n = source.size
    vwap = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    spv = sv = spv2 = 0.0
    for i in range(n):
        if starts[i] != 0.0 or i == 0:
            spv = sv = spv2 = 0.0
        spv += source[i] * volume[i]
        sv += volume[i]
        spv2 += source[i] * source[i] * volume[i]
        if sv > 0:
            vwap[i] = spv / sv
            sd[i] = np.sqrt(max(0.0, spv2 / sv - vwap[i] ** 2))
        else:
            vwap[i] = source[i]
            sd[i] = 0.0
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


def sar(high, low, acceleration, maximum):
    high, low = _f(high), _f(low)
    if HAVE_RUST:
        return _rs.sar(high, low, float(acceleration), float(maximum))
    raise RuntimeError("sar requires the _oaindicators extension")


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


def rvi_vigor(open_, high, low, close, period):
    open_, high, low, close = _f(open_), _f(high), _f(low), _f(close)
    period = int(period)
    if HAVE_RUST:
        return _rs.rvi_vigor(open_, high, low, close, period)
    sco = swma(close - open_)
    shl = swma(high - low)
    n = close.size
    rvi = np.full(n, np.nan)
    for i in range(period + 2, n):
        ns = ds = 0.0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(sco[j]):
                ns += sco[j]
            if not np.isnan(shl[j]):
                ds += shl[j]
        rvi[i] = ns / ds if ds != 0.0 else 0.0
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
    data = _f(data)
    period = int(period)
    if HAVE_RUST:
        return _rs.cmo(data, period)
    raise RuntimeError("cmo requires the _oaindicators extension")


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
    ulcer = np.sqrt(sma(dd ** 2, int(smooth_length)))
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
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = data[:period].sum() / period
    for i in range(period, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
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
    c = np.cumsum(data)
    out[period - 1] = c[period - 1]
    if n > period:
        out[period:] = c[period:] - c[:-period]
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
    data = _f(data)
    return rsi(stdev(data, int(stdev_period)), int(rsi_period))


def ultosc(high, low, close, period1, period2, period3):
    high, low, close = _f(high), _f(low), _f(close)
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

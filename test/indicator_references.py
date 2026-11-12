"""Plain-Python references for `openalgo.ta` indicators, written from their published
formulas and nothing else.

Each function follows the formula stated in `docs/INDICATOR_CONVENTIONS.md` (and, where
it has one, in the `_backend` docstring of the same indicator) with explicit loops over
explicit windows.
None of them calls `openalgo`, so a comparison with `ta` can fail: a wrong Rust kernel
or a wrong NumPy fallback no longer agrees with them. `test_indicators_fixes.py` runs
every reference against `ta` in CI, and the parity scripts in `benchmark/` use them
where the indicator class no longer keeps a kernel of its own.

The references take complete bars only (no NaN), the case the formulas describe; how
each indicator treats a NaN bar is pinned by hand-worked tests instead. Arrays are
float64 NumPy arrays; periods are in bars. A NaN input raises ValueError.
"""

import math

import numpy as np

# Fisher Transform: value = 0.66 * (position - 0.5) + 0.67 * value[1], held at
# +-0.999 once it passes +-0.99 (the published form of the transform).
FISHER_POSITION_WEIGHT = 0.66
FISHER_MEMORY = 0.67
FISHER_LIMIT = 0.99
FISHER_HELD_AT = 0.999

# FRAMA: weight exp(-4.6 * (D - 1)) held between 0.01 and 1, and a trailing mean of
# the last five filter values (the published FRAMA).
FRAMA_DECAY = -4.6
FRAMA_MIN_WEIGHT = 0.01
FRAMA_SMOOTHING = 5

# The symmetric weighted moving average of the Relative Vigor Index: 1-2-2-1 over four bars.
SWMA_WEIGHTS = (1.0, 2.0, 2.0, 1.0)


def _complete(*series: np.ndarray) -> list[np.ndarray]:
    """The inputs as float64 arrays, refusing a NaN bar (the references assume none)."""
    arrays = [np.asarray(values, dtype=np.float64) for values in series]
    if any(np.isnan(values).any() for values in arrays):
        raise ValueError("the references take complete bars: drop or fill the NaN bars first")
    return arrays


def wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted moving average: weights 1, 2, ..., period, the newest bar weighted most.

    NaN for the first period - 1 bars.
    """
    (data,) = _complete(data)
    out = np.full(data.size, np.nan)
    weights = np.arange(1.0, period + 1.0)
    for i in range(period - 1, data.size):
        window = data[i - period + 1 : i + 1]
        out[i] = float(np.dot(window, weights)) / weights.sum()
    return out


def frama(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Fractal adaptive moving average of the bar midpoint (high + low) / 2.

    For the first `period` bars the filter is the midpoint. On each later bar, with
    `half` = period / 2 bars in each half of the last `period` bars:
    N3 = range of the last `period` bars / period, N1 and N2 = range of the newer and
    of the older half / half, D = (ln(N1 + N2) - ln(N3)) / ln 2 (1 when a range is not
    positive), weight = exp(-4.6 * (D - 1)) held between 0.01 and 1, and
    filter = weight * midpoint + (1 - weight) * previous filter. The result is the
    midpoint for the first period + 1 bars and the mean of the last five filter values
    after that. `period` must be even and at least 4, so every later bar has five.
    """
    high, low = _complete(high, low)
    if period < 4 or period % 2:
        raise ValueError("the FRAMA reference takes an even period of 4 or more")
    midpoint = (high + low) / 2.0
    half = period // 2
    filt = midpoint.copy()
    for i in range(period, midpoint.size):
        whole = (high[i - period + 1 : i + 1].max() - low[i - period + 1 : i + 1].min()) / period
        newer = (high[i - half + 1 : i + 1].max() - low[i - half + 1 : i + 1].min()) / half
        older = (
            high[i - period + 1 : i - half + 1].max() - low[i - period + 1 : i - half + 1].min()
        ) / half
        if newer > 0 and older > 0 and whole > 0:
            dimension = (math.log(newer + older) - math.log(whole)) / math.log(2.0)
        else:
            dimension = 1.0
        weight = min(max(math.exp(FRAMA_DECAY * (dimension - 1.0)), FRAMA_MIN_WEIGHT), 1.0)
        filt[i] = weight * midpoint[i] + (1.0 - weight) * filt[i - 1]
    out = midpoint.copy()
    for i in range(period + 1, midpoint.size):
        out[i] = sum(filt[i - FRAMA_SMOOTHING + 1 : i + 1]) / FRAMA_SMOOTHING
    return out


def fisher(high: np.ndarray, low: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Fisher Transform of the midpoint (high + low) / 2: returns (fisher, trigger).

    With `position` = (price - lowest) / (highest - lowest) over the last `length` bars:
    value = 0.66 * (position - 0.5) + 0.67 * value[1], held at +-0.999 once it passes
    +-0.99, and fisher = 0.5 * ln((1 + value) / (1 - value)) + 0.5 * fisher[1]; value
    and fisher start at 0. A window without range leaves both unchanged. The trigger is
    the fisher value before the bar (0 at the start). NaN for the first length - 1 bars.
    """
    high, low = _complete(high, low)
    price = (high + low) / 2.0
    fish = np.full(price.size, np.nan)
    trigger = np.full(price.size, np.nan)
    value = previous = 0.0
    for i in range(length - 1, price.size):
        window = price[i - length + 1 : i + 1]
        trigger[i] = previous
        if window.max() != window.min():
            position = (price[i] - window.min()) / (window.max() - window.min())
            value = FISHER_POSITION_WEIGHT * (position - 0.5) + FISHER_MEMORY * value
            if abs(value) > FISHER_LIMIT:
                value = math.copysign(FISHER_HELD_AT, value)
            previous = 0.5 * math.log((1.0 + value) / (1.0 - value)) + 0.5 * previous
        fish[i] = previous
    return fish, trigger


def cmo(data: np.ndarray, period: int) -> np.ndarray:
    """Chande Momentum Oscillator: 100 * (up - down) / (up + down).

    `up` and `down` sum the rises and the falls among the last `period` bar-to-bar
    changes; 0 when nothing moved. NaN for the first `period` bars.
    """
    (data,) = _complete(data)
    out = np.full(data.size, np.nan)
    for i in range(period, data.size):
        changes = [data[j] - data[j - 1] for j in range(i - period + 1, i + 1)]
        up = sum(change for change in changes if change > 0)
        down = -sum(change for change in changes if change < 0)
        out[i] = 0.0 if up + down == 0 else 100.0 * (up - down) / (up + down)
    return out


def cmf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int
) -> np.ndarray:
    """Chaikin Money Flow: sum(money flow volume) / sum(volume) over `period` bars.

    Money flow volume = ((close - low) - (high - close)) / (high - low) * volume, 0 on a
    bar without range; the result is 0 when the window's volume is not positive. NaN
    for the first period - 1 bars.
    """
    high, low, close, volume = _complete(high, low, close, volume)
    flow = [
        0.0 if h == lo else ((c - lo) - (h - c)) / (h - lo) * v
        for h, lo, c, v in zip(high, low, close, volume, strict=True)
    ]
    out = np.full(close.size, np.nan)
    for i in range(period - 1, close.size):
        window_volume = volume[i - period + 1 : i + 1].sum()
        window_flow = sum(flow[i - period + 1 : i + 1])
        out[i] = window_flow / window_volume if window_volume > 0 else 0.0
    return out


def mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int
) -> np.ndarray:
    """Money Flow Index: 100 - 100 / (1 + positive flow / negative flow), 0 to 100.

    A bar's flow is typical price (high + low + close) / 3 times volume: positive when
    the typical price rose from the bar before, negative when it fell, neither on an
    unchanged bar or on bar 0. Flows are summed over `period` bars; 100 when the window
    has no negative flow. NaN for bar 0 and the first period - 1 bars.
    """
    high, low, close, volume = _complete(high, low, close, volume)
    typical = (high + low + close) / 3.0
    out = np.full(close.size, np.nan)
    for i in range(max(1, period - 1), close.size):
        rising = falling = 0.0
        for j in range(max(1, i - period + 1), i + 1):
            if typical[j] > typical[j - 1]:
                rising += typical[j] * volume[j]
            elif typical[j] < typical[j - 1]:
                falling += typical[j] * volume[j]
        out[i] = 100.0 if falling == 0 else 100.0 - 100.0 / (1.0 + rising / falling)
    return out


def _swma(values: np.ndarray) -> np.ndarray:
    """1-2-2-1 weighted mean of the last four bars, NaN for the first three."""
    out = np.full(values.size, np.nan)
    span = len(SWMA_WEIGHTS)
    for i in range(span - 1, values.size):
        out[i] = float(np.dot(values[i - span + 1 : i + 1], SWMA_WEIGHTS)) / sum(SWMA_WEIGHTS)
    return out


def rvi(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Relative Vigor Index: returns (rvi, signal).

    rvi = sum(swma(close - open)) / sum(swma(high - low)) over `period` bars, where swma
    is the 1-2-2-1 weighted mean of the last four bars; signal = swma(rvi).
    """
    open_, high, low, close = _complete(open_, high, low, close)
    body, bar_range = _swma(close - open_), _swma(high - low)
    out = np.full(close.size, np.nan)
    for i in range(len(SWMA_WEIGHTS) + period - 2, close.size):
        out[i] = body[i - period + 1 : i + 1].sum() / bar_range[i - period + 1 : i + 1].sum()
    signal = np.full(close.size, np.nan)
    first = len(SWMA_WEIGHTS) + period - 2
    signal[first:] = _swma(out[first:])
    return out, signal


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume-weighted average of the typical price (high + low + close) / 3 from the
    first bar on: sum(price * volume) / sum(volume), NaN while no volume has traded."""
    high, low, close, volume = _complete(high, low, close, volume)
    typical = (high + low + close) / 3.0
    out = np.full(close.size, np.nan)
    traded = weighted = 0.0
    for i in range(close.size):
        traded += volume[i]
        weighted += typical[i] * volume[i]
        if traded > 0:
            out[i] = weighted / traded
    return out


def sar(
    high: np.ndarray, low: np.ndarray, acceleration: float, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic SAR: returns (stop, trend), trend +1 while long and -1 while short.

    Bar 0 starts long with stop = low and extreme point EP = high. On every later bar:
    the stop moves by AF * (EP - stop); a long position reverses when the stop is
    above the low (to short, stop = max(high, EP), EP = low) and a short one when it
    is below the high (to long, stop = min(low, EP), EP = high), with AF reset to
    `acceleration`; without a reversal a new extreme becomes EP and raises AF by
    `acceleration` up to `maximum`. Last, the stop is held at or below the lows of the
    two previous bars when long, at or above their highs when short.
    """
    high, low = _complete(high, low)
    stops = np.full(high.size, np.nan)
    trends = np.full(high.size, np.nan)
    long, stop, extreme, factor = True, low[0], high[0], acceleration
    stops[0], trends[0] = stop, 1.0
    for i in range(1, high.size):
        stop += factor * (extreme - stop)
        if long and stop > low[i]:
            long, stop, extreme, factor = False, max(high[i], extreme), low[i], acceleration
        elif not long and stop < high[i]:
            long, stop, extreme, factor = True, min(low[i], extreme), high[i], acceleration
        elif (long and high[i] > extreme) or (not long and low[i] < extreme):
            extreme = high[i] if long else low[i]
            factor = min(factor + acceleration, maximum)
        previous = slice(max(0, i - 2), i)
        stop = min(stop, low[previous].min()) if long else max(stop, high[previous].max())
        stops[i], trends[i] = stop, 1.0 if long else -1.0
    return stops, trends

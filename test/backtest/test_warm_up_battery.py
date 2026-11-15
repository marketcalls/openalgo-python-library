"""The warm-up each common indicator measures on the engine's reference bars
(`_warm_up_measure`) holds on prices of other shapes: on every series of a fixed battery,
the indicator computed on only its last `need` bars ends at its full-history value
within the window tolerance.

The battery: eight daily series of 4,000 bars from seeded draws (NumPy's default
generator), each a shape that makes some indicator slow to settle: a steady rise, a
steady fall, a tight range, a volatile walk (3 percent a bar, with opening gaps), trends
that reverse every 300 bars, a walk with jumps of 4 to 10 percent, a cycle, and regimes
of trend with clustered volatility. "Within the tolerance" is the rule's own: the
difference at most 1e-6 of the larger value, or of a thousandth of the output's median
magnitude when that is larger (a value near zero), at 50 bars spread over each series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openalgo import ta
from openalgo.backtest._warm_up_measure import (
    NEAR_ZERO_FRACTION,
    WINDOW_TOLERANCE,
    ReferenceCall,
    ReferenceNeeds,
)

BARS = 4_000
POINTS = 50


def bars_of(closes: np.ndarray, rng: np.random.Generator, gap: float = 0.002,
            spread: float = 0.006) -> dict:  # fmt: skip
    """OHLCV around `closes`: opens moved from the previous close by `gap`, a range of
    about `spread`, volume 1 to 3 lakh units; times one a day."""
    opens = np.concatenate(([closes[0]], closes[:-1])) * np.exp(rng.normal(0, gap, len(closes)))
    span = np.abs(rng.normal(0, spread, len(closes))) * closes
    times = pd.date_range("1990-01-01 09:15", periods=len(closes), freq="D", tz="Asia/Kolkata")
    return {
        "open": opens,
        "high": np.maximum(opens, closes) + span / 2,
        "low": np.minimum(opens, closes) - span / 2,
        "close": closes,
        "volume": rng.uniform(1e5, 3e5, len(closes)),
        "times": times,
    }


def walk(seed: int, drift, vol: float, start: float = 100.0, **shape) -> dict:
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(drift + rng.normal(0, vol, BARS)))
    return bars_of(closes, rng, **shape)


def tight_range(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    level = np.zeros(BARS)
    for k in range(1, BARS):
        level[k] = 0.97 * level[k - 1] + rng.normal(0, 0.004)
    return bars_of(500 * np.exp(level), rng)


def jumps(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sizes = rng.choice([-1, 1], BARS) * rng.uniform(0.04, 0.1, BARS)
    jumped = np.where(rng.uniform(size=BARS) < 1 / 30, sizes, 0.0)
    return bars_of(150 * np.exp(np.cumsum(rng.normal(0, 0.012, BARS) + jumped)), rng)


def cycle(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    closes = 100 + 20 * np.sin(np.arange(BARS) / 15.0) + rng.normal(0, 0.5, BARS)
    return bars_of(closes, rng)


def regimes(seed: int) -> dict:
    """Trends of 50 to 600 bars (drift up to 0.4 percent a bar either way) with volatility
    that clusters (each bar's variance follows the last bar's move)."""
    rng = np.random.default_rng(seed)
    drift = np.empty(BARS)
    k = 0
    while k < BARS:
        span = int(rng.integers(50, 600))
        drift[k : k + span] = rng.uniform(-0.004, 0.004)
        k += span
    shocks, moves, vol = rng.standard_normal(BARS), np.empty(BARS), 0.015
    for k in range(BARS):
        last = vol * shocks[k - 1] if k else 0.0
        vol = np.sqrt(0.000004 + 0.08 * last**2 + 0.9 * vol**2)
        moves[k] = drift[k] + vol * shocks[k]
    return bars_of(100 * np.exp(np.cumsum(moves)), rng, gap=0.005)


BATTERY = {
    "rise": walk(1, 0.0008, 0.008),
    "fall": walk(2, -0.0006, 0.008, start=1000.0),
    "range": tight_range(3),
    "volatile": walk(4, 0.0, 0.03, start=300.0, gap=0.01, spread=0.03),
    "reversals": walk(5, np.where((np.arange(BARS) // 300) % 2 == 0, 0.004, -0.004), 0.01),
    "jumps": jumps(6),
    "cycle": cycle(7),
    "regimes": regimes(11),
}


def fields(bars, *names):
    """The named fields of `bars` as pandas Series by bar time."""
    times = bars["times"]
    return [pd.Series(np.asarray(bars[name], dtype=float), index=times) for name in names]


def on_close(function, *arguments):
    return lambda bars: function(*fields(bars, "close"), *arguments)


def on_prices(function, *arguments, names=("high", "low", "close")):
    return lambda bars: function(*fields(bars, *names), *arguments)


def ichimoku_causal(bars):
    """The causal outputs of ta.ichimoku: its lagging span looks ahead."""
    return ta.ichimoku(*fields(bars, "high", "low", "close"))[:4]


COMMON = {
    "ema 20": on_close(ta.ema, 20),
    "ema 50": on_close(ta.ema, 50),
    "ema 200": on_close(ta.ema, 200),
    "sma 50": on_close(ta.sma, 50),
    "rsi 14": on_close(ta.rsi, 14),
    "macd 12/26/9": on_close(ta.macd, 12, 26, 9),
    "supertrend 10/3": on_prices(ta.supertrend, 10, 3),
    "psar": on_prices(ta.psar, names=("high", "low")),
    "atr 14": on_prices(ta.atr, 14),
    "adx 14": on_prices(ta.adx),
    "bbands 20": on_close(ta.bbands),
    "stochastic": on_prices(ta.stochastic),
    "kama": on_close(ta.kama),
    "dema 20": on_close(ta.dema, 20),
    "tsi": on_close(ta.tsi),
    "mfi 14": on_prices(ta.mfi, names=("high", "low", "close", "volume")),
    "ichimoku": ichimoku_causal,
}


def outputs(returned) -> list[np.ndarray]:
    parts = returned if isinstance(returned, tuple | list) else (returned,)
    return [np.asarray(part, dtype=float) for part in parts]


def within_tolerance(window: float, whole: float, floor: float) -> bool:
    if np.isnan(window) and np.isnan(whole):
        return True
    scale = max(abs(window), abs(whole), floor)
    return abs(window - whole) <= WINDOW_TOLERANCE * scale


@pytest.fixture(scope="module")
def needs() -> ReferenceNeeds:
    """One set of reference bars for the module (they take a few milliseconds to make)."""
    return ReferenceNeeds()


@pytest.mark.parametrize("label", list(COMMON))
def test_the_measured_need_reproduces_the_full_history_values_on_every_shape(label, needs):
    compute = COMMON[label]
    need = needs.need(label, ReferenceCall(lambda bars: outputs(compute(bars)))).bars
    assert need is not None and need < BARS
    misses = []
    for shape, bars in BATTERY.items():
        whole = outputs(compute(bars))
        floors = [NEAR_ZERO_FRACTION * np.nanmedian(np.abs(values)) for values in whole]
        for bar in np.linspace(need - 1, BARS - 1, POINTS).astype(int):
            window = {name: values[bar - need + 1 : bar + 1] for name, values in bars.items()}
            for part, values, floor in zip(outputs(compute(window)), whole, floors, strict=True):
                if not within_tolerance(part[-1], values[bar], floor):
                    misses.append((shape, int(bar)))
    assert misses == []

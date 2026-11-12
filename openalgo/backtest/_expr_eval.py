"""Evaluating condition expressions (`_expr`) once per run, over every strategy bar of
the strategy's instrument, into arrays the Rust signal driver consumes (14-strategy-api
4.3, Phases 1 to 3).

Each node becomes one NumPy array with a value per strategy bar, the value at that
bar's close. Crossings and rolling extremes run through `openalgo.ta` (`crossover`,
`crossunder`, `highest`, `lowest`), and indicators are `openalgo.ta` functions, so a
condition and the same logic written with `ta` in an `on_bar` or a `from_signals`
function give the same values. Comparisons with NaN (an indicator still warming up) are
false. The whole series is visible here, but no operation reads a later bar: the value
at bar t depends only on bars up to t.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators import ta

from . import _expr as x
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ._series import FIELDS, ist_time
from .errors import ConfigError

SECONDS_PER_DAY = 86_400
NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
# 1970-01-01, day 0 of the epoch, was a Thursday (Monday is 0).
EPOCH_WEEKDAY = 3
ARITHMETIC = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
COMPARISON = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}
LOGICAL = {"&": np.logical_and, "|": np.logical_or}


@dataclass(frozen=True)
class StrategyBars:
    """Every strategy bar of one instrument over the run: `ts_open` and `ts_close` as
    int64 UTC nanoseconds, prices in rupees, volume in units."""

    symbol: str
    ts_open: np.ndarray
    ts_close: np.ndarray
    columns: dict[str, np.ndarray]

    @classmethod
    def from_native(cls, symbol: str, native: dict[str, np.ndarray]) -> StrategyBars:
        """From the engine's `strategy_bars` dict."""
        columns = {name: np.asarray(native[name], dtype=np.float64) for name in FIELDS}
        ts_open = np.asarray(native["ts_open"], dtype=np.int64)
        return cls(symbol, ts_open, np.asarray(native["ts_close"], dtype=np.int64), columns)

    def __len__(self) -> int:
        return len(self.ts_open)

    def head(self, count: int) -> StrategyBars:
        """The first `count` bars."""
        columns = {name: values[:count] for name, values in self.columns.items()}
        return StrategyBars(self.symbol, self.ts_open[:count], self.ts_close[:count], columns)

    def frame(self) -> pd.DataFrame:
        """The bars as a DataFrame indexed by IST bar open (as `Strategy.history`)."""
        index = pd.DatetimeIndex([ist_time(int(ts)) for ts in self.ts_open], name="time")
        return pd.DataFrame({name: self.columns[name] for name in FIELDS}, index=index)


def condition_values(condition: Any, bars: StrategyBars, what: str) -> np.ndarray:
    """A condition as a bool array with one value per strategy bar. `what` names it in
    errors ("entry_when"). Raises ConfigError for anything outside the grammar, a
    condition that is a number, or arrays that do not cover the bars."""
    expr = x.typed(condition, x.BOOL, what)
    return np.ascontiguousarray(evaluate(expr, bars, {}), dtype=bool)


def evaluate(expr: x.Expr, bars: StrategyBars, cache: dict[int, Any]) -> np.ndarray:
    """The values of a node over the bars. `cache` holds indicator results by call, so a
    function with several outputs runs once."""
    handler = HANDLERS.get(type(expr))
    if handler is None:
        raise x.outside(f"a {type(expr).__name__} node")
    return handler(expr, bars, cache)


def _column(node: x.Column, bars: StrategyBars, cache: dict) -> np.ndarray:
    if node.symbol != bars.symbol:
        raise ConfigError(
            f"a condition strategy reads its own symbol {bars.symbol}; {node.symbol} is "
            "another one: use on_bar for several symbols"
        )
    return bars.columns[node.field].copy()


def _constant(node: x.Constant, bars: StrategyBars, cache: dict) -> np.ndarray:
    return np.full(len(bars), node.value)


def _values(node: x.Values, bars: StrategyBars, cache: dict) -> np.ndarray:
    if node.times is None:
        if len(node.values) != len(bars):
            raise ConfigError(
                f"an array has {len(node.values)} values and the run has {len(bars)} "
                "strategy bars: give one value per strategy bar, or a Series indexed by "
                "bar open time"
            )
        return node.values
    index = pd.Index(node.times)
    if not index.is_unique:
        raise ConfigError("a condition Series has the same bar open time more than once")
    rows = index.get_indexer(bars.ts_open)
    if (rows < 0).any():
        first = int(bars.ts_open[int(np.argmax(rows < 0))])
        raise ConfigError(
            f"a condition Series has no value for the strategy bar opening at "
            f"{ist_time(first).isoformat()}: index it by the run's strategy-bar open times"
        )
    return node.values[rows]


def _indicator(node: x.Indicator, bars: StrategyBars, cache: dict) -> np.ndarray:
    from openalgo.indicators import causality

    call = node.call
    position = 0 if node.output is None else node.output
    name = getattr(call.func, "__name__", repr(call.func))
    if not causality.is_causal(call.func, position):
        raise ConfigError(
            f"ta.{name} output {causality.outputs(call.func)[position]!r} looks ahead "
            "(openalgo.indicators.causality): a strategy cannot use it"
        )
    if id(call) not in cache:
        args = [evaluate(a, bars, cache) if isinstance(a, x.Expr) else a for a in call.args]
        cache[id(call)] = call.func(*args, **call.params)
    result = cache[id(call)]
    if isinstance(result, tuple) != (node.output is not None):
        raise ConfigError(f"ta.{name} returned {type(result).__name__}, not the outputs expected")
    if node.output is not None:
        if node.output >= len(result):
            raise ConfigError(f"ta.{name} returned {len(result)} outputs, no output {position}")
        result = result[node.output]
    values = np.asarray(result, dtype=np.float64)
    if values.shape != (len(bars),):
        raise ConfigError(f"ta.{name} returned {values.shape} values for {len(bars)} bars")
    return values


def _arithmetic(node: x.Arithmetic, bars: StrategyBars, cache: dict) -> np.ndarray:
    left, right = evaluate(node.left, bars, cache), evaluate(node.right, bars, cache)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ARITHMETIC[node.op](left, right)


def _comparison(node: x.Comparison, bars: StrategyBars, cache: dict) -> np.ndarray:
    left, right = evaluate(node.left, bars, cache), evaluate(node.right, bars, cache)
    with np.errstate(invalid="ignore"):
        return COMPARISON[node.op](left, right)


def _logical(node: x.Logical, bars: StrategyBars, cache: dict) -> np.ndarray:
    return LOGICAL[node.op](evaluate(node.left, bars, cache), evaluate(node.right, bars, cache))


def _negation(node: x.Negation, bars: StrategyBars, cache: dict) -> np.ndarray:
    return np.logical_not(evaluate(node.operand, bars, cache))


def _crossing(node: x.Crossing, bars: StrategyBars, cache: dict) -> np.ndarray:
    left, right = evaluate(node.left, bars, cache), evaluate(node.right, bars, cache)
    cross = ta.crossover if node.direction == "above" else ta.crossunder
    return np.asarray(cross(left, right), dtype=bool)


def _shifted(node: x.Shifted, bars: StrategyBars, cache: dict) -> np.ndarray:
    values = evaluate(node.operand, bars, cache)
    filler = False if node.kind == x.BOOL else np.nan
    shifted = np.full(len(values), filler, dtype=values.dtype)
    if node.bars < len(values):
        shifted[node.bars :] = values[: len(values) - node.bars]
    return shifted


def _rolling(node: x.RollingExtreme, bars: StrategyBars, cache: dict) -> np.ndarray:
    values = evaluate(node.operand, bars, cache)
    extreme = ta.highest if node.extreme == "max" else ta.lowest
    return np.asarray(extreme(values, node.bars), dtype=np.float64)


def _ist_seconds(ts_close: np.ndarray) -> np.ndarray:
    """Seconds since IST midnight of each instant."""
    return ((ts_close + IST_OFFSET_NANOS) % NANOS_PER_DAY) // NANOS_PER_SECOND


def _time_between(node: x.TimeBetween, bars: StrategyBars, cache: dict) -> np.ndarray:
    seconds = _ist_seconds(bars.ts_close)
    return (seconds >= node.start) & (seconds < node.end)


def _weekday(node: x.Weekday, bars: StrategyBars, cache: dict) -> np.ndarray:
    days = (bars.ts_close + IST_OFFSET_NANOS) // NANOS_PER_DAY
    return np.isin((days + EPOCH_WEEKDAY) % 7, sorted(node.days))


HANDLERS: dict[type, Callable[[Any, StrategyBars, dict], np.ndarray]] = {
    x.Column: _column,
    x.Constant: _constant,
    x.Values: _values,
    x.Indicator: _indicator,
    x.Arithmetic: _arithmetic,
    x.Comparison: _comparison,
    x.Logical: _logical,
    x.Negation: _negation,
    x.Crossing: _crossing,
    x.Shifted: _shifted,
    x.RollingExtreme: _rolling,
    x.TimeBetween: _time_between,
    x.Weekday: _weekday,
}

__all__ = ["StrategyBars", "condition_values", "evaluate"]

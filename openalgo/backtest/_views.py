"""Values aligned to bars that are revealed as they close: what series views, indicators
and the expressions built from them have in common (14-strategy-api 2.2).

A view holds the values of the whole run, computed ahead, and shows only those of the
bars its store has revealed: `[-1]` is the latest closed bar, slices and `len()` work,
`.values` is a NumPy copy of what is visible. Arithmetic (`+ - * /`, unary minus,
`abs`) and comparisons (`< <= > >= == !=`) with a number or another view of the same
store give a new view; so do `.crossed_above(other)` and `.crossed_below(other)`, which
use `openalgo.ta.crossover` and `crossunder`. Views of booleans combine with `&`, `|`
and `~`; `.shift(n)`, `.rolling_max(n)` and `.rolling_min(n)` look back `n` bars. These
make the condition grammar of 14-strategy-api 4.2 (`_conditions`). A view of booleans
is true when its latest value is (false before any bar); a numeric view has no truth
value, so `if view:` is an error that points to `view[-1]`.

While the strategy's `init` declares it, no view has a truth value: `and`, `or`, `not`,
`if` and a chained comparison such as `a < b < c` need one true or false, a condition
over the whole run has none, and reading "false, no bar has closed" would quietly drop
part of a condition (`a and b` would become `a`). A truth test then raises ConfigError
saying to combine conditions with `&`, `|` and `~`. The run context marks the phase on
its stores (`declaring`, see `Store`), so the rule needs no state of its own.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .errors import ConfigError


class Store(Protocol):
    """What a view needs from the bars it is aligned to.

    A store of a run also carries `declaring`, true while the strategy's `init` runs
    (`_context.RunContext` sets it on every store it makes); a store without it, such as
    one built outside a run, is never declaring."""

    symbol: str
    interval: str

    def visible(self) -> int:
        """How many bars are revealed now."""
        ...

    def times(self) -> pd.DatetimeIndex:
        """IST open of every bar of the run."""
        ...


def _truth_in_init(label: str) -> str:
    """The error for a truth test of a view while `init` declares the strategy."""
    return (
        f"{label} was tested for truth in init: and, or, not, if and a chained comparison "
        "such as a < b < c need one true or false, and a condition over the whole run has "
        "none, so part of the condition would be lost: combine conditions with & (and), "
        "| (or) and ~ (not), each comparison in brackets: "
        "(fast > slow) & (close > ema); write a < b < c as (a < b) & (b < c)"
    )


def _bar_count(value: Any, what: str, minimum: int = 1) -> int:
    """A whole number of bars, `minimum` or more. Raises ConfigError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        later = " (a negative shift would read a later bar)" if what == "shift" else ""
        raise ConfigError(f"{what} takes a whole number of bars, {minimum} or more{later}")
    return int(value)


class TimelineView:
    """Values aligned to a store's bars; see the module docstring."""

    # Views compare element-wise, so they are not hashable.
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, store: Store, values: np.ndarray, label: str) -> None:
        self._store = store
        self._values = values
        self._label = label

    @property
    def label(self) -> str:
        """How the view is named in messages, for example `ema(SBIN.close, 20)`."""
        return self._label

    def _shown(self) -> np.ndarray:
        return self._values[: self._store.visible()]

    def __len__(self) -> int:
        return len(self._shown())

    def __getitem__(self, key: int | slice) -> Any:
        shown = self._shown()
        if isinstance(key, slice):
            return shown[key].copy()
        try:
            return shown[key].item()
        except IndexError:
            raise IndexError(
                f"{self._label}: bar {key} is not available, {len(shown)} bars have closed"
            ) from None

    @property
    def values(self) -> np.ndarray:
        """A copy of the values revealed so far."""
        return self._shown().copy()

    # ----- building new views ----------------------------------------------------

    def _operand(self, other: Any) -> Any:
        if isinstance(other, TimelineView):
            if other._store is not self._store:
                raise ConfigError(
                    f"{self._label} and {other._label} belong to different series "
                    f"({self._store.symbol} {self._store.interval} and {other._store.symbol} "
                    f"{other._store.interval}): combine series of one symbol and interval"
                )
            return other._values
        if isinstance(other, bool | int | float | np.number):
            return other
        return NotImplemented

    def _derive(self, other: Any, function: Callable, symbol: str, flip: bool = False) -> Any:
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented
        left, right = (operand, self._values) if flip else (self._values, operand)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = function(left, right)
        name = other.label if isinstance(other, TimelineView) else repr(other)
        names = (name, self._label) if flip else (self._label, name)
        return TimelineView(self._store, np.asarray(values), f"({names[0]} {symbol} {names[1]})")

    def __add__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.add, "+")

    def __radd__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.add, "+", flip=True)

    def __sub__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.sub, "-")

    def __rsub__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.sub, "-", flip=True)

    def __mul__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.mul, "*")

    def __rmul__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.mul, "*", flip=True)

    def __truediv__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.truediv, "/")

    def __rtruediv__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.truediv, "/", flip=True)

    def __neg__(self) -> TimelineView:
        return TimelineView(self._store, -self._values, f"-{self._label}")

    def __abs__(self) -> TimelineView:
        return TimelineView(self._store, np.abs(self._values), f"abs({self._label})")

    def __lt__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.lt, "<")

    def __le__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.le, "<=")

    def __gt__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.gt, ">")

    def __ge__(self, other: Any) -> TimelineView:
        return self._derive(other, operator.ge, ">=")

    def __eq__(self, other: Any) -> TimelineView:  # type: ignore[override]
        return self._derive(other, operator.eq, "==")

    def __ne__(self, other: Any) -> TimelineView:  # type: ignore[override]
        return self._derive(other, operator.ne, "!=")

    def crossed_above(self, other: Any) -> TimelineView:
        """True on a bar where this view moves above `other` (a number or a view of the
        same store): above on that bar, at or below on the bar before."""
        return self._cross(other, "crossover", "crossed_above")

    def crossed_below(self, other: Any) -> TimelineView:
        """True on a bar where this view moves below `other`."""
        return self._cross(other, "crossunder", "crossed_below")

    def _cross(self, other: Any, function: str, name: str) -> TimelineView:
        from openalgo import ta

        operand = self._operand(other)
        if operand is NotImplemented:
            raise ConfigError(f"{name} takes a number or a series view, got {type(other).__name__}")
        line = np.broadcast_to(np.asarray(operand, dtype=float), self._values.shape)
        values = np.asarray(getattr(ta, function)(self._values, np.ascontiguousarray(line)))
        label = other.label if isinstance(other, TimelineView) else repr(other)
        return TimelineView(self._store, values.astype(bool), f"{self._label}.{name}({label})")

    # ----- combining conditions, and the windows of 14-strategy-api 4.2 ----------------

    def _logical(self, other: Any, function: Callable, symbol: str, flip: bool = False) -> Any:
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented
        for values in (self._values, np.asarray(operand)):
            if values.dtype != np.bool_:
                raise ConfigError(
                    f"{symbol} combines conditions (true or false), and {self._label} {symbol} "
                    f"{other!r} has numbers: compare them first, for example view > 0"
                )
        return self._derive(other, function, symbol, flip)

    def __and__(self, other: Any) -> TimelineView:
        return self._logical(other, np.logical_and, "&")

    def __rand__(self, other: Any) -> TimelineView:
        return self._logical(other, np.logical_and, "&", flip=True)

    def __or__(self, other: Any) -> TimelineView:
        return self._logical(other, np.logical_or, "|")

    def __ror__(self, other: Any) -> TimelineView:
        return self._logical(other, np.logical_or, "|", flip=True)

    def __invert__(self) -> TimelineView:
        if self._values.dtype != np.bool_:
            raise ConfigError(f"~ negates a condition; {self._label} is a series of numbers")
        return TimelineView(self._store, ~self._values, f"~{self._label}")

    def shift(self, bars: int) -> TimelineView:
        """The value `bars` bars earlier (0 or more): NaN for numbers and false for a
        condition before the first bar. A negative shift would read a later bar."""
        count = _bar_count(bars, "shift", minimum=0)
        condition = self._values.dtype == np.bool_
        values = np.full(len(self._values), False if condition else np.nan)
        if count < len(values):
            values[count:] = self._values[: len(values) - count]
        return TimelineView(self._store, values, f"{self._label}.shift({count})")

    def rolling_max(self, bars: int) -> TimelineView:
        """The highest of the last `bars` values, this bar's included (`ta.highest`);
        NaN until `bars` bars have closed."""
        return self._window(bars, "highest", "rolling_max")

    def rolling_min(self, bars: int) -> TimelineView:
        """The lowest of the last `bars` values, this bar's included (`ta.lowest`)."""
        return self._window(bars, "lowest", "rolling_min")

    def _window(self, bars: int, function: str, name: str) -> TimelineView:
        from openalgo import ta

        count = _bar_count(bars, name)
        if self._values.dtype == np.bool_:
            raise ConfigError(f"{name} takes a series of numbers; {self._label} is a condition")
        source = np.ascontiguousarray(self._values, dtype=float)
        values = np.asarray(getattr(ta, function)(source, count), dtype=float)
        return TimelineView(self._store, values, f"{self._label}.{name}({count})")

    def __bool__(self) -> bool:
        if getattr(self._store, "declaring", False):
            raise ConfigError(_truth_in_init(self._label))
        if self._values.dtype != np.bool_:
            raise TypeError(
                f"{self._label} is a series of numbers: use {self._label}[-1] for the latest value"
            )
        shown = self._shown()
        return bool(shown[-1]) if len(shown) else False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._label}, {len(self)} bars)"


__all__ = ["Store", "TimelineView"]

"""The condition grammar of 14-strategy-api 4.2: what `entry_when`, `exit_when`,
`short_when` and `cover_when` accept.

    fast, slow = indicator(ta.ema, self.close, 10), indicator(ta.ema, self.close, 30)
    cond = fast.crossed_above(slow) & (self.close > self.open) & bt.time_between("09:30", "14:30")
    self.entry_when(cond, stop=bt.pct(1))

A condition is an expression tree, built here and evaluated once per run over the
strategy bars by `_expr_eval`. Operands are series views (`self.close`,
`self.data[s].high`), `openalgo.ta` indicators, numbers and parameters, and arrays with
one value per strategy bar (a NumPy array or a list in bar order, or a pandas Series
indexed by bar open time; SPEC-29 reads every array input the same way). Operations:
`+ - * /`, the comparisons `< <= > >=`, `&`, `|`, `~`, `.crossed_above(x)`,
`.crossed_below(x)`, `.shift(n)`, `.rolling_max(n)`, `.rolling_min(n)`, and the clock
conditions `bt.time_between` and `bt.weekday`. Every value is the value at a bar's close
computed from bars up to that close: no operation reads a later bar.

Anything else raises ConfigError saying to write the logic in `on_bar`; that includes
`==` and `!=` (prices are compared with `< <= > >=`), `and`, `or`, `not` and `if` (use
`&`, `|`, `~`), and `True` or `False` as an operand. An object takes part in conditions
by implementing `__condition_expr__()` returning the `Expr` it stands for (series views
do, and so should the handles `self.I` returns).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ._calendar_convert import IST, to_seconds
from .errors import ConfigError

NUMBER, BOOL = "number", "bool"
GRAMMAR = "the condition grammar of 14-strategy-api 4.2"
DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# "mon" and "monday" both name Monday (0); Python's date.weekday() numbering.
DAY_NUMBERS = {
    **{name: number for number, name in enumerate(DAY_NAMES)},
    **{name[:3]: number for number, name in enumerate(DAY_NAMES)},
}


def outside(what: str) -> ConfigError:
    """The error for anything outside the grammar."""
    return ConfigError(f"{what} is outside {GRAMMAR}; write this logic in on_bar instead")


class ConditionOperators:
    """The operators of the grammar, for any class whose `__condition_expr__()` returns
    the `Expr` it stands for. Each builds a new expression; nothing is computed here."""

    # NumPy defers to these operators instead of applying them element by element, so
    # `array & condition` builds an expression too.
    __array_ufunc__ = None

    def __condition_expr__(self) -> Expr:
        raise NotImplementedError

    def __add__(self, other: Any) -> Expr:
        return Arithmetic.of("+", self, other)

    def __radd__(self, other: Any) -> Expr:
        return Arithmetic.of("+", other, self)

    def __sub__(self, other: Any) -> Expr:
        return Arithmetic.of("-", self, other)

    def __rsub__(self, other: Any) -> Expr:
        return Arithmetic.of("-", other, self)

    def __mul__(self, other: Any) -> Expr:
        return Arithmetic.of("*", self, other)

    def __rmul__(self, other: Any) -> Expr:
        return Arithmetic.of("*", other, self)

    def __truediv__(self, other: Any) -> Expr:
        return Arithmetic.of("/", self, other)

    def __rtruediv__(self, other: Any) -> Expr:
        return Arithmetic.of("/", other, self)

    def __lt__(self, other: Any) -> Expr:
        return Comparison.of("<", self, other)

    def __le__(self, other: Any) -> Expr:
        return Comparison.of("<=", self, other)

    def __gt__(self, other: Any) -> Expr:
        return Comparison.of(">", self, other)

    def __ge__(self, other: Any) -> Expr:
        return Comparison.of(">=", self, other)

    def __and__(self, other: Any) -> Expr:
        return Logical.of("&", self, other)

    def __rand__(self, other: Any) -> Expr:
        return Logical.of("&", other, self)

    def __or__(self, other: Any) -> Expr:
        return Logical.of("|", self, other)

    def __ror__(self, other: Any) -> Expr:
        return Logical.of("|", other, self)

    def __invert__(self) -> Expr:
        return Negation(typed(self, BOOL, "~"))

    def crossed_above(self, other: Any) -> Expr:
        """True at a bar that closes above `other` after the bar before closed at or
        below it (`openalgo.ta.crossover`); false where a value is NaN."""
        return Crossing("above", typed(self, NUMBER, "crossed_above"), typed(other, NUMBER, "x"))

    def crossed_below(self, other: Any) -> Expr:
        """True at a bar that closes below `other` after the bar before closed at or
        above it (`openalgo.ta.crossunder`); false where a value is NaN."""
        return Crossing("below", typed(self, NUMBER, "crossed_below"), typed(other, NUMBER, "x"))

    def shift(self, bars: int) -> Expr:
        """The value `bars` strategy bars earlier, 0 or more (NaN for a number and false
        for a condition before the first bar). A negative shift would read a later bar."""
        return Shifted(as_expr(self), _bar_count(bars, "shift", minimum=0))

    def rolling_max(self, bars: int) -> Expr:
        """The highest value of the last `bars` strategy bars, this one included
        (`openalgo.ta.highest`); NaN until `bars` bars have closed."""
        operand = typed(self, NUMBER, "rolling_max")
        return RollingExtreme("max", operand, _bar_count(bars, "rolling_max"))

    def rolling_min(self, bars: int) -> Expr:
        """The lowest value of the last `bars` strategy bars, this one included
        (`openalgo.ta.lowest`); NaN until `bars` bars have closed."""
        operand = typed(self, NUMBER, "rolling_min")
        return RollingExtreme("min", operand, _bar_count(bars, "rolling_min"))


class Expr(ConditionOperators):
    """A node of a condition expression; `kind` is NUMBER or BOOL. Nodes are immutable
    and compared by identity."""

    kind: ClassVar[str] = NUMBER

    def __condition_expr__(self) -> Expr:
        return self

    def __bool__(self) -> bool:
        raise outside("the truth value of a condition (and, or, not, if, chained <)")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        raise outside("== (compare prices with < <= > >=)")

    def __ne__(self, other: object) -> bool:  # type: ignore[override]
        raise outside("!= (compare prices with < <= > >=)")

    __hash__ = object.__hash__


def _bar_count(value: Any, what: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        reason = " (a negative shift would read a later bar)" if what == "shift" else ""
        raise ConfigError(f"{what} takes a whole number of bars, {minimum} or more{reason}")
    return int(value)


def as_expr(value: Any) -> Expr:
    """The expression an operand stands for. Raises ConfigError for anything outside
    the grammar."""
    hook = getattr(value, "__condition_expr__", None)
    if hook is not None:
        return hook()
    if isinstance(value, bool | np.bool_):
        raise outside("True or False as an operand (== and != are not in the grammar)")
    if isinstance(value, numbers.Real):
        return Constant(float(value))
    if isinstance(value, pd.Series | np.ndarray | list | tuple):
        return Values.of(value)
    raise outside(f"a {type(value).__name__} operand")


def typed(value: Any, kind: str, where: str) -> Expr:
    """`as_expr`, checked to be a number or a condition as `where` needs. Raises
    ConfigError otherwise."""
    expr = as_expr(value)
    if expr.kind != kind:
        needed = "a number" if kind == NUMBER else "a condition (true or false)"
        raise ConfigError(f"{where} needs {needed}, got a {expr.kind} ({GRAMMAR})")
    return expr


@dataclass(frozen=True, eq=False)
class Column(Expr):
    """A column of a symbol's strategy bars: open, high, low, close or volume."""

    symbol: str
    field: str


@dataclass(frozen=True, eq=False)
class Constant(Expr):
    """A number: a constant or a parameter's value."""

    value: float


@dataclass(frozen=True, eq=False)
class Values(Expr):
    """Values given as an array: one per strategy bar in bar order (`times` None), or
    keyed by bar open time (`times`, int64 UTC nanoseconds). `value_kind` is NUMBER or
    BOOL."""

    values: np.ndarray
    times: np.ndarray | None
    value_kind: str

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.value_kind

    @classmethod
    def of(cls, data: Any) -> Values:
        """From a Series with a DatetimeIndex (aligned by bar open time; a naive index is
        IST) or from values in bar order. Raises ConfigError for values that are neither
        booleans nor numbers, or not one-dimensional."""
        times = None
        if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
            index = data.index if data.index.tz is not None else data.index.tz_localize(IST)
            # As nanoseconds whatever resolution pandas inferred for the index.
            times = index.tz_convert("UTC").as_unit("ns").asi8.astype(np.int64)
        values = np.asarray(data.to_numpy() if isinstance(data, pd.Series) else data)
        if values.ndim != 1:
            raise ConfigError(f"a condition array must be one-dimensional, got {values.shape}")
        if values.dtype == bool:
            return cls(values, times, BOOL)
        if np.issubdtype(values.dtype, np.number):
            return cls(values.astype(np.float64), times, NUMBER)
        raise ConfigError(f"a condition array holds booleans or numbers, not {values.dtype}")


@dataclass(frozen=True, eq=False)
class IndicatorCall:
    """One call of an `openalgo.ta` function: `args` are expressions (series,
    indicators, arrays) or plain values passed as they are (periods, multipliers)."""

    func: Any
    args: tuple
    params: dict


@dataclass(frozen=True, eq=False)
class Indicator(Expr):
    """One output of an indicator call; `output` is None for a single-output function."""

    call: IndicatorCall
    output: int | None


@dataclass(frozen=True, eq=False)
class Arithmetic(Expr):
    """`left op right` with op one of + - * /; a division by zero gives inf or NaN."""

    op: str
    left: Expr
    right: Expr

    @classmethod
    def of(cls, op: str, left: Any, right: Any) -> Arithmetic:
        return cls(op, typed(left, NUMBER, op), typed(right, NUMBER, op))


@dataclass(frozen=True, eq=False)
class Comparison(Expr):
    """`left op right` with op one of < <= > >=; false where a value is NaN."""

    kind: ClassVar[str] = BOOL
    op: str
    left: Expr
    right: Expr

    @classmethod
    def of(cls, op: str, left: Any, right: Any) -> Comparison:
        return cls(op, typed(left, NUMBER, op), typed(right, NUMBER, op))


@dataclass(frozen=True, eq=False)
class Logical(Expr):
    """`left & right` or `left | right` on conditions."""

    kind: ClassVar[str] = BOOL
    op: str
    left: Expr
    right: Expr

    @classmethod
    def of(cls, op: str, left: Any, right: Any) -> Logical:
        return cls(op, typed(left, BOOL, op), typed(right, BOOL, op))


@dataclass(frozen=True, eq=False)
class Negation(Expr):
    """`~operand`."""

    kind: ClassVar[str] = BOOL
    operand: Expr


@dataclass(frozen=True, eq=False)
class Crossing(Expr):
    """`left.crossed_above(right)` ("above") or `.crossed_below` ("below")."""

    kind: ClassVar[str] = BOOL
    direction: str
    left: Expr
    right: Expr


@dataclass(frozen=True, eq=False)
class Shifted(Expr):
    """The operand `bars` strategy bars earlier."""

    operand: Expr
    bars: int

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.operand.kind


@dataclass(frozen=True, eq=False)
class RollingExtreme(Expr):
    """The highest ("max") or lowest ("min") of the last `bars` values."""

    extreme: str
    operand: Expr
    bars: int


@dataclass(frozen=True, eq=False)
class TimeBetween(Expr):
    """True when the bar's close (its decision instant) is at or after `start` and
    before `end`, IST seconds since midnight."""

    kind: ClassVar[str] = BOOL
    start: int
    end: int


@dataclass(frozen=True, eq=False)
class Weekday(Expr):
    """True when the bar closes on one of `days` (IST; Monday 0 to Sunday 6)."""

    kind: ClassVar[str] = BOOL
    days: frozenset


def column(symbol: str, field: str) -> Column:
    """A column of a symbol's strategy bars as an operand (what a series view stands
    for)."""
    return Column(symbol, field)


def indicator(func: Any, *inputs: Any, **params: Any) -> Indicator | tuple[Indicator, ...]:
    """An `openalgo.ta` function over series views, indicators or arrays, as an operand:
    `indicator(ta.ema, self.close, 10)`. A function with several outputs (`ta.macd`)
    gives one operand per output, in the order `openalgo.indicators.causality.outputs`
    lists them; the function runs once per run for all of them. Raises ConfigError for a
    function that is not in `openalgo.ta`; an output that looks ahead
    (`openalgo.indicators.causality`) raises ConfigError when a condition uses it."""
    from openalgo.indicators import causality

    try:
        names = causality.outputs(func)
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"indicators come only from openalgo.ta: {exc}") from None
    args = tuple(_indicator_argument(value) for value in inputs)
    call = IndicatorCall(func, args, dict(params))
    if len(names) == 1:
        return Indicator(call, None)
    return tuple(Indicator(call, position) for position in range(len(names)))


def _indicator_argument(value: Any) -> Any:
    """Series, indicators and arrays become expressions; periods and options stay."""
    if hasattr(value, "__condition_expr__") or isinstance(value, pd.Series | np.ndarray | list):
        return typed(value, NUMBER, "an indicator input")
    return value


def time_between(start: str, end: str) -> TimeBetween:
    """A condition true at bars whose close (the decision instant) is at or after
    `start` and before `end`, IST "HH:MM": the half-open rule of the entry window.
    Raises ConfigError for an unreadable time or unless start is before end."""
    first = to_seconds(start, what="time_between start")
    last = to_seconds(end, what="time_between end")
    if first >= last:
        raise ConfigError(f"time_between needs start before end, got {start!r} and {end!r}")
    return TimeBetween(first, last)


def weekday(*days: str | int) -> Weekday:
    """A condition true at bars that close on one of `days` (IST): names ("Mon",
    "monday", any case) or numbers, Monday 0 to Sunday 6. Raises ConfigError otherwise."""
    if not days:
        raise ConfigError("weekday needs at least one day, for example bt.weekday('Mon')")
    return Weekday(frozenset(_day_number(day) for day in days))


def _day_number(day: str | int) -> int:
    if isinstance(day, int) and not isinstance(day, bool) and 0 <= day <= 6:
        return day
    if isinstance(day, str) and day.strip().lower() in DAY_NUMBERS:
        return DAY_NUMBERS[day.strip().lower()]
    raise ConfigError(f"weekday takes day names (Mon to Sun) or 0 (Monday) to 6, got {day!r}")


__all__ = [
    "BOOL",
    "NUMBER",
    "ConditionOperators",
    "Expr",
    "as_expr",
    "column",
    "indicator",
    "outside",
    "time_between",
    "typed",
    "weekday",
]

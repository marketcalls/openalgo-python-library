"""Units for risk arguments and parameters: `bt.points`, `bt.pct`, `bt.rupees`, `bt.ticks`,
`bt.atr`, the trailing forms, `bt.breakeven`, `bt.lock` and `bt.param`.

Conventions:

* A bare number where a distance is expected is points (rupees of price).
* `bt.pct(x)` is x percent of the leg's fill average (`bt.pct(1)` is 1 percent).
* `bt.rupees(x)` is rupees of the leg's MTM; `bt.ticks(n)` whole ticks.
* `bt.atr(mult, period)` is a multiple of the ATR from `openalgo.ta`, taken at the close
  before the entry.
* `bt.trail(x, y)` is OpenAlgo's trailing stop: `y = 0` continuous, `y > 0` stepped.
  `bt.trail_points(x)` is `bt.trail(x, 0)`; `bt.trail_pct(p)` and `bt.trail_atr(...)` are
  the same with the distance in percent or ATR.

These objects only describe; the kernel evaluates them. Raises ConfigError for a value
that is not a positive number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ConfigError

UNITS = ("points", "pct", "ticks", "rupees", "atr")


def _positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{what} must be a positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Distance:
    """A distance from the reference price: `unit` is one of UNITS, `value` is in that
    unit, `period` is the ATR period for the "atr" unit (`value` is then the multiple)."""

    unit: str
    value: float
    period: int | None = None

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ConfigError(f"unknown distance unit {self.unit!r}: use one of {UNITS}")
        _positive(self.value, f"a distance in {self.unit}")


def points(value: float) -> Distance:
    """A distance in points (rupees of price)."""
    return Distance("points", _positive(value, "points"))


def pct(value: float) -> Distance:
    """A distance in percent of the leg's fill average (`pct(1)` is 1 percent)."""
    return Distance("pct", _positive(value, "percent"))


def rupees(value: float) -> Distance:
    """A distance in rupees of the leg's MTM, or a money amount where one is expected."""
    return Distance("rupees", _positive(value, "rupees"))


def ticks(count: int) -> Distance:
    """A distance in whole ticks of the instrument."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ConfigError(f"ticks must be a positive whole number, got {count!r}")
    return Distance("ticks", float(count))


def atr(multiple: float, period: int = 14) -> Distance:
    """A multiple of the ATR of `period` bars (from `openalgo.ta`)."""
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ConfigError(f"the ATR period must be a positive whole number, got {period!r}")
    return Distance("atr", _positive(multiple, "the ATR multiple"), period)


def as_distance(value: object, what: str) -> Distance:
    """A Distance from a Distance or a bare number (points). Raises ConfigError."""
    if isinstance(value, Distance):
        return value
    return points(_positive(value, what))


@dataclass(frozen=True)
class Trail:
    """OpenAlgo's `{x, y}` trailing stop: `trigger` is x; `step` is y, None when
    continuous."""

    trigger: Distance
    step: Distance | None = None


def trail(x: float | Distance, y: float | Distance = 0) -> Trail:
    """OpenAlgo's trailing stop `{x, y}`; bare numbers are points."""
    step = None if (not isinstance(y, Distance) and y == 0) else as_distance(y, "trail step")
    return Trail(as_distance(x, "trail trigger"), step)


def trail_points(x: float) -> Trail:
    """A continuous trail of x points (`trail(x, 0)`)."""
    return Trail(points(x))


def trail_pct(p: float) -> Trail:
    """A continuous trail of p percent of the fill average."""
    return Trail(pct(p))


def trail_atr(multiple: float, period: int = 14) -> Trail:
    """A continuous trail of an ATR multiple."""
    return Trail(atr(multiple, period))


@dataclass(frozen=True)
class Breakeven:
    """Move the stop to entry plus `offset` once the favourable move reaches `trigger`."""

    trigger: Distance
    offset: Distance | None = None


def breakeven(trigger: float | Distance, offset: float | Distance | None = None) -> Breakeven:
    """A breakeven move; bare numbers are points, no offset moves the stop to entry."""
    moved = None if offset is None else as_distance(offset, "breakeven offset")
    return Breakeven(as_distance(trigger, "breakeven trigger"), moved)


@dataclass(frozen=True)
class Lock:
    """Lock profit: arm at `reach` rupees of MTM, lock `lock` rupees, trail by
    `trail_step` rupees when given (OpenAlgo's lock and lock_and_trail)."""

    reach: float
    lock: float
    trail_step: float | None = None


def lock(reach: float, lock_amount: float, trail_step: float | None = None) -> Lock:
    """Lock profit in rupees, a combined rule over the run's positions. A run with
    `lock_profit` is not supported in this version and raises NotImplementedError."""
    step = None if trail_step is None else _positive(trail_step, "trail_step")
    return Lock(_positive(reach, "reach"), _positive(lock_amount, "lock"), step)


@dataclass(frozen=True)
class Param:
    """A strategy parameter: its default and the values a sweep tries."""

    default: Any
    values: tuple[Any, ...] = ()


def param(default: Any, values: Any = ()) -> Param:
    """Declare a parameter with its sweep range: `fast = bt.param(10, range(5, 30))`."""
    return Param(default, tuple(values))


__all__ = [
    "UNITS",
    "Breakeven",
    "Distance",
    "Lock",
    "Param",
    "Trail",
    "as_distance",
    "atr",
    "breakeven",
    "lock",
    "param",
    "pct",
    "points",
    "rupees",
    "ticks",
    "trail",
    "trail_atr",
    "trail_pct",
    "trail_points",
]

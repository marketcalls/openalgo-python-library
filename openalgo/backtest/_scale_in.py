"""The scale-in limits a strategy declares as class attributes (12-position-sizing 2,
14-strategy-api 2.1, decision SPEC-15), in the units the kernel takes
(`rust/oa_bt/src/rms/scale_in.rs`).

    class Pyramid(bt.Strategy):
        max_adds = 2                      # adds to one position
        max_position = bt.size.lots(4)    # or units (a number), bt.rupees(x), bt.pct(x)
        min_spacing = 3                   # base bars, or bt.points(x), bt.pct(x), bt.ticks(n)
        max_risk = bt.rupees(5_000)       # or a number of rupees, or bt.pct(x) of equity
        allow_averaging_down = False      # the default

Each limit sets nothing unless it is given. Averaging down, adding while the decision
price is worse than the average price (below it for a long, above it for a short), is
refused unless `allow_averaging_down = True`. The kernel checks the limits on every add
at submission and rejects an add that breaks one with reject code `scale_in_limit`; an
entry from flat is not an add.

The forms, and what they mean:

* `max_position`: a number or `bt.size.qty(n)` is units; `bt.size.lots(n)` lots; a value
  in rupees at the decision price is `bt.rupees(x)` or `bt.size.value(x)`; a percent of
  equity is `bt.pct(x)` or `bt.size.percent(x)`.
* `min_spacing`: a whole number is base bars closed after the bar in which the last
  tranche filled; `bt.points(x)`, `bt.pct(x)` (of the last tranche's price) and
  `bt.ticks(n)` are the least price distance from the last tranche.
* `max_risk`: the open risk of the whole position after the add (average price to the
  stop, times units and multiplier). A number or `bt.rupees(x)` is rupees, `bt.pct(x)` a
  percent of equity. A position without a stop has no bounded risk, so with `max_risk`
  set its adds are refused.
"""

from __future__ import annotations

from typing import Any

from .errors import ConfigError
from .size import Sizer
from .units import Distance

# The class attributes this module reads, with the value that sets no limit.
LIMITS = {
    "max_adds": None,
    "max_position": None,
    "min_spacing": None,
    "max_risk": None,
    "allow_averaging_down": False,
}
PERCENT = 100.0
NativeLimit = tuple[str, float]


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _positive(value: Any) -> bool:
    return _is_number(value) and value > 0


def _max_adds(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"max_adds takes a whole number of adds, 0 or more, got {value!r}")
    return value


def position_limit(value: Any, name: str = "max_position") -> NativeLimit | None:
    """A position limit in the kernel's (kind, amount) form, or None; the forms are those
    of `max_position` above (also read by `bt.size.caps`). Raises ConfigError naming
    `name` for any other value."""
    if value is None:
        return None
    if _positive(value):
        return ("units", float(value))
    if isinstance(value, Distance) and value.unit in ("rupees", "pct"):
        return ("rupees" if value.unit == "rupees" else "percent", value.value)
    if isinstance(value, Sizer) and value.kind == "qty":
        return ("units", float(value.amount))
    if isinstance(value, Sizer) and value.kind == "lots":
        return ("lots", float(value.amount))
    if isinstance(value, Sizer) and value.kind == "value":
        return ("rupees", float(value.amount))
    if isinstance(value, Sizer) and value.kind == "percent" and value.options["basis"] == "equity":
        return ("percent", float(value.amount))
    raise ConfigError(
        f"{name} takes units (a positive number or bt.size.qty), bt.size.lots(n), "
        "rupees of value (bt.rupees or bt.size.value) or a percent of equity (bt.pct or "
        f"bt.size.percent), got {value!r}"
    )


def _min_spacing(value: Any) -> NativeLimit | None:
    if value is None:
        return None
    if _is_number(value) and isinstance(value, int) and value >= 1:
        return ("bars", float(value))
    if isinstance(value, Distance) and value.unit in ("points", "pct", "ticks"):
        return (value.unit, value.value)
    raise ConfigError(
        "min_spacing takes a whole number of base bars (1 or more), or a price distance "
        f"bt.points(x), bt.pct(x) or bt.ticks(n), got {value!r}"
    )


def risk_limit(value: Any, name: str = "max_risk") -> NativeLimit | None:
    """A risk limit in the kernel's (kind, amount) form, or None; the forms are those of
    `max_risk` above (also read by `bt.size.caps`). Raises ConfigError naming `name`
    for any other value."""
    if value is None:
        return None
    if _positive(value):
        return ("rupees", float(value))
    if isinstance(value, Distance) and value.unit == "rupees":
        return ("rupees", value.value)
    if isinstance(value, Distance) and value.unit == "pct" and value.value < PERCENT:
        return ("percent", value.value)
    raise ConfigError(
        f"{name} takes rupees (a positive number or bt.rupees) or a percent of equity "
        f"below 100 (bt.pct), got {value!r}"
    )


def _flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"allow_averaging_down takes True or False, got {value!r}")
    return value


def scale_limits(strategy: type) -> dict[str, Any]:
    """The strategy's scale-in limits as the entries of the kernel's risk dict. Raises
    ConfigError for a value of the wrong kind."""
    given = {name: getattr(strategy, name, unset) for name, unset in LIMITS.items()}
    return {
        "max_adds": _max_adds(given["max_adds"]),
        "max_position": position_limit(given["max_position"]),
        "min_spacing": _min_spacing(given["min_spacing"]),
        "max_risk": risk_limit(given["max_risk"]),
        "allow_averaging_down": _flag(given["allow_averaging_down"]),
    }


_POSITION_UNITS = {
    "units": "units",
    "lots": "lots",
    "rupees": "rupees",
    "percent": "percent of equity",
}
_SPACING_UNITS = {"bars": "base bars", "points": "points", "pct": "percent", "ticks": "ticks"}
_RISK_UNITS = {"rupees": "rupees", "percent": "percent of equity"}


def _text(limit: NativeLimit | None, units: dict[str, str]) -> str | None:
    if limit is None:
        return None
    kind, amount = limit
    return f"{amount:g} {units[kind]}"


def describe(risk: dict[str, Any]) -> dict[str, Any]:
    """The scale-in limits of a resolved risk dict, for the Assumptions block."""
    return {
        "max_adds": risk["max_adds"],
        "max_position": _text(risk["max_position"], _POSITION_UNITS),
        "min_spacing": _text(risk["min_spacing"], _SPACING_UNITS),
        "max_risk": _text(risk["max_risk"], _RISK_UNITS),
        "allow_averaging_down": risk["allow_averaging_down"],
    }


__all__ = ["LIMITS", "describe", "position_limit", "risk_limit", "scale_limits"]

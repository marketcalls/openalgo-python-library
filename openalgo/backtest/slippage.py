"""Slippage for `bt.backtest(slippage=...)` (04-execution-and-market-model section 4.3).

Every MARKET fill, entry or exit, is worse than its reference price by the slippage for
its side (a buy pays more, a sell receives less) and is rounded to the tick away from
the trader (decision SPEC-27). The forms:

    "default"              the dated default table, one row per slippage class (the
                           default; see `defaults()`)
    0                      no slippage
    bt.ticks(n)            n ticks per side
    bt.pct(p)              p percent of the price (of the premium for options), at least
                           one tick
    bt.slippage.bps(b)     b basis points of the price, at least one tick
    bt.slippage.table(overrides={"INDEX_FUT": bt.ticks(2)}, low_volume=("IDEA",))
                           the default table with some classes overridden and the cash-
                           equity symbols to treat as low volume
    {"INDEX_FUT": bt.ticks(2)}   shorthand for `table(overrides=...)`

Slippage classes (`CLASSES`): EQ_LARGE_MID (cash equity, the default), EQ_LOW_VOLUME
(cash equity the run names as low volume), INDEX_FUT, STOCK_FUT, INDEX_OPT, STOCK_OPT,
MCX_FUT, MCX_OPT and CDS. The figures live in the kernel's data file
`rust/oa_bt/src/market/slippage_defaults.csv`, each row with its source and
verification status.

Raises ConfigError for any other value.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ._native import native
from .errors import ConfigError
from .units import Distance

#: The slippage classes of the default table, in table order.
CLASSES: tuple[str, ...] = tuple(dict.fromkeys(row[1] for row in native.slippage_defaults()))
#: The least slippage, in ticks, of the bps and percent forms (04-execution 4.3).
MIN_TICKS = 1
_ROW_FIELDS = (
    "row_id",
    "slippage_class",
    "valid_from",
    "valid_to",
    "model",
    "amount",
    "min_ticks",
    "source",
    "status",
    "notes",
)
_FORMS = "'default', 0, bt.ticks(n), bt.pct(p), bt.slippage.bps(b) or bt.slippage.table(...)"


@dataclass(frozen=True)
class Slippage:
    """One slippage form as the kernel reads it: `model` is "zero", "ticks", "bps" or
    "percent", `amount` is in that unit, `min_ticks` is the least slippage in ticks."""

    model: str
    amount: float
    min_ticks: int = 0

    def to_native(self) -> tuple[str, float, int]:
        """The (kind, amount, min_ticks) tuple of the binding."""
        return (self.model, float(self.amount), int(self.min_ticks))


@dataclass(frozen=True)
class SlippageTable:
    """The default table with some classes overridden (`overrides`, by class code) and
    the cash-equity symbols treated as low volume (`low_volume`)."""

    overrides: Mapping[str, Slippage] = field(default_factory=lambda: MappingProxyType({}))
    low_volume: tuple[str, ...] = ()


def _positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{what} must be a positive number, got {value!r}")
    return float(value)


def bps(points: float, min_ticks: int = MIN_TICKS) -> Slippage:
    """`points` basis points of the price per side (2 is 0.02 percent), at least
    `min_ticks` ticks. Raises ConfigError for a value that is not positive."""
    if isinstance(min_ticks, bool) or not isinstance(min_ticks, int) or min_ticks < 0:
        raise ConfigError(f"min_ticks must be a whole number, 0 or more, got {min_ticks!r}")
    return Slippage("bps", _positive(points, "bps"), min_ticks)


def form(value: object) -> Slippage:
    """One slippage form from 0, bt.ticks(n), bt.pct(p) or a `Slippage`.

    Raises ConfigError for anything else.
    """
    if isinstance(value, Slippage):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value == 0:
        return Slippage("zero", 0.0)
    if isinstance(value, Distance) and value.unit == "ticks":
        return Slippage("ticks", value.value)
    if isinstance(value, Distance) and value.unit == "pct":
        return Slippage("percent", value.value, MIN_TICKS)
    raise ConfigError(f"slippage takes {_FORMS}, got {value!r}")


def table(
    overrides: Mapping[str, object] | None = None, low_volume: tuple[str, ...] | list[str] = ()
) -> SlippageTable:
    """The default table with `overrides` (class code to a form) and the cash-equity
    symbols in `low_volume` priced by the EQ_LOW_VOLUME row.

    Raises ConfigError for an unknown class or form.
    """
    resolved = {}
    for code, value in dict(overrides or {}).items():
        if code not in CLASSES:
            raise ConfigError(f"unknown slippage class {code!r}: use one of {CLASSES}")
        resolved[code] = form(value)
    symbols = tuple(str(symbol) for symbol in low_volume)
    return SlippageTable(MappingProxyType(resolved), symbols)


def resolve(value: object) -> Slippage | SlippageTable:
    """A `slippage=` argument as one form for every instrument, or a table.

    Raises ConfigError for a value that is not one of the forms in the module docstring.
    """
    if isinstance(value, str):
        if value != "default":
            raise ConfigError(f"slippage takes {_FORMS}, got {value!r}")
        return SlippageTable()
    if isinstance(value, SlippageTable):
        return value
    if isinstance(value, Mapping):
        return table(value)
    return form(value)


def label(value: object) -> str:
    """A short description of a `slippage=` argument for the Assumptions block."""
    resolved = resolve(value)
    if isinstance(resolved, Slippage):
        return "zero" if resolved.model == "zero" else f"{resolved.model} {resolved.amount:g}"
    parts = [f"{code}: {item.model} {item.amount:g}" for code, item in resolved.overrides.items()]
    if resolved.low_volume:
        parts.append(f"low volume: {', '.join(resolved.low_volume)}")
    return "default table" + (f" ({'; '.join(parts)})" if parts else "")


def _date(day: int | None) -> dt.date | None:
    return None if day is None else dt.date(1970, 1, 1) + dt.timedelta(days=day)


def defaults() -> list[dict[str, Any]]:
    """The built-in default table: one dict per row with row_id, slippage_class,
    valid_from and valid_to (dates, None for open), model, amount, min_ticks, source,
    status ("verified" or "unverified") and notes."""
    rows = []
    for raw in native.slippage_defaults():
        row = dict(zip(_ROW_FIELDS, raw, strict=True))
        row["valid_from"], row["valid_to"] = _date(row["valid_from"]), _date(row["valid_to"])
        rows.append(row)
    return rows


__all__ = [
    "CLASSES",
    "MIN_TICKS",
    "Slippage",
    "SlippageTable",
    "bps",
    "defaults",
    "form",
    "label",
    "resolve",
    "table",
]

"""`margin=`: the futures margin of a backtest (decision D23; 04-execution 5).

By default every futures contract, on any exchange and underlying (NSE index and stock
futures, BSE, MCX), blocks the default row of the kernel's margin table: 20 percent of
its contract value (5x), on any date. Real margins differ by underlying and change every
day, so a run can give its own:

    bt.backtest(..., margin=bt.pct(15))                      # every futures contract
    bt.backtest(..., margin={"NIFTY": bt.pct(12), "CRUDEOIL": bt.rupees(450_000)})

A value is a percent of the contract value, `bt.pct(x)` (above 0 and at most 100), or
rupees per lot of the contract traded, `bt.rupees(x)`: the figure a broker's margin
calculator shows for one lot. A bare number is refused, since it could be read as
either. A dict is keyed by underlying symbol, in capitals as OpenAlgo writes it; an
underlying it does not name keeps the default. Rupees per lot count the lots of the
contract traded (its own lot size), so a lot size changed at a roll is the next
contract's.

It is a backtest assumption only: in live trading the broker computes margin, so
`bt.live` takes no `margin=`. The parsed form crosses to the kernel per instrument
(`_market`), is stated in the Assumptions block and recorded in the manifest.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any

from ._calendar_convert import to_day
from ._futures_contracts import underlying_of
from ._native import native
from .errors import BacktestError, ConfigError
from .reference import Instrument
from .units import Distance

# A futures margin as the kernel takes it: ("percent", percent of contract value) or
# ("rupees_per_lot", rupees).
MarginRule = tuple[str, float]
# The parsed `margin=`: None (the default), one rule for every futures contract, or
# rules by underlying symbol.
MarginSetting = MarginRule | dict[str, MarginRule] | None

PERCENT = "percent"
RUPEES_PER_LOT = "rupees_per_lot"
# A margin cannot block more than the whole contract value.
MOST_PERCENT_OF_VALUE = 100.0
# An underlying as OpenAlgo writes it: capitals, digits, "&" and "-" (M&M, BAJAJ-AUTO).
_UNDERLYING = re.compile(r"[A-Z0-9][A-Z0-9&_-]*")
# The expiry date inside an OpenAlgo derivative symbol (NIFTY28NOV24FUT).
_CONTRACT_DATE = re.compile(r"\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}")
_FORMS = (
    "bt.pct(x) for x percent of the contract value (above 0, at most 100) or bt.rupees(x) "
    "for x rupees per lot, the figure a broker's margin calculator shows"
)
# The kernel's code of a futures margin-table row (`InstrumentKind::Future`).
_FUTURES_KIND = "FUT"


def _rule(value: Any, where: str) -> MarginRule:
    """One value of `margin=` as a kernel rule. Raises ConfigError naming `where`."""
    if isinstance(value, Distance) and value.unit == "pct":
        if not value.value <= MOST_PERCENT_OF_VALUE:
            raise ConfigError(
                f"{where}: a futures margin is at most 100 percent of the contract value, "
                f"got bt.pct({value.value:g})"
            )
        return (PERCENT, value.value)
    if isinstance(value, Distance) and value.unit == "rupees":
        if not math.isfinite(value.value):
            raise ConfigError(f"{where}: rupees per lot must be a finite amount, got {value!r}")
        return (RUPEES_PER_LOT, value.value)
    if not isinstance(value, bool) and isinstance(value, int | float):
        raise ConfigError(
            f"{where}: a bare number could be a percent or rupees; write {_FORMS}, got {value!r}"
        )
    raise ConfigError(f"{where} takes {_FORMS}, got {value!r}")


def _underlying_name(key: Any) -> str:
    """A dict key of `margin=` as an underlying symbol. Raises ConfigError for anything
    that is not one, saying what to write."""
    shape = (
        "margin= takes a dict keyed by underlying symbol ('NIFTY', 'CRUDEOIL', 'M&M'), "
        f"got the key {key!r}"
    )
    if not isinstance(key, str):
        raise ConfigError(shape)
    capitals = key.upper()
    if _CONTRACT_DATE.search(capitals):
        raise ConfigError(
            f"margin= is set per underlying, not per contract: write the underlying of "
            f"{key!r} (for example 'NIFTY'), not the contract symbol"
        )
    if not _UNDERLYING.fullmatch(capitals):
        raise ConfigError(shape)
    if key != capitals:
        raise ConfigError(
            f"margin= names underlyings in capitals, as OpenAlgo writes them: write "
            f"{capitals!r}, not {key!r}"
        )
    return key


def parse_margin(value: Any) -> MarginSetting:
    """The parsed `margin=` (see the module docstring), dict keys sorted. Raises
    ConfigError for a bare number, a percent above 100, a value that is not
    `bt.pct(x)` or `bt.rupees(x)`, an empty dict or a key that is not an underlying
    symbol."""
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            raise ConfigError(
                "margin= got an empty dict: name at least one underlying "
                "({'NIFTY': bt.pct(12)}), or leave margin= out for the default"
            )
        rules = {
            _underlying_name(key): _rule(item, f"margin[{key!r}]") for key, item in value.items()
        }
        return dict(sorted(rules.items()))
    return _rule(value, "margin=")


def _name(instrument: Instrument) -> str:
    """The name `margin=` knows a future by: its underlying (as given, else read from its
    symbol, as the futures series do), or its own symbol when it has none."""
    return underlying_of(instrument) or instrument.symbol


def own_margin(setting: MarginSetting, instrument: Instrument) -> MarginRule | None:
    """The run's own margin for `instrument`: None for anything but a future, and for a
    future whose underlying `margin=` does not name (the default applies)."""
    if setting is None or instrument.kind != "future":
        return None
    if isinstance(setting, dict):
        return setting.get(_name(instrument))
    return setting


def default_percent(day: dt.date) -> float:
    """The default futures margin on `day`, percent of contract value, from the kernel's
    margin table (rows as `native.margin_defaults()` gives them). Raises BacktestError
    when the table has no futures row for the day, a defect of the release: it holds
    one for every date."""
    number = to_day(day)
    for row in native.margin_defaults():
        _, kind, _, valid_from, valid_to, _, percent, *_ = row
        covers = (valid_from is None or valid_from <= number) and (
            valid_to is None or number <= valid_to
        )
        if kind == _FUTURES_KIND and covers:
            return percent
    raise BacktestError(
        f"the built-in margin table has no futures row for {day}; reinstall openalgo"
    )


def _number(value: float) -> str:
    """A figure without an exponent or trailing zeros (450000, 12.5)."""
    return f"{value:.15g}"


def _text(rule: MarginRule) -> str:
    kind, amount = rule
    if kind == PERCENT:
        return f"{_number(amount)} percent of contract value (margin=)"
    return f"{_number(amount)} rupees per lot (margin=)"


def _futures_underlyings(instruments: list[Instrument]) -> dict[str, Instrument]:
    """The futures the run trades, one per underlying (`_name`), in registration order."""
    traded: dict[str, Instrument] = {}
    for instrument in instruments:
        if instrument.kind == "future":
            traded.setdefault(_name(instrument), instrument)
    return traded


def margin_assumption(
    setting: MarginSetting, instruments: list[Instrument], day: dt.date
) -> str | None:
    """The Assumptions line: the futures margin in force for each underlying traded and
    whether it is the default or the run's own, or None when the run trades no futures.
    `day` is the run's first day, for the default row."""
    traded = _futures_underlyings(instruments)
    if not traded:
        return None
    default = f"{_number(default_percent(day))} percent of contract value (default)"
    parts = []
    for name, instrument in traded.items():
        rule = own_margin(setting, instrument)
        parts.append(f"{name} {default if rule is None else _text(rule)}")
    return "; ".join(parts)


def margin_warnings(setting: MarginSetting, instruments: list[Instrument]) -> list[str]:
    """Warnings for a `margin=` this run does not use: given while the run trades no
    futures, or naming underlyings it does not trade as futures."""
    if setting is None:
        return []
    traded = _futures_underlyings(instruments)
    if not traded:
        return [
            "margin= sets the margin of futures contracts; this run trades none, so it is unused"
        ]
    if not isinstance(setting, dict):
        return []
    unused = [name for name in setting if name not in traded]
    if not unused:
        return []
    values = "its value is" if len(unused) == 1 else "their values are"
    return [
        f"margin= names {', '.join(unused)}, which this run does not trade as futures; "
        f"{values} unused"
    ]


__all__ = [
    "MarginRule",
    "MarginSetting",
    "default_percent",
    "margin_assumption",
    "margin_warnings",
    "own_margin",
    "parse_margin",
]

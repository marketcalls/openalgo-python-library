"""What a backtest does with the corporate actions of its data (FR-MKT-10; PRD 04
section 7 and PRD 06 section 5).

1. `declared_for` gives each symbol its declarations from the run's
   `corporate_actions=` (a list for a one-symbol run, a dict by symbol otherwise). The
   data layer's split check (`data.splits`) then stops the load on a suspected
   unadjusted split or bonus that was not declared; `is_split_error` tells the choice
   of the base interval not to try another interval for it.
2. `run_warnings` puts what the split check made of the declarations in the result's
   warnings.
3. A declared split or bonus that the bars are not adjusted for (`unadjusted_events`)
   is allowed while nothing is held across it. A position, or a GTT resting, at the
   first open of its ex-date would need its units, average price, levels and triggers
   adjusted, which is FR-MKT-11 (Phase 5): `check_crossings` raises NotImplementedError.
   Orders decided before the ex-date that fill at its open are not adjusted: live, they
   would reach the exchange at that open as they were sent.
4. `manifest_form` is the argument as the run manifest records it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .data import BarsResult
from .data.corporate_actions import declared
from .data.splits import (
    ACKNOWLEDGED_CHECK,
    ADJUSTED_CHECK,
    CHECK,
    UNADJUSTED,
    UNADJUSTED_CHECK,
)
from .errors import ConfigError, DataQualityError

# The split check's findings a result reports: the errors stop the load instead.
_REPORTED_CHECKS = (UNADJUSTED_CHECK, ADJUSTED_CHECK, ACKNOWLEDGED_CHECK)
# The GTT events after which a GTT still rests (`exec::gtt_event` in the kernel).
_RESTING_GTT_EVENTS = ("GttPlaced", "GttModified")


@dataclass(frozen=True)
class UnadjustedEvent:
    """A declared split or bonus the bars of an instrument are not adjusted for.

    instrument: the registration index; symbol: its symbol. ex_date: ISO IST date.
    ex_open_ns: UTC nanoseconds of the first bar open on the ex-date. reading: the
    declaration in words.
    """

    instrument: int
    symbol: str
    ex_date: str
    ex_open_ns: int
    reading: str


def declared_for(argument: Any, symbol: str, symbols: Sequence[str]) -> Any:
    """The declarations of `symbol` in the run's `corporate_actions=`, as `load_bars`
    takes them (the data layer checks the items). Raises ConfigError for a dict that
    names a symbol the run does not trade, and for a list in a run of several symbols."""
    if argument is None:
        return None
    if isinstance(argument, dict):
        unknown = sorted(set(argument) - set(symbols))
        if unknown:
            raise ConfigError(
                f"corporate_actions names {unknown}, which the run does not trade ({list(symbols)})"
            )
        return argument.get(symbol)
    if len(symbols) > 1:
        raise ConfigError(
            "a run of several symbols takes corporate_actions as a dict by symbol: "
            "{symbol: [CorporateAction, ...]}"
        )
    return argument


def is_split_error(error: Exception) -> bool:
    """True for a DataQualityError raised for a suspected unadjusted split or bonus. It
    is in the share's prices at every interval, so loading another interval cannot help."""
    report = getattr(error, "report", None)
    return isinstance(error, DataQualityError) and report is not None and report.count(CHECK) > 0


def run_warnings(symbols: Sequence[str], loaded: Sequence[BarsResult]) -> list[str]:
    """What the split check made of each symbol's declarations, one sentence per check,
    in registration order."""
    return [
        f"{symbol}: {issue.message}"
        for symbol, result in zip(symbols, loaded, strict=True)
        for issue in result.meta.quality.issues
        if issue.check in _REPORTED_CHECKS
    ]


def unadjusted_events(
    symbols: Sequence[str], loaded: Sequence[BarsResult]
) -> tuple[UnadjustedEvent, ...]:
    """The declared splits and bonuses each instrument's bars are not adjusted for."""
    return tuple(
        UnadjustedEvent(index, symbol, gap.ex_date, gap.ex_open_ns, gap.reading)
        for index, (symbol, result) in enumerate(zip(symbols, loaded, strict=True))
        for gap in result.meta.quality.price_gaps
        if gap.status == UNADJUSTED
    )


def _position_open(levels: dict[str, list], instrument: int, before_ns: int) -> bool:
    """True when the position history leaves `instrument` with units in any product at
    the end of the last slice opening before `before_ns`."""
    last: dict[str, float] = {}
    rows = zip(
        levels["slice_ts"],
        levels["instrument"],
        levels["product"],
        levels["quantity_units"],
        strict=True,
    )
    for slice_ts, index, product, units in rows:
        if index == instrument and slice_ts < before_ns:
            last[product] = units
    return any(units != 0.0 for units in last.values())


def _gtt_resting(events: list[dict], instrument: int, before_ns: int) -> bool:
    """True when a GTT of `instrument` placed in a slice opening before `before_ns` has
    neither fired nor been cancelled by the end of that slice."""
    latest: dict[int, str] = {}
    for event in events:
        ours = event["type"] == "gtt" and event["instrument"] == instrument
        if ours and event["slice_ts"] < before_ns:
            latest[event["gtt_id"]] = event["name"]
    return any(name in _RESTING_GTT_EVENTS for name in latest.values())


def check_crossings(output: dict[str, Any], events: Sequence[UnadjustedEvent]) -> None:
    """Raise NotImplementedError when a finished run held a position, or had a GTT
    resting, across an ex-date its bars are not adjusted for (item 3 of the module
    docstring)."""
    for event in events:
        if _position_open(output["levels"], event.instrument, event.ex_open_ns):
            held = "a position is held"
        elif _gtt_resting(output["events"], event.instrument, event.ex_open_ns):
            held = "a GTT rests"
        else:
            continue
        raise NotImplementedError(
            f"{event.symbol}: {held} across {event.ex_date}, the ex-date of {event.reading} "
            "that the bars are not adjusted for. Adjusting the units, average price, levels "
            "and GTTs of a holding at a corporate action is FR-MKT-11, planned for Phase 5. "
            "Supply bars adjusted for it (data=), or run on dates that do not span it"
        )


def manifest_form(argument: Any) -> Any:
    """The run's `corporate_actions=` as JSON types: None, a list of declarations, or
    a dict of them by symbol in symbol order."""
    if argument is None:
        return None
    if isinstance(argument, dict):
        return {
            symbol: [action.to_dict() for action in declared(argument[symbol])]
            for symbol in sorted(argument)
        }
    return [action.to_dict() for action in declared(argument)]


__all__ = [
    "UnadjustedEvent",
    "check_crossings",
    "declared_for",
    "is_split_error",
    "manifest_form",
    "run_warnings",
    "unadjusted_events",
]

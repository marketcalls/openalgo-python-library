"""The monthly contracts of a futures underlying, before the run picks the ones of its
rank (`_futures`), and the record of which of them the run relies on.

The contracts come from `instrument=` when the caller gives them for the underlying,
else from the reference tables' monthly expiry schedule, listed from the run's start to
LOOKAHEAD_DAYS past its end (final dates, so a re-dated contract keeps the last trading
day it really had). That look-ahead names contracts the run may never use, so it is
listed without recording any row: a table checked up to some date warns only when a row
is used for a later date, and an unused contract must not warn. The rows the run does
use are recorded afterwards: the expiries that set a hand-over inside the run
(`record_expiries`), and the contracts it trades for the dates it trades them (their
terms, read by `_futures`).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from . import reference
from .errors import ReferenceDataError
from .reference import Instrument, SymbolError, format_symbol, parse_symbol

# Calendar days past the run's end for which the reference tables are asked for
# contracts: a monthly contract expires within about five weeks of any date, and the
# next-month rank needs one contract more.
LOOKAHEAD_DAYS = 75


@dataclass(frozen=True)
class Contract:
    """A contract before its terms: its symbol, last trading day and, when the caller
    gave it, its instrument."""

    symbol: str
    last_day: dt.date
    instrument: Instrument | None


def _given(given: Any) -> list[Instrument]:
    if isinstance(given, Instrument):
        return [given]
    if isinstance(given, dict):
        return [value for value in given.values() if isinstance(value, Instrument)]
    return []


def _underlying_of(instrument: Instrument) -> str | None:
    if instrument.underlying:
        return instrument.underlying
    try:
        return parse_symbol(instrument.symbol, instrument.exchange).underlying
    except SymbolError:
        return None


def _last_day(instrument: Instrument) -> dt.date:
    return instrument.last_trading_day or instrument.expiry


def listed_contracts(name: str, plan: Any, given: Any, calendar: Any) -> list[Contract]:
    """The underlying's monthly contracts, nearest first (see the module docstring).
    Raises ReferenceDataError when the reference tables cannot list them."""
    mine = [
        i
        for i in _given(given)
        if i.kind == "future" and i.exchange == plan.exchange and _underlying_of(i) == name
    ]
    if mine:
        mine.sort(key=_last_day)
        return [Contract(i.symbol, _last_day(i), i) for i in mine]
    tables = reference.default_tables()
    last = plan.end + dt.timedelta(days=LOOKAHEAD_DAYS)
    try:
        found = tables.expiry_schedule(
            name, plan.exchange, plan.start, last, calendar, series="monthly", record_use=False
        )
    except ReferenceDataError as exc:
        raise ReferenceDataError(
            f"{name} on {plan.exchange}: the reference tables cannot list its monthly "
            f"futures ({exc}). Pass the contracts with instrument={{symbol: Instrument(...)}} "
            "giving each one's expiry and last trading day"
        ) from None
    return [
        Contract(format_symbol(name, item.expiry), item.last_trading_day, None) for item in found
    ]


def record_expiries(
    name: str,
    contracts: list[Contract],
    chosen: list[tuple[Contract, dt.date | None]],
    rank: int,
    roll: int | None,
    plan: Any,
    calendar: Any,
) -> None:
    """Record in the reference tables the last trading day of every listed contract the
    run relies on, so that a row used past the tables' check date warns: the nearest
    contract (`rank` places before the one held) whose expiry sets a hand-over inside
    the run, and with `roll=None` a contract settled at its expiry inside the run.
    `chosen` holds the contracts the run holds and the session each hands over on
    (None for the last)."""
    relied = [
        contracts[contracts.index(held) - rank]
        for held, day in chosen
        if day is not None or (roll is None and held.last_day <= plan.end)
    ]
    tables = reference.default_tables()
    for contract in relied:
        if contract.instrument is None:
            tables.expiry_schedule(
                name, plan.exchange, contract.last_day, contract.last_day, calendar, "monthly"
            )


__all__ = ["LOOKAHEAD_DAYS", "Contract", "listed_contracts", "record_expiries"]

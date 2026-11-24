"""The positional exit day of each instrument of a run (design 20-positional-live section 9;
decision 11.2, extended by the lead to options and MCX delivery contracts): the IST date
on which the kernel exits every position still open on a derivatives contract, at the
exchange's square-off time, before the exchange settles it by delivery (exit reason
`expiry_exit`; rust/oa_bt/src/engine/expiry.rs). The kernel holds no market facts: the
day is worked out here from the reference tables and passed with the instrument.

- An option, on any exchange: its last trading day.
- An NFO or BFO future settled by delivery (stock futures, `settlement.csv`): its last
  trading day.
- An MCX future settled by delivery: the trading day before its tender period, the
  `tender_days` of `mcx_contracts.csv` trading days that end on its last trading day.
- An MCX delivery future the table gives no tender period for: none. It settles at expiry
  as before, and the run warns, naming the contract and the table.
- A future settled in cash (index futures, MCX cash contracts, currency futures): none. It
  settles at expiry (`settlement.rs`).
- Cash equity: none.

A figure from an unverified row is used with the reference tables' `ReferenceDataWarning`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import ReferenceDataError
from .reference import Instrument, default_tables

#: The exchange whose delivery contracts end in a tender period.
MCX = "MCX"
#: The exchanges whose futures settled by delivery are exited on their last trading day.
LISTED_DERIVATIVES = ("NFO", "BFO")
#: The settlement of a contract settled by delivery (`reference.instruments`).
PHYSICAL = "physical"
#: Calendar days searched back for the sessions of a tender period: far more than any
#: tender period and the holidays inside it.
SEARCH_DAYS = 60
ONE_DAY = dt.timedelta(days=1)


@dataclass(frozen=True)
class PositionalExits:
    """Each instrument's positional exit day, in registration order (None where it has
    none), and the warnings of the delivery contracts left to settle at expiry."""

    days: tuple[dt.date | None, ...]
    warnings: tuple[str, ...]


def _last_trading_day(instrument: Instrument) -> dt.date | None:
    return instrument.last_trading_day or instrument.expiry


def _day_before_tender(calendar: Any, last_day: dt.date, tender_days: int) -> dt.date | None:
    """The MCX trading day before the `tender_days` trading days that end on `last_day`
    (its last trading day, counted as the first); None when the calendar has too few
    sessions before it."""
    counted, day = 0, last_day
    for _ in range(SEARCH_DAYS):
        if calendar.is_open(MCX, day):
            counted += 1
            if counted > tender_days:
                return day
        day -= ONE_DAY
    return None


def _no_tender_warning(symbol: str) -> str:
    return (
        f"{symbol} is settled by delivery, and the reference table mcx_contracts gives no "
        "tender period for it: a position held into its tender period is settled at expiry "
        "in cash, where MCX would mark it for delivery. Exit it before its tender period, or "
        "give its tender_days (the trading days of the tender period, the expiry day "
        "included) in a user file of the mcx_contracts table"
    )


def _mcx_exit_day(
    instrument: Instrument, trade_day: dt.date, calendar: Any
) -> tuple[dt.date | None, str | None]:
    """(exit day, warning) of an MCX future settled by delivery (module docstring)."""
    last_day = _last_trading_day(instrument)
    underlying = instrument.underlying or instrument.symbol
    tender_days = None
    if last_day is not None and instrument.expiry is not None:
        try:
            tender_days = default_tables().tender_days(underlying, trade_day, instrument.expiry)
        except ReferenceDataError:
            tender_days = None
    if tender_days is None or last_day is None:
        return None, _no_tender_warning(instrument.symbol)
    day = _day_before_tender(calendar, last_day, tender_days)
    if day is None:
        return None, _no_tender_warning(instrument.symbol)
    return day, None


def exit_day(
    instrument: Instrument, trade_day: dt.date, calendar: Any
) -> tuple[dt.date | None, str | None]:
    """(positional exit day, warning) of `instrument` traded from `trade_day`, on the
    session `calendar` (module docstring). Raises nothing: a delivery contract the tables
    do not cover is left to settle at expiry, with a warning."""
    if instrument.kind == "option":
        return _last_trading_day(instrument), None
    if instrument.kind != "future" or instrument.settlement != PHYSICAL:
        return None, None
    if instrument.exchange in LISTED_DERIVATIVES:
        return _last_trading_day(instrument), None
    if instrument.exchange == MCX:
        return _mcx_exit_day(instrument, trade_day, calendar)
    return None, None


def positional_exits(
    instruments: Sequence[Instrument], trade_day: dt.date, calendar: Any
) -> PositionalExits:
    """The positional exit day of each of `instruments` (registration order), traded from
    `trade_day` on the session `calendar`, and the warnings of the delivery contracts the
    tables give no tender period for."""
    found = [exit_day(instrument, trade_day, calendar) for instrument in instruments]
    days = tuple(day for day, _ in found)
    warnings = tuple(warning for _, warning in found if warning is not None)
    return PositionalExits(days, warnings)


__all__ = ["PositionalExits", "exit_day", "positional_exits"]

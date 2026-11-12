"""Tick sizes chosen by a price slab, resolved from the run's data (04-execution 1; the
NSE and BSE circulars cited in `reference/data/tick_sizes.csv`).

On NFO and BFO some contracts' tick is fixed once a month by a reference close: an index
future's by the index close on the last trading day of the previous month, a stock
future's or option's by the underlying share's close that day (the slab's `basis`). The
reference tables hold the slabs but not that close, so they leave the tick unknown; a
backtest reads the close from its data instead of asking for `instrument=`.

For every month in which a contract is traded in the run, the close the circular names
is loaded through the data layer from the run's own source, on the exchange's last
session of the previous month:

1. the underlying's close (an index on NSE_INDEX or BSE_INDEX, a share on NSE or BSE),
   when the source has bars of it;
2. otherwise the contract's own close that day, which stands in for it: a future trades
   within a small basis of its underlying, far smaller than the width of a slab. The
   Assumptions line of the instrument says which close was used.

The engine holds one tick per contract for the run, so every month must give the same
tick. ConfigError is raised when no close of either series exists in the data, or when
the tick would change during the run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Any

from . import reference
from ._layout import parse_interval
from .data import load_bars
from .errors import ConfigError, NoDataError, ReferenceDataError, TimestampError
from .reference import Instrument

SLAB_EXCHANGES = ("NFO", "BFO")
INDEX_EXCHANGE_OF = {"NFO": "NSE_INDEX", "BFO": "BSE_INDEX"}
CASH_EXCHANGE_OF = {"NFO": "NSE", "BFO": "BSE"}
# Longest run of closed days walked back over to the previous month's last session.
MAX_CLOSED_RUN_DAYS = 15
ONE_MINUTE = "1m"


@dataclass(frozen=True)
class ReferenceClose:
    """A reference close: its IST day, the price in rupees and the series it came from."""

    day: dt.date
    price: float
    series: str


def months(first: dt.date, last: dt.date) -> list[dt.date]:
    """The first day of every month from `first`'s to `last`'s."""
    found, month = [], first.replace(day=1)
    while month <= last:
        found.append(month)
        month = (month + dt.timedelta(days=32)).replace(day=1)
    return found


def last_session_before(day: dt.date, exchange: str, calendar: Any) -> dt.date | None:
    """The exchange's last session before `day`, or None when the calendar has none soon
    before it."""
    for back in range(1, MAX_CLOSED_RUN_DAYS + 1):
        candidate = day - dt.timedelta(days=back)
        if calendar.session(exchange, candidate) is not None:
            return candidate
    return None


class _Source:
    """The run's data source, asked for one day's close of a series."""

    def __init__(self, args: dict[str, Any], client: Any, calendar: Any, interval: str) -> None:
        self._args, self._client, self._calendar = args, client, calendar
        # Own intraday data is read as daily bars resampled from its own interval.
        self._bases = [None, ONE_MINUTE]
        if parse_interval(interval)[0] == "intraday" and interval != ONE_MINUTE:
            self._bases.append(interval)

    def own(self, symbol: str, contract: bool) -> Any:
        """The user's data for a symbol: its entry in a dict by symbol (None when it has
        none); data that is not a dict is the contract's and never its underlying's."""
        data = self._args["data"]
        if isinstance(data, dict):
            return data.get(symbol)
        return data if contract else None

    def close(self, symbol: str, exchange: str, day: dt.date, own: Any) -> float | None:
        """The day's close of a series, or None when the source has no bar that day."""
        if own is None and self._client is None:
            return None
        for base in self._bases:
            try:
                result = load_bars(
                    symbol,
                    exchange,
                    "D",
                    day,
                    day,
                    source=self._args["source"],
                    client=self._client,
                    data=own,
                    calendar=self._calendar,
                    tz=self._args["tz"],
                    base=base,
                )
            except NoDataError:
                return None
            except TimestampError:
                continue
            closes = result.bars["close"]
            return float(closes.iloc[-1]) if len(closes) else None
        return None


def _underlying_series(instrument: Instrument, day: dt.date) -> tuple[str, str] | None:
    """(symbol, exchange) of the underlying whose close the circular names, if known."""
    underlying = instrument.underlying
    if underlying is None:
        return None
    try:
        kind = reference.default_tables().underlying_type(underlying, instrument.exchange, day)
    except ReferenceDataError:
        return None
    exchanges = INDEX_EXCHANGE_OF if kind == "index" else CASH_EXCHANGE_OF
    return underlying, exchanges[instrument.exchange]


def _reference_close(
    instrument: Instrument, day: dt.date, source: _Source
) -> ReferenceClose | None:
    underlying = _underlying_series(instrument, day)
    if underlying is not None:
        symbol, exchange = underlying
        price = source.close(symbol, exchange, day, source.own(symbol, contract=False))
        if price is not None:
            return ReferenceClose(day, price, f"the {symbol} close")
    own = source.own(instrument.symbol, contract=True)
    price = source.close(instrument.symbol, instrument.exchange, day, own)
    if price is None:
        return None
    return ReferenceClose(
        day, price, f"{instrument.symbol}'s own close, standing in for its underlying's"
    )


def _month_tick(
    instrument: Instrument, month: dt.date, first: dt.date, source: _Source, calendar: Any
) -> tuple[float, ReferenceClose]:
    day = last_session_before(month, instrument.exchange, calendar)
    close = None if day is None else _reference_close(instrument, day, source)
    if close is None:
        raise ConfigError(
            f"{instrument.symbol}: its tick size depends on a price slab chosen by a close on "
            f"the last session before {month.isoformat()}, and the data has no close of its "
            f"underlying or of the contract that day; load data that covers it, or pass "
            "instrument=Instrument(..., tick_size=...)"
        )
    found = reference.default_tables().instrument(
        instrument.symbol,
        instrument.exchange,
        max(month, first),
        reference_price=close.price,
        calendar=calendar,
    )
    return found.tick_size, close


def _resolve_one(
    instrument: Instrument, window: tuple[dt.date, dt.date], source: _Source, calendar: Any
) -> tuple[Instrument, str]:
    first, last = window
    found = [
        _month_tick(instrument, month, first, source, calendar) for month in months(first, last)
    ]
    ticks = sorted({tick for tick, _ in found})
    if len(ticks) != 1:
        changes = ", ".join(f"{close.day.isoformat()}: {tick}" for tick, close in found)
        raise ConfigError(
            f"{instrument.symbol}: its tick size changes during the run (by the slab of each "
            f"month's reference close: {changes}); the engine holds one tick per contract, so "
            "split the run at the month boundary or pass instrument=Instrument(..., "
            "tick_size=...)"
        )
    tick, close = found[0]
    freeze = (
        "no freeze quantity" if instrument.freeze_qty is None else f"freeze {instrument.freeze_qty}"
    )
    line = (
        f"{instrument.symbol}: lot {instrument.lot_size}, tick {tick} by the price slab of "
        f"{close.series} on {close.day.isoformat()} ({close.price}), {freeze}, from the "
        "reference tables"
    )
    return dataclasses.replace(instrument, tick_size=tick), line


def resolve_slab_ticks(
    instruments: list[Instrument],
    lines: dict[str, str],
    plan: Any,
    args: dict[str, Any],
    client: Any,
    calendar: Any,
    windows: dict[str, tuple[dt.date, dt.date]] | None,
) -> tuple[list[Instrument], dict[str, str]]:
    """The run's instruments with every slab tick resolved from the data, and their
    Assumptions lines. Raises ConfigError as the module docstring says."""
    source = _Source(args, client, calendar, plan.interval)
    resolved, lines = [], dict(lines)
    for instrument in instruments:
        if instrument.tick_size is not None or instrument.exchange not in SLAB_EXCHANGES:
            resolved.append(instrument)
            continue
        window = (windows or {}).get(instrument.symbol, (plan.start, plan.end))
        instrument, lines[instrument.symbol] = _resolve_one(instrument, window, source, calendar)
        resolved.append(instrument)
    return resolved, lines


__all__ = ["ReferenceClose", "last_session_before", "months", "resolve_slab_ticks"]

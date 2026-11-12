"""The instruments of a run: lot size, tick size, freeze quantity and expiry for each
traded symbol (14-strategy-api 6; 03-architecture 7).

- `instrument=` (an `Instrument`, or a dict of them by symbol) is used as given.
- Cash equity on NSE or BSE without one is the generic cash-equity instrument (lot 1,
  tick 0.05, no freeze quantity, the exchange's equity costs), as 14-strategy-api 6
  fixes. The reference tables hold nothing more for a share: its lot is 1, no freeze
  quantity is modelled, and its tick by price slab needs the previous month's closing
  price, which the run's data does not hold.
- Futures and options are resolved by the reference tables
  (`openalgo.backtest.reference`) as on the run's first date: lot size, freeze
  quantity, expiry and, where one tick applies to every price, the tick. Where the tick
  depends on a price slab it is resolved from the run's data (`_slab_ticks`).

Every instrument gets one line in the Assumptions block that says where its terms
came from.
"""

from __future__ import annotations

from typing import Any

from . import reference
from .errors import ConfigError
from .reference import Instrument

CASH_EXCHANGES = ("NSE", "BSE")


def _terms(instrument: Instrument) -> str:
    freeze = f"freeze {instrument.freeze_qty}"
    if instrument.freeze_qty is None:
        freeze = "no freeze quantity"
    return f"lot {instrument.lot_size}, tick {instrument.tick_size}, {freeze}"


def _given(given: Any) -> dict[str, Instrument]:
    if given is None:
        return {}
    if isinstance(given, Instrument):
        return {given.symbol: given}
    if isinstance(given, dict) and all(isinstance(v, Instrument) for v in given.values()):
        return dict(given)
    raise ConfigError("instrument= takes an Instrument or a dict of them by symbol")


def _from_tables(symbol: str, plan: Any, calendar: Any) -> tuple[Instrument, str]:
    """The instrument from the reference tables. A tick chosen by a price slab is left
    None here and resolved from the run's data (`_slab_ticks`)."""
    tables = reference.default_tables()
    instrument = tables.instrument(symbol, plan.exchange, plan.start, calendar=calendar)
    line = f"{symbol}: {_terms(instrument)} from the reference tables ({tables.version})"
    return instrument, line


def resolve_instruments(
    plan: Any, given: Any, calendar: Any
) -> tuple[list[Instrument], dict[str, str], str | None]:
    """(instruments in registration order, an Assumptions line by symbol, the
    reference-table version if the tables were used). Raises ConfigError for an
    `instrument=` of another exchange or one that cannot be used, and the reference
    tables' errors for a contract they do not cover."""
    by_symbol = _given(given)
    instruments, lines, version = [], {}, None
    for symbol in plan.symbols:
        if symbol in by_symbol:
            instrument = by_symbol[symbol]
            if instrument.exchange != plan.exchange:
                raise ConfigError(
                    f"instrument= for {symbol} is on {instrument.exchange}, the strategy "
                    f"trades on {plan.exchange}"
                )
            line = f"{symbol}: as given, {_terms(instrument)}"
        elif plan.exchange in CASH_EXCHANGES:
            instrument = Instrument.cash_equity(symbol, plan.exchange)
            line = (
                f"{symbol}: generic cash-equity instrument assumed ({_terms(instrument)}, "
                f"{plan.exchange} equity costs); pass instrument= for its own terms"
            )
        else:
            instrument, line = _from_tables(symbol, plan, calendar)
            version = reference.default_tables().version
        instruments.append(instrument)
        lines[symbol] = line
    return instruments, lines, version


__all__ = ["CASH_EXCHANGES", "resolve_instruments"]

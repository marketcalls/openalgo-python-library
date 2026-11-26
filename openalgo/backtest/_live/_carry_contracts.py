"""A carried position whose contract the session does not trade, judged at the start before
the preflight (20-positional-live 9 and 4.2 step 4; the lead's rulings N5a-1 and N5a-3,
review round 1 of unit N5a). Split from `_carry_open` so each module keeps one job.

A strategy's day-close record names the contracts it carries positions on. When the
session's strategy of that name trades another contract:

- **Every carried contract has expired** (its last trading day, from the reference tables
  as of the record's day, is before today): the exchange settled it (an index future, or
  an MCX contract settled in cash). The strategy starts flat on the contract it trades
  now: its carry is dropped, so nothing of the expired contract is looked up in today's
  master contract, restored or traded; its journal GTTs on the expired contracts are
  ended (the broker ends a GTT of a contract that no longer trades); and the settlement
  is named (`CarryAtStart.expired`): a SESSION alert and a warning of the result say the
  contract, its units, its last trading day and its last close the journal holds (the
  record's, of the day it was carried from: the contract note holds the settlement
  price). The close writes the strategy's flat day-close record, so it is named once.
- **A carried contract still trades**: what the strategy carries continues only on its
  own contract (the engine's restore refuses other instruments), so the start is refused
  (PreflightError) saying so, and what to do.

A strategy that still names its expired contract is refused before this, when the plan
resolves the contract for today (the reference tables have no terms for it): the
operator names the contract it trades now, and the start then settles the old one as
above. Days are IST trading days, quantities signed whole units, prices rupees. Run by
the main thread at the start, before the order worker follows any GTT.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .._futures_contracts import last_day
from .._instruments import CASH_EXCHANGES
from ..errors import ConfigError, ReferenceDataError
from ..reference import default_tables
from ._carry_start import Carried, CarryAtStart
from ._journal_gtt_rows import GttState
from ._preflight_carry import CarriedPosition
from .journal import Journal

#: What a start refused for a strategy moved off a contract it carries asks of the
#: operator.
START_ON_ITS_OWN = (
    "close the position and cancel its GTTs in OpenAlgo by hand, then start it under "
    "another account name (account=)"
)


def settle_expired(
    journal: Journal, carry: CarryAtStart, strategies: Sequence[Any]
) -> tuple[CarryAtStart, list[str]]:
    """Judge each carried strategy of `strategies` (each a `plan.StrategySpec`) whose
    record carries a contract it does not trade now (module docstring): (the carry
    without those the exchange settled, with their words; a refusal for each other).
    Raises JournalError when the journal cannot end a settled contract's GTTs."""
    refusals: list[str] = []
    expired = dict(carry.expired)
    kept = dict(carry.carried)
    for spec in strategies:
        carried = carry.of(spec.name)
        if carried is None:
            continue
        elsewhere = [
            p for p in carried.positions if (p.symbol, p.exchange) != (spec.symbol, spec.exchange)
        ]
        if not elsewhere:
            continue
        ended = {p: _last_day(p, carried) for p in elsewhere}
        trading = [p for p, last in ended.items() if last is None or last >= carry.day]
        if trading:
            refusals.append(_moved(spec, carried, trading[0]))
            continue
        _end_gtts(journal, spec.name, {(p.symbol, p.exchange) for p in elsewhere}, carry.day)
        del kept[spec.name]
        expired[spec.name] = "; ".join(_settled(spec, carried, p, ended[p]) for p in elsewhere)
    return replace(carry, carried=kept, expired=expired), refusals


def _last_day(position: CarriedPosition, carried: Carried) -> dt.date | None:
    """The last trading day of a carried position's contract, from the reference tables
    as of the day it was carried from; None for a cash instrument, or one the tables do
    not resolve (the operator's own terms: taken as still trading)."""
    if position.exchange in CASH_EXCHANGES:
        return None
    try:
        instrument = default_tables().instrument(
            position.symbol, position.exchange, carried.record.day
        )
    except (ConfigError, ReferenceDataError):
        return None
    return None if instrument.expiry is None else last_day(instrument)


def _end_gtts(journal: Journal, name: str, contracts: set[tuple[str, str]], day: dt.date) -> None:
    """End strategy `name`'s journal GTTs on `contracts` that have not ended: the
    exchange settled each contract before `day`."""
    for gtt in journal.open_gtts():
        if gtt.strategy == name and (gtt.symbol, gtt.exchange) in contracts:
            note = f"its contract expired before {day}: the exchange settled it"
            state = GttState.ENDED if gtt.trigger_id is not None else GttState.NOT_PLACED
            journal.set_gtt_state(gtt.seq, state, note=note)


def _settled(spec: Any, carried: Carried, position: CarriedPosition, last: dt.date) -> str:
    """The words of a carried position the exchange settled (module docstring)."""
    return (
        f"{spec.name}: {position.units:g} {position.symbol} {position.exchange} "
        f"({position.product}) carried from {carried.record.day} expired on {last}: the "
        f"exchange settled it (its last close in the journal {_last_close(carried, position)}, "
        f"of {carried.record.day}; the contract note holds the settlement price), so it starts "
        f"flat on {spec.symbol} {spec.exchange}"
    )


def _last_close(carried: Carried, position: CarriedPosition) -> str:
    """The record's last close of a carried position's contract, in rupees, or "not
    kept"."""
    state = carried.record.carried
    for row in state.get("last_closes", []):
        instrument = state["instruments"][int(row["instrument"])]
        if (instrument["symbol"], instrument["exchange"]) == (position.symbol, position.exchange):
            return f"{float(row['close_rupees']):.2f}"
    return "not kept"


def _moved(spec: Any, carried: Carried, position: CarriedPosition) -> str:
    """The refusal of a strategy moved off a carried contract that still trades."""
    return (
        f"{spec.name}: what it carries from {carried.record.day} is {position.units:g} "
        f"{position.symbol} {position.exchange} ({position.product}), and this start trades "
        f"{spec.symbol} {spec.exchange}: a carried position continues only on its own "
        f"contract, so start it on {position.symbol} {position.exchange}; or "
        f"{START_ON_ITS_OWN}"
    )


__all__ = ["START_ON_ITS_OWN", "settle_expired"]

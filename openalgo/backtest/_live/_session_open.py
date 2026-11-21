"""The positions a live session's close lists open (16-session-design 1.4 `open_positions`,
5.4 "As built" rounds 7 and 8, 7.1; 18-multi-strategy 4.11; moved from `_session_close` in
unit M7).

- **A strategy's leg** the engine holds open, at the engine's units, or, when its
  difference with OpenAlgo's books no later read confirms (`_reconcile_close.differing`),
  at OpenAlgo's units then, moved by the engine's own fills since; a leg those units leave
  flat is no open position (round 8). A member of a shared instrument is listed at its own
  units: no member's share of the group's net is known. An entry the order book cannot
  tell apart from another order adds its units, with no stop confirmed (F60). Each row has
  its broker-held stop and whether it was confirmed (`_stops_report.stop_state`) and a note
  saying what to do.
- **An instrument several strategies share**, after the legs, when OpenAlgo's net is a
  position, or flat while a member is listed open at units their sum does not match
  (`_session_named.SharedPosition.listed`): its members, OpenAlgo's net for them together
  (not a position of its own) and whether it equals their sum.

Quantities are signed units. Read while the kernel sessions run: a finished one has no
units to read. Used by the close only (`_session_close`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._group_rules import and_join
from ._reconcile_close import differing
from ._session_loop import Session
from ._session_named import SharedPosition
from ._stops_report import stop_state

#: How an open position names the units of an entry the order book cannot tell apart from
#: another order (16-session-design F60): the session cannot say which is its own.
AMBIGUOUS_ENTRY_NOTE = (
    "an entry of {units} units cannot be told apart in the order book from another order: "
    "one of them is the session's, at the broker with no stop; check the order book in "
    "OpenAlgo and close it by hand"
)


@dataclass(frozen=True)
class OpenPosition:
    """A position still open when the session ended: the strategy, the instrument, its
    signed units (OpenAlgo's at the last read when they differed from the engine's and no
    later read confirmed it, moved by the engine's own fills since: `_reconcile_close`), a
    note for the operator, its stop's broker order ID ("" for none, or one not bound) and
    whether that stop was confirmed working at the broker for no more units than the
    position holds (one larger, which a trigger would take through zero, is not, nor any
    while OpenAlgo's books show other units than the engine's:
    `_stops_report.stop_state`).

    The row of an instrument several strategies share names its `members` ("A and B" as
    its strategy), and its units are OpenAlgo's net for them together, not a position of
    its own; `members` is empty for a strategy's leg."""

    strategy: str
    symbol: str
    exchange: str
    units: float
    note: str
    stop_order_id: str = ""
    stop_confirmed: bool = False
    members: tuple[str, ...] = ()


def open_positions(
    session: Session, shared: tuple[SharedPosition, ...]
) -> tuple[OpenPosition, ...]:
    """Each leg the close lists open, then each `shared` instrument it lists (module
    docstring). `session` is the `_session_loop.Session`."""
    stops, positions = session.stops, []
    for trader in session.traders:
        differs = differing(session.reconcile, trader)
        units = trader.open_units() if differs is None else differs.open_units
        unsure = stops.leg(trader).ambiguous_entry_units()
        if units == 0 and unsure == 0:
            continue
        note = (
            "known as of the mode breach" if session.locked else stops.after.note(session, trader)
        )
        stop = stop_state(stops, trader)
        if unsure:
            words = AMBIGUOUS_ENTRY_NOTE.format(units=abs(unsure))
            note, stop = (f"{note}; {words}" if units else words), (stop[0], False)
        positions.append(
            OpenPosition(trader.name, trader.symbol, trader.exchange, units + unsure, note, *stop)
        )
    rows = [
        _instrument_row(session, each, positions)
        for each in shared
        if each.listed(any(_of(p, each) for p in positions))
    ]
    return (*positions, *rows)


def _of(position: OpenPosition, shared: SharedPosition) -> bool:
    """Whether `position` is the leg of a member of the `shared` instrument."""
    return position.strategy in shared.members and position.symbol == shared.symbol


def _instrument_row(
    session: Session, shared: SharedPosition, legs: list[OpenPosition]
) -> OpenPosition:
    """The row of a shared instrument (18-multi-strategy 4.11): its members, OpenAlgo's
    net and the note (`SharedPosition.note`); no stop of its own, and its stop confirmed
    when the net, read in time, equals the members' sum and each member's row has its stop
    confirmed."""
    rows = [p for p in legs if _of(p, shared)]
    covered = shared.agrees and bool(rows) and all(p.stop_confirmed for p in rows)
    note = shared.note()
    if session.locked:
        note = f"known as of the mode breach; {note}"
    whose, units = and_join(shared.members), shared.net or 0
    return OpenPosition(
        whose, shared.symbol, shared.exchange, units, note, "", covered, shared.members
    )


__all__ = ["AMBIGUOUS_ENTRY_NOTE", "OpenPosition", "open_positions"]

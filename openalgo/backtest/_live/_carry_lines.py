"""The start banner's lines of what a session continues from earlier days
(20-positional-live 4.2 steps 1 and 2, 4.3; the lead's rulings N5a-1 and N5a-3), after the
restart's lines and before the funds:

- each continued leg: the day its record was written, its units and its stop GTT as the
  stops supervisor knows it once the start phase is over (`continued_line`); a
  `positional_stops="rearm_daily"` leg's stop level and whether its day stop is placed yet,
  as the start's warning of such legs says it (`_session_rearm.stop_at_start`);
- the trading days of its exchange since that record on which no session ran: a stop GTT
  that fired on one is recorded at its child's limit, and targets reached then were
  missed (`missed_line`); a rearm_daily leg has no stop GTT, and N5b's start warning
  names those days for it (`_session_rearm`), so it has no such line;
- a strategy whose version changed since its record: carried all the same, its leg keeps
  the levels and rules saved with it (`version_line`, review L4);
- each position the other mode's ledger carries, named and never restored
  (`_preflight_carry.other_ledger_lines`: "... held in sandbox, not continued in live");
- each strategy whose holdings of an earlier day no record carried were found closed at
  the broker, so it starts flat (`_carry_unrecorded`).

Days are IST trading days. Used by the loop at the start.
"""

from __future__ import annotations

from typing import Any

from ..calendar import default_calendar
from ._carry_start import Carried
from ._preflight_carry import missed_days, other_ledger_lines
from ._session_rearm import rearm_daily, stop_at_start
from ._session_trader import Trader
from ._stops_report import stop_words


def carry_lines(session: Any) -> list[str]:
    """The banner's lines of what `session` (a `_session_loop.Session`) continues from
    earlier days (module docstring); none when it continues nothing."""
    carry = session.parts.carry
    if carry is None:
        return []
    lines = []
    for trader in session.traders:
        carried = carry.of(trader.name)
        if carried is None:
            continue
        lines.append(continued_line(session, trader, carried))
        missed = "" if rearm_daily(trader) else missed_line(trader, carried, carry.day)
        lines += [missed] if missed else []
        version = version_line(trader, carried)
        lines += [version] if version else []
    lines += other_ledger_lines(carry.other, carry.mode)
    return [*lines, *carry.settled]


def continued_line(session: Any, trader: Trader, carried: Carried) -> str:
    """The line of a continued leg (module docstring)."""
    units = trader.open_units()
    stop = (
        stop_at_start(session, trader) if rearm_daily(trader) else stop_words(session.stops, trader)
    )
    held = f"{units:g} {trader.symbol} {trader.exchange}, {stop}" if units else "flat now"
    return f"{trader.name}: continued from its day-close record of {carried.record.day}: {held}"


def missed_line(trader: Trader, carried: Carried, today: Any) -> str:
    """The line naming the trading days no session of `trader` ran since its record, or
    "" for none (module docstring)."""
    days = missed_days(trader.exchange, carried.record.day, today, default_calendar())
    if not days:
        return ""
    named = ", ".join(day.isoformat() for day in days)
    return (
        f"{trader.name}: no session ran on {named}, since its day-close record of "
        f"{carried.record.day}: a stop GTT that fired then is recorded at its child's limit, "
        "flagged price_unverified and missed_day, and targets reached then were missed"
    )


def version_line(trader: Trader, carried: Carried) -> str:
    """The line of a strategy whose version changed since its record, or "" (module
    docstring)."""
    now, then = trader.part.spec.version.version_id, carried.record.version_id
    if now == then:
        return ""
    return (
        f"{trader.name}: its version changed since its day-close record of "
        f"{carried.record.day}, from {then} to {now}: the position it carries keeps the levels "
        "and rules saved with it, and new parameters apply to new entries"
    )


__all__ = ["carry_lines", "continued_line", "missed_line", "version_line"]

"""Each strategy's `Trader` at the start of a live session (16-session-design 3.1 step 11;
unit 8), split from `_session_start` so each module keeps one job: its venue map, numbered
with the session's request IDs, its square-off and its exchange's close, its sizing capped
by the funds available, and the order worker's record of locked entries
(`_locked_entries`), which its entry locks are copied to.

Times are UTC nanoseconds, amounts rupees.
"""

from __future__ import annotations

import itertools

from .._engine import call
from ..calendar import default_calendar
from ._session_loop import Session
from ._session_trader import Trader, ist_instant
from .venue_map import VenueMap


def build_traders(session: Session, funds_available: float) -> tuple[Trader, ...]:
    """Each strategy's `Trader` (module docstring), in the plan's order; the caller locks
    their entries until trading begins."""
    parts = session.parts
    calendar, day = default_calendar(), parts.spec.day
    numbers = itertools.count(1)
    traders = []
    for part in parts.plan.strategies:
        venue = VenueMap(
            part.engine,
            strategy=part.spec.name,
            instruments=part.setup.instruments,
            session=parts.journal_session,
            request_ids=numbers,
            alerts=parts.alerts,
            now=session.now,
        )
        times = calendar.intraday_times(part.spec.exchange, day)
        square_off = part.spec.plan.square_off
        square_off_ns = times.square_off if square_off is None else ist_instant(day, square_off)
        trader = Trader(
            part,
            venue,
            square_off_ns=square_off_ns,
            close_ns=times.session_close,
            now=session.now,
            event=session.event,
            decision=parts.logs.decision,
            warn=session.warn,
            locked_entries=parts.worker.locked_entries,
        )
        call(part.engine.cap_sizing_equity, float(funds_available))
        traders.append(trader)
    return tuple(traders)


__all__ = ["build_traders"]

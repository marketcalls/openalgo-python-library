"""Each strategy of a live session as the loop drives it (16-session-design 3.1 step 11; unit
8): its `Trader`, with its venue map numbered with the session's request IDs and carrying
the strategy's version id onto every intent (18-multi-strategy 12.1), its entries locked
until trading begins (`STARTING_LOCK`), and its sizing capped by the funds OpenAlgo reported
at the start (R18). Part of the start (`_session_start`). Amounts are rupees, times UTC
nanoseconds.
"""

from __future__ import annotations

import itertools
from typing import Any

from .._engine import call
from ..calendar import default_calendar
from ._session_trader import Trader, ist_instant
from .venue_map import VenueMap

#: The entry lock of the start phase (3.1 steps 11 to 15).
STARTING_LOCK = "starting"


def traders(session: Any, funds_available: float) -> tuple[Trader, ...]:
    """Each strategy's `Trader` of `session` (a `_session_loop.Session` whose plan is
    built), in call order (module docstring); `funds_available` is the rupees OpenAlgo
    reported available at the start."""
    parts = session.parts
    calendar, day = default_calendar(), parts.spec.day
    numbers = itertools.count(1)
    made = []
    for part in parts.plan.strategies:
        venue = VenueMap(
            part.engine,
            strategy=part.spec.name,
            instruments=part.setup.instruments,
            session=parts.journal_session,
            request_ids=numbers,
            alerts=parts.alerts,
            now=session.now,
            version_id=part.spec.version.version_id,
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
        )
        trader.lock(STARTING_LOCK)
        call(part.engine.cap_sizing_equity, float(funds_available))
        made.append(trader)
    return tuple(made)


__all__ = ["STARTING_LOCK", "traders"]

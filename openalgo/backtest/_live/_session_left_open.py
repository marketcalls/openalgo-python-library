"""The lines of the legs a live session leaves open (16-session-design 3.3 step 4, 3.4;
20-positional-live 6 and 7.3; review findings 6 c and 7 of unit N3b), split from
`_session_stop` so each module keeps one job.

One console line per leg left open, a carried CNC or NRML leg's at the session's end among
them, with what holds it: its stop (a CNC or NRML leg's stop GTT by its trigger ID, and
whether the close's read of the GTT book confirmed it) and what works toward its exit
(`_stops_report.left_open`). An ERROR alert goes for each leg the operator must act on: one
without a confirmed stop covering it, or with another order working; a leg its stop
protects is information only. At the exchange's close plus the grace (`every_leg`) an MIS
position still open is the operator's whatever its stop; a carried leg its stop GTT
protects is not, and one the close's read did not confirm has had its own alert
(`_session_carry`). A `positional_stops="rearm_daily"` leg with no stop and nothing
working toward it is left so by design until the next start places its day stop: its line
says so (`REARM_LEFT_OPEN`), and the close's WARN alert names it (`_session_rearm`), so it
has no ERROR alert of its own; nor has a future the exchange settles today, whose stop GTT
the kernel cancelled at the close: its line says the exchange settles it
(`_session_expiry`).

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from ._session_expiry import settles_today, settles_words
from ._session_rearm import day_orders, rearm_daily
from ._session_trader import Trader
from ._stops_gtt import overnight
from ._stops_report import left_open, leg_words, protected
from ._types import LOGGER_NAME, AlertKind

logger = logging.getLogger(LOGGER_NAME)

#: The line of a rearm_daily leg left with no stop and nothing working (module docstring).
REARM_LEFT_OPEN = (
    "no stop at the broker until your next start, which places its day stop at the open "
    '(positional_stops="rearm_daily")'
)


def left_open_lines(session: Any, *, every_leg: bool) -> None:
    """The lines and alerts of the legs `session` (a `_session_loop.Session`) leaves open
    (module docstring); `every_leg` at the exchange's close plus the grace."""
    for trader in session.traders:
        units = trader.open_units()
        if units == 0:
            continue
        settles = settles_today(trader, session.parts.spec.day) and not trader.working()
        bare = settles or _rearm_bare(session, trader)
        note = left_open(session.stops, trader)
        if bare:
            note = settles_words(trader) if settles else REARM_LEFT_OPEN
        line = f"{leg_words(trader, units)}: {note}"
        session.emit(line)
        if not bare and _must_act(session, trader, every_leg=every_leg):
            session.alert(AlertKind.ERROR, line, echo=False)
        else:
            logger.info(line)


def _rearm_bare(session: Any, trader: Trader) -> bool:
    """Whether `trader` is a rearm_daily leg with no stop at the broker and no DAY order
    working toward it (module docstring; the strategy's own GTT orders rest overnight by
    design, `_session_rearm.day_orders`)."""
    if not rearm_daily(trader) or day_orders(trader):
        return False
    return session.stops.covered(trader) == 0 and session.stops.broker_units(trader) is None


def _must_act(session: Any, trader: Trader, *, every_leg: bool) -> bool:
    """Whether the operator must act on `trader`'s leg left open (module docstring)."""
    if overnight(trader):
        alerted = session.news.carry.alerted(trader.name)
        return not alerted and not protected(session.stops, trader)
    return every_leg or not protected(session.stops, trader)


__all__ = ["REARM_LEFT_OPEN", "left_open_lines"]

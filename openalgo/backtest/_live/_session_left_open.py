"""The lines of the legs a live session leaves open (16-session-design 3.3 step 4, 3.4;
20-positional-live 7.3; review findings 6 c and 7 of unit N3b), split from `_session_stop`
so each module keeps one job.

One console line per leg left open, a carried CNC or NRML leg's at the session's end among
them, with what holds it: its stop (a CNC or NRML leg's stop GTT by its trigger ID, and
whether the close's read of the GTT book confirmed it) and what works toward its exit
(`_stops_report.left_open`). An ERROR alert goes for each leg the operator must act on: one
without a confirmed stop covering it, or with another order working; a leg its stop
protects is information only. At the exchange's close plus the grace (`every_leg`) an MIS
position still open is the operator's whatever its stop; a carried leg its stop GTT
protects is not, and one the close's read did not confirm has had its own alert
(`_session_carry`).

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from ._session_trader import Trader
from ._stops_gtt import overnight
from ._stops_report import left_open, leg_words, protected
from ._types import LOGGER_NAME, AlertKind

logger = logging.getLogger(LOGGER_NAME)


def left_open_lines(session: Any, *, every_leg: bool) -> None:
    """The lines and alerts of the legs `session` (a `_session_loop.Session`) leaves open
    (module docstring); `every_leg` at the exchange's close plus the grace."""
    for trader in session.traders:
        units = trader.open_units()
        if units == 0:
            continue
        line = f"{leg_words(trader, units)}: {left_open(session.stops, trader)}"
        session.emit(line)
        if _must_act(session, trader, every_leg=every_leg):
            session.alert(AlertKind.ERROR, line, echo=False)
        else:
            logger.info(line)


def _must_act(session: Any, trader: Trader, *, every_leg: bool) -> bool:
    """Whether the operator must act on `trader`'s leg left open (module docstring)."""
    if overnight(trader):
        alerted = session.news.carry.alerted(trader.name)
        return not alerted and not protected(session.stops, trader)
    return every_leg or not protected(session.stops, trader)


__all__ = ["left_open_lines"]

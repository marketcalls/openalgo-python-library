"""The ERROR alert of a live session whose mode guard locked (16-session-design 6.2;
20-positional-live 10.4; the lead's rulings 9b and 9c of review round 4 of unit N5a), split
from `_session_loop` so each module keeps one job.

The session sends nothing more once the guard locks, so the alert names each position as
last known with its stop, what becomes of them by the product they are held in, and each
daily or longer strategy's pre-open decision whose orders, waiting for the open, the switch
drops before they reach the broker (`_session_pre_open.dropped_unsent`); each such line is
a warning of the result too. The broker's MIS square-off is named only in a session that
runs an MIS strategy (`_guard_breach.breach_alert`).

Quantities are signed units. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from ._guard_breach import breach_alert
from ._session_pre_open import dropped_unsent
from ._stops_gtt import overnight
from ._stops_report import stop_words
from ._types import AlertKind


def alert_breach(session: Any) -> None:
    """The ERROR alert of `session` (a `_session_loop.Session`) whose guard has just locked,
    and a warning of each pre-open decision the switch drops (module docstring)."""
    positions = [
        f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units, "
        f"{stop_words(session.stops, trader)}"
        for trader in session.traders
        if (units := trader.open_units()) != 0
    ]
    products = {trader.product for trader in session.traders if trader.open_units()}
    mode = session.parts.mode
    dropped = [line for t in session.traders if (line := dropped_unsent(t, mode)) is not None]
    for line in dropped:
        session.warn(line)
    intraday = any(not overnight(trader) for trader in session.traders)
    text = breach_alert(positions, products, intraday=intraday, dropped=dropped)
    session.alert(AlertKind.ERROR, text)


__all__ = ["alert_breach"]

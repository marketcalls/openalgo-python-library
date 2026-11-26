"""The CNC and NRML legs a session leaves carried, as its close names them
(20-positional-live 2 "Day 1, the close", 4.1; the lead's ruling N5a-4): one line per leg
with its units, the day and price it was entered at, its stop level and its stop GTT by
trigger ID with its status (confirmed at the broker, or not with what to do), in one
SESSION alert at every close and every Ctrl+C that keeps positions (`alert_carried`), and
the same lines in the daily summary (`SessionSummary.carried`).

A stop GTT the close's read did not confirm asks the operator to check the order book and
the position, then place it by hand (`_stops_gtt.CHECK_THEN_PLACE`): it may have fired. A
`positional_stops="rearm_daily"` leg has no stop GTT: its line gives its stop level and
what the broker holds from the close on in the words of the close's WARN alert of such legs
(`_session_rearm.stop_at_close`), never a stop resting overnight.

Quantities are signed units, prices rupees, days IST. Used by the loop at the close.
"""

from __future__ import annotations

from typing import Any

from .._engine import call
from .._series import ist_time
from ._plan_catch_up import INSTRUMENT
from ._session_expiry import settles_today, settles_words
from ._session_rearm import rearm_daily, stop_at_close
from ._session_trader import Trader
from ._stops_gtt import CHECK_THEN_PLACE, gtt_unconfirmed, gtt_words, overnight, stop_gtts
from ._stops_report import NO_STOP
from ._types import AlertKind

#: The first line of the close's alert of the carried legs; each line below it gives its
#: leg's stop GTT and whether the close confirmed it.
CARRIED_TITLE = "Carried overnight:"


def carried_lines(session: Any) -> tuple[str, ...]:
    """One line per CNC or NRML leg `session` (a `_session_loop.Session`) leaves open, in
    the strategies' order (module docstring); none when it carries none."""
    return tuple(
        carried_words(session, trader)
        for trader in session.traders
        if overnight(trader) and trader.open_units() != 0
    )


def carried_words(session: Any, trader: Trader) -> str:
    """The line of `trader`'s carried leg (module docstring): "CncHold SBIN NSE 10 units
    entered on 2026-11-16 at 500.00, stop 497.00: stop GTT 261116000001 confirmed at the
    broker, trigger 497.00 for 10 units"; a rearm_daily leg's ends "stop 497.00: no stop
    at the broker until your next start"."""
    units = trader.open_units()
    held = f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units{_entered(trader)}"
    if settles_today(trader, session.parts.spec.day):
        return f"{held}: {settles_words(trader)}"
    if rearm_daily(trader):
        return f"{held}, {stop_at_close(session, trader)}"
    gtts = stop_gtts(trader)
    level = f", stop {gtts[-1]['trigger_price']:.2f}" if gtts else ""
    status = gtt_words(trader) or NO_STOP
    if gtt_unconfirmed(trader):
        status = f"{status}: {CHECK_THEN_PLACE}"
    return f"{held}{level}: {status}"


def _entered(trader: Trader) -> str:
    """When and at what price `trader`'s position was entered, in a line's words: " entered
    on 2026-11-16 at 500.00", or "" when its kernel session does not say."""
    position = call(trader.engine.position, INSTRUMENT, trader.product) or {}
    price = position.get("average_price_rupees")
    opened = [tranche[3] for tranche in position.get("tranches", []) if tranche[1]]
    if price is None or not opened:
        return ""
    return f" entered on {ist_time(min(opened)).date().isoformat()} at {price:.2f}"


def alert_carried(session: Any, lines: tuple[str, ...]) -> None:
    """The close's SESSION alert of the carried legs, `lines` (`carried_lines`); none when
    the session carries none."""
    if lines:
        session.alert(AlertKind.SESSION, "\n".join([CARRIED_TITLE, *lines]))


__all__ = ["CARRIED_TITLE", "alert_carried", "carried_lines", "carried_words"]

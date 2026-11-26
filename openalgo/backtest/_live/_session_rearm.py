"""A live session's CNC and NRML legs under `positional_stops="rearm_daily"` (design
20-positional-live section 6, decisions P2, P5 and 11.4; the lead's ruling N5b-2).

Such a leg's stop is no GTT: it is a day stop order, as an MIS leg's is (SL-M for cash
equity and futures, a stop-limit 5 percent beyond the trigger for options), which the
kernel places as the session reaches each trading day's open and the close ends
(rust/oa_bt/src/engine/rearm.rs). The stops supervisor follows it as an MIS leg's stop
(`_stops_gtt.stop_gtt`). Nothing protects the position at the broker from the close until
the next start, and the user is told so at every start and every close (decision P5):

- **At every start** (`warn_at_start`, before the banner): the sentence of section 6
  (`REARM_DAILY_SENTENCE`), then each carried leg with its units, its stop level and
  whether its day stop is placed yet (placed, sent and not confirmed yet, waiting for the
  open, or not placed), then the trading days since the leg's last close on which no
  session ran (`_preflight_carry.missed_days`). Each is a warning of the session: printed
  in the banner and kept in `SessionResult.warnings`. The banner's line of a leg continued
  from its day-close record gives its stop in the same words (`stop_at_start`,
  `_carry_lines`).
- **Before its day stop is placed** a leg with a stop level counts as guarded
  (`rearm_guarded`), never as a position opened with no stop; while the market is closed
  (`market_closed`) its units are uncovered for `_stops_cover.MARKET_CLOSED` in the
  result's unprotected table: `protect` answers `waits_for_open` before the open, which is
  no alert.
- **At every close** (`warn_at_close`, the session's end, a Ctrl+C, an error): one WARN
  alert (urgent, as the design's WARN; review round 4 of unit N5a) listing each carried leg
  with its units and stop level, as without a stop at the broker until the next start,
  then the sentence (a future the exchange settles today is left out: it is not carried
  past its expiry, `_session_expiry`); the same lines in `SessionResult.warnings` and in
  the daily summary, and the console prints them as the session's warnings. The close's
  alert of the carried legs (`_carry_close`) gives each such leg's stop in the same words
  (`stop_at_close`), so the two never disagree and neither says a stop rests at the
  broker overnight.

Quantities are signed units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._engine import call
from .._orders import VALIDITIES
from ..calendar import default_calendar
from ._plan_catch_up import INSTRUMENT
from ._preflight_carry import missed_days
from ._preflight_gtt import REARM_DAILY, REARM_DAILY_SENTENCE
from ._session_expiry import settles_today
from ._stops_gtt import overnight, stop_gtt
from ._types import AlertKind

#: What each carried leg's line says at the close.
NO_STOP_UNTIL_START = "no stop at the broker until your next start"
#: The same, for a leg whose day stop still works (a Ctrl+C that keeps it before the close).
STOP_UNTIL_CLOSE = (
    "its day stop works at the broker until the close, then no stop until your next start"
)
#: Whether a carried leg's day stop is placed yet, as the start's line says it.
DAY_STOP_PLACED = "its day stop is placed at the broker"
DAY_STOP_SENT = "its day stop is sent, not confirmed by the broker yet"
DAY_STOP_WAITS = "its day stop waits for the open"
DAY_STOP_NOT_PLACED = "its day stop is not placed yet"


def rearm_daily(trader: Any) -> bool:
    """Whether `trader` holds CNC or NRML positions with `positional_stops="rearm_daily"`:
    its leg is carried overnight and its stop is a day stop order."""
    return overnight(trader) and not stop_gtt(trader)


def day_orders(trader: Any) -> list[int]:
    """`trader`'s working kernel orders that end with the day (the kernel's DAY validity):
    its day stop and its exits among them, which the close ends; the strategy's own
    `validity="gtt"` orders rest at the broker overnight by design, so they hold up neither
    the session's end nor make a leg less bare at it."""
    return [
        kernel_order
        for kernel_order in trader.working()
        if call(trader.engine.order, kernel_order)["validity"] == VALIDITIES["day"]
    ]


def market_closed(trader: Any, now_ns: int) -> bool:
    """Whether `trader`'s exchange is outside today's session at `now_ns`: before the open
    of its first slice, or at or after its close."""
    return now_ns < trader.part.slices[0][0] or now_ns >= trader.close_ns


def stop_level(trader: Any) -> float | None:
    """The stop level of `trader`'s leg, rupees, as its kernel session holds it (its live
    state's leg); None for a leg with no stop level (no stop rule, or a trail not armed)."""
    for leg in call(trader.engine.live_state)["legs"]:
        if leg["instrument"] == INSTRUMENT and leg["product"] == trader.product:
            stop = leg.get("stop")
            return None if stop is None else float(stop["level_rupees"])
    return None


def rearm_guarded(trader: Any) -> bool:
    """Whether `trader` is a rearm_daily leg with units open and a stop level: a leg the
    stops supervisor counts as guarded before its day stop is placed (while the market is
    closed, or at an open that exits it flagged `stop_crossed`), not as a position opened
    with no stop. Reads the kernel's state, so the supervisor asks it only of a leg that
    has had no stop placed in the session (module docstring)."""
    if not rearm_daily(trader) or trader.open_units() == 0:
        return False
    return stop_level(trader) is not None


def leg_line(trader: Any) -> str:
    """A carried leg in a line's words: "CncHold 10 SBIN NSE (CNC), stop 497.00"."""
    return (
        f"{trader.name} {trader.open_units():g} {trader.symbol} {trader.exchange} "
        f"({trader.product}), {_level_words(trader)}"
    )


def stop_at_start(session: Any, trader: Any) -> str:
    """A carried rearm_daily leg's stop at the start, in a line's words (module docstring):
    "stop 497.00: its day stop waits for the open"."""
    return f"{_level_words(trader)}: {_day_stop_words(session, trader)}"


def stop_at_close(session: Any, trader: Any) -> str:
    """A carried rearm_daily leg's stop from the close's moment on, in a line's words
    (module docstring): "stop 497.00: no stop at the broker until your next start"."""
    return f"{_level_words(trader)}: {_at_close_words(session, trader)}"


def _level_words(trader: Any) -> str:
    level = stop_level(trader)
    return "no stop level" if level is None else f"stop {level:.2f}"


def warn_at_start(session: Any) -> None:
    """The start's warnings of a session with a rearm_daily strategy (module docstring):
    the sentence, each carried leg and its day stop, the days no session ran. None for a
    session without one. Raises JournalError when the journal's day-close records cannot be
    read."""
    traders = [trader for trader in session.traders if rearm_daily(trader)]
    if not traders:
        return
    session.warn(REARM_DAILY_SENTENCE)
    for trader in traders:
        if trader.open_units() != 0:
            session.warn(f"{leg_line(trader)}: {_day_stop_words(session, trader)}")
        missed = _missed_words(session, trader)
        if missed:
            session.warn(missed)


def warn_at_close(session: Any) -> tuple[str, ...]:
    """The close's alert of the rearm_daily legs carried (module docstring), sent once, and
    its lines kept as the session's warnings; returns those lines for the daily summary,
    none when no such leg is open. Read while the kernel sessions run."""
    day = session.parts.spec.day
    legs = [
        trader
        for trader in session.traders
        if rearm_daily(trader) and trader.open_units() and not settles_today(trader, day)
    ]
    if not legs:
        return ()
    lines = tuple(f"{leg_line(trader)}: {_at_close_words(session, trader)}" for trader in legs)
    session.alert(
        AlertKind.WARN,
        f'positional_stops="{REARM_DAILY}": {"; ".join(lines)}. {REARM_DAILY_SENTENCE}',
    )
    for line in lines:
        session.warn(line)
    return lines


def _covered(session: Any, trader: Any) -> bool:
    units = round(abs(trader.open_units()))
    return 0 < units <= session.stops.covered(trader)


def _day_stop_words(session: Any, trader: Any) -> str:
    """Whether a carried leg's day stop is placed yet, at the start (module docstring)."""
    if _covered(session, trader):
        return DAY_STOP_PLACED
    if session.now() < trader.part.slices[0][0]:
        return DAY_STOP_WAITS
    leg = session.stops.leg(trader)
    if leg.stop is not None and leg.stop in trader.working():
        return DAY_STOP_SENT
    return DAY_STOP_NOT_PLACED


def _at_close_words(session: Any, trader: Any) -> str:
    """What a carried leg holds at the broker from the close's moment on."""
    if _covered(session, trader) and not market_closed(trader, session.now()):
        return STOP_UNTIL_CLOSE
    return NO_STOP_UNTIL_START


def _missed_words(session: Any, trader: Any) -> str | None:
    """The trading days since `trader`'s last close on which no session ran, in a warning's
    words, or None for none (its latest day-close record, in the session's ledger)."""
    record = session.parts.journal.day_close(trader.name)
    day = session.parts.spec.day
    if record is None or record.day >= day:
        return None
    days = missed_days(trader.exchange, record.day, day, default_calendar())
    if not days:
        return None
    named = ", ".join(str(missed) for missed in days)
    return (
        f"{trader.name}: no session ran on {named}, the trading days since its last close "
        f"on {record.day}"
    )


__all__ = [
    "DAY_STOP_NOT_PLACED",
    "DAY_STOP_PLACED",
    "DAY_STOP_SENT",
    "DAY_STOP_WAITS",
    "NO_STOP_UNTIL_START",
    "STOP_UNTIL_CLOSE",
    "day_orders",
    "leg_line",
    "market_closed",
    "rearm_daily",
    "rearm_guarded",
    "stop_at_close",
    "stop_at_start",
    "stop_level",
    "warn_at_close",
    "warn_at_start",
]

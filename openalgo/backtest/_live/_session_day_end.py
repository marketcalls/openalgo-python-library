"""The end of a live session's day (16-session-design 3.4; unit 8), split from
`_session_stop` so each module keeps one job: when the day's session ends by itself, and
the alert of a strategy still not flat after its square-off.

Once every strategy is past its MIS square-off, flat and with nothing working, the session
ends ("session_end"). From the square-off plus `_session_stop.SQUARE_OFF_VERIFY_AFTER_S` a
strategy not flat gets the ERROR alert "still not flat after square-off", each
`_session_stop.NOT_FLAT_REPEAT_S` while it lasts. A leg still open at the exchange's
close plus `_session_stop.SESSION_END_GRACE_S` ends the session too, with an ERROR alert
for every such leg (the caller's lines of the legs left open). After a guard lock the
session ends at that same time ("mode_breach").

**CNC and NRML legs** (20-positional-live 7.3; the lead's ruling N3b-4) are carried, never
squared off: no "still not flat after square-off" alert names them, and they need not be
flat. The session ends at the latest close among their exchanges (MCX by the calendar's
dated rows), once every MIS leg is done and the close's GTT book read has been judged
(`_session_carry.CarriedLegs.confirmed`, asked for at that close whether the MIS legs are
done or not); the safe stop then names each carried leg in a line (`_session_stop`). A
session with no CNC or NRML strategy ends as before.

Times are UTC nanoseconds, quantities units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._session_expiry import settles_today
from ._session_rearm import rearm_daily
from ._session_trader import Trader
from ._stops_gtt import gtt_cancelling, gtt_cover, gtt_waiting, overnight
from ._types import AlertKind

#: How a session ended by the day (1.4 `stopped_by`).
SESSION_END, MODE_BREACH = "session_end", "mode_breach"


class DayEnd:
    """The end of the day of `session`, a `_session_loop.Session` (module docstring):
    `not_flat_ns`, by strategy, when a strategy not flat is first alerted; `end_ns`, when
    the session ends whatever is open; `repeat_ns`, between two alerts of one strategy."""

    def __init__(
        self, session: Any, *, not_flat_ns: Mapping[str, int], end_ns: int, repeat_ns: int
    ) -> None:
        self._session = session
        carried = [t.close_ns for t in session.traders if overnight(t)]
        #: When the latest exchange of a CNC or NRML strategy closes (UTC ns), or None.
        self._carry_ns = max(carried) if carried else None
        self._not_flat_ns = dict(not_flat_ns)
        self._end_ns = end_ns
        self._repeat_ns = repeat_ns

    @property
    def end_ns(self) -> int:
        """When the session ends whatever is open (UTC ns): the latest close of its
        exchanges plus the grace."""
        return self._end_ns

    def due(self, now_ns: int) -> tuple[str | None, bool]:
        """(how the session ends now, or None; whether every leg left open is to be named
        in the lines and alerted) at `now_ns` (module docstring)."""
        session = self._session
        if session.locked:
            return (MODE_BREACH if now_ns >= self._end_ns else None), False
        intraday = [t for t in session.traders if not overnight(t)]
        done = all(now_ns >= t.square_off_ns and t.flat for t in intraday)
        # Asked for from the latest close of the carried legs, whether the MIS legs are
        # done or not, so it is judged by the time the session ends whatever is open.
        carried_done = self._carried_done(now_ns)
        if done and (self._carry_ns is None or carried_done):
            return SESSION_END, False
        if now_ns >= self._end_ns:
            return SESSION_END, True
        for trader in intraday:
            if now_ns >= self._not_flat_ns[trader.name] and not trader.flat:
                self._not_flat_ns[trader.name] = now_ns + self._repeat_ns
                self._not_flat(trader)
        return None, False

    def _carried_done(self, now_ns: int) -> bool:
        """Whether the carried legs let the session end (module docstring): the close's GTT
        book read judged, and every carried leg's own close answered (`_closing`), so the
        lines see where it ended."""
        if self._carry_ns is None or now_ns < self._carry_ns:
            return False
        if any(self._closing(trader) for trader in self._session.traders):
            return False
        return bool(self._session.news.carry.confirmed())

    def _closing(self, trader: Trader) -> bool:
        """Whether `trader`'s carried leg is still closing: a `positional_stops="rearm_daily"`
        leg with an order working (its day stop's end at the close not answered yet,
        `_session_rearm`), or a future the exchange settles today whose stop GTT rests or
        is being cancelled (the kernel cancels it at the close, `_session_expiry`)."""
        if rearm_daily(trader):
            return bool(trader.working())
        if settles_today(trader, self._session.parts.spec.day):
            return bool(gtt_cover(trader) or gtt_cancelling(trader) or gtt_waiting(trader))
        return False

    def _not_flat(self, trader: Trader) -> None:
        exits = [trader.order_name(kernel_order) for kernel_order in trader.working()]
        self._session.alert(
            AlertKind.ERROR,
            f"still not flat after square-off: {trader.name} {trader.symbol} "
            f"{trader.exchange} {trader.open_units():g} units, working orders "
            f"{', '.join(exits) or 'none'}",
        )


__all__ = ["MODE_BREACH", "SESSION_END", "DayEnd"]

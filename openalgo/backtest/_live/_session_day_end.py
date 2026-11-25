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

Times are UTC nanoseconds, quantities units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._session_trader import Trader
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
        traders = session.traders
        if all(now_ns >= t.square_off_ns and t.flat for t in traders):
            return SESSION_END, False
        if now_ns >= self._end_ns:
            return SESSION_END, True
        for trader in traders:
            if now_ns >= self._not_flat_ns[trader.name] and not trader.flat:
                self._not_flat_ns[trader.name] = now_ns + self._repeat_ns
                self._not_flat(trader)
        return None, False

    def _not_flat(self, trader: Trader) -> None:
        exits = [trader.order_name(kernel_order) for kernel_order in trader.working()]
        self._session.alert(
            AlertKind.ERROR,
            f"still not flat after square-off: {trader.name} {trader.symbol} "
            f"{trader.exchange} {trader.open_units():g} units, working orders "
            f"{', '.join(exits) or 'none'}",
        )


__all__ = ["MODE_BREACH", "SESSION_END", "DayEnd"]

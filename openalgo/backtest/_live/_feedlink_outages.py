"""The spans in which the feed was down, and the order stream's gap, for `feedlink`
(16-session-design 2.5 A.12 and 6.5), from the feed's events in the order they came.

A disconnect the session did not ask for opens an outage at the loop's clock (a second
drop while one is open keeps its start). It ends at the connect of the connection that
restores the session, whose ticks flow from then on: the SDK fires its reconnect hook
only once the server has answered every replayed subscription, or it stopped waiting for
the answers (`replay_ack` in `openalgo/_feed_state.py`, 10 s by default), and a tick
taken meanwhile must not fall inside the span, or its slice would only seed. That
reconnect hook says the order stream is back: from it the loop asks the order worker for
`stream_gap(since_ns)`, from the first drop since the order stream was last restored, so
a fill while it was down is polled for.

The ticks' receive times come from the same clock as the loop takes them. A feed that
stops ticking while its connection stays up is dead, not down: the bars' `FeedHealth`
finds that from the ticks (6.5). Times are UTC nanoseconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ._tick_spans import Outage
from ._types import LOGGER_NAME, Now

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class Recovery:
    """What the loop does as the feed comes back (16-session-design 2.5 A.12), from
    `FeedLink.note`. `outage`, when set: the span the feed was down, to `resume` every
    bar builder over. `stream_gap_since_ns`, when set: when the order stream was lost
    (UTC nanoseconds), to ask the order worker for `stream_gap(stream_gap_since_ns)`, so
    a fill while it was down is polled for. At least one is set."""

    outage: Outage | None = None
    stream_gap_since_ns: int | None = None


class Outages:
    """The feed's outages and the order stream's gap (see the module docstring), on the
    loop's clock `now`."""

    def __init__(self, now: Now) -> None:
        self._now = now
        self._down_since_ns: int | None = None
        self._gap_since_ns: int | None = None
        self._ended: list[Outage] = []

    @property
    def down_since_ns(self) -> int | None:
        """When the outage under way began, or None while the feed is up."""
        return self._down_since_ns

    @property
    def ended(self) -> tuple[Outage, ...]:
        """The outages that ended, in order."""
        return tuple(self._ended)

    def dropped(self, seq: int | None, reason: object, will_reconnect: bool) -> None:
        """A connection the session did not close dropped: an outage opens now, unless
        one is open; the order stream's gap opens now, unless one is open."""
        if self._down_since_ns is not None:
            return
        self._down_since_ns = self._now()
        if self._gap_since_ns is None:
            self._gap_since_ns = self._down_since_ns
        will = "it reconnects by itself" if will_reconnect else "nothing reconnects it"
        logger.warning("the feed is down (connection %s: %s); %s", seq, reason, will)

    def restored(self, seq: int | None) -> Recovery | None:
        """The connection that restores the session is up: the outage under way ends
        now. Returns it, or None when none was open."""
        if self._down_since_ns is None:
            return None
        return Recovery(outage=self._back(seq))

    def stream_back(self, seq: int | None) -> Recovery | None:
        """The SDK's reconnect hook: the order stream is back. Returns the stream gap to
        ask for, and the outage when it is still open (no connect was noted before:
        the span is never lost); None when neither was open."""
        outage = None if self._down_since_ns is None else self._back(seq)
        since, self._gap_since_ns = self._gap_since_ns, None
        if outage is None and since is None:
            return None
        return Recovery(outage=outage, stream_gap_since_ns=since)

    def close(self) -> None:
        """The session closes the feed: an outage under way ends now."""
        if self._down_since_ns is not None:
            self._end()

    def _back(self, seq: int | None) -> Outage:
        outage = self._end()
        seconds = (outage.end_ns - outage.start_ns) / NANOS_PER_SECOND
        logger.warning("the feed is back (connection %s) after %.1f s down", seq, seconds)
        return outage

    def _end(self) -> Outage:
        start_ns = self._down_since_ns
        # The wall clock may step back between two reads: an outage never ends before it
        # starts, or the bar builders would refuse it.
        outage = Outage(start_ns, max(self._now(), start_ns))
        self._ended.append(outage)
        self._down_since_ns = None
        return outage


__all__ = ["Outages", "Recovery"]

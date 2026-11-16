"""The clock skew of a live session, estimated from the feed (16-session-design 6.4;
unit 11): how far this computer's clock is from the exchange's.

The preflight measures nothing against a server on this computer, so the session
estimates the skew from the ticks' exchange times where the feed gives them:

- **Samples.** A tick whose day's cumulative volume rose since its instrument's last
  tick (a trade just printed, so its last trade time is fresh) and that carries the
  exchange's last trade time gives one sample: the feed callback's clock read minus that
  time, which is the network delay plus this clock's offset from the exchange's.
- **Windows.** Every `SKEW_WINDOW_S` the window closes. The instruments with at least
  `SKEW_MIN_TRADES` samples in it (liquid ones) give the estimate: the least of their
  samples, the least delayed tick. A window without a liquid instrument gives none.
- **Feeds that cannot be measured.** A broker whose feed adapter stamps the last trade
  time with the OpenAlgo server's own clock (`SERVER_CLOCK_LTT_BROKERS`) gives samples
  that read near 0 for a server on this computer whatever the true skew: none is taken,
  and `unmeasurable_note` is the banner's warning. A feed that gives no exchange time at
  all is found from its ticks: a window in which an instrument traded liquidly with no
  exchange time on any of its trades sets `untimed`.

Beyond `SKEW_ALERT_S` either way the session alerts (`skew_words`); a skew locks
nothing. Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .._layout import NANOS_PER_SECOND
from ._types import Tick

#: Seconds of one estimate's window (16-session-design 6.4).
SKEW_WINDOW_S = 60.0
#: Trade ticks with an exchange time an instrument needs in a window to count as liquid.
SKEW_MIN_TRADES = 20
#: Seconds of skew either way beyond which the session alerts: the preflight's limit
#: (`_preflight_clock.MAX_CLOCK_SKEW_SECONDS`, 08-live-trading section 4).
SKEW_ALERT_S = 2.0
#: The brokers whose feed adapter in the OpenAlgo server stamps a tick's last trade time
#: with the server's own clock in LTP and quote mode, the mode the session subscribes in
#: (the tick adapter's per-broker survey, merge notes "tick adapter and recorder done";
#: broker/zerodha/streaming/zerodha_websocket.py:740 in the OpenAlgo server).
SERVER_CLOCK_LTT_BROKERS = frozenset({"zerodha"})

_WINDOW_NS = round(SKEW_WINDOW_S * NANOS_PER_SECOND)
_Key = tuple[str, str]


@dataclass(frozen=True)
class SkewEstimate:
    """One window's estimate: `seconds`, this clock minus the exchange's at most
    (positive when this clock is ahead; the least delayed tick still holds its network
    delay), from the least of `samples` trade ticks of the liquid `symbols`, the window
    closing at `window_end_ns` (UTC nanoseconds)."""

    seconds: float
    samples: int
    symbols: tuple[str, ...]
    window_end_ns: int

    @property
    def beyond_limit(self) -> bool:
        """True when the skew is beyond `SKEW_ALERT_S` either way."""
        return abs(self.seconds) > SKEW_ALERT_S


@dataclass
class _Window:
    start_ns: int
    least_ns: dict[_Key, int] = field(default_factory=dict)
    timed: dict[_Key, int] = field(default_factory=dict)
    untimed: dict[_Key, int] = field(default_factory=dict)


class ClockSkew:
    """The skew estimate of a session whose broker is `broker` (module docstring).
    `estimate` is the last window's estimate, None before any; `untimed` turns True once
    a window showed that the feed gives no exchange time."""

    def __init__(self, broker: str) -> None:
        self.broker = broker
        #: False for a broker whose feed's last trade time is the server's clock.
        self.measurable = broker.strip().lower() not in SERVER_CLOCK_LTT_BROKERS
        self.estimate: SkewEstimate | None = None
        self.untimed = False
        self._volume: dict[_Key, float] = {}
        self._window: _Window | None = None

    def unmeasurable_note(self) -> str | None:
        """The banner's warning when the broker's feed cannot be measured, else None."""
        return None if self.measurable else unmeasured_words(self.broker)

    def start(self, now_ns: int) -> None:
        """Begin the first window at `now_ns` (the session starts watching the feed)."""
        self._window = _Window(now_ns)

    def sample(self, tick: Tick, arrived_ns: int) -> None:
        """Take a tick the feed's callback received at `arrived_ns` (the module
        docstring says which give a sample). Nothing before `start`."""
        key = (tick.symbol, tick.exchange)
        previous = self._volume.get(key)
        if tick.volume is not None:
            self._volume[key] = tick.volume
        window = self._window
        traded = previous is not None and tick.volume is not None and tick.volume > previous
        if window is None or not traded or not self.measurable:
            return
        if tick.ts_event_ns is None:
            window.untimed[key] = window.untimed.get(key, 0) + 1
            return
        offset_ns = arrived_ns - tick.ts_event_ns
        window.timed[key] = window.timed.get(key, 0) + 1
        window.least_ns[key] = min(offset_ns, window.least_ns.get(key, offset_ns))

    def close_due(self, now_ns: int) -> SkewEstimate | None:
        """Close the window once `SKEW_WINDOW_S` have passed since it began, and begin the
        next at `now_ns`: the estimate it gives (kept as `estimate`), None when no window
        closed or the one that did had no liquid instrument."""
        window = self._window
        if window is None or now_ns < window.start_ns + _WINDOW_NS:
            return None
        self._window = _Window(now_ns)
        liquid = sorted(key for key, count in window.timed.items() if count >= SKEW_MIN_TRADES)
        if not liquid:
            if any(count >= SKEW_MIN_TRADES for count in window.untimed.values()):
                self.untimed = True
            return None
        least_ns = min(window.least_ns[key] for key in liquid)
        self.estimate = SkewEstimate(
            seconds=least_ns / NANOS_PER_SECOND,
            samples=sum(window.timed[key] for key in liquid),
            symbols=tuple(symbol for symbol, _ in liquid),
            window_end_ns=now_ns,
        )
        return self.estimate


def skew_words(estimate: SkewEstimate) -> str:
    """The ERROR alert of a skew beyond the limit, and the result's warning."""
    way = "ahead of" if estimate.seconds > 0 else "behind"
    return (
        f"the local clock is about {abs(estimate.seconds):.1f} s {way} the exchange's (the "
        f"least delayed of {estimate.samples} trade ticks in the last minute): bars are cut "
        "by this computer's clock, so trades at each minute's end may be dropped as late. "
        "Synchronise this computer's clock with an internet time server"
    )


def unmeasured_words(broker: str, *, untimed: bool = False) -> str:
    """The warning that the clock cannot be checked through `broker`'s feed: its last
    trade time is the server's clock, or (`untimed`) its ticks carry no exchange time."""
    why = (
        "its ticks carry no exchange time"
        if untimed
        else "its last trade time is the OpenAlgo server's own clock"
    )
    return (
        f"the clock against the exchange cannot be checked through {broker}'s feed ({why}): "
        "keep this computer's clock synchronised"
    )


__all__ = [
    "SERVER_CLOCK_LTT_BROKERS",
    "SKEW_ALERT_S",
    "SKEW_MIN_TRADES",
    "SKEW_WINDOW_S",
    "ClockSkew",
    "SkewEstimate",
    "skew_words",
    "unmeasured_words",
]

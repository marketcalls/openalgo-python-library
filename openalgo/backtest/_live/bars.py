"""Live bars (08-live-trading section 5, D27): bars built from WebSocket ticks and stamped
at their close, the check against the broker's candles, and the health of the feed.

- `TickBarBuilder` turns one exchange's ticks of one session into bars on the session's
  grid: the calendar's base-interval slices of the session, the grid the kernel's live
  session is built on (`Engine::new_session`). Slices open at IST midnight plus the
  exchange's regular open plus whole intervals, so the first one may open before the
  open of a special session, and the last is cut at the session close. A bar goes to
  the kernel as the base bar of the slice that opens at its `ts_open_ns`
  (`Engine::add_bar`) once its `ts_close_ns`, the slice's close, has passed.
- `candle_bar` reads a broker candle row as a bar, and `CandleCheck` compares a
  tick-built bar with the broker's candle of the same slice: the differences go to the
  drift log, and indicators continue from the candle when it starts on time.
- `FeedHealth` tells a dead feed (no tick on any symbol: entries lock) from a quiet
  symbol (an illiquid one, while the others tick).

Times are UTC nanoseconds, prices rupees, volume units. Not thread-safe: the session loop
owns these objects, as it owns the engine; the feed's thread only queues ticks for it.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass

from .._calendar_convert import DateLike
from .._layout import NANOS_PER_SECOND
from ..calendar import NO_BAR, Calendar, default_calendar
from ..errors import CalendarError, ConfigError
from ._bars_candle import (
    CANDLE_SOURCE,
    TICKS_SOURCE,
    BarDrift,
    CandleCheck,
    candle_bar,
    check_interval,
)
from ._bars_health import DEFAULT_DEAD_AFTER_S, FeedHealth
from ._types import LOGGER_NAME, LiveBar, Tick

#: The interval of the bars when none is given: one minute, the interval of the broker
#: candles the bars are checked against (08-live-trading section 5, rule 2).
DEFAULT_INTERVAL_NS = 60 * NANOS_PER_SECOND

logger = logging.getLogger(LOGGER_NAME)


@dataclass(slots=True)
class _Forming:
    """The bar of one symbol that is still forming. `first_ns` and `last_ns` are the
    times of the ticks its open and close came from, so a tick that arrives out of order
    inside the bar still sets the open or close by its time."""

    slot: int
    open: float
    high: float
    low: float
    close: float
    first_ns: int
    last_ns: int
    ticks: int = 1

    def add(self, price: float, at_ns: int) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        if at_ns < self.first_ns:
            self.open, self.first_ns = price, at_ns
        if at_ns >= self.last_ns:
            self.close, self.last_ns = price, at_ns
        self.ticks += 1


@dataclass(slots=True)
class _Instrument:
    """One symbol's state. `next_slot` is the first slice a tick may still go to: the
    slices before it have a bar closed or forming. `volume_high` is the highest
    cumulative day volume seen, `volume_base` its value when the last bar closed."""

    forming: _Forming | None = None
    next_slot: int = 0
    volume_high: float | None = None
    volume_base: float = 0.0


def _usable(tick: Tick) -> bool:
    """A tick with a positive, finite LTP and, when given, a finite cumulative volume
    that is not negative."""
    if not (math.isfinite(tick.ltp) and tick.ltp > 0):
        return False
    return tick.volume is None or (math.isfinite(tick.volume) and tick.volume >= 0)


class TickBarBuilder:
    """Builds the bars of one exchange's session from its ticks (D27).

    Bar `k` covers the slice `[open_k, close_k)` of `slices`. A tick is placed by
    `ts_event_ns` when the feed gives it, else by `ts_recv_ns`, and updates the forming
    bar of its symbol: high and low from every LTP, open and close from the earliest and
    latest tick by that time. `ticks` counts the ticks of the bar and `source` is
    "ticks". A symbol is registered by its first accepted tick; bars that close together
    are returned in time order, symbols in registration order.

    A bar closes when a tick of its symbol falls in a later slice (`on_tick`) or when the
    clock passes its close (`close_due`). A slice in which a symbol had no tick makes no
    bar for it, as a backtest runs a missing bar. Ticks outside the session are ignored
    and counted (`outside_ticks`); a tick for a slice that has closed (or that is before
    its symbol's forming bar) is dropped and counted (`late_ticks`); a tick with an LTP
    that is not a positive number, or a cumulative volume that is negative or not a
    number, is dropped and counted (`invalid_ticks`).

    Volume comes from the day's cumulative volume: a bar's volume is the highest
    cumulative volume seen up to its close minus that value at the previous bar's close
    (the count only rises, so an older value adds nothing). A symbol whose first volume
    is in the session's first slice counts from zero, so the first bar holds the day's
    volume so far; one whose first volume is later counts from that tick. Volume is 0.0
    while the feed gives none.

    `exchange` is an OpenAlgo exchange code and `day` an IST date (`datetime.date` or ISO
    string). `calendar` is the package default calendar when None. Raises ConfigError
    for an interval that is not a positive whole number of seconds, and CalendarError
    for an unknown exchange or a day on which the exchange has no session.
    """

    def __init__(
        self,
        exchange: str,
        day: DateLike,
        *,
        interval_ns: int = DEFAULT_INTERVAL_NS,
        calendar: Calendar | None = None,
    ) -> None:
        interval_ns = check_interval(interval_ns)
        calendar = default_calendar() if calendar is None else calendar
        session = calendar.session(exchange, day)
        if session is None:
            raise CalendarError(
                f"{exchange} has no session on {day}: live bars are built on a trading day "
                "only; check the date, or add the session to the calendar"
            )
        base = f"{interval_ns // NANOS_PER_SECOND}s"
        grid = calendar.union_grid([exchange], day, day, base=base)
        closes = grid[f"close_{grid.attrs['exchanges'][0]}"].to_numpy()
        has_bar = closes != NO_BAR
        self._exchange = exchange
        self._session = session
        self._opens: list[int] = grid["ts"].to_numpy()[has_bar].tolist()
        self._closes: list[int] = closes[has_bar].tolist()
        self._instruments: dict[str, _Instrument] = {}
        # Slices whose close the clock has passed (`close_due`): closed for every symbol.
        self._due_slots = 0
        self._late = self._outside = self._invalid = 0

    @property
    def exchange(self) -> str:
        """The exchange whose ticks this builder takes."""
        return self._exchange

    @property
    def session(self) -> tuple[int, int]:
        """(open, close) of the session, UTC nanoseconds."""
        return self._session

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        """(open, close) of every slice of the session in order, UTC nanoseconds: the
        exchange's slices of the kernel's session grid."""
        return tuple(zip(self._opens, self._closes, strict=True))

    @property
    def late_ticks(self) -> int:
        """Ticks dropped because their slice had closed or was before the forming bar."""
        return self._late

    @property
    def outside_ticks(self) -> int:
        """Ticks ignored because they fall outside the session."""
        return self._outside

    @property
    def invalid_ticks(self) -> int:
        """Ticks dropped for an unusable LTP or cumulative volume."""
        return self._invalid

    def on_tick(self, tick: Tick) -> list[LiveBar]:
        """Add a tick to the forming bar of its symbol; return the bar it closes (at most
        one: the symbol's forming bar, when the tick falls in a later slice).

        Raises ConfigError for a tick of another exchange.
        """
        if tick.exchange != self._exchange:
            raise ConfigError(
                f"a tick of {tick.exchange}:{tick.symbol} reached the bar builder of "
                f"{self._exchange}: route each exchange's ticks to its own builder"
            )
        if not _usable(tick):
            self._invalid += 1
            logger.debug("tick dropped, unusable LTP or volume: %r", tick)
            return []
        at_ns = tick.ts_event_ns if tick.ts_event_ns is not None else tick.ts_recv_ns
        if not self._session[0] <= at_ns < self._session[1]:
            self._outside += 1
            return []
        slot = bisect.bisect_right(self._opens, at_ns) - 1
        state = self._instruments.get(tick.symbol)
        if slot < max(self._due_slots, 0 if state is None else state.next_slot):
            self._late += 1
            logger.debug("late tick dropped: %r", tick)
            return []
        if state is None:
            state = self._instruments[tick.symbol] = _Instrument()
        closed: list[LiveBar] = []
        if state.forming is not None and slot > state.forming.slot:
            closed.append(self._close(tick.symbol, state, state.forming))
        if tick.volume is not None:
            if state.volume_high is None:
                state.volume_base = 0.0 if slot == 0 else tick.volume
            state.volume_high = max(tick.volume, state.volume_high or 0.0)
        if state.forming is None:
            price = tick.ltp
            state.forming = _Forming(slot, price, price, price, price, at_ns, at_ns)
            state.next_slot = slot
        else:
            state.forming.add(tick.ltp, at_ns)
        return closed

    def close_due(self, now_ns: int) -> list[LiveBar]:
        """Close, by the clock, every forming bar whose close is at or before `now_ns`
        (UTC nanoseconds), such as a bar whose slice ended with no later tick. From then
        on a tick for any slice that closed by `now_ns` is late."""
        self._due_slots = max(self._due_slots, bisect.bisect_right(self._closes, now_ns))
        bars = [
            self._close(symbol, state, state.forming)
            for symbol, state in self._instruments.items()
            if state.forming is not None and state.forming.slot < self._due_slots
        ]
        bars.sort(key=lambda bar: bar.ts_open_ns)
        return bars

    def _close(self, symbol: str, state: _Instrument, forming: _Forming) -> LiveBar:
        volume = 0.0
        if state.volume_high is not None:
            volume = state.volume_high - state.volume_base
            state.volume_base = state.volume_high
        state.forming, state.next_slot = None, forming.slot + 1
        return LiveBar(
            symbol=symbol,
            exchange=self._exchange,
            ts_open_ns=self._opens[forming.slot],
            ts_close_ns=self._closes[forming.slot],
            open=forming.open,
            high=forming.high,
            low=forming.low,
            close=forming.close,
            volume=volume,
            ticks=forming.ticks,
            source=TICKS_SOURCE,
        )


__all__ = [
    "CANDLE_SOURCE",
    "DEFAULT_DEAD_AFTER_S",
    "DEFAULT_INTERVAL_NS",
    "TICKS_SOURCE",
    "BarDrift",
    "CandleCheck",
    "FeedHealth",
    "TickBarBuilder",
    "candle_bar",
]

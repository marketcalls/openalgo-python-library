"""Live bars (08-live-trading section 5, D27): bars built from WebSocket ticks and stamped
at their close, the check against the broker's candles, and the health of the feed.

- `TickBarBuilder` turns one exchange's ticks of one session into bars on the session's
  grid: the calendar's base-interval slices of the session, the grid the kernel's live
  session is built on (`Engine::new_session`). Slices open at IST midnight plus the
  exchange's regular open plus whole intervals, so the first one may open before the
  open of a special session, and the last is cut at the session close. A bar is the
  base bar of the slice that opens at its `ts_open_ns` (`Engine::add_bar`); `close_due`
  says which slices it closed, so the session releases exactly those to the kernel.
- `candle_bar` reads a broker candle row as a bar, and `CandleCheck` compares a
  tick-built bar with the broker's candle of the same slice: the differences go to the
  drift log, and indicators continue from the candle when it covers the same slice.
- `FeedHealth` tells a dead feed (no tick on any symbol: entries lock) from a quiet
  symbol (an illiquid one, while the others tick).

Times are UTC nanoseconds, prices rupees, volume units. Not thread-safe: the session loop
owns these objects, as it owns the engine; the feed's thread only queues ticks for it.
"""

from __future__ import annotations

import bisect
import copy
import datetime as dt
import logging
import math
import numbers
from dataclasses import dataclass

import pandas as pd

from .._calendar_convert import IST, DateLike, from_day, to_day
from .._layout import NANOS_PER_SECOND
from ..calendar import NO_BAR, Calendar, default_calendar
from ..errors import CalendarError, ConfigError
from ..reference.symbols import INDEX_EXCHANGES
from ._bars_candle import (
    CANDLE_SOURCE,
    TICKS_SOURCE,
    BarDrift,
    CandleCheck,
    candle_bar,
    check_interval,
)
from ._bars_health import DEFAULT_DEAD_AFTER_S, FeedHealth
from ._bars_trades import PlacedTick, SliceTrades, SymbolTrades
from ._types import LOGGER_NAME, LiveBar, Tick

#: The interval of the bars when none is given: one minute, the interval of the broker
#: candles the bars are checked against (08-live-trading section 5, rule 2).
DEFAULT_INTERVAL_NS = 60 * NANOS_PER_SECOND
#: Seconds after a slice's close during which a tick of the slice still on its way is
#: counted in it; a tick received later is late.
DEFAULT_CLOSE_GRACE_S = 1.0
#: How far a tick's exchange time may be from the instant it was received and still be
#: used to place it.
EVENT_TIME_TOLERANCE_NS = 5 * NANOS_PER_SECOND
_TOLERANCE_S = EVENT_TIME_TOLERANCE_NS // NANOS_PER_SECOND
# An event time before 2000 is no time of a trade (a value in milliseconds, say), so it
# marks no snapshot of an earlier session.
_EARLIEST_EVENT_NS = int(dt.datetime(2000, 1, 1, tzinfo=dt.UTC).timestamp()) * NANOS_PER_SECOND

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class ClosedSlices:
    """What one `close_due` call closed: `slices`, the (open, close) of every slice it
    closed in order (UTC nanoseconds), which the session releases to the kernel, and
    `bars`, the bars of their trades in time order, symbols in registration order."""

    slices: tuple[tuple[int, int], ...]
    bars: tuple[LiveBar, ...]


def _usable(tick: Tick) -> bool:
    """A tick with a positive, finite LTP and, when given, a finite cumulative volume
    that is not negative."""
    if not (math.isfinite(tick.ltp) and tick.ltp > 0):
        return False
    return tick.volume is None or (math.isfinite(tick.volume) and tick.volume >= 0)


def _instant(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigError(f"{name} is an instant in UTC nanoseconds (an int), got {value!r}")
    return int(value)


def _grace_ns(seconds: float) -> int:
    usable = isinstance(seconds, numbers.Real) and not isinstance(seconds, bool)
    if not (usable and math.isfinite(seconds) and seconds >= 0):
        raise ConfigError(f"close_grace_s must be 0 or more seconds, got {seconds!r}")
    return round(seconds * NANOS_PER_SECOND)


def _ist_day_bounds(day: DateLike) -> tuple[int, int]:
    """[start, end) of an IST date as UTC nanoseconds."""
    date = from_day(to_day(day))
    start, end = (pd.Timestamp(d).tz_localize(IST) for d in (date, date + dt.timedelta(days=1)))
    return int(start.value), int(end.value)


class TickBarBuilder:
    """Builds the bars of one exchange's session from its ticks (D27).

    Bar `k` covers the slice `[open_k, close_k)` of `slices` and is made of the trades of
    its symbol in that slice (`_bars_trades` says which ticks are trades): open and close
    are the first and last trade received, high and low over all of them, `ticks` counts
    them, `volume` is the rise of the day's cumulative volume they carry (a float, 0.0
    while the feed gives none) and `source` is "ticks". A slice without a trade makes no
    bar for its symbol, as a backtest runs a missing bar.

    A tick's time places it in a slice: `ts_event_ns` when it is within
    `EVENT_TIME_TOLERANCE_NS` of `ts_recv_ns` and on the session's IST day, else
    `ts_recv_ns`. An event time of 0 is a missing one, and an older time of that day (the
    last trade of an illiquid stock, the pre-open's) is simply not used. A time of an
    earlier day marks a snapshot of an earlier session, which only seeds; it and any
    other time (ahead of receipt, before 2000, a later day) are counted
    (`event_time_rejected`) with one warning per symbol.

    A tick is late when it was received at or after its slice's close plus
    `close_grace_s`: it is dropped and, when it was a trade, counted (`late_ticks`) with
    one warning per symbol (the local clock may be ahead of the exchange).

    The session loop has two duties, so that the bars depend on the ticks' receive times
    only and a replay driven by receive times gives the live bars: it stamps
    `ts_recv_ns` when it takes the tick off its queue, and it hands over every tick
    received up to `now` before it calls `close_due(now)`.

    `watching_from_ns` is the instant the session started watching the feed. From the
    session open or earlier, every slice is whole and a symbol's volume counts from 0; a
    tick before the open seeds its symbol's last known LTP. After a start in the middle
    of the session, the slice in which watching began (and any before it) makes no bar:
    its ticks only seed each symbol's volume level and last known LTP. `resume` does the
    same for the slices of a feed outage.

    Ticks outside the session are counted (`outside_ticks`); a tick with an LTP that is
    not a positive number, or a cumulative volume that is negative or not a number, is
    dropped and counted (`invalid_ticks`). A symbol is registered by its first tick.

    `exchange` is an OpenAlgo exchange code and `day` an IST date (`datetime.date` or ISO
    string); `calendar` is the package default calendar when None. Raises ConfigError
    for an interval that is not a positive whole number of seconds shorter than a day
    (whatever its type), a `watching_from_ns` that is not an int or a negative grace,
    and CalendarError for an unknown exchange or a day the exchange has no session on.
    """

    def __init__(
        self,
        exchange: str,
        day: DateLike,
        *,
        watching_from_ns: int,
        interval_ns: int = DEFAULT_INTERVAL_NS,
        close_grace_s: float = DEFAULT_CLOSE_GRACE_S,
        calendar: Calendar | None = None,
    ) -> None:
        interval_ns = check_interval(interval_ns)
        watching_from_ns = _instant(watching_from_ns, "watching_from_ns")
        self._grace_ns = _grace_ns(close_grace_s)
        calendar = default_calendar() if calendar is None else calendar
        session = calendar.session(exchange, day)
        if session is None:
            raise CalendarError(
                f"{exchange} has no session on {day}: live bars are built on a trading day "
                "only; check the date, or add the session to the calendar"
            )
        grid = calendar.union_grid([exchange], day, day, base=f"{interval_ns // NANOS_PER_SECOND}s")
        closes = grid[f"close_{grid.attrs['exchanges'][0]}"].to_numpy()
        has_bar = closes != NO_BAR
        self._exchange = exchange
        self._index = exchange in INDEX_EXCHANGES
        self._session = session
        self._day_ns = _ist_day_bounds(day)
        self._opens: list[int] = grid["ts"].to_numpy()[has_bar].tolist()
        self._closes: list[int] = closes[has_bar].tolist()
        self._from_open = watching_from_ns <= session[0]
        # Ranges of slices that only seed, first and last inclusive: the slice watching
        # began in and any before it, then the slices of each feed outage.
        self._seeding: list[tuple[int, int]] = []
        if not self._from_open:
            self._seeding.append((0, bisect.bisect_left(self._opens, watching_from_ns) - 1))
        self._symbols: dict[str, SymbolTrades] = {}
        self._due_slots = 0
        self._arrivals = 0
        self._late = self._outside = self._invalid = self._event_rejected = 0
        self._noted: set[tuple[str, str]] = set()

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
        """Trades dropped because they were received after their slice's close plus the
        grace."""
        return self._late

    @property
    def outside_ticks(self) -> int:
        """Ticks outside the session."""
        return self._outside

    @property
    def invalid_ticks(self) -> int:
        """Ticks dropped for an unusable LTP or cumulative volume."""
        return self._invalid

    @property
    def event_time_rejected(self) -> int:
        """Ticks whose event time was not used for being ahead of receipt or of another
        day (a snapshot, a value in the wrong unit), placed by their receive time."""
        return self._event_rejected

    def on_tick(self, tick: Tick) -> None:
        """Take a tick for the slice its time falls in; bars come from `close_due`.

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
            return
        at_ns, snapshot = self._place(tick)
        symbol = self._symbol(tick.symbol)
        ltp = float(tick.ltp)
        volume = None if tick.volume is None else float(tick.volume)
        if not self._session[0] <= at_ns < self._session[1]:
            self._outside += 1
            self._note_once("outside", tick, logging.DEBUG, "ticks outside the session")
            if at_ns < self._session[0]:
                symbol.seed(tick.ts_recv_ns, ltp, volume)
            return
        slot = bisect.bisect_right(self._opens, at_ns) - 1
        if tick.ts_recv_ns >= self._closes[slot] + self._grace_ns or slot < self._due_slots:
            if not snapshot and symbol.late_trade(ltp, volume):
                self._late += 1
                self._note_once(
                    "late",
                    tick,
                    logging.WARNING,
                    "trades received after their slice closed (plus close_grace_s) are "
                    "dropped; the local clock may be ahead of the exchange",
                )
            return
        self._arrivals += 1
        symbol.add(slot, PlacedTick(tick.ts_recv_ns, self._arrivals, ltp, volume, snapshot))

    def close_due(self, now_ns: int) -> ClosedSlices:
        """Close every slice whose close plus the grace is at or before `now_ns` (UTC
        nanoseconds) and give out the bars of its trades. The loop hands over every tick
        received up to `now_ns` first."""
        first = self._due_slots
        due = bisect.bisect_right(self._closes, now_ns - self._grace_ns)
        self._due_slots = max(first, due)
        bars = [
            self._bar(name, slot, trades)
            for name, symbol in self._symbols.items()
            for slot, trades in symbol.close_before(self._due_slots)
            if not any(first <= slot <= last for first, last in self._seeding)
        ]
        bars.sort(key=lambda bar: bar.ts_open_ns)
        closed = tuple(
            zip(
                self._opens[first : self._due_slots],
                self._closes[first : self._due_slots],
                strict=True,
            )
        )
        return ClosedSlices(closed, tuple(bars))

    def open_so_far(self, symbol: str, ts_open_ns: int) -> float | None:
        """Rupees: the open `symbol`'s bar of the slice opening at `ts_open_ns` will have,
        from the ticks taken so far of a slice not closed yet: a bar's open is its first
        trade, which later ticks do not change. They are judged on a copy, so the slice
        closes as it would have. None for a slice with no trade so far, one closed or that
        only seeds, and an instant that opens no slice."""
        slot = bisect.bisect_left(self._opens, ts_open_ns)
        trades = self._symbols.get(symbol)
        seeding = any(first <= slot <= last for first, last in self._seeding)
        if trades is None or seeding or not self._due_slots <= slot < len(self._opens):
            return None
        if self._opens[slot] != ts_open_ns:
            return None
        judged = copy.deepcopy(trades).close_before(slot + 1)
        return next((bar.open for at, bar in judged if at == slot), None)

    def resume(self, outage_start_ns: int, now_ns: int) -> None:
        """The feed, out since `outage_start_ns`, resumed at `now_ns` (UTC nanoseconds).
        The slices from the one the outage began in to the one it ended in only seed, and
        from the first of them every symbol's volume level and last known LTP start
        again, as after a start in the middle of the session: slices completed before the
        outage keep their bars, and its volume lands on no bar. An outage that ended
        before the open or began at or after the close changes nothing.

        Raises ConfigError unless both are ints with the outage starting first.
        """
        start_ns = _instant(outage_start_ns, "outage_start_ns")
        if not start_ns <= _instant(now_ns, "now_ns"):
            raise ConfigError(f"the outage starts after it ends: {start_ns} > {now_ns}")
        if now_ns < self._session[0] or start_ns >= self._session[1]:
            return
        first = bisect.bisect_right(self._opens, max(start_ns, self._session[0])) - 1
        last = bisect.bisect_right(self._opens, min(now_ns, self._session[1] - 1)) - 1
        self._seeding.append((first, last))
        self._from_open = False
        for symbol in self._symbols.values():
            symbol.restart_from(first)

    def _symbol(self, name: str) -> SymbolTrades:
        symbol = self._symbols.get(name)
        if symbol is None:
            symbol = SymbolTrades(index=self._index, from_open=self._from_open)
            self._symbols[name] = symbol
        return symbol

    def _place(self, tick: Tick) -> tuple[int, bool]:
        """(the instant a tick is placed at, whether it is a snapshot of an earlier
        session): its event time when plausible, else its receive time."""
        event_ns, recv_ns = tick.ts_event_ns, tick.ts_recv_ns
        if not event_ns:
            return recv_ns, False
        on_day = self._day_ns[0] <= event_ns < self._day_ns[1]
        if on_day and abs(event_ns - recv_ns) <= EVENT_TIME_TOLERANCE_NS:
            return event_ns, False
        if on_day and event_ns < recv_ns:
            return recv_ns, False
        self._event_rejected += 1
        self._note_once(
            "event",
            tick,
            logging.WARNING,
            f"event times ahead of receipt by more than {_TOLERANCE_S} s or of another "
            "day are not used, and a tick of an earlier day is a snapshot that only seeds; "
            "such ticks are placed by their receive time",
        )
        return recv_ns, _EARLIEST_EVENT_NS <= event_ns < self._day_ns[0]

    def _note_once(self, kind: str, tick: Tick, level: int, text: str) -> None:
        """Log `text` for the tick's symbol the first time `kind` happens to it."""
        if (kind, tick.symbol) not in self._noted:
            self._noted.add((kind, tick.symbol))
            logger.log(level, "%s:%s: %s (first: %r)", tick.exchange, tick.symbol, text, tick)

    def _bar(self, symbol: str, slot: int, trades: SliceTrades) -> LiveBar:
        return LiveBar(
            symbol=symbol,
            exchange=self._exchange,
            ts_open_ns=self._opens[slot],
            ts_close_ns=self._closes[slot],
            open=trades.open,
            high=trades.high,
            low=trades.low,
            close=trades.close,
            volume=float(trades.volume),
            ticks=trades.trades,
            source=TICKS_SOURCE,
        )


__all__ = [
    "CANDLE_SOURCE",
    "DEFAULT_CLOSE_GRACE_S",
    "DEFAULT_DEAD_AFTER_S",
    "DEFAULT_INTERVAL_NS",
    "EVENT_TIME_TOLERANCE_NS",
    "TICKS_SOURCE",
    "BarDrift",
    "CandleCheck",
    "ClosedSlices",
    "FeedHealth",
    "TickBarBuilder",
    "candle_bar",
]

"""Which of one symbol's ticks are trades, and what the trades of a slice make (D27).
Part of `bars`: `TickBarBuilder` places the ticks in the slices and closes them.

A bar is made of trades only, as the broker's candles and the backtest's bars are: a
quote update with nothing traded makes no bar, so a minute without trades has none. The
ticks of a slice are judged when the slice closes, in the order they were received (ties
in the order they were handed over): the server sends one symbol's ticks in order, while
the times that place them mix event and receive times.

The feed's trade time never decides a trade: the SDK fills a missing field with 0, and
the server may set it to the packet's time or its own clock. A tick is a trade when:

- for a tradable instrument, its cumulative volume rose (when the feed gives a volume)
  or its LTP changed from the last known LTP. The first LTP only seeds the last known
  LTP, so an option not traded today, quoting the previous close with volume 0, makes
  no bar.
- for an index (on NSE_INDEX, BSE_INDEX or another index exchange), always: every tick
  of a watched slice is a print of its value, changed or not.

A tick that is a snapshot of an earlier session (its event time on an earlier day) is
never a trade: it only seeds the last known LTP.

Volume is the day's cumulative count, which never falls: a value at or below the highest
seen adds nothing. Watched from the open, a symbol's count rises from 0. After a start in
the middle of the session, or after the feed resumed, the first positive count only sets
the level: a trade it comes with still makes the bar, adding no volume (the candle check
corrects it). A symbol that reported a count of 0 then (it had not traded today) counts
its first positive count as traded volume. Prices are rupees, volume units, times UTC
nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlacedTick:
    """A tick taken for a slice: `recv_ns`, when it was received; `arrival`, its number in
    the order the ticks were handed over; its LTP; `volume`, its cumulative volume (0 when
    it reported 0) or None; `snapshot`, True when its event time is of an earlier day."""

    recv_ns: int
    arrival: int
    ltp: float
    volume: float | None
    snapshot: bool = False


@dataclass(frozen=True, slots=True)
class SliceTrades:
    """The trades of one symbol in one slice: their OHLC (rupees), the volume they added
    (units) and how many there were."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


class SymbolTrades:
    """One symbol's ticks of the slices not closed yet, and what the closed ones left:
    the highest cumulative volume (the level) and the last known LTP.

    `index` is True for an index; `from_open` is True when watching began at or before
    the session open.
    """

    def __init__(self, *, index: bool, from_open: bool) -> None:
        self._index = index
        self._level: float | None = 0.0 if from_open else None
        # A judged tick showed a positive count today: a later 0 is a missing count.
        self._counted_today = False
        self._last_price: float | None = None
        # The latest tick before the open (receive time, LTP): the first known LTP.
        self._pre_open: tuple[int, float] | None = None
        self._pending: dict[int, list[PlacedTick]] = {}
        # Slices from which the level and the last known LTP start again (a resume).
        self._restarts: list[int] = []
        # What the ticks taken so far show, to judge a tick that came too late.
        self._seen_volume: float | None = None
        self._seen_price: float | None = None

    def seed(self, recv_ns: int, ltp: float, volume: float | None) -> None:
        """Take a tick from before the session open: it only seeds the last known LTP."""
        if self._pre_open is None or recv_ns >= self._pre_open[0]:
            self._pre_open = (recv_ns, ltp)
        self._observe(ltp, volume)

    def add(self, slot: int, tick: PlacedTick) -> None:
        """Hold a tick for its slice until the slice closes."""
        self._pending.setdefault(slot, []).append(tick)
        self._observe(tick.ltp, tick.volume)

    def late_trade(self, ltp: float, volume: float | None) -> bool:
        """Whether a tick that came after its slice closed was a trade, judged against
        the ticks taken so far: an index tick always is; a tradable one when its volume
        is above their highest or its LTP differs from the last one's."""
        seen_volume, seen_price = self._seen_volume, self._seen_price
        self._observe(ltp, volume)
        if self._index:
            return True
        rose = volume is not None and seen_volume is not None and volume > seen_volume
        return rose or (seen_price is not None and ltp != seen_price)

    def restart_from(self, slot: int) -> None:
        """From `slot` on, forget the level and the last known LTP, as after a start in
        the middle of the session: the feed was out and its trades were not seen."""
        self._restarts.append(slot)
        self._restarts.sort()

    def close_before(self, end_slot: int) -> list[tuple[int, SliceTrades]]:
        """Judge the ticks of every held slice before `end_slot`, oldest slice first, and
        return (slice, trades) for each slice that had a trade."""
        closed = []
        for slot in sorted(slot for slot in self._pending if slot < end_slot):
            self._restart_through(slot)
            trades = self._judge(self._pending.pop(slot))
            if trades is not None:
                closed.append((slot, trades))
        self._restart_through(end_slot - 1)
        return closed

    def _restart_through(self, slot: int) -> None:
        while self._restarts and self._restarts[0] <= slot:
            self._restarts.pop(0)
            self._level = None
            self._last_price = None
            self._pre_open = None

    def _observe(self, ltp: float, volume: float | None) -> None:
        self._seen_price = ltp
        if volume:
            self._seen_volume = max(volume, self._seen_volume or volume)

    def _judge(self, ticks: list[PlacedTick]) -> SliceTrades | None:
        ticks.sort(key=lambda tick: (tick.recv_ns, tick.arrival))
        if self._last_price is None and self._pre_open is not None:
            self._last_price = self._pre_open[1]
        prices: list[float] = []
        volume = 0.0
        for tick in ticks:
            added = None if tick.snapshot else self._volume_if_trade(tick)
            if added is not None:
                prices.append(tick.ltp)
                volume += added
            self._last_price = tick.ltp
        if not prices:
            return None
        return SliceTrades(prices[0], max(prices), min(prices), prices[-1], volume, len(prices))

    def _volume_if_trade(self, tick: PlacedTick) -> float | None:
        """The volume a trade adds (0.0 when its count is unknown), or None when the tick
        is no trade."""
        rise = self._count(tick.volume)
        if self._index:
            return rise
        changed = self._last_price is not None and tick.ltp != self._last_price
        return rise if rise > 0 or changed else None

    def _count(self, volume: float | None) -> float:
        """Move the level by a tick's cumulative volume; the rise it brings."""
        if volume is None:
            return 0.0
        if volume == 0:
            if self._level is None and not self._counted_today:
                self._level = 0.0  # it reported 0: it had not traded today
            return 0.0
        self._counted_today = True
        if self._level is None:
            self._level = volume
            return 0.0
        if volume <= self._level:
            return 0.0
        rise, self._level = volume - self._level, volume
        return rise


__all__ = ["PlacedTick", "SliceTrades", "SymbolTrades"]

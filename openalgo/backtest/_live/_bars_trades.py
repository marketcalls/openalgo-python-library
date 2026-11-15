"""Which of one symbol's ticks are trades, and what the trades of a slice make (D27).
Part of `bars`: `TickBarBuilder` places the ticks in the slices and closes them.

A bar is made of trades only, as the broker's candles and the backtest's bars are: a
quote update with nothing traded makes no bar, so a minute without trades has none. The
ticks of a slice are judged when the slice closes, in the order of their times (ties by
receive time, then arrival), so the bars do not depend on the order the ticks arrived in.

The feed's trade time never decides a trade: the SDK fills a missing field with 0, and
the server may set it to the packet's time or its own clock. A tick is a trade when:

- for a tradable instrument, its cumulative volume rose (when the feed gives a volume)
  or its LTP changed from the last known LTP. The first LTP only seeds the last known
  LTP, so an option not traded today, quoting the previous close with volume 0, makes
  no bar.
- for an index (on NSE_INDEX, BSE_INDEX or another index exchange), its value changed;
  when watching began at or before the open, its first tick in the session opens its
  bar.

Volume is the day's cumulative count, which never falls: a value at or below the highest
seen adds nothing, and 0 (which the SDK sends for a missing count) is no value. Watched
from the open, a symbol's count rises from 0. After a start in the middle of the session
(or the feed's resumption) the first count only sets the level: a trade it comes with
still makes the bar, with that trade's volume unknown (0.0; the candle check corrects
it). Prices are rupees, volume units, times UTC nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlacedTick:
    """A tick placed in time. `at_ns` is its event time when plausible, else its receive
    time `recv_ns`; `arrival` numbers the ticks in the order they arrived; `volume` is
    its positive cumulative volume or None."""

    at_ns: int
    recv_ns: int
    arrival: int
    ltp: float
    volume: float | None


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
        self._last_price: float | None = None
        # An index watched from the open: its first tick in the session opens its bar.
        self._first_opens = index and from_open
        # The latest tick before the open (time, LTP): the first known LTP.
        self._pre_open: tuple[int, float] | None = None
        self._pending: dict[int, list[PlacedTick]] = {}
        # What the ticks taken so far show, to judge a tick that came too late.
        self._seen_volume: float | None = None
        self._seen_price: float | None = None

    def seed(self, at_ns: int, ltp: float, volume: float | None) -> None:
        """Take a tick from before the session open: it only seeds the last known LTP."""
        if self._pre_open is None or at_ns >= self._pre_open[0]:
            self._pre_open = (at_ns, ltp)
        self._observe(ltp, volume)

    def add(self, slot: int, tick: PlacedTick) -> None:
        """Hold a tick for its slice until the slice closes."""
        self._pending.setdefault(slot, []).append(tick)
        self._observe(tick.ltp, tick.volume)

    def late_trade(self, ltp: float, volume: float | None) -> bool:
        """Whether a tick that came after its slice closed was a trade, judged against
        the ticks taken so far (its volume above their highest, or its LTP other than the
        last one's; an index by its value)."""
        rose = volume is not None and self._seen_volume is not None and volume > self._seen_volume
        changed = self._seen_price is not None and ltp != self._seen_price
        self._observe(ltp, volume)
        return changed if self._index else (rose or changed)

    def restart(self) -> None:
        """Forget the level and the last known LTP, as after a start in the middle of the
        session: the feed resumed after an outage whose trades were not seen."""
        self._level = None
        self._last_price = None
        self._pre_open = None
        self._first_opens = False

    def close_before(self, end_slot: int) -> list[tuple[int, SliceTrades]]:
        """Judge the ticks of every held slice before `end_slot`, oldest slice first, and
        return (slice, trades) for each slice that had a trade."""
        closed = []
        for slot in sorted(slot for slot in self._pending if slot < end_slot):
            trades = self._judge(self._pending.pop(slot))
            if trades is not None:
                closed.append((slot, trades))
        return closed

    def _observe(self, ltp: float, volume: float | None) -> None:
        self._seen_price = ltp
        if volume is not None:
            self._seen_volume = max(volume, self._seen_volume or volume)

    def _judge(self, ticks: list[PlacedTick]) -> SliceTrades | None:
        ticks.sort(key=lambda tick: (tick.at_ns, tick.recv_ns, tick.arrival))
        if self._last_price is None and self._pre_open is not None:
            self._last_price = self._pre_open[1]
        prices: list[float] = []
        volume = 0.0
        for tick in ticks:
            added = self._volume_if_trade(tick)
            if added is not None:
                prices.append(tick.ltp)
                volume += added
            self._last_price = tick.ltp
        if not prices:
            return None
        return SliceTrades(prices[0], max(prices), min(prices), prices[-1], volume, len(prices))

    def _volume_if_trade(self, tick: PlacedTick) -> float | None:
        """The volume a trade adds (0.0 when its count is unknown), or None when the tick
        is no trade. Moves the symbol's level."""
        rise = 0.0
        if tick.volume is not None:
            if self._level is None:
                self._level = tick.volume
            elif tick.volume > self._level:
                rise, self._level = tick.volume - self._level, tick.volume
        first, self._first_opens = self._first_opens, False
        changed = self._last_price is not None and tick.ltp != self._last_price
        traded = (first or changed) if self._index else (rise > 0 or changed)
        return rise if traded else None


__all__ = ["PlacedTick", "SliceTrades", "SymbolTrades"]

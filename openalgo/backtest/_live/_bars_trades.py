"""Which of one symbol's ticks are trades, and what the trades of a slice make (D27).
Part of `bars`: `TickBarBuilder` places the ticks in the slices and closes them.

A bar is made of trades only, as the broker's candles and the backtest's bars are: a
quote update with nothing traded makes no bar, so a minute without trades has none. The
ticks of a slice are judged when the slice closes, in the order of their times (ties in
the order they arrived), so the bars do not depend on the order the ticks arrived in.
A tick is a trade when:

1. it carries a positive cumulative volume (the day's count) and that count rose. The
   count never falls, so a value at or below the highest seen is no trade. A symbol's
   first count rises from 0 when the session was watched from its open; after a start
   in the middle of the session it only sets the level later counts rise from.
2. else (no volume, or 0, which the SDK sends for a missing count) it carries a
   plausible last trade time and that time advanced: past the symbol's last one, or for
   its first one, at or after the instant watching began.
3. else its LTP differs from the symbol's previous LTP. A symbol's first tick in the
   session is a trade when the session was watched from its open and the tick carries
   no trade time at all (an index such as NIFTY): its first LTP opens its bar. The first
   LTP only seeds in the partial slice after a start in the middle of the session, and
   when the tick carries a trade time that is not plausible (a stale one, such as the
   previous day's last trade of an option not traded yet today).

Prices are rupees, volume units, times UTC nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlacedTick:
    """A tick placed in time. `at_ns` is its event time when plausible, else its receive
    time; `arrival` numbers the ticks in the order they arrived; `volume` is its positive
    cumulative volume or None; `trade_ns` its plausible last trade time or None;
    `time_rejected` is True when it carried a trade time (not 0, which means none) that
    was not plausible."""

    at_ns: int
    arrival: int
    ltp: float
    volume: float | None
    trade_ns: int | None
    time_rejected: bool = False


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
    the highest cumulative volume, the last trade time and the last LTP.

    `watch_start_ns` is the later of the session open and the instant watching began;
    `from_open` is True when watching began at or before the session open.
    """

    def __init__(self, *, watch_start_ns: int, from_open: bool) -> None:
        self._watch_start_ns = watch_start_ns
        self._from_open = from_open
        self._level: float | None = 0.0 if from_open else None
        self._last_trade_ns: int | None = None
        self._last_price: float | None = None
        self._pending: dict[int, list[PlacedTick]] = {}

    def add(self, slot: int, tick: PlacedTick) -> None:
        """Hold a tick for its slice until the slice closes."""
        self._pending.setdefault(slot, []).append(tick)

    def close_before(self, end_slot: int) -> list[tuple[int, SliceTrades]]:
        """Judge the ticks of every held slice before `end_slot`, oldest slice first, and
        return (slice, trades) for each slice that had a trade."""
        closed = []
        for slot in sorted(slot for slot in self._pending if slot < end_slot):
            trades = self._judge(self._pending.pop(slot))
            if trades is not None:
                closed.append((slot, trades))
        return closed

    def _judge(self, ticks: list[PlacedTick]) -> SliceTrades | None:
        ticks.sort(key=lambda tick: (tick.at_ns, tick.arrival))
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
        """The volume a trade adds (0.0 for a trade judged without a count), or None
        when the tick is no trade. Moves the symbol's level and last trade time."""
        last_trade_ns = self._last_trade_ns
        if tick.trade_ns is not None:
            self._last_trade_ns = max(tick.trade_ns, last_trade_ns or tick.trade_ns)
        if tick.volume is not None:
            level = self._level
            if level is None or tick.volume <= level:
                self._level = tick.volume if level is None else level
                return None
            self._level = tick.volume
            return tick.volume - level
        if tick.trade_ns is not None:
            if last_trade_ns is None:
                return 0.0 if tick.trade_ns >= self._watch_start_ns else None
            return 0.0 if tick.trade_ns > last_trade_ns else None
        if self._last_price is None:
            first_is_trade = self._from_open and not tick.time_rejected
            return 0.0 if first_is_trade else None
        return 0.0 if tick.ltp != self._last_price else None


__all__ = ["PlacedTick", "SliceTrades", "SymbolTrades"]

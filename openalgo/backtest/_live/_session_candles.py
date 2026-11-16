"""One strategy's broker candles in a live session (16-session-design 2.5 A.10 and B, 3.1
steps 11 to 15; unit 8): what the candle checks ask for, and what each read's candles do.

- A candle of a slice before the first slice released live (`first_live`) goes to the
  plan's catch-up, which releases the slices it still owns (`StrategyPlan.catch_up`,
  bounded below `first_live`); those whose last candle check passed with none read are
  given up on as a read with no candle would (`give_up`).
- A candle of a live slice already released is skipped when it equals the bar the series
  has; checked against the slice's tick bar when it has one (`CandleCheck.accepted`: the
  candle stands for the bar when it covers the slice, and its drift is kept); taken as
  the slice's bar when it made none (a minute with no tick, a seed minute). The kernel
  keeps the bar it ran.
- A slice whose candle no read holds, while a read begun `CANDLE_LAST_READ_S` or more
  after its close holds a candle of a later slice, had no trade: its missing mark is
  lifted without a bar (`LiveSeries.no_trade`) and it is asked for no more.

`wanted` lists what the next candle check asks for: the live slices closed without an
accepted candle, the slices marked missing, and the slices before `first_live` closed by
then and not released. Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._bars_candle import BarDrift, CandleCheck, candle_bar
from ._plan_catch_up import StrategyPlan
from ._plan_release import CANDLE_LAST_READ_S, missing_note
from ._types import LiveBar
from .bars import DEFAULT_INTERVAL_NS

_LAST_READ_NS = round(CANDLE_LAST_READ_S * NANOS_PER_SECOND)


class CandleKeeper:
    """The broker candles of `part`, a strategy's share of the plan (module docstring);
    `event` writes the event stream (each accepted drift)."""

    def __init__(self, part: StrategyPlan, *, event: Callable[..., None]) -> None:
        self._part, self._event = part, event
        self._opens = [opened for opened, _ in part.slices]
        self._closes = {opened: closed for opened, closed in part.slices}
        self._first_live: int | None = None
        self._needs: set[int] = set()
        self._checks = CandleCheck()

    @property
    def drifts(self) -> list[BarDrift]:
        """The drift of every candle accepted for a tick bar, in order."""
        return self._checks.drifts

    def watch(self, first_live: int) -> None:
        """The session releases the slices from index `first_live` live."""
        self._first_live = first_live

    def closed(self, slice_open_ns: int) -> None:
        """A live slice closed: its candle is asked for until one is accepted."""
        self._needs.add(slice_open_ns)

    def wanted(self, now_ns: int) -> tuple[int, ...]:
        """The opens of the slices the next candle check asks for (module docstring)."""
        part = self._part
        gap = {
            opened
            for opened, closed in part.slices[part.released : self._first_live or 0]
            if closed <= now_ns
        }
        return tuple(sorted(self._needs | set(part.missing) | gap))

    def take(self, rows: Iterable[Any], read_ns: int) -> list[str]:
        """Take a read of the broker's candles begun at `read_ns` (module docstring), each
        row as `bars.candle_bar` reads it; the warnings of the slices the catch-up marked
        missing."""
        part = self._part
        spec = part.spec
        candles = [
            candle_bar(spec.symbol, spec.exchange, row, DEFAULT_INTERVAL_NS, calendar=part.calendar)
            for row in rows
        ]
        # Once the session trades the kernel's commands are the session's to take (an exit
        # sent when a cancel ends, an engine exit), not a catch-up's to refuse.
        marked = part.catch_up(candles, read_ns, before=self._first_live, refuse_orders=False)
        for candle in candles:
            index = bisect.bisect_left(self._opens, candle.ts_open_ns)
            if self._first_live is not None and self._first_live <= index < part.released:
                self._live(candle)
        self._no_trade({candle.ts_open_ns for candle in candles}, read_ns)
        return self._notes(marked, none_today=not candles)

    def give_up(self, now_ns: int) -> list[str]:
        """Release, as a read with no candle begun at `now_ns` would, the slices before
        `first_live` whose last candle check has passed; their warnings."""
        part = self._part
        pending = part.slices[part.released : self._first_live or 0]
        if not pending or now_ns < pending[0][1] + _LAST_READ_NS:
            return []
        marked = part.catch_up((), now_ns, before=self._first_live, refuse_orders=False)
        return self._notes(marked, none_today=True)

    def _live(self, candle: LiveBar) -> None:
        series, opened = self._part.series, candle.ts_open_ns
        if series.bar(opened) == candle:
            self._needs.discard(opened)
            return
        tick_bar = series.tick_bar(opened)
        if tick_bar is None:
            series.take(candle)
            self._needs.discard(opened)
            return
        if self._checks.accepted(tick_bar, candle) is candle:
            series.take(candle)
            self._needs.discard(opened)
            drift = self._checks.drifts[-1]
            self._event(
                "drift",
                strategy=self._part.spec.name,
                slice_open_ns=opened,
                open=drift.open,
                high=drift.high,
                low=drift.low,
                close=drift.close,
                volume=drift.volume,
            )

    def _no_trade(self, read: set[int], read_ns: int) -> None:
        if not read:
            return
        latest, part = max(read), self._part
        released = set(self._opens[: part.released])
        for opened in sorted(self._needs | set(part.missing)):
            done = read_ns >= self._closes[opened] + _LAST_READ_NS
            if opened in released and opened < latest and opened not in read and done:
                part.series.no_trade(opened)
                self._needs.discard(opened)

    def _notes(self, marked: tuple[int, ...], *, none_today: bool) -> list[str]:
        if not marked:
            return []
        spec = self._part.spec
        return [missing_note(spec.name, spec.symbol, spec.exchange, marked, none_today=none_today)]


__all__ = ["CandleKeeper"]

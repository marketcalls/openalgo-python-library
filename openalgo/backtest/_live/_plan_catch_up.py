"""One strategy's part of the session plan, `StrategyPlan` (16-session-design 3.1 steps 11
to 15, 2.5 B; units 7 and 8): its kernel session, live series and strategy, the slices of
the day released to the kernel, and the catch-up that releases them from broker candles.
Part of `plan`, which re-exports `StrategyPlan` and describes the catch-up.

Slices are released in order, from the open. The catch-up releases those before the
session watches the feed (`catch_up`, bounded by `before` once the session watches); the
session loop releases the later ones as their slices close (`release_next`), each with
the bar built from the feed's ticks or none.

Times are UTC nanoseconds. Not thread-safe: the main thread builds the plan, and the
session loop owns it afterwards.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Iterable
from typing import Any

from .._calendar_convert import format_ist
from .._driver import deliver
from .._engine import call
from .._warm_up import _ready_at
from ..calendar import Calendar
from ..errors import ConfigError, LiveError
from ..strategy import Strategy
from ._bars_candle import CANDLE_SOURCE
from ._plan_release import MISSING, SETTLE_NS, releases
from ._plan_setup import StrategySetup
from ._plan_strategies import StrategySpec
from ._types import LOGGER_NAME, LiveBar
from .series import LiveRunContext, LiveSeries

#: Each strategy trades one instrument, registered first (1.1).
INSTRUMENT = 0

logger = logging.getLogger(LOGGER_NAME)


class StrategyPlan:
    """One strategy's part of the plan (`plan` module docstring): its `spec`, its `setup`,
    its kernel session (`engine`), its live `series`, the strategy instance and its live
    run `context`. `released` counts the slices of the day released to the kernel, by
    the catch-up and then by the session loop. `kernel_instruments` are the binding's
    instrument dicts the kernel session was built on (`_plan_engine.kernel_instruments`;
    none for a plan built without them), which the session saves."""

    #: The binding's instrument dicts of the kernel session.
    kernel_instruments: tuple[dict[str, Any], ...] = ()

    def __init__(
        self,
        spec: StrategySpec,
        setup: StrategySetup,
        engine: Any,
        series: LiveSeries,
        strategy: Strategy,
        context: LiveRunContext,
        calendar: Calendar,
    ) -> None:
        self.spec, self.setup, self.engine = spec, setup, engine
        self.series, self.strategy, self.context = series, strategy, context
        self.calendar = calendar
        self._slices = series.slices
        self._closes = [close for _, close in self._slices]
        self._released = 0

    @property
    def released(self) -> int:
        """How many slices of the day, from the open, are released to the kernel."""
        return self._released

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        """(open, close) of every slice of the day, in order (UTC nanoseconds)."""
        return self._slices

    @property
    def missing(self) -> tuple[int, ...]:
        """The opens (UTC nanoseconds) of the released slices marked missing whose candle
        has not been read since, oldest first: the session's candle checks ask for them
        (2.5 B)."""
        if not self._released:
            return ()
        return self.series.missing_slices(self._slices[0][0], self._closes[self._released - 1])

    def uncaught(self, read_ns: int, *, before: int | None = None) -> tuple[int, ...]:
        """The opens of the slices settled at a read of the broker's candles begun at
        `read_ns` (UTC nanoseconds), those closed by `read_ns` less `CANDLE_SETTLE_S`,
        that are not released yet, oldest first; with `before`, only those of index below
        it (the slices the catch-up still owns once the session watches the feed)."""
        return tuple(opened for opened, _ in self._settled(read_ns, before))

    def catch_up(
        self,
        bars: Iterable[LiveBar],
        read_ns: int,
        *,
        before: int | None = None,
        refuse_orders: bool = True,
    ) -> tuple[int, ...]:
        """Catch up to a read of today's candles begun at `read_ns` (UTC nanoseconds; the
        start of the read, not its end, since a slice that settled during it may be
        missing from it) whose candles are `bars` (the strategy's broker candles of
        today, `source` "candle"; none for a read that held none). The slices settled at
        the read (`uncaught`, below index `before` when given) are released by the rule
        of the `plan` module docstring, each run deciding nothing, and a candle of a
        slice marked missing is taken by the series. Candles of the other slices
        released already are left alone. Returns the opens of the slices it marked
        missing, oldest first. Once the session trades (`refuse_orders` False) the
        kernel's commands are the session's to take: an exit sent when a cancel ends, an
        engine exit.

        Raises ConfigError for a bar that is not a candle of the strategy's instrument,
        and LiveError when the catch-up made an order (with `refuse_orders`) or the
        exchange's session ended."""
        candles = self._candles(bars)
        for opened in self.missing:
            if opened in candles:
                self.series.take(candles[opened])
        found = releases(self._settled(read_ns, before), candles, read_ns)
        for release in found:
            bar = release.bar
            if bar is not None:
                self.series.take(bar)
                self._add(bar)
            elif release.kind == MISSING:
                self.series.mark_missing(release.opened)
            self._release()
            self._drive()
        if found and refuse_orders:
            self._refuse_orders()
        return tuple(release.opened for release in found if release.kind == MISSING)

    def release_next(self, bar: LiveBar | None, *, missing: bool = False) -> int:
        """Release the next slice of the day to the kernel live, as the session loop does
        once the slice has closed: with `bar`, the slice's bar built from the feed's
        ticks (taken by the series and added to the kernel), or with none (the kernel
        runs a missing bar; `missing` marks the slice missing in the series, so no
        decision is made on a bar holding it until its candle is read). The kernel's
        pauses are the session's to drive. Returns the slice's index.

        Raises ConfigError for a bar that is not the next slice's, and LiveError when
        every slice of the day is released."""
        index = self._released
        if index >= len(self._slices):
            raise LiveError(f"{self.spec.name}: every slice of the day is released already")
        opened = self._slices[index][0]
        if bar is not None:
            if bar.ts_open_ns != opened:
                raise ConfigError(
                    f"{self.spec.name}: the bar opening at {format_ist(bar.ts_open_ns)} is "
                    f"not the next slice's, which opens at {format_ist(opened)}"
                )
            self.series.take(bar)
            self._add(bar)
        elif missing:
            self.series.mark_missing(opened)
        self._release()
        return index

    def indicators_ready(self, decided_at_ns: int) -> bool:
        """True when every indicator need of the strategy is met by the bars closed at
        the decision `decided_at_ns` (D25: the need-th bar of its series, history
        counted, or for an anchored need its whole periods), as the backtest's warm-up
        rule finds it (`_warm_up`): the session decides only then (2.5 C)."""
        return all(_ready_at(need, self.context) <= decided_at_ns for need in self.spec.needs)

    def _add(self, bar: LiveBar) -> None:
        prices = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        call(self.engine.add_bar, INSTRUMENT, (bar.ts_open_ns, *prices))

    def _release(self) -> None:
        self._released += 1
        call(self.engine.release, self._released)

    def _settled(self, read_ns: int, before: int | None) -> list[tuple[int, int]]:
        """(open, close) of the slices settled at a read begun at `read_ns` and not
        released yet, below index `before` when given, oldest first."""
        end = bisect.bisect_right(self._closes, read_ns - SETTLE_NS)
        if before is not None:
            end = min(end, before)
        return list(self._slices[self._released : end])

    def _candles(self, bars: Iterable[LiveBar]) -> dict[int, LiveBar]:
        """`bars` by open, each checked to be a broker candle of the instrument."""
        found: dict[int, LiveBar] = {}
        for bar in bars:
            if bar.source != CANDLE_SOURCE:
                raise ConfigError(f"the catch-up takes broker candles, not {bar.source!r} bars")
            if (bar.symbol, bar.exchange) != (self.spec.symbol, self.spec.exchange):
                raise ConfigError(
                    f"a candle of {bar.exchange}:{bar.symbol} reached the catch-up of "
                    f"{self.spec.name}, which trades {self.spec.exchange}:{self.spec.symbol}"
                )
            found[bar.ts_open_ns] = bar
        return found

    def _drive(self) -> None:
        """Run the kernel's pauses until it waits, revealing bars and delivering events,
        deciding nothing (no `on_bar`, no timer)."""
        while True:
            kind, decided_at, rows, events, _timers = call(self.engine.advance)
            if kind == "waiting":
                return
            self.context.reveal(decided_at, rows)
            if events:
                deliver(self.strategy, self.context, events)
            if kind == "finished":
                raise LiveError(
                    f"{self.spec.exchange}'s session of {self.spec.plan.start} ended before "
                    f"the plan of {self.spec.name} was built: nothing is left to trade today"
                )
            if kind == "timers":
                logger.info(
                    "%s: the timers due at %s were missed before the session started and are "
                    "skipped",
                    self.spec.name,
                    format_ist(decided_at),
                )

    def _refuse_orders(self) -> None:
        """Raise LiveError when the catch-up made an order: a catch-up sends nothing, and
        an order from a callback run on a past bar is not a decision of now."""
        commands = call(self.engine.take_commands)
        if commands:
            raise LiveError(
                f"{self.spec.name}: {len(commands)} orders were made while the session caught "
                "up on today's past bars, and a catch-up sends nothing: place orders in "
                "on_bar, which runs once the session trades"
            )


__all__ = ["INSTRUMENT", "StrategyPlan"]

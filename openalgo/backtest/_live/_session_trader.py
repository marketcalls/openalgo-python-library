"""One strategy of a live session as the session loop runs it (16-session-design 2.5 B, C
and A.10, 3.4; unit 8): its part of the plan (kernel session, live series, strategy), its
venue map, the slices it releases live, its decisions and its broker candles.

- **Releases.** From the slice the session began watching the feed in (`first_live`),
  each slice is released to the kernel once the bar builder has closed it, in order,
  with its bar built from ticks or none (a minute without a trade runs as a missing bar).
  The slices before it that the plan's catch-up has not released yet are released from
  the broker's candles first (`candles`), or given up on at their last candle check; a
  live slice waits behind them. After a start in the middle of the
  session, the slice watching began in only seeds the bar builder: it is released with no
  bar and marked missing until its candle is read.
- **Decisions** (`drive`): at each strategy bar close the bars are revealed and the
  kernel's events delivered; `on_bar` runs when the strategy decides: trading has begun
  (the decision at or after `decide_from_ns`), its indicators are ready, and no base
  slice of the bar, nor of the bar before it in the series, is marked missing: a
  crossing reads that bar too, and the candle that fills such a hole (the slice
  watching began in) usually comes seconds later (`_unread`). Timers fire on the same
  terms. A skipped decision is logged, and kept when it was skipped for missing data. An
  exception of the strategy's code is raised as `StrategyFailed` once the kernel waits
  again (it refuses the venue's results at a bar close), after which the strategy is
  called no more. Every strategy bar gets a line in the decisions file, the one whose
  `on_bar` raised among them, and each order the strategy's code submitted is looked at
  for a rejection (`notes`, `_session_decisions.DecisionNotes`).
- **Orders**: the kernel's commands as the order worker's requests, its position and
  working orders, and the exits and cancels the session asks of it are
  `_session_orders.TraderOrders`'.
- **Candles**: the strategy's broker candles are `candles`, a `_session_candles.
  CandleKeeper`.
- **Bookkeeping**: its entry locks (shared with the order worker through
  `locked_entries`) and its saved `self.state` are `_session_bookkeeping.Bookkeeping`'s.

Times are UTC nanoseconds, prices rupees, quantities units. Used by the loop only.
"""

from __future__ import annotations

import bisect
import datetime as dt
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .._calendar_convert import IST, format_ist
from .._driver import deliver
from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._locked_entries import LockedEntries
from ._plan_catch_up import StrategyPlan
from ._session_bookkeeping import Bookkeeping
from ._session_candles import CandleKeeper
from ._session_decisions import DecisionNotes
from ._session_orders import TraderOrders
from ._session_records import StrategyRecords
from ._types import LOGGER_NAME, LiveBar, Now
from .venue_map import VenueMap

logger = logging.getLogger(LOGGER_NAME)

#: An `on_bar` that takes longer is warned about, once per strategy (16-session-design
#: 2.5 C): the loop does nothing else meanwhile, so ticks and order news wait for it.
SLOW_CALLBACK_WARN_S = 1.0
#: Why a decision was skipped (`Skipped.reason`).
SKIP_MISSING = "missing data"

_SLOW_NS = round(SLOW_CALLBACK_WARN_S * NANOS_PER_SECOND)


class StrategyFailed(Exception):
    """The code of strategy `strategy` raised `error` (in `on_bar`, a callback or a
    timer), after it `submitted` what a decision line lists (`DecisionNotes.submitted`).
    The session stops safely, then raises `error` again (16-session-design 3.5)."""

    def __init__(
        self, strategy: str, error: BaseException, submitted: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(f"{strategy} raised {type(error).__name__}: {error}")
        self.strategy = strategy
        self.error = error
        self.submitted = submitted or []


@dataclass(frozen=True)
class Skipped:
    """A strategy bar closing at `decided_at_ns` whose `on_bar` was skipped, and why."""

    strategy: str
    decided_at_ns: int
    reason: str


def ist_instant(day: dt.date, hhmm: str) -> int:
    """UTC nanoseconds of the IST wall-clock time "HH:MM" on `day`."""
    clock = dt.time.fromisoformat(hhmm)
    return int(dt.datetime.combine(day, clock, tzinfo=IST).timestamp()) * NANOS_PER_SECOND


class Trader(Bookkeeping, TraderOrders):
    """One strategy of the session (module docstring): `part`, its share of the plan,
    `venue`, its venue map; `square_off_ns` its MIS square-off and `close_ns` its
    exchange's session close (UTC nanoseconds); `now` the session's clock, `event` the
    event stream's writer and `decision` the decisions file's; `locked_entries` the order
    worker's record of the strategies whose entries are locked (`_locked_entries`)."""

    def __init__(
        self,
        part: StrategyPlan,
        venue: VenueMap,
        *,
        square_off_ns: int,
        close_ns: int,
        now: Now,
        event: Callable[..., None],
        decision: Callable[..., None],
        locked_entries: LockedEntries,
    ) -> None:
        self.part, self.venue = part, venue
        self.locked_entries = locked_entries
        self.name, self.symbol = part.spec.name, part.spec.symbol
        self.exchange = part.spec.exchange
        self.square_off_ns, self.close_ns = square_off_ns, close_ns
        self._now, self._event = now, event
        self._opens = [opened for opened, _ in part.slices]
        self.candles = CandleKeeper(part, event=event)
        self.notes = DecisionNotes(self.name, part, decision)
        self.first_live: int | None = None
        self.decide_from_ns: int | None = None
        self._seed: int | None = None
        self._closed: dict[int, LiveBar | None] = {}
        self.skipped: list[Skipped] = []
        #: The kernel's reason of each risk exit placed, by kernel order.
        self.reasons: dict[int, str] = {}
        #: The position's units (unsigned) when each order's placement was taken from the
        #: kernel, by kernel order: what an exit was for part of (`_session_exits`).
        self.units_at_place: dict[int, int] = {}
        self.failed = False
        #: True once the kernel session has finished: it takes no call that changes it.
        self.finished = False
        #: Its entry locks and the round trips it closed, for the result.
        self.records = StrategyRecords(self.name)
        self._slow_warned = self._state_warned = False
        self._state_text = self._state_json()

    # ----- releases ----------------------------------------------------------------

    def watch(self, watching_from_ns: int, *, from_open: bool) -> None:
        """The session watches the feed from `watching_from_ns` (`from_open` when that is
        at or before the session's open): set the first slice released live and the first
        decision, the close of the first whole slice watched (16-session-design 3.1 step
        14). A slice the bar builder only seeds (the one watching began in, after the open)
        is released with no bar and marked missing."""
        opens, closes = self._opens, [closed for _, closed in self.part.slices]
        if from_open:
            first, seed = 0, None
        else:
            index = bisect.bisect_left(opens, watching_from_ns)
            if index < len(opens) and opens[index] == watching_from_ns:
                first, seed = index, None
            else:
                first = seed = max(0, index - 1)
        self.first_live, self._seed = first, seed
        self.candles.watch(first)
        whole = first if seed is None else first + 1
        self.decide_from_ns = closes[min(whole, len(closes) - 1)]

    def closed(self, slices: Sequence[tuple[int, int]], bars: Iterable[LiveBar]) -> None:
        """The bar builder closed `slices` ((open, close) in order) with `bars` (those of
        every symbol of the exchange): note this strategy's live slices among them."""
        mine = {bar.ts_open_ns: bar for bar in bars if bar.symbol == self.symbol}
        for opened, _ in slices:
            index = bisect.bisect_left(self._opens, opened)
            if self.first_live is None or index < self.first_live:
                continue
            self._closed[index] = mine.get(opened)
            self.candles.closed(opened)

    def release_due(self) -> int:
        """Release every closed live slice that is next in order; how many."""
        part, released = self.part, 0
        while part.released in self._closed:
            index = part.released
            bar = self._closed.pop(index)
            part.release_next(bar, missing=bar is None and index == self._seed)
            released += 1
        return released

    # ----- decisions ---------------------------------------------------------------

    def drive(self, *, deciding: bool) -> None:
        """Run the kernel's pauses until it waits (module docstring). `deciding` False
        reveals and delivers but decides nothing (the session is stopping). A kernel
        session that finished by itself (the exchange's close, with nothing working) calls
        `on_stop`, as a backtest's end does, and is driven no more (`finished`). Raises
        StrategyFailed for an exception of the strategy's code, once the kernel waits
        again: the pauses after it are run through without calling the strategy, so the
        order news the loop takes next is never refused at a bar close."""
        failure: StrategyFailed | None = None
        while not self.finished:
            pause = call(self.part.engine.advance)
            if pause[0] == "waiting":
                break
            try:
                self._pause(*pause, deciding=deciding)
            except StrategyFailed as failed:
                failure = failure or failed
        if failure is not None:
            raise failure

    def _pause(
        self,
        kind: str,
        decided_at: int,
        rows: list[tuple],
        events: list[Any],
        timers: Any,
        *,
        deciding: bool,
    ) -> None:
        """One pause of the kernel: reveal its bars, deliver its events, then decide."""
        part = self.part
        bars = part.context.reveal(decided_at, rows)
        if kind == "finished":
            # Set first: a callback that raises here must leave no call to a finished
            # kernel session behind it (the safe stop's exit, say).
            self.finished = True
            logger.info("%s: the session's close ended its kernel session", self.name)
        if events and not self.failed:
            self._calling(deliver, part.strategy, part.context, events)
        if kind == "finished":
            if not self.failed:
                self._calling(part.strategy.on_stop)
            return
        decide_from_ns = self.decide_from_ns
        decides = deciding and not self.failed
        decides = decides and decide_from_ns is not None and decided_at >= decide_from_ns
        if kind == "timers":
            if decides:
                self._calling(part.context.timers.fire, timers)
            else:
                logger.info("%s: timers due at %s skipped", self.name, format_ist(decided_at))
        elif bars and decides:
            self._decide(next(iter(bars.values())), decided_at)
        elif bars:
            self.notes.not_deciding(decided_at, failed=self.failed, deciding=deciding)

    def _decide(self, bar: Any, decided_at: int) -> None:
        part = self.part
        if not part.indicators_ready(decided_at):
            self.notes.not_ready(decided_at)
            return
        missing = self._unread(bar.ts_open, bar.ts_close)
        if missing:
            skipped = Skipped(self.name, decided_at, SKIP_MISSING)
            self.skipped.append(skipped)
            logger.warning(
                "%s: the decision at %s is skipped: its bar or the bar before it holds %d "
                "minute(s) with no data (from %s)",
                self.name,
                format_ist(decided_at),
                len(missing),
                format_ist(missing[0]),
            )
            self._event("skipped_decision", strategy=self.name, decided_at_ns=decided_at)
            self.notes.skipped(decided_at, SKIP_MISSING)
            return
        # What code outside a pause submitted (a catch-up's callbacks) is not this decision's.
        self.notes.submitted()
        started = self._now()
        try:
            submitted = self._calling(part.strategy.on_bar, bar)
        except StrategyFailed as failed:
            # The bar the strategy failed on is the one most worth a line.
            self.notes.decided(
                decided_at, bar, failed.submitted, self.open_units(), error=failed.error
            )
            raise
        if self._now() - started > _SLOW_NS and not self._slow_warned:
            self._slow_warned = True
            logger.warning(
                "%s: on_bar took more than %s s: ticks and order news wait meanwhile",
                self.name,
                SLOW_CALLBACK_WARN_S,
            )
        self.notes.decided(decided_at, bar, submitted, self.open_units())

    def _unread(self, ts_open_ns: int, ts_close_ns: int) -> tuple[int, ...]:
        """The opens (UTC nanoseconds, oldest first) of the slices marked missing in the
        strategy bar [ts_open_ns, ts_close_ns) or in the bar before it in the strategy's
        series: a crossing, `shift(1)` or `[-2]` reads that bar too, so a hole there (the
        minute watching began in, after a start in the middle of the session, until its
        candle is read) would make the decision on values a backtest never has. A hole
        further back skips no decision: a minute whose candle never comes holds up one
        decision more, not the rest of the day."""
        opens = self.part.context.store(self.symbol).column("ts_open")
        since = int(opens[-2]) if len(opens) > 1 else ts_open_ns
        return self.part.series.missing_slices(since, ts_close_ns)

    def _calling(self, function: Callable[..., Any], *args: Any) -> list[dict[str, Any]]:
        """Call the strategy's code; what it submitted (`DecisionNotes.submitted`)."""
        try:
            function(*args)
        # The strategy's own code: any exception it raises stops the session safely.
        except Exception as error:
            self.failed = True
            raise StrategyFailed(self.name, error, self.notes.submitted()) from error
        return self.notes.submitted()


__all__ = [
    "SKIP_MISSING",
    "SLOW_CALLBACK_WARN_S",
    "Skipped",
    "StrategyFailed",
    "Trader",
    "ist_instant",
]

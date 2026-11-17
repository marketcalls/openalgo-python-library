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
  slice of the bar is marked missing. Timers fire on the same terms. A skipped decision
  is logged, and kept when it was skipped for missing data. An exception of the
  strategy's code is raised as `StrategyFailed` once the kernel waits again (it refuses
  the venue's results at a bar close), after which the strategy is called no more.
- **Commands** (`commands`): the kernel's commands as the order worker's requests. A
  placement the kernel cancelled before the loop took it (the safe stop's cancel of an
  entry decided in the same loop step) is withdrawn: it is never sent, and the kernel is
  told it was cancelled (`apply_cancel`).
- **Candles**: the strategy's broker candles are `candles`, a `_session_candles.
  CandleKeeper`.

Times are UTC nanoseconds, prices rupees, quantities units. Used by the loop only.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .._calendar_convert import IST, format_ist
from .._driver import deliver
from .._engine import call
from .._layout import NANOS_PER_SECOND
from ..errors import JournalError, LiveError
from ._journal_rows import state_json
from ._plan_catch_up import INSTRUMENT, StrategyPlan
from ._session_candles import CandleKeeper
from ._types import LOGGER_NAME, LiveBar, Now
from ._venue_requests import LIVE_PRODUCT, OrderRequest
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
    timer). The session stops safely, then raises `error` again (16-session-design 3.5)."""

    def __init__(self, strategy: str, error: BaseException) -> None:
        super().__init__(f"{strategy} raised {type(error).__name__}: {error}")
        self.strategy = strategy
        self.error = error


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


class Trader:
    """One strategy of the session (module docstring): `part`, its share of the plan,
    `venue`, its venue map; `square_off_ns` its MIS square-off and `close_ns` its
    exchange's session close (UTC nanoseconds); `now` the session's clock and `event`
    the event stream's writer."""

    def __init__(
        self,
        part: StrategyPlan,
        venue: VenueMap,
        *,
        square_off_ns: int,
        close_ns: int,
        now: Now,
        event: Callable[..., None],
    ) -> None:
        self.part, self.venue = part, venue
        self.name, self.symbol = part.spec.name, part.spec.symbol
        self.exchange = part.spec.exchange
        self.square_off_ns, self.close_ns = square_off_ns, close_ns
        self._now, self._event = now, event
        self._opens = [opened for opened, _ in part.slices]
        self.candles = CandleKeeper(part, event=event)
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
        self._slow_warned = self._state_warned = False
        self._state_text = self._state_json()

    @property
    def engine(self) -> Any:
        """The strategy's kernel session."""
        return self.part.engine

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

    def _decide(self, bar: Any, decided_at: int) -> None:
        part = self.part
        if not part.indicators_ready(decided_at):
            logger.debug("%s: indicators not ready at %s", self.name, format_ist(decided_at))
            return
        missing = part.series.missing_slices(bar.ts_open, bar.ts_close)
        if missing:
            skipped = Skipped(self.name, decided_at, SKIP_MISSING)
            self.skipped.append(skipped)
            logger.warning(
                "%s: the decision at %s is skipped: the bar holds %d minute(s) with no data "
                "(from %s)",
                self.name,
                format_ist(decided_at),
                len(missing),
                format_ist(missing[0]),
            )
            self._event("skipped_decision", strategy=self.name, decided_at_ns=decided_at)
            return
        started = self._now()
        self._calling(part.strategy.on_bar, bar)
        if self._now() - started > _SLOW_NS and not self._slow_warned:
            self._slow_warned = True
            logger.warning(
                "%s: on_bar took more than %s s: ticks and order news wait meanwhile",
                self.name,
                SLOW_CALLBACK_WARN_S,
            )

    def _calling(self, function: Callable[..., Any], *args: Any) -> None:
        try:
            function(*args)
        # The strategy's own code: any exception it raises stops the session safely.
        except Exception as error:
            self.failed = True
            raise StrategyFailed(self.name, error) from error

    # ----- orders and positions ----------------------------------------------------

    def commands(self) -> list[OrderRequest]:
        """The kernel's commands since the last call, as the order worker's requests (a
        placement the kernel cancelled before this call withdrawn: module docstring); the
        reason of each risk exit placed is kept (`reasons`), and the position's units at
        each placement (`units_at_place`)."""
        if self.finished:
            return []
        requests: list[OrderRequest] = []
        commands = call(self.engine.take_commands)
        while commands:
            kept = self._withdraw_cancelled(commands)
            for command in kept:
                if command.get("command") != "place":
                    continue
                kernel_order = int(command["order_id"])
                self.units_at_place[kernel_order] = round(abs(self.open_units()))
                if command.get("reason"):
                    self.reasons[kernel_order] = str(command["reason"])
            requests += self.venue.requests(kept)
            # A withdrawal ends an order in the kernel, which may free an exit that waited
            # for it: its commands go out in this same call.
            commands = call(self.engine.take_commands) if len(kept) < len(commands) else []
        return requests

    def _withdraw_cancelled(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`commands` without those of the orders both placed and cancelled in them, each
        such order confirmed cancelled to the kernel (it never reached the broker)."""
        placed = {int(c["order_id"]) for c in commands if c.get("command") == "place"}
        cancelled = {int(c["order_id"]) for c in commands if c.get("command") == "cancel"}
        withdrawn = placed & cancelled
        for kernel_order in sorted(withdrawn):
            call(self.engine.apply_cancel, kernel_order)
            logger.info("%s: order o%d was cancelled before it was sent", self.name, kernel_order)
            self._event("withdrawn", strategy=self.name, kernel_order=kernel_order)
        return [c for c in commands if int(c["order_id"]) not in withdrawn]

    def open_units(self) -> float:
        """The kernel's position in the strategy's instrument, signed units (0 flat)."""
        position = call(self.engine.position, INSTRUMENT, LIVE_PRODUCT)
        return 0.0 if position is None else float(position["quantity_units"])

    def working(self) -> list[int]:
        """The kernel orders that can still fill."""
        return list(call(self.engine.working_orders))

    def order_name(self, kernel_order: int) -> str:
        """How an alert names a kernel order: the broker's order ID once bound, else
        "o<n>" (also for an order the kernel made in this loop step, which the venue map
        has not taken yet)."""
        try:
            order_id = self.venue.order_id_of(kernel_order)
        # The venue map knows only the orders whose commands it has taken.
        except LiveError:
            order_id = ""
        return order_id or f"o{kernel_order}"

    def exit_orders(self) -> list[int]:
        """The working orders that close the position (reduce-only)."""
        entries = set(self.entry_orders())
        return [kernel_order for kernel_order in self.working() if kernel_order not in entries]

    @property
    def flat(self) -> bool:
        """True with no position and no working order."""
        return self.open_units() == 0 and not self.working()

    def entry_orders(self) -> list[int]:
        """The working orders that open or add to the position: those not reduce-only
        (the kernel classes a sale beyond what closes the position as an entry too)."""
        return [
            kernel_order
            for kernel_order in self.working()
            if not call(self.engine.order, kernel_order)["reduce_only"]
        ]

    def lock(self, reason: str) -> bool:
        """Lock the strategy's entries for `reason`; True when they were not locked for
        it (False once the kernel session finished: it takes no entry)."""
        return not self.finished and bool(call(self.engine.lock_entries, reason))

    def unlock(self, reason: str) -> bool:
        """Lift the entry lock of `reason`; True when it was set."""
        return not self.finished and bool(call(self.engine.unlock_entries, reason))

    def request_exit(self, reason: str) -> str | None:
        """Ask the kernel to exit the position for `reason` (one of its RISK_EXIT_REASONS);
        what it did (its EXIT_REQUESTS), or None once the kernel session finished."""
        if self.finished:
            return None
        return str(call(self.engine.request_exit, INSTRUMENT, LIVE_PRODUCT, reason))

    def cancel(self, kernel_order: int) -> None:
        """Ask the kernel to cancel a working order (nothing once it finished)."""
        if not self.finished:
            call(self.engine.cancel, kernel_order)

    def state_changed(self) -> dict[str, Any] | None:
        """The strategy's `self.state` when it changed since the last call, else None. A
        state the journal cannot keep (a value JSON does not hold) is warned about once
        and not saved: the session goes on, and a restart would not have it."""
        text = self._state_json()
        if text is None or text == self._state_text:
            return None
        self._state_text = text
        return json.loads(text)

    def _state_json(self) -> str | None:
        try:
            return state_json(self.part.strategy.state, "self.state", self.name)
        except JournalError as error:
            if not self._state_warned:
                self._state_warned = True
                logger.warning("%s: %s; it is not saved", self.name, error)
            return None


__all__ = [
    "SKIP_MISSING",
    "SLOW_CALLBACK_WARN_S",
    "Skipped",
    "StrategyFailed",
    "Trader",
    "ist_instant",
]

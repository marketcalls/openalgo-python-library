"""The session loop (16-session-design 2.1, 2.5, 2.9, 2.11; unit 8): the one thread that owns
the kernel sessions, and its iteration, `Session.step`.

    A. take in what arrived (ticks, the workers' answers, the feed's events), in arrival
       order, a bounded batch (`_session_news`)
    B. close slices by the clock: each bar builder's due slices, released to the kernel
       in order (`_session_trader`), and the broker candle checks asked for, one per
       instrument (`_session_candles`)
    C. drive the kernel sessions to their decisions (`Trader.drive`), each member of a
       shared instrument's group given its gates first (`_groups`), and each strategy
       the account's trade cap (`_account_watch`)
    D. duties by time: the stops supervisor's (`stops`), the positions a closing order's
       rejection holds on a shared instrument (`_margin_reject`), the risk watch
       (`watch`), the reconciler's reads (`reconcile`), the safe stop, the end of the day
       (`_session_stop.Control`)
    E. hand the kernel's commands to the order worker, each strategy that made any saved
       to journal L first (`_session_saved`, design rule 4), none after a guard lock and
       none the reconciler's exit guard holds; the stops supervisor then reviews each
       leg's unprotected window, and the safe stop's lines asked for in D are taken
       (`Control.after_hand_over`)
    F. save each strategy that released a slice for a restart, and its `self.state` when
       it changed (`_session_saved.save_released`)
    G. wait for the next slice close, at most `wait_max_s`

The loop never makes a network call: the order worker and the data worker do, on their
own threads. Under the test seam (`threads` False) `step` runs the order worker's and the
data worker's own steps, and posts the queued alerts, on the loop's thread before A, and
every wait is the injected `sleep`, so a test runs on a manual clock with nothing waiting
for real (2.11). The loop's lag (the age of the oldest message it takes) is watched with
threads only (`_session_lag`).

A guard lock (`guard_locked`, 6.2) locks every strategy's entries and stops every request
to the worker; the ERROR alert lists each position and its stop. The emergency stop's
protect by state turns the loop with `protecting` (3.5): no slice is released and no
strategy called. Times are UTC nanoseconds.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable, Sequence
from typing import Any

from .._calendar_convert import format_ist
from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._account_watch import AccountTrades
from ._groups import Groups
from ._guard_breach import ModeBreach, breach_alert, breach_text
from ._margin_reject import MarginHolds
from ._session_candles import ask_candles, ask_first_candles
from ._session_exits import ExitRetries
from ._session_figures import DayFigures, SessionFigures, StrategyFigures
from ._session_lag import LoopLag
from ._session_news import News
from ._session_parts import SessionParts, SessionStopped, StopFlag
from ._session_records import FillRecord
from ._session_saved import save_released, saved_commands
from ._session_trader import Trader
from ._shared_alerts import warning_line
from ._stops_report import stop_words
from ._types import LOGGER_NAME, AlertKind, Now, Sleep, Tick
from ._venue_requests import request_fields
from ._worker_queue import Request
from .bars import DEFAULT_CLOSE_GRACE_S, TickBarBuilder
from .data_worker import DataWorker
from .feedlink import FeedLink
from .inbox import Envelope
from .reconcile import Reconciler
from .stops import StopsSupervisor
from .ticks import TickTally
from .watch import RiskWatch

logger = logging.getLogger(LOGGER_NAME)

#: Seconds the loop waits on its inbox at most (16-session-design 2.5 G): a Ctrl+C or a
#: message is noticed within it.
LOOP_WAIT_MAX_S = 0.1
#: Messages, and seconds, one iteration takes in at most, so its duties never wait long.
LOOP_BATCH_MAX = 2000
LOOP_BATCH_MAX_S = 0.02
#: Seconds A keeps draining what arrived before it began once a slice close is due, so
#: the closing slice's ticks are stamped before it closes (the bars' second duty).
LOOP_DRAIN_OVERRUN_S = 0.2
#: The shortest wait, so a loop with work due still yields.
LOOP_WAIT_MIN_S = 0.001
#: The entry lock of the session's guard lock (6.2).
MODE_BREACH_LOCK = "mode_breach"

_NS = NANOS_PER_SECOND


class Session:
    """The loop of one live session (module docstring), over `parts`. `now` gives UTC
    nanoseconds and `sleep` waits seconds; `threads` False is the test seam; `stop` is
    the session's Ctrl+C; `emit` prints a line to the operator; `wait_max_s` bounds a
    wait (`LOOP_WAIT_MAX_S`)."""

    def __init__(
        self,
        parts: SessionParts,
        *,
        now: Now,
        sleep: Sleep,
        threads: bool,
        stop: StopFlag,
        emit: Callable[[str], None],
        wait_max_s: float = LOOP_WAIT_MAX_S,
    ) -> None:
        self.parts = parts
        self.now, self._sleep, self.threads = now, sleep, threads
        self.stop, self.emit = stop, emit
        self._wait_max_ns = round(wait_max_s * _NS)
        self._traders: tuple[Trader, ...] = ()
        self.groups = Groups((), event=self.event)
        self.trades = AccountTrades(self)
        self.news = News(self)
        self.exits = ExitRetries(self)
        self.margin = MarginHolds(self)
        self.stops = StopsSupervisor(self)
        self.risk = RiskWatch(self)
        self.reconcile = Reconciler(self)
        self.control: Any = None
        self.data: DataWorker | None = None
        self.link: FeedLink | None = None
        self.builders: dict[str, TickBarBuilder] = {}
        # Per exchange: the closes of its slices, the grace included, and how many closed.
        self._closes: dict[str, list[int]] = {}
        self._closed: dict[str, int] = {}
        self.trading = False
        self.deciding = True
        self.breach: ModeBreach | None = None
        self.breach_text: str | None = None
        #: Orders the broker or OpenAlgo refused, and fills the kernel refused (the
        #: daily summary's counts).
        self.rejected = 0
        self.mismatches = 0
        self.tally = TickTally()
        self.last_tick: dict[tuple[str, str], Tick] = {}
        self.warnings: list[str] = []
        #: True once the start banner is printed: a later warning is printed at once.
        self.banner_shown = False
        #: Every fill the kernel took, and what the day's earlier starts left (the start
        #: sets it from journal L at a restart).
        self.fills: list[FillRecord] = []
        self.figures = DayFigures({}, StrategyFigures(), SessionFigures())
        self.started_ns = now()
        self._held: collections.deque[Envelope] = collections.deque()
        self._event_failed = False
        self._lag = LoopLag(self)

    # ----- what the handlers and the stop use -----------------------------------------

    @property
    def locked(self) -> bool:
        """True once the mode guard locked: nothing more is sent."""
        return self.breach is not None

    @property
    def inbox_waiting(self) -> int:
        """The messages put on the loop's inbox and not taken yet."""
        return len(self.parts.inbox) + len(self._held)

    def trader_named(self, name: str) -> Trader:
        """The strategy named `name`. Raises LiveError for none."""
        for trader in self.traders:
            if trader.name == name:
                return trader
        raise LiveError(f"no strategy {name!r} runs in this session")

    @property
    def traders(self) -> tuple[Trader, ...]:
        """The strategies, in call order (set by the start once their venue maps exist),
        and with them their instrument groups (`groups`)."""
        return self._traders

    @traders.setter
    def traders(self, traders: tuple[Trader, ...]) -> None:
        self._traders, self.groups = traders, Groups(traders, event=self.event)

    def event(self, kind: str, **fields: Any) -> None:
        """Write one line to the event stream. One that cannot be written is logged to
        the day's log, alerted once and kept in the warnings; the session goes on
        (16-session-design 3.5)."""
        try:
            self.parts.logs.event(kind, **fields)
        except LiveError as error:
            logger.error("session event %s not written: %s", kind, error)
            if not self._event_failed:
                self._event_failed = True
                text = f"the session's event stream cannot be written ({error}); it goes on"
                self.warn(text, printed=True)
                self.alert(AlertKind.ERROR, text)

    def alert(self, kind: AlertKind, text: str, *, echo: bool = True) -> None:
        """Send an alert through the session's shared alerts (which log it, and echo it on
        the console unless `echo` is False: a line the caller prints itself)."""
        self.parts.alerts.send(kind, text, echo=echo)

    def warn(self, text: str, *, printed: bool = False) -> None:
        """Keep a warning for the result, once, and log it; once the start banner is out
        (it prints the warnings found before it) print it on the console too, unless
        `printed`: an alert of the same words, which the console echoes, goes with it."""
        if text not in self.warnings:
            self.warnings.append(text)
            logger.warning(text)
            if self.banner_shown and not printed:
                self.emit(warning_line(text))

    def hand_over(self, requests: Sequence[Request]) -> None:
        """Give the order worker `requests`, in order, but those the reconciler's exit
        guard holds; none after a guard lock (each is noted as not sent)."""
        handed = []
        for request in self.reconcile.guard.admit(requests):
            fields = request_fields(request)
            if self.locked:
                self.event("not_sent_locked", **fields)
                continue
            self.event("request", **fields)
            self.parts.worker.put(request)
            handed.append(request)
        self.stops.handed(handed)

    def guard_locked(self, breach: ModeBreach) -> None:
        """The mode guard locked (16-session-design 6.2): every strategy's entries are
        locked, nothing more is handed to the worker, and an ERROR alert lists every
        position as last known."""
        if self.breach is not None:
            return
        self.breach, self.breach_text = breach, breach_text(breach)
        for trader in self.traders:
            trader.lock(MODE_BREACH_LOCK)
        self.event("guard_locked", text=self.breach_text)
        positions = [
            f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units, "
            f"{stop_words(self.stops, trader)}"
            for trader in self.traders
            if (units := trader.open_units()) != 0
        ]
        self.alert(AlertKind.ERROR, breach_alert(positions))

    # ----- the iteration -------------------------------------------------------------

    def step(self, until_ns: int | None = None, *, protecting: bool = False) -> None:
        """One iteration (module docstring), waiting at most until `until_ns`. With
        `protecting` (the emergency stop's protect by state, the close's final read) it
        only takes in what arrived, runs the stops supervisor and hands over the
        commands."""
        if not self.threads:
            self._step_workers()
        self._take_in()
        released: set[str] = set()
        if not protecting:
            released = self._close_slices(self.now())
            for trader in self.traders:
                self.groups.before_decisions(trader)
                self.trades.before_decisions(trader)
                trader.drive(deciding=self.deciding)
        self.stops.duties(self.now(), protecting=protecting)
        # The commands are taken after the duties, so an entry the safe stop cancels in
        # the step that decided it is withdrawn before it reaches the worker.
        if not protecting:
            self.margin.duties(self.now())
            self.risk.duties(self.now())
            self.reconcile.duties(self.now())
            if self.control is not None:
                self.control.duties(self.now())
        self.hand_over(saved_commands(self))
        self.stops.review(self.now())
        if self.control is not None and not protecting:
            self.control.after_hand_over()
        if released:
            save_released(self, released)
        self._wait(until_ns)

    def pause(self, seconds: float) -> None:
        """Wait `seconds` while the loop keeps turning (the start's waits: the countdown,
        the feed's connect). Raises SessionStopped once the stop flag is set."""
        until_ns = self.now() + round(seconds * _NS)
        while True:
            if self.stop.count:
                raise SessionStopped("the session was asked to stop during its start")
            if self.now() >= until_ns:
                return
            self.step(until_ns)

    def _step_workers(self) -> None:
        self.parts.worker.step()
        if self.data is not None:
            self.data.step()
        self.parts.sender.deliver_queued()

    def _take_in(self) -> None:
        start_ns = self.now()
        due = self._close_due(start_ns)
        for _ in range(LOOP_BATCH_MAX):
            spent = self.now() - start_ns
            over = spent >= LOOP_BATCH_MAX_S * _NS
            if over and (not due or spent >= (LOOP_BATCH_MAX_S + LOOP_DRAIN_OVERRUN_S) * _NS):
                return
            envelope = self._next(start_ns if over else None)
            if envelope is None:
                return
            if self.threads:
                self._lag.note(envelope)
            self.news.take(envelope.message)

    def _next(self, arrived_before_ns: int | None) -> Envelope | None:
        if self._held:
            head = self._held[0]
            if arrived_before_ns is not None and head.put_ns >= arrived_before_ns:
                return None
            return self._held.popleft()
        return self.parts.inbox.take(arrived_before_ns=arrived_before_ns)

    def _close_due(self, now_ns: int) -> bool:
        return any(close_ns <= now_ns for close_ns in self._next_closes())

    def _next_closes(self) -> list[int]:
        """When each exchange's next slice closes, its grace included."""
        return [
            closes[self._closed[exchange]]
            for exchange, closes in self._closes.items()
            if self._closed[exchange] < len(closes)
        ]

    def _close_slices(self, now_ns: int) -> set[str]:
        """B: the slices due close; returns the names of the strategies that released one
        (catch-up releases included)."""
        before = {trader.name: trader.part.released for trader in self.traders}
        for exchange, builder in self.builders.items():
            closed = builder.close_due(now_ns)
            if not closed.slices:
                continue
            self._closed[exchange] += len(closed.slices)
            traders = [trader for trader in self.traders if trader.exchange == exchange]
            for trader in traders:
                trader.closed(closed.slices, closed.bars)
            ask_candles(self, traders, closed.slices[-1][1])
        for trader in self.traders:
            self.risk.released(trader, trader.release_due())
        return {t.name for t in self.traders if t.part.released != before[t.name]}

    def _wait(self, until_ns: int | None) -> None:
        now_ns = self.now()
        due = [now_ns + self._wait_max_ns]
        if until_ns is not None:
            due.append(until_ns)
        due += self._next_closes()
        wait_s = max((min(due) - now_ns) / _NS, LOOP_WAIT_MIN_S)
        if not self.threads:
            self._sleep(wait_s)
            return
        envelope = self.parts.inbox.take(wait_s=wait_s)
        if envelope is not None:
            self._held.append(envelope)

    # ----- the start phase's steps 14 and 15 -------------------------------------------

    def watch(self, link: FeedLink) -> None:
        """Start watching the connected feed (3.1 step 14): a bar builder per exchange
        from now, each strategy's first live slice and first decision, and the first
        candle checks."""
        self.link = link
        watching_ns = self.now()
        day = self.parts.spec.day
        grace_ns = round(DEFAULT_CLOSE_GRACE_S * _NS)
        for exchange in dict.fromkeys(trader.exchange for trader in self.traders):
            builder = TickBarBuilder(exchange, day, watching_from_ns=watching_ns)
            self.builders[exchange] = builder
            self._closes[exchange] = [closed + grace_ns for _, closed in builder.slices]
            self._closed[exchange] = 0
        self.risk.start(watching_ns)
        for trader in self.traders:
            builder = self.builders[trader.exchange]
            trader.watch(watching_ns, from_open=watching_ns <= builder.session[0])
            self.event("watching", strategy=trader.name, decide_from_ns=trader.decide_from_ns)
        logger.info("watching the feed from %s", format_ist(watching_ns))
        ask_first_candles(self)


__all__ = [
    "LOOP_BATCH_MAX",
    "LOOP_BATCH_MAX_S",
    "LOOP_DRAIN_OVERRUN_S",
    "LOOP_WAIT_MAX_S",
    "LOOP_WAIT_MIN_S",
    "MODE_BREACH_LOCK",
    "Session",
]

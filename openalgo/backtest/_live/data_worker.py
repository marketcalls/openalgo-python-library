"""The data worker of a live session (16-session-design 2.1, 2.5 B, 6.5; unit 9): the
REST reads of market data, made on their own thread with the session's data client
(`clients.SessionClients.data`, from 3.1 step 12), so a slow read never holds the loop.
Its results reach the loop inbox as `inbox.Quote`, `inbox.Candles` and
`inbox.DataFailed`.

Two schedules, quotes first:

- **Quote fallback** (08-live-trading section 2a; 6.5). While the loop names
  instruments (`quote`: the feed is dead, or a quiet symbol has an open position), a
  round reads each one's REST quote, a round every `QUOTE_FALLBACK_EVERY_S`, and each
  LTP goes to the loop, which pushes it to the kernel as a price and never to a bar
  builder (a quote is not a trade). The reads are paced at `QUOTE_CALLS_PER_SECOND`, so
  a round of many instruments is not refused by the server's limit on the quotes route;
  a round slower than its period is followed by the next at once. Quotes go first: a
  due quote is read before any candle read, and no candle read starts while a round has
  reads left, so a backlog of history reads never delays them.
- **Candle checks** (08-live-trading section 5 rule 2; 2.5 B). At a slice close the loop
  asks per instrument, never per slice (`check_candles`), for every slice closed since
  watching began that has no accepted candle yet: slices whose tick bar is to be
  checked, and gaps (no tick bar: missed trades, a feed outage) whose candle fills the
  series. One `history` read of today's 1-minute broker candles at the close plus 2 s
  gives the rows of the slices asked for; while one is missing the read is made again
  at plus 4 and plus 7 s, then the next close's check asks for it again. A newer check
  of an instrument replaces the older, since it asks for every slice still missing.
  History reads are paced at `HISTORY_CALLS_PER_SECOND` across instruments. While the
  feed is dead (`feed_dead` to `feed_alive`) no history read is made: the checks waiting
  then, and any asked meanwhile, are dropped, since the first one asked after the feed
  returns covers the outage's slices.

A read that fails reaches the loop as `DataFailed` with its request and the error's
text, so it is logged and, for quotes, alerted (6.5: the broker login may have
expired); a candle check that failed is read again on its schedule. The reads and what
their answers carry are `_data_reads`.

`step` does what is due and says when the next read is due; the worker's thread
(`start`) runs it and waits between steps, woken by each request. The session's test
seam calls `step` on its own thread on a manual clock instead (2.11). Every step ends,
however slow the reads: it takes the loop's requests before each read (a feed death,
new quotes or a check asked during a slow read count for the next read), stops before
its next read once `close` is called, and makes each read at most once (a quote round,
or an instrument's candle check, due again within the step ends it). Times are UTC
nanoseconds from the injected clock, prices rupees.
"""

from __future__ import annotations

import logging
import numbers
import queue
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .._calendar_convert import DateLike, from_day, to_day
from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._data_reads import ReadFailed, call, candle_rows, ltp_of
from ._types import LOGGER_NAME, Now
from .clients import DATA_CLIENT_TIMEOUT_S
from .inbox import Candles, DataFailed, Inbox, Quote

#: History reads per second at most, across instruments: under the server's pace for
#: broker history, about 3 per second server-wide (services/history_service.py,
#: `_MIN_HISTORY_INTERVAL` of 0.35 s between calls), so the session's checks leave room
#: for the user's other scripts and never meet the broker's refusal of a burst.
HISTORY_CALLS_PER_SECOND = 2
#: Seconds from the start of one quote round to the start of the next while the feed is
#: silent (16-session-design 6.5; REST quotes after 10 s without ticks, 08-live-trading
#: 2a).
QUOTE_FALLBACK_EVERY_S = 2.0
#: Quote reads per second at most: half the server's limit on the quotes route, 10 per
#: second by default (restx_api/quotes.py, API_RATE_LIMIT, counted per caller in a moving
#: window), so a round is never refused and the user's other scripts keep the other half.
QUOTE_CALLS_PER_SECOND = 5
#: Seconds after a slice's close at which its candles are read, the first read and then
#: the retries while one is missing (08-live-trading section 5 rule 2).
CANDLE_READ_OFFSETS_S = (2.0, 4.0, 7.0)
#: The broker candles the live bars are checked against: 1-minute, from the broker
#: (08-live-trading section 5 rules 2 and 6).
CANDLE_INTERVAL = "1m"
#: The `source` of those history reads: the broker ("api"), not candles stored in
#: OpenAlgo (08-live-trading section 5 rule 6).
HISTORY_SOURCE = "api"
#: The worker's thread (16-session-design 2.1).
THREAD_NAME = "openalgo-live-data"
#: Seconds `close` waits for the thread: a read in flight ends within the data client's
#: timeout.
CLOSE_WAIT_S = DATA_CLIENT_TIMEOUT_S + 1.0
#: Seconds the thread pauses after a fault outside a read, so a fault that repeats is not
#: retried in a tight loop.
FAULT_PAUSE_S = 1.0

_HISTORY_SPACING_NS = NANOS_PER_SECOND // HISTORY_CALLS_PER_SECOND
_QUOTE_EVERY_NS = round(QUOTE_FALLBACK_EVERY_S * NANOS_PER_SECOND)
_QUOTE_SPACING_NS = NANOS_PER_SECOND // QUOTE_CALLS_PER_SECOND
_OFFSETS_NS = tuple(round(offset * NANOS_PER_SECOND) for offset in CANDLE_READ_OFFSETS_S)

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class CandleRead:
    """A candle check of one instrument: `slices`, the opens (UTC nanoseconds) of today's
    closed base slices it asks candles of, oldest first, and `closed_ns`, the slice close
    it was asked at; its reads are due at that close plus each of
    `CANDLE_READ_OFFSETS_S`. A failed read's `DataFailed` carries it, with the slices
    still asked for."""

    symbol: str
    exchange: str
    slices: tuple[int, ...]
    closed_ns: int


@dataclass(frozen=True)
class QuoteRead:
    """A quote read of one instrument, as a failed read's `DataFailed` carries it."""

    symbol: str
    exchange: str


@dataclass(frozen=True)
class _Pending:
    """A candle check waiting: its request and how many of its reads were made."""

    read: CandleRead
    reads: int

    @property
    def due_ns(self) -> int:
        return self.read.closed_ns + _OFFSETS_NS[self.reads]


def _instant(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigError(f"{name} is an instant in UTC nanoseconds (an int), got {value!r}")
    return int(value)


class DataWorker:
    """The data worker of one session's IST trading `day` (module docstring): reads
    through `client` (an `openalgo.api`, the session's data client) and puts what it read
    on the loop's `inbox`; `now` is the session's clock (UTC nanoseconds).

    `check_candles`, `quote`, `feed_dead` and `feed_alive` may be called from any thread
    (the loop's); `step`, `start` and `close` from one thread at a time.
    """

    def __init__(self, client: Any, inbox: Inbox, *, day: DateLike, now: Now) -> None:
        self._client = client
        self._inbox = inbox
        self._now = now
        self._day = from_day(to_day(day)).isoformat()
        self._requests: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._wake = threading.Event()
        self._checks: dict[tuple[str, str], _Pending] = {}
        # The instruments named, the reads left in the round under way, when the next
        # round starts and when the pace lets the next quote read start (UTC nanoseconds).
        self._quotes: tuple[QuoteRead, ...] = ()
        self._round: tuple[QuoteRead, ...] = ()
        self._round_due_ns = 0
        self._quote_pace_ns = 0
        self._next_history_ns = 0
        self._dead = False
        self._stopping = False
        self._thread: threading.Thread | None = None

    # ----- requests (any thread) -------------------------------------------------

    def check_candles(
        self, symbol: str, exchange: str, slices: Sequence[int], closed_ns: int
    ) -> None:
        """Ask for the candles of today's closed slices opening at `slices` (UTC
        nanoseconds, every slice of the instrument closed since watching began that has
        no accepted candle yet), at the slice close `closed_ns`. Replaces the
        instrument's waiting check. Raises ConfigError for no slices or a value that is
        not an instant."""
        opens = tuple(sorted({_instant(value, "a slice open") for value in slices}))
        if not opens:
            raise ConfigError("check_candles asks for at least one slice's candle")
        read = CandleRead(symbol, exchange, opens, _instant(closed_ns, "closed_ns"))
        self._request("candles", read)

    def quote(self, instruments: Iterable[tuple[str, str]]) -> None:
        """Read the quote of each (symbol, exchange) in a round every
        `QUOTE_FALLBACK_EVERY_S`, the first round at once (as the pace allows), until the
        next call; none for an empty one."""
        self._request(
            "quotes", tuple(QuoteRead(symbol, exchange) for symbol, exchange in instruments)
        )

    def feed_dead(self) -> None:
        """The feed is dead: no history read until `feed_alive` (module docstring)."""
        self._request("feed", True)

    def feed_alive(self) -> None:
        """The feed ticks again: candle checks are read again."""
        self._request("feed", False)

    def _request(self, kind: str, value: Any) -> None:
        self._requests.put((kind, value))
        self._wake.set()

    # ----- the work (the worker's thread) -----------------------------------------

    def step(self) -> int | None:
        """Read what is due now, quotes first (module docstring), and return when the
        next read is due (UTC nanoseconds; at or before now when a read is due again),
        None when none waits or once `close` was called. The requests that came are
        taken before every read, and each read is made at most once per step, so a step
        ends however slow the reads are."""
        quoted = 0
        checked: set[tuple[str, str]] = set()
        while True:
            self._take_requests()
            if self._stopping:
                return None
            now_ns = self._now()
            quote_due = self._quote_due_ns()
            if quote_due is not None and quote_due <= now_ns:
                if quoted >= len(self._quotes):
                    return quote_due
                self._read_quote(now_ns)
                quoted += 1
                continue
            if self._round:
                return quote_due
            pending = self._due_check(now_ns)
            if pending is None:
                return self._next_due()
            key = (pending.read.symbol, pending.read.exchange)
            if key in checked:
                return pending.due_ns
            checked.add(key)
            self._read_candles(pending, now_ns)

    def _take_requests(self) -> None:
        while True:
            try:
                kind, value = self._requests.get_nowait()
            except queue.Empty:
                return
            if kind == "feed":
                self._set_dead(value)
            elif kind == "quotes":
                self._set_quotes(value)
            elif self._dead:
                logger.debug("candle check dropped while the feed is dead: %r", value)
            else:
                self._checks[(value.symbol, value.exchange)] = _Pending(value, 0)

    def _set_dead(self, dead: bool) -> None:
        if dead and self._checks:
            logger.debug("feed dead: %d waiting candle checks dropped", len(self._checks))
            self._checks.clear()
        self._dead = dead

    def _set_quotes(self, quotes: tuple[QuoteRead, ...]) -> None:
        if quotes != self._quotes:
            self._quotes = quotes
            self._round = ()
            self._round_due_ns = self._now()

    def _quote_due_ns(self) -> int | None:
        """When the next quote read may start, None when no instrument is named: the
        pace's instant within a round; between rounds, the next round's start, or the
        pace's instant when later."""
        if not self._quotes:
            return None
        if self._round:
            return self._quote_pace_ns
        return max(self._round_due_ns, self._quote_pace_ns)

    def _due_check(self, now_ns: int) -> _Pending | None:
        """The candle check to read now: the one due first, when one is due and the pace
        allows a history read."""
        if not self._checks or now_ns < self._next_history_ns:
            return None
        pending = min(self._checks.values(), key=lambda waiting: waiting.due_ns)
        return pending if pending.due_ns <= now_ns else None

    def _next_due(self) -> int | None:
        quote_due = self._quote_due_ns()
        due = [] if quote_due is None else [quote_due]
        if self._checks:
            first = min(waiting.due_ns for waiting in self._checks.values())
            due.append(max(first, self._next_history_ns))
        return min(due) if due else None

    def _read_quote(self, now_ns: int) -> None:
        """Read the round's next quote, starting a round when none is under way. The
        next round is due a period after this one started (at once when the round took
        longer); the pace counts from each read's start."""
        if not self._round:
            self._round = self._quotes
            self._round_due_ns = now_ns + _QUOTE_EVERY_NS
        read, self._round = self._round[0], self._round[1:]
        self._quote_pace_ns = now_ns + _QUOTE_SPACING_NS
        try:
            ltp = ltp_of(call(self._client, "quotes", symbol=read.symbol, exchange=read.exchange))
        except ReadFailed as failed:
            self._inbox.put(DataFailed(read, str(failed)))
            return
        self._inbox.put(Quote(read.symbol, read.exchange, ltp, self._now()))

    def _read_candles(self, pending: _Pending, now_ns: int) -> None:
        read = pending.read
        self._next_history_ns = now_ns + _HISTORY_SPACING_NS
        try:
            answer = call(
                self._client,
                "history",
                symbol=read.symbol,
                exchange=read.exchange,
                interval=CANDLE_INTERVAL,
                start_date=self._day,
                end_date=self._day,
                source=HISTORY_SOURCE,
            )
            rows = candle_rows(answer, set(read.slices))
        except ReadFailed as failed:
            self._inbox.put(DataFailed(read, str(failed)))
            rows = {}
        if rows:
            self._inbox.put(Candles(read.symbol, read.exchange, tuple(rows.values()), now_ns))
        missing = tuple(opened for opened in read.slices if opened not in rows)
        key = (read.symbol, read.exchange)
        if missing and pending.reads + 1 < len(_OFFSETS_NS):
            self._checks[key] = _Pending(replace(read, slices=missing), pending.reads + 1)
        else:
            del self._checks[key]

    # ----- the thread -------------------------------------------------------------

    def start(self) -> None:
        """Start the worker's thread (`THREAD_NAME`). Raises ConfigError when it runs."""
        if self._thread is not None:
            raise ConfigError("the data worker's thread is started once")
        self._thread = threading.Thread(target=self.run, name=THREAD_NAME, daemon=True)
        self._thread.start()

    def run(self) -> None:
        """The worker's thread: step, then wait until the next read is due or a request
        comes, until `close`. A fault outside a read is logged and the thread goes on."""
        while True:
            # The wake is cleared before the stop is looked at: a close between the two
            # leaves the stop set, and one after leaves the wake set, so none is lost.
            self._wake.clear()
            if self._stopping:
                return
            try:
                due_ns = self.step()
            # A fault must not end the worker, or quotes and candle checks would stop
            # unseen: it is logged with its traceback, and the worker pauses and goes on.
            except Exception:
                logger.exception("the data worker failed; it goes on")
                self._wake.wait(FAULT_PAUSE_S)
                continue
            timeout = None if due_ns is None else max(due_ns - self._now(), 0) / NANOS_PER_SECOND
            self._wake.wait(timeout)

    def close(self, wait_s: float = CLOSE_WAIT_S) -> None:
        """Stop the thread and wait for it at most `wait_s` seconds (a read in flight
        ends within the data client's timeout). Safe to call twice, and before `start`.
        The client is the session's: it is not closed here."""
        self._stopping = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(wait_s)
            if self._thread.is_alive():
                logger.warning("the data worker's thread did not end within %s s", wait_s)


__all__ = [
    "CANDLE_INTERVAL",
    "CANDLE_READ_OFFSETS_S",
    "CLOSE_WAIT_S",
    "HISTORY_CALLS_PER_SECOND",
    "HISTORY_SOURCE",
    "QUOTE_CALLS_PER_SECOND",
    "QUOTE_FALLBACK_EVERY_S",
    "THREAD_NAME",
    "CandleRead",
    "DataWorker",
    "QuoteRead",
]

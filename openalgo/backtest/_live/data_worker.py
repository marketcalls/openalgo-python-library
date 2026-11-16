"""The data worker of a live session (16-session-design 2.1, 2.5 B, 6.5; unit 9): the
REST reads of market data, made on their own thread with the session's data client
(`clients.SessionClients.data`, from 3.1 step 12), so a slow read never holds the loop.
Its results reach the loop inbox as `inbox.Quote`, `inbox.Candles` and
`inbox.DataFailed`.

Two schedules, quotes first:

- **Quote fallback** (08-live-trading section 2a; 6.5). While the loop names
  instruments (`quote`: the feed is dead, or a quiet symbol has an open position), each
  one's REST quote is read every `QUOTE_FALLBACK_EVERY_S` and its LTP goes to the loop,
  which pushes it to the kernel as a price and never to a bar builder (a quote is not a
  trade). Due quotes are read before any candle read, so a backlog of history reads
  never delays them.
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
expired); a candle check that failed is read again on its schedule.

`step` does what is due and says when the next read is due; the worker's thread
(`start`) runs it and waits between steps, woken by each request. The session's test
seam calls `step` on its own thread on a manual clock instead (2.11). Times are UTC
nanoseconds from the injected clock, prices rupees.
"""

from __future__ import annotations

import logging
import math
import numbers
import queue
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from .._calendar_convert import DateLike, from_day, to_day
from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._types import LOGGER_NAME, Now
from .clients import DATA_CLIENT_TIMEOUT_S
from .inbox import Candles, DataFailed, Inbox, Quote

#: History reads per second at most, across instruments: under the server's pace for
#: broker history, about 3 per second server-wide (services/history_service.py,
#: `_MIN_HISTORY_INTERVAL` of 0.35 s between calls), so the session's checks leave room
#: for the user's other scripts and never meet the broker's refusal of a burst.
HISTORY_CALLS_PER_SECOND = 2
#: Seconds between two quote reads of an instrument while the feed is silent for it
#: (16-session-design 6.5; REST quotes after 10 s without ticks, 08-live-trading 2a).
QUOTE_FALLBACK_EVERY_S = 2.0
#: Seconds after a slice's close at which its candles are read, the first read and then
#: the retries while one is missing (08-live-trading section 5 rule 2).
CANDLE_READ_OFFSETS_S = (2.0, 4.0, 7.0)
#: The broker candles the live bars are checked against: 1-minute, from the broker
#: (08-live-trading section 5 rules 2 and 6).
CANDLE_INTERVAL = "1m"
CANDLE_SOURCE = "api"
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
_OFFSETS_NS = tuple(round(offset * NANOS_PER_SECOND) for offset in CANDLE_READ_OFFSETS_S)
_PRICES = ("open", "high", "low", "close")

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


class _ReadFailed(Exception):
    """An answer that is not the data asked for; the text says what it was."""


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
        self._quotes: tuple[QuoteRead, ...] = ()
        self._next_quote_ns = 0
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
        """Read the quote of each (symbol, exchange) every `QUOTE_FALLBACK_EVERY_S`, the
        first at once, until the next call; none for an empty one."""
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
        """Take the requests that came, then read what is due: every due quote, then
        the due candle checks the pace allows, quotes again first after each. Returns
        when the next read is due (UTC nanoseconds), None when none waits."""
        self._take_requests()
        while True:
            now_ns = self._now()
            if self._quotes and now_ns >= self._next_quote_ns:
                self._read_quotes(now_ns)
                continue
            pending = self._due_check(now_ns)
            if pending is None:
                return self._next_due()
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
            self._next_quote_ns = self._now()

    def _due_check(self, now_ns: int) -> _Pending | None:
        """The candle check to read now: the one due first, when one is due and the pace
        allows a history read."""
        if not self._checks or now_ns < self._next_history_ns:
            return None
        pending = min(self._checks.values(), key=lambda waiting: waiting.due_ns)
        return pending if pending.due_ns <= now_ns else None

    def _next_due(self) -> int | None:
        due = [self._next_quote_ns] if self._quotes else []
        if self._checks:
            first = min(waiting.due_ns for waiting in self._checks.values())
            due.append(max(first, self._next_history_ns))
        return min(due) if due else None

    def _read_quotes(self, now_ns: int) -> None:
        self._next_quote_ns = now_ns + _QUOTE_EVERY_NS
        for read in self._quotes:
            try:
                ltp = _ltp(self._call("quotes", symbol=read.symbol, exchange=read.exchange))
            except _ReadFailed as failed:
                self._inbox.put(DataFailed(read, str(failed)))
                continue
            self._inbox.put(Quote(read.symbol, read.exchange, ltp, self._now()))

    def _read_candles(self, pending: _Pending, now_ns: int) -> None:
        read = pending.read
        self._next_history_ns = now_ns + _HISTORY_SPACING_NS
        try:
            answer = self._call(
                "history",
                symbol=read.symbol,
                exchange=read.exchange,
                interval=CANDLE_INTERVAL,
                start_date=self._day,
                end_date=self._day,
                source=CANDLE_SOURCE,
            )
            rows = _rows_of(answer, set(read.slices))
        except _ReadFailed as failed:
            self._inbox.put(DataFailed(read, str(failed)))
            rows = {}
        if rows:
            self._inbox.put(Candles(read.symbol, read.exchange, tuple(rows.values()), self._now()))
        missing = tuple(opened for opened in read.slices if opened not in rows)
        key = (read.symbol, read.exchange)
        if missing and pending.reads + 1 < len(_OFFSETS_NS):
            self._checks[key] = _Pending(replace(read, slices=missing), pending.reads + 1)
        else:
            del self._checks[key]

    def _call(self, endpoint: str, **arguments: Any) -> Any:
        """The client's answer to one read. Raises _ReadFailed when the client raised:
        the SDK answers HTTP and transport errors with a dict, so an exception is
        unexpected, and is logged with its traceback."""
        try:
            return getattr(self._client, endpoint)(**arguments)
        # One failed read must not stop the others: it reaches the loop as DataFailed.
        except Exception as error:
            logger.exception("the data client raised on %s %r", endpoint, arguments)
            raise _ReadFailed(f"{type(error).__name__}: {error}") from error

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
        while not self._stopping:
            self._wake.clear()
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


def _ltp(answer: Any) -> float:
    """The last traded price of a quote answer, rupees. Raises _ReadFailed for an error
    answer or a quote without a positive LTP (none traded yet)."""
    if not isinstance(answer, dict) or answer.get("status") != "success":
        raise _ReadFailed(_words(answer))
    data = answer.get("data")
    ltp = data.get("ltp") if isinstance(data, dict) else None
    usable = isinstance(ltp, numbers.Real) and not isinstance(ltp, bool)
    if not (usable and math.isfinite(ltp) and ltp > 0):
        raise _ReadFailed(f"the quote carries no positive last traded price (ltp {ltp!r})")
    return float(ltp)


def _rows_of(answer: Any, wanted: set[int]) -> dict[int, dict[str, Any]]:
    """The candle rows of the slices opening at `wanted` (UTC nanoseconds) in a history
    answer, by open: each an aware IST "timestamp", prices and volume, as
    `bars.candle_bar` reads a row. "No data" is an answer without rows. Raises
    _ReadFailed for an error answer or a frame whose times have no zone."""
    if isinstance(answer, dict) and answer.get("error_type") == "no_data":
        return {}
    if not isinstance(answer, pd.DataFrame):
        raise _ReadFailed(_words(answer))
    if getattr(answer.index, "tz", None) is None:
        raise _ReadFailed("the history rows carry times without a zone: no candle is placed")
    rows: dict[int, dict[str, Any]] = {}
    for stamp, row in answer.iterrows():
        opened = int(pd.Timestamp(stamp).value)
        if opened in wanted:
            prices = {name: row[name] for name in _PRICES if name in row}
            rows[opened] = {"timestamp": stamp, **prices, "volume": row.get("volume")}
    return rows


def _words(answer: Any) -> str:
    """What an answer that is not the data said."""
    if isinstance(answer, dict):
        return str(answer.get("message") or answer)
    return f"an answer of type {type(answer).__name__}"


__all__ = [
    "CANDLE_INTERVAL",
    "CANDLE_READ_OFFSETS_S",
    "CANDLE_SOURCE",
    "CLOSE_WAIT_S",
    "HISTORY_CALLS_PER_SECOND",
    "QUOTE_FALLBACK_EVERY_S",
    "THREAD_NAME",
    "CandleRead",
    "DataWorker",
    "QuoteRead",
]

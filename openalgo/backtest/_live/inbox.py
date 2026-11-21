"""The messages between the threads of a live session, and the inboxes that carry them
(16-session-design 2.3).

The loop inbox (`Inbox`) takes what the feed's threads, the order worker and the data
worker put on it; only the loop takes from it, in the order the messages were put:

| Message | From | Carries |
|---|---|---|
| `FeedTick` | feed thread | a market-data message and the callback's clock read |
| `FeedEvent` | feed thread | a connection hook: connect, disconnect, reconnect, error, with `seq` |
| `WorkerReady` | order worker | the journal session, after `start_session` and `restore` |
| `OrderNews` | order worker | an update the order path gave as news, its intent |
| `OtherFill` | order worker | a fill the order stream reported of an order not the session's |
| `SendResult` | order worker | a request sent: its ID, intent or order ID, the outcome |
| `NotSent` | order worker | a request not sent, and why |
| `Throttled` | order worker | a request the order rate held back too long; withdrawn or not |
| `Settled` | order worker | an unanswered or refused modify or cancel, and what settled it |
| `Resolutions` | order worker | unbound intents settled against the order book |
| `HeldFills` | order worker | the fills the order path holds, when they changed |
| `GuardLocked` | order worker | the mode guard's breach |
| `Books` | order worker | the books and funds read for a reconciliation |
| `PositionRows` | order worker | a position book read for a held cancel |
| `WorkerFailed` | order worker | a fatal error |
| `Candles` | data worker | closed 1-minute broker candles of one instrument |
| `Quote` | data worker | one instrument's REST quote, while the feed is silent |
| `DataFailed` | data worker | a data read that failed |

A put never waits: the queue is unbounded, because a feed callback must never wait, and
the loop watches its depth instead (2.9). The order worker's inbox is its request queue
(the worker's own) plus `RawUpdates`, the FIFO of the order updates the feed's order
callback receives (`RawOrderUpdate`; both in `_inbox_raw`, re-exported here).

Every message is a frozen dataclass; a raw message from the feed is the callback's own
copy, which only its reader looks at. Times are UTC nanoseconds, prices rupees.
"""

from __future__ import annotations

import enum
import queue
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._bars_candle import Row
from ._guard_breach import ModeBreach
from ._inbox_raw import RawOrderUpdate, RawUpdates
from ._journal_reads import JournalSession
from ._order_answers import SendOutcome
from ._order_book import Resolution
from ._order_check import OrderCheck
from ._order_state import HeldFill
from ._types import Now, OrderUpdate


class FeedEventKind(enum.Enum):
    """Which hook of the feed fired."""

    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"
    ERROR = "error"


@dataclass(frozen=True)
class FeedTick:
    """A market-data message as a data callback of the feed received it, and
    `arrived_ns`, the callback's clock read. The loop adapts it (`ticks.tick_from_feed`)
    and stamps the tick's receive time when it takes it (2.5 A.1)."""

    message: Mapping[str, Any]
    arrived_ns: int


@dataclass(frozen=True)
class FeedEvent:
    """A connection hook of the feed: its `kind`; `payload`, the dict the hook received
    (for an error, its context: `kind`, `callback`, `message`, and for a refusal by the
    server `code` and `rejected`); `seq`, the connection number every payload carries
    (None when one carries none); `arrived_ns`, the hook's clock read; and for an error,
    the exception."""

    kind: FeedEventKind
    payload: Mapping[str, Any]
    seq: int | None
    arrived_ns: int
    error: BaseException | None = None

    @property
    def requested(self) -> bool:
        """True for a disconnect the session asked for (its own `disconnect()`)."""
        return self.kind is FeedEventKind.DISCONNECT and self.payload.get("requested") is True


@dataclass(frozen=True)
class WorkerReady:
    """The order worker has started the journal's session and restored the order path:
    the journal session (mode, starts, last start)."""

    session: JournalSession


@dataclass(frozen=True)
class OrderNews:
    """An order update the order path gave as news, the intent of its order, and whether
    the worker cleared the leg's journal stop with it (a stop's terminal news, 4.1)."""

    update: OrderUpdate
    intent_id: str
    stop_cleared: bool = False


@dataclass(frozen=True)
class OtherFill:
    """A fill the order stream reported of an order the session did not send, on
    `exchange:symbol`, received at `arrived_ns`: the loop asks for a comparison soon when
    the session trades the instrument (5.1; 7.1 "As built (unit 14")."""

    symbol: str
    exchange: str
    arrived_ns: int


@dataclass(frozen=True)
class SendResult:
    """The outcome of a request the worker sent: the request's ID, and the intent ID of a
    placement or the broker's order ID of a modify or cancel."""

    request_id: int
    outcome: SendOutcome
    intent_id: str = ""
    order_id: str = ""


class NotSentReason(enum.Enum):
    """Why the worker did not send a request."""

    #: The mode guard has locked: nothing is sent any more.
    LOCKED = "locked"
    #: A stop, an exit or a protective modify held back, to be tried again.
    HELD = "held"
    #: OpenAlgo is silent: nothing is sent until it answers.
    SILENT = "silent"
    #: An entry whose decision grew too old before its turn.
    STALE = "stale"
    #: A modify or cancel of an order already ended.
    ORDER_ENDED = "order_ended"
    #: Refused before it was sent (an entry or add is not tried again).
    REFUSED = "refused"
    #: An entry or add the order rate held back too long (18-multi-strategy 2.5a rule 12).
    THROTTLED = "order_rate"
    #: An entry or add the kernel cancelled while it waited in the worker's queue.
    CANCELLED = "cancelled"
    #: An entry or add that waited in the worker's queue when its strategy's entries
    #: locked (`_venue_withdraw`), or that the worker took after they locked
    #: (`_locked_entries`).
    ENTRIES_LOCKED = "entries_locked"


@dataclass(frozen=True)
class NotSent:
    """A request the worker did not send (the worker's own request object), why, and the
    error's text when there was one."""

    request: object
    reason: NotSentReason
    error: str = ""


@dataclass(frozen=True)
class Throttled:
    """A request (the worker's own request object) the order rate held back for
    `waited_s` seconds; `withdrawn` when it is an entry or add that is not sent
    (`_order_throttle`)."""

    request: Any
    waited_s: float
    withdrawn: bool


@dataclass(frozen=True)
class Settled:
    """A modify or cancel (the worker's request) that was unanswered or refused, and the
    `OrderCheck` that settled it, or None when it was still unsettled past its deadline
    (4.3, 4.5)."""

    request: object
    check: OrderCheck | None

    @property
    def expired(self) -> bool:
        """True when no read settled it before its deadline."""
        return self.check is None


@dataclass(frozen=True)
class Resolutions:
    """What `resolve_unbound` settled, in its order."""

    resolutions: tuple[Resolution, ...]


@dataclass(frozen=True)
class HeldFills:
    """The fills the order path holds now (`held_fills()`), sent when they changed."""

    held: tuple[HeldFill, ...]


@dataclass(frozen=True)
class GuardLocked:
    """The mode guard locked: why."""

    breach: ModeBreach


@dataclass(frozen=True)
class Books:
    """The answers of one reconciliation's reads, as the SDK returned them, numbered by
    `generation`; the reads ran from `started_ns` to `ended_ns`."""

    generation: int
    positionbook: Mapping[str, Any]
    orderbook: Mapping[str, Any]
    tradebook: Mapping[str, Any]
    funds: Mapping[str, Any]
    started_ns: int
    ended_ns: int


@dataclass(frozen=True)
class PositionRows:
    """A position book read for a held cancel (4.11), as the SDK returned it, and when it
    was read; with the order book read after it when it was asked for
    (`ReadPositions.orders`), else None."""

    positionbook: Mapping[str, Any]
    read_ns: int
    orderbook: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class WorkerFailed:
    """The order worker cannot go on: a JournalError while it wrote an intent or a stop,
    or an exception nothing expected (the loop runs the emergency stop, 3.5)."""

    error: BaseException


@dataclass(frozen=True)
class Candles:
    """Today's closed 1-minute broker candles of one instrument that were asked for, each
    a row `bars.candle_bar` reads, and `read_ns`, when the read began: a slice that
    settled while the read was under way may be missing from it (the session plan's
    catch-up counts from the start)."""

    symbol: str
    exchange: str
    rows: tuple[Row, ...]
    read_ns: int


@dataclass(frozen=True)
class Quote:
    """One instrument's last traded price from a REST quote (rupees), read while the feed
    is silent for it (6.5), and when it was read."""

    symbol: str
    exchange: str
    ltp: float
    read_ns: int


@dataclass(frozen=True)
class DataFailed:
    """A read of the data worker failed: what was asked (the data worker's own request
    object) and the error's text."""

    request: object
    error: str


#: Everything the loop inbox carries.
Message = (
    FeedTick
    | FeedEvent
    | WorkerReady
    | OrderNews
    | OtherFill
    | SendResult
    | NotSent
    | Throttled
    | Settled
    | Resolutions
    | HeldFills
    | GuardLocked
    | Books
    | PositionRows
    | WorkerFailed
    | Candles
    | Quote
    | DataFailed
)


@dataclass(frozen=True)
class Envelope:
    """A message as the loop takes it: `put_ns`, when it was put on the inbox, and the
    message. The age of the oldest one the loop takes is the loop's lag (2.9)."""

    put_ns: int
    message: Message


class Inbox:
    """The loop's inbox. Any thread puts, and a put never waits; one thread takes (the
    loop), oldest first. `now` stamps each message as it is put."""

    def __init__(self, now: Now) -> None:
        self._now = now
        self._queue: queue.SimpleQueue[Envelope] = queue.SimpleQueue()
        # Taken off the queue but left first for the next take (the loop's thread only).
        self._head: Envelope | None = None

    def put(self, message: Message) -> None:
        """Put `message` on the inbox, stamped now. Raises TypeError for anything that is
        not one of the messages of the module docstring."""
        if not isinstance(message, Message):
            raise TypeError(
                f"the loop inbox carries the messages of inbox.py, not {type(message).__name__}"
            )
        self._queue.put(Envelope(self._now(), message))

    def take(self, *, wait_s: float = 0.0, arrived_before_ns: int | None = None) -> Envelope | None:
        """The oldest message, or None when none came within `wait_s` seconds (a finite
        number; 0, the default, looks once and returns at once). With `arrived_before_ns`, also None
        when the oldest was put at or after that instant (UTC nanoseconds): it stays
        first for the next take, so the loop can drain what arrived before a slice's
        close and nothing later (2.5 A). Call it from the loop's thread only."""
        envelope = self._head
        if envelope is None:
            try:
                if wait_s > 0:
                    envelope = self._queue.get(timeout=wait_s)
                else:
                    envelope = self._queue.get_nowait()
            except queue.Empty:
                return None
        if arrived_before_ns is not None and envelope.put_ns >= arrived_before_ns:
            self._head = envelope
            return None
        self._head = None
        return envelope

    def __len__(self) -> int:
        """The messages waiting, as of now (another thread may put more meanwhile)."""
        return self._queue.qsize() + (self._head is not None)


__all__ = [
    "Books",
    "Candles",
    "DataFailed",
    "Envelope",
    "FeedEvent",
    "FeedEventKind",
    "FeedTick",
    "GuardLocked",
    "HeldFills",
    "Inbox",
    "Message",
    "NotSent",
    "NotSentReason",
    "OrderNews",
    "OtherFill",
    "PositionRows",
    "Quote",
    "RawOrderUpdate",
    "RawUpdates",
    "Resolutions",
    "SendResult",
    "Settled",
    "Throttled",
    "WorkerFailed",
    "WorkerReady",
]

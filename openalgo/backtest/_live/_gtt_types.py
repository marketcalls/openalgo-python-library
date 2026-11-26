"""The GTT requests the order worker takes and the answers it gives the session
(20-positional-live 3; the lead's rulings N3a-1 and N3a-2), in the style of the order
requests (`_venue_requests`): the session (unit N3b) makes a `GttRequest` from each of the
kernel's GTT commands (place, modify, cancel), and applies each `GttAnswer` to the kernel;
the GTT follower's findings come as `GttOutcome`s. Every GTT is a SINGLE GTT (P6).

- **A request** carries the kernel's GTT id, the instrument and product, the child order's
  side, the trigger and the child's limit in rupees, the units, where the trigger sits
  against the price it was placed from (`Direction`: OpenAlgo's `triggerprice_sl` below
  it, `triggerprice_tg` above it, 3.6), what the GTT is for (a leg's stop or a strategy's
  GTT entry) and its priority; a modify or cancel names the journal's GTT it is of (`seq`,
  from the place's answer). A place that names one (`seq`) is a place the journal shows
  under way and unbound at a restart: the worker settles it as an unanswered place and
  sends it only as that rule allows (`_gtt_send`; the lead's ruling N5a-5).
- **An answer** to a request (`GttAnswerKind`): placed (the broker holds it, a book read
  showed it), refused with the broker's message (kept across starts when it repeats a
  refusal of an earlier day, 3.5), modified, modify refused, cancelled, cancel refused, or
  unresolved (the worker cannot tell, and says why). It names the journal's GTT that now
  holds the kernel's GTT: a modify that had to be cancelled and placed again names the new
  one. That place again can be refused: a MODIFY answered REFUSED (`seq` None) means the
  old GTT was cancelled and nothing rests for the leg now.
- **An outcome** of the follower (`GttEvent`): in flight (fired or gone, its child not
  resolved yet: the leg is in flight for the reconciler), resting again (a read showed a
  GTT taken as gone resting: nothing fired), triggered (its child found), child missed
  (the position moved by the GTT's units with no child found: the fill's price, flagged
  when the trade book did not give it), ended outside the session, expired, rejected
  after placement, or a mismatch the engine never guesses through; and other terms (a
  resting GTT whose row shows another trigger or other units than the journal's, which
  stays followed: the broker holds a stop the session did not ask for).

**Order.** The news reaches the loop in the order the order worker learned it: a call's
answer comes before any outcome of the GTT it names (`GttNews`).

**What the session must do with them (unit N3b).** Every outcome but IN_FLIGHT lets the
leg's flight go (`_gtt_flight`), TRIGGERED included. OTHER_TERMS leaves the GTT resting
and followed: the session must not protect the leg again beside it (a second stop), but
alert and set the journal's terms again (a modify). The session binds the child
(`BindGttChild`) in the same loop step it takes TRIGGERED, or the child's fill may be
compared as an external one. A request the worker could not send is posted as
`inbox.NotSent` naming it (OpenAlgo silent, the mode guard locked); the session hands it to
the kernel as for an order request. The follower reads the GTT book on the clock only when
the worker is given `market_open` (`worker.OrderWorker`).

Prices are rupees, quantities whole units, times UTC nanoseconds.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass

from .._products import OVERNIGHT_PRODUCTS
from ..errors import LiveError
from ._journal_gtt_rows import GTT_PURPOSES, GTT_SIDES, GttTerms
from ._types import Purpose
from ._venue_requests import Priority

#: The flag of a fill whose price the books did not give: the GTT child's limit stands in.
PRICE_UNVERIFIED = "price_unverified"
#: The flag of a fill made on a trading day no session ran (20-positional-live 4.3): a
#: GTT that fired after its last resting read, with a trading day between that read's day
#: and today; the binding's `VENUE_FILL_FLAGS` hold both.
MISSED_DAY = "missed_day"


class GttAction(enum.Enum):
    """What a GTT request asks of the broker."""

    PLACE = "place"
    MODIFY = "modify"
    CANCEL = "cancel"


class Direction(enum.Enum):
    """Where a SINGLE GTT's trigger sits against the price it was placed from, which
    OpenAlgo reads from the field it is sent in (20-positional-live 3.6)."""

    #: Below the price (a long's stop): sent in `triggerprice_sl`.
    BELOW = "below"
    #: Above the price (a short's stop): sent in `triggerprice_tg`.
    ABOVE = "above"


@dataclass(frozen=True)
class GttRequest:
    """One GTT call for the order worker (module docstring). `request_id` is the session's
    request number, `gtt_id` the kernel's; `seq` the journal's GTT a modify or cancel is of,
    or a place under way since before a restart is (None for a new place). Raises LiveError
    when the request cannot be sent as it stands."""

    request_id: int
    action: GttAction
    purpose: Purpose
    strategy: str
    gtt_id: int
    symbol: str
    exchange: str
    product: str
    side: str
    trigger_price: float
    limit_price: float
    quantity: int
    direction: Direction
    priority: Priority
    seq: int | None = None

    def __post_init__(self) -> None:
        problem = _problem(self)
        if problem:
            raise LiveError(f"GTT request {self.request_id} cannot be sent: {problem}")

    @property
    def terms(self) -> GttTerms:
        """The terms the request asks the broker to hold."""
        return GttTerms(self.side, self.trigger_price, self.limit_price, self.quantity)

    @property
    def leg(self) -> str:
        """The leg in words, for logs and alerts."""
        return f"{self.strategy} on {self.exchange}:{self.symbol} ({self.product})"


class GttAnswerKind(enum.Enum):
    """What became of a GTT request (module docstring)."""

    PLACED = "placed"
    REFUSED = "refused"
    MODIFIED = "modified"
    MODIFY_REFUSED = "modify_refused"
    CANCELLED = "cancelled"
    CANCEL_REFUSED = "cancel_refused"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class GttAnswer:
    """The answer to `request` (module docstring): `seq` and `trigger_id`, the journal's
    GTT and the broker's that hold the kernel's GTT now (None and "" when none does);
    `message`, the broker's words of a refusal or why it is unresolved; `kept`, a refusal
    kept across starts (20-positional-live 3.5)."""

    request: GttRequest
    kind: GttAnswerKind
    seq: int | None = None
    trigger_id: str = ""
    message: str = ""
    kept: bool = False


class GttEvent(enum.Enum):
    """What the GTT follower found of a GTT (module docstring)."""

    IN_FLIGHT = "in_flight"
    #: A GTT taken as gone (a read that left it out) that a later read shows resting.
    RESTING = "resting"
    TRIGGERED = "triggered"
    CHILD_MISSED = "child_missed"
    ENDED_OUTSIDE = "ended_outside"
    EXPIRED = "expired"
    REJECTED = "rejected"
    MISMATCH = "mismatch"
    #: A resting GTT whose row shows other terms than the journal's (it rests still).
    OTHER_TERMS = "other_terms"

    @property
    def resolves(self) -> bool:
        """True for every event but IN_FLIGHT: nothing of the GTT is in flight."""
        return self is not GttEvent.IN_FLIGHT


@dataclass(frozen=True)
class GttOutcome:
    """A finding of the GTT follower about the journal's GTT `seq` (`event`, module
    docstring), of `strategy`'s kernel GTT `gtt_id`, broker GTT `trigger_id`: for
    TRIGGERED the child's broker `order_id`; for CHILD_MISSED the `units` and `price` of
    the fill the position's move stands for, and its `flags`; for MISMATCH the order IDs
    that could each be its child (`candidates`); for ENDED_OUTSIDE the leg's signed units
    at the broker in the read that judged it (`held`, None for none read); `message` says
    what was seen."""

    seq: int
    strategy: str
    gtt_id: int
    trigger_id: str
    symbol: str
    exchange: str
    product: str
    event: GttEvent
    order_id: str = ""
    units: int = 0
    price: float = 0.0
    flags: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    message: str = ""
    held: int | None = None


@dataclass(frozen=True)
class GttNews:
    """The loop inbox's message (`inbox`) of a GTT call's answer or a finding of the GTT
    follower, in the order the order worker learned them."""

    news: GttAnswer | GttOutcome


@dataclass(frozen=True)
class ReadGtts:
    """Read the GTT book now (the start's read and the close's, 20-positional-live 3.1):
    the follower judges every GTT it follows. With `at_close` the read's findings are
    posted as `GttBookSeen` too, the units of every GTT leg read with it."""

    at_close: bool = False

    priority = Priority.UPKEEP


@dataclass(frozen=True)
class GttBookSeen:
    """The close's read of the GTT book (`ReadGtts` with `at_close`; 20-positional-live 2,
    "Day 1, the close"), as the loop takes it: when it began (UTC ns), each resting GTT's
    triggers (rupees) and units by its broker trigger ID (None for a book that could not
    be read), and each GTT leg's (symbol, exchange, product) units at the broker read just
    before it (None where they could not be read), with the legs whose own fills had
    settled then (`_gtt_reads.BookRead`)."""

    read_ns: int
    resting: Mapping[str, tuple[tuple[float, ...], int]] | None
    units: Mapping[tuple[str, str, str], int | None]
    settled: frozenset[tuple[str, str, str]]


@dataclass(frozen=True)
class BindGttChild:
    """Bind `order_id`, the child the follower found for the journal's GTT `seq`, as the
    kernel's order `kernel_order` (the session gives it once the kernel follows the child
    as the leg's exit): the GTT is then fired in the journal."""

    seq: int
    order_id: str
    kernel_order: int

    priority = Priority.PROTECT


#: The FIFO key of a GTT's requests in the worker's queue: ("gtt", strategy, kernel GTT).
GttKey = tuple[str, str, int]


def gtt_key(request: GttRequest) -> GttKey:
    """The FIFO key of `request`: every request of one kernel GTT queues behind the last."""
    return "gtt", request.strategy, request.gtt_id


def _problem(request: GttRequest) -> str:
    """Why `request` cannot be sent, or ""."""
    numbers = (request.trigger_price, request.limit_price)
    if not isinstance(request.action, GttAction) or not isinstance(request.direction, Direction):
        return "its action and direction are a GttAction and a Direction"
    if request.purpose not in GTT_PURPOSES:
        return f"a GTT is a leg's stop or a strategy's entry, not {request.purpose!r}"
    if request.product not in OVERNIGHT_PRODUCTS:
        return f"a GTT holds a CNC or NRML position, not {request.product!r}"
    if request.side not in GTT_SIDES:
        return f"its side is BUY or SELL, not {request.side!r}"
    if not all(isinstance(x, int | float) and math.isfinite(x) for x in numbers):
        return "its trigger and limit are finite prices"
    if request.trigger_price <= 0 or request.limit_price <= 0:
        # Every child is a LIMIT order (`_journal_gtt_rows.CHILD_PRICE_TYPE`): a limit of 0 would
        # sell at any price, or be refused by the broker.
        return "its trigger and limit are above 0 rupees"
    if isinstance(request.quantity, bool) or not isinstance(request.quantity, int):
        return f"its units are a whole number, not {request.quantity!r}"
    if request.quantity <= 0:
        return f"its units are above 0, not {request.quantity}"
    if request.seq is None and request.action is not GttAction.PLACE:
        return "a modify or cancel names the journal GTT it is of"
    return ""


__all__ = [
    "GTT_PURPOSES",
    "MISSED_DAY",
    "PRICE_UNVERIFIED",
    "BindGttChild",
    "Direction",
    "GttAction",
    "GttAnswer",
    "GttAnswerKind",
    "GttBookSeen",
    "GttEvent",
    "GttKey",
    "GttNews",
    "GttOutcome",
    "GttRequest",
    "ReadGtts",
    "gtt_key",
]

"""The books read at every start, and a restart judged against them (16-session-design 3.1
steps 9 and 10, 3.6, 5.5; unit 12), before the plan is built.

**Step 9** (`read_start`): while the journal holds unbound intents of the day (a restart),
the order worker resolves them against the order book before anything else is sent; an
ERROR alert names them after `UNBOUND_AT_START_ALERT_S`, and an intent the book cannot
tell from another order refuses the start (LiveError: the user checks the order book).
**Step 10**: the worker reads the position, order and trade books (`ReadBooks`), again
every `START_BOOKS_RETRY_S` while one cannot be read, for at most `START_BOOKS_WAIT_S`
(then LiveError). The messages the worker posts meanwhile are kept for the loop
(`StartBooks.deferred`), but the resolutions, which the journal holds already.

A strategy whose saved state is of another instrument than the session trades for it
(the same class name moved to another symbol or contract) refuses the start, naming both
and what to do (`_session_saved.another_instrument`); its position is not judged against
the other instrument's books, and it is never started afresh, since its saved state may
hold working orders. Every refusal of the start is raised at once, so a trader sees each
strategy to fix before starting again.

**The journal position** of each strategy (`judge_start`, 3.6): its saved position, plus,
for each order its saved state worked, the units filled beyond its saved ledger by the
journal's last update or the trade book, whichever says more. The restart rules (5.5,
`_reconcile_rules.restart_kind`) judge it against the broker's: a broker position on a
traded symbol that no journal position holds, and a journal position the broker does not
hold that no trade of another order explains, refuse the start (LiveError naming them;
nothing has been sent). An instrument several strategies share is judged as one, the sum
of its members' journal positions against the broker's net (`_restart_groups`,
18-multi-strategy 4.10). A resumed leg whose saved stop does not work in the order book is
an ERROR alert now, saying what the restart does about it (`_missing_stop_words`): its
stop is placed again at the end of step 11, unless the books show the broker holding
fewer units that no trade explains, or the other side (the exit guard then holds a stop
for the journal's units), or other orders closed the position.

Times are UTC nanoseconds, quantities signed units. Run by the main thread at the start.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._group_rules import GroupFigure, and_join
from ._groups import InstrumentGroup, instrument_groups
from ._order_book import Resolved
from ._reconcile_rules import Kind, LegBooks, RestartKind, judge, leg_books, restart_kind
from ._reconcile_rules import unexplained_by as unexplained_units
from ._restart_groups import judge_group
from ._session_saved import Saved, another_instrument, read_saved
from ._types import LOGGER_NAME, AlertKind
from ._worker_queue import ReadBooks
from .inbox import Books, Envelope, GuardLocked, NotSent, Resolutions, WorkerFailed

logger = logging.getLogger(LOGGER_NAME)

#: Seconds unbound intents may wait at a start before an ERROR alert names them (3.1 step 9).
UNBOUND_AT_START_ALERT_S = 60.0
#: Seconds the start waits for readable books, and between two reads (3.1 step 10).
START_BOOKS_WAIT_S = 30.0
START_BOOKS_RETRY_S = 2.0
#: Seconds the start waits between two looks at the worker under the test seam.
START_POLL_S = 0.5
#: The generation of the start's book reads (the reconciler numbers its own from 1).
_START_GENERATION = 0

_NS = NANOS_PER_SECOND


@dataclass(frozen=True)
class StartBooks:
    """The start's book read (`books`) and the worker's messages kept for the loop."""

    books: Books
    deferred: tuple[Envelope, ...] = ()


@dataclass(frozen=True)
class LegStart:
    """One strategy at the start: its saved state (None: nothing saved today), its journal
    position, the broker's units, the books of its instrument, how the restart judged it
    and, for a difference, how the reconciler would (`_reconcile_rules.Kind`); for a
    member of a shared instrument, the group it was judged with (`group`, None alone)."""

    saved: Saved | None
    journal_units: int
    broker_units: int
    books: LegBooks
    restart: RestartKind
    kind: Kind = Kind.AGREE
    group: GroupFigure | None = None


@dataclass(frozen=True)
class _Found:
    """One strategy before it is judged: its spec, saved state, the books of its
    instrument (readable), the broker's units in them and its journal position."""

    spec: Any
    saved: Saved | None
    books: LegBooks
    broker_units: int
    journal_units: int


@dataclass(frozen=True)
class Restart:
    """What the start found: each strategy's `LegStart` by name, the broker order IDs of
    the day's orders in the journal (`own`), when the books were read."""

    legs: Mapping[str, LegStart]
    own: frozenset[str]
    read_ns: int


def read_start(
    worker: Any, inbox: Any, settings: Any, journal: Any, day: Any, alerts: Any
) -> StartBooks:
    """Steps 9 and 10 (module docstring). Raises LiveError for an ambiguous unbound
    intent, a mode change, or books that cannot be read in time; the worker's own failure
    as it failed."""
    deferred: list[Envelope] = []
    began_ns, alerted = settings.now(), False
    while unbound := journal.unbound(day=day):
        if not alerted and settings.now() - began_ns >= round(UNBOUND_AT_START_ALERT_S * _NS):
            alerted = True
            names = ", ".join(intent.intent_id for intent in unbound)
            alerts.send(
                AlertKind.ERROR,
                f"the start waits for the order book to show intents {names}: nothing is "
                "sent for their instruments until it does",
            )
        _turn(worker, inbox, settings, deferred)
    until_ns = settings.now() + round(START_BOOKS_WAIT_S * _NS)
    while True:
        worker.put(ReadBooks(_START_GENERATION))
        asked_ns = settings.now()
        while (books := _turn(worker, inbox, settings, deferred)) is None:
            if settings.now() >= until_ns:
                raise LiveError(_UNREAD)
        if _readable(books):
            return StartBooks(books, tuple(deferred))
        while settings.now() < asked_ns + round(START_BOOKS_RETRY_S * _NS):
            _turn(worker, inbox, settings, deferred)
        if settings.now() >= until_ns:
            raise LiveError(_UNREAD)


_UNREAD = (
    "OpenAlgo's position, order and trade books could not be read at the start, so the "
    "session cannot tell what the broker holds: check OpenAlgo and the broker's login, "
    "then start again"
)


def _turn(worker: Any, inbox: Any, settings: Any, deferred: list[Envelope]) -> Books | None:
    """Let the worker work once (inline under the test seam) and take what it posted: the
    books when they came, else None. Raises as `read_start` does."""
    if settings.threads:
        envelope = inbox.take(wait_s=START_POLL_S)
        envelopes = [] if envelope is None else [envelope]
    else:
        worker.step()
        envelopes = []
        while (envelope := inbox.take()) is not None:
            envelopes.append(envelope)
        if not envelopes:
            settings.sleep(START_POLL_S)
    books = None
    for envelope in envelopes:
        message = envelope.message
        if isinstance(message, Books):
            books = message
        elif isinstance(message, WorkerFailed):
            raise message.error
        elif isinstance(message, GuardLocked):
            raise LiveError(
                "OpenAlgo's mode changed during the start: start again in the mode wanted"
            )
        elif isinstance(message, NotSent) and isinstance(message.request, ReadBooks):
            raise LiveError(f"the books were not read at the start: {message.error}")
        elif isinstance(message, Resolutions):
            _ambiguous(message)
        else:
            deferred.append(envelope)
    return books


def _ambiguous(message: Resolutions) -> None:
    for resolution in message.resolutions:
        if resolution.kind is Resolved.AMBIGUOUS:
            raise LiveError(
                f"intent {resolution.intent.intent_id} of {resolution.intent.symbol} cannot "
                f"be told apart in the order book from {', '.join(resolution.candidates)}: "
                "check the order book in OpenAlgo, then start again"
            )


def _readable(books: Books) -> bool:
    return all(
        answer.get("status") == "success"
        for answer in (books.positionbook, books.orderbook, books.tradebook)
    )


def judge_start(spec: Any, journal: Any, start: StartBooks, alerts: Any) -> Restart:
    """Each strategy's journal position judged against the broker's, an instrument several
    strategies share as one (module docstring). Raises LiveError, once, for the positions
    a restart cannot resume and the strategies saved on another instrument, naming them."""
    day, books = spec.day, start.books
    own = frozenset(order.order_id for order in journal.bound_orders(day=day))
    found: dict[str, _Found] = {}
    moved: list[str] = []
    for each in spec.strategies:
        saved = read_saved(journal.rms_state(each.name, day=day), each.name)
        # Before its position is judged against another instrument's books.
        elsewhere = another_instrument(saved, each.name, each.symbol, each.exchange)
        if elsewhere is not None:
            moved.append(elsewhere)
            continue
        leg = leg_books(
            books.positionbook, books.tradebook, books.orderbook, each.symbol, each.exchange
        )
        if leg.units is None or leg.trades is None:
            raise LiveError(_UNREAD)
        units = _journal_units(saved, journal, day, leg)
        found[each.name] = _Found(each, saved, leg, leg.units, units)
    legs: dict[str, LegStart] = {}
    refused: list[str] = []
    for group in instrument_groups((s.name, s.symbol, s.exchange) for s in spec.strategies):
        members = [found[name] for name in group.members if name in found]
        if not members:
            continue
        starts, refusal = (
            _judge_shared(group, members, own) if group.shared else _judge_leg(members[0], own)
        )
        legs.update({m.spec.name: leg for m, leg in zip(members, starts, strict=True)})
        refused.extend([] if refusal is None else [refusal])
    if refused or moved:
        positions = ["the start cannot resume: " + "; ".join(refused)] if refused else []
        raise LiveError(". ".join([*positions, *moved]))
    _alert_missing_stops(spec, legs, journal, alerts)
    return Restart(legs, own, books.started_ns)


def _judge_leg(found: _Found, own: frozenset[str]) -> tuple[list[LegStart], str | None]:
    """A strategy alone on its instrument judged against the broker's units (module
    docstring): its `LegStart`, and the words of its refusal (None when it resumes)."""
    each, leg, units, broker = found.spec, found.books, found.journal_units, found.broker_units
    external = sum(trade.signed for trade in leg.external(own))
    adopted = 0 if found.saved is None else found.saved.adopted
    unexplained = unexplained_units(units, broker, external, adopted)
    verdict = restart_kind(units, broker, unexplained)
    start = LegStart(found.saved, units, broker, leg, verdict, judge(units, broker, unexplained))
    if verdict is RestartKind.UNKNOWN_POSITION:
        return [start], (
            f"OpenAlgo's books show {broker} units of {each.symbol} {each.exchange}, which "
            f"{each.name} trades, and the journal holds none: close them or start with "
            "another symbol"
        )
    if verdict is RestartKind.UNEXPLAINED_FLAT:
        return [start], (
            f"the journal holds {units} units of {each.symbol} {each.exchange} for "
            f"{each.name}, OpenAlgo's books show none and no trade explains it: check the "
            "account in OpenAlgo"
        )
    return [start], None


def _judge_shared(
    group: InstrumentGroup, members: list[_Found], own: frozenset[str]
) -> tuple[list[LegStart], str | None]:
    """The members of a shared instrument judged together (`_restart_groups`): each
    member's `LegStart`, in `members`' order, and the words of the group's refusal (None
    when it resumes)."""
    books = members[0].books
    external = sum(trade.signed for trade in books.external(own))
    adopted = sum(0 if m.saved is None else m.saved.adopted for m in members)
    units = {m.spec.name: m.journal_units for m in members}
    group_start = judge_group(group, units, members[0].broker_units, external, adopted)
    starts = [
        LegStart(
            m.saved,
            m.journal_units,
            m.broker_units,
            m.books,
            group_start.restart,
            group_start.kind,
            group_start.figure,
        )
        for m in members
    ]
    return starts, group_start.refusal


def _journal_units(saved: Saved | None, journal: Any, day: Any, leg: LegBooks) -> int:
    """The journal position (module docstring)."""
    if saved is None:
        return 0
    units = saved.units
    for old, record in saved.orders.items():
        if not saved.sent.get(old, False):
            continue
        order_id = record.order_id or journal.order_id_of(record.intent.intent_id) or ""
        if not order_id:
            continue
        last = journal.last_update(order_id, day=day)
        filled = max(0 if last is None else last.filled_quantity, leg.fills(order_id)[0])
        more = max(0, filled - record.filled_units)
        units += more if record.intent.action == "BUY" else -more
    return units


def _alert_missing_stops(
    spec: Any, legs: Mapping[str, LegStart], journal: Any, alerts: Any
) -> None:
    """An ERROR alert for each leg that resumes with a position whose saved stop does not
    work in the order book (3.1 step 10)."""
    for each in spec.strategies:
        leg = legs[each.name]
        stop = None if leg.saved is None else leg.saved.stop
        if leg.journal_units == 0 or stop is None:
            continue
        order_id = stop.order_id or journal.order_id_of(stop.intent.intent_id) or ""
        row = leg.books.row(order_id) if order_id else None
        if row is not None and row.working:
            continue
        alerts.send(
            AlertKind.ERROR,
            f"{each.name}: {leg.journal_units} units of {each.symbol} {each.exchange} have no "
            f"working stop at the broker (stop {order_id or 'not sent'}): "
            f"{_missing_stop_words(leg)}",
        )


def _missing_stop_words(leg: LegStart) -> str:
    """What the restart does about a resumed leg's missing stop, by how the start judged
    the leg against the broker's books (module docstring); a member of a shared
    instrument's words name the group and the sum it was judged by."""
    shown = f"OpenAlgo's books show {leg.broker_units} units"
    if leg.group is not None:
        shown += f" for {and_join(leg.group.members)} together, who hold {leg.group.units}"
    if leg.kind is Kind.EXPLAINED_FLAT:
        return (
            "OpenAlgo's books show the position closed by other orders, which the engine "
            "takes as its own, so no stop is placed"
        )
    if leg.kind is Kind.UNEXPLAINED:
        return (
            f"{shown}, which no trade explains, so no stop is placed for the journal's units "
            "while they disagree: check the account in OpenAlgo"
        )
    if leg.kind is Kind.EXPLAINED_SHARED:
        return (
            f"{shown}; trades of other orders explain it, which no strategy of a shared "
            "instrument takes as its own, so no stop is placed for the journal's units while "
            "they disagree: check the account in OpenAlgo"
        )
    if leg.kind is Kind.OTHER_SIDE:
        return (
            f"{shown}, the other side, so no stop is placed: close the position by hand in OpenAlgo"
        )
    return "it is placed again once the engine is restored"


__all__ = [
    "START_BOOKS_RETRY_S",
    "START_BOOKS_WAIT_S",
    "START_POLL_S",
    "UNBOUND_AT_START_ALERT_S",
    "LegStart",
    "Restart",
    "StartBooks",
    "judge_start",
    "read_start",
]

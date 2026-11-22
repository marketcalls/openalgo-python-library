"""What the session invariants are checked over (16-session-design 7.1, unit 14): every change
of every order at the fake OpenAlgo, recorded as it happens, and what each session run on
that fake left.

- **The order changes** (`OrderChange`): each change of an order of either book, at the
  moment the fake makes it (its own book, not the order stream, which a fault may hold or
  lose, and which never carries a sandbox modify), with the client that placed the order
  and the net position of its instrument in that book right after it. Recorded from the
  fake's construction on, so every position is the sum of the fills before it.
- **The runs** (`Run`): each session that opened on the fake, in order: the session
  object, the names of its clients (`FakeOpenAlgo.client`), its event stream and its
  alerts as it sent them, where the fake's log stood when it opened, when its mode guard
  locked and when it closed, each strategy's kernel state as its close stopped it
  (`Closing`), the outcome its close returned (None for a run that crashed or whose
  close failed), each order word a kernel took that was not of its own orders
  (`Run.strayed`, 18-multi-strategy 5.1 I5), and how many of its book's order changes
  while it was open had the members' books read and how many could not (`Run.sampled`,
  `Run.unsampled`).
- **The members' books at each change** (`OrderChange.members`, 18-multi-strategy 5.1 I4
  and I6): for each strategy of the run open then that trades the change's instrument in
  its book, its kernel's units, the units its kernel adopted from other orders' trades,
  and what each of its bound orders' fill ledgers has given the kernel.

`session_hooks.trace_sessions` installs the recording for one fake and has the invariants
run when the test ends.

A test whose session cannot keep one of the checks says which and why (`SessionTrace.waive`);
one whose row leaves a leg without a confirmed stop names the leg
(`SessionTrace.leave_unprotected`). A test whose other client sells by hand says what the
decided end it plays leaves at the fake beyond 7.1's list, each such test listed in
FAULT_ROWS.md for the lead: a stop larger than the position, or working on a flat one,
that the close names (`SessionTrace.kept_stop_by_decision`), or the session's stop filled
against what the sale left, through zero (`SessionTrace.oversold_by_decision`).

Times are UTC nanoseconds, quantities signed units, prices rupees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fake_openalgo import FakeOpenAlgo

from openalgo.backtest._engine import call
from openalgo.backtest._live._order_state import ENTRY_PURPOSES
from openalgo.backtest._live._types import Mode

#: The checks of 7.1 (`session_invariants`), by the name a waiver gives.
ONE_STOP = "one stop per leg"
WITHIN = "exits within the position"
OTHER_SIDE = "never the other side"
NO_ORPHAN = "no orphan"
PROTECTED = "protected"
BOOKS_AGREE = "books agree"
ANSWERED = "every modify and cancel answered"
GUARD_LOCK = "nothing after a guard lock"
PRICED = "priced fills"
SUM_RULE = "the sum rule"
ISOLATION = "isolation"
ONE_SIDE = "one side"
#: I3 per member (18-multi-strategy 5.1), waived on its own: a test may waive the group's
#: exits against the broker's net (`WITHIN`) and keep each member's against its own units.
MEMBERS_WITHIN = "each member's exits within its own position"
CHECKS = frozenset(
    {
        ONE_STOP,
        WITHIN,
        OTHER_SIDE,
        NO_ORPHAN,
        PROTECTED,
        BOOKS_AGREE,
        ANSWERED,
        GUARD_LOCK,
        PRICED,
        SUM_RULE,
        ISOLATION,
        ONE_SIDE,
        MEMBERS_WITHIN,
    }
)
#: The rows 7.1 lists as leaving a leg without a confirmed stop at the end, and the two
#: rows of a shared instrument 18-multi-strategy 5.1 (I7) adds.
LISTED_ROWS = frozenset({"F35", "F36", "F60", "F77 variant", "F80", "F105", "F118"})


@dataclass(frozen=True)
class Member:
    """A strategy's books as one change at the fake found them (module docstring): its
    kernel's signed units, the signed units it adopted from other orders' trades, and the
    units each of its bound orders' fill ledgers has given the kernel, by broker order."""

    strategy: str
    units: int
    adopted: int
    given: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class OrderChange:
    """An order as it stood right after one change at the fake (module docstring): the
    change's place in the sequence, its time, the book's mode, the order's terms and
    state, the units and price of the fill the change made (0 for none), the net
    position of the order's instrument in that book after it, the run's members of the
    instrument then (None while no run is open, or a kernel cannot be read) and the
    names of that run's clients."""

    seq: int
    ts_ns: int
    mode: Mode
    order_id: str
    client: str
    strategy: str
    symbol: str
    exchange: str
    action: str
    price_type: str
    status: str
    quantity: int
    filled: int
    fill_units: int
    fill_price: float
    position: int
    members: tuple[Member, ...] | None = None
    run_clients: frozenset[str] = frozenset()

    @property
    def working(self) -> bool:
        """True while the order can still fill."""
        return self.status in ("open", "trigger pending")

    @property
    def remaining(self) -> int:
        """The units not filled yet."""
        return self.quantity - self.filled

    @property
    def instrument(self) -> tuple[Mode, str, str]:
        return (self.mode, self.symbol, self.exchange)


@dataclass
class Closing:
    """A strategy's kernel state as the close stopped its kernel session: its signed units,
    whether its leg wants a broker-held stop (a stop level is set), its working orders'
    broker order IDs (those bound) and how many work with none bound yet (sent with no
    answer, or not sent), and the units its stop covers at the broker as the stops
    supervisor confirmed it working (`StopsSupervisor.covered`: 0 for none)."""

    strategy: str
    symbol: str
    exchange: str
    units: float
    wants_stop: bool
    working_order_ids: frozenset[str]
    stop_covers: int
    unbound_working: int = 0


@dataclass
class Run:
    """One session run on the fake (module docstring). `ended_at` is how many order changes
    the fake had made when its close returned (None while it has not); `log_from` and
    `log_to` are how many entries the fake's log held when the run opened and when its
    close returned; `locked_at` how many it held when the run's mode guard locked (None:
    it never did). `alerts` are the (kind, text) of each alert the run sent. `sampled`
    and `unsampled` count the order changes of its book while it was open whose members'
    books the trace read, and those it could not read (a kernel that failed)."""

    session: Any
    clients: frozenset[str]
    events: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[tuple[Any, str]] = field(default_factory=list)
    closing: dict[str, Closing] = field(default_factory=dict)
    outcome: Any = None
    ended_at: int | None = None
    log_from: int = 0
    log_to: int | None = None
    locked_at: int | None = None
    strayed: list[str] = field(default_factory=list)
    sampled: int = 0
    unsampled: int = 0

    @property
    def mode(self) -> Mode:
        return self.session.parts.mode


class SessionTrace:
    """The recording of one fake's order changes and session runs (module docstring)."""

    def __init__(self, fake: FakeOpenAlgo) -> None:
        self.fake = fake
        self.changes: list[OrderChange] = []
        self.runs: list[Run] = []
        #: The strategies whose leg the test leaves without a confirmed stop, with the
        #: listed row or the decided end that leaves it so.
        self.unprotected: dict[str, str] = {}
        #: The checks the test's session cannot keep, with why.
        self.waived: dict[str, str] = {}
        #: The strategies whose stop the test's decided end keeps larger than the position
        #: another client's sale left, or on the flat position it left, with that end.
        self.kept_stops: dict[str, str] = {}
        #: The strategies whose stop the test's decided end lets fill against what another
        #: client's sale left, taking the account through zero, with that end.
        self.oversold: dict[str, str] = {}
        #: Whether the invariants were checked already (`check_now`).
        self.checked = False
        self._positions: dict[tuple[Mode, str, str], int] = {}
        self._last: Any = None
        venue = fake._venue
        publish = venue._publish

        def published(book: Any, change: Any) -> None:
            # A change the book's modify returned is recorded already (`_record_modifies`).
            if change is not self._last:
                self._record(book.mode, change)
            publish(book, change)

        venue._publish = published
        for book in venue.books.values():
            self._record_modifies(book)

    def leave_unprotected(self, strategy: str, row: str) -> None:
        """The test (one of the rows 7.1 lists: F35, F36, F60, the F77 variant, F80, and
        18-multi-strategy's F105 and F118) asserts `strategy`'s leg unprotected in
        `open_positions` and the ERROR alert that names it: the end's protection
        invariant takes that instead of a stop. Raises ValueError for a row 7.1 does not
        list (`left_unprotected_by_decision`)."""
        if row not in LISTED_ROWS:
            raise ValueError(f"7.1 lists {sorted(LISTED_ROWS)}, not {row!r}")
        self.unprotected[strategy] = row

    def left_unprotected_by_decision(self, strategy: str, decision: str) -> None:
        """As `leave_unprotected`, for a decided end of 16-session-design that leaves the
        leg without a confirmed stop (`decision` names the paragraph); each is listed in
        FAULT_ROWS.md for the lead to add to 7.1's list."""
        self.unprotected[strategy] = decision

    def kept_stop_by_decision(self, strategy: str, decision: str) -> None:
        """Another client sells part or all of `strategy`'s position by hand while its stop
        works, and the decided end of 16-session-design that `decision` names keeps that
        stop at the broker, larger than what is left, to the end. The checks then take a
        stop that was working when the sale came, beyond the position by no more than the
        sale left it there, once an ERROR alert of the run names its order ID; an exit
        placed or raised after the sale gets no such room. Listed in FAULT_ROWS.md for the
        lead."""
        self.kept_stops[strategy] = decision

    def oversold_by_decision(self, strategy: str, decision: str) -> None:
        """Another client sells part of `strategy`'s position by hand while its stop works,
        and the decided end of 16-session-design that `decision` names lets the stop fill
        for the whole position before the session fits it (5.3 fits it on the second
        comparison that shows the sale), taking the account through zero. The checks then
        take such a fill of a stop that was working when the sale came, through zero by
        no more than the sale left it beyond the position, once the run lists the leg in
        `open_positions` at the units the fake holds, with no confirmed stop, and an ERROR
        alert names the strategy. Listed in FAULT_ROWS.md for the lead."""
        self.oversold[strategy] = decision

    def waive(self, check: str, reason: str) -> None:
        """Do not assert `check` (one of `CHECKS`) for this test, for `reason` (said in the
        test too); every other check still runs. Raises ValueError for an unknown check."""
        if check not in CHECKS:
            raise ValueError(f"no check {check!r}: use one of {sorted(CHECKS)}")
        self.waived[check] = reason

    def check_now(self, check: Callable[[SessionTrace, FakeOpenAlgo], None]) -> None:
        """Run `check` over this trace now, inside the test, and not again when it ends:
        for a self-test of the invariants that expects `check` to raise InvariantBroken
        on a session that breaks one."""
        self.checked = True
        check(self, self.fake)

    def session_clients(self) -> frozenset[str]:
        """The names of every client a run of this trace used."""
        return frozenset(name for run in self.runs for name in run.clients)

    def entry_orders(self) -> frozenset[str]:
        """The broker order IDs of the orders the runs' venue maps placed as entries or
        adds (a reversal is an entry, `venue_roles`): what may move a leg to the other
        side (`session_invariants_changes`)."""
        return frozenset(
            placed.order_id
            for run in self.runs
            for trader in run.session.traders
            for _, placed in trader.venue.placed()
            if placed.order_id and placed.intent.purpose in ENTRY_PURPOSES
        )

    def _record_modifies(self, book: Any) -> None:
        """Record each modify of `book` as the book makes it: the sandbox's is never
        published (`Venue.modify`), and the broker's is published right after."""
        modify = book.modify

        def modified(order: Any, **terms: Any) -> Any:
            change = modify(order, **terms)
            self._record(book.mode, change)
            self._last = change
            return change

        book.modify = modified

    def _record(self, mode: Mode, change: Any) -> None:
        order, fill = change.order, change.fill
        key = (mode, order.symbol, order.exchange)
        units = 0 if fill is None else fill.quantity
        if units:
            signed = units if order.action == "BUY" else -units
            self._positions[key] = self._positions.get(key, 0) + signed
        self.changes.append(
            OrderChange(
                len(self.changes),
                self.fake.clock.now(),
                mode,
                order.order_id,
                order.client,
                order.strategy,
                order.symbol,
                order.exchange,
                order.action,
                order.price_type,
                order.status,
                order.quantity,
                order.filled,
                units,
                0.0 if fill is None else fill.price,
                self._positions.get(key, 0),
                *self._members(mode, order.symbol, order.exchange),
            )
        )

    def _members(
        self, mode: Mode, symbol: str, exchange: str
    ) -> tuple[tuple[Member, ...] | None, frozenset[str]]:
        """The open run's strategies of `exchange:symbol` in book `mode`, as their kernels
        and venue maps stand now (module docstring), and the run's clients."""
        run = next((run for run in reversed(self.runs) if run.ended_at is None), None)
        if run is None or run.mode is not mode:
            return None, frozenset()
        session = run.session
        try:
            members = tuple(
                _member(session, trader)
                for trader in session.traders
                if (trader.symbol, trader.exchange) == (symbol, exchange)
            )
        # Any failure of a kernel (a test's hook broke it, the run ended): no sample,
        # counted, so that a run with none at all fails I4 and I6 rather than pass them.
        except Exception:
            run.unsampled += 1
            return None, frozenset()
        run.sampled += 1
        return members, run.clients


def _member(session: Any, trader: Any) -> Member:
    given = tuple(
        (placed.order_id, trader.venue.filled(kernel_order))
        for kernel_order, placed in trader.venue.placed()
        if placed.order_id
    )
    adopted = session.reconcile.adopted(trader.name)
    return Member(trader.name, round(trader.open_units()), adopted, given)


def closing_of(session: Any, trader: Any) -> Closing | None:
    """`trader`'s `Closing` read from its running kernel session and `session`'s stops
    supervisor, None when the kernel cannot be read (a test's hook broke it)."""
    try:
        units = trader.open_units()
        state = call(trader.engine.live_state)
        working = set(trader.working())
        bound = {kernel: placed.order_id for kernel, placed in trader.venue.placed()}
        covers = int(session.stops.covered(trader))
    # Any failure of the kernel: the end checks then take what the fake shows.
    except Exception:
        return None
    wants = any(leg.get("stop") is not None for leg in state.get("legs", []))
    order_ids = frozenset(bound[k] for k in working if bound.get(k))
    unbound = len(working) - len(order_ids)
    return Closing(
        trader.name, trader.symbol, trader.exchange, units, wants, order_ids, covers, unbound
    )


__all__ = [
    "ANSWERED",
    "BOOKS_AGREE",
    "CHECKS",
    "GUARD_LOCK",
    "ISOLATION",
    "LISTED_ROWS",
    "MEMBERS_WITHIN",
    "NO_ORPHAN",
    "ONE_SIDE",
    "ONE_STOP",
    "OTHER_SIDE",
    "PRICED",
    "PROTECTED",
    "SUM_RULE",
    "WITHIN",
    "Closing",
    "Member",
    "OrderChange",
    "Run",
    "SessionTrace",
    "closing_of",
]

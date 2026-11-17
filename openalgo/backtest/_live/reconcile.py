"""The reconciler of a live session (16-session-design 5; unit 12): the broker's books read
on a schedule and compared, leg by leg, with what the engine knows, by the rules of
`_reconcile_rules` (decided 2026-11-16: a mismatch locks entries and alerts, with only the
safe corrections; it never exits a position and never fires the kill switch).

**Schedule** (5.1): a read when trading begins, every `RECONCILE_EVERY_S`,
`RECONCILE_AFTER_FILL_S` after a fill, `RECONCILE_AGAIN_AFTER_S` after a comparison that
found a difference the one before it did not (the second confirms it; one seen twice in a
row waits for the schedule), `SQUARE_OFF_READ_BEFORE_S` before and
`SQUARE_OFF_READ_AFTER_S` after each strategy's square-off, when the safe stop's flatten
begins (`read_before_exits`), and at the common close (`_reconcile_final`); never after
the mode guard locked. One read at a time, numbered: the order worker reads the position,
order and trade books and the funds (`ReadBooks`) and answers `Books`. Before the flatten
and the square-off the exit guard holds each leg's exits until a read begun after the
request has compared it (a sale by hand since the last read is not known before), read
again every `RECONCILE_AGAIN_AFTER_S` while a leg cannot be compared.

**In flight** (`_reconcile_act.in_flight`). A read compares a leg only when it was out of
flight from before the read began (a fill restarts that) and still is; a leg in flight
past `RECONCILE_IN_FLIGHT_MAX_S` is compared anyway (late): a difference is alerted, with
no automatic action and no sighting. A fill of the leg's own orders the read's trade book
gives the kernel restarts flight too, as its position book, read first, may predate it.

**A comparison** (5.2), in order: (1) the trade book's trades of the session's own orders
go to their orders' fill ledgers as news (a fill the stream and the polls missed is applied
to its own order; one applied already changes nothing); (2) an order the engine works whose
row has ended asks for a poll of the followed orders; an order of the session's working at
the broker that the engine does not hold is an orphan, and a working row naming the
strategy that the session did not send is another's; (3) the positions are judged
(`_reconcile_rules.judge`); (4) a working stop's units and trigger against the engine's.
A finding seen in two comparisons in a row acts, once (`_reconcile_findings`). From
the first comparison showing the broker holding fewer units, none or the other side, the
exit guard (`_reconcile_guard`) holds every closing order beyond what the broker holds,
and the stop's cancel that makes way for one, until a comparison agrees. Once the close
has printed its last lines (`end`), a difference a leg's last comparison saw for the
first time is never confirmed: the close reports the leg at OpenAlgo's units, moved by
the engine's own fills since (`unconfirmed`, `_reconcile_close`; 5.4 "As built", rounds 7
to 9), and one no comparison judged after such a fill as not confirmed (`unjudged`). A
leg whose last comparison was late and saw a difference is named by that read's figures
instead, neither carried nor confirmed (`late`; the unit 12 follow-up).

Each read's position book also counts for the kill switch (`KillSwitch.positions`, its
interface to the reconciler, which never fires it), and its funds cap each strategy's
sizing (6.6, R18). Every comparison is an event; a difference is a row of `rows` when
first seen and when seen again in the next comparison (what was done then), not at each
read while it lasts (`clean` while none was found). Times are UTC nanoseconds,
quantities signed units, prices rupees. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._reconcile_act import Corrections, attribute, in_flight, order_findings, own_orders
from ._reconcile_findings import Findings, Reconciliation, Sighting
from ._reconcile_guard import ExitGuard
from ._reconcile_rules import (
    GUARDED,
    Kind,
    LegBooks,
    available_cash,
    judge,
    leg_books,
    unexplained_by,
)
from ._session_trader import Trader
from ._worker_queue import ReadBooks, StreamGap
from .inbox import Books, PositionRows

#: Seconds between two reads (08-live-trading section 7).
RECONCILE_EVERY_S = 30.0
#: Seconds after the last fill, and after a comparison that differed, before a read.
RECONCILE_AFTER_FILL_S = 3.0
RECONCILE_AGAIN_AFTER_S = 3.0
#: Seconds a leg may stay in flight before it is compared anyway: the order path's longest
#: legitimate wait (`BROKER_WINDOW_NS`, 150 s) plus two book reads and a margin (5.1).
RECONCILE_IN_FLIGHT_MAX_S = 180.0
#: Seconds after a strategy's square-off before the books are read: the session's own
#: check that every leg is flat then (16-session-design 3.4).
SQUARE_OFF_READ_AFTER_S = 30.0
#: Seconds before a strategy's square-off at which the books are read, so its exit knows
#: what the broker holds (5.4 "As built"): a read of the four books takes a second or two.
SQUARE_OFF_READ_BEFORE_S = 5.0
#: Seconds after which a read asked for and never answered is asked for again: well past
#: the orders client's own wait (the SDK's 120 s), as for the kill switch's reads.
RECONCILE_READ_LOST_AFTER_S = 300.0
#: What the "reconcile" event of a leg says it found when a fill of the leg's own orders
#: landed during the read: nothing judged (module docstring).
FILL_DURING_READ = "fill_during_read"

_NS = NANOS_PER_SECOND


class Reconciler:
    """The reconciler of `session`, a `_session_loop.Session` (module docstring). `guard`
    is its exit guard, `rows` its differences."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.guard = ExitGuard(session)
        self._adopted: dict[str, int] = {}
        self._orphans: set[str] = set()
        corrections = Corrections(session, self._adopted, self._orphans)
        self._findings = Findings(session, corrections, RECONCILE_IN_FLIGHT_MAX_S)
        self.rows: list[Reconciliation] = self._findings.rows
        self._generation = 0
        self._asked_ns: int | None = None
        self._answered = True
        self._next_ns: int | None = None
        self._soon_ns: int | None = None
        self._square_offs: list[int] = []
        self._before: dict[str, int] = {}
        self._quiet: dict[str, int | None] = {}
        self._flight: dict[str, int | None] = {}
        self._earlier: frozenset[str] = frozenset()
        # The strategies the read asked for compares, all for None (`ask_of`).
        self._only: frozenset[str] | None = None
        self._stopped = False
        self._ended = False

    # ----- the start and what the loop tells it -----------------------------------------

    def start(self, adopted: Mapping[str, int], earlier: frozenset[str]) -> None:
        """Before trading: the units each strategy adopted before a restart, and the
        broker order IDs of the day's earlier starts (the session's own, by the journal)."""
        self._adopted.update(adopted)
        self._earlier = earlier

    def differs_at_start(
        self, trader: Trader, kind: Kind, journal_units: int, books: LegBooks
    ) -> None:
        """A restart found `trader`'s journal position (`journal_units`) different from
        the broker's (5.5): acted on at once, as 5.3 would after two comparisons, with the
        strategy's entries locked from the start; the exit guard opened while the broker
        holds fewer units that nothing explains, or the other side. When the restored
        engine holds other units (a fill reached the journal after the start read the
        books: the order worker follows the day's orders during the warm-up), those books
        no longer describe the leg: its entries are locked, and its exits wait for a
        comparison of a fresh read, which judges it."""
        own = own_orders(trader, self._earlier)
        if round(trader.open_units()) != journal_units:
            self._findings.at_start(trader, kind, journal_units, books, own, stale=True)
            self.guard.await_read(trader.name, self._session.now())
            return
        self._findings.at_start(trader, kind, journal_units, books, own)
        if kind in (Kind.UNEXPLAINED, Kind.OTHER_SIDE) and books.units is not None:
            self.guard.open(trader.name, books.units)

    def adopted(self, name: str) -> int:
        """The signed units strategy `name` adopted from other orders' trades today."""
        return self._adopted.get(name, 0)

    def filled(self, trader: Trader, at_ns: int) -> None:
        """A fill of `trader`'s reached the kernel at `at_ns`: a read is due shortly, and
        only a read begun after it compares the leg."""
        if self._quiet.get(trader.name) is not None:
            self._quiet[trader.name] = at_ns
        self._soon(at_ns + _s(RECONCILE_AFTER_FILL_S))

    def holds(self, trader: Trader) -> bool:
        """True while the exit guard holds `trader`'s leg."""
        return self.guard.guards(trader.name)

    def orphan(self, order_id: str) -> bool:
        """True for an orphan the reconciler asked to cancel: its news is expected."""
        return order_id in self._orphans

    def end(self) -> None:
        """The close has printed its last lines: the exit guard ends (`ExitGuard.end`),
        and no later comparison will confirm a difference first seen (module docstring)."""
        self._ended = True
        self._findings.end()
        self.guard.end()

    def unconfirmed(self, name: str) -> Sighting | None:
        """Once the close has printed its last lines, the position difference strategy
        `name`'s last comparison saw for the first time, else None; None too when its last
        comparison was late with a difference (`late` names the leg then)."""
        if not self._ended or self._findings.late(name) is not None:
            return None
        return self._findings.unconfirmed(name)

    def unjudged(self, name: str) -> bool:
        """Once the close has printed its last lines, True when no comparison judged
        strategy `name`'s position since a fill of its own orders landed during a read,
        unless a late comparison since saw a difference (`late` names the leg then)."""
        findings = self._findings
        return self._ended and findings.unjudged(name) and findings.late(name) is None

    def late(self, name: str) -> Sighting | None:
        """Once the close has printed its last lines, the position difference strategy
        `name`'s last comparison saw when it was late (the leg in flight past the limit):
        named at the close as that read saw it, neither carried nor confirmed; else None."""
        return self._findings.late(name) if self._ended else None

    def broker_units(self, name: str) -> int | None:
        """OpenAlgo's signed units of strategy `name`'s leg at the last read while they
        differ from the engine's with nothing taking them in: while the exit guard holds
        the leg (`ExitGuard.broker_units`), and after the last lines when no later
        comparison will confirm the difference (`unconfirmed`); else None."""
        guarded, seen = self.guard.broker_units(name), self.unconfirmed(name)
        return guarded if guarded is not None or seen is None else seen.broker_units

    def not_read(self) -> None:
        """The worker did not read the books (the guard locked): nothing more is read."""
        self._stopped = True

    @property
    def waiting(self) -> bool:
        """True while a read asked for has not come."""
        return not self._answered and not self._stopped

    @property
    def stopped(self) -> bool:
        """True once the worker did not read the books (`not_read`)."""
        return self._stopped

    @property
    def clean(self) -> bool:
        """True while no comparison has differed."""
        return not self.rows

    # ----- step D -----------------------------------------------------------------------

    def duties(self, now_ns: int) -> None:
        """Note each leg in flight or out of it, and ask for a read when one is due."""
        session = self._session
        if not self._reads(session):
            return
        for trader in session.traders:
            name = trader.name
            if in_flight(session, trader):
                self._quiet[name] = None
                self._flight[name] = self._flight.get(name) or now_ns
            else:
                self._flight[name] = None
                if self._quiet.get(name) is None:
                    self._quiet[name] = now_ns
        if self._next_ns is None:
            self._next_ns = now_ns
            after, before = _s(SQUARE_OFF_READ_AFTER_S), _s(SQUARE_OFF_READ_BEFORE_S)
            self._square_offs = sorted({t.square_off_ns + after for t in session.traders})
            self._before = {t.name: t.square_off_ns - before for t in session.traders}
        due = [name for name, before_ns in self._before.items() if before_ns <= now_ns]
        if due:
            self._before = {n: at for n, at in self._before.items() if n not in due}
            self.read_before_exits(due)
        self._ask_when_due(now_ns)

    def read_before_exits(self, names: Iterable[str] | None = None) -> None:
        """The exits of strategies `names` (all for None) wait for a comparison of a read
        asked for now (module docstring): the safe stop's flatten and the square-off. A
        finished kernel session is never compared again, so it waits for nothing. The
        read is asked for in this step, so the exits wait one read and no more; while one
        is asked for already, its answer asks for the next at once when it began earlier
        (`books`)."""
        session = self._session
        now_ns = session.now()
        for trader in session.traders:
            if not trader.finished and (names is None or trader.name in names):
                self.guard.await_read(trader.name, now_ns)
        if not self._answered:
            return
        self._soon(now_ns)
        if self._reads(session):
            self._ask_when_due(now_ns)

    def ask(self, now_ns: int) -> None:
        """Ask the order worker for a read now, of every leg."""
        self._only = None
        self._generation += 1
        self._asked_ns, self._answered = now_ns, False
        self._next_ns, self._soon_ns = now_ns + _s(RECONCILE_EVERY_S), None
        self._square_offs = [instant for instant in self._square_offs if instant > now_ns]
        self._session.hand_over([ReadBooks(self._generation)])

    def ask_of(self, names: frozenset[str] | None) -> None:
        """Ask for a read now that compares the legs of strategies `names` alone (all for
        None), each out of flight from now: the close's reads (`_reconcile_final`), after
        the last lines, when no duty notes flight."""
        now_ns = self._session.now()
        traders = self._session.traders
        self._quiet.update({t.name: now_ns for t in traders if names is None or t.name in names})
        self.ask(now_ns)
        self._only = names

    def _reads(self, session: Any) -> bool:
        """Whether reads are made: once trading began, never after a guard lock."""
        return session.trading and not (session.locked or self._stopped)

    def _ask_when_due(self, now_ns: int) -> None:
        # While OpenAlgo is silent the order worker's probe asks it (4.9): a read could
        # hold the worker for the client's whole timeout, ahead of the stops and exits.
        if self._session.parts.worker.silent_since_ns is not None:
            return
        if self._answered:
            if self._due(now_ns):
                self.ask(now_ns)
        elif self._asked_ns is None or now_ns >= self._asked_ns + _s(RECONCILE_READ_LOST_AFTER_S):
            self.ask(now_ns)

    def _due(self, now_ns: int) -> bool:
        due = [self._next_ns if self._next_ns is not None else now_ns]
        due += [self._soon_ns] if self._soon_ns is not None else []
        due += [instant for instant in self._square_offs if instant <= now_ns]
        return now_ns >= min(due)

    def _soon(self, due_ns: int) -> None:
        self._soon_ns = due_ns if self._soon_ns is None else min(self._soon_ns, due_ns)

    # ----- a read -----------------------------------------------------------------------

    def books(self, message: Books) -> None:
        """A read came: compare each leg it may compare (module docstring)."""
        session = self._session
        if message.generation != self._generation or session.locked:
            return
        self._answered = True
        session.risk.positions(PositionRows(message.positionbook, message.started_ns))
        # The funds cap each strategy's sizing by the account's available cash (6.6, R18).
        available = available_cash(message.funds)
        polls = False
        for trader in session.traders:
            if available is not None and not trader.finished:
                call(trader.engine.cap_sizing_equity, available)
            if trader.finished or (self._only is not None and trader.name not in self._only):
                continue
            since = self._flight.get(trader.name)
            late = since is not None and session.now() - since > _s(RECONCILE_IN_FLIGHT_MAX_S)
            quiet = self._quiet.get(trader.name)
            if not late and (quiet is None or quiet > message.started_ns):
                continue
            if not late and in_flight(session, trader):
                continue
            polls |= self._compare(trader, message, late=late)
        if polls:
            session.hand_over([StreamGap(message.started_ns)])
        awaited = self.guard.awaited_since()
        if awaited:
            # A wait that began after this read began needs a read at once; a leg the read
            # could not compare (in flight, unreadable) another after a short while.
            again = 0 if max(awaited) > message.started_ns else _s(RECONCILE_AGAIN_AFTER_S)
            self._soon(session.now() + again)

    def _compare(self, trader: Trader, message: Books, *, late: bool) -> bool:
        """Compare one leg (module docstring); True when an order's row has ended while
        the engine works it, so the followed orders are polled."""
        books = leg_books(
            message.positionbook,
            message.tradebook,
            message.orderbook,
            trader.symbol,
            trader.exchange,
        )
        own = own_orders(trader, self._earlier)
        late_or_unread = late or books.trades is None
        if not late_or_unread and attribute(self._session, trader, books, message.ended_ns):
            # The read took its position book first, which may predate the fill: it judges
            # nothing of the leg, and the fill asked for a read soon (`filled`).
            self._findings.fill_during_read(trader.name)
            self._noted(trader, FILL_DURING_READ, books.units, 0)
            return False
        engine = round(trader.open_units())
        findings, polls = order_findings(self._session, trader, books, own)
        kind = Kind.UNREADABLE
        if books.units is not None and books.trades is not None:
            external = sum(trade.signed for trade in books.external(own))
            unexplained = unexplained_by(engine, books.units, external, self.adopted(trader.name))
            kind = judge(engine, books.units, unexplained)
        if kind not in (Kind.AGREE, Kind.UNREADABLE):
            findings.add((kind, engine, books.units))
        self._noted(trader, kind.value, books.units, len(findings))
        if kind in GUARDED and books.units is not None:
            self.guard.open(trader.name, books.units)
        if self._findings.judged(trader, findings, books, own, late=late):
            self._soon(self._session.now() + _s(RECONCILE_AGAIN_AFTER_S))
        if kind in (Kind.AGREE, Kind.MORE) or round(trader.open_units()) == books.units:
            self.guard.close(trader.name)
        if kind is not Kind.UNREADABLE:
            self.guard.read(trader.name, message.started_ns)
        return polls

    def _noted(self, trader: Trader, found: str, broker_units: int | None, findings: int) -> None:
        """A comparison's event: what it found of the leg, both figures, the findings."""
        self._session.event(
            "reconcile",
            strategy=trader.name,
            found=found,
            engine_units=round(trader.open_units()),
            broker_units=broker_units,
            findings=findings,
        )


def _s(seconds: float) -> int:
    return round(seconds * _NS)


__all__ = [
    "FILL_DURING_READ",
    "RECONCILE_AFTER_FILL_S",
    "RECONCILE_AGAIN_AFTER_S",
    "RECONCILE_EVERY_S",
    "RECONCILE_IN_FLIGHT_MAX_S",
    "RECONCILE_READ_LOST_AFTER_S",
    "SQUARE_OFF_READ_AFTER_S",
    "SQUARE_OFF_READ_BEFORE_S",
    "Reconciler",
]

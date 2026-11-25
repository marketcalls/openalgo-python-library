"""The reconciler of a live session (16-session-design 5; unit 12): the broker's books read
on a schedule and compared, leg by leg, with what the engine knows, by the rules of
`_reconcile_rules` (decided 2026-11-16: a mismatch locks entries and alerts, with only the
safe corrections; it never exits a position and never fires the kill switch).

**Schedule** (5.1): when a read is due, and the reads the session asks for, are
`_reconcile_schedule`'s. An instrument several strategies of the session share in one
product is compared as one group (`_reconcile_groups`, 18-multi-strategy 4.6); each leg's
broker units are its product's (`_broker_units`), a CNC leg's judged one-sided: the user's
own units beyond the strategy's are reported once (a SESSION alert) and never a mismatch
(20-positional-live 7.2).

**In flight** (`_reconcile_steps.in_flight`; a GTT fired or gone and not resolved holds its
leg in flight too, `gtts`). A read compares a leg only when it was out of
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
instead, neither carried nor confirmed (`late`; the unit 12 follow-up), and counted no
more when comparisons in time had acted on it (`late_counted`; the unit 13 follow-up).
These queries of the close are `_reconcile_ended`'s.

Each read's position book also counts for the kill switch (`KillSwitch.positions`, its
interface to the reconciler, which never fires it), and its funds cap each strategy's
sizing (6.6, R18). Every comparison is an event; a difference is a row of `rows` when
first seen and when seen again in the next comparison (what was done then), not at each
read while it lasts (`clean` while none was found). Times are UTC nanoseconds,
quantities signed units, prices rupees. Used by the loop only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._broker_units import own_units, strategy_units
from ._group_rules import GroupFigure
from ._groups import GroupKey
from ._gtt_flight import GttFlight
from ._reconcile_act import Corrections
from ._reconcile_ended import EndQueries
from ._reconcile_findings import FILL_DURING_READ, Findings, Reconciliation
from ._reconcile_groups import GroupComparison, GroupRead
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
from ._reconcile_schedule import (
    RECONCILE_AFTER_FILL_S,
    RECONCILE_AGAIN_AFTER_S,
    RECONCILE_EVERY_S,
    RECONCILE_READ_LOST_AFTER_S,
    SQUARE_OFF_READ_AFTER_S,
    SQUARE_OFF_READ_BEFORE_S,
    ReadSchedule,
)
from ._reconcile_steps import attribute, in_flight, order_findings, own_orders
from ._session_trader import Trader
from ._types import AlertKind
from ._worker_queue import StreamGap
from .inbox import Books, PositionRows

#: Seconds a leg may stay in flight before it is compared anyway: the order path's longest
#: legitimate wait (`BROKER_WINDOW_NS`, 150 s) plus two book reads and a margin (5.1).
RECONCILE_IN_FLIGHT_MAX_S = 180.0
_NS = NANOS_PER_SECOND
#: The differences a restart acts on with nothing adopted: the exit guard holds the leg.
UNADOPTED = frozenset({Kind.UNEXPLAINED, Kind.OTHER_SIDE})


class Reconciler(EndQueries, GroupComparison, ReadSchedule):
    """The reconciler of `session`, a `_session_loop.Session` (module docstring). `guard`
    is its exit guard, `rows` its differences."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.guard = ExitGuard(session)
        #: The legs a GTT holds in flight until the GTT follower resolves it (`_gtt_flight`).
        self.gtts = GttFlight()
        self._own_reported: set[str] = set()
        self._adopted: dict[str, int] = {}
        self._orphans: set[str] = set()
        # By strategy: its shared instrument's group as the last comparison judged it.
        self._figures: dict[str, GroupFigure] = {}
        # By shared instrument and product: its last read (the broker's net, the members'
        # sum, late).
        self._group_reads: dict[GroupKey, GroupRead] = {}
        corrections = Corrections(session, self._adopted, self._orphans, self._figures)
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
        self,
        trader: Trader,
        kind: Kind,
        journal_units: int,
        books: LegBooks,
        group: GroupFigure | None = None,
    ) -> None:
        """A restart found `trader`'s journal position (`journal_units`), or its `group`'s
        sum on a shared instrument, different from the broker's (5.5; 18-multi-strategy
        4.10): acted on at once, as 5.3 would after two comparisons, with the strategy's
        entries locked from the start; the exit guard opened while the broker holds fewer
        units that nothing adopts, or the other side (every closing order of a member of
        a group held). When the restored engine holds other units (a fill reached the
        journal after the start read the books: the order worker follows the day's
        orders during the warm-up), those books no longer describe the leg: its entries
        are locked, and its exits wait for a comparison of a fresh read, which judges it."""
        own = own_orders(trader, self._earlier)
        if group is not None:
            self._figures[trader.name] = group
        if round(trader.open_units()) != journal_units:
            self._findings.at_start(
                trader, kind, journal_units, books, own, stale=True, group=group
            )
            self.guard.await_read(trader.name, self._session.now())
            return
        self._findings.at_start(trader, kind, journal_units, books, own, group=group)
        if kind in UNADOPTED and books.units is not None:
            self.guard.open(trader.name, books.units, hold_all=group is not None)

    def adopted(self, name: str) -> int:
        """The signed units strategy `name` adopted from other orders' trades today."""
        return self._adopted.get(name, 0)

    def holds(self, trader: Trader) -> bool:
        """True while the exit guard holds `trader`'s leg."""
        return self.guard.guards(trader.name)

    def orphan(self, order_id: str) -> bool:
        """True for an orphan the reconciler asked to cancel: its news is expected."""
        return order_id in self._orphans

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

    # ----- a read -----------------------------------------------------------------------

    def books(self, message: Books) -> None:
        """A read came: compare each leg it may compare (module docstring)."""
        session = self._session
        if message.generation != self._generation or session.locked:
            return
        self._answered = True
        rows = PositionRows(message.positionbook, message.started_ns, holdings=message.holdings)
        session.risk.positions(rows)
        # The funds cap each strategy's sizing by the account's available cash (6.6, R18).
        available = available_cash(message.funds)
        polls = False
        for trader in session.traders:
            if available is not None and not trader.finished:
                call(trader.engine.cap_sizing_equity, available)
            if session.groups.shared(trader) or trader.finished:
                continue
            if self._only is not None and trader.name not in self._only:
                continue
            since = self._flight.get(trader.name)
            late = since is not None and session.now() - since > _s(RECONCILE_IN_FLIGHT_MAX_S)
            quiet = self._quiet.get(trader.name)
            if not late and (quiet is None or quiet > message.started_ns):
                continue
            if not late and in_flight(session, trader):
                continue
            polls |= self._compare(trader, message, late=late)
        limits = {"late_after_s": RECONCILE_IN_FLIGHT_MAX_S, "again_s": RECONCILE_AGAIN_AFTER_S}
        polls |= self._compare_shared(message, **limits)
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
            product=trader.product,
            holdings=message.holdings,
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
        own_held = own_units(trader.product, engine, books.units)
        if own_held:
            self._own_held(trader, own_held)
        books = dataclasses.replace(
            books, units=strategy_units(trader.product, engine, books.units)
        )
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

    def _own_held(self, trader: Trader, units: int) -> None:
        """The user's own units beyond a CNC strategy's (`_broker_units`, 20-positional-live
        7.2): reported once per strategy as information, never a mismatch."""
        if trader.name in self._own_reported:
            return
        self._own_reported.add(trader.name)
        self._session.event("own_holdings", strategy=trader.name, units=units)
        self._session.alert(
            AlertKind.SESSION,
            f"{trader.name}: the account holds {units} units of {trader.symbol} "
            f"{trader.exchange} ({trader.product}) beyond the strategy's: they are your own, "
            "and the session never sells them or covers them with its stop",
        )

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

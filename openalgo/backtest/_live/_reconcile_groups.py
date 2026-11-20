"""A comparison of an instrument several strategies of the session trade (18-multi-strategy
4.1, 4.2, 4.6 and 4.7; unit M6, the part the session needs before the group rows of unit
M7).

A broker nets the members of an instrument group into one position, so their legs are
compared together, never one by one against the broker's net (`GroupComparison`, a part
of `reconcile.Reconciler`). A group of one is compared as a leg always was
(`reconcile.Reconciler._compare`).

- **In flight**: the group is compared only when every member may be (out of flight
  since before the read began, and still), or, once a member has been in flight past
  `RECONCILE_IN_FLIGHT_MAX_S`, late: alerted, nothing done.
- **Trades first**: each member's own orders' trades go through its own fill ledgers
  (attribution by order ID, never by the strategy's name); a fill any member's ledger
  takes means the read judges nothing of the group. The trades of orders bound to no
  member (and not of the day's earlier starts) are the group's external trades.
- **Orders**: each member's rows as for a leg, but a row bound to another member is that
  member's, and an order of an earlier start that no member holds is the group's orphan,
  found by its first member only (`order_findings`).
- **Positions**: the sum of the members' units against the broker's net
  (`_group_rules.group_kind`). A difference trades of other orders explain
  (`Kind.EXPLAINED_SHARED`) is shared out among the members it closes, the most recently
  opened position first (`_reconcile_allocate.allocate`, their positions' oldest open
  tranche telling when each opened). Each member's finding is the group's kind with its
  own units and the broker's net, acted on, when two comparisons in a row show it, by the
  rules of a leg (16-session-design 5.3), its alert naming the group's sum: the entries
  locked and alerted; an explained difference's share adopted by each member, which cuts
  its stop to what is left; on the other side each member's own stop cancelled. While
  the broker holds fewer units than the sum, or none, or the other side, every closing
  order of every member is held by the exit guard until a comparison of the group agrees
  (18-multi-strategy 4.8: holding the whole group is simpler to verify than a room per
  member).
- **Agreement** tells the margin holds (`_margin_reject`): a position a closing order's
  rejection held may be exited again.
- **The last read** of each shared instrument (`group_read`, `GroupRead`; 18-multi-strategy
  4.11): the broker's net of every comparison that read its position book, kept with the
  members' sum once the comparison has acted and whether it was late, for the close's row
  and line of the instrument. A late read's position book may hold a fill of a member's
  order in flight that its engine hears only later, so the close names its net as that
  read saw it and never carries it by the members' fills since, as for a leg
  (`_reconcile_close.LATE_READ`).

Quantities are signed units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._group_rules import GroupFigure, Share, group_kind
from ._plan_catch_up import INSTRUMENT
from ._reconcile_allocate import allocate
from ._reconcile_findings import FILL_DURING_READ
from ._reconcile_rules import GUARDED, Kind, LegBooks, Trade, leg_books, unexplained_by
from ._reconcile_steps import Finding, attribute, in_flight, order_findings, own_orders
from ._session_trader import Trader
from ._venue_requests import LIVE_PRODUCT
from .inbox import Books

_NS = NANOS_PER_SECOND


@dataclass(frozen=True)
class GroupRead:
    """The last read of a shared instrument that a comparison of its group took
    (`GroupComparison.group_read`): the broker's `net` and the members' sum `held` once
    that comparison had acted (signed units), and whether it compared `late` (a member in
    flight past the limit: module docstring)."""

    net: int
    held: int
    late: bool = False


class GroupComparison:
    """The comparison of a shared instrument's group, part of `reconcile.Reconciler`
    (module docstring), which holds the session (`_session`), the exit guard (`guard`),
    the findings (`_findings`), the earlier starts' orders (`_earlier`), each leg's
    flight and quiet times (`_flight`, `_quiet`), the read's filter (`_only`) and its
    helpers."""

    _session: Any
    guard: Any
    _findings: Any
    _earlier: frozenset[str]
    _flight: dict[str, int | None]
    _quiet: dict[str, int | None]
    _only: frozenset[str] | None
    _figures: dict[str, GroupFigure]
    _group_reads: dict[tuple[str, str], GroupRead]
    adopted: Any
    _noted: Any
    _soon: Any

    def group_read(self, symbol: str, exchange: str) -> GroupRead | None:
        """The last read of the shared instrument `exchange:symbol` that a comparison of
        its group took (`GroupRead`); None when no comparison of this start read its
        position book."""
        return self._group_reads.get((symbol, exchange))

    def _compare_shared(self, message: Books, *, late_after_s: float, again_s: float) -> bool:
        """Compare each shared instrument's group with a read (`_compare_group`), in the
        order of their first members. True when an order's row has ended while its engine
        works it."""
        groups, polls = self._session.groups, False
        for group in groups.groups:
            if group.shared:
                members = groups.of(group.symbol, group.exchange)
                polls |= self._compare_group(
                    members, message, late_after_s=late_after_s, again_s=again_s
                )
        return polls

    def _compare_group(
        self, members: Sequence[Trader], message: Books, *, late_after_s: float, again_s: float
    ) -> bool:
        """Compare a shared instrument's `members` (call order) with a read (module
        docstring): late once a member has been in flight past `late_after_s`; a difference
        seen for the first time asks for a read `again_s` later. True when an order's row
        has ended while its engine works it."""
        session = self._session
        active = [
            m for m in members if not m.finished and (self._only is None or m.name in self._only)
        ]
        if not active:
            return False
        now_ns, max_ns = session.now(), round(late_after_s * _NS)
        late = any(
            (since := self._flight.get(m.name)) is not None and now_ns - since > max_ns
            for m in active
        )
        if not late and not all(self._comparable(m, message.started_ns) for m in members):
            return False
        books = leg_books(
            message.positionbook,
            message.tradebook,
            message.orderbook,
            members[0].symbol,
            members[0].exchange,
        )
        if not late and books.trades is not None and self._fills_landed(active, books, message):
            return False
        return self._judge_group(members, active, books, message, late=late, again_s=again_s)

    def _comparable(self, member: Trader, started_ns: int) -> bool:
        quiet = self._quiet.get(member.name)
        if member.finished:
            return True
        return quiet is not None and quiet <= started_ns and not in_flight(self._session, member)

    def _fills_landed(self, active: Sequence[Trader], books: LegBooks, message: Books) -> bool:
        """Give each member its own trades of the read; True when any took a fill, so the
        read judges nothing of the group (its position book may predate the fill)."""
        landed = False
        for member in active:
            landed |= attribute(self._session, member, books, message.ended_ns)
        if landed:
            for member in active:
                self._findings.fill_during_read(member.name)
                self._noted(member, FILL_DURING_READ, books.units, 0)
        return landed

    def _judge_group(
        self,
        members: Sequence[Trader],
        active: Sequence[Trader],
        books: LegBooks,
        message: Books,
        *,
        late: bool,
        again_s: float,
    ) -> bool:
        session = self._session
        bound = {m.name: frozenset(m.venue.bound()) for m in members}
        everyone = frozenset().union(*bound.values()) | self._earlier
        units = {m.name: round(m.open_units()) for m in members}
        group_units = sum(units.values())
        kind, shares, pending = Kind.UNREADABLE, (), None
        if books.units is not None and books.trades is not None:
            trades = books.external(everyone)
            external = sum(trade.signed for trade in trades)
            adopted = sum(self.adopted(m.name) for m in members)
            pending = external - adopted
            unexplained = unexplained_by(group_units, books.units, external, adopted)
            kind = group_kind(tuple(units.values()), books.units, unexplained)
            if kind is Kind.EXPLAINED_SHARED:
                shares = _shared_out(members, units, books.units - group_units, trades)
        names, held = tuple(units), tuple(units.values())
        figure = GroupFigure(names, group_units, shares, held, pending)
        polls = False
        for member in active:
            self._figures[member.name] = figure
            others = frozenset().union(*(ids for name, ids in bound.items() if name != member.name))
            findings: set[Finding]
            findings, polled = order_findings(
                session,
                member,
                books,
                own_orders(member, self._earlier),
                others=others,
                orphans_of_earlier=member is active[0],
            )
            polls |= polled
            if kind not in (Kind.AGREE, Kind.UNREADABLE):
                findings.add((kind, units[member.name], books.units))
            self._noted(member, kind.value, books.units, len(findings))
            if kind in GUARDED and books.units is not None:
                self.guard.open(member.name, books.units, hold_all=True)
            if self._findings.judged(member, findings, books, everyone, late=late, group=figure):
                self._soon(session.now() + round(again_s * _NS))
            if kind in (Kind.AGREE, Kind.MORE):
                self.guard.close(member.name)
            if kind is not Kind.UNREADABLE:
                self.guard.read(member.name, message.started_ns)
        if kind is Kind.AGREE and not late:
            session.margin.agreed(list(active), message.started_ns)
        if books.units is not None:
            held_now = sum(round(m.open_units()) for m in members)
            read = GroupRead(books.units, held_now, late)
            self._group_reads[(members[0].symbol, members[0].exchange)] = read
        return polls


def _shared_out(
    members: Sequence[Trader],
    units: dict[str, int],
    difference: int,
    trades: Sequence[Trade],
) -> tuple[Share, ...]:
    """The allocation of an explained `difference` (the broker's net less the members'
    sum) among the members whose kernel session runs, whatever members a read asked for:
    each share is adopted when that member's comparison confirms it."""
    holders = [(m.name, units[m.name], _opened_ns(m)) for m in members if not m.finished]
    return allocate(holders, difference, trades)


def _opened_ns(trader: Trader) -> int | None:
    """When `trader`'s current position opened, UTC nanoseconds: its oldest open
    tranche's opening; None when it holds none."""
    position = call(trader.engine.position, INSTRUMENT, LIVE_PRODUCT)
    tranches = () if position is None else position["tranches"]
    opened = [int(at) for _number, units, _price, at in tranches if units]
    return min(opened) if opened else None


__all__ = ["GroupComparison", "GroupRead"]

"""A comparison of an instrument several strategies of the session trade (18-multi-strategy
4.1, 4.2 and 4.6; unit M6, the part the session needs before the group rows of unit M7).

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
- **Positions**: the sum of the members' units against the broker's net (`judge`). Until
  the allocation of unit M7, a difference trades of other orders explain is judged as one
  they do not: nothing is adopted. Each member's finding is the group's kind with its own
  units and the broker's net, acted on by the rules of a leg (16-session-design 5.3): the
  entries locked and alerted; on the other side each member's own stop cancelled. While
  the broker holds fewer units than the sum, or none, or the other side, each member's
  closing orders are held to its share of what the broker holds (`share_of`), so they can
  never together close more than it holds.
- **Agreement** tells the margin holds (`_margin_reject`): a position a closing order's
  rejection held may be exited again.

Quantities are signed units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._reconcile_act import Finding, attribute, in_flight, order_findings, own_orders
from ._reconcile_findings import FILL_DURING_READ
from ._reconcile_rules import GUARDED, Kind, LegBooks, judge, leg_books, unexplained_by
from ._session_trader import Trader
from .inbox import Books

_NS = NANOS_PER_SECOND


def share_of(broker_units: int, group_units: int, member_units: int) -> int:
    """The broker's units a member's closing orders may be for while the group is guarded:
    what the broker holds on the group's side beyond the other members' units (none when
    the broker is flat or on the other side; the broker's own figure then, which leaves no
    room). Summed over the members it never exceeds what the broker holds."""
    if broker_units * group_units <= 0:
        return broker_units
    others = abs(group_units) - abs(member_units)
    share = max(0, abs(broker_units) - others)
    return share if broker_units > 0 else -share


def group_kind(group_units: int, broker_units: int | None, unexplained: int) -> Kind:
    """The group's position against the broker's (module docstring): `judge`'s kind of the
    sum, a difference other orders' trades explain taken as unexplained (nothing is
    adopted on a shared instrument before unit M7's allocation)."""
    kind = judge(group_units, broker_units, unexplained)
    if kind in (Kind.EXPLAINED_FLAT, Kind.EXPLAINED_FEWER):
        return Kind.UNEXPLAINED
    return kind


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
    adopted: Any
    _noted: Any
    _soon: Any

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
        kind = Kind.UNREADABLE
        if books.units is not None and books.trades is not None:
            external = sum(trade.signed for trade in books.external(everyone))
            adopted = sum(self.adopted(m.name) for m in members)
            unexplained = unexplained_by(group_units, books.units, external, adopted)
            kind = group_kind(group_units, books.units, unexplained)
        polls = False
        for member in active:
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
                share = share_of(books.units, group_units, units[member.name])
                self.guard.open(member.name, share)
            if self._findings.judged(member, findings, books, everyone, late=late):
                self._soon(session.now() + round(again_s * _NS))
            if kind in (Kind.AGREE, Kind.MORE):
                self.guard.close(member.name)
            if kind is not Kind.UNREADABLE:
                self.guard.read(member.name, message.started_ns)
        if kind is Kind.AGREE and not late:
            session.margin.agreed(list(active), message.started_ns)
        return polls


__all__ = ["GroupComparison", "group_kind", "share_of"]

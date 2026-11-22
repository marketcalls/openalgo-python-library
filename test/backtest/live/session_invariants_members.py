"""The session invariant I3 restated per member of an instrument group (18-multi-strategy
5.1; unit M8), checked over every session test with the others (`session_invariants`), at
every change of an order at the fake while a run is open (`OrderChange.members`). A
strategy alone on its instrument is a group of one, for which it is its leg's rule.

**I3 per member.** A member's working closing orders, its stop included, never hold more
units than its own kernel position at any change: the session's working orders of the
member's strategy on the instrument, on the side that closes the member's position, a
reversal (an entry or add against the position, `entries`) counting only the position it
closes; for a flat member, its working stops. The rule of the group (`within_the_position`,
the instrument's exits against the broker's net) cannot see one member's stop sized from
another member's units, or a flat member's stop standing on another member's position.

A member's kernel adopts another client's trade before its stop is cut at the broker
(18-multi-strategy 4.14b ruling: the adoption first, the cut after it in the same
correction). So the member's exits that were working when it adopted may stand beyond its
position by no more than the units it adopted, until its exits fit its position again;
an exit placed or raised after the adoption gets no such room.

Quantities are signed units.
"""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass, field

from session_invariants_changes import STOP_TYPES, Instrument
from session_trace import Member, OrderChange


@dataclass
class AdoptionRoom:
    """What one member's adoptions left beside its exits since they last fitted its
    position: the units it `adopted`, and the exits working when it adopted (`overtaken`),
    with the units each had not filled then."""

    adopted: int = 0
    overtaken: dict[str, int] = field(default_factory=dict)

    def room(self, exits: Iterable[OrderChange]) -> int:
        """Units the member's working `exits` may hold beyond its position: those of the
        overtaken ones (never more than each had when it adopted), up to `adopted`."""
        kept = sum(min(o.remaining, self.overtaken[o.order_id]) for o in exits if self.took(o))
        return min(self.adopted, kept)

    def took(self, order: OrderChange) -> bool:
        """Whether `order` was working when the member adopted."""
        return order.order_id in self.overtaken


def members_within(
    changes: Iterable[OrderChange], mine: frozenset[str], entries: Set[str] = frozenset()
) -> None:
    """I3 per member (module docstring) over `changes` (in order), `mine` the session's
    clients and `entries` the broker order IDs of the session's entries and adds."""
    orders: dict[str, OrderChange] = {}
    adopted_seen: dict[tuple[Instrument, str], int] = {}
    rooms: dict[tuple[Instrument, str], AdoptionRoom] = {}
    for change in changes:
        orders[change.order_id] = change
        for member in change.members or ():
            key = (change.instrument, member.strategy)
            own = [
                o for o in orders.values()
                if o.client in mine and o.working and o.strategy == member.strategy
                and o.instrument == change.instrument
            ]  # fmt: skip
            exits, closing = closing_orders(own, member.units, entries)
            room = rooms.setdefault(key, AdoptionRoom())
            newly = abs(member.adopted - adopted_seen.get(key, 0))
            adopted_seen[key] = member.adopted
            if newly:
                room.adopted += newly
                for order in exits:
                    room.overtaken.setdefault(order.order_id, order.remaining)
            if closing <= abs(member.units):
                rooms[key] = AdoptionRoom()
                continue
            assert closing <= abs(member.units) + room.room(exits), _broken(change, member, own)


def closing_orders(
    own: list[OrderChange], units: int, entries: Set[str]
) -> tuple[list[OrderChange], int]:
    """A member's working orders `own` that close its position of `units` (module
    docstring), and the units they hold."""
    if units == 0:
        stops = [o for o in own if o.price_type in STOP_TYPES]
        return stops, sum(o.remaining for o in stops)
    closing = "SELL" if units > 0 else "BUY"
    against = [o for o in own if o.action == closing]
    exits = [o for o in against if o.order_id not in entries]
    reversing = sum(min(o.remaining, abs(units)) for o in against if o.order_id in entries)
    return exits, sum(o.remaining for o in exits) + reversing


def _broken(change: OrderChange, member: Member, own: list[OrderChange]) -> str:
    working = [(o.order_id, o.action, o.price_type, o.remaining) for o in own]
    return (
        f"18-multi-strategy I3: {member.strategy}'s working closing orders {working} hold "
        f"more than its {member.units} units of {change.symbol} after {change}"
    )


__all__ = ["AdoptionRoom", "closing_orders", "members_within"]

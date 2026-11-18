"""The session invariants checked at every change of an order at the fake OpenAlgo, as it
happened (`session_invariants` 1 to 3; 16-session-design 7.1 "at any instant", unit 14).

1. **One stop.** At most one stop of the session's (SL or SL-M) works per leg (book,
   strategy, instrument).
2. **Exits within the position.** At every change, the session's working orders on the
   closing side of an instrument (the side opposite its first order there) hold no more
   units than the position on the open side. Another client's sale by hand can overtake
   the session's exits: the session learns of it only by its comparisons, and 5.3 fits
   its stop on the second that shows the sale. So the exits that were working when the
   sale filled may stand beyond the position by as much as the sale left them beyond it,
   never more than the units sold (`SaleRoom`), until the session's exits fit the position
   again; an exit placed after the sale, or the units an exit is raised by, get no such
   room (unit 14 round 2, finding 2).
3. **Never the other side.** No fill of the session's orders leaves the instrument's
   position on the other side of the leg, unless another client's fills had taken it
   there already and the session's fill moved it towards flat, or the test's decided end
   lets a stop that a sale by hand overtook fill through zero by no more than the sale
   left it beyond the position (`SessionTrace.oversold_by_decision`; unit 14 round 2,
   finding 1).

Quantities are signed units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from session_trace import OrderChange

from openalgo.backtest._live._types import Mode

#: The price types of a stop at the broker.
STOP_TYPES = frozenset({"SL", "SL-M"})

Instrument = tuple[Mode, str, str]


@dataclass
class SaleRoom:
    """What other clients' sales by hand of one instrument left beside the session's exits
    since those last fitted the position: the units `sold`; the units the session's exits
    stood beyond the position as the latest sale filled, no more than those sold
    (`beyond`); and the session's exits that were working when a sale filled
    (`overtaken`), with the units each had not filled then."""

    sold: int = 0
    beyond: int = 0
    overtaken: dict[str, int] = field(default_factory=dict)

    def room(self, exits: Iterable[OrderChange]) -> int:
        """Units the session's working `exits` may hold beyond the position: those of the
        overtaken ones (never more than each had when the sale came), up to `beyond`."""
        kept = sum(min(o.remaining, self.overtaken[o.order_id]) for o in exits if self.took(o))
        return min(self.beyond, kept)

    def took(self, order: OrderChange) -> bool:
        """Whether `order` was working when a sale by hand filled."""
        return order.order_id in self.overtaken


def sides_of(own: list[OrderChange]) -> dict[Instrument, str]:
    """The open side of each instrument the session traded: its first order's action."""
    sides: dict[Instrument, str] = {}
    for change in own:
        sides.setdefault(change.instrument, change.action)
    return sides


def one_stop_per_leg(changes: list[OrderChange], mine: frozenset[str]) -> None:
    """1 (module docstring), at each change of a stop of the session's."""
    orders: dict[str, OrderChange] = {}
    for change in changes:
        orders[change.order_id] = change
        if change.client not in mine or change.price_type not in STOP_TYPES:
            continue
        leg = (change.mode, change.strategy, change.symbol, change.exchange)
        stops = [
            o.order_id
            for o in orders.values()
            if o.client in mine and o.working and o.price_type in STOP_TYPES
            and (o.mode, o.strategy, o.symbol, o.exchange) == leg
        ]  # fmt: skip
        assert len(stops) <= 1, f"7.1: two stops of {leg} work at once: {stops} ({change})"


def within_the_position(
    changes: list[OrderChange],
    mine: frozenset[str],
    sides: Mapping[Instrument, str],
    oversold: Mapping[str, str] | None = None,
    *,
    within: bool = True,
    side: bool = True,
    used: set[str] | None = None,
) -> dict[Instrument, SaleRoom]:
    """2 and 3 (module docstring), `within` and `side` saying which is asserted, and
    `oversold` the strategies whose decided end lets an overtaken stop fill through zero
    (each one whose end a fill took is added to `used`, when given). Returns, for each
    instrument whose session exits a sale by hand left beyond the position at the end,
    its `SaleRoom`."""
    orders: dict[str, OrderChange] = {}
    rooms: dict[Instrument, SaleRoom] = {}
    for change in changes:
        orders[change.order_id] = change
        instrument, opens = change.instrument, sides.get(change.instrument)
        if opens is None:
            continue
        held = change.position if opens == "BUY" else -change.position
        room = rooms.setdefault(instrument, SaleRoom())
        exits = [
            o for o in orders.values()
            if o.client in mine and o.working and o.instrument == instrument and o.action != opens
        ]  # fmt: skip
        closing = sum(o.remaining for o in exits)
        if change.client in mine:
            allowed = side and _never_the_other_side(change, opens, held, room, oversold or {})
            if allowed and used is not None:
                used.add(change.strategy)
        elif change.fill_units and change.action != opens:
            room.sold += change.fill_units
            room.beyond = min(room.sold, max(closing - max(held, 0), 0))
            for order in exits:
                room.overtaken.setdefault(order.order_id, order.remaining)
        if closing <= max(held, 0):
            rooms[instrument] = SaleRoom()
            continue
        assert not within or closing <= max(held, 0) + room.room(exits), (
            f"7.1: the session's working exits of {instrument} hold {closing} units "
            f"against a position of {change.position} ({change})"
        )
    return {instrument: room for instrument, room in rooms.items() if room.beyond}


def _never_the_other_side(
    change: OrderChange, opens: str, held: int, room: SaleRoom, oversold: Mapping[str, str]
) -> bool:
    """3 (module docstring) for a change of the session's order: `held` is the position on
    the open side after it, `room` what sales by hand left before it. True when the test's
    decided end (`oversold`) is what allows it."""
    if not change.fill_units or held >= 0:
        return False
    moved = change.fill_units if change.action == opens else -change.fill_units
    if held - moved < 0 and moved > 0:
        return False
    overtaken = change.strategy in oversold and room.took(change) and -held <= room.beyond
    assert overtaken, (
        f"7.1: a fill of the session's order {change.order_id} left {change.instrument} at "
        f"{change.position}, the other side of the leg ({change})"
    )
    return True


__all__ = [
    "STOP_TYPES",
    "Instrument",
    "SaleRoom",
    "one_stop_per_leg",
    "sides_of",
    "within_the_position",
]

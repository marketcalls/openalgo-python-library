"""The instrument groups of a live session (18-multi-strategy 4.1 and 4.13; unit M6): the
strategies that trade one instrument and product, which a broker nets into one position,
and the two entry gates the session sets on each member of a group of two or more.

Every instrument the session trades has a group (`InstrumentGroup`), its members the
strategies trading it in call order; most have one member. Each member keeps its own
engine, position, orders and broker-held stop; what belongs to the broker's net position
(the comparison with the books, the exit guard) is the group's.

**The gates** (`Groups.before_decisions`, run before each member's decisions in the
loop's step C, so a member deciding after another in the same step sees the other's new
order). A group of one sets neither, so its member trades as a strategy alone does.

- *Resting orders* (`set_shared`): a strategy's own LIMIT, SL and SL-M orders, entry or
  exit, are refused at submission (`shared_resting`); set once, before the member's first
  decision. Targets, the leg's stop and the engine's exits are not strategy orders.
- *The other side* (`set_group_side`): the side the other members hold or are entering,
  so an entry, add or reversal that would put the member on the other side is refused at
  submission (`group_other_side`). A member enters the side each working order that opens
  or adds opens (an entry of a flat member, an add, a reversal: an entry against its
  position, whose role is "entry"); its resulting side is that one, else the side it
  holds. While another member's working order would take it to the other side (a
  reversal not yet filled), or the others' resulting sides differ, every entry of the
  member is refused (the entry lock `GROUP_REVERSING_LOCK`): an entry on either side
  would stand against one of them at some instant (18-multi-strategy 4.14a item 1).

Exits are never gated. Quantities are signed units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .._engine import call
from ..errors import LiveError
from ._order_state import ENTRY_PURPOSES
from ._plan_catch_up import INSTRUMENT
from ._session_trader import Trader
from ._venue_requests import LIVE_PRODUCT

#: The sides `set_group_side` takes (the binding's `GROUP_SIDES`).
LONG = "long"
SHORT = "short"
#: The roles of the kernel's orders that open or add (the binding's `ORDER_ROLES`): an
#: entry (a reversal too) and an add.
OPENING_ROLES = frozenset({"entry", "add"})
#: The entry lock of a member while another member reverses, or the others' resulting
#: sides differ (module docstring).
GROUP_REVERSING_LOCK = "group_reversing"


@dataclass(frozen=True)
class InstrumentGroup:
    """One instrument and product the session trades: `symbol`, `exchange`, `product`,
    and `members`, the names of the strategies trading it, in call order."""

    symbol: str
    exchange: str
    product: str
    members: tuple[str, ...]

    @property
    def shared(self) -> bool:
        """True when several strategies of the session trade the instrument."""
        return len(self.members) > 1


def instrument_groups(strategies: Iterable[tuple[str, str, str]]) -> tuple[InstrumentGroup, ...]:
    """The groups of a session's strategies, each given as (name, symbol, exchange) in
    call order: one per instrument in the live product, ordered by its first member."""
    members: dict[tuple[str, str], list[str]] = {}
    for name, symbol, exchange in strategies:
        members.setdefault((symbol, exchange), []).append(name)
    return tuple(
        InstrumentGroup(symbol, exchange, LIVE_PRODUCT, tuple(names))
        for (symbol, exchange), names in members.items()
    )


@dataclass(frozen=True)
class Stand:
    """Where a member stands on its instrument: the side it holds ("long", "short" or
    None when flat) and the sides its working orders that open or add enter."""

    held: str | None
    entering: frozenset[str]

    @property
    def resulting(self) -> frozenset[str]:
        """The sides it will hold once its working opening orders fill: those, else the
        side it holds."""
        if self.entering:
            return self.entering
        return frozenset() if self.held is None else frozenset({self.held})

    @property
    def reversing(self) -> bool:
        """True while a working order would take it to the other side."""
        return self.held is not None and bool(self.entering - {self.held})


def stand_of(trader: Trader) -> Stand:
    """Where `trader` stands (`Stand`), from its kernel session: its position, and each
    working order that is not reduce-only and whose role opens or adds. An order whose
    command the venue map has taken carries its role there (as its intent's purpose);
    one decided since is read from the kernel's saved state."""
    units = trader.open_units()
    held = None if units == 0 else (LONG if units > 0 else SHORT)
    entering: set[str] = set()
    roles: dict[int, str] | None = None
    for kernel_order in trader.working():
        order = call(trader.engine.order, kernel_order)
        if order is None or order["reduce_only"]:
            continue
        opens = _taken_opens(trader, kernel_order)
        if opens is None:
            roles = _roles(trader) if roles is None else roles
            opens = roles.get(kernel_order) in OPENING_ROLES
        if opens:
            entering.add(LONG if order["side"] == "BUY" else SHORT)
    return Stand(held, frozenset(entering))


def _taken_opens(trader: Trader, kernel_order: int) -> bool | None:
    """Whether a kernel order the venue map has taken opens or adds; None for one it has
    not taken yet (decided in this loop step)."""
    try:
        intent = trader.venue.intent_of(kernel_order)
    # The venue map knows only the orders whose commands it has taken.
    except LiveError:
        return None
    return intent.purpose in ENTRY_PURPOSES


def _roles(trader: Trader) -> dict[int, str]:
    """The role of each working order of `trader`'s kernel session, by kernel order."""
    orders = call(trader.engine.live_state)["orders"]
    return {int(each["order"]): str(each["role"]) for each in orders if each.get("role")}


class Groups:
    """The instrument groups of a session's `traders` (call order), and the gates they
    set (module docstring); `event` writes the session's event stream (each lock and
    unlock of `GROUP_REVERSING_LOCK`)."""

    def __init__(self, traders: Sequence[Trader], *, event: Callable[..., None]) -> None:
        self._event = event
        self.groups = instrument_groups((t.name, t.symbol, t.exchange) for t in traders)
        self._members: dict[tuple[str, str], tuple[Trader, ...]] = {}
        for trader in traders:
            key = (trader.symbol, trader.exchange)
            self._members[key] = (*self._members.get(key, ()), trader)
        self._shared_set: set[str] = set()
        self._sides: dict[str, str | None] = {}
        self._reversing: set[str] = set()

    def of(self, symbol: str, exchange: str) -> tuple[Trader, ...]:
        """The members trading `exchange:symbol`, in call order (none for another)."""
        return self._members.get((symbol, exchange), ())

    def members(self, trader: Trader) -> tuple[Trader, ...]:
        """The members of `trader`'s group, `trader` among them, in call order."""
        return self.of(trader.symbol, trader.exchange)

    def shared(self, trader: Trader) -> bool:
        """True when `trader`'s instrument is traded by another strategy of the session."""
        return len(self.members(trader)) > 1

    def before_decisions(self, trader: Trader) -> None:
        """Set the gates of `trader` from where the other members stand now (module
        docstring); nothing for a member of a group of one or a finished kernel
        session."""
        if trader.finished or not self.shared(trader):
            return
        if trader.name not in self._shared_set:
            call(trader.engine.set_shared, INSTRUMENT, LIVE_PRODUCT, True)
            self._shared_set.add(trader.name)
        stands = [stand_of(other) for other in self.members(trader) if other is not trader]
        resulting = frozenset().union(*(stand.resulting for stand in stands))
        side = next(iter(resulting)) if len(resulting) == 1 else None
        if trader.name not in self._sides or self._sides[trader.name] != side:
            call(trader.engine.set_group_side, INSTRUMENT, LIVE_PRODUCT, side)
            self._sides[trader.name] = side
        reversing = len(resulting) > 1 or any(stand.reversing for stand in stands)
        if reversing and trader.name not in self._reversing:
            self._reversing.add(trader.name)
            if trader.lock(GROUP_REVERSING_LOCK):
                self._event("lock", strategy=trader.name, reason=GROUP_REVERSING_LOCK)
        elif not reversing and trader.name in self._reversing:
            self._reversing.discard(trader.name)
            if trader.unlock(GROUP_REVERSING_LOCK):
                self._event("unlock", strategy=trader.name, reason=GROUP_REVERSING_LOCK)


def shared_lines(groups: Iterable[InstrumentGroup]) -> list[str]:
    """The start banner's lines of the shared instruments (18-multi-strategy 4.11): each
    with its members, then the rule on trading by hand; none without one."""
    lines = []
    for group in groups:
        if not group.shared:
            continue
        names = ", ".join(group.members[:-1]) + f" and {group.members[-1]}"
        lines.append(
            f"{group.symbol} ({group.exchange}, {group.product}): {names} share one broker "
            "position; each holds its own stop; they trade one side at a time and with "
            "market orders only"
        )
    if lines:
        lines.append("Never trade a session's symbol by hand while it runs")
    return lines


__all__ = [
    "GROUP_REVERSING_LOCK",
    "LONG",
    "OPENING_ROLES",
    "SHORT",
    "Groups",
    "InstrumentGroup",
    "Stand",
    "instrument_groups",
    "shared_lines",
    "stand_of",
]

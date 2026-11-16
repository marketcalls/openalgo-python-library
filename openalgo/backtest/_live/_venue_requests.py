"""The order requests the loop hands the order worker, built from the kernel's command dicts
(16-session-design 2.7), with their priorities (2.4), and the verdicts a targeted read of
an order gives on a modify or a cancel whose answer was lost or refused (4.3, 4.5).

A command crosses the binding as a dict (rust/oa_py/src/backtest/engine/venue.rs):
`command` ("place", "modify", "cancel") and `order_id` (the kernel order); "place" adds
`instrument` (the registration index), `product`, `side`, `role` and `reason`; "place" and
"modify" add `quantity_units`, `order_type`, `price` (None for MARKET and SL-M) and
`trigger_price` (None for MARKET and LIMIT).

An intent ID is `"{day:%Y%m%d}-{mode}-{starts}-{strategy}-o{order_id}"`: the journal
session's IST day, mode ("sandbox" or "live") and start count, so a morning sandbox
rehearsal, a live session and every restart of a day each have their own, since kernel
order IDs start at o1 in every engine and the journal refuses an ID already in either
ledger. Quantities are whole units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import LiveError
from ._journal_reads import JournalSession
from ._order_check import OrderCheck
from ._order_state import ENTRY_PURPOSES
from ._types import OrderIntent, Purpose

#: What each kernel order role is for (the binding's `ORDER_ROLES`, R1).
ROLE_PURPOSES: Mapping[str, Purpose] = {
    "entry": Purpose.ENTRY,
    "add": Purpose.ADD,
    "leg_stop": Purpose.STOP,
    "exit": Purpose.EXIT,
    "risk_exit": Purpose.RISK_EXIT,
    "square_off": Purpose.SQUARE_OFF,
}
#: The binding's sides, as OpenAlgo's actions.
ACTIONS = frozenset({"BUY", "SELL"})
#: The product a live session trades in Tier 1 (CNC and NRML positions need GTT live).
LIVE_PRODUCT = "MIS"
#: The price types of a broker stop: a modify that keeps the units and one of these is a
#: trigger move (trailing), not a change of what protects the position.
STOP_TYPES = frozenset({"SL", "SL-M"})
#: The price types whose limit price is part of the order's terms.
LIMIT_TYPES = frozenset({"LIMIT", "SL"})
#: Rupees within which a row's price is the price the session sent (a broker writes a
#: price on the tick back with float noise only).
SAME_PRICE_RUPEES = 1e-6


class Priority(enum.IntEnum):
    """How urgent a request is to the order worker, lower first (16-session-design 2.4)."""

    #: Stops, exits and what protects a position: placements of stops and exits, a stop's
    #: units, a reprice to marketable, cancels of closing orders.
    PROTECT = 0
    #: A trailing stop's trigger, a change or cancel of a working entry.
    ADJUST = 1
    #: Entries and adds, which expire.
    ENTER = 2
    #: Releases of held fills, book reads, probes.
    UPKEEP = 3


@dataclass(frozen=True)
class OrderTerms:
    """An order's terms at the broker: whole units, the price type ("MARKET", "LIMIT",
    "SL", "SL-M"), the limit and the trigger in rupees (0.0 where they do not apply)."""

    quantity: int
    price_type: str
    price: float = 0.0
    trigger_price: float = 0.0


@dataclass(frozen=True)
class PlaceRequest:
    """Place kernel order `kernel_order` as `intent`, once."""

    request_id: int
    kernel_order: int
    intent: OrderIntent
    priority: Priority


@dataclass(frozen=True)
class ModifyRequest:
    """Change broker order `order_id` (kernel order `kernel_order`, placed as `intent`)
    from the terms it `replaces` to `terms`: a full replacement, as every kernel modify
    is, carrying the order's total units."""

    request_id: int
    kernel_order: int
    intent: OrderIntent
    order_id: str
    terms: OrderTerms
    replaces: OrderTerms
    priority: Priority

    def fields(self) -> dict[str, Any]:
        """The keyword arguments of `OrderPath.modify` (those of `modifyorder`)."""
        return {
            "symbol": self.intent.symbol,
            "action": self.intent.action,
            "exchange": self.intent.exchange,
            "product": self.intent.product,
            "price_type": self.terms.price_type,
            "quantity": self.terms.quantity,
            "price": self.terms.price,
            "trigger_price": self.terms.trigger_price,
        }


@dataclass(frozen=True)
class CancelRequest:
    """Cancel broker order `order_id` (kernel order `kernel_order`, placed as `intent`)."""

    request_id: int
    kernel_order: int
    intent: OrderIntent
    order_id: str
    priority: Priority


#: A request for one kernel order.
OrderRequest = PlaceRequest | ModifyRequest | CancelRequest


class Verdict(enum.Enum):
    """What a targeted read of an order (`OrderCheck`) says of a modify or a cancel."""

    #: A modify: the row shows its terms.
    CONFIRMED = "confirmed"
    #: A modify: the row shows the old terms. A cancel: the order works on.
    REFUSED = "refused"
    #: The order is terminal: its news decides.
    ENDED = "ended"
    #: The row cannot be read or shows neither: read it again.
    UNSETTLED = "unsettled"


def intent_id(session: JournalSession, strategy: str, kernel_order: int) -> str:
    """The intent ID of kernel order `kernel_order` of `strategy` in `session` (module
    docstring)."""
    return f"{session.day:%Y%m%d}-{session.mode.value}-{session.starts}-{strategy}-o{kernel_order}"


def intent_fields(
    command: Mapping[str, Any],
    *,
    strategy: str,
    markets: Sequence[tuple[str, str]],
    session: JournalSession,
    created_ns: int,
) -> OrderIntent:
    """The intent of a "place" command of `strategy`'s engine, whose instruments are
    `markets` ((symbol, exchange) in registration order), created at `created_ns`.

    Raises LiveError for a fraction of a unit (the kernel sizes in whole units in a live
    session), an unknown role, side or instrument, or a product other than MIS.
    """
    kernel_order = int(command["order_id"])
    role, side, product = command.get("role"), command.get("side"), command.get("product")
    index = command.get("instrument")
    if role not in ROLE_PURPOSES or side not in ACTIONS or product != LIVE_PRODUCT:
        raise LiveError(
            f"kernel order o{kernel_order} cannot be sent: a live session places "
            f"{sorted(ROLE_PURPOSES)} orders, BUY or SELL, in {LIVE_PRODUCT}, not role "
            f"{role!r}, side {side!r}, product {product!r}"
        )
    if not isinstance(index, int) or not 0 <= index < len(markets):
        raise LiveError(f"kernel order o{kernel_order} names instrument {index!r}, unregistered")
    symbol, exchange = markets[index]
    terms = terms_of(command)
    return OrderIntent(
        intent_id=intent_id(session, strategy, kernel_order),
        strategy=strategy,
        symbol=symbol,
        exchange=exchange,
        action=side,
        quantity=terms.quantity,
        price_type=terms.price_type,
        product=product,
        purpose=ROLE_PURPOSES[role],
        created_ns=created_ns,
        price=terms.price,
        trigger_price=terms.trigger_price,
    )


def terms_of(command: Mapping[str, Any]) -> OrderTerms:
    """The terms a "place" or "modify" command gives. Raises LiveError for a quantity that
    is not a whole number of units above 0."""
    units = command["quantity_units"]
    if not (math.isfinite(units) and units > 0 and float(units).is_integer()):
        raise LiveError(
            f"kernel order o{command['order_id']} asks for {units!r} units: a live session "
            "sends whole units only (venue quantity), so this is a defect of the kernel"
        )
    return OrderTerms(
        quantity=int(units),
        price_type=str(command["order_type"]),
        price=float(command["price"] or 0.0),
        trigger_price=float(command["trigger_price"] or 0.0),
    )


def place_priority(intent: OrderIntent) -> Priority:
    """An entry or add at ENTER; a stop or any exit at PROTECT."""
    return Priority.ENTER if intent.purpose in ENTRY_PURPOSES else Priority.PROTECT


def modify_priority(intent: OrderIntent, old: OrderTerms, new: OrderTerms) -> Priority:
    """A change of a working entry or add, and a closing order's trigger move that keeps
    its units and its stop type (trailing), at ADJUST; any other change of a closing
    order (its units, a reprice to marketable) at PROTECT."""
    if intent.purpose in ENTRY_PURPOSES:
        return Priority.ADJUST
    trailing = (
        old.quantity == new.quantity
        and old.price_type == new.price_type
        and new.price_type in STOP_TYPES
    )
    return Priority.ADJUST if trailing else Priority.PROTECT


def cancel_priority(intent: OrderIntent) -> Priority:
    """The cancel of an entry or add at ADJUST; of a stop or any exit at PROTECT (an exit
    waits for the cancels of its position's closing orders)."""
    return Priority.ADJUST if intent.purpose in ENTRY_PURPOSES else Priority.PROTECT


def modify_verdict(request: ModifyRequest, check: OrderCheck) -> Verdict:
    """What `check`, a read of the order after its modify was unanswered or refused, says
    of the modify: its terms (CONFIRMED), the old terms (REFUSED), a terminal order
    (ENDED), or neither, unread (UNSETTLED). A row's units, price type and trigger count,
    and its limit price for a LIMIT or SL order."""
    if check.status is None:
        return Verdict.UNSETTLED
    if check.status.terminal:
        return Verdict.ENDED
    if _shows(check, request.terms):
        return Verdict.CONFIRMED
    if _shows(check, request.replaces):
        return Verdict.REFUSED
    return Verdict.UNSETTLED


def cancel_verdict(check: OrderCheck) -> Verdict:
    """What `check`, a read of the order after its cancel was unanswered or refused, says:
    ENDED when terminal, REFUSED while it works on, UNSETTLED when unread."""
    if check.status is None:
        return Verdict.UNSETTLED
    return Verdict.ENDED if check.status.terminal else Verdict.REFUSED


def _shows(check: OrderCheck, terms: OrderTerms) -> bool:
    if check.quantity != terms.quantity or check.price_type != terms.price_type:
        return False
    if terms.price_type in STOP_TYPES and not _same(check.trigger_price, terms.trigger_price):
        return False
    return terms.price_type not in LIMIT_TYPES or _same(check.price, terms.price)


def _same(seen: float | None, sent: float) -> bool:
    return seen is not None and math.isclose(seen, sent, rel_tol=0.0, abs_tol=SAME_PRICE_RUPEES)


__all__ = [
    "ACTIONS",
    "LIMIT_TYPES",
    "LIVE_PRODUCT",
    "ROLE_PURPOSES",
    "SAME_PRICE_RUPEES",
    "STOP_TYPES",
    "CancelRequest",
    "ModifyRequest",
    "OrderRequest",
    "OrderTerms",
    "PlaceRequest",
    "Priority",
    "Verdict",
    "cancel_priority",
    "cancel_verdict",
    "intent_fields",
    "intent_id",
    "modify_priority",
    "modify_verdict",
    "place_priority",
    "terms_of",
]

"""The GTT records of the live journal (`_journal_gtt`; 20-positional-live 3): where a GTT
the session placed stands (`GttState`), its terms (`GttTerms`), its binding to the broker
(`GttBinding`) and the child order it placed when it fired (`GttChild`), each checked when
it is written and read back from its row.

A GTT here is a SINGLE GTT (20-positional-live P6): one trigger, one child order. Prices
are rupees, quantities whole units, times UTC nanoseconds.
"""

from __future__ import annotations

import enum
import sqlite3
from dataclasses import dataclass
from datetime import date

from ...gtt import GTT_PRODUCTS
from ..errors import JournalError
from ._journal_file import check_name
from ._journal_rows import finite_number, unreadable, whole_number

#: The sides of a GTT's child order, as OpenAlgo's actions.
GTT_SIDES = ("BUY", "SELL")


class GttState(enum.Enum):
    """Where a GTT the session placed stands, as the journal keeps it (20-positional-live
    3.1): the session writes each change, and the journal refuses one that would move a
    GTT on from an end (`terminal`)."""

    #: Written before the place call: the broker's answer is not known yet.
    REQUESTED = "requested"
    #: The place answered with the broker's trigger ID; no book read has shown it yet.
    PLACED = "placed"
    #: A book read showed it resting: its place, or its last modify, confirmed.
    ACTIVE = "active"
    #: A modify was sent and is not confirmed yet.
    MODIFYING = "modifying"
    #: A cancel was sent and is not confirmed yet.
    CANCELLING = "cancelling"
    #: It fired, or went from the book while the session did not cancel it: its child is
    #: not resolved yet (20-positional-live 3.2, 3.3).
    TRIGGERED = "triggered"
    #: It fired and is resolved: its child bound, or the position's fall applied as its fill.
    FIRED = "fired"
    #: The session cancelled it, its cancel confirmed.
    CANCELLED = "cancelled"
    #: It ended without firing outside the session: cancelled by hand, expired, or refused
    #: by the broker after its placement (protected again by a new GTT, 3.4).
    ENDED = "ended"
    #: The broker refused its placement, or the place never reached it.
    NOT_PLACED = "not_placed"

    @property
    def terminal(self) -> bool:
        """True once the GTT can change no more: fired, cancelled, ended or not placed."""
        return self in (GttState.FIRED, GttState.CANCELLED, GttState.ENDED, GttState.NOT_PLACED)


@dataclass(frozen=True)
class GttTerms:
    """A SINGLE GTT's terms: its child order's `side` ("BUY" or "SELL"), its trigger and
    its child's limit in rupees (the limit as the broker holds it, which brokers set their
    own way: 20-positional-live 3.6), and its whole units."""

    side: str
    trigger_price: float
    limit_price: float
    quantity: int

    def checked(self) -> GttTerms:
        """These terms, checked. Raises JournalError for a side other than BUY or SELL, a
        trigger that is not a finite price above 0, a limit that is not a finite price of
        0 or more, or units that are not a whole number above 0."""
        if self.side not in GTT_SIDES:
            raise JournalError(f"a GTT's side is BUY or SELL, not {self.side!r}")
        trigger = finite_number("trigger_price", self.trigger_price)
        limit = finite_number("limit_price", self.limit_price)
        quantity = whole_number("quantity", self.quantity)
        if trigger <= 0 or limit < 0 or quantity <= 0:
            raise JournalError(
                f"a GTT needs a trigger above 0 rupees, a limit of 0 rupees or more and "
                f"units above 0, not a trigger of {trigger}, a limit of {limit} and "
                f"{quantity} units"
            )
        return GttTerms(self.side, trigger, limit, quantity)


@dataclass(frozen=True)
class GttBinding:
    """A GTT the session placed (`_journal_gtt`): `seq`, the journal's number of the place
    request that made it (its row, written before the place call); the trading `day` it
    was recorded on; the `strategy` and the kernel's `gtt_id` of that day's session; the
    order worker's `request_id` of the place request; the broker's `trigger_id` (None until
    the place answered with one); its instrument and product; its `terms` as the broker
    holds them; its `state`; and when it was recorded and last changed."""

    seq: int
    day: date
    strategy: str
    gtt_id: int
    request_id: int
    trigger_id: str | None
    symbol: str
    exchange: str
    product: str
    terms: GttTerms
    state: GttState
    recorded_ns: int
    changed_ns: int


@dataclass(frozen=True)
class GttChild:
    """The child order a fired GTT placed (20-positional-live 3.2): the GTT's `seq`, the
    trading `day` it was bound on, the broker's `order_id` and the `kernel_order` the
    session follows it as, and when it was bound."""

    gtt_seq: int
    day: date
    order_id: str
    kernel_order: int
    bound_ns: int


def check_leg(strategy: str, symbol: str, exchange: str, product: str) -> None:
    """Raise JournalError for a blank strategy, symbol or exchange, or a product a GTT
    cannot hold (OpenAlgo takes CNC and NRML only: MIS is squared off the same day)."""
    for name, value in (("strategy", strategy), ("symbol", symbol), ("exchange", exchange)):
        check_name(name, value)
    if product not in GTT_PRODUCTS:
        raise JournalError(
            f"a GTT holds a position overnight, so its product is CNC or NRML, not {product!r}"
        )


def binding_from(row: sqlite3.Row) -> GttBinding:
    """The GTT a `gtts` row holds. Raises JournalError for one it cannot read."""
    try:
        terms = GttTerms(row["side"], row["trigger_price"], row["limit_price"], row["quantity"])
        return GttBinding(
            seq=row["seq"],
            day=date.fromisoformat(row["day"]),
            strategy=row["strategy"],
            gtt_id=row["gtt_id"],
            request_id=row["request_id"],
            trigger_id=row["trigger_id"],
            symbol=row["symbol"],
            exchange=row["exchange"],
            product=row["product"],
            terms=terms,
            state=GttState(row["state"]),
            recorded_ns=row["recorded_ns"],
            changed_ns=row["changed_ns"],
        )
    except ValueError as error:
        raise JournalError(unreadable("a GTT", error)) from error


def child_from(row: sqlite3.Row) -> GttChild:
    """The child a `gtt_children` row holds. Raises JournalError for one it cannot read."""
    try:
        day = date.fromisoformat(row["day"])
    except ValueError as error:
        raise JournalError(unreadable("a GTT's child", error)) from error
    return GttChild(row["gtt_seq"], day, row["order_id"], row["kernel_order"], row["bound_ns"])


__all__ = [
    "GTT_SIDES",
    "GttBinding",
    "GttChild",
    "GttState",
    "GttTerms",
    "binding_from",
    "check_leg",
    "child_from",
]

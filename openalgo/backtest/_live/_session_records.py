"""What a live session keeps for its result as it runs (16-session-design 1.4; unit 13): every
fill it gave the kernel, every entry lock of each strategy, the round trips each strategy
closed, and, at the close, every order its venue maps placed.

- **Fills** (`FillRecord`): each increment a fill ledger gave the kernel (the stream's, a
  poll's, a released held fill's, the trade book's at a reconciliation or a restart) and
  each fill of other orders the reconciler adopted (`source` "external"), with whether
  its price was released unconfirmed (`estimated`) or may lie on another tick
  (`derived_from_average`).
- **Locks** (`LockRecord`): each reason a strategy's entries were locked for, when, and
  when it was lifted (None while it stands). The kernel's own halts (its daily loss
  limit, its drawdown halt) are its trading state, in each strategy's result.
- **Round trips** (`StrategyRecords.closed_trades`): counted as each fill takes a
  position to flat or through it, so that the count saved with a strategy's state is the
  count that state holds (`_session_figures`).
- **Orders** (`OrderRecord`, `order_records`): every order the venue maps placed, with
  how far it got at the broker (the stops supervisor's word: handed, held, unanswered,
  ambiguous, bound, confirmed, ended, or never handed) and the kernel's final status.

Times are UTC nanoseconds, prices rupees, quantities units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: How far an order got when the session never handed it to the order worker (an exit
#: the exit guard held, a request after a guard lock).
NOT_HANDED = "not handed"
#: The purpose and source of a fill of other orders the reconciler adopted.
EXTERNAL = "external"


@dataclass(frozen=True)
class FillRecord:
    """A fill the kernel took: when (the word's receive time), the strategy, the broker's
    order ID ("" for an adopted fill of other orders), BUY or SELL, whole units, the
    price, the order's purpose ("external" for an adopted fill), where the word came from
    (`source`), and the two flags of the module docstring."""

    at_ns: int
    strategy: str
    symbol: str
    exchange: str
    order_id: str
    action: str
    units: int
    price: float
    purpose: str
    source: str
    estimated: bool = False
    derived_from_average: bool = False


def adopted_fill(trader: Any, action: str, units: int, price: float, at_ns: int) -> FillRecord:
    """The record of `units` of other orders' trades on `action`'s side at `price` that
    the reconciler adopted into `trader`'s leg at `at_ns` (`EXTERNAL`)."""
    return FillRecord(
        at_ns=at_ns,
        strategy=trader.name,
        symbol=trader.symbol,
        exchange=trader.exchange,
        order_id="",
        action=action,
        units=units,
        price=price,
        purpose=EXTERNAL,
        source=EXTERNAL,
    )


@dataclass
class LockRecord:
    """An entry lock of strategy `strategy`: its reason, when it was set and lifted (None
    while it stands)."""

    strategy: str
    reason: str
    set_ns: int
    lifted_ns: int | None = None


@dataclass
class StrategyRecords:
    """One strategy's records (module docstring): its entry locks, oldest first, and the
    round trips it closed in this start."""

    strategy: str
    locks: list[LockRecord] = field(default_factory=list)
    closed_trades: int = 0

    def locked(self, reason: str, at_ns: int) -> None:
        """Its entries were locked for `reason` at `at_ns`."""
        self.locks.append(LockRecord(self.strategy, reason, at_ns))

    def unlocked(self, reason: str, at_ns: int) -> None:
        """The lock of `reason` was lifted at `at_ns`."""
        for record in reversed(self.locks):
            if record.reason == reason and record.lifted_ns is None:
                record.lifted_ns = at_ns
                return

    def filled(self, before_units: float, after_units: float) -> None:
        """A fill took the position from `before_units` to `after_units` (signed): a round
        trip closed when it went to flat or through it."""
        if before_units and (not after_units or (after_units > 0) != (before_units > 0)):
            self.closed_trades += 1


@dataclass(frozen=True)
class OrderRecord:
    """An order a venue map placed: the strategy, the kernel's order number, the intent
    (its ID, purpose, symbol, exchange, action, price type, units, limit and trigger in
    rupees, 0 for none, and when it was made), the broker's order ID ("" when never
    bound), how far it got at the broker (`reached`, module docstring), and the kernel's
    final status, units filled, average fill price (None for none) and reject code (None
    for none)."""

    strategy: str
    kernel_order: int
    intent_id: str
    order_id: str
    purpose: str
    symbol: str
    exchange: str
    action: str
    price_type: str
    quantity: int
    price: float
    trigger_price: float
    created_ns: int
    reached: str
    status: str
    filled_units: float
    average_price: float | None
    reject_code: str | None


def order_records(session: Any, outputs: Mapping[str, Any]) -> tuple[OrderRecord, ...]:
    """Every order `session`'s venue maps placed, by strategy in call order and then by
    kernel order (module docstring); `outputs` are the kernel sessions' `finish()` tables
    by strategy name."""
    records: list[OrderRecord] = []
    for trader in session.traders:
        final = _final_orders(outputs[trader.name]["orders"])
        watched = session.stops.leg(trader).orders
        for kernel_order, placed in trader.venue.placed():
            intent, row = placed.intent, final.get(kernel_order, {})
            seen = watched.get(kernel_order)
            records.append(
                OrderRecord(
                    strategy=trader.name,
                    kernel_order=kernel_order,
                    intent_id=intent.intent_id,
                    order_id=placed.order_id,
                    purpose=intent.purpose.value,
                    symbol=intent.symbol,
                    exchange=intent.exchange,
                    action=intent.action,
                    price_type=intent.price_type,
                    quantity=intent.quantity,
                    price=intent.price,
                    trigger_price=intent.trigger_price,
                    created_ns=intent.created_ns,
                    reached=NOT_HANDED if seen is None else seen.state.value,
                    status=str(row.get("status", "")),
                    filled_units=float(row.get("filled_units", 0.0)),
                    average_price=row.get("average_fill_rupees"),
                    reject_code=row.get("reject_code"),
                )
            )
    return tuple(records)


def _final_orders(columns: Mapping[str, Sequence[Any]]) -> dict[int, dict[str, Any]]:
    """The kernel's orders table (a list per column) as one dict per order, by order id."""
    names = list(columns)
    rows = zip(*(columns[name] for name in names), strict=True)
    return {int(row["order_id"]): row for row in (dict(zip(names, r, strict=True)) for r in rows)}


__all__ = [
    "EXTERNAL",
    "NOT_HANDED",
    "FillRecord",
    "LockRecord",
    "OrderRecord",
    "StrategyRecords",
    "adopted_fill",
    "order_records",
]

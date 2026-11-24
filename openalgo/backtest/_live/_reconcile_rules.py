"""The rules of reconciliation (16-session-design 5.2, 5.3, 5.5; decided 2026-11-16): what
the broker's books say of one leg, how a difference with the engine is judged, and what
each judgement allows. Pure: the reconciler (`reconcile`) and the restart
(`_session_restore`) act on what these give.

**A leg in the books** (`LegBooks`, `leg_books`): the broker's units of the instrument in
the leg's product (`_broker_units`: the position book, and for CNC the holdings too), its
trades and its order-book rows in that product (the trade book, the order book); the
account's available cash in the funds (`available_cash`).

**A position difference** (`judge`, 5.3). The engine's signed units are compared, after
its own orders' trades are attributed to them (5.2 step 1), with the broker's. A
difference is **explained** when the trades of other orders on the instrument (a manual
order, the broker's own square-off), less what the engine has adopted of them already,
come to it exactly; else it is **unexplained**. Two comparisons in a row showing one lock
the strategy's entries, alert, and allow only this (`RULES`):

| Engine | Broker | Kind | Allowed |
|---|---|---|---|
| N | the same | AGREE | nothing to do |
| N | flat, explained | EXPLAINED_FLAT | adopt flat; the kernel cancels the stop with it |
| N | same side, fewer, explained | EXPLAINED_FEWER | adopt; the stop cut (a held cancel goes) |
| N | flat or fewer, unexplained | UNEXPLAINED | nothing: a lag of the broker's books |
| N or flat | same side, more | MORE | nothing: the excess is not the engine's |
| N | the other side | OTHER_SIDE | cancel the leg's own working stop; close by hand |

On an instrument several strategies of the session share, an explained difference of the
members' sum is EXPLAINED_SHARED: each member adopts its share of it
(`_reconcile_allocate`, 18-multi-strategy 4.7).

Adopting is `apply_external_fill` of the difference at the other orders' price. From the
first comparison that shows the broker holding fewer units on the engine's side,
or none, or the other side, until one agrees, no closing order of the leg is sent for
more than the broker holds on that side (`exit_room`). Nothing ever trades to make the
broker match the engine, and no rule exits a position or fires the kill switch.

**At a restart** (`restart_kind`, 5.5) the journal position stands for the engine's: a
match resumes the leg; a broker position on a traded symbol with none in the journal, or a
journal position the broker does not hold and nothing explains, refuses to resume; the
rest are judged as above, with entries locked from the start.

Quantities are signed units (long above 0), prices rupees.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..errors import LiveError
from ._broker_units import broker_units
from ._order_book import book_rows
from ._order_updates import status_of
from ._types import OrderStatus


class Kind(enum.Enum):
    """How the broker's position of a leg stands against the engine's (module docstring)."""

    AGREE = "agree"
    EXPLAINED_FLAT = "explained_flat"
    EXPLAINED_FEWER = "explained_fewer"
    UNEXPLAINED = "unexplained"
    MORE = "more"
    OTHER_SIDE = "other_side"
    #: A shared instrument's broker holds fewer units than its members' sum, or none, and
    #: trades of other orders explain it: each member adopts its share (`_reconcile_allocate`).
    EXPLAINED_SHARED = "explained_shared"
    #: The position book could not be read: nothing is judged.
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class Rule:
    """What a difference allows once two comparisons in a row show it: adopt the broker's
    units through `apply_external_fill`, cancel the leg's own working stop, and the words
    of its RECONCILE_MISMATCH alert (what is done, what to do). A change of the stop is
    named as asked for: it goes to the order worker, which may close before sending it
    (a final read whose books came as its seconds ran out). Every difference locks the
    strategy's entries for the rest of the session ("reconciliation")."""

    adopt: bool
    cancel_stop: bool
    words: str


#: The rules of 5.3 (module docstring), decided 2026-11-16: only the safe corrections.
RULES: Mapping[Kind, Rule] = MappingProxyType(
    {
        Kind.EXPLAINED_FLAT: Rule(
            True,
            False,
            "trades of other orders closed it at the broker: the engine takes them as its "
            "own and asks for a stop of the leg to be cancelled with them",
        ),
        Kind.EXPLAINED_FEWER: Rule(
            True,
            False,
            "trades of other orders reduced it at the broker: the engine takes them as its "
            "own and asks for a stop of the leg to be cut down to what is left",
        ),
        Kind.UNEXPLAINED: Rule(
            False,
            False,
            "no trade explains it, so it is taken as a lag of the broker's books: nothing is "
            "adopted, and no exit is sent for more than the broker holds. Check the position "
            "in OpenAlgo",
        ),
        Kind.MORE: Rule(
            False,
            False,
            "the excess is not the engine's: nothing is sent for it and it has no stop at the "
            "broker. Close it by hand in OpenAlgo",
        ),
        Kind.OTHER_SIDE: Rule(
            False,
            True,
            "the broker holds the other side: the engine asks for its stop to be cancelled "
            "and sends nothing else. Close the position by hand in OpenAlgo",
        ),
    }
)
#: The words of an explained fewer difference whose adoption sent the cancel of the leg's
#: stop that the exit guard held (16-session-design 3.3 step 5 "As built"): an exit asked
#: for that cancel, so the stop's cancel is asked for rather than its cut.
FEWER_STOP_CANCELLED = (
    "trades of other orders reduced it at the broker: the engine takes them as its own and "
    "asks for the stop of the leg to be cancelled (an exit had asked for its cancel, held "
    "back until the books agreed)"
)
#: The kinds during which no closing order goes beyond what the broker holds.
GUARDED = frozenset(
    {
        Kind.EXPLAINED_FLAT,
        Kind.EXPLAINED_FEWER,
        Kind.UNEXPLAINED,
        Kind.OTHER_SIDE,
        Kind.EXPLAINED_SHARED,
    }
)


class RestartKind(enum.Enum):
    """What a restart does with a leg (16-session-design 5.5)."""

    #: The journal position matches the broker's: the leg resumes.
    RESUME = "resume"
    #: The broker holds a position the journal does not: the restart is refused.
    UNKNOWN_POSITION = "unknown_position"
    #: The journal holds a position the broker does not, and nothing explains it: refused.
    UNEXPLAINED_FLAT = "unexplained_flat"
    #: Any other difference: judged as mid-session, with entries locked from the start.
    DIFFERS = "differs"


@dataclass(frozen=True)
class Trade:
    """One trade-book row: the broker's order ID, the side, whole units and the price."""

    order_id: str
    side: str
    units: int
    price: float

    @property
    def signed(self) -> int:
        """The units the trade adds to a position (a sale takes them away)."""
        return self.units if self.side == "BUY" else -self.units


@dataclass(frozen=True)
class Row:
    """One order-book row: the order ID, its status, whole units, price type, trigger in
    rupees, and the strategy it names ("" when the book names none: a broker's)."""

    order_id: str
    status: OrderStatus
    quantity: int
    price_type: str
    trigger_price: float
    strategy: str

    @property
    def working(self) -> bool:
        """True while the order can still fill."""
        return not self.status.terminal


@dataclass(frozen=True)
class LegBooks:
    """What the books say of one instrument in a leg's product: the broker's net `units`
    (None when a book they are read from cannot be read), its `trades` and `rows` (None
    when their book cannot be read)."""

    units: int | None
    trades: tuple[Trade, ...] | None
    rows: tuple[Row, ...] | None

    def fills(self, order_id: str) -> tuple[int, float]:
        """An order's units filled in the trade book and their average price (0, 0.0 for
        none)."""
        own = [trade for trade in self.trades or () if trade.order_id == order_id]
        units = sum(trade.units for trade in own)
        value = sum(trade.units * trade.price for trade in own)
        return units, (value / units if units else 0.0)

    def external(self, own: frozenset[str]) -> tuple[Trade, ...]:
        """The trades of orders that are not in `own` (the session's orders by ID)."""
        return tuple(trade for trade in self.trades or () if trade.order_id not in own)

    def row(self, order_id: str) -> Row | None:
        """The order-book row of an order, or None."""
        return next((row for row in self.rows or () if row.order_id == order_id), None)


def leg_books(
    positionbook: Mapping[str, Any],
    tradebook: Mapping[str, Any],
    orderbook: Mapping[str, Any],
    symbol: str,
    exchange: str,
    *,
    product: str,
    holdings: Mapping[str, Any] | None = None,
) -> LegBooks:
    """The books' answers (as the SDK returns them) read for `exchange:symbol` in
    `product`, with the holdings' answer when the session reads them (None when it does
    not: a CNC leg's units are then unknown). A row that cannot be read makes its whole
    book unreadable: a guess is never made. Raises LiveError for a product that is not
    MIS, CNC or NRML."""
    return LegBooks(
        broker_units(positionbook, holdings, symbol, exchange, product),
        _trades(tradebook, symbol, exchange, product),
        _rows(orderbook, symbol, exchange, product),
    )


def judge(engine_units: int, broker_units: int | None, unexplained_by: int) -> Kind:
    """How the broker's position stands against the engine's (module docstring);
    `unexplained_by` is the broker's units less the engine's that the trades of other
    orders, less what the engine adopted of them, do not account for."""
    if broker_units is None:
        return Kind.UNREADABLE
    if broker_units == engine_units:
        return Kind.AGREE
    if engine_units * broker_units < 0:
        return Kind.OTHER_SIDE
    if abs(broker_units) > abs(engine_units):
        return Kind.MORE
    if unexplained_by != 0:
        return Kind.UNEXPLAINED
    return Kind.EXPLAINED_FLAT if broker_units == 0 else Kind.EXPLAINED_FEWER


def unexplained_by(engine_units: int, broker_units: int, external: int, adopted: int) -> int:
    """Units of the broker's difference from the engine that the trades of other orders
    (`external`, signed) do not account for once `adopted` (signed) are taken out."""
    return (broker_units - engine_units) - (external - adopted)


def available_cash(funds: Mapping[str, Any]) -> float | None:
    """Rupees of available cash in a funds answer (as the SDK returns it), or None when it
    is not a success or the figure is not a finite number."""
    data = funds.get("data") if funds.get("status") == "success" else None
    value = data.get("availablecash") if isinstance(data, Mapping) else None
    try:
        cash = float(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None
    return cash if cash is not None and math.isfinite(cash) else None


def adoption_price(external: Sequence[Trade], units: int, side: str) -> float:
    """The price the engine adopts `units` on `side` at: the trades of other orders on that
    side, newest first, up to `units`, averaged by units. Raises ValueError for none."""
    taken, value = 0, 0.0
    for trade in reversed([trade for trade in external if trade.side == side]):
        part = min(trade.units, units - taken)
        taken, value = taken + part, value + part * trade.price
        if taken == units:
            break
    if taken == 0:
        raise ValueError(f"no trade on the {side} side to adopt {units} units at")
    return value / taken


def exit_room(engine_units: int, broker_units: int) -> int:
    """Units a closing order of the engine's position may be for: what the broker holds on
    the engine's side (none when it is flat or on the other side)."""
    if engine_units * broker_units <= 0:
        return 0
    return min(abs(engine_units), abs(broker_units))


def restart_kind(journal_units: int, broker_units: int, unexplained: int) -> RestartKind:
    """What a restart does with a leg (module docstring): `journal_units` the journal
    position, `broker_units` the broker's, `unexplained` what the trades of other orders do
    not account for (`unexplained_by`)."""
    if journal_units == broker_units:
        return RestartKind.RESUME
    if journal_units == 0:
        return RestartKind.UNKNOWN_POSITION
    if broker_units == 0 and unexplained != 0:
        return RestartKind.UNEXPLAINED_FLAT
    return RestartKind.DIFFERS


def _trades(
    answer: Mapping[str, Any], symbol: str, exchange: str, product: str
) -> tuple[Trade, ...] | None:
    data = answer.get("data") if answer.get("status") == "success" else None
    if not isinstance(data, list):
        return None
    trades = []
    for row in data:
        if not isinstance(row, Mapping):
            return None
        if (row.get("symbol"), row.get("exchange"), row.get("product")) != (
            symbol,
            exchange,
            product,
        ):
            continue
        try:
            units, price = float(row["quantity"]), float(row["average_price"])
        except (KeyError, TypeError, ValueError):
            return None
        side = str(row.get("action", "")).upper()
        if side not in ("BUY", "SELL") or not units.is_integer() or units <= 0 or price <= 0:
            return None
        trades.append(Trade(str(row.get("orderid") or ""), side, int(units), price))
    return tuple(trades)


def _rows(
    answer: Mapping[str, Any], symbol: str, exchange: str, product: str
) -> tuple[Row, ...] | None:
    """The order-book rows of the leg; a row that names no product is taken as the leg's
    (a book that leaves the product out shows the leg's orders all the same)."""
    if answer.get("status") != "success":
        return None
    try:
        rows = book_rows(answer)
    except LiveError:
        return None
    read = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if (row.get("symbol"), row.get("exchange")) != (symbol, exchange):
            continue
        if row.get("product", product) != product:
            continue
        order_id = str(row.get("orderid") or "")
        try:
            status = status_of(row, order_id)
            quantity = float(row["quantity"])
            trigger = float(row.get("trigger_price") or 0.0)
        except (LiveError, KeyError, TypeError, ValueError):
            return None
        price_type = str(row.get("pricetype") or row.get("price_type") or "")
        strategy = str(row.get("strategy") or "")
        read.append(Row(order_id, status, int(quantity), price_type, trigger, strategy))
    return tuple(read)


__all__ = [
    "FEWER_STOP_CANCELLED",
    "GUARDED",
    "RULES",
    "Kind",
    "LegBooks",
    "RestartKind",
    "Row",
    "Rule",
    "Trade",
    "adoption_price",
    "available_cash",
    "exit_room",
    "judge",
    "leg_books",
    "restart_kind",
    "unexplained_by",
]

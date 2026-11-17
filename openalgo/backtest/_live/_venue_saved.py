"""The venue map's part for a restart (16-session-design 2.5 E, 3.6): the record of each
kernel order it placed (`VenueOrder`), the saved form journal L keeps of those the kernel
still works, and the map rebuilt from the kernel's (saved id, new id) pairs alone
(`RestoredOrders`, a part of `venue_map.VenueMap`): a key the earlier start kept for an
order that has since ended names another order now.

A record is plain data that survives a JSON round trip: the kernel order's saved id
(`order`), its intent (every `OrderIntent` field, the purpose by its value), the terms the
broker holds as far as the kernel knows, the broker's order ID once bound ("" before),
the fills its ledger gave the kernel (`filled_units`, `filled_value`), the modify waiting
for a result (its terms and the terms it replaces, or None), whether a cancel waits for
its result, and the kernel's modify and cancel commands that waited for the bind. A
record that cannot be read raises LiveError naming the field: a restart never guesses.

Quantities are whole units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import LiveError
from ._fill_ledger import FillLedger
from ._types import OrderIntent, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, OrderRequest, OrderTerms

#: The intent fields a record keeps, in `OrderIntent`'s order.
_INTENT_FIELDS = tuple(field.name for field in dataclasses.fields(OrderIntent))
_TERMS_FIELDS = tuple(field.name for field in dataclasses.fields(OrderTerms))


@dataclass
class VenueOrder:
    """A kernel order the venue map placed: its intent, the terms the broker holds as far
    as the kernel knows (`terms`), its fills' ledger, the broker's order ID once bound,
    whether it ended, the modify and the cancel waiting for a result, and the commands
    waiting for the bind."""

    intent: OrderIntent
    terms: OrderTerms
    ledger: FillLedger
    order_id: str = ""
    ended: bool = False
    modify: ModifyRequest | None = None
    cancel: CancelRequest | None = None
    deferred: list[Mapping[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SavedOrder:
    """One kernel order of a saved venue map (module docstring)."""

    intent: OrderIntent
    terms: OrderTerms
    order_id: str
    filled_units: int
    filled_value: float
    modify: tuple[OrderTerms, OrderTerms] | None
    cancel: bool
    deferred: tuple[dict[str, Any], ...]


def order_record(kernel_order: int, saved: SavedOrder) -> dict[str, Any]:
    """The plain-data record of `saved`, kernel order `kernel_order`."""
    intent = {name: getattr(saved.intent, name) for name in _INTENT_FIELDS}
    intent["purpose"] = saved.intent.purpose.value
    modify = None
    if saved.modify is not None:
        modify = {
            "terms": _terms_record(saved.modify[0]),
            "replaces": _terms_record(saved.modify[1]),
        }
    return {
        "order": kernel_order,
        "intent": intent,
        "terms": _terms_record(saved.terms),
        "order_id": saved.order_id,
        "filled_units": saved.filled_units,
        "filled_value": saved.filled_value,
        "modify": modify,
        "cancel": saved.cancel,
        "deferred": [dict(command) for command in saved.deferred],
    }


def read_records(records: Any) -> dict[int, SavedOrder]:
    """The saved orders of a venue map's records, by saved kernel id. Raises LiveError for
    records that cannot be read, naming the field."""
    if not isinstance(records, Sequence) or isinstance(records, str):
        raise LiveError(f"the saved venue map is not a list of orders: {records!r}")
    read: dict[int, SavedOrder] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise LiveError(f"a saved venue map order is not a record: {record!r}")
        kernel_order = _whole(record, "order")
        read[kernel_order] = _read(record, kernel_order)
    return read


def _read(record: Mapping[str, Any], kernel_order: int) -> SavedOrder:
    where = f"saved venue map order o{kernel_order}"
    try:
        intent_fields = dict(record["intent"])
        intent_fields["purpose"] = Purpose(intent_fields["purpose"])
        intent = OrderIntent(**{name: intent_fields[name] for name in _INTENT_FIELDS})
        modify = record["modify"]
        pending = None
        if modify is not None:
            pending = (_terms(modify["terms"]), _terms(modify["replaces"]))
        return SavedOrder(
            intent=intent,
            terms=_terms(record["terms"]),
            order_id=str(record["order_id"]),
            filled_units=_whole(record, "filled_units"),
            filled_value=float(record["filled_value"]),
            modify=pending,
            cancel=bool(record["cancel"]),
            deferred=tuple(dict(command) for command in record["deferred"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LiveError(f"the {where} cannot be read ({error!r}): {dict(record)!r}") from None


class RestoredOrders:
    """The venue map's part for a restart (module docstring). Mixed into
    `venue_map.VenueMap`, whose state it reads (the engine, its orders, the order ID and
    intent indexes, the instruments' ticks, the alerts) and whose `_request` makes a
    modify's or a cancel's request."""

    _engine: Any
    _alerts: Any
    _markets: tuple[tuple[str, str], ...]
    _ticks: tuple[float, ...]
    _orders: dict[int, VenueOrder]
    _by_order_id: dict[str, int]
    _by_intent: dict[str, int]

    def saved(self) -> list[dict[str, Any]]:
        """The records of the orders the kernel still works, oldest first."""
        working = set(self._engine.working_orders())
        return [
            order_record(kernel_order, _saved(order))
            for kernel_order, order in sorted(self._orders.items())
            if kernel_order in working and not order.ended
        ]

    def restore(
        self, saved: Mapping[int, SavedOrder], pairs: Mapping[int, int]
    ) -> dict[int, SavedOrder]:
        """Rebuild the map's orders from `saved` (by saved kernel id) and the kernel's
        `pairs` ({saved id: new id}); the saved order of each new id rebuilt. An order of
        the pairs with no record (a strategy exit the kernel had not sent, whose Place it
        makes later) is left to that Place. Raises LiveError for an instrument the
        engine does not trade, or a map that holds orders already."""
        if self._orders:
            raise LiveError("a venue map is restored before it places any order")
        rebuilt: dict[int, SavedOrder] = {}
        for old, new in sorted(pairs.items(), key=lambda pair: pair[1]):
            record = saved.get(old)
            if record is None:
                continue
            intent = record.intent
            market = (intent.symbol, intent.exchange)
            if market not in self._markets:
                raise LiveError(f"the saved order o{old} trades {market}, not this engine's")
            ledger = FillLedger(
                self._engine,
                new,
                tick=self._ticks[self._markets.index(market)],
                alerts=self._alerts,
                filled_units=record.filled_units,
                filled_value=record.filled_value,
            )
            deferred = [{**command, "order_id": new} for command in record.deferred]
            self._orders[new] = VenueOrder(intent, record.terms, ledger, deferred=deferred)
            self._by_intent[intent.intent_id] = new
            rebuilt[new] = record
        return rebuilt

    def restored_changes(self, kernel_order: int, saved: SavedOrder) -> list[OrderRequest]:
        """The modify and cancel of a restored, bound order whose results the restart
        cannot know: the saved pending ones and the commands that waited for the bind,
        as requests the order worker settles by reading the order (3.6). A saved modify
        replaces the order's saved terms, the last the kernel confirmed."""
        order = self._orders[kernel_order]
        made: list[OrderRequest] = []
        if saved.modify is not None:
            command = {"command": "modify", "order_id": kernel_order}
            made.append(self._request(order, command | _command_terms(saved.modify[0])))
        if saved.cancel:
            made.append(self._request(order, {"command": "cancel", "order_id": kernel_order}))
        deferred, order.deferred = order.deferred, []
        made += [self._request(order, command) for command in deferred]
        return made

    def changing(self, kernel_order: int) -> bool:
        """True while a modify or cancel of the order waits for its result or its bind."""
        order = self._orders.get(kernel_order)
        return order is not None and bool(order.modify or order.cancel or order.deferred)

    def bound(self) -> dict[str, int]:
        """Every order this map bound, by broker order ID: its kernel order."""
        return dict(self._by_order_id)

    def filled(self, kernel_order: int) -> int:
        """The units of a kernel order its ledger has given the kernel."""
        return self._orders[kernel_order].ledger.filled_units

    def bind_restored(self, kernel_order: int, order_id: str) -> None:
        """Bind a restored order to its broker order ID, as the journal binds its intent;
        its modify and cancel are `restored_changes`, never released as new requests."""
        self._orders[kernel_order].order_id = order_id
        self._by_order_id[order_id] = kernel_order

    def _request(self, order: VenueOrder, command: Mapping[str, Any]) -> OrderRequest:
        """A modify's or a cancel's request (`venue_map.VenueMap` makes it)."""
        raise NotImplementedError


def _saved(order: VenueOrder) -> SavedOrder:
    modify = None if order.modify is None else (order.modify.terms, order.modify.replaces)
    return SavedOrder(
        intent=order.intent,
        terms=order.terms,
        order_id=order.order_id,
        filled_units=order.ledger.filled_units,
        filled_value=order.ledger.filled_value,
        modify=modify,
        cancel=order.cancel is not None,
        deferred=tuple(dict(command) for command in order.deferred),
    )


def _command_terms(terms: OrderTerms) -> dict[str, Any]:
    """A kernel command's terms fields for `terms` (`_venue_requests.terms_of` reads them)."""
    return {
        "quantity_units": float(terms.quantity),
        "order_type": terms.price_type,
        "price": terms.price or None,
        "trigger_price": terms.trigger_price or None,
    }


def _terms_record(terms: OrderTerms) -> dict[str, Any]:
    return {name: getattr(terms, name) for name in _TERMS_FIELDS}


def _terms(record: Mapping[str, Any]) -> OrderTerms:
    return OrderTerms(
        quantity=int(record["quantity"]),
        price_type=str(record["price_type"]),
        price=float(record["price"]),
        trigger_price=float(record["trigger_price"]),
    )


def _whole(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveError(f"the saved venue map's {name} is not a whole number: {value!r}")
    return value


__all__ = ["RestoredOrders", "SavedOrder", "VenueOrder", "order_record", "read_records"]

"""Held fills and held cancels (16-session-design 4.11): words the order path holds for
the tradebook, released by the session so the kernel always knows the position it must
protect.

- **A held fill** (units filled whose price the tradebook has not given: a fill reported
  at price 0, or a polled fill awaiting the tradebook) is released after
  `HELD_FILL_RELEASE_AFTER_S` (`release_fill`) at, in order: the held word's own average
  price when positive, the feed's last price of the instrument, the order's limit or
  trigger, the last bar close. It is an estimate, kept in `estimates`, and never
  re-priced (the first positive price stands).
- **A held cancel** (cancelled with no fill the tradebook has confirmed: a polled stream
  always says 0 filled) holds the kernel's exit owner while the order is gone at the
  broker. After `HELD_CANCEL_DECIDE_AFTER_S`, when it is the only held word of its leg,
  the position book is read (`ReadPositions`), and every `POSITION_READ_EVERY_S` until it
  decides: the broker's net units of the instrument (MIS) equal to the kernel's, the
  order filled nothing more and the cancel is released as reported; fewer on the same
  side by `d` at most the order's units left, it filled `d` before its cancel, released
  with them at the estimate (`release_cancel`); anything else, or a book that cannot be
  read, an ERROR alert at once and a read again.

A word is held's when its filled units are above what the kernel has of the order (a
fill), else its terminal word waits (a cancel). Each held word is released once: it is
known by its order and the time it was first held, so a later word of the same order (a
fill released, then the order's cancel held) is a word of its own. Times are UTC
nanoseconds, prices rupees, quantities units. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._order_state import HeldFill
from ._plan_catch_up import INSTRUMENT
from ._session_trader import Trader
from ._stops_kernel import order_of
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import LIVE_PRODUCT
from ._worker_queue import ReadPositions, ReleaseCancel, ReleaseFill
from .inbox import PositionRows

logger = logging.getLogger(LOGGER_NAME)

#: Seconds a held fill waits for the tradebook before the session releases it.
HELD_FILL_RELEASE_AFTER_S = 2.0
#: Seconds a held cancel waits for the tradebook before the position book decides it.
HELD_CANCEL_DECIDE_AFTER_S = 3.0
#: Seconds between two position-book reads for a held cancel.
POSITION_READ_EVERY_S = 2.0

_NS = NANOS_PER_SECOND


@dataclass(frozen=True)
class Estimate:
    """Units of order `order_id` of strategy `strategy` released at a price the books did
    not confirm: `filled_quantity` units filled in all, at `price` rupees, at `at_ns`."""

    strategy: str
    order_id: str
    filled_quantity: int
    price: float
    at_ns: int


class HeldWords:
    """The held words of one session (module docstring). `session` is the
    `_session_loop.Session` whose strategies it watches."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._held: tuple[HeldFill, ...] = ()
        # The held words released, by order ID and the time each was first held; kept
        # while the worker's held list still shows them (it catches up a step later).
        self._released: set[tuple[str, int]] = set()
        self._read_due_ns: int | None = None
        self._alerted: set[str] = set()
        #: The fills released at an estimate, oldest first.
        self.estimates: list[Estimate] = []

    def update(self, held: tuple[HeldFill, ...]) -> None:
        """The order path's held words, as the worker last posted them."""
        self._held = held
        self._released &= {_key(word) for word in held}

    def held_cancel(self, trader: Trader, kernel_order: int) -> bool:
        """True while the order's cancel is held (it is gone at the broker)."""
        order_id = trader.venue.order_id_of(kernel_order)
        return any(w.order_id == order_id and self._is_cancel(trader, w) for w in self._held)

    def estimated(self, order_id: str, filled_quantity: int) -> bool:
        """True when the order's word of `filled_quantity` units filled in all was
        released at an estimate."""
        return any(
            (estimate.order_id, estimate.filled_quantity) == (order_id, filled_quantity)
            for estimate in self.estimates
        )

    def due(self, now_ns: int) -> None:
        """Release the held fills past their wait; ask for the position book for a held
        cancel past its wait."""
        cancels = []
        for word in self._held:
            owner = self._owner(word.order_id)
            if owner is None:
                continue
            trader, kernel_order = owner
            if not self._is_cancel(trader, word):
                if now_ns - word.held_since_ns >= round(HELD_FILL_RELEASE_AFTER_S * _NS):
                    self._release(trader, kernel_order, word, word.filled_quantity, now_ns)
            elif self._waits(word, now_ns) and (
                sum(self._leg_of(w) == trader.name for w in self._held) == 1
            ):
                cancels.append(word)
        if cancels and (self._read_due_ns is None or now_ns >= self._read_due_ns):
            self._read_due_ns = now_ns + round(POSITION_READ_EVERY_S * _NS)
            self._session.hand_over([ReadPositions()])

    def _waits(self, word: HeldFill, now_ns: int) -> bool:
        """Whether a held cancel past its wait is still to be decided: not released yet
        (the worker's next word of the held list drops it)."""
        waited = now_ns - word.held_since_ns >= round(HELD_CANCEL_DECIDE_AFTER_S * _NS)
        return waited and _key(word) not in self._released

    def positions(self, message: PositionRows) -> None:
        """A position-book read: decide each held cancel past its wait."""
        for word in self._held:
            owner = self._owner(word.order_id)
            if owner is None or not self._is_cancel(owner[0], word):
                continue
            trader, kernel_order = owner
            net = _net_units(message.positionbook, trader.symbol, trader.exchange)
            filled = self._decide(trader, kernel_order, net)
            if filled is None:
                self._undecided(trader, word.order_id, net)
                continue
            self._release(trader, kernel_order, word, filled, message.read_ns)

    def _decide(self, trader: Trader, kernel_order: int, net: int | None) -> int | None:
        """The units the order filled in all, from the broker's net position; None when
        the book does not tell."""
        order = order_of(trader, kernel_order)
        if net is None or order is None:
            return None
        known, kernel = round(order["filled_units"]), trader.open_units()
        left = round(order["quantity_units"]) - known
        if net == kernel:
            return known
        same_side = net == 0 or (net > 0) == (kernel > 0)
        missing = round(abs(kernel) - abs(net))
        if same_side and 0 < missing <= left:
            return known + missing
        return None

    def _release(
        self, trader: Trader, kernel_order: int, word: HeldFill, filled: int, now_ns: int
    ) -> None:
        if _key(word) in self._released:
            return
        self._released.add(_key(word))
        price = self._price(trader, kernel_order, word)
        if self._is_cancel(trader, word):
            request: Any = ReleaseCancel(word.order_id, filled, price)
        else:
            request = ReleaseFill(word.order_id, price)
        self.estimates.append(Estimate(trader.name, word.order_id, filled, price, now_ns))
        self._session.event(
            "held_released",
            strategy=trader.name,
            order_id=word.order_id,
            filled=filled,
            price=price,
        )
        self._session.hand_over([request])

    def _price(self, trader: Trader, kernel_order: int, word: HeldFill) -> float:
        """The estimate's price (module docstring), rupees."""
        if word.average_price > 0:
            return word.average_price
        tick = self._session.last_tick.get((trader.symbol, trader.exchange))
        if tick is not None and tick.ltp > 0:
            return float(tick.ltp)
        order = order_of(trader, kernel_order) or {}
        for key in ("limit_rupees", "trigger_rupees"):
            if order.get(key):
                return float(order[key])
        return float(call(trader.engine.last_close, INSTRUMENT) or 0.0)

    def _undecided(self, trader: Trader, order_id: str, net: int | None) -> None:
        if order_id in self._alerted:
            return
        self._alerted.add(order_id)
        book = "cannot be read" if net is None else f"shows {net} units"
        self._session.alert(
            AlertKind.ERROR,
            f"{trader.symbol}: stop {order_id} cancelled, its fills not confirmed (the "
            f"position book {book}, the engine holds {trader.open_units():g}), no exit sent: "
            "check the order book",
        )

    def _is_cancel(self, trader: Trader, word: HeldFill) -> bool:
        kernel_order = trader.venue.kernel_order_of(word.order_id)
        order = None if kernel_order is None else order_of(trader, kernel_order)
        return order is not None and word.filled_quantity <= round(order["filled_units"])

    def _owner(self, order_id: str) -> tuple[Trader, int] | None:
        for trader in self._session.traders:
            kernel_order = trader.venue.kernel_order_of(order_id)
            if kernel_order is not None:
                return trader, kernel_order
        return None

    def _leg_of(self, word: HeldFill) -> str | None:
        owner = self._owner(word.order_id)
        return None if owner is None else owner[0].name


def _key(word: HeldFill) -> tuple[str, int]:
    """What tells one held word from another: its order and when it was first held."""
    return word.order_id, word.held_since_ns


def _net_units(answer: Mapping[str, Any], symbol: str, exchange: str) -> int | None:
    """The broker's net MIS units of an instrument in a position-book answer, signed; None
    when the answer cannot be read."""
    rows = answer.get("data") if answer.get("status") == "success" else None
    if not isinstance(rows, list):
        return None
    net = 0
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if (row.get("symbol"), row.get("exchange"), row.get("product")) != (
            symbol,
            exchange,
            LIVE_PRODUCT,
        ):
            continue
        try:
            net += round(float(row.get("quantity", 0)))
        except (TypeError, ValueError):
            return None
    return net


__all__ = [
    "HELD_CANCEL_DECIDE_AFTER_S",
    "HELD_FILL_RELEASE_AFTER_S",
    "POSITION_READ_EVERY_S",
    "Estimate",
    "HeldWords",
]

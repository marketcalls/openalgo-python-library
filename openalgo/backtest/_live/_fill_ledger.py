"""The one door through which every fill of the session's own orders reaches the kernel
(16-session-design 2.6), and the terminal word that ends an order there.

The order path hands over cumulative updates: the filled quantity and the average price
of the whole order, every fill priced. The kernel takes increments. A `FillLedger`, one
per kernel order, knows what the kernel was given (`filled_units`, and `filled_value`,
the broker's cumulative value as of the last increment applied) and turns each
cumulative word into the units newly filled and their price, so a fill seen by two
sources (the stream and a poll, a released held fill, the tradebook rows a
reconciliation attributes to the order, the journal at a restart) is applied once:

- Fewer units than known: stale, logged and ignored. The same units: nothing new.
- More: `units = F - filled_units` at `(A * F - filled_value) / units`, rounded to the
  instrument's tick (a trade is on the tick; a cumulative average is not). A derived
  price not above 0, or outside `INCREMENT_PRICE_BAND_PCT` of A (an average the broker
  rounded too coarsely), gives way to A itself, flagged `derived_from_average`.
- The kernel takes it with `apply_fill(order, units, price, at=ts_recv_ns)`, and the
  ledger moves to F and A * F. A refusal (`VenueRefusalError`) is a reconciliation
  matter: logged with both figures, an ERROR alert, and the result says a book read is
  due. The ledger stays where it was, since nothing was recorded, except after
  "would_reverse", where the kernel recorded the closing part and refused the rest: the
  ledger moves to the broker's figures, so the excess is reported once, not on every
  later word.

Then a terminal word: REJECTED is `apply_reject(order, "venue")`, CANCELLED
`apply_cancel(order)`; COMPLETE needs nothing once the kernel's order is filled, and an
order the kernel still holds working is cancelled for the rest, with an ERROR alert. A
terminal result for an order the kernel had already ended is the race it expects
("order_ended"), logged at debug level.

Quantities are whole units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

import enum
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..errors import LiveError, VenueRefusalError
from ._types import LOGGER_NAME, AlertKind, AlertsLike, OrderStatus, OrderUpdate

logger = logging.getLogger(LOGGER_NAME)

#: A price derived from two cumulative averages must lie within this percentage of the
#: newer average; outside it the averages were rounded too coarsely to derive a price
#: from, and the average itself is used (16-session-design 2.6).
INCREMENT_PRICE_BAND_PCT = 2.0
#: The reject code the kernel records for a refusal by the broker or OpenAlgo
#: (`_reject_reasons.REASONS["venue"]`).
VENUE_REJECT_CODE = "venue"
#: The refusal kind of a terminal result for an order the kernel had already ended: a
#: fill, a cancel or a rejection reported after the kernel completed it is the race the
#: kernel expects (`errors.VENUE_REFUSAL_KINDS`).
ENDED_RACE = "order_ended"
#: The refusal kind after which the kernel recorded the units that closed the position.
WOULD_REVERSE = "would_reverse"
_PERCENT = 100.0


class FillOutcome(enum.Enum):
    """What became of one cumulative word's fills."""

    #: An increment reached the kernel.
    APPLIED = "applied"
    #: The kernel already has these units (a second source of the same fill).
    NOTHING_NEW = "nothing_new"
    #: Fewer units than the kernel has: a stale word, ignored.
    STALE = "stale"
    #: The kernel refused the increment: a reconciliation matter.
    REFUSED = "refused"


@dataclass(frozen=True)
class Increment:
    """The units newly filled (whole units) at `price` rupees; `derived_from_average` when
    the broker's average was used because no price could be derived on the tick."""

    units: int
    price: float
    derived_from_average: bool = False


@dataclass(frozen=True)
class FillResult:
    """What one word did to kernel order `kernel_order`: its fills' `outcome`, the
    `increment` given (None when none was), the refusal's `refusal` kind when REFUSED,
    `ended` the terminal status it applied (None for none), and `mismatch` when the
    kernel and the broker now differ, so the loop asks for a book read."""

    kernel_order: int
    outcome: FillOutcome
    increment: Increment | None = None
    refusal: str = ""
    ended: OrderStatus | None = None
    mismatch: bool = False


def next_increment(
    filled_units: int, filled_value: float, filled_quantity: int, average_price: float, tick: float
) -> tuple[FillOutcome, Increment | None]:
    """The increment a cumulative word gives: the kernel has `filled_units` worth
    `filled_value` rupees (the broker's cumulative value at the last increment), the word
    says `filled_quantity` units at an average of `average_price` rupees, and the
    instrument's tick is `tick` rupees. (NOTHING_NEW or STALE, None) when it gives no
    units; (APPLIED, the increment) otherwise (module docstring)."""
    if filled_quantity < filled_units:
        return FillOutcome.STALE, None
    if filled_quantity == filled_units:
        return FillOutcome.NOTHING_NEW, None
    units = filled_quantity - filled_units
    derived = on_tick((average_price * filled_quantity - filled_value) / units, tick)
    band = average_price * INCREMENT_PRICE_BAND_PCT / _PERCENT
    if derived > 0 and abs(derived - average_price) <= band:
        return FillOutcome.APPLIED, Increment(units, derived)
    return FillOutcome.APPLIED, Increment(units, average_price, derived_from_average=True)


def on_tick(price: float, tick: float) -> float:
    """`price` rounded to the nearest multiple of `tick` (rupees), to the tick's own
    decimals so that 0.05 times 10003 reads 500.15."""
    places = max(0, -Decimal(repr(tick)).normalize().as_tuple().exponent)
    return round(round(price / tick) * tick, places)


class FillLedger:
    """The fills one kernel order of `engine` (an `Engine.session`) has been given, and
    the door for more (module docstring). `kernel_order` is its ID, `tick` the
    instrument's tick in rupees, `alerts` the session's; a restart seeds `filled_units`
    and `filled_value` from journal L. Raises LiveError for a tick that is not a positive
    number of rupees."""

    def __init__(
        self,
        engine: Any,
        kernel_order: int,
        *,
        tick: float,
        alerts: AlertsLike,
        filled_units: int = 0,
        filled_value: float = 0.0,
    ) -> None:
        real = isinstance(tick, float | int) and not isinstance(tick, bool)
        if not (real and math.isfinite(tick) and tick > 0):
            raise LiveError(
                f"the fills of kernel order o{kernel_order} are priced on the instrument's tick, "
                f"a positive number of rupees, not {tick!r}: take it from the preflight's facts"
            )
        self._engine = engine
        self._order = kernel_order
        self._tick = tick
        self._alerts = alerts
        self._filled_units = filled_units
        self._filled_value = filled_value

    @property
    def filled_units(self) -> int:
        """The units the kernel has been given (after a "would_reverse", the broker's)."""
        return self._filled_units

    @property
    def filled_value(self) -> float:
        """Rupees: the broker's cumulative value (units times average) at the last word
        applied."""
        return self._filled_value

    def apply(self, update: OrderUpdate) -> FillResult:
        """Give the kernel the fills `update` (a cumulative word of this order) adds, then
        its terminal status (module docstring). Never raises for a refusal by the kernel:
        it is in the result."""
        outcome, increment = next_increment(
            self._filled_units,
            self._filled_value,
            update.filled_quantity,
            update.average_price,
            self._tick,
        )
        refusal = ""
        if outcome is FillOutcome.STALE:
            told, known = update.filled_quantity, self._filled_units
            logger.info("order %s: a stale word, %d filled of %d", update.order_id, told, known)
        elif increment is not None:
            refusal = self._give(update, increment)
            if refusal:
                outcome, increment = FillOutcome.REFUSED, None
        ended, differs = self._end(update)
        return FillResult(
            self._order, outcome, increment, refusal, ended, mismatch=bool(refusal) or differs
        )

    def _give(self, update: OrderUpdate, increment: Increment) -> str:
        """Hand the increment to the kernel; the refusal's kind, or "" when it took it."""
        try:
            self._engine.apply_fill(
                self._order, float(increment.units), increment.price, update.ts_recv_ns
            )
        except VenueRefusalError as error:
            if error.kind == WOULD_REVERSE:
                self._move(update)
            self._mismatch(
                f"the engine refused {increment.units} units at {increment.price} of order "
                f"{update.order_id} ({update.symbol} {update.exchange}): the broker reports "
                f"{update.filled_quantity} filled at {update.average_price}, the engine had "
                f"{self._filled_units} ({error.kind}: {error}). The books are read again; "
                "check the position in OpenAlgo"
            )
            return error.kind
        self._move(update)
        return ""

    def _move(self, update: OrderUpdate) -> None:
        self._filled_units = update.filled_quantity
        self._filled_value = update.average_price * update.filled_quantity

    def _end(self, update: OrderUpdate) -> tuple[OrderStatus | None, bool]:
        """Apply `update`'s terminal status; (the status applied or None, whether the
        engine and the broker now differ)."""
        status = update.status
        if status is OrderStatus.REJECTED:
            logger.warning("order %s rejected: %s", update.order_id, update.rejection_reason)
            return status, self._ended(update, self._reject)
        if status is OrderStatus.CANCELLED:
            return status, self._ended(update, self._cancel)
        if status is not OrderStatus.COMPLETE or self._order not in self._engine.working_orders():
            return None, False
        self._mismatch(
            f"OpenAlgo reports order {update.order_id} ({update.symbol} {update.exchange}) "
            f"complete with {update.filled_quantity} units filled while the engine waits for "
            "more: the rest is taken as cancelled. Check the order in OpenAlgo"
        )
        self._ended(update, self._cancel)
        return status, True

    def _reject(self) -> None:
        self._engine.apply_reject(self._order, VENUE_REJECT_CODE)

    def _cancel(self) -> None:
        self._engine.apply_cancel(self._order)

    def _ended(self, update: OrderUpdate, call: Callable[[], None]) -> bool:
        """Make the terminal `call`; True when the engine refused it for another reason
        than having ended the order already (logged and alerted)."""
        try:
            call()
        except VenueRefusalError as error:
            if error.kind == ENDED_RACE:
                logger.debug("order %s had ended in the engine already: %s", update.order_id, error)
                return False
            self._mismatch(
                f"the engine refused the {update.status.value} word of order "
                f"{update.order_id} ({error.kind}: {error}); check the order in OpenAlgo"
            )
            return True
        return False

    def _mismatch(self, text: str) -> None:
        logger.error(text)
        self._alerts.send(AlertKind.ERROR, text)


__all__ = [
    "ENDED_RACE",
    "INCREMENT_PRICE_BAND_PCT",
    "VENUE_REJECT_CODE",
    "WOULD_REVERSE",
    "FillLedger",
    "FillOutcome",
    "FillResult",
    "Increment",
    "next_increment",
    "on_tick",
]

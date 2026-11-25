"""The one door through which every fill of the session's own orders reaches the kernel
(16-session-design 2.6), and the terminal word that ends an order there.

The order path hands over cumulative updates: the filled quantity and the average price
of the whole order, every fill priced. The kernel takes increments. A `FillLedger`, one
per kernel order, knows what the kernel was given (`filled_units`, and `filled_value`,
those units times the prices they were given at) and turns each cumulative word into the
units newly filled and their price, so a fill seen by two sources (the stream and a poll,
a released held fill, the tradebook rows a reconciliation attributes to the order, the
journal at a restart) is applied once:

- Fewer units than known: stale, logged and ignored. The same units: nothing new.
- More: `units = F - filled_units` at the derived price `(A * F - filled_value) / units`.
  The broker rounds (or cuts) A to its last decimal, at the paisa or finer
  (`AVERAGE_DECIMALS_AT_LEAST`), so the derived price is off by less than one unit of
  that decimal times F over units: its rounding bound. The ticks strictly within the
  bound tell the units' price. One tick: that is the price (a trade is on the tick).
  None: the units are several trades whose average is the derived price itself. Two or
  more (a large order's average priced against a few units, or a later part of an order
  at a 0.01 tick): the tick cannot be told, and the derived price is used as it is,
  flagged `derived_from_average`, still within its bound, where the average itself could
  be far from the units' price. A first fill is never flagged: its derived price is the
  broker's own average of those units. A derived price not above 0 (figures that do not
  agree) gives way to A itself, flagged. So a fill is flagged only when its true price
  may lie on another tick than the one shown. Since `filled_value` is what the kernel was
  given, the next word's derived price makes good any error, and the kernel's value
  follows the broker's A times F.
- The kernel takes it with `apply_fill(order, units, price, at=ts_recv_ns)`, and the
  ledger moves to F and adds the increment's value. A refusal (`VenueRefusalError`) is
  a reconciliation matter: logged with both figures, an ERROR alert, and the result says
  a book read is due. The ledger stays where it was, since nothing was recorded, except
  after "would_reverse", where the kernel recorded the closing part and refused the rest:
  the result reports the units the kernel recorded (read from the kernel's order), and
  the ledger moves past the whole increment, so the excess is reported once, not on
  every later word.

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

#: The fewest decimals an average price is taken to be given to, the paisa: OpenAlgo's
#: broker mappings and its sandbox round an average to 2 decimals, none coarser, and a
#: number cannot show its trailing zeros (an average of 500.10 reads 500.1).
AVERAGE_DECIMALS_AT_LEAST = 2
#: Decimals a derived price or value keeps: finer digits are the noise of float arithmetic,
#: not money (the paisa is 2 decimals, the finest tick 4).
PRICE_DECIMALS = 6
#: The reject code the kernel records for a refusal by the broker or OpenAlgo, or by the
#: session before sending (`_session_unsent`; `_reject_reasons.REASONS["venue"]`).
VENUE_REJECT_CODE = "venue"
#: The refusal kind of a terminal result for an order the kernel had already ended: a
#: fill, a cancel or a rejection reported after the kernel completed it is the race the
#: kernel expects (`errors.VENUE_REFUSAL_KINDS`).
ENDED_RACE = "order_ended"
#: The refusal kind after which the kernel recorded the units that closed the position.
WOULD_REVERSE = "would_reverse"


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
    the units' true price may lie on another tick than `price`: the cumulative averages
    were rounded too coarsely for so few units of a later fill (the derived price, within
    its rounding bound), or did not agree (the newer average itself)."""

    units: int
    price: float
    derived_from_average: bool = False


@dataclass(frozen=True)
class FillResult:
    """What one word did to kernel order `kernel_order`: its fills' `outcome`, the
    `increment` the kernel recorded (None when it recorded none; after a "would_reverse"
    refusal, the closing part), the refusal's `refusal` kind when REFUSED,
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
    `filled_value` rupees (those units times the prices they were given at), the word
    says `filled_quantity` units at an average of `average_price` rupees, and the
    instrument's tick is `tick` rupees. (NOTHING_NEW or STALE, None) when it gives no
    units; (APPLIED, the increment) otherwise (module docstring)."""
    if filled_quantity < filled_units:
        return FillOutcome.STALE, None
    if filled_quantity == filled_units:
        return FillOutcome.NOTHING_NEW, None
    units = filled_quantity - filled_units
    derived = round((average_price * filled_quantity - filled_value) / units, PRICE_DECIMALS)
    if derived <= 0:
        return FillOutcome.APPLIED, Increment(units, average_price, derived_from_average=True)
    bound = _rounding_bound(average_price, filled_quantity, units)
    nearest = on_tick(derived, tick)
    if not _within(nearest, derived, bound):
        return FillOutcome.APPLIED, Increment(units, derived)
    # The tick next nearest the derived price is the second one within the bound, if any.
    second = nearest + (tick if derived >= nearest else -tick)
    if not _within(second, derived, bound):
        return FillOutcome.APPLIED, Increment(units, nearest)
    first_fill = filled_units == 0
    return FillOutcome.APPLIED, Increment(units, derived, derived_from_average=not first_fill)


def _within(price: float, derived: float, bound: float) -> bool:
    """Whether `price` lies strictly within `bound` rupees of `derived`: the true price is
    off by less than the rounding bound, never by it."""
    return round(abs(price - derived) - bound, PRICE_DECIMALS) < 0


def _rounding_bound(average_price: float, filled_quantity: int, units: int) -> float:
    """Rupees by which the price of `units` derived from an average of `average_price`
    over `filled_quantity` units can be off: one unit of the average's last decimal (at
    the paisa or finer, `AVERAGE_DECIMALS_AT_LEAST`), whether the broker rounded or cut
    it, times `filled_quantity` over `units`."""
    shown = -Decimal(repr(average_price)).normalize().as_tuple().exponent
    decimals = max(AVERAGE_DECIMALS_AT_LEAST, shown)
    return 10.0**-decimals * filled_quantity / units


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
        # The units the kernel recorded: `filled_units` but after a "would_reverse".
        self._recorded_units = filled_units

    @property
    def filled_units(self) -> int:
        """The units the kernel has been given (after a "would_reverse", the broker's)."""
        return self._filled_units

    @property
    def filled_value(self) -> float:
        """Rupees: `filled_units` times the prices they were given at."""
        return self._filled_value

    def apply(self, update: OrderUpdate, flags: tuple[str, ...] = ()) -> FillResult:
        """Give the kernel the fills `update` (a cumulative word of this order) adds, with
        the venue's `flags` (`VENUE_FILL_FLAGS`: a GTT child's fill priced at its limit,
        made on a day no session ran), then its terminal status (module docstring). Never
        raises for a refusal by the kernel: it is in the result. Raises LiveError when the
        kernel no longer knows the order whose fill it refused in part."""
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
            refusal, increment = self._give(update, increment, flags)
            if refusal:
                outcome = FillOutcome.REFUSED
        ended, differs = self._end(update)
        return FillResult(
            self._order, outcome, increment, refusal, ended, mismatch=bool(refusal) or differs
        )

    def _give(
        self, update: OrderUpdate, increment: Increment, flags: tuple[str, ...]
    ) -> tuple[str, Increment | None]:
        """Hand the increment to the kernel with `flags`: ("", the increment) when it took
        it, else the refusal's kind and what it recorded (None but after a
        "would_reverse")."""
        units, price, at = float(increment.units), increment.price, update.ts_recv_ns
        try:
            # Flags go only with a fill that has them: every other fill's call is as before.
            if flags:
                self._engine.apply_fill(self._order, units, price, at, flags=list(flags))
            else:
                self._engine.apply_fill(self._order, units, price, at)
        except VenueRefusalError as error:
            self._mismatch(
                f"the engine refused {increment.units} units at {increment.price} of order "
                f"{update.order_id} ({update.symbol} {update.exchange}): the broker reports "
                f"{update.filled_quantity} filled at {update.average_price}, the engine had "
                f"{self._filled_units} ({error.kind}: {error}). The books are read again; "
                "check the position in OpenAlgo"
            )
            if error.kind != WOULD_REVERSE:
                return error.kind, None
            recorded = self._recorded_of(increment)
            self._move(update, increment)
            return error.kind, recorded
        self._recorded_units += increment.units
        self._move(update, increment)
        return "", increment

    def _recorded_of(self, increment: Increment) -> Increment | None:
        """The units the kernel recorded of an increment it refused in part (read from its
        order), at the increment's price; None for none. Raises LiveError when the kernel
        no longer knows the order."""
        kernel = self._engine.order(self._order)
        if kernel is None:
            raise LiveError(f"the engine no longer knows kernel order o{self._order}")
        recorded = round(kernel["filled_units"]) - self._recorded_units
        self._recorded_units += recorded
        if recorded <= 0:
            return None
        return Increment(recorded, increment.price, increment.derived_from_average)

    def _move(self, update: OrderUpdate, increment: Increment) -> None:
        self._filled_units = update.filled_quantity
        value = self._filled_value + increment.units * increment.price
        self._filled_value = round(value, PRICE_DECIMALS)

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
    "AVERAGE_DECIMALS_AT_LEAST",
    "ENDED_RACE",
    "PRICE_DECIMALS",
    "VENUE_REJECT_CODE",
    "WOULD_REVERSE",
    "FillLedger",
    "FillOutcome",
    "FillResult",
    "Increment",
    "next_increment",
    "on_tick",
]

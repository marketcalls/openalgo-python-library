"""Tick and quantity helpers.

Pure functions with no table lookups: the caller supplies the tick size, the lot size
and the freeze quantity (from an `Instrument` or from `ReferenceTables`). Lot sizes,
freeze limits and order quantities may be any integer type, numpy integers included.

Prices are f64 rupees in the engine, so a price that should sit exactly on a tick can
arrive as 100.15000000000002. Every helper here first snaps a value that is within a
tiny tolerance of a whole number of ticks (or lots) onto it, so floating-point noise
never moves a price by a tick or a quantity by a lot.

One rule for an exact half, in both helpers: "nearest" rounds a value exactly half way
between two multiples away from zero (100.025 to 100.05, -100.025 to -100.05).
"""

from __future__ import annotations

import math
import numbers
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

from .._messages import counted
from ..errors import ConfigError

# ROUND_HALF_UP in `decimal` rounds an exact half away from zero.
_TICK_ROUNDING = {"nearest": ROUND_HALF_UP, "down": ROUND_FLOOR, "up": ROUND_CEILING}
_ABS_TOLERANCE = Decimal("1e-7")
# 64 units in the last place of an f64 (64 x 2**-52), relative: far above the rounding
# error of the few arithmetic steps a price goes through, and far below a tick even for
# a price of several lakh rupees on a 0.0025 tick (a looser bound snapped real prices).
_REL_TOLERANCE = Decimal(64) * Decimal(2) ** -52


def _whole(value: object) -> int | None:
    """`value` as a Python int when it is an integer of any type (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _decimal(value: object, what: str) -> Decimal:
    if isinstance(value, bool):
        raise ConfigError(f"{what} must be a number, got {value!r}")
    whole = _whole(value)
    if whole is not None:
        return Decimal(whole)
    if isinstance(value, Decimal):
        number = value
    else:
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{what} must be a number, got {value!r}") from None
        if not math.isfinite(as_float):
            raise ConfigError(f"{what} must be finite, got {value!r}")
        number = Decimal(repr(as_float))
    if not number.is_finite():
        raise ConfigError(f"{what} must be finite, got {value!r}")
    return number


def _snap(steps: Decimal) -> Decimal:
    """Move a value that is a whole number up to floating-point noise onto that number."""
    whole = steps.to_integral_value(rounding=ROUND_HALF_EVEN)
    tolerance = max(_ABS_TOLERANCE, abs(steps) * _REL_TOLERANCE)
    return whole if abs(steps - whole) <= tolerance else steps


def _mode(mode: str, table: dict) -> str:
    if mode not in table:
        raise ConfigError(f"mode must be 'nearest', 'down' or 'up', got {mode!r}")
    return table[mode]


def round_to_tick(price: float, tick_size: float, mode: str = "nearest") -> float:
    """Round a price in rupees to a multiple of `tick_size` (rupees).

    `mode` is "nearest" (an exact half rounds away from zero), "down" (towards minus
    infinity) or "up" (towards plus infinity). Raises `ConfigError` for a price that is
    not a finite number, a tick that is not positive, or an unknown mode.
    """
    tick = _decimal(tick_size, "tick_size")
    if tick <= 0:
        raise ConfigError(f"tick_size must be positive, got {tick_size!r}")
    steps = _snap(_decimal(price, "price") / tick)
    whole = steps.to_integral_value(rounding=_mode(mode, _TICK_ROUNDING))
    return float(whole * tick)


def is_tick_valid(price: float, tick_size: float) -> bool:
    """True when `price` (rupees) is a whole number of ticks of `tick_size` (rupees).

    Floating-point noise does not make a price off-tick. Raises `ConfigError` for a price
    that is not finite or a tick that is not positive.
    """
    tick = _decimal(tick_size, "tick_size")
    if tick <= 0:
        raise ConfigError(f"tick_size must be positive, got {tick_size!r}")
    steps = _snap(_decimal(price, "price") / tick)
    return steps == steps.to_integral_value()


def _lot(lot_size: object) -> int:
    lot = _whole(lot_size)
    if lot is None or lot < 1:
        raise ConfigError(f"lot_size must be a whole number of units, 1 or more, got {lot_size!r}")
    return lot


# On a magnitude, so "down" is towards zero and "up" away from zero.
_LOT_ROUNDING = {"nearest": ROUND_HALF_UP, "down": ROUND_FLOOR, "up": ROUND_CEILING}


def round_to_lot(quantity: float, lot_size: int, mode: str = "down") -> int:
    """Round a quantity in units to a whole multiple of `lot_size`; the result is in units.

    The default "down" rounds towards zero, so a sized order never becomes larger than
    was asked for. "up" rounds away from zero and "nearest" sends an exact half lot away
    from zero. The sign of `quantity` is kept. Raises `ConfigError` for a lot size that
    is not a whole number of 1 or more, a quantity that is not finite, or an unknown mode.
    """
    lot = _lot(lot_size)
    number = _decimal(quantity, "quantity")
    sign = -1 if number < 0 else 1
    lots = _snap(abs(number) / lot)
    whole = lots.to_integral_value(rounding=_mode(mode, _LOT_ROUNDING))
    return sign * int(whole) * lot


def units_to_lots(quantity: int, lot_size: int) -> int:
    """Convert a quantity in units to lots, at an adapter boundary that counts lots.

    Raises `ConfigError` when the quantity is not a whole number of lots.
    """
    lot = _lot(lot_size)
    lots = _snap(_decimal(quantity, "quantity") / lot)
    if lots != lots.to_integral_value():
        raise ConfigError(
            f"quantity {quantity!r} is not a whole number of lots of {counted(lot, 'unit')}; "
            "quantity is in units, not lots"
        )
    return int(lots)


def _freeze(freeze_qty: object) -> int | None:
    if freeze_qty is None:
        return None
    limit = _whole(freeze_qty)
    if limit is None or limit < 1:
        raise ConfigError(f"freeze_qty must be a whole number of units or None, got {freeze_qty!r}")
    return limit


def check_freeze(quantity: float, freeze_qty: int | None) -> bool:
    """True when a single order of `quantity` units is within the freeze limit.

    The exchange freezes an order whose quantity is greater than the limit, so a
    quantity equal to the limit passes. `freeze_qty=None` means no limit applies.
    Raises `ConfigError` for a limit that is not a whole number of 1 or more.
    """
    limit = _freeze(freeze_qty)
    number = abs(_decimal(quantity, "quantity"))
    return True if limit is None else number <= limit


def max_order_quantity(freeze_qty: int | None, lot_size: int = 1) -> int | None:
    """The largest single order, in units, that is a lot multiple within the freeze limit.

    None when there is no limit. Raises `ConfigError` when the limit is smaller than one
    lot, which means the reference data is inconsistent.
    """
    lot = _lot(lot_size)
    limit = _freeze(freeze_qty)
    if limit is None:
        return None
    if limit < lot:
        raise ConfigError(
            f"freeze quantity {limit} is smaller than one lot of {counted(lot, 'unit')}; "
            "the reference data for this instrument is inconsistent"
        )
    return (limit // lot) * lot


def split_by_freeze(quantity: int, freeze_qty: int | None, lot_size: int = 1) -> list[int]:
    """Slice an order that exceeds the freeze limit into orders that do not.

    `quantity` must be a positive whole number of lots (in units). Every slice is a lot
    multiple no larger than the freeze limit, the slices are as large as allowed, and
    they add up to `quantity`. Raises `ConfigError` when `quantity` is not a positive
    whole number of lots or the limit is smaller than one lot.
    """
    lot = _lot(lot_size)
    units = _whole(quantity)
    if units is None or units < 1:
        raise ConfigError(f"quantity must be a positive whole number of units, got {quantity!r}")
    if units % lot:
        raise ConfigError(
            f"quantity {units} is not a whole number of lots of {counted(lot, 'unit')}"
        )
    largest = max_order_quantity(freeze_qty, lot)
    if largest is None or units <= largest:
        return [units]
    full, rest = divmod(units, largest)
    return [largest] * full + ([rest] if rest else [])


__all__ = [
    "check_freeze",
    "is_tick_valid",
    "max_order_quantity",
    "round_to_lot",
    "round_to_tick",
    "split_by_freeze",
    "units_to_lots",
]

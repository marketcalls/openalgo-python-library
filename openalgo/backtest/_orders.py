"""Order requests and handles: what `buy`, `sell`, `exit` and `reduce` send to the kernel,
and the `Order` handle they return (14-strategy-api 2.4).

The request is a dict whose keys the binding reads (`rust/oa_py/src/backtest/engine/
request.rs`). This module only translates arguments; the kernel decides what it can
run, and anything not built yet comes back as NotImplementedError naming the feature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import units
from .errors import ConfigError
from .size import Sizer

VALIDITIES = {"day": "DAY", "gtt": "GTT"}
AFTER_FIRST_TARGET = (None, "keep", "breakeven")


def _distance(value: object, what: str) -> tuple[str, float, float | None]:
    distance = units.as_distance(value, what)
    if distance.unit == "atr":
        raise NotImplementedError(
            "ATR distances (bt.atr, bt.trail_atr) are not implemented yet (Phase 1 work in "
            "progress): the ATR value at the decision is resolved through openalgo.ta"
        )
    return (distance.unit, distance.value, None)


def _trail(value: object) -> tuple[Any, Any] | None:
    if value is None:
        return None
    if not isinstance(value, units.Trail):
        raise ConfigError(
            "trail= takes bt.trail(x, y), bt.trail_points, bt.trail_pct or bt.trail_atr"
        )
    step = None if value.step is None else _distance(value.step, "trail step")
    return (_distance(value.trigger, "trail trigger"), step)


def _breakeven(value: object) -> tuple[Any, Any] | None:
    if value is None:
        return None
    if not isinstance(value, units.Breakeven):
        raise ConfigError("breakeven= takes bt.breakeven(trigger, offset)")
    offset = None if value.offset is None else _distance(value.offset, "breakeven offset")
    return (_distance(value.trigger, "breakeven trigger"), offset)


def _targets(value: object) -> list[tuple[Any, float]]:
    if value is None:
        return []
    pairs = []
    for item in value:  # type: ignore[union-attr]
        level, fraction = item
        pairs.append((_distance(level, "target"), float(fraction)))
    return pairs


def leg_rules(
    stop: object,
    target: object,
    targets: object,
    trail: object,
    breakeven: object,
    after_first_target: str | None,
) -> dict[str, Any] | None:
    """The rules dict of a request, or None when no rule is given."""
    if after_first_target not in AFTER_FIRST_TARGET:
        raise ConfigError("after_first_target takes 'breakeven' or 'keep'")
    rules = {
        "stop": None if stop is None else _distance(stop, "stop"),
        "target": None if target is None else _distance(target, "target"),
        "targets": _targets(targets),
        "trail": _trail(trail),
        "breakeven": _breakeven(breakeven),
        "after_first_target": after_first_target,
    }
    if all(not value for value in rules.values()):
        return None
    return rules


def quantity(
    qty: float | None, lots: int | None, size: Sizer | None, default: Sizer | None
) -> dict:
    """The quantity dict: qty or lots, else size, else the strategy default, else the
    engine default (03-architecture 7). Raises ConfigError for qty together with lots."""
    if qty is not None and lots is not None:
        raise ConfigError("give qty= or lots=, not both")
    if qty is not None:
        return {"kind": "units", "value": float(qty)}
    if lots is not None:
        return {"kind": "lots", "value": int(lots)}
    sizer = size if size is not None else default
    if sizer is None:
        return {"kind": "sizer", "sizer": "default", "amount": None}
    if not isinstance(sizer, Sizer):
        raise ConfigError("size= takes a sizer from bt.size (bt.size.percent(50), ...)")
    return sizer.to_native()


def order_shape(price: float | None) -> dict[str, Any]:
    """MARKET without a price, LIMIT with one."""
    if price is None:
        return {"order_type": "MARKET", "price": None, "trigger": None}
    return {"order_type": "LIMIT", "price": float(price), "trigger": None}


def request(instrument: int, side: str, product: str, **fields: Any) -> dict[str, Any]:
    """A complete request dict with the defaults of a plain DAY order."""
    base = {
        "instrument": instrument,
        "side": side,
        "product": product,
        "validity": "DAY",
        "reduce_only": False,
        "reverse": False,
        "exit_reason": None,
        "rules": None,
        "leverage": None,
        "tag": None,
        **order_shape(None),
    }
    base.update(fields)
    return base


def validity(value: str) -> str:
    """The kernel's validity code for "day" or "gtt". Raises ConfigError otherwise."""
    if value not in VALIDITIES:
        raise ConfigError(f"validity takes 'day' or 'gtt', got {value!r}")
    return VALIDITIES[value]


class Order:
    """The handle `buy`, `sell`, `exit` and `reduce` return.

    `id` is the order id ("o1"), `status` its state ("Submitted", "Filled", "Rejected",
    ...), `filled_qty` the units filled and `avg_price` the average fill price in rupees;
    `reject_code` says why it was refused. `cancel()` cancels it while it works.
    """

    def __init__(self, number: int, lookup: Callable[[int], dict], cancel: Callable[[int], str]):
        self._number = number
        self._lookup = lookup
        self._cancel = cancel

    @property
    def id(self) -> str:
        return f"o{self._number}"

    @property
    def number(self) -> int:
        """The order id as a number (1 for "o1")."""
        return self._number

    @property
    def status(self) -> str:
        return self._lookup(self._number)["status"]

    @property
    def filled_qty(self) -> float:
        return self._lookup(self._number)["filled_units"]

    @property
    def avg_price(self) -> float:
        return self._lookup(self._number)["average_fill_rupees"]

    @property
    def reject_code(self) -> str | None:
        return self._lookup(self._number)["reject_code"]

    def cancel(self) -> str:
        """Cancel the order if it still works; returns its state afterwards."""
        return self._cancel(self._number)

    def __repr__(self) -> str:
        return f"Order({self.id}, {self.status})"


__all__ = ["Order", "leg_rules", "order_shape", "quantity", "request", "validity"]

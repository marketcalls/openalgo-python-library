"""The ATRs a condition strategy's order names (`bt.atr`, `bt.trail_atr` and
`bt.size.atr_risk` in `entry_when` and `short_when`), as the per-bar series the Rust
signal driver reads (`rust/oa_bt/src/engine/signals/atr.rs`).

An `on_bar` order resolves each ATR when it is placed (`_atr.decision_atr`,
`_atr.sizer_atr`). A condition's order template is built once, before the run, so each
ATR it names is computed here over the run's strategy bars (`_atr.atr_values`), one value
per bar, and the driver puts the deciding bar's value into the order. Both paths read
the same `openalgo.ta` ATR, causal by the registry, so they place the same orders. The
template holds NaN where an ATR goes (`TEMPLATE_ATR`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import units
from ._atr import atr_values
from ._series import SeriesStore

# One ATR as the driver reads it: (slot, target index or None, period, values per bar).
AtrInput = tuple[str, int | None, int, np.ndarray]
# A slot of the template: (slot, target index or None, period).
Slot = tuple[str, int | None, int]


def template_atr(period: int) -> float:
    """The ATR an order template holds: none yet; the driver fills it at each bar."""
    return math.nan


def _distance_slot(value: object, slot: str, index: int | None = None) -> list[Slot]:
    if isinstance(value, units.Distance) and value.unit == "atr" and value.period is not None:
        return [(slot, index, value.period)]
    return []


def atr_slots(args: dict[str, Any], quantity: dict[str, Any]) -> list[Slot]:
    """Every ATR the order arguments `args` and the resolved `quantity` name, as (slot,
    target index, period), with the slot names of `signals.rs`. The arguments were
    checked when the template was built (`_orders.leg_rules`)."""
    slots = _distance_slot(args.get("stop"), "stop")
    slots += _distance_slot(args.get("target"), "target")
    for index, (level, _fraction) in enumerate(args.get("targets") or ()):
        slots += _distance_slot(level, "targets", index)
    trail = args.get("trail")
    if isinstance(trail, units.Trail):
        slots += _distance_slot(trail.trigger, "trail_trigger")
        slots += _distance_slot(trail.step, "trail_step")
    breakeven = args.get("breakeven")
    if isinstance(breakeven, units.Breakeven):
        slots += _distance_slot(breakeven.trigger, "breakeven_trigger")
        slots += _distance_slot(breakeven.offset, "breakeven_offset")
    if quantity.get("sizer") == "atr_risk":
        slots.append(("sizer", None, int(quantity["period"])))
    return slots


def atr_inputs(slots: list[Slot], store: SeriesStore) -> list[AtrInput]:
    """The ATR of each slot at every strategy bar of `store` (the whole run), rupees, NaN
    while fewer than its period of bars have closed. Raises ConfigError when the
    causality registry does not clear the ATR (`_atr.atr_values`)."""
    by_period: dict[int, np.ndarray] = {}
    inputs = []
    for slot, index, period in slots:
        if period not in by_period:
            values = atr_values(store, period, whole_run=True)
            by_period[period] = np.ascontiguousarray(values, dtype=np.float64)
        inputs.append((slot, index, period, by_period[period]))
    return inputs


__all__ = ["AtrInput", "Slot", "atr_inputs", "atr_slots", "template_atr"]

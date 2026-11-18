"""The objects a live session's record keeps as tagged JSON objects (`session_record`'s
format): `bt.live` arguments that are not JSON values but hold what a replay must give
the session again. Each is written as an object whose `tag` names its type, and read back
as the object it was (`untagged`).

- **A distance** (`bt.points`, `bt.pct`, `bt.rupees`, `bt.ticks`, `bt.atr`):
  {"tag": "distance", "unit", "value", "period"}.
- **A brokerage preset** (`bt.costs.zerodha()`, `zero_brokerage`, `flat`, `percent`,
  `custom`): {"tag": "brokerage", "name", "dp_fee"} and either `preset`, the name of
  the built-in preset it was built from, or `rules`, its four rules (delivery, intraday,
  futures, options) as `bt.costs.Brokerage.rules` gives them, each [kind, first,
  second]: ["zero", 0, 0], ["flat", rupees an executed order, 0] or ["rate", fraction of
  the order's value, cap in rupees an executed order or null for no cap]. `dp_fee` is
  the DP fee in rupees a scrip a sell day before GST when the user gave one, or null for
  the preset's own (Zerodha's, standing in for a preset with none).
- **A cost model** (`bt.costs.CostModel`): {"tag": "cost_model", "brokerage" (a tagged
  brokerage), "rounding", "backfill", "schedules"}, `schedules` null for the built-in
  versions, else every version of the model as `CostModel.schedules()` gives it, dates
  as ISO text.

An object is tagged only when it reads back equal to itself; one that would not, and an
object of another type, is not kept (`tagged`).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Callable
from typing import Any

from openalgo.backtest import costs
from openalgo.backtest._costs.kernel import call, native
from openalgo.backtest.units import Distance

DISTANCE_TAG, BROKERAGE_TAG, COST_MODEL_TAG = "distance", "brokerage", "cost_model"
#: The built-in brokerage presets, by name (`bt.costs.preset`'s docstring).
PRESETS = ("zerodha", "zero_brokerage")
#: What a record keeps of an argument, for a refusal's message.
KEPT = (
    "JSON values (numbers, text, lists and dicts of them), a distance (bt.points, bt.pct, "
    "bt.rupees, bt.ticks, bt.atr), a cost preset (a preset name, or a preset from "
    "bt.costs) or a bt.costs.CostModel"
)
#: The source of a DP fee the user gave (`bt.costs.Brokerage.dp_fee_source`).
_SUPPLIED = "supplied"


def tagged(value: Any) -> dict[str, Any] | None:
    """`value` as its tagged object (module docstring), or None for a value of a type the
    record keeps no tag for. Raises ValueError for one that would not read back equal to
    itself."""
    if isinstance(value, Distance):
        return {"tag": DISTANCE_TAG, **dataclasses.asdict(value)}
    if isinstance(value, costs.Brokerage):
        found = _brokerage_tag(value)
        back = _brokerage(found)
        same = (back, back.name) == (value, value.name)
    elif isinstance(value, costs.CostModel):
        found = _cost_model_tag(value)
        same = _cost_model_tag(_cost_model(found)) == found
    else:
        return None
    if not same:
        raise ValueError(f"{value!r} would not read back from its record as itself")
    return found


def untagged(value: Any) -> Any:
    """`value` read from a record, each tagged object (module docstring) the object it
    was."""
    if isinstance(value, dict):
        tag = value.get("tag")
        if isinstance(tag, str) and frozenset(value) in _KEYS.get(tag, ()):
            return _READERS[tag](value)
        return {key: untagged(item) for key, item in value.items()}
    if isinstance(value, list):
        return [untagged(item) for item in value]
    return value


def _brokerage_tag(brokerage: costs.Brokerage) -> dict[str, Any]:
    supplied = brokerage.dp_fee_source == _SUPPLIED
    found = {
        "tag": BROKERAGE_TAG,
        "name": brokerage.name,
        "dp_fee": brokerage.dp_fee if supplied else None,
    }
    # A built-in preset is kept by its name: Zerodha's own DP fee belongs to its preset
    # alone, so the same rules built otherwise would not read back equal.
    for preset in PRESETS:
        if _brokerage({**found, "preset": preset}) == brokerage:
            return {**found, "preset": preset}
    return {**found, "rules": [_rule(rule) for rule in brokerage.rules.values()]}


def _rule(rule: tuple) -> list[Any]:
    """A rule as `Brokerage.rules` gives it, as the record keeps it (module docstring)."""
    kind, first, second = (*rule, 0.0, 0.0)[:3]
    return [kind, first, None if math.isinf(second) else second]


def _brokerage(found: dict[str, Any]) -> costs.Brokerage:
    if "preset" in found:
        kernel = call(native.CostBrokerage.named, found["preset"])
    else:
        # The kernel's own constructor, as `bt.costs.custom` takes a rate in percent
        # units and the record keeps it as the fraction the kernel holds: built from
        # that fraction, it reads back to the bit.
        rules = [
            (kind, first, math.inf if cap is None else cap) for kind, first, cap in found["rules"]
        ]
        kernel = call(native.CostBrokerage.custom, *rules)
    if found["dp_fee"] is not None:
        kernel = call(kernel.with_dp_fee, found["dp_fee"])
    return costs.Brokerage(found["name"], kernel)


def _cost_model_tag(model: costs.CostModel) -> dict[str, Any]:
    versions = model.schedules()
    return {
        "tag": COST_MODEL_TAG,
        "brokerage": _brokerage_tag(model.brokerage),
        "rounding": model.rounding,
        "backfill": model.backfill,
        "schedules": None if versions == costs.schedules() else _iso_dates(versions),
    }


def _cost_model(found: dict[str, Any]) -> costs.CostModel:
    versions = found["schedules"]
    return costs.CostModel(
        _brokerage(found["brokerage"]),
        found["rounding"],
        schedules=versions,
        include_builtin=versions is None,
        backfill=found["backfill"],
    )


def _iso_dates(value: Any) -> Any:
    """`value` with each date as ISO text, as `CostModel(schedules=...)` reads it back."""
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _iso_dates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_iso_dates(item) for item in value]
    return value


def _distance(found: dict[str, Any]) -> Distance:
    return Distance(found["unit"], found["value"], found["period"])


#: The keys of each tagged object, so that a dict of the user's own that happens to
#: carry a `tag` is read back as the dict it was.
_KEYS = {
    DISTANCE_TAG: (frozenset({"tag", "unit", "value", "period"}),),
    BROKERAGE_TAG: tuple(
        frozenset({"tag", "name", "dp_fee", terms}) for terms in ("preset", "rules")
    ),
    COST_MODEL_TAG: (frozenset({"tag", "brokerage", "rounding", "backfill", "schedules"}),),
}
_READERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    DISTANCE_TAG: _distance,
    BROKERAGE_TAG: _brokerage,
    COST_MODEL_TAG: _cost_model,
}


__all__ = [
    "BROKERAGE_TAG",
    "COST_MODEL_TAG",
    "DISTANCE_TAG",
    "KEPT",
    "PRESETS",
    "tagged",
    "untagged",
]

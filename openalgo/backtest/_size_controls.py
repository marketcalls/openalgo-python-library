"""`reinvest` and `size_caps`: the controls a strategy sets for every sizer
(12-position-sizing 5, FR-SIZE-07), as class attributes, in the units the kernel takes
(`rust/oa_bt/src/sizing/caps.rs`).

    class Breakout(bt.Strategy):
        size = bt.size.fixed_risk(1)
        reinvest = False                  # size from the initial capital
        size_caps = bt.size.caps(position=bt.pct(20), leverage=1, risk=bt.rupees(5_000))

`reinvest` (True by default) sizes from the current equity, so sizes compound; False
sizes every sizer, every cap given in percent or leverage and `target(percent=)` from
the capital allocated at the start. `size_caps` (None by default: no cap) bounds the
quantity any sizer gives (`bt.size.caps`). Both enter the kernel's risk dict and the
Assumptions block.
"""

from __future__ import annotations

from typing import Any

from .errors import ConfigError
from .size import SizeCaps

# The class attributes this module reads, with their defaults.
CONTROLS = {"reinvest": True, "size_caps": None}
_POSITION_UNITS = {
    "units": "units",
    "lots": "lots",
    "rupees": "rupees",
    "percent": "percent of equity",
}
_RISK_UNITS = {"rupees": "rupees", "percent": "percent of equity"}


def size_controls(strategy: type) -> dict[str, Any]:
    """The strategy's `reinvest` and `size_caps` as entries of the kernel's risk dict.
    Raises ConfigError for a `reinvest` that is not True or False, or `size_caps` that
    is not made by `bt.size.caps`."""
    reinvest = getattr(strategy, "reinvest", CONTROLS["reinvest"])
    if not isinstance(reinvest, bool):
        raise ConfigError(f"reinvest takes True or False, got {reinvest!r}")
    caps = getattr(strategy, "size_caps", CONTROLS["size_caps"])
    if caps is not None and not isinstance(caps, SizeCaps):
        raise ConfigError(
            f"size_caps takes bt.size.caps(position=..., leverage=..., risk=...), got {caps!r}"
        )
    return {"reinvest": reinvest, "size_caps": None if caps is None else caps.to_native()}


def _caps_text(caps: dict[str, Any] | None) -> str | None:
    if caps is None:
        return None
    parts = []
    if caps["position"] is not None:
        kind, amount = caps["position"]
        parts.append(f"position {amount:g} {_POSITION_UNITS[kind]}")
    if caps["leverage"] is not None:
        parts.append(f"leverage {caps['leverage']:g}x")
    if caps["risk"] is not None:
        kind, amount = caps["risk"]
        parts.append(f"risk {amount:g} {_RISK_UNITS[kind]}")
    return "; ".join(parts)


def describe(risk: dict[str, Any]) -> dict[str, Any]:
    """`reinvest` and the caps of a resolved risk dict, for the Assumptions block."""
    return {"reinvest": risk["reinvest"], "size_caps": _caps_text(risk["size_caps"])}


__all__ = ["CONTROLS", "describe", "size_controls"]

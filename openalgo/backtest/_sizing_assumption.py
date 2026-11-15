"""The `sizing` line of the Assumptions block: how quantities are rounded and what the
engine default keeps back in cash.

In venue mode (whole shares, the default) an order with no quantity is sized at the
decision close, a bar before it fills, so the kernel keeps a fixed share of its budget in
cash (`VENUE_CASH_BUFFER` in rust/oa_bt/src/sizing/default.rs, read here from the
binding) for the price's move to the next open, as live trading does, and takes the
order's estimated charges off the rest. In fractional mode it is sized at the fill price
and spends the whole budget less the charges. The `bt.size` sizers and `target()` keep
no buffer; `rebalance` keeps its own `cash_buffer`.
"""

from __future__ import annotations

from ._native import native

PERCENT = 100.0
VENUE = "venue"


def cash_buffer_percent() -> float:
    """The engine default's cash buffer in venue mode, in percent (2.0 is 2 percent)."""
    return native.VENUE_CASH_BUFFER * PERCENT


def sizing_assumption(quantity_mode: str) -> str:
    """The Assumptions line for a run's `quantity_mode` ("venue" or "fractional", as
    `_settings.resolve` checked it)."""
    if quantity_mode == VENUE:
        return (
            "whole shares and whole lots; an order with no quantity is sized at the "
            f"decision close and keeps {cash_buffer_percent():g} percent of its budget in "
            "cash for the move to the next open (the engine's cash buffer, the same in live "
            "trading), then takes off its estimated charges; bt.size sizers and target() "
            "keep no buffer, and rebalance keeps its own cash_buffer"
        )
    return (
        "part shares of cash equity, whole lots of derivatives; an order with no quantity "
        "is sized at the fill price and spends its whole budget less its estimated "
        "charges, with no cash buffer"
    )


__all__ = ["cash_buffer_percent", "sizing_assumption"]

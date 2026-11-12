"""Why orders a sizer sized were rejected for margin, as a warning of the `Result`.

A sizer's quantity is checked against margin when the order fills: an order whose margin
the free cash does not cover is rejected with reject code "margin" (12-position-sizing
5; 05-risk-management 7). The sizer is not cut down to what the cash allows. A risk
sizer on intraday bars (`bt.size.atr_risk`, `bt.size.fixed_risk`), whose stop is close
to the price, can ask for many times the capital, so every entry is rejected and the run
makes no trade. The order's reject code alone does not say that the sizer asked for too
much, so the result says it and names the bound, the class attribute `size_caps`.

The engine default never reaches here as the cause: it sizes from the free cash itself.
"""

from __future__ import annotations

from typing import Any

from ._messages import counted

MARGIN = "margin"
# The sizer codes of the orders table (`SizerSpec::code` in the kernel) and how a user
# writes each; the engine default ("default") is left out on purpose (module docstring).
SIZERS = {
    "qty": "bt.size.qty",
    "lots": "bt.size.lots",
    "value": "bt.size.value",
    "percent": "bt.size.percent",
    "fixed_risk": "bt.size.fixed_risk",
    "atr_risk": "bt.size.atr_risk",
    "function": "a sizing function",
}


def margin_rejections(orders: dict[str, list[Any]]) -> list[str]:
    """One warning when orders sized by a sizer were rejected for margin, else none.
    `orders` is the kernel's orders table, a list per column (`sizer`, `reject_code`)."""
    sizers = [
        sizer
        for sizer, code in zip(orders["sizer"], orders["reject_code"], strict=True)
        if code == MARGIN and sizer in SIZERS
    ]
    if not sizers:
        return []
    names = " and ".join(SIZERS[sizer] for sizer in dict.fromkeys(sizers))
    rejected = counted(len(sizers), "order", "orders")
    were = "was" if len(sizers) == 1 else "were"
    return [
        f"{rejected} sized by {names} {were} rejected for margin (reject_code 'margin' in "
        "result.orders): the sizer asked for more than the margin allows, a position whose "
        "margin is more than the free cash. Bound what it gives with the class attribute "
        "size_caps, for example size_caps = bt.size.caps(leverage=1) for a position of at "
        "most 1x the equity"
    ]


__all__ = ["margin_rejections"]

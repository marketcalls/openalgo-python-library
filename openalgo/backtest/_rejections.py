"""Rejected orders as warnings of the `Result`, and as a line of its printout.

A check that refuses an order gives it a reject code (`result.orders["reject_code"]`)
and the run goes on, so a run whose entries were all refused can end with no trade and
nothing else to say why. The result therefore says, for each reject code that occurred,
how many of the run's orders it refused, when the first of them was decided (IST), what
the code means and what to do (`_reject_reasons`). One line per code, however many
orders it refused: an entry cut-off can refuse an order at every bar after it.

Orders a sizer sized and the margin refused get one more clause: a risk sizer
(`bt.size.atr_risk`, `bt.size.fixed_risk`) on intraday bars, whose stop is close to the
price, can ask for many times the capital, and the sizer is not cut down to what the cash
allows (12-position-sizing 5), so the warning names the sizer. The engine default never
causes it: it sizes from the free cash itself.
"""

from __future__ import annotations

from typing import Any

from ._calendar_convert import format_ist
from ._messages import counted
from ._reject_reasons import PRODUCT_RULE, PRODUCT_RULE_CASES, REASONS

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
# The columns of the orders table these messages read.
COLUMNS = ("reject_code", "decided_at", "side", "product", "validity", "sizer")


def _rejected_by_code(orders: dict[str, list[Any]]) -> dict[str, list[dict[str, Any]]]:
    """The rejected orders grouped by reject code, codes in the order they first
    occurred and orders in id order (the kernel's table is in id order, which is the
    order they were decided in). `orders` is the kernel's orders table, a list per
    column."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for values in zip(*(orders[name] for name in COLUMNS), strict=True):
        row = dict(zip(COLUMNS, values, strict=True))
        if row["reject_code"] is not None:
            groups.setdefault(row["reject_code"], []).append(row)
    return groups


def product_rule_case(order: dict[str, Any]) -> str:
    """Which product rule refused `order` (the kernel's submission and fill checks): a
    delivery (CNC) sale opens a short, a GTT on MIS, or else an order for an index."""
    if order["product"] == "CNC" and order["side"] == "SELL":
        return "delivery_short"
    if order["validity"] == "GTT" and order["product"] == "MIS":
        return "gtt_on_mis"
    return "index"


def _sized(rows: list[dict[str, Any]]) -> str:
    """The margin clause naming the sizers that sized refused orders, or "" for none."""
    sizers = [row["sizer"] for row in rows if row["sizer"] in SIZERS]
    if not sizers:
        return ""
    names = " and ".join(SIZERS[sizer] for sizer in dict.fromkeys(sizers))
    which = "" if len(sizers) == len(rows) else f"{len(sizers)} of them "
    return f" ({which}sized by {names}, which asked for more than the margin allows)"


def _explanation(code: str, rows: list[dict[str, Any]]) -> str:
    """What the code means and what to do, for the orders it refused."""
    if code == PRODUCT_RULE:
        cases = dict.fromkeys(product_rule_case(row) for row in rows)
        return ". ".join("; ".join(PRODUCT_RULE_CASES[case]) for case in cases)
    meaning, advice = REASONS[code]
    detail = _sized(rows) if code == MARGIN else ""
    return f"{meaning}{detail}; {advice}"


def reject_warnings(orders: dict[str, list[Any]]) -> list[str]:
    """One warning per reject code that occurred, in the order the codes first occurred:
    for example "4 of 28 orders were rejected with reject_code 'margin'
    (result.orders), the first decided at 2026-10-05 10:15 IST: ...". `orders` is the
    kernel's orders table, a list per column."""
    total = len(orders["reject_code"])
    lines = []
    for code, rows in _rejected_by_code(orders).items():
        were = "was" if len(rows) == 1 else "were"
        first = "decided at" if len(rows) == 1 else "the first decided at"
        lines.append(
            f"{len(rows)} of {counted(total, 'order')} {were} rejected with reject_code "
            f"'{code}' (result.orders), {first} {format_ist(rows[0]['decided_at'])}: "
            f"{_explanation(code, rows)}"
        )
    return lines


def rejected_summary(orders: dict[str, list[Any]]) -> str | None:
    """The printout's line, for example "rejected orders: 4 of 28 (margin 4)"; None when
    no order was rejected."""
    groups = _rejected_by_code(orders)
    if not groups:
        return None
    rejected = sum(len(rows) for rows in groups.values())
    counts = ", ".join(f"{code} {len(rows)}" for code, rows in groups.items())
    return f"rejected orders: {rejected} of {len(orders['reject_code'])} ({counts})"


__all__ = ["product_rule_case", "rejected_summary", "reject_warnings"]

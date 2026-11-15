"""The warning `bt.backtest` ends with when orders of the run were not placed.

Every reject code and skip reason that occurred is described in `result.warnings` and
counted in `print(result)` (`_rejections`, `_skips`), but a reader who prints only a few
metrics sees none of it: a daily delivery strategy whose shorts are all rejected, or one
whose share costs more than its capital in whole shares, just shows no trades. So a run
in which orders were rejected or skipped for a reason the run did not ask for ends with
one `OrdersNotPlacedWarning` that says how many and the most common reason in plain
words, and points to `result.warnings` for the rest. Library code never prints; a warning
reaches the screen and can be filtered by its class.

What the run asked for is left out: orders a rule of the run refused as it intends (the
MIS entry window, a daily loss limit, a cooldown, ...: `_reject_reasons.RUN_RULE_CODES`;
a size cap: `_skip_reasons.RUN_RULE_REASONS`). Their lines in `result.warnings` stay, and
a run with both kinds counts only the others.

It is issued once per run, by `bt.backtest` only (and so by `Result.rerun`, which calls
it): the combinations of `bt.optimize` run through `_pipeline.execute` directly and never
issue it, in this process or in its workers.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ._market import PACKAGE_DIR
from ._messages import counted
from ._reject_reasons import PRODUCT_RULE, PRODUCT_RULE_CASES, REASONS, RUN_RULE_CODES
from ._rejections import product_rule_case
from ._skip_reasons import RUN_RULE_REASONS, SKIP_REASONS
from .errors import OrdersNotPlacedWarning

# The columns of `Result.orders` a reason is read from.
REASON_COLUMNS = ("reject_code", "side", "product", "instrument", "decided_at")
SEE_DETAILS = "See result.warnings."
REJECT_LABEL = "reject_code"
SKIP_LABEL = "skip reason"

# One reason an order was not placed: (label, code, what happened in plain words).
Reason = tuple[str, str, str]


def _rejection(order: dict[str, Any], kinds: Sequence[str]) -> Reason:
    """The reason of a rejected order; the product rule has one meaning per case
    (`_rejections.product_rule_case`)."""
    code = order["reject_code"]
    if code == PRODUCT_RULE:
        meaning = PRODUCT_RULE_CASES[product_rule_case(order, kinds[order["instrument"]])][0]
    else:
        meaning = REASONS[code][0]
    return REJECT_LABEL, code, meaning


def _not_asked_for(
    orders: pd.DataFrame, skipped: pd.DataFrame, kinds: Sequence[str]
) -> list[tuple[Reason, pd.Timestamp]]:
    """(reason, decision instant) of every rejected order, in id order, then of every
    skipped one, in decision order, leaving out the reasons the run asked for."""
    refused = orders["reject_code"].notna() & ~orders["reject_code"].isin(RUN_RULE_CODES)
    found = [
        (_rejection(order, kinds), order["decided_at"])
        for order in orders.loc[refused, list(REASON_COLUMNS)].to_dict("records")
    ]
    for decided_at, reason in zip(skipped["decided_at"], skipped["reason"], strict=True):
        if reason not in RUN_RULE_REASONS:
            found.append(((SKIP_LABEL, reason, SKIP_REASONS[reason][0]), decided_at))
    return found


def _were(count: int) -> str:
    return "was" if count == 1 else "were"


def _how_many(rejected: int, skipped: int, orders: int | None) -> str:
    """How many: "3 of 5 orders were rejected", "2 orders were skipped", or both.
    `orders` counts `Result.orders`, which a skipped order never reaches; None leaves
    the total out, when orders a rule of the run refused are not counted (the others
    were not all placed)."""
    if not rejected:
        return f"{counted(skipped, 'order')} {_were(skipped)} skipped"
    of = (
        counted(rejected, "order")
        if orders is None
        else f"{rejected} of {counted(orders, 'order')}"
    )
    told = f"{of} {_were(rejected)} rejected"
    return f"{told} and {skipped} more {_were(skipped)} skipped" if skipped else told


def unplaced_notice(
    orders: pd.DataFrame, skipped: pd.DataFrame, kinds: Sequence[str]
) -> str | None:
    """The text of the warning, or None when no order was rejected or skipped for a
    reason the run did not ask for. For example "3 of 3 orders were rejected because a
    delivery (CNC) short is not allowed in Indian cash markets (reject_code
    'product_rule'). See result.warnings." With several reasons it names the one behind
    the most orders ("1 of 1 order was rejected and 2 more were skipped, 2 of them
    because ... (skip reason 'rounded_to_zero', the most common reason)"); of reasons
    behind as many, the one decided first. `orders` is `Result.orders`, `skipped`
    `Result.skipped`; `kinds` the `Instrument.kind` of each instrument by the kernel's
    instrument index."""
    found = _not_asked_for(orders, skipped, kinds)
    if not found:
        return None
    counts = Counter(reason for reason, _ in found)
    first: dict[Reason, pd.Timestamp] = {}
    for reason, decided_at in found:
        first[reason] = min(first.get(reason, decided_at), decided_at)
    label, code, meaning = min(counts, key=lambda reason: (-counts[reason], first[reason]))
    count = counts[(label, code, meaning)]
    rejected = sum(n for (kind, _, _), n in counts.items() if kind == REJECT_LABEL)
    left_out = bool(orders["reject_code"].isin(RUN_RULE_CODES).any())
    told = _how_many(rejected, len(found) - rejected, None if left_out else len(orders))
    if count == len(found):
        return f"{told} because {meaning} ({label} '{code}'). {SEE_DETAILS}"
    return (
        f"{told}, {count} of them because {meaning} ({label} '{code}', the most common "
        f"reason). {SEE_DETAILS}"
    )


def announce_unplaced(result: Any) -> None:
    """Issue the notice of a finished run (`unplaced_notice`) as one
    OrdersNotPlacedWarning attributed to the caller's line, or nothing when every order
    was placed or refused as the run asked. `result` is the run's `Result`."""
    kinds = [instrument["kind"] for instrument in result.manifest["instruments"]]
    notice = unplaced_notice(result.orders, result.skipped, kinds)
    if notice is not None:
        warnings.warn(notice, OrdersNotPlacedWarning, skip_file_prefixes=(PACKAGE_DIR,))


__all__ = ["announce_unplaced", "unplaced_notice"]

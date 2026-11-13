"""The warning `bt.backtest` ends with when orders of the run were rejected.

Every reject code that occurred is described in `result.warnings` and counted in
`print(result)` (`_rejections`), but a reader who prints only a few metrics sees none of
it: a daily delivery strategy whose shorts are all rejected just shows no trades. So a
run in which any order was rejected ends with one `RejectedOrdersWarning` that says how
many orders were rejected and the most common reason in plain words, and points to
`result.warnings` for the rest. Library code never prints; a warning reaches the screen
and can be filtered by its class.

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
from ._reject_reasons import PRODUCT_RULE, PRODUCT_RULE_CASES, REASONS
from ._rejections import product_rule_case
from .errors import RejectedOrdersWarning

# The columns of `Result.orders` a reason is read from.
REASON_COLUMNS = ("reject_code", "side", "product", "instrument")
SEE_DETAILS = "See result.warnings."


def _reason(order: dict[str, Any], kinds: Sequence[str]) -> tuple[str, str]:
    """(reject code, what happened in plain words) of a rejected order; the product rule
    has one meaning per case (`_rejections.product_rule_case`)."""
    code = order["reject_code"]
    if code == PRODUCT_RULE:
        case = product_rule_case(order, kinds[order["instrument"]])
        return code, PRODUCT_RULE_CASES[case][0]
    return code, REASONS[code][0]


def rejection_notice(orders: pd.DataFrame, kinds: Sequence[str]) -> str | None:
    """The text of the warning, or None when no order was rejected. For example "3 of 3
    orders were rejected because a delivery (CNC) short is not allowed in Indian cash
    markets (reject_code 'product_rule'). See result.warnings." With several reasons it
    names the one that rejected the most orders ("4 of 4 orders were rejected, 3 of them
    because ... (reject_code 'entry_window', the most common reason)"); of reasons that
    rejected as many, the one whose first order came first in id order. `orders` is
    `Result.orders`; `kinds` the `Instrument.kind` of each instrument by the kernel's
    instrument index."""
    rejected = orders.loc[orders["reject_code"].notna(), list(REASON_COLUMNS)]
    if rejected.empty:
        return None
    # Counter keeps first-seen order among equal counts, which settles a tie.
    reasons = Counter(_reason(order, kinds) for order in rejected.to_dict("records"))
    (code, meaning), count = reasons.most_common(1)[0]
    of_all = f"{len(rejected)} of {counted(len(orders), 'order')}"
    if count == len(rejected):
        were = "was" if count == 1 else "were"
        told = f"{of_all} {were} rejected because {meaning} (reject_code '{code}')"
    else:
        told = (
            f"{of_all} were rejected, {count} of them because {meaning} (reject_code "
            f"'{code}', the most common reason)"
        )
    return f"{told}. {SEE_DETAILS}"


def announce_rejections(result: Any) -> None:
    """Issue the notice of a finished run (`rejection_notice`) as one
    RejectedOrdersWarning attributed to the caller's line, or nothing when no order was
    rejected. `result` is the run's `Result`."""
    kinds = [instrument["kind"] for instrument in result.manifest["instruments"]]
    notice = rejection_notice(result.orders, kinds)
    if notice is not None:
        warnings.warn(notice, RejectedOrdersWarning, skip_file_prefixes=(PACKAGE_DIR,))


__all__ = ["announce_rejections", "rejection_notice"]

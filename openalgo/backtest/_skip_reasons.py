"""What each skip reason of `result.skipped` means and what to do about it: the wording of
the run's skip warnings (`_skips`) and of its end-of-run notice (`_rejection_notice`).

The reasons are the kernel's (`SkipReason::code` in rust/oa_bt/src/sizing/mod.rs). A
skipped order was never placed, so it has no order id and is not in `result.orders`: a
sizer, a sizing function, `target()` or `rebalance()` gave it no quantity. Each entry is
(what happened, what to do).

`RUN_RULE_REASONS` are the skips of a rule the strategy set working as it intends: a size
cap (`size_caps`). The end-of-run notice leaves them out (`_rejection_notice`).
"""

from __future__ import annotations

from types import MappingProxyType

ROUNDED_TO_ZERO = "rounded_to_zero"
CAP = "cap"
RUN_RULE_REASONS = frozenset({CAP})

SKIP_REASONS = MappingProxyType(
    {
        # A sizing function's 0 is told here too: the kernel gives it this reason.
        ROUNDED_TO_ZERO: (
            "the sized quantity was less than one share or, for a derivative, one lot",
            "give the strategy more capital, size with a larger percent, trade a cheaper "
            'share, or pass quantity_mode="fractional" for part shares (a sizing function '
            "that returns 0 skips its order on purpose)",
        ),
        CAP: (
            "a size cap (size_caps) left no room for the order, or a risk cap met an order "
            "with no stop",
            "loosen size_caps, or give the order a stop so a risk cap can size it",
        ),
        "margin": (
            "the free cash left no room for the order",
            "raise capital or lower the size",
        ),
    }
)

__all__ = ["CAP", "ROUNDED_TO_ZERO", "RUN_RULE_REASONS", "SKIP_REASONS"]

"""Skipped orders as warnings of the `Result`, and as a line of its printout.

A sizer, a sizing function, `target()` or `rebalance()` that gives an order no quantity
skips it before it is placed (`result.skipped`: the decision instant, the symbol and the
reason) and the run goes on, so a run whose entries were all skipped can end with no
trade and nothing else to say why: with whole shares, a share dearer than the capital
skips every entry. The result therefore says, for each skip reason that occurred, how
many orders it skipped, when the earliest of them was decided (IST), what happened and
what to do (`_skip_reasons`). One line per reason, however many orders it skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._calendar_convert import format_ist
from ._messages import counted
from ._skip_reasons import SKIP_REASONS

# A skipped order as the run context records it: (decision instant in UTC nanoseconds,
# symbol, skip reason).
Skip = tuple[int, str, str]


def _decisions_by_reason(skipped: Sequence[Skip]) -> dict[str, list[int]]:
    """The decision instants of the skipped orders by reason, reasons in the order they
    first occurred."""
    groups: dict[str, list[int]] = {}
    for decided_at, _, reason in skipped:
        groups.setdefault(reason, []).append(decided_at)
    return groups


def skip_warnings(skipped: Sequence[Skip]) -> list[str]:
    """One warning per skip reason that occurred, in the order the reasons first
    occurred: for example "3 orders were skipped with skip reason 'rounded_to_zero'
    (result.skipped), the first decided at 2026-11-02 15:30 IST: the sized quantity was
    below one share ...; give the strategy more capital, ...". `skipped` is what the
    run context recorded, in decision order."""
    lines = []
    for reason, decisions in _decisions_by_reason(skipped).items():
        meaning, advice = SKIP_REASONS[reason]
        were = "was" if len(decisions) == 1 else "were"
        first = "decided at" if len(decisions) == 1 else "the first decided at"
        lines.append(
            f"{counted(len(decisions), 'order')} {were} skipped with skip reason '{reason}' "
            f"(result.skipped), {first} {format_ist(min(decisions))}: {meaning}; {advice}"
        )
    return lines


def skipped_summary(skipped: Sequence[Skip]) -> str | None:
    """The printout's line, for example "skipped orders: 249 (rounded_to_zero 249)";
    None when no order was skipped."""
    groups = _decisions_by_reason(skipped)
    if not groups:
        return None
    counts = ", ".join(f"{reason} {len(decisions)}" for reason, decisions in groups.items())
    return f"skipped orders: {len(skipped)} ({counts})"


__all__ = ["skip_warnings", "skipped_summary"]

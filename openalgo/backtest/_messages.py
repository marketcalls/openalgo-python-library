"""The wording that messages a user sees share: a count with its noun in the right
number, and the error for a part of the engine this version does not run.

A message never cites an internal planning document (a requirement, decision or phase
number): it says what is not supported and what the user can do instead, as the
kernel's own not-built errors do (`rust/oa_bt/src/model.rs`, `write_not_supported`).
"""

from __future__ import annotations

# (what is not supported, what to do instead) of a run of several strategies, which both
# `bt.backtest` and the several-strategy views of a `Result` refuse.
SEVERAL_STRATEGIES = (
    "a run of several strategies",
    "run each strategy with its own bt.backtest call",
)


def counted(count: int, singular: str, plural: str | None = None) -> str:
    """`count` followed by `singular` when it is exactly 1, else by `plural` (by default
    `singular` with an "s"): "1 row", "3 rows", "0 rows". Give `plural` for a noun that
    does not take an "s", or for a phrase whose verb agrees ("row has", "rows have")."""
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def not_supported(feature: str, instead: str) -> NotImplementedError:
    """The NotImplementedError for `feature`, which this version does not run, saying
    what the user can do `instead`."""
    return NotImplementedError(f"{feature} is not supported in this version: {instead}")


__all__ = ["SEVERAL_STRATEGIES", "counted", "not_supported"]

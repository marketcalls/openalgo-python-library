"""The wording that messages a user sees share: a count with its noun in the right
number, names listed as a sentence lists them, the error for a part of the engine this
version does not run, and where a strategy of a portfolio run finds its full set of
overfitting checks.

A message never cites an internal planning document (a requirement, decision or phase
number): it says what is not supported and what the user can do instead, as the
kernel's own not-built errors do (`rust/oa_bt/src/model.rs`, `write_not_supported`).
"""

from __future__ import annotations

from collections.abc import Sequence

# (what is not supported, what to do instead) of the views of several strategies on a
# single strategy's `Result`, which a list of strategies gives in its `PortfolioResult`.
SEVERAL_STRATEGIES = (
    "result.strategy(), result.combined or result.correlation on a single strategy's Result",
    "pass a list of strategies to bt.backtest: its PortfolioResult has strategy(name), "
    "the account's tables and the correlation of the strategies",
)
#: Where one strategy of a portfolio run finds the full set of overfitting checks: its own
#: `Result.robustness()` is refused (a rerun of it alone would not face the account it
#: shared), so a run of it alone. A failed quick check's warning in the portfolio and the
#: combined page's Overfitting checks section end with it.
ALONE_POINTER = (
    "result.robustness() on a run of the strategy alone with bt.backtest runs the full set "
    "of checks, with charts"
)


def counted(count: int, singular: str, plural: str | None = None) -> str:
    """`count` followed by `singular` when it is exactly 1, else by `plural` (by default
    `singular` with an "s"): "1 row", "3 rows", "0 rows". Give `plural` for a noun that
    does not take an "s", or for a phrase whose verb agrees ("row has", "rows have")."""
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def spoken_list(names: Sequence[str]) -> str:
    """Names as a sentence says them: "A", "A and B", "A, B and C"."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def unsupported(feature: str, instead: str) -> str:
    """The message for `feature`, which this version does not run, saying what the user
    can do `instead`; for an error whose type says more than NotImplementedError would."""
    return f"{feature} is not supported in this version: {instead}"


def not_supported(feature: str, instead: str) -> NotImplementedError:
    """The NotImplementedError for `feature`, which this version does not run, saying
    what the user can do `instead`."""
    return NotImplementedError(unsupported(feature, instead))


def one_strategy_only(verb: str) -> NotImplementedError:
    """The NotImplementedError of `verb` ("bt.optimize"), which takes one strategy in this
    version, given a list of them."""
    return not_supported(
        f"{verb} with a list of strategies",
        f"call {verb} for each strategy of the list, one at a time",
    )


__all__ = [
    "SEVERAL_STRATEGIES",
    "counted",
    "not_supported",
    "one_strategy_only",
    "spoken_list",
    "unsupported",
]

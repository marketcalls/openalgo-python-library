"""The verdicts of the robustness checks and `CheckResult`, one check's result.

Wording rule (design 17, section 0, as amended on 2026-11-18): passing a check proves
nothing on its own; failing one says the edge did not hold up under that check, so it
may be fitted to the past or too small to tell apart from the noise of the data tested,
and no single failure is called proof of overfitting. The trade-list checks cannot see a
strategy picked as the best of many tried, and the rule names the checks that can. Nor
can the checks that rerun the strategy with its parameters as chosen: they judge those
settings, not the search that chose them (an EMA crossover picked as the best of 100
settings on a random walk passed noise in 3 of 3 seeds and the permutation test in 1 of
3), so the rule says so too. A check with too little to judge says "not judged" and why,
and never passes. A check
whose rule needs a limit the user did not give is "informational": it shows its figures
and judges nothing.

The rule is made of sentences that the quick checks' reading (`_quick.QUICK_READING`)
shares, so the two never drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

PASSED = "passed"
FAILED = "failed"
NOT_JUDGED = "not judged"
INFORMATIONAL = "informational"
NOT_BUILT = "not built"
#: Every verdict, in the order a scorecard counts them.
VERDICTS = (PASSED, FAILED, NOT_JUDGED, INFORMATIONAL, NOT_BUILT)
#: What a pass shows.
PASSING = "Passing a check proves nothing on its own."
#: What a failure shows, and what it does not.
FAILING = (
    "Failing one says the edge did not hold up under that check, so it may be fitted to "
    "the past or too small to tell apart from the noise of the data tested; a single "
    "failure is not proof of overfitting."
)
#: What the trade-list checks cannot see, and the checks that can.
PICKED = (
    "The checks of the trade list (random out of sample, trade reshuffle, trade resample "
    "and variance) cannot see a strategy picked as the best of many tried; the parameter "
    "neighbourhood, the noise-robust optimization and the optimizer's deflated Sharpe "
    "ratio and probability of backtest overfitting can."
)
#: What the checks that rerun the chosen settings cannot see either (module docstring).
FIXED = (
    "Nor can the checks that rerun the strategy with its parameters as chosen (versus "
    "random, permutation, noise, other markets and synthetic stress): they judge the "
    "settings chosen, not the search that chose them."
)
#: What a check with too little to judge says.
UNJUDGED = "A check with too little to judge says not judged and why, and never passes."
#: The wording rule every full report states at its top.
WORDING_RULE = " ".join((PASSING, FAILING, PICKED, FIXED, UNJUDGED))


@dataclass(frozen=True, eq=False)
class Outcome:
    """What a check computes: its verdict, its figure, its rule and a note (plain text),
    a table with a row per run, the summary figures and the blocks its section draws
    (`_blocks`)."""

    verdict: str
    figure: str
    rule: str
    note: str
    table: pd.DataFrame
    summary: Mapping[str, float]
    blocks: tuple = ()


def not_judged(rule: str, why: str) -> Outcome:
    """The outcome of a check with too little to judge: no figure, no runs, `why` as
    its note."""
    return Outcome(NOT_JUDGED, "n/a", rule, why, pd.DataFrame(), {})


@dataclass(frozen=True, eq=False)
class CheckResult:
    """One check of a robustness report.

    name: the check's name ("resample"); title: its title ("Trade resample");
    question: what it answers; verdict: "passed", "failed", "not judged" or
    "informational"; figure: the figure it judged, as text; rule: its pass rule, as
    text; note: how it was measured, or why it was not judged.
    table: a row per run (an assignment, an order, a path, a rerun), empty when not
    judged; other markets has a row per symbol and synthetic stress a row per rule.
    summary: its figures by name; fractions are fractions (0.95 is 95 percent), money
    is rupees, and drawdowns are zero or negative.
    settings: the settings it ran with (its defaults with the check's dict applied);
    seed: the seed of its random stream, or None for a check that draws nothing.
    """

    name: str
    title: str
    question: str
    verdict: str
    figure: str
    rule: str
    note: str
    table: pd.DataFrame = field(repr=False)
    summary: Mapping[str, float] = field(repr=False)
    settings: Mapping[str, Any] = field(repr=False)
    seed: int | None = None

    @property
    def passed(self) -> bool:
        """True only for the verdict "passed"."""
        return self.verdict == PASSED


__all__ = [
    "FAILED",
    "FAILING",
    "FIXED",
    "INFORMATIONAL",
    "NOT_BUILT",
    "NOT_JUDGED",
    "PASSED",
    "PASSING",
    "PICKED",
    "UNJUDGED",
    "VERDICTS",
    "WORDING_RULE",
    "CheckResult",
    "Outcome",
    "not_judged",
]

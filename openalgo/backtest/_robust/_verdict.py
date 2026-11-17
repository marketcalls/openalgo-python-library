"""The verdicts of the robustness checks and `CheckResult`, one check's result.

Wording rule (design 17, section 0): passing a check proves nothing on its own; failing
one is strong evidence of overfitting. A check with too little to judge says "not
judged" and why, and never passes. A check whose rule needs a limit the user did not
give is "informational": it shows its figures and judges nothing.
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
#: The wording rule every report states at its top.
WORDING_RULE = (
    "Passing a check proves nothing on its own; failing one is strong evidence of "
    "overfitting. A check with too little to judge says not judged and why, and never "
    "passes."
)


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
    table: a row per run (an assignment, an order, a path), empty when not judged.
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
    _blocks: tuple = field(default=(), repr=False)

    @property
    def passed(self) -> bool:
        """True only for the verdict "passed"."""
        return self.verdict == PASSED


__all__ = [
    "FAILED",
    "INFORMATIONAL",
    "NOT_BUILT",
    "NOT_JUDGED",
    "PASSED",
    "VERDICTS",
    "WORDING_RULE",
    "CheckResult",
    "Outcome",
    "not_judged",
]

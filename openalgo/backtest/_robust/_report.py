"""`RobustnessReport`: what `bt.robustness` and `Result.robustness` return.

    report.verdicts          # DataFrame: check, question, figure, rule, verdict, note
    report.check("resample") # one check's CheckResult: .table, .summary, .verdict
    report.passed            # every judged check passed (and at least one was judged)
    report.plan              # the plan reported before the checks ran, untimed
    report.tearsheet(path)   # the HTML robustness report

Wording rule (design 17, section 0): passing a check proves nothing on its own; failing
one is strong evidence of overfitting; "not judged" never passes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import pandas as pd

from ..errors import ConfigError
from ._registry import UNITS, find, not_built
from ._request import Request
from ._scorecard import counts_line
from ._verdict import FAILED, NOT_BUILT, PASSED, CheckResult


@dataclass(frozen=True)
class RunFacts:
    """What a report says about the run behind it: the strategy's name, its first and
    last session (IST dates; None without a session), its sessions, its closed trades
    and the positions open at the end, its capital in rupees, and its assumption and
    warning lines ("key: value", "warning: ...")."""

    strategy: str
    first_session: pd.Timestamp | None
    last_session: pd.Timestamp | None
    sessions: int
    closed: int
    open_trades: int
    capital: float
    lines: tuple[str, ...]


class RobustnessReport:
    """The result of the robustness checks of one run; see the module docstring. Made by
    `bt.robustness` and `Result.robustness`; `blocks` holds what each check's section of
    the HTML report draws, by check name."""

    def __init__(
        self,
        facts: RunFacts,
        request: Request,
        results: tuple[CheckResult, ...],
        plan: str,
        blocks: Mapping[str, tuple],
    ) -> None:
        self._facts = facts
        self._request = request
        self._results = {result.name: result for result in results}
        self._plan = plan
        self._blocks = dict(blocks)

    @property
    def plan(self) -> str:
        """What was reported before the checks ran, without how long the backtest
        took."""
        return self._plan

    @property
    def verdicts(self) -> pd.DataFrame:
        """One row per listed check, indexed by its number in report order: check (its
        name), question, figure, rule, verdict and note (plain text). A check this
        version does not run has the verdict "not built" and names the unit that
        brings it."""
        rows = []
        for check in self._request.listed:
            if check.built:
                result = self._results[check.name]
                cells = (result.figure, result.rule, result.verdict, result.note)
            else:
                note = f"Comes with robustness unit {check.unit}, {UNITS[check.unit]}."
                cells = ("", "", NOT_BUILT, note)
            rows.append((check.number, check.name, check.question, *cells))
        columns = ["number", "check", "question", "figure", "rule", "verdict", "note"]
        return pd.DataFrame(rows, columns=columns).set_index("number")

    def _counts(self) -> dict[str, int]:
        """How many listed checks had each verdict."""
        return dict(Counter(self.verdicts["verdict"]))

    def check(self, name: str) -> CheckResult:
        """The result of the check named `name`. Raises ConfigError for an unknown name or
        a check this report did not run, and NotImplementedError for a check this version
        does not run."""
        check = find(name)
        if not check.built:
            raise not_built(check)
        if name not in self._results:
            raise ConfigError(f"this report did not run {name}; it ran {list(self._results)}")
        return self._results[name]

    @property
    def passed(self) -> bool:
        """True when at least one check was judged and every judged check passed. A
        check that was not judged, informational or not built is not judged."""
        judged = [r.verdict for r in self._results.values() if r.verdict in (PASSED, FAILED)]
        return bool(judged) and all(verdict == PASSED for verdict in judged)

    def tearsheet(self, path: str | PathLike | None = None) -> str:
        """The HTML robustness report (`_html`): returned as a string, or written to
        `path` as UTF-8 and the path returned."""
        from ._html import render

        results = tuple(self._results.values())
        page = render(self._facts, self._request, results, self._blocks, self._counts(), self._plan)
        if path is None:
            return page
        target = Path(path)
        target.write_text(page, encoding="utf-8")
        return str(target)

    def __repr__(self) -> str:
        return f"RobustnessReport({self._facts.strategy}: {counts_line(self._counts())})"


__all__ = ["RobustnessReport", "RunFacts"]

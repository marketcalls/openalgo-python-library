"""The quick overfitting checks of every backtest (design 17, section 4): the trade-list
checks (random out of sample, trade reshuffle, trade resample and variance) at
`QUICK_RUNS` runs with the default seed, settings and objective, from the trade list
alone, with no rerun. They never change a figure of the run.

The standard tearsheet shows them in its "Overfitting checks" section, `Result.summary()`
in one line, and `Result.warnings` with a line for each one that fails. A run with fewer
closed trades than the checks need says "too few trades to judge" instead.

What a quick check's verdict means (`QUICK_READING`): these checks read the run's own
trades, so a failure says the edge is fragile or small next to the noise of those
trades, and none of them can see a strategy picked as the best of many tried. At their
default rules an honest edge with an annual Sharpe ratio near 1 over a few years fails
random out of sample and variance more often than not, so their failure is not worded
as strong evidence of overfitting, the reading of the full report's checks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..report._performance import Performance
from ._assess import run_check
from ._request import DEFAULT_OBJECTIVE, DEFAULT_SEED, build_request
from ._scorecard import Row, counts_line
from ._trade_common import MIN_TRADES
from ._trades import trade_run
from ._verdict import FAILED, CheckResult

#: The runs of each quick check (design 17, section 4).
QUICK_RUNS = 200
#: Where the full set of checks is.
POINTER = "result.robustness() runs the full set of checks, with charts"
#: How to read the quick checks' verdicts (module docstring).
QUICK_READING = (
    "Passing proves nothing on its own. These checks read the run's own trades: a failure "
    "says the edge is fragile or small next to the noise of those trades, and none of them "
    "can see a strategy picked as the best of many tried. A check with too little to judge "
    "says not judged and why, and never passes."
)


@dataclass(frozen=True, eq=False)
class QuickChecks:
    """The quick checks of one run: their results in report order, the closed trades,
    and the runs and seed they used."""

    results: tuple[CheckResult, ...]
    closed: int
    runs: int
    seed: int

    @property
    def too_few_trades(self) -> bool:
        """Whether the run has fewer closed trades than the checks need to judge."""
        return self.closed < MIN_TRADES

    def counts(self) -> dict[str, int]:
        """How many quick checks had each verdict."""
        return dict(Counter(result.verdict for result in self.results))

    def failed(self) -> list[CheckResult]:
        """The quick checks that failed, in report order."""
        return [result for result in self.results if result.verdict == FAILED]

    def summary_line(self) -> str:
        """One line for `Result.summary()`: "too few trades to judge (12 closed, 30
        needed)", or how many quick checks had each verdict with the failed ones named:
        "quick checks: 2 passed, 1 failed (variance), 1 informational; see
        result.robustness()"."""
        if self.too_few_trades:
            return f"too few trades to judge ({self.closed} closed, {MIN_TRADES} needed)"
        counted = counts_line(self.counts())
        failed = [result.name for result in self.failed()]
        if failed:
            counted = counted.replace(FAILED, f"{FAILED} ({', '.join(failed)})", 1)
        return f"quick checks: {counted}; see result.robustness()"

    def warnings(self) -> list[str]:
        """A line for `Result.warnings` for each quick check that failed, saying what a
        failure of the trade list shows (`QUICK_READING`)."""
        return [
            f"Overfitting check failed, {result.title.lower()}: {result.figure}, where the rule "
            f"is {result.rule}. The edge is fragile or small next to the noise of the run's "
            f"own trades; {POINTER}"
            for result in self.failed()
        ]

    def rows(self) -> list[Row]:
        """The scorecard rows of the quick checks."""
        from ._registry import BY_NAME

        return [
            Row(BY_NAME[r.name].number, r.title, r.question, r.figure, r.rule, r.verdict)
            for r in self.results
        ]


def quick_checks(
    performance: Performance, *, runs: int = QUICK_RUNS, seed: int = DEFAULT_SEED
) -> QuickChecks:
    """The quick checks of `performance` (module docstring)."""
    request = build_request(
        checks="quick",
        objective=DEFAULT_OBJECTIVE,
        runs=runs,
        seed=seed,
        workers=None,
        others=None,
        check_settings={},
        ranges={},
    )
    trades = trade_run(performance)
    results = tuple(run_check(check, trades, request).result for check in request.to_run)
    return QuickChecks(results, trades.closed, runs, seed)


__all__ = ["POINTER", "QUICK_READING", "QUICK_RUNS", "QuickChecks", "quick_checks"]

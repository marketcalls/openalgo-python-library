"""The reports of a `PortfolioResult`: its plain-text summary and its combined tearsheet.

The summary is `Result.summary()`'s for the account (`_summary`): its net profit, return,
maximum drawdown, closed trades and benchmark, with the quick overfitting checks of each
strategy's own trades on a line each, and none on the account's trades together (the
strategies' trades overlap in time, so they are not one sequence the checks can read).
A run that states its scope in its assumptions (`SCOPE_NOTE`: a restarted live session's
portfolio, which covers its start only) has that line first; a backtest states none.

The tearsheet is the account's page drawn by the report's own machinery
(`report.tearsheet`), through the adapter `Report`: every built-in section from the
account's `Performance`, the Strategies section (each strategy's `performance`, the
mapping `report.tearsheet` reads as `strategies`), each strategy's quick checks in a row
of the Overfitting checks section, the Portfolio section's four parts (fed the portfolio
itself, `report._dashboard.portfolio`) and the Notes with the portfolio's assumptions and
warnings. A custom layout's callables receive the `PortfolioResult`. Each strategy's own
page is `pf.strategy(name).tearsheet()`, unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._messages import spoken_list
from ._result_report import benchmark_name
from ._summary import summary_text
from .report._dashboard.portfolio_tables import REFUSED_NOTE
from .report._performance import Performance
from .report.custom import Layout
from .report.tearsheet import write_tearsheet

if TYPE_CHECKING:
    from .portfolio_result import PortfolioResult

#: The summary's overfitting line of a portfolio, over a line per strategy.
PER_STRATEGY = "each strategy's own trades below; none of the account's trades together"
#: A strategy's line when it has no equity point, so no performance to check.
NOTHING_TO_CHECK = "no equity point, nothing to check"
#: The start of the combined page's title, before the strategies' names.
TITLE = "Portfolio"
#: The key of the Assumptions line in which a run states what it covers when that is less
#: than the run (a restarted live session's portfolio covers its start only:
#: `_live._result_portfolio.restart_scope`); the summary prints it as its first line.
SCOPE_NOTE = "scope"


@dataclass(frozen=True)
class Report:
    """What the combined tearsheet reads (`report.tearsheet`'s attributes): the
    account's performance, the page's title, the assumption lines, each strategy's
    performance by name (the Strategies section, with two or more), the benchmark's name
    (None without one) and each strategy's quick overfitting checks by name (None for a
    strategy with no equity point)."""

    performance: Performance
    name: str
    assumptions: list[str]
    strategies: Mapping[str, Performance]
    benchmark_name: str | None
    overfitting: Mapping[str, Any]


def _benchmark(portfolio: PortfolioResult) -> str | None:
    """The account's benchmark as the page names it: the first strategy's, when the
    account has one (`_result_report.benchmark_name`)."""
    if portfolio.performance.benchmark_returns is None:
        return None
    first = next(iter(portfolio.results.values()))
    return benchmark_name(first.manifest)


def _quick_checks(portfolio: PortfolioResult) -> dict[str, Any]:
    """Each strategy's quick overfitting checks, as its own `Result` computed them."""
    return {name: result._quick for name, result in portfolio.results.items()}


def report(portfolio: PortfolioResult) -> Report:
    """The adapter the combined tearsheet reads (module docstring)."""
    # The words of an account that refused nothing are its refusals card's, not a line.
    notes = portfolio.assumptions.items()
    lines = [f"{key}: {value}" for key, value in notes if key != REFUSED_NOTE]
    lines += [f"warning: {text}" for text in portfolio.warnings]
    return Report(
        performance=portfolio.performance,
        name=f"{TITLE}: {spoken_list(list(portfolio.strategies))}",
        assumptions=lines,
        strategies={name: result.performance for name, result in portfolio.results.items()},
        benchmark_name=_benchmark(portfolio),
        overfitting=_quick_checks(portfolio),
    )


def tearsheet(
    portfolio: PortfolioResult,
    path: str | os.PathLike | None,
    *,
    mode: str,
    log_scale: bool,
    layout: Layout | None,
    title: str | None = None,
) -> str:
    """The combined tearsheet of `portfolio` (module docstring) in `mode`, or of a custom
    `layout` whose blocks' callables receive `portfolio`, titled `title` (by default
    "Portfolio:" and the strategies' names); returned, or written to `path` (and the path
    returned)."""
    return write_tearsheet(
        report(portfolio),
        path,
        subject=portfolio,
        title=title,
        mode=mode,
        log_scale=log_scale,
        layout=layout,
        portfolio=portfolio,
    )


def summary(portfolio: PortfolioResult) -> str:
    """The plain-text summary of `portfolio` (module docstring)."""
    details = [
        (name, NOTHING_TO_CHECK if quick is None else quick.verdict_line())
        for name, quick in _quick_checks(portfolio).items()
    ]
    text = summary_text(portfolio.metrics, _benchmark(portfolio), PER_STRATEGY, details)
    scope = portfolio.assumptions.get(SCOPE_NOTE)
    return text if scope is None else f"{scope[:1].upper()}{scope[1:]}\n{text}"


__all__ = [
    "NOTHING_TO_CHECK",
    "PER_STRATEGY",
    "SCOPE_NOTE",
    "TITLE",
    "Report",
    "report",
    "summary",
    "tearsheet",
]

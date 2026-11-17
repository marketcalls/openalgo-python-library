"""Running the checks of a request on one run, and the plan reported before they run.

Each check runs with its resolved settings (`runs` None takes the report's runs) and the
seed of its own stream (`_random.check_seed`). The plan says what will run, on how many
closed trades, which checks are not built in this version, and what reruns the checks
need: none, because every check this version runs reads the run's trade list, so no
worker process starts. It is reported through `progress` (a function taking one line of
text) or, without one, the `openalgo.backtest.robustness` logger at INFO, before any
check runs, with how long the backtest took when the report ran it. The report keeps the
plan without that time, so the same run and seed give the same page.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .._messages import counted
from ..errors import ConfigError
from . import _random
from . import _words as words
from ._registry import CHECKS, CheckSpec
from ._report import RobustnessReport, RunFacts
from ._request import Request
from ._trades import TradeRun, trade_run
from ._verdict import CheckResult

LOGGER = logging.getLogger("openalgo.backtest.robustness")


@dataclass(frozen=True, eq=False)
class Checked:
    """A check that ran: its `result`, and the `blocks` its section of the HTML report
    draws (`_blocks`), which stay out of the public `CheckResult`."""

    result: CheckResult
    blocks: tuple


def run_check(check: CheckSpec, trades: TradeRun, request: Request) -> Checked:
    """The result of `check` on `trades` under `request`, with what its section draws."""
    settings = dict(request.settings[check.name])
    runs = settings["runs"] if settings["runs"] is not None else request.runs
    settings["runs"] = runs
    seed = _random.check_seed(request.seed, check.number)
    outcome = check.run(trades, settings, runs=runs, seed=seed, objective=request.objective)
    result = CheckResult(
        check.name,
        check.title,
        check.question,
        outcome.verdict,
        outcome.figure,
        outcome.rule,
        outcome.note,
        outcome.table,
        dict(outcome.summary),
        settings,
        seed,
    )
    return Checked(result, outcome.blocks)


def plan_text(strategy: str, request: Request, trades: TradeRun) -> str:
    """The plan of the checks (module docstring), without the backtest's duration."""
    names = [check.name for check in request.to_run]
    text = (
        f"bt.robustness: {counted(len(names), 'check')} of {len(CHECKS)} for {strategy} "
        f"({', '.join(names)}), {words.whole(request.runs, 'run')} each from seed "
        f"{request.seed}, on {words.whole(trades.closed, 'closed trade')} over "
        f"{words.whole(len(trades.sessions), 'session')}. They read the run's trade list, so "
        "the backtest is not rerun and no worker process starts."
    )
    later = [check.name for check in request.listed if not check.built]
    if later:
        text += (
            f" {counted(len(later), 'check is', 'checks are')} not built in this version and "
            f"listed as such: {', '.join(later)}."
        )
    return text


def _facts(result: Any, trades: TradeRun) -> RunFacts:
    """The run behind a report: its strategy, sessions and trades, and the run's own
    assumptions and warnings (not the quick checks', which this report supersedes)."""
    sessions = trades.sessions
    lines = [f"{key}: {value}" for key, value in result.assumptions.items()]
    lines += [f"warning: {text}" for text in result._warnings]
    return RunFacts(
        strategy=result.manifest["strategy"]["name"],
        first_session=sessions[0] if len(sessions) else None,
        last_session=sessions[-1] if len(sessions) else None,
        sessions=len(sessions),
        closed=trades.closed,
        open_trades=trades.open_trades,
        capital=trades.capital,
        lines=tuple(lines),
    )


def assess(
    result: Any,
    request: Request,
    progress: Callable[[str], Any] | None,
    seconds: float | None,
) -> RobustnessReport:
    """The report of `request` on `result` (a `Result`); `seconds` is how long the
    backtest took when the report ran it, None for an existing result. Raises
    ConfigError for a `progress` that is not a function, and what `result.performance`
    raises."""
    if progress is not None and not callable(progress):
        raise ConfigError(f"progress takes a function of one line of text, got {progress!r}")
    trades = trade_run(result.performance)
    facts = _facts(result, trades)
    plan = plan_text(facts.strategy, request, trades)
    timed = plan if seconds is None else f"{plan} The backtest took {seconds:.2f} s."
    (progress or LOGGER.info)(timed)
    checked = [run_check(check, trades, request) for check in request.to_run]
    results = tuple(done.result for done in checked)
    blocks = {done.result.name: done.blocks for done in checked}
    return RobustnessReport(facts, request, results, plan, blocks)


__all__ = ["LOGGER", "Checked", "assess", "plan_text", "run_check"]

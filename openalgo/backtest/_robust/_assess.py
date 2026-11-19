"""Running the checks of a request on one run, and the plan reported before they run.

Each check runs with its resolved settings (`runs` None takes the report's runs) and the
seed of its own stream (`_random.check_seed`). The trade-list checks read the run's trade
list (`_trades.TradeRun`); the bar checks and the searching checks rerun the backtest
(`_bar_setup`), so when the request lists one the run's bars are read again first, and
the trade-list checks then have the run's prices too (`TradeRun.prices`, the chart of
random out of sample).

The plan says what will run, on how many closed trades, and what reruns the checks that
rerun plan. It is reported through `progress` (a
function taking one line of text) or, without one, the `openalgo.backtest.robustness`
logger at INFO, before any check runs, with how long the backtest took when the report
ran it and an estimate of the reruns timed on one rerun (`_bar_setup`). The report keeps
the plan without those times and the worker count, so the same run and seed give the
same page.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .._messages import counted
from ..errors import ConfigError
from . import _random
from . import _words as words
from ._bar_setup import BarContext, bar_context
from ._registry import CHECKS, CheckSpec
from ._report import RobustnessReport, RunFacts
from ._request import Request
from ._rerun_plan import plan_words
from ._trades import TradeRun, trade_run
from ._verdict import CheckResult

LOGGER = logging.getLogger("openalgo.backtest.robustness")


@dataclass(frozen=True, eq=False)
class Checked:
    """A check that ran: its `result`, and the `blocks` its section of the HTML report
    draws (`_blocks`), which stay out of the public `CheckResult`."""

    result: CheckResult
    blocks: tuple


def run_check(check: CheckSpec, subject: Any, request: Request) -> Checked:
    """The result of `check` on `subject` (the run's `TradeRun` for a trade-list check,
    its `_bar_common.BarRun` for a bar check) under `request`, with what its section
    draws."""
    settings = dict(request.settings[check.name])
    runs = request.runs_of(check)
    if "runs" in settings:
        settings["runs"] = runs
    seed = _random.check_seed(request.seed, check.number)
    outcome = check.run(subject, settings, runs=runs, seed=seed, objective=request.objective)
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


def plan_text(
    strategy: str, request: Request, trades: TradeRun, context: BarContext | None = None
) -> str:
    """The plan of the checks (module docstring), without times or the worker count."""
    names = [check.name for check in request.listed]
    reruns = plan_words(None, True, 0.0, 0.0) if context is None else context.words()
    return (
        f"bt.robustness: {counted(len(names), 'check')} of {len(CHECKS)} for {strategy} "
        f"({', '.join(names)}), {words.whole(request.runs, 'run')} each from seed "
        f"{request.seed}, on {words.whole(trades.closed, 'closed trade')} over "
        f"{words.whole(len(trades.sessions), 'session')}. {reruns}"
    )


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


def _timed(plan: str, seconds: float | None, context: BarContext | None) -> str:
    """The plan as reported: with the backtest's duration and the reruns' estimate."""
    timed = plan if seconds is None else f"{plan} The backtest took {seconds:.2f} s."
    if context is not None:
        timed += context.timing()
    return timed


def _subject(
    check: CheckSpec, trades: TradeRun, result: Any, request: Request, context: Any
) -> Any:
    """What `check` reads: the trade list, the run as the bar checks read it, or the run
    as the searching checks read it."""
    if check.trade_list:
        return trades
    return context.search if check.searches else context.subject(trades, result, request)


def assess(
    result: Any,
    request: Request,
    progress: Callable[[str], Any] | None,
    seconds: float | None,
) -> RobustnessReport:
    """The report of `request` on `result` (a `Result`); `seconds` is how long the
    backtest took when the report ran it, None for an existing result. Raises
    ConfigError for a `progress` that is not a function, what `result.performance`
    raises, and for a bar check what reading the run's bars again and rerunning it
    raise (`_bar_setup.bar_context`)."""
    if progress is not None and not callable(progress):
        raise ConfigError(f"progress takes a function of one line of text, got {progress!r}")
    trades = trade_run(result.performance)
    facts = _facts(result, trades)
    context = None
    if request.reruns:
        context = bar_context(result, request, trades)
        trades = dataclasses.replace(trades, prices=context.bars.session_closes(trades.sessions))
    plan = plan_text(facts.strategy, request, trades, context)
    (progress or LOGGER.info)(_timed(plan, seconds, context))
    closing = context.reruns if context is not None else contextlib.nullcontext()
    with closing:
        checked = [
            run_check(check, _subject(check, trades, result, request, context), request)
            for check in request.listed
        ]
    results = tuple(done.result for done in checked)
    blocks = {done.result.name: done.blocks for done in checked}
    return RobustnessReport(facts, request, results, plan, blocks)


__all__ = ["LOGGER", "Checked", "assess", "plan_text", "run_check"]

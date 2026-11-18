"""Robustness and overfitting checks of a backtest (design 17): `bt.robustness`,
`Result.robustness`, `RobustnessReport` and `CheckResult`.

    report = bt.robustness(EmaCross, client, capital=100_000, start=..., end=...,
                           checks="all", objective="sharpe", runs=1000, seed=0)
    report = result.robustness(checks="all")   # the same run, its arguments reused
    report.verdicts, report.check("resample"), report.passed, report.tearsheet(path)

Private: `import openalgo.backtest` does not import it; `bt.robustness`,
`Result.robustness`, the tearsheet, `Result.summary()` and `Result.warnings` import it
when they need it. Every check is deterministic for a seed on every platform: the bar
checks rerun the backtest, an ordinary run of the deterministic kernel, on bars that are
a pure function of the run's bars and the seed.

| Module | Responsibility |
|---|---|
| `_registry` | The seventeen checks in report order, the ones built, the not-built error |
| `_request` | The checks, objective, runs, seed, workers and settings, checked first |
| `_setting` | A check's settings: defaults, the values each takes, applying a check's dict |
| `_assess` | Running the checks on one run, and the plan reported before they run |
| `_quick` | The quick checks of every backtest: tearsheet, summary line and warnings |
| `_report` | `RobustnessReport` |
| `_verdict` | The verdicts, the wording rule and `CheckResult` |
| `_trades` | What the trade-list checks read: closed trades, sessions, capital |
| `_random` | Seeded random draws, the same on every platform |
| `_paths` | Equity paths of booked trades, their drawdowns and fans |
| `_trade_common` | Settings and the summary block the trade-list checks share |
| `_part_objective` | The objective measured on a part of the trades |
| `_check_random_oos` | Random out of sample |
| `_check_reshuffle` | Trade reshuffle |
| `_check_resample` | Trade resample |
| `_check_variance` | Variance |
| `_ohlc` | Sessions of bars, prices on the tick grid, a bar made valid |
| `_generate_permuted` | Permuted bars |
| `_generate_noisy` | Noisy bars |
| `_stress_rules` | The rules that keep a synthetic stress candidate |
| `_run_bars` | The bars a run read, and the arguments of its reruns |
| `_rerun_one` | One rerun on given bars, and what a check keeps of it |
| `_random_entries` | Random entries with the run's trades |
| `_reruns` | Families of reruns, in this process or worker processes, kept once |
| `_bar_setup` | The bar checks' setup: the bars, a first rerun, the plan's reruns |
| `_bar_common` | What the bar checks share: the run they read, break-even, rerun sets |
| `_check_vs_random` | Versus random |
| `_check_permutation` | Permutation test |
| `_check_noise` | Noise |
| `_check_other_markets` | Other markets |
| `_check_stress` | Synthetic stress |
| `_blocks` | What a check's section draws, as data |
| `_charts` | The fan, histogram and sessions charts as inline SVG |
| `_charts_bars` | The candles, lines, groups of paths and tables of the bar checks |
| `_scorecard` | The scorecard table and its top line |
| `_html` | The HTML robustness report |
| `_words` | Figures as plain text |
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..errors import BacktestError, ConfigError
from ._assess import assess
from ._check_other_markets import check_strategy
from ._report import RobustnessReport
from ._request import (
    DEFAULT_OBJECTIVE,
    DEFAULT_RUNS,
    DEFAULT_SEED,
    build_request,
    split_keywords,
)
from ._verdict import CheckResult


def robustness(
    strategy: type,
    client: Any = None,
    *,
    checks: Any = "all",
    objective: str | Callable[[Any], float] = DEFAULT_OBJECTIVE,
    runs: int = DEFAULT_RUNS,
    seed: int = DEFAULT_SEED,
    workers: int | None = None,
    progress: Callable[[str], Any] | None = None,
    others: Any = None,
    **settings: Any,
) -> RobustnessReport:
    """Backtest `strategy` once with the run arguments in `settings` and run the
    robustness checks on it (see `bt.robustness`). Raises ConfigError for unusable
    arguments, NotImplementedError for a check or parameter ranges this version does not
    run (and other markets for a strategy that trades no one named symbol), and whatever
    the backtest and its reruns raise."""
    from ..run import backtest
    from ..strategy import Strategy

    if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
        raise ConfigError("bt.robustness takes a bt.Strategy subclass")
    run_arguments, check_settings, ranges = split_keywords(strategy, settings)
    request = build_request(
        checks=checks,
        objective=objective,
        runs=runs,
        seed=seed,
        workers=workers,
        others=others,
        check_settings=check_settings,
        ranges=ranges,
    )
    if request.others:
        check_strategy(strategy, run_arguments)
    started = time.perf_counter()
    result = backtest(strategy, client, **run_arguments)
    return assess(result, request, progress, time.perf_counter() - started)


def result_robustness(
    result: Any,
    *,
    checks: Any = "all",
    objective: str | Callable[[Any], float] = DEFAULT_OBJECTIVE,
    runs: int = DEFAULT_RUNS,
    seed: int = DEFAULT_SEED,
    workers: int | None = None,
    progress: Callable[[str], Any] | None = None,
    others: Any = None,
    **settings: Any,
) -> RobustnessReport:
    """The robustness checks of an existing `result`, reusing its run (see
    `Result.robustness`). Raises BacktestError for a result not made by `bt.backtest`,
    ConfigError for a run argument (the run's own are reused) and as `robustness`."""
    record = result._record
    if record is None:
        raise BacktestError("this result was not made by bt.backtest, so it has no run to check")
    run_arguments, check_settings, ranges = split_keywords(record.strategy, settings)
    if run_arguments:
        names = sorted(run_arguments)
        raise ConfigError(
            f"result.robustness reuses this run's arguments, so it takes no {names}; to "
            f"change them, call result.rerun({names[0]}=...).robustness()"
        )
    request = build_request(
        checks=checks,
        objective=objective,
        runs=runs,
        seed=seed,
        workers=workers,
        others=others,
        check_settings=check_settings,
        ranges=ranges,
    )
    if request.others:
        check_strategy(record.strategy, record.arguments)
    return assess(result, request, progress, None)


__all__ = ["CheckResult", "RobustnessReport", "result_robustness", "robustness"]

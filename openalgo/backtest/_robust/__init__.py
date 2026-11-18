"""Robustness and overfitting checks of a backtest (design 17): `bt.robustness`,
`Result.robustness`, `RobustnessReport` and `CheckResult`; `bt.walk_forward` and
`WalkForwardResult`; and the overfitting figures of `bt.optimize` (`OptimizeResult`).

    report = bt.robustness(EmaCross, client, capital=100_000, start=..., end=...,
                           checks="all", objective="sharpe", runs=1000, seed=0,
                           fast=range(5, 30, 5), slow=range(20, 100, 10))
    report = result.robustness(checks="all")   # the same run, its arguments reused
    report.verdicts, report.check("resample"), report.passed, report.tearsheet(path)

Private: `import openalgo.backtest` does not import it; `bt.robustness`,
`Result.robustness`, the tearsheet, `Result.summary()` and `Result.warnings` import it
when they need it. Every check is deterministic for a seed on every platform: the bar
checks rerun the backtest, an ordinary run of the deterministic kernel, on bars that are
a pure function of the run's bars and the seed, or on the run's own bars with one
setting changed (a delay, a fill model, shifted strategy bars, exits drawn from the
seed, other parameters, a window of the run's sessions).

| Module | Responsibility |
|---|---|
| `_registry` | The seventeen checks in report order and what each reads |
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
| `_session_entries` | Random entries of an all-intraday run kept within one session each |
| `_reruns` | Families of reruns, in this process or worker processes, kept once |
| `_bar_setup` | The rerunning checks' setup: the bars, a first rerun, the runners |
| `_rerun_plan` | What the rerunning checks plan, and the plan's words |
| `_bar_common` | What the bar checks share: the run they read, break-even, rerun sets |
| `_check_vs_random` | Versus random |
| `_check_permutation` | Permutation test |
| `_check_noise` | Noise |
| `_check_other_markets` | Other markets |
| `_check_stress` | Synthetic stress |
| `_stress_table` | Synthetic stress's candidates and the figures of each rule over them |
| `_variant_common` | What the checks on the run's bars with one setting changed share |
| `_exit_draws` | Each run's exit factors and exit parameters, for randomized exits |
| `_check_shifted` | Shifted bars |
| `_check_random_exits` | Randomized exits |
| `_check_delayed` | Delayed orders |
| `_check_liquidity` | Liquidity |
| `_grid` | The parameter combinations of a search: a grid, or the run's neighbourhood |
| `_windows` | Windows of the run's sessions: in and out of sample, walk-forward folds |
| `_search_setup` | The searching checks' setup: their bars and what they read |
| `_search_common` | What the searching checks share: reasons, values as text, the grid drawn |
| `_neighbourhood` | The parameter-neighbourhood rule: a plateau or a spike |
| `_stitch` | Out-of-sample runs stitched into one equity curve |
| `_walk` | The walk-forward: folds searched in sample, traded out of sample |
| `_walk_tables` | The walk-forward's folds as a table |
| `_walk_forward` | `bt.walk_forward` and `WalkForwardResult` |
| `_check_out_of_sample` | Out of sample |
| `_check_walk_forward` | Walk-forward |
| `_check_parameters` | Parameter neighbourhood |
| `_check_noise_optimization` | Noise-robust optimization |
| `_deflated_sharpe` | The deflated Sharpe ratio of a sweep's best |
| `_overfitting_probability` | The probability of backtest overfitting |
| `_optimizer_checks` | The overfitting figures of a `bt.optimize` sweep |
| `_optimizer_page` | The heatmap page of a `bt.optimize` sweep |
| `_blocks` | What a check's section draws, as data |
| `_charts` | The fan, histogram and sessions charts as inline SVG |
| `_charts_bars` | The candles, lines, groups of paths and tables of the bar checks |
| `_charts_grid` | An objective over two parameters: the heatmap and the isometric surface |
| `_charts_folds` | The fold diagram and the table with an equity line a row |
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
from ._walk_forward import WalkForwardResult


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
    robustness checks on it (see `bt.robustness`); the other keywords of `settings` are
    a check's dict or a parameter range. Raises ConfigError for unusable arguments,
    NotImplementedError for other markets of a strategy that trades no one named symbol,
    and whatever the backtest and its reruns raise."""
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


__all__ = [
    "CheckResult",
    "RobustnessReport",
    "WalkForwardResult",
    "result_robustness",
    "robustness",
]

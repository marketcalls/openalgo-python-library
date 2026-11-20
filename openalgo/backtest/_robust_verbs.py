"""The verbs of the robustness and overfitting checks: `bt.robustness` and
`bt.walk_forward`, re-exported unchanged by `run`. Each takes one strategy; both load the
checks (`_robust`) when they run, not before.
"""

from __future__ import annotations

from typing import Any

from ._messages import one_strategy_only
from .strategy import Strategy


def robustness(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Backtest the strategy once and run the robustness and overfitting checks on it;
    returns a `RobustnessReport` (`.verdicts`, `.check(name)`, `.passed`, `.plan`,
    `.tearsheet(path)`). Passing a check proves nothing on its own; failing one says
    the edge did not hold up under that check (it may be fitted to the past or too small
    to tell apart from the noise of the data tested), never that it proves overfitting;
    a check with too little to judge is "not judged" and never passes. The trade-list
    checks and the checks that rerun the chosen settings judge those settings, not the
    search that chose them; the parameter neighbourhood and the noise-robust
    optimization (and the deflated Sharpe ratio and the probability of backtest
    overfitting of `bt.optimize`) can see a strategy picked as the best of many tried.

    Takes the run arguments of `bt.backtest` and:

    checks: "all" (the default: every check), "quick" (the trade-list checks, no rerun)
        or a list of check names: the trade-list checks (random_out_of_sample, reshuffle,
        resample, variance); the searching checks, which rerun the backtest over the
        parameter ranges given (or declared with `bt.param`): out_of_sample (the last 30
        percent of the sessions run with the parameters searched best on the first 70,
        or as given without ranges), walk_forward (as `bt.walk_forward` runs it),
        parameters (every combination of the ranges, or each number parameter at minus,
        zero and plus 10 percent without, judged on the chosen setting's neighbours) and
        noise_optimization (every combination on 50 noisy copies, scored by its median;
        informational, its figure saying whether the best score sits on a plateau);
        and the bar checks, which rerun the backtest on generated bars, other symbols'
        bars or the run's own bars with one setting changed:
        vs_random (random entries with the run's trades, and the strategy on permuted
        bars), permutation (the same moves in another order), noise (every price moved
        within a fifth of its bar's average true range), shifted (strategy bars ending
        1 to n - 1 base bars earlier, or days ending 15 to 60 minutes earlier),
        other_markets, stress (the noisy and permuted series whose drawdown, time to a
        new high or fall below the average is half as bad again as the run's own, judged
        against every generated series), random_exits (each trade's stop, target, trail
        and breakeven distances scaled within a quarter, and exit_params drawn per run),
        delayed (entries and exits held 0 to 2 strategy bars; not judged when the run a
        bar late entered again while its entry was held, as a strategy deciding by
        `self.position.is_flat` without `self.orders` does) and liquidity (fills at the
        next bar's close, midpoint or average price, or their two-bar averages).
    objective: what the checks measure: a metric name or a function of a `Result`
        ("sharpe" by default). Random out of sample measures it on parts of the trades,
        so it takes "sharpe", "sortino", "net_profit", "expectancy" or
        "profit_factor"; "win_rate" (a win rate is not a measure of profit), another
        metric or a function leaves that check not judged. The bar checks measure it on
        each rerun; noise, vs_random, shifted and delayed also need its break-even value
        (0, or 1 for a profit factor), and an objective without one (a win rate, a
        function, whose value for a run that makes nothing is not known) leaves them not
        judged.
    ranges: keywords named after strategy parameters (`fast=range(5, 30, 5)`; the
        `bt.param` ranges the strategy declares otherwise) are what the searching checks
        search; `params=` holds the run's own parameters, its chosen settings, which the
        parameter neighbourhood judges among their neighbours. A parameter named in
        `params=` keeps its declared range searched (so the best of `bt.optimize`, given
        as `params=`, is checked inside the grid it was picked from); a one-value range
        keyword (`fast=[10]`) holds it at one value.
    others: the symbols other_markets runs the strategy on, unchanged (["ICICIBANK",
        "HDFCBANK"] on the strategy's exchange, or (symbol, exchange) pairs), read from
        the run's source: a `data=` dict by symbol holding them, or OpenAlgo through
        `client`. Without it other_markets is not judged.
    runs: alternative runs of each check (1000); seed: the seed of every check's
        stream (0), the same report for the same seed on every platform.
    workers, progress: as in `bt.optimize`: the checks' reruns run in `workers`
        processes (the CPU count by default; `workers=1` keeps them in this process),
        and the plan, with the reruns it needs and an estimate of their time, is reported
        through `progress` (or the `openalgo.backtest.robustness` logger) before the
        checks run.
    A keyword named after a check holds its settings as a dict (`variance={"points":
        10}`, `noise={"runs": 500, "close": 0.75}`); `report.check(name).settings`
        shows each check's.

    The bar checks read the run's bars again from its source and rerun it as ordinary
    backtests of the same kernel, the warm-up included; nothing changes the run's own
    figures.

    Raises ConfigError for unusable arguments or settings (naming what they take), and
    for parameter ranges when checks= lists no check that searches them, the backtest's
    errors, and NotImplementedError for the checks that rerun a futures series and for
    other_markets of a strategy that trades no one named symbol."""
    from ._robust import robustness as run_checks

    if isinstance(strategy, list | tuple):
        raise one_strategy_only("bt.robustness")
    return run_checks(strategy, client, **settings)


def walk_forward(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Walk the strategy forward through time and return a `WalkForwardResult`
    (`.folds`, `.equity`, `.performance`, `.efficiency`, `.profitable_share`, `.drift`,
    `.net_profit`, `.plan`).

    The run's sessions are cut into `folds` (5) folds, each `in_sample` (0.7) of its
    sessions in sample: rolling folds (the default) keep an in-sample window of fixed
    length before each out-of-sample one, `anchored=True` keeps every session before it.
    Each fold searches the parameter ranges on its in-sample window, by `objective`
    ("net_profit" by default, any `bt.optimize` objective, higher better), trades the
    best combination (the first of equal ones) on the next out-of-sample window, and the
    out-of-sample runs, which cover the run after the first fold's sample once, are
    stitched into one equity curve by their P&L. Each window is an ordinary backtest
    that opens flat with the run's capital and reads its warm-up from the bars before it.

    Takes the run arguments of `bt.backtest`, the parameter ranges named after strategy
    parameters (`fast=range(5, 30, 5)`; `bt.param` ranges otherwise, a parameter that
    `params=` names being fixed as in `bt.optimize`), and `folds`, `anchored`,
    `in_sample`, `objective`, `workers` and `progress` as `bt.optimize` takes them (the
    plan, with the reruns and an estimate, is reported before they run, through
    `progress` or the `openalgo.backtest.walk_forward` logger). The run's bars are read
    once in this process, through `client` or from `data=`; worker processes get the
    bars, never the client. The efficiency is the stitched out-of-sample objective per
    year over the mean in-sample one, both from the objective's break-even (a total such
    as the net profit is divided by the years each lasted). `.folds` counts each fold's
    closed trades as whole numbers. `bt.robustness(checks=["walk_forward"], ...)` runs
    the same walk-forward, judges it and draws its folds; there `params=` names the run's
    chosen settings, so a declared range stays searched.

    Raises ConfigError for unusable arguments, a parameter both fixed by `params=` and
    given a range, nothing to search, or too few sessions for the folds (each window
    needs two), and whatever a backtest raises."""
    from ._robust._walk_forward import walk_forward as walk

    if isinstance(strategy, list | tuple):
        raise one_strategy_only("bt.walk_forward")
    return walk(strategy, client, **settings)


__all__ = ["robustness", "walk_forward"]

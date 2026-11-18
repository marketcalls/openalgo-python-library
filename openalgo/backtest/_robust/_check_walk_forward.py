"""Walk-forward (design 17, section 2.3): does it hold when re-fitted through time?

`folds` (5) folds of the run's sessions, rolling (the default) or `anchored`, each
`in_sample` (70 percent) in sample (`_windows.folds`): each fold searches the parameter
ranges on its in-sample window and trades the next out-of-sample window with the
combination that searched best, and the out-of-sample runs are stitched into one equity
curve (`_walk`, as `bt.walk_forward` runs it).

Figures (`_walk`): the walk-forward efficiency, the stitched out-of-sample objective per
year over the mean in-sample one, from break-even; the share of folds profitable out of
sample; the parameter drift. Pass: the stitched out-of-sample run profitable (at least
half a paisa), the efficiency at least `efficiency` (0.5), and at least `fold_share` (60
percent) of the folds that traded profitable.

Not judged when no parameter range is given or declared (nothing to re-fit), when the
objective has no break-even or is a function of a whole `Result` (the stitched run has
none), when a window would hold fewer than two sessions, when no fold could choose, or
when the stitched out-of-sample runs closed fewer than `min_trades` (30) trades.

Charts: the fold diagram; the stitched out-of-sample equity; the fold table (dates,
parameters chosen, a small equity line, net P&L, drawdown, P&L over drawdown, trades,
win rate), the stitched row first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..errors import ConfigError
from . import _words as words
from ._bar_common import break_even, unmeasured
from ._blocks import BOLD, FoldsBlock, FoldSpan, Line, LinesBlock
from ._part_objective import objective_name
from ._reruns import Family, Later
from ._search_setup import SearchRun
from ._setting import Setting, between, whole
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES
from ._verdict import FAILED, NOT_JUDGED, PASSED, Outcome, not_judged
from ._walk import WalkOutcome, walk
from ._walk_tables import drift_words, fold_frame, fold_table
from ._windows import Fold, folds

SETTINGS = (
    ENOUGH_TRADES,
    Setting("folds", 5, whole(1), "a whole number, 1 or more", "the folds of the run's sessions"),
    Setting(
        "anchored",
        False,
        lambda value: value if isinstance(value, bool) else None,
        "True or False",
        "True keeps every session before a fold in sample; False rolls a window of fixed length",
    ),
    Setting(
        "in_sample",
        0.7,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "the share of each fold in sample",
    ),
    Setting(
        "efficiency",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the walk-forward efficiency the stitched run must reach",
    ),
    Setting(
        "fold_share",
        0.6,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of folds that must be profitable out of sample",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"the stitched out-of-sample run is profitable, its walk-forward efficiency (the "
        f"{objective_name(objective)} per year out of sample over in sample) is at least "
        f"{words.percent(settings['efficiency'], 0)}, and at least "
        f"{words.percent(settings['fold_share'], 0)} of the folds are profitable out of sample"
    )


def plan_folds(subject: SearchRun, settings: Mapping[str, Any]) -> tuple[Fold, ...]:
    """The folds of the run's sessions. Raises ConfigError for too few sessions."""
    count, share = settings["folds"], settings["in_sample"]
    return folds(len(subject.dates), count, share, settings["anchored"])


def unjudged(subject: SearchRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None."""
    text = rule(settings, objective)
    if subject.grid is None:
        return not_judged(
            text,
            "Nothing to re-fit: give ranges named after the strategy's parameters "
            "(fast=range(5, 30, 5)) or declare them with bt.param(default, values).",
        )
    if break_even(objective) is None:
        return not_judged(text, unmeasured(objective))
    try:
        plan_folds(subject, settings)
    except ConfigError as error:
        return not_judged(text, f"Too few sessions to judge: {error}.")
    return None


def reruns(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family | Later, int]]:
    """(family, runs) of what the check reruns: each combination in each fold's sample,
    then each fold's choice out of sample; none when it is not judged before any rerun."""
    if unjudged(subject, settings, objective) is not None:
        return []
    grid, made = subject.grid, plan_folds(subject, settings)
    searched = [
        (subject.family(grid, k, fold.in_sample), 1)
        for fold in made
        for k in range(len(grid.combos))
    ]
    return [*searched, (Later("walk_forward", "each fold's choice out of sample"), len(made))]


def run(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    outcome = walk(subject, subject.reruns, plan_folds(subject, settings), objective)
    traded = outcome.traded
    if not traded:
        return not_judged(text, "No fold could choose a combination with a defined objective.")
    table = fold_frame(outcome, subject.dates)
    blocks = _blocks(subject, outcome)
    closed = int(outcome.performance.metrics["trades"])
    if closed < settings["min_trades"]:
        why = (
            f"Too few trades to judge: the stitched out-of-sample runs closed "
            f"{words.whole(closed, 'trade')}, {settings['min_trades']} needed."
        )
        return Outcome(NOT_JUDGED, "n/a", text, why, table, {}, blocks)
    summary = _summary(outcome)
    passed = (
        summary["net_profit"] >= BREAK_EVEN_RUPEES
        and summary["efficiency"] >= settings["efficiency"]
        and summary["profitable_share"] >= settings["fold_share"]
    )
    name = objective_name(objective)
    figure = (
        f"stitched out of sample {words.rupees(summary['net_profit'])}; efficiency "
        f"{words.percent(summary['efficiency'])}; {words.percent(summary['profitable_share'])} "
        f"of {words.whole(len(traded), 'fold')} profitable; drift {drift_words(outcome)}"
    )
    mode = "anchored" if settings["anchored"] else "rolling"
    note = (
        f"{words.whole(len(outcome.folds), 'fold').capitalize()}, {mode}, each "
        f"{words.percent(settings['in_sample'], 0)} in sample: each searched "
        f"{words.whole(len(outcome.grid.combos), 'combination')} in sample by {name} and "
        "traded the best out of sample, each window an ordinary backtest that opens flat "
        "with the run's capital; the out-of-sample runs are stitched by their P&L. The "
        f"efficiency is measured from {break_even(objective):g}, a run that makes nothing."
    )
    return Outcome(PASSED if passed else FAILED, figure, text, note, table, summary, blocks)


def _summary(outcome: WalkOutcome) -> dict[str, float]:
    drift = outcome.drift()
    return {
        "net_profit": outcome.net_profit,
        "efficiency": outcome.efficiency,
        "profitable_share": outcome.profitable_share,
        "in_sample": outcome.in_sample_annual,
        "out_of_sample": outcome.stitched_annual,
        "folds": float(len(outcome.folds)),
        "traded": float(len(outcome.traded)),
        "drift": max(drift.values()) if drift else math.nan,
    }


def _blocks(subject: SearchRun, outcome: WalkOutcome) -> tuple:
    dates = pd.DatetimeIndex(subject.dates)
    spans = tuple(
        FoldSpan(
            f"Fold {fold.fold.number}",
            fold.fold.in_sample.first,
            fold.fold.in_sample.last,
            fold.fold.out_of_sample.first,
            fold.fold.out_of_sample.last,
        )
        for fold in outcome.folds
    )
    diagram = FoldsBlock(
        "Folds", "In sample, then out of sample, on the run's sessions", dates, spans
    )
    equity = outcome.performance.equity if outcome.performance is not None else None
    lines = ()
    if equity is not None and len(equity):
        grown = equity.to_numpy(dtype=float) / subject.capital - 1.0
        lines = (
            Line("the stitched out-of-sample run", pd.DatetimeIndex(equity.index), grown, BOLD),
        )
    curve = LinesBlock(
        "Stitched out of sample",
        "Each fold's out-of-sample run, one after another",
        lines,
        "Cumulative return on the capital.",
    )
    return ((diagram,), (curve,), (fold_table(outcome, subject.dates, subject.capital),))


__all__ = ["SETTINGS", "plan_folds", "reruns", "rule", "run", "unjudged"]

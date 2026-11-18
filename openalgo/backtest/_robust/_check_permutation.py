"""Permutation test (design 17, section 2.5): could the edge come from the price path's
distribution alone?

`runs` alternative price paths with the same moves in another order
(`_generate_permuted`: within each session, and the sessions' first bars with their
overnight gaps as one more set; several instruments with one order), and the strategy
rerun on each (`_reruns`). Figure: the p-value, (the measured runs whose objective is at
or above the run's + 1) / (the measured runs + 1). Pass: p at most `max_p` (0.05).

A run whose objective is undefined (no trade, returns that never vary) or whose bars the
data checks refused says nothing about the order of the moves, so it is left out of both
counts, which can only raise p; when too few runs are measured for p to reach `max_p`
(fewer than 1 / `max_p` - 1), the check is not judged. Not judged either when the run did
not make money or has too few closed trades (`_trade_common`).

The summary block, over the permuted runs that ran: the shallowest, deepest and average
maximum drawdown; the average largest win and loss; the longest win and loss streaks of
any run and the average of each run's longest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _words as words
from ._bar_common import (
    BarRun,
    RerunSet,
    objective_histogram,
    permuted_family,
    rerun_set,
    session_fan,
)
from ._blocks import LIMIT, Mark, Tile, TilesBlock
from ._part_objective import objective_name
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUNS, exact_mean, no_edge, too_few
from ._verdict import FAILED, PASSED, Outcome, not_judged

SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    Setting(
        "max_p",
        0.05,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "the largest p-value that passes",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"p at most {settings['max_p']:g}: few permuted paths reach the run's "
        f"{objective_name(objective)}"
    )


def p_value(values: np.ndarray, real: float) -> tuple[float, int, int]:
    """(p, runs at or above `real`, runs measured) of the finite `values` (module
    docstring)."""
    measured = values[np.isfinite(values)]
    above = int((measured >= real).sum())
    return (above + 1) / (len(measured) + 1), above, len(measured)


def _summary(set_: RerunSet, p: float, above: int, measured: int) -> dict[str, float]:
    ran = set_.ran
    drawdowns = set_.max_drawdown[ran]
    return {
        "runs": float(len(set_.objective)),
        "measured": float(measured),
        "at_or_above": float(above),
        "p_value": p,
        "median_permuted": float(np.median(set_.objective[set_.measured])),
        "best_drawdown": float(drawdowns.max()),
        "worst_drawdown": float(drawdowns.min()),
        "average_drawdown": exact_mean(drawdowns),
        "average_largest_win": exact_mean(np.nan_to_num(set_.largest_win[ran])),
        "average_largest_loss": exact_mean(np.nan_to_num(set_.largest_loss[ran])),
        "longest_win_streak": float(set_.win_streak[ran].max()),
        "average_win_streak": exact_mean(set_.win_streak[ran].astype(float)),
        "longest_loss_streak": float(set_.loss_streak[ran].max()),
        "average_loss_streak": exact_mean(set_.loss_streak[ran].astype(float)),
    }


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    trades = subject.trades
    short = too_few(trades, settings["min_trades"], text) or no_edge(trades, text)
    if short is not None:
        return short
    name = objective_name(objective)
    if not math.isfinite(subject.real):
        return not_judged(text, f"The run's {name} is undefined, so no path can reach it.")
    set_ = rerun_set(subject.reruns.rows(permuted_family(subject.seed), range(runs)))
    p, above, measured = p_value(set_.objective, subject.real)
    least = math.ceil(1.0 / settings["max_p"]) - 1
    if measured < least:
        why = (
            f"Too few runs to judge: {words.whole(measured, 'permuted run')} measured, and p "
            f"can reach {settings['max_p']:g} only with {least} or more.{set_.failure_note()}"
        )
        return not_judged(text, why)
    verdict = PASSED if p <= settings["max_p"] else FAILED
    figure = (
        f"p = {p:.3f}: {words.number(above, 0)} of {words.whole(measured, 'permuted run')} "
        f"at or above the run's {name} of {words.number(subject.real)}"
    )
    note = (
        "Each run rebuilds the bars from the run's own moves in another order: within each "
        "session the bars after its first, and the sessions' first bars with their overnight "
        "gaps as one more set, the history before start apart; volumes travel with their "
        "bars. Runs whose objective is undefined are left out of both counts." + set_.failure_note()
    )
    summary = _summary(set_, p, above, measured)
    table = set_.table()
    return Outcome(
        verdict, figure, text, note, table, summary, _blocks(subject, set_, summary, name, settings)
    )


def _tiles(summary: Mapping[str, float]) -> TilesBlock:
    tiles = (
        Tile("p-value", f"{summary['p_value']:.3f}", "at or above the run, + 1, over runs + 1"),
        Tile("Median permuted", words.number(summary["median_permuted"]), "objective"),
        Tile("Best drawdown", words.percent(summary["best_drawdown"]), "the shallowest", "neg"),
        Tile("Worst drawdown", words.percent(summary["worst_drawdown"]), "the deepest", "neg"),
        Tile("Average drawdown", words.percent(summary["average_drawdown"]), "maximum", "neg"),
        Tile("Average largest win", words.rupees(summary["average_largest_win"]), "per run"),
        Tile("Average largest loss", words.rupees(summary["average_largest_loss"]), "per run"),
        Tile("Longest win streak", words.number(summary["longest_win_streak"], 0), "trades"),
        Tile(
            "Average win streak",
            words.number(summary["average_win_streak"], 1),
            "each run's longest",
        ),
        Tile("Longest loss streak", words.number(summary["longest_loss_streak"], 0), "trades"),
        Tile(
            "Average loss streak",
            words.number(summary["average_loss_streak"], 1),
            "each run's longest",
        ),
    )
    return TilesBlock("Summary", "Over the permuted runs", tiles)


def _blocks(
    subject: BarRun,
    set_: RerunSet,
    summary: Mapping[str, float],
    name: str,
    settings: Mapping[str, Any],
) -> tuple:
    fan = session_fan("Permuted equity", "The strategy on each permuted path", set_, subject)
    values = set_.objective[set_.measured]
    limit = float(np.quantile(values, 1.0 - settings["max_p"])) if len(values) else math.nan
    marks = (Mark(limit, f"top {words.percent(settings['max_p'], 0)} of runs", LIMIT),)
    histogram = objective_histogram(
        "Permuted objectives",
        f"The {name} of each permuted run; p = {summary['p_value']:.3f}",
        values,
        subject.real,
        marks if math.isfinite(limit) else (),
    )
    return ((fan, histogram), (_tiles(summary),))


__all__ = ["SETTINGS", "p_value", "rule", "run"]

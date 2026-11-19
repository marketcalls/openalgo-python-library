"""What the checks that rerun the strategy on its own bars with one setting changed share
(design 17, sections 2.7, 2.12, 2.14 and 2.15: shifted bars, randomized exits, delayed
orders, liquidity): the reasons they leave a run not judged before any rerun, the runs of
a list of variants, the ratio of a variant's objective to the run's, and their charts.

A variant is one family of one run (`_reruns.Family` on the run's own bars), so a check's
variants are each an ordinary backtest of the same kernel with one setting changed.

The ratio of a variant: its objective over the run's, both measured from the objective's
break-even (`_bar_common.break_even`): (variant - break-even) / (run - break-even); NaN
for a variant whose objective is undefined. The checks that judge a ratio need the run's
objective defined, with a break-even, and above it (`objective_unjudged`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import _words as words
from ._bar_common import (
    BarRun,
    RerunSet,
    below_break_even,
    break_even,
    finite,
    joined,
    real_path,
    rerun_set,
    undefined,
    unmeasured,
)
from ._blocks import PathGroup, PathsBlock, TableBlock
from ._part_objective import objective_name
from ._reruns import Family
from ._trade_common import no_edge, too_few
from ._verdict import Outcome, not_judged


def run_unjudged(subject: BarRun, settings: Mapping[str, Any], rule: str) -> Outcome | None:
    """The "not judged" outcome of a run with too few closed trades or no edge, else
    None (`_trade_common`)."""
    return too_few(subject.trades, settings["min_trades"], rule) or no_edge(subject.trades, rule)


def objective_unjudged(subject: Any, objective: Any, rule: str, compared: str) -> Outcome | None:
    """The "not judged" outcome of a run (a `BarRun` or a `_search_setup.SearchRun`: its
    objective's value `real`) whose objective is undefined, has no break-even or is not
    above it, so no `compared` run's ratio can be measured; else None."""
    name, level = objective_name(objective), break_even(objective)
    if not finite(subject.real):
        return not_judged(rule, undefined(name, compared))
    if level is None:
        return not_judged(rule, unmeasured(objective))
    if not subject.real > level:
        return not_judged(rule, below_break_even(name, subject.real, level))
    return None


def variant_set(subject: BarRun, families: Sequence[Family]) -> RerunSet:
    """The runs of `families`, one each (run 0), in order."""
    return joined([rerun_set(subject.reruns.rows(family, [0])) for family in families])


def ratios(values: np.ndarray, real: float, objective: Any) -> np.ndarray:
    """Each value's ratio to the run's objective `real`, from the break-even (module
    docstring); NaN without a break-even or for an undefined value."""
    level = break_even(objective)
    if level is None or not finite(real) or real == level:
        return np.full(len(values), np.nan)
    return (np.asarray(values, dtype=float) - level) / (real - level)


def variants_chart(
    title: str,
    subtitle: str,
    groups: Sequence[tuple[str, Sequence[int]]],
    set_: RerunSet,
    subject: BarRun,
) -> PathsBlock:
    """The equity of the variants that ran, a group of them in a colour and dash each
    (`groups`: (label, positions in `set_`)), with the run's own equity bold."""
    setup = subject.reruns.setup
    drawn = []
    for label, positions in groups:
        mine = [k for k in positions if set_.ran[k]]
        if mine:
            drawn.append(PathGroup(label, set_.paths[mine]))
    return PathsBlock(
        title,
        subtitle,
        setup.steps,
        pd.DatetimeIndex(setup.dates),
        tuple(drawn),
        real_path(subject.trades, setup.steps),
    )


def variants_table(
    title: str,
    subtitle: str,
    labels: Sequence[str],
    set_: RerunSet,
    ratio: np.ndarray,
    name: str,
) -> TableBlock:
    """A row per variant: its label, objective, ratio to the run's, net P&L, maximum
    drawdown and closed trades (a refused variant says so)."""
    rows = []
    for k, label in enumerate(labels):
        if not set_.ran[k]:
            rows.append((label, "refused", "", "", "", ""))
            continue
        rows.append(
            (
                label,
                words.number(set_.objective[k]),
                words.number(ratio[k]),
                words.rupees(set_.net_profit[k]),
                words.percent(set_.max_drawdown[k]),
                words.number(set_.trades[k], 0),
            )
        )
    columns = ("Run", name.capitalize(), "Over the run", "Net P&L", "Max drawdown", "Trades")
    return TableBlock(title, subtitle, columns, tuple(rows))


def labelled_table(set_: RerunSet, ratio: np.ndarray, **labels: Sequence[Any]) -> pd.DataFrame:
    """`set_.table()` with the variants' `labels` (columns by name) first and their ratio
    to the run's objective after the objective."""
    frame = set_.table()
    for position, (name, values) in enumerate(labels.items()):
        frame.insert(position, name, list(values))
    frame.insert(frame.columns.get_loc("objective") + 1, "ratio", ratio)
    return frame


__all__ = [
    "labelled_table",
    "objective_unjudged",
    "ratios",
    "run_unjudged",
    "variant_set",
    "variants_chart",
    "variants_table",
]

"""Noise-robust optimization (design 17, section 2.17): which settings hold across noisy
histories?

Every combination of the parameter ranges reruns on `samples` (50) noisy copies of the
run's bars, made as the noise check makes them (`_generate_noisy`; its chances, fraction
and period, here this check's own settings with the noise check's defaults), from this
check's own stream; every combination reads the same copies. Each combination's noisy
score is its median objective over the copies whose objective is defined. The robust
choice is the combination with the highest score (the first of equal ones); the plain
optimum, the one with the highest objective on the run's own bars, which every
combination also reruns on.

Figures: the robust choice and its score; its neighbourhood on the noisy score, by the
parameter-neighbourhood rule (`_neighbourhood`: a neighbour made money when its median
noisy net profit did); its objective on the run's own bars against the plain optimum's.
Verdict: informational, as design 17 sets it, with the parameter-neighbourhood rule
applied to the robust choice and its outcome stated in the figure and the summary
(`plateau`, 1 when the rule holds, else 0): the robust choice's 25th percentile neighbour
above break-even on the noisy score and its median neighbour keeping at least `keep`
(half) of its score. It counts neither as a pass nor as a failure.

Not judged when no parameter range is given or declared (there is nothing to search),
when the objective has no break-even, when no combination has a defined score, or when
the robust choice's score is not above break-even (no setting holds an edge on the noisy
histories).

Charts: the heatmap of the noisy score beside the heatmap of the objective on the run's
own bars, the robust choice and the plain optimum outlined on both.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import _random
from . import _words as words
from ._bar_common import break_even, rerun_set, unmeasured
from ._check_noise import SETTINGS as NOISE_SETTINGS
from ._check_noise import noise_of
from ._grid import Grid
from ._neighbourhood import DEFAULT_KEEP, Plateau, plateau, rule_words
from ._part_objective import objective_name
from ._reruns import NOISY, Family
from ._search_common import grid_block, params_text
from ._search_setup import SearchRun
from ._setting import Setting, between, whole
from ._trade_common import BREAK_EVEN_RUPEES
from ._verdict import INFORMATIONAL, Outcome, not_judged
from ._walk import best

#: The number of this check: the stream of its noisy copies.
NUMBER = 17
#: The noise check's settings that shape a noisy copy, which this check takes too.
SHAPE = ("open", "high", "low", "close", "fraction", "period")

SETTINGS = (
    Setting(
        "samples",
        50,
        whole(1),
        "a whole number, 1 or more",
        "the noisy copies each combination reruns on",
    ),
    *(setting for setting in NOISE_SETTINGS if setting.name in SHAPE),
    Setting(
        "keep",
        DEFAULT_KEEP,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the robust choice's score its median neighbour must keep",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    level = break_even(objective)
    words_ = rule_words(
        objective_name(objective), 0.0 if level is None else level, settings["keep"]
    )
    return (
        "informational: the parameter-neighbourhood rule applied to the robust choice on "
        f"the noisy score, {words_}"
    )


def unjudged(subject: SearchRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None."""
    text = rule(settings, objective)
    if subject.grid is None:
        return not_judged(
            text,
            "No parameter range to search: give ranges named after the strategy's parameters "
            "(fast=range(5, 30, 5)) or declare them with bt.param(default, values).",
        )
    if break_even(objective) is None:
        return not_judged(text, unmeasured(objective))
    return None


def _noisy(subject: SearchRun, settings: Mapping[str, Any], position: int) -> Family:
    seed = _random.check_seed(subject.seed, NUMBER)
    params = subject.grid.changes(position)
    return Family(NOISY, seed, noise_of(settings), params=params)


def reruns(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: each combination on every noisy copy and
    once on the run's own bars, none when it is not judged before any rerun."""
    if unjudged(subject, settings, objective) is not None:
        return []
    count = len(subject.grid.combos)
    noisy = [(_noisy(subject, settings, k), settings["samples"]) for k in range(count)]
    return noisy + [(subject.family(subject.grid, k), 1) for k in range(count)]


def _scores(subject: SearchRun, settings: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(each combination's median noisy objective, its median noisy net profit)."""
    samples = range(settings["samples"])
    scores, profits = [], []
    for position in range(len(subject.grid.combos)):
        set_ = rerun_set(subject.reruns.rows(_noisy(subject, settings, position), samples))
        measured = set_.objective[set_.measured]
        ran = set_.net_profit[set_.ran]
        scores.append(float(np.median(measured)) if len(measured) else np.nan)
        profits.append(float(np.median(ran)) if len(ran) else np.nan)
    return np.array(scores), np.array(profits)


def run(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    grid, level, name = subject.grid, break_even(objective), objective_name(objective)
    scores, profits = _scores(subject, settings)
    families = [subject.family(grid, k) for k in range(len(grid.combos))]
    plain = np.array([row.objective for row in subject.reruns.firsts(families)], dtype=float)
    robust, optimum = best(scores), best(plain)
    if robust is None:
        return not_judged(text, "No combination has a defined score on the noisy copies.")
    if not scores[robust] > level:
        return not_judged(
            text,
            f"No edge to judge: the best noisy score, {words.number(scores[robust])}, is not "
            f"above {level:g}, the value of a run that makes nothing.",
        )
    near = grid.neighbours(robust)
    others = [k for k in range(len(grid.combos)) if k != robust]
    profitable = profits[near] >= BREAK_EVEN_RUPEES
    found = plateau(
        scores[robust], scores[near], profitable, scores[others], level, settings["keep"]
    )
    if found is None:
        return not_judged(text, "No neighbour of the robust choice has a defined score.")
    table = _table(grid, scores, profits, plain, robust, optimum)
    summary = {
        **found.summary(),
        "plateau": float(found.passed),
        "robust_real": float(plain[robust]),
        "optimum_real": float(plain[optimum]) if optimum is not None else np.nan,
    }
    figure = _figure(grid, found, plain, robust, optimum, name)
    note = (
        f"Each of {words.whole(len(grid.combos), 'combination')} reran on "
        f"{words.whole(settings['samples'], 'noisy copy', 'noisy copies')} of the run's bars, "
        "the same copies for every combination, and once on the run's own bars; its score is "
        f"its median {name} over the copies. The neighbourhood rule reads the noisy score; "
        "its outcome is stated, not counted as a pass or a failure (design 17)."
    )
    blocks = _blocks(grid, scores, plain, robust, optimum, name, level)
    return Outcome(INFORMATIONAL, figure, text, note, table, summary, blocks)


def _figure(
    grid: Grid, found: Plateau, plain: np.ndarray, robust: int, optimum: int | None, name: str
) -> str:
    against = (
        "the plain optimum is the same setting"
        if optimum == robust
        else f"the plain optimum ({params_text(grid.swept(optimum))}) makes "
        f"{words.number(plain[optimum])}"
        if optimum is not None
        else "no plain optimum is defined"
    )
    shape = "on a plateau" if found.passed else "a lone spike, not a plateau"
    return (
        f"robust choice {params_text(grid.swept(robust))} scores {words.number(found.chosen)}, "
        f"by the neighbourhood rule {shape}: {words.percent(found.profitable_share)} of its "
        f"neighbours profitable, the median keeps {words.percent(found.median_ratio)}; on "
        f"the run's bars it makes {words.number(plain[robust])} {name}, and {against}"
    )


def _table(
    grid: Grid,
    scores: np.ndarray,
    profits: np.ndarray,
    plain: np.ndarray,
    robust: int,
    optimum: int | None,
) -> pd.DataFrame:
    frame = pd.DataFrame({name: [combo[name] for combo in grid.combos] for name in grid.names})
    frame["noisy_score"] = scores
    frame["noisy_net_profit"] = profits
    frame["objective"] = plain
    frame["robust"] = [k == robust for k in range(len(grid.combos))]
    frame["optimum"] = [k == optimum for k in range(len(grid.combos))]
    frame.index.name = "combination"
    return frame


def _blocks(
    grid: Grid,
    scores: np.ndarray,
    plain: np.ndarray,
    robust: int,
    optimum: int | None,
    name: str,
    level: float,
) -> tuple:
    marked = [(robust, "robust")]
    if optimum is not None and optimum != robust:
        marked.append((optimum, "plain optimum"))
    noisy = grid_block(
        "Noisy score",
        f"The median {name} over the noisy copies",
        grid,
        scores,
        False,
        marked,
        level,
    )
    real = grid_block(
        "Plain objective", f"The {name} on the run's own bars", grid, plain, False, marked, level
    )
    return ((noisy, real),)


__all__ = ["NUMBER", "SETTINGS", "reruns", "rule", "run", "unjudged"]

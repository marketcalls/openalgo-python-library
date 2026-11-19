"""Versus random (design 17, section 2.4): is it better than random entries with the same
exposure?

Three families of `runs` reruns each (`families`, all three by default):

    random_entries        random entries on the run's own bars (`_random_entries`: the
                          run's trade count, directions, quantities and holding times,
                          placed at random strategy bars, each exited after its holding
                          time)
    permuted              the strategy on permuted bars (`_generate_permuted`): the
                          permutation test's runs, its stream, so a report that runs both
                          reruns them once
    random_on_permuted    random entries on those permuted bars

Figure: the run's objective over the best random one (the best of every measured run of
every family, not the average), both measured from the objective's break-even
(`_bar_common`). Pass: the run's objective above the best by `margin` (0 by default; 0.03
asks for 3 percent more): run - break-even > (1 + margin) x (best - break-even), which a
run above break-even passes whenever every random run is at or below it.

Random entries need the run's trades to fit among its strategy bars, and permuted bars
need every symbol's bars at the same times; the strategy on permuted bars needs a
strategy that reads them, so it is left out for one made by `bt.from_arrays`, whose
decisions are arrays fixed in advance (`_bar_common.FIXED_ARRAYS`; the random entries
stay). Families that cannot run are left out and the note says why. Not judged when the
run did not make money, has too few closed trades, has an objective that is undefined,
has no break-even or is not above it, when no family can run, or when no random run
could be measured.

Draws: random entries on the run's bars from part 0 of the check's stream
(`_random.part_seed`), random entries on permuted bars from part 1, on the permutation
test's permuted bars.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import _random
from . import _words as words
from ._bar_common import (
    FIXED_ARRAYS,
    BarRun,
    RerunSet,
    below_break_even,
    break_even,
    finite,
    fixed_decisions,
    grid_problem,
    joined,
    objective_histogram,
    permuted_family,
    rerun_set,
    session_fan,
    undefined,
    unmeasured,
)
from ._blocks import TableBlock, Tile, TilesBlock
from ._part_objective import objective_name
from ._reruns import PERMUTED, REAL, Family
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUNS, no_edge, too_few
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: The families, in the order a report lists them, and what each runs.
FAMILIES = {
    "random_entries": "random entries on the run's bars",
    "permuted": "the strategy on permuted bars",
    "random_on_permuted": "random entries on permuted bars",
}
#: The families that draw random entries.
RANDOM = ("random_entries", "random_on_permuted")
#: The families on permuted bars.
ON_PERMUTED = ("permuted", "random_on_permuted")


def families_value(value: Any) -> tuple[str, ...] | None:
    """The `families` setting: a non-empty list of family names, in report order; None
    for anything else."""
    if isinstance(value, str) or not isinstance(value, list | tuple) or not value:
        return None
    if not all(isinstance(name, str) and name in FAMILIES for name in value):
        return None
    return tuple(name for name in FAMILIES if name in value)


SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    Setting(
        "margin",
        0.0,
        between(0.0, 10.0, low_open=False, high_open=False),
        "a fraction from 0 to 10",
        "how far above the best random run the run must be, a fraction of it",
    ),
    Setting(
        "families",
        tuple(FAMILIES),
        families_value,
        f"a list of {list(FAMILIES)}",
        "the families of random runs",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    above = "above" if not settings["margin"] else f"{words.percent(settings['margin'], 0)} above"
    return (
        f"the run's {objective_name(objective)} is {above} the best of every random run "
        f"({', '.join(settings['families'])})"
    )


def family(name: str, seed: int, report_seed: int) -> Family:
    """The reruns of family `name` (module docstring)."""
    permuted = permuted_family(report_seed)
    if name == "random_entries":
        return Family(REAL, entries_seed=_random.part_seed(seed, 0))
    if name == "permuted":
        return permuted
    return Family(PERMUTED, permuted.bars_seed, entries_seed=_random.part_seed(seed, 1))


def _placed(subject: BarRun, names: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """(the families that can run, why the others cannot, "" when all can)."""
    entries = subject.entries
    unplaced = entries.unplaced() if entries is not None else []
    problem = grid_problem(subject.bars)
    fixed = fixed_decisions(subject) and "permuted" in names and not problem
    why = ""
    if unplaced:
        why += (
            f" Random entries are left out: the trades of {', '.join(unplaced)} do not fit "
            "among its strategy bars of the run."
        )
    if problem:
        why += f" The families on permuted bars are left out: {problem}."
    if fixed:
        why += f" The strategy on permuted bars is left out: {FIXED_ARRAYS}."
    left_out = (
        (RANDOM if unplaced else ())
        + (ON_PERMUTED if problem else ())
        + (("permuted",) if fixed else ())
    )
    return tuple(name for name in names if name not in left_out), why


def unjudged(subject: BarRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None:
    the one source of whether the check reruns (`reruns`)."""
    text = rule(settings, objective)
    short = too_few(subject.trades, settings["min_trades"], text) or no_edge(subject.trades, text)
    if short is not None:
        return short
    name, level = objective_name(objective), break_even(objective)
    if not finite(subject.real):
        return not_judged(text, undefined(name, "random"))
    if level is None:
        return not_judged(text, unmeasured(objective))
    if not subject.real > level:
        return not_judged(text, below_break_even(name, subject.real, level))
    names, left_out = _placed(subject, settings["families"])
    if not names:
        return not_judged(text, "No family of random runs can run." + left_out)
    return None


def reruns(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: each family that can run, none when the
    check is not judged before any rerun (`unjudged`)."""
    if unjudged(subject, settings, objective) is not None:
        return []
    names, _ = _placed(subject, settings["families"])
    return [(family(each, seed, subject.seed), runs) for each in names]


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    name, level = objective_name(objective), break_even(objective)
    names, left_out = _placed(subject, settings["families"])
    sets = {
        each: rerun_set(subject.reruns.rows(family(each, seed, subject.seed), range(runs)))
        for each in names
    }
    pooled = np.concatenate([set_.objective[set_.measured] for set_ in sets.values()])
    if not len(pooled):
        return not_judged(text, "No random run could be measured." + left_out)
    best, real = float(pooled.max()), subject.real
    passed = real - level > (1.0 + settings["margin"]) * (best - level)
    summary = _summary(sets, pooled, real, level)
    figure = _figure(name, real, best, level, summary)
    note = (
        "Random entries hold the run's own trades (count, direction, quantity, holding time) "
        "at random strategy bars, under the run's rules; the strategy on permuted bars is the "
        "permutation test's runs." + left_out + "".join(s.failure_note() for s in sets.values())
    )
    table = _table(sets)
    blocks = _blocks(subject, sets, pooled, summary, name)
    return Outcome(PASSED if passed else FAILED, figure, text, note, table, summary, blocks)


def _summary(
    sets: Mapping[str, RerunSet], pooled: np.ndarray, real: float, level: float
) -> dict[str, float]:
    best = float(pooled.max())
    summary = {
        "runs": float(sum(len(set_.objective) for set_ in sets.values())),
        "measured": float(len(pooled)),
        "best_random": best,
        "ratio": (real - level) / (best - level) if best > level else np.inf,
        "percentile": float((pooled < real).mean()),
    }
    for each, set_ in sets.items():
        values = set_.objective[set_.measured]
        summary[f"best_{each}"] = float(values.max()) if len(values) else np.nan
    return summary


def _figure(name: str, real: float, best: float, level: float, summary: Mapping) -> str:
    if best <= level:
        return (
            f"the run's {name} {words.number(real)} is above every random run (the best, "
            f"{words.number(best)}, is at or below {level:g})"
        )
    return (
        f"the run's {name} {words.number(real)} is {words.number(summary['ratio'])} times the "
        f"best random run's {words.number(best)}; above "
        f"{words.percent(summary['percentile'])} of random runs"
    )


def _table(sets: Mapping[str, RerunSet]) -> pd.DataFrame:
    """A row per run of every family, with its family."""
    frames = []
    for each, set_ in sets.items():
        frame = set_.table()
        frame.insert(0, "family", each)
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)


def _blocks(
    subject: BarRun,
    sets: Mapping[str, RerunSet],
    pooled: np.ndarray,
    summary: Mapping[str, float],
    name: str,
) -> tuple:
    every = joined(list(sets.values()))
    fan = session_fan("Random equity", "Every random run, the strategy bold", every, subject)
    histogram = objective_histogram(
        "Random objectives",
        f"The {name} of every random run; the run above {words.percent(summary['percentile'])}",
        pooled,
        subject.real,
    )
    rows = []
    for each, set_ in sets.items():
        values = set_.objective[set_.measured]
        rows.append(
            (
                FAMILIES[each],
                words.number(len(values), 0),
                words.number(float(values.max()) if len(values) else np.nan),
                words.number(float(np.median(values)) if len(values) else np.nan),
                words.percent(float((values < subject.real).mean()) if len(values) else np.nan),
                words.number(float(set_.trades[set_.ran].mean()) if set_.ran.any() else np.nan, 1),
            )
        )
    columns = ("Family", "Measured", "Best", "Median", "Below the run", "Average trades")
    subtitle = (
        f"The {name} of each family's runs; the run has "
        f"{words.whole(subject.trades.closed, 'closed trade')}"
    )
    table = TableBlock("By family", subtitle, columns, tuple(rows))
    tiles = (
        Tile("The run", words.number(subject.real), name),
        Tile("Best random", words.number(summary["best_random"]), name),
        Tile("Run over best", words.number(summary["ratio"]), "from break-even"),
        Tile("Above", words.percent(summary["percentile"]), "of random runs"),
    )
    return ((fan, histogram), (table,), (TilesBlock("Summary", "Over every random run", tiles),))


__all__ = [
    "FAMILIES",
    "SETTINGS",
    "families_value",
    "family",
    "reruns",
    "rule",
    "run",
    "unjudged",
]

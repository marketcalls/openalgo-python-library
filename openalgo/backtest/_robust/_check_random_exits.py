"""Randomized exits (design 17, section 2.12): are the exits fitted?

The strategy keeps its decisions and reruns on its own bars `runs` times with its exits
varied (`_exit_draws`): the distances of its engine-held exits (stop, target or targets,
trail, breakeven) are multiplied per trade by a factor drawn uniformly from 1 - `spread`
to 1 + `spread` (0.25; the kernel's `engine::exit_scale`), and each strategy parameter
named in `exit_params` ({"atr_stop": [1.5, 2.0, 2.5]}, a list or range of values each)
takes one of its values per run. The entries the strategy decides are unchanged; what
they meet after is not, so a run may trade differently where an exit moved.

Figure: the share of runs profitable (net profit of at least half a paisa). Pass: at
least `profitable` (0.7).

Not judged when the run did not make money or has too few closed trades, or when there
is nothing to vary: no leg of the run held an engine exit (`Result.positions`) and no
`exit_params` were named.

Charts: the fan of the runs with the run's bold; the histogram of their objectives with
the run's marked; a summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ConfigError
from . import _random
from . import _words as words
from ._bar_common import BarRun, RerunSet, objective_histogram, rerun_set, session_fan
from ._blocks import Tile, TilesBlock
from ._exit_draws import ExitDraws
from ._part_objective import objective_name
from ._reruns import REAL, Family
from ._setting import Setting, between
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES, RUNS
from ._variant_common import run_unjudged
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: What the run's position history says of a leg that holds an engine exit.
HELD_COLUMNS = ("stop_rupees", "trail_armed", "breakeven_armed")


def exit_params_value(value: Any) -> tuple[tuple[str, tuple[Any, ...]], ...] | None:
    """The `exit_params` setting: a dict of parameter names to a non-empty list, tuple or
    range of values, as (name, values) pairs in the order given; None for anything
    else."""
    if not isinstance(value, dict) or not value:
        return None
    pairs = []
    for name, values in value.items():
        if not isinstance(name, str) or not isinstance(values, list | tuple | range):
            return None
        if not len(values):
            return None
        pairs.append((name, tuple(values)))
    return tuple(pairs)


SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    Setting(
        "spread",
        0.25,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "how far a trade's exit distances may move, a fraction of each",
    ),
    Setting(
        "exit_params",
        None,
        exit_params_value,
        "a dict of strategy parameter names to lists of values",
        "the exit parameters drawn per run, among their values",
    ),
    Setting(
        "profitable",
        0.7,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of runs that must make money",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    share = words.percent(settings["profitable"], 0)
    return f"at least {share} of the runs with varied exits profitable"


def holds_engine_exits(result: Any) -> bool:
    """Whether a leg of the run held an engine exit: a stop, a target, a trail or a
    breakeven (`Result.positions`)."""
    levels = result.positions
    if levels.empty:
        return False
    if any(levels[name].notna().any() for name in HELD_COLUMNS if name in levels):
        return True
    return "targets" in levels and bool(levels["targets"].map(lambda t: len(t or ()) > 0).any())


def _draws(subject: BarRun, settings: Mapping[str, Any], seed: int) -> ExitDraws:
    """What each run's exits are drawn from: one factor for each strategy bar of the run
    on every symbol, and one more."""
    bars = sum(len(closes) for closes in subject.result._strategy_closes)
    params = settings["exit_params"] or ()
    return ExitDraws(_random.part_seed(seed, 0), float(settings["spread"]), params, bars + 1)


def unjudged(subject: BarRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None:
    the one source of whether the check reruns (`reruns`). Raises ConfigError for an
    exit parameter the strategy does not have."""
    text = rule(settings, objective)
    named = settings["exit_params"] or ()
    strategy = subject.result._record.strategy
    unknown = [name for name, _ in named if name not in strategy.parameters()]
    if unknown:
        raise ConfigError(
            f"random_exits: exit_params names {unknown}, which are not parameters of "
            f"{strategy.__name__} ({sorted(strategy.parameters())})"
        )
    short = run_unjudged(subject, settings, text)
    if short is None and not named and not holds_engine_exits(subject.result):
        short = not_judged(
            text,
            "Nothing to vary: no trade held an engine exit (stop, target, trail or "
            "breakeven), and no exit_params were named.",
        )
    return short


def reruns(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: `runs` runs with varied exits, none when
    it is not judged before any rerun (`unjudged`)."""
    if unjudged(subject, settings, objective) is not None:
        return []
    return [(Family(REAL, exits=_draws(subject, settings, seed)), runs)]


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    draws = _draws(subject, settings, seed)
    set_ = rerun_set(subject.reruns.rows(Family(REAL, exits=draws), range(runs)))
    if not set_.ran.any():
        return not_judged(text, "No run with varied exits ran." + set_.failure_note())
    name = objective_name(objective)
    summary = _summary(set_, subject.real)
    passed = summary["profitable_share"] >= settings["profitable"]
    figure = (
        f"{words.percent(summary['profitable_share'])} of "
        f"{words.whole(int(summary['runs']), 'run')} with varied exits profitable; median "
        f"{name} {words.number(summary['median'])} against the run's {words.number(subject.real)}"
    )
    varied = ", ".join(n for n, _ in settings["exit_params"] or ()) or "none"
    note = (
        f"Each run multiplied every trade's engine-held exit distances by a factor of its "
        f"own from {1 - settings['spread']:g} to {1 + settings['spread']:g}; exit "
        f"parameters drawn per run: {varied}. The strategy's entry decisions are its own."
        + set_.failure_note()
    )
    table = _table(set_, draws, runs)
    return Outcome(
        PASSED if passed else FAILED,
        figure,
        text,
        note,
        table,
        summary,
        _blocks(subject, set_, summary, name),
    )


def _table(set_: RerunSet, draws: ExitDraws, runs: int) -> pd.DataFrame:
    """A row per run, with the exit parameters it drew."""
    frame = set_.table()
    for name, _ in draws.params:
        frame[name] = [draws.values(r)[name] for r in range(runs)]
    return frame


def _summary(set_: RerunSet, real: float) -> dict[str, float]:
    ran, values = set_.ran, set_.objective[set_.measured]
    return {
        "runs": float(ran.sum()),
        "profitable_share": float((set_.net_profit[ran] >= BREAK_EVEN_RUPEES).mean()),
        "median": float(np.median(values)) if len(values) else np.nan,
        "percentile": float((values < real).mean()) if len(values) else np.nan,
    }


def _blocks(subject: BarRun, set_: RerunSet, summary: Mapping[str, float], name: str) -> tuple:
    fan = session_fan("Equity with varied exits", "Each run's exits varied", set_, subject)
    histogram = objective_histogram(
        "Objectives with varied exits", f"The {name} of each run", set_.objective, subject.real
    )
    tiles = TilesBlock(
        "Summary",
        "Over the runs with varied exits",
        (
            Tile("Profitable", words.percent(summary["profitable_share"]), "of the runs"),
            Tile("Median", words.number(summary["median"]), name),
            Tile(
                "The run's percentile",
                words.ordinal(summary["percentile"]),
                f"{words.percent(summary['percentile'])} of the runs below it",
            ),
        ),
    )
    return ((fan, histogram), (tiles,))


__all__ = [
    "HELD_COLUMNS",
    "SETTINGS",
    "exit_params_value",
    "holds_engine_exits",
    "reruns",
    "rule",
    "run",
    "unjudged",
]

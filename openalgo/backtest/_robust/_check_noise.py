"""Noise (design 17, section 2.6): does it survive small changes to every bar?

`runs` noisy copies of the run's bars (`_generate_noisy`: each price moves with its
chance, `open` 0.40, `high` 0.60, `low` 0.50 and `close` 0.75, by a uniform draw within
`fraction` 0.20 of the bar's average true range of `period` 14 base bars, and each bar is
made valid on the tick grid), and the strategy rerun on each. Figures, over the measured
runs (`_bar_common`): the median noise objective over the run's, both measured from the
objective's break-even; the share of noise runs profitable; the run's percentile among
them (the share of noise runs below it). Pass: the ratio at least `keep` (0.5), at least
`profitable` (0.8) of the runs profitable, and the run not above the `top` (0.95)
percentile of the noise objectives (linear between order statistics).

Not judged when the run did not make money, has too few closed trades, has an objective
with no break-even or one not above it, or when no noise run could be measured.

Charts: the noise equity fan with the run bold; the histogram of the noise objectives
with the run, its limit at the top percentile and the keep line marked; candles of the
run's last `CANDLES` base bars beside the same bars of the first noisy copy.
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
    below_break_even,
    break_even,
    objective_histogram,
    rerun_set,
    session_fan,
    unmeasured,
)
from ._blocks import LIMIT, CandlesBlock, Mark, Tile, TilesBlock
from ._generate_noisy import Noise, noisy
from ._part_objective import objective_name
from ._reruns import NOISY, Family
from ._setting import Setting, between, whole
from ._trade_common import ENOUGH_TRADES, RUNS, no_edge, too_few
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: The base bars the candle charts show.
CANDLES = 60


def _chance(name: str, default: float) -> Setting:
    return Setting(
        name,
        default,
        between(0.0, 1.0, low_open=False, high_open=False),
        "a fraction from 0 to 1",
        f"the chance that a bar's {name} moves",
    )


def _share(name: str, default: float, meaning: str) -> Setting:
    return Setting(
        name,
        default,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        meaning,
    )


SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    _chance("open", 0.40),
    _chance("high", 0.60),
    _chance("low", 0.50),
    _chance("close", 0.75),
    Setting(
        "fraction",
        0.20,
        between(0.0, 10.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 10",
        "the most a price moves, as a fraction of its bar's average true range",
    ),
    Setting(
        "period", 14, whole(1), "a whole number, 1 or more", "the bars of the average true range"
    ),
    _share("keep", 0.5, "the share of the run's objective the median noise run must keep"),
    _share("profitable", 0.8, "the share of noise runs that must make money"),
    _share("top", 0.95, "the percentile of the noise runs the run must not be above"),
)


def noise_of(settings: Mapping[str, Any]) -> Noise:
    """The `Noise` the settings describe."""
    chances = tuple(float(settings[name]) for name in ("open", "high", "low", "close"))
    return Noise(chances, float(settings["fraction"]), int(settings["period"]))


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"the median noise run keeps {words.percent(settings['keep'], 0)} of the run's "
        f"{objective_name(objective)}, at least {words.percent(settings['profitable'], 0)} of "
        f"noise runs are profitable, and the run is not above their "
        f"{words.percent(settings['top'], 0)} percentile"
    )


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    trades = subject.trades
    short = too_few(trades, settings["min_trades"], text) or no_edge(trades, text)
    if short is not None:
        return short
    name, level = objective_name(objective), break_even(objective)
    if level is None:
        return not_judged(text, unmeasured(objective))
    if not subject.real > level:
        return not_judged(text, below_break_even(name, subject.real, level))
    family = Family(NOISY, seed, noise_of(settings))
    set_ = rerun_set(subject.reruns.rows(family, range(runs)))
    values = set_.objective[set_.measured]
    if not len(values):
        return not_judged(text, "No noise run could be measured." + set_.failure_note())
    summary = _summary(set_, values, subject.real, level, settings)
    passed = (
        summary["ratio"] >= settings["keep"]
        and summary["profitable_share"] >= settings["profitable"]
        and subject.real <= summary["top_limit"]
    )
    figure = (
        f"median {name} {words.number(summary['median'])} keeps "
        f"{words.percent(summary['ratio'])} of the run's {words.number(subject.real)}; "
        f"{words.percent(summary['profitable_share'])} of noise runs profitable; the run at "
        f"the {words.percent(summary['percentile'])} percentile"
    )
    note = (
        f"Each run moves every price of every bar, with its chance, by up to "
        f"{words.percent(settings['fraction'], 0)} of the bar's average true range of "
        f"{settings['period']} base bars, history included, then puts the bar back on the "
        f"tick grid with its high the largest and its low the smallest. The ratio is "
        f"measured from {level:g}, a run that makes nothing." + set_.failure_note()
    )
    keep_line = level + settings["keep"] * (subject.real - level)
    blocks = _blocks(subject, set_, summary, name, settings, seed, keep_line)
    return Outcome(PASSED if passed else FAILED, figure, text, note, set_.table(), summary, blocks)


def _summary(
    set_: RerunSet, values: np.ndarray, real: float, level: float, settings: Mapping[str, Any]
) -> dict[str, float]:
    median = float(np.median(values))
    return {
        "runs": float(len(set_.objective)),
        "measured": float(len(values)),
        "median": median,
        "ratio": (median - level) / (real - level),
        "profitable_share": float(set_.profitable[set_.ran].mean()) if set_.ran.any() else math.nan,
        "percentile": float((values < real).mean()),
        "top_limit": float(np.quantile(values, settings["top"])),
    }


def _candles(
    subject: BarRun, settings: Mapping[str, Any], seed: int
) -> tuple[CandlesBlock, CandlesBlock]:
    """The run's last base bars of its first symbol beside the same bars of the first
    noisy copy (run 0 of the check's stream)."""
    bars = subject.bars
    copy = noisy(bars.frames, bars.ticks, noise_of(settings), seed, 0)
    real = bars.frames[0].iloc[-CANDLES:]
    moved = copy[0].iloc[-CANDLES:]
    symbol = bars.symbols[0]
    return (
        CandlesBlock("The run's bars", f"{symbol}, its last {len(real)} base bars", real),
        CandlesBlock("One noisy copy", f"{symbol}, the same bars with noise", moved),
    )


def _blocks(
    subject: BarRun,
    set_: RerunSet,
    summary: Mapping[str, float],
    name: str,
    settings: Mapping[str, Any],
    seed: int,
    keep_line: float,
) -> tuple:
    """The fan, the histogram with the run, its top percentile and the keep line the
    median must reach, the candles and the summary."""
    fan = session_fan("Noise equity", "The strategy on each noisy copy", set_, subject)
    marks = (
        Mark(summary["top_limit"], f"{words.percent(settings['top'], 0)} percentile", LIMIT),
        Mark(keep_line, f"the median keeps {words.percent(settings['keep'], 0)}", LIMIT),
    )
    histogram = objective_histogram(
        "Noise objectives", f"The {name} of each noisy copy", set_.objective, subject.real, marks
    )
    tiles = (
        Tile("Median kept", words.percent(summary["ratio"]), f"of the run's {name}"),
        Tile("Profitable", words.percent(summary["profitable_share"]), "of the noise runs"),
        Tile("The run's percentile", words.percent(summary["percentile"]), "noise runs below it"),
        Tile("Median", words.number(summary["median"]), name),
        Tile(
            "Measured",
            words.whole(int(summary["measured"]), "run"),
            f"of {words.number(summary['runs'], 0)}",
        ),
    )
    summary_block = TilesBlock("Summary", "Over the noise runs", tiles)
    return ((fan, histogram), _candles(subject, settings, seed), (summary_block,))


__all__ = ["CANDLES", "SETTINGS", "noise_of", "rule", "run"]

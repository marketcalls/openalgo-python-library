"""Out of sample (design 17, section 2.1): does it work on bars the search never saw?

The run's sessions are split in time (`_windows.split`): the first `in_sample` (70
percent) in sample, the rest out of sample. With parameter ranges (given or declared with
`bt.param`), every combination reruns on the in-sample window and the one with the
highest objective (the first of equal ones) reruns once on the out-of-sample window;
without, the strategy as given reruns on each. Each part is an ordinary backtest of its
window, its warm-up read from the bars before it, opening flat with the run's capital.

Figure: the out-of-sample objective over the in-sample one, both per year (`_walk.annual`:
a total such as the net profit over the years each part lasted) and measured from the
objective's break-even. Pass: the out-of-sample run profitable (a net profit of at least
half a paisa) and the ratio at least `keep` (0.5).

Not judged when the run has too few closed trades or did not make money, when its
objective is undefined, has no break-even or is not above it, when either part would
hold fewer than two sessions, when the in-sample part did not make more than break-even
(nothing to keep), or when an out-of-sample objective is undefined.

Charts: the price with the out-of-sample sessions shaded; the equity of the two parts
stitched (`_stitch`) with the out-of-sample sessions shaded, so its start is marked; a
table of in sample against out of sample (CAGR, Sharpe, profit factor, win rate, maximum
drawdown, trades per year, and the objective per year).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ConfigError
from . import _words as words
from ._bar_common import break_even
from ._blocks import SessionsBlock, TableBlock
from ._part_objective import objective_name
from ._rerun_one import RerunRow
from ._reruns import Family, Later
from ._search_common import objective_unjudged, params_text, search_unjudged
from ._search_setup import SearchRun
from ._setting import Setting, between
from ._stitch import stitched_equity
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES
from ._verdict import FAILED, PASSED, Outcome, not_judged
from ._walk import annual, best
from ._windows import Window, split

SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "in_sample",
        0.7,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "the share of the sessions in sample, the first ones",
    ),
    Setting(
        "keep",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the in-sample objective the out-of-sample run must keep",
    ),
)
#: The rows of the table of in sample against out of sample: label, figure, and how it
#: reads ("percent", "number").
TABLE_ROWS = (
    ("CAGR", "cagr", "percent"),
    ("Sharpe", "sharpe", "number"),
    ("Profit factor", "profit_factor", "number"),
    ("Win rate", "win_rate", "percent"),
    ("Max drawdown", "max_drawdown", "percent"),
    ("Trades per year", "trades_per_year", "number"),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"the out-of-sample run is profitable and keeps {words.percent(settings['keep'], 0)} "
        f"of the in-sample {objective_name(objective)} per year (the last "
        f"{words.percent(1.0 - settings['in_sample'], 0)} of the sessions out of sample)"
    )


def parts(subject: SearchRun, settings: Mapping[str, Any]) -> tuple[Window, Window]:
    """(in sample, out of sample) of the run's sessions. Raises ConfigError for too few."""
    return split(len(subject.dates), settings["in_sample"])


def unjudged(subject: SearchRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None."""
    text = rule(settings, objective)
    short = search_unjudged(subject, settings, text) or objective_unjudged(
        subject, objective, text, "out-of-sample"
    )
    if short is not None:
        return short
    try:
        parts(subject, settings)
    except ConfigError as error:
        return not_judged(text, f"Too few sessions to judge: {error}.")
    return None


def _searched(subject: SearchRun, inside: Window) -> list[Family]:
    grid = subject.grid
    return [subject.family(grid, k, inside) for k in range(len(grid.combos))]


def reruns(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family | Later, int]]:
    """(family, runs) of what the check reruns: each combination in sample with ranges,
    then the chosen one on both parts; none when it is not judged before any rerun."""
    if unjudged(subject, settings, objective) is not None:
        return []
    inside, outside = parts(subject, settings)
    if subject.grid is None:
        return [
            (subject.family(window=inside, detail=True), 1),
            (subject.family(window=outside, detail=True), 1),
        ]
    # The chosen combination's two runs are known once the search has run.
    chosen = Later("out_of_sample", "the combination searched best, on both parts")
    return [(family, 1) for family in _searched(subject, inside)] + [(chosen, 2)]


def _chosen(subject: SearchRun, inside: Window) -> tuple[int | None, pd.DataFrame]:
    """(the grid position searched best in sample, a row per combination searched)."""
    grid = subject.grid
    if grid is None:
        return None, pd.DataFrame()
    rows = subject.reruns.firsts(_searched(subject, inside))
    values = np.array([row.objective for row in rows], dtype=float)
    frame = pd.DataFrame({name: [combo[name] for combo in grid.combos] for name in grid.names})
    frame["in_sample"] = values
    frame["in_sample_net_profit"] = [row.net_profit for row in rows]
    frame.index.name = "combination"
    return best(values), frame


def run(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    inside, outside = parts(subject, settings)
    position, table = _chosen(subject, inside)
    if subject.grid is not None and position is None:
        return not_judged(text, "No combination has a defined objective in sample.")
    grid = subject.grid
    first, second = subject.reruns.firsts(
        [
            subject.family(grid, position, inside, detail=True),
            subject.family(grid, position, outside, detail=True),
        ]
    )
    level, name = break_even(objective), objective_name(objective)
    in_year = annual(first.objective, first.years, objective)
    out_year = annual(second.objective, second.years, objective)
    chosen = "" if grid is None else f" with {params_text(grid.swept(position))}, searched best"
    if first.failure or second.failure:
        return not_judged(text, f"A part did not run: {first.failure or second.failure}.")
    if not in_year > level:
        return not_judged(
            text,
            f"Nothing to keep: in sample{chosen} the {name} per year is "
            f"{words.number(in_year)}, not above {level:g}.",
        )
    if not math.isfinite(out_year):
        return not_judged(text, f"The out-of-sample {name} is undefined.")
    ratio = (out_year - level) / (in_year - level)
    profitable = bool(second.net_profit >= BREAK_EVEN_RUPEES)
    passed = profitable and ratio >= settings["keep"]
    summary = {
        "in_sample": in_year,
        "out_of_sample": out_year,
        "ratio": ratio,
        "out_of_sample_net_profit": second.net_profit,
        "in_sample_sessions": float(inside.sessions),
        "out_of_sample_sessions": float(outside.sessions),
    }
    figure = (
        f"out of sample the {name} per year is {words.number(out_year)}, "
        f"{words.percent(ratio)} of the in-sample {words.number(in_year)}; the run made "
        f"{words.rupees(second.net_profit)} out of sample"
    )
    note = (
        f"The first {words.whole(inside.sessions, 'session')} are in sample and the last "
        f"{words.whole(outside.sessions, 'session')} out of sample, each part an ordinary "
        f"backtest of its window that opens flat with the run's capital{chosen}. "
        f"The ratio is measured from {level:g}, a run that makes nothing."
    )
    frame = _frame(table, first, second, position)
    blocks = _blocks(subject, first, second, outside, name, in_year, out_year)
    return Outcome(PASSED if passed else FAILED, figure, text, note, frame, summary, blocks)


def _frame(
    searched: pd.DataFrame, first: RerunRow, second: RerunRow, position: int | None
) -> pd.DataFrame:
    """A row per part (in sample, out of sample) with its figures, and the search's rows
    after them when there was a search."""
    parts_ = pd.DataFrame(
        [
            {"part": "in sample", "objective": first.objective, **first.detail.figures},
            {"part": "out of sample", "objective": second.objective, **second.detail.figures},
        ]
    )
    if searched.empty:
        return parts_
    searched = searched.assign(chosen=[k == position for k in range(len(searched))])
    return pd.concat([parts_, searched.reset_index()], ignore_index=True)


def _blocks(
    subject: SearchRun,
    first: RerunRow,
    second: RerunRow,
    outside: Window,
    name: str,
    in_year: float,
    out_year: float,
) -> tuple:
    dates = pd.DatetimeIndex(subject.dates)
    marked = np.arange(len(dates)) >= outside.first
    symbol, closes = subject.bars.session_closes(dates)
    price = SessionsBlock(
        "Two parts in time",
        f"{symbol} close; shaded sessions are out of sample",
        dates,
        closes,
        marked,
        percent=False,
        line=f"{symbol}'s close at each session close",
        shaded="out of sample",
    )
    equity = stitched_equity([first.detail, second.detail], subject.capital)
    days = pd.DatetimeIndex(equity.index).tz_localize(None).normalize()
    grown = pd.Series(equity.to_numpy() / subject.capital - 1.0, index=days)
    values = grown.reindex(dates, method="ffill").fillna(0.0).to_numpy(dtype=float)
    curve = SessionsBlock(
        "Equity of the two parts",
        "In sample, then out of sample from its start",
        dates,
        values,
        marked,
        percent=True,
        line="the two parts' cumulative return on the capital, out of sample after in sample",
        shaded="out of sample",
    )
    return ((price, curve), (_compare(first, second, name, in_year, out_year),))


def _compare(
    first: RerunRow, second: RerunRow, name: str, in_year: float, out_year: float
) -> TableBlock:
    rows = [
        (label, *(_shown(part.detail.figures[key], how) for part in (first, second)))
        for label, key, how in TABLE_ROWS
    ]
    rows.append((f"{name.capitalize()} per year", words.number(in_year), words.number(out_year)))
    return TableBlock(
        "In sample against out of sample",
        "Each part's figures",
        ("Figure", "In sample", "Out of sample"),
        tuple(rows),
    )


def _shown(value: float, how: str) -> str:
    return words.percent(value) if how == "percent" else words.number(value)


__all__ = ["SETTINGS", "TABLE_ROWS", "parts", "reruns", "rule", "run", "unjudged"]

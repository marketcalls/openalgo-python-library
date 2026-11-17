"""Random out of sample (design 17, section 2.2): was the out-of-sample period just lucky?

One backtest over the whole period, no rerun. Each of `runs` assignments puts a random
`oos` share of the run's sessions (rounded to whole sessions, at least one in each part)
out of sample and the rest in sample, and every closed trade belongs to the part of its
entry session. The run's objective is measured on each part (`_part_objective`).

An assignment keeps its in-sample figure when the in-sample objective is above zero and
the out-of-sample one is at least `keep` times it; an assignment whose in-sample
objective is zero, negative or undefined does not keep (a strategy that made nothing in
sample has nothing to keep). Figure: the share of assignments that keep. Pass: at least
`pass_share`.

Draws: assignment r orders the sessions by draws r x sessions onwards of the check's
stream (`_random.orders`) and puts the first of that order out of sample.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import _random
from . import _words as words
from ._blocks import REFERENCE, HistogramBlock, Mark, SessionsBlock, Tile, TilesBlock
from ._part_objective import MEASURED, Part, objective_name, part_measure
from ._paths import DRAWS_PER_CHUNK
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUNS, too_few
from ._trades import TradeRun
from ._verdict import FAILED, PASSED, Outcome, not_judged

SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    Setting(
        "oos",
        0.30,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "the share of the sessions each assignment puts out of sample",
    ),
    Setting(
        "keep",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the in-sample objective the out-of-sample one must keep",
    ),
    Setting(
        "pass_share",
        0.8,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of assignments that must keep it",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"at least {words.percent(settings['pass_share'], 0)} of assignments keep "
        f"{words.percent(settings['keep'], 0)} of the in-sample {objective_name(objective)} "
        f"out of sample ({words.percent(settings['oos'], 0)} of the sessions out of sample)"
    )


def _assignments(
    trades: TradeRun, measure: Callable[[Part], float], held_out: int, seed: int, runs: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """A row per assignment, and the out-of-sample sessions of the first."""
    count = len(trades.sessions)
    session_pnl = np.bincount(trades.entry_session, weights=trades.pnl, minlength=count)
    rows, first_marked = [], np.zeros(count, dtype=bool)
    for first, size in _random.run_chunks(runs, count, DRAWS_PER_CHUNK):
        for offset, order in enumerate(_random.orders(seed, first, size, count)):
            marked = np.zeros(count, dtype=bool)
            marked[order[:held_out]] = True
            if first + offset == 0:
                first_marked = marked
            out_trade = marked[trades.entry_session]
            inside = measure(Part(session_pnl[~marked], trades.pnl[~out_trade]))
            outside = measure(Part(session_pnl[marked], trades.pnl[out_trade]))
            rows.append((int((~out_trade).sum()), int(out_trade.sum()), inside, outside))
    columns = ["in_sample_trades", "out_of_sample_trades", "in_sample", "out_of_sample"]
    table = pd.DataFrame(rows, columns=columns)
    table.index.name = "run"
    table["difference"] = table["out_of_sample"] - table["in_sample"]
    return table, first_marked


def _kept(table: pd.DataFrame, keep: float) -> pd.Series:
    inside, outside = table["in_sample"], table["out_of_sample"]
    # NaN compares False, so an undefined figure never keeps.
    return (inside > 0) & (outside >= keep * inside)


def run(
    trades: TradeRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = too_few(trades, settings["min_trades"], text)
    if short is not None:
        return short
    name = objective_name(objective)
    measure = part_measure(objective, trades)
    if measure is None:
        return not_judged(
            text,
            f"The objective {name} cannot be measured on a part of the trades; "
            f"give objective= one of {', '.join(MEASURED)} to judge this check.",
        )
    count = len(trades.sessions)
    held_out = round(settings["oos"] * count)
    if not 1 <= held_out < count:
        return not_judged(
            text,
            f"Too few sessions to judge: {words.whole(count, 'session')} leave no session in "
            f"one of the parts at {words.percent(settings['oos'], 0)} out of sample.",
        )
    table, first_marked = _assignments(trades, measure, held_out, seed, runs)
    table["kept"] = _kept(table, settings["keep"])
    share = float(table["kept"].mean())
    verdict = PASSED if share >= settings["pass_share"] else FAILED
    summary = {
        "runs": float(runs),
        "kept_share": share,
        "out_of_sample_sessions": float(held_out),
        "sessions": float(count),
        "median_in_sample": float(table["in_sample"].median()),
        "median_out_of_sample": float(table["out_of_sample"].median()),
        "median_difference": float(table["difference"].median()),
    }
    figure = f"{words.percent(share)} of {words.whole(runs, 'assignment')} kept it"
    note = (
        f"Each assignment puts {words.whole(held_out, 'session')} of {count} out of sample "
        f"at random; a closed trade belongs to the part of its entry session, and {name} is "
        "measured on each part. An assignment whose in-sample figure is not above zero "
        "does not keep."
    )
    blocks = _blocks(trades, table, first_marked, summary, name)
    return Outcome(verdict, figure, text, note, table, summary, blocks)


def _blocks(
    trades: TradeRun,
    table: pd.DataFrame,
    marked: np.ndarray,
    summary: dict[str, float],
    name: str,
) -> tuple:
    differences = table["difference"].to_numpy(dtype=float)
    shown = differences[np.isfinite(differences)]
    tiles = (
        Tile("Kept", words.percent(summary["kept_share"]), "of the assignments"),
        Tile("Median in sample", words.number(summary["median_in_sample"]), name),
        Tile("Median out of sample", words.number(summary["median_out_of_sample"]), name),
        Tile("Median difference", words.number(summary["median_difference"]), "out minus in"),
        Tile(
            "Out of sample",
            words.whole(int(summary["out_of_sample_sessions"]), "session"),
            f"of {int(summary['sessions'])}",
        ),
        Tile(
            "Undefined",
            words.whole(int((~np.isfinite(differences)).sum()), "assignment"),
            "a part with no figure",
        ),
    )
    return (
        (
            SessionsBlock(
                "One assignment",
                "The run's cumulative return; shaded sessions are out of sample",
                trades.sessions,
                trades.cumulative,
                marked,
            ),
            HistogramBlock(
                "Out of sample minus in sample",
                f"The {name} of each assignment",
                shown,
                False,
                (Mark(0.0, "no change", REFERENCE),),
            ),
        ),
        (TilesBlock("Summary", "Over the assignments", tiles),),
    )


__all__ = ["SETTINGS", "rule", "run"]

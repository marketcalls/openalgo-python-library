"""Random out of sample (design 17, section 2.2): was the out-of-sample period just lucky?

One backtest over the whole period, no rerun. Each of `runs` assignments puts a random
`oos` share of the run's sessions out of sample and the rest in sample, and every closed
trade belongs to the part of its entry session. The share is rounded to whole sessions
(halves to even, Python's `round`). The run's objective is measured on each part
(`_part_objective`), and the check is not judged when the run did not make money
(`_trade_common.no_edge`), or when a part would hold fewer sessions than the objective
needs (two for a ratio of session returns).

An assignment can be judged when its out-of-sample part holds a closed trade and the
objective of both parts is defined; one that cannot (a part with no trade, or whose
session returns never vary) says nothing either way, so it is left out rather than
counted as not keeping. When fewer than `pass_share` of the assignments can be judged,
the check is not judged and says how many could; the summary shows both counts.

A judged assignment keeps its in-sample figure when the in-sample objective is above its
break-even figure (0, or 1 for a profit factor) and the out-of-sample one keeps at least
`keep` times the in-sample one's part above break-even; one whose in-sample objective is
at or below break-even does not keep (a strategy that made nothing in sample has nothing
to keep). Figure: the share of the judged assignments that keep. Pass: at least
`pass_share`.

Chart: when the report has the run's bars (`TradeRun.prices`, loaded for the checks that
rerun the backtest), the strategy bars' close at each session close with one
assignment's out-of-sample sessions shaded; without them, the run's cumulative return,
and the chart's key says why.

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
from ._part_objective import (
    BREAK_EVEN,
    MEASURED,
    NO_BREAK_EVEN,
    Measure,
    Part,
    objective_name,
    part_measure,
)
from ._paths import DRAWS_PER_CHUNK
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUNS, no_edge, too_few
from ._trades import TradeRun
from ._verdict import FAILED, NOT_JUDGED, PASSED, Outcome, not_judged

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
    name = objective_name(objective)
    break_even = BREAK_EVEN.get(name, 0.0)
    above = f" above {break_even:g}" if break_even else ""
    return (
        f"at least {words.percent(settings['pass_share'], 0)} of assignments keep "
        f"{words.percent(settings['keep'], 0)} of the in-sample {name}{above} out of sample "
        f"({words.percent(settings['oos'], 0)} of the sessions out of sample)"
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


def kept(table: pd.DataFrame, keep: float, break_even: float) -> pd.Series:
    """Whether each assignment of `table` (its "in_sample" and "out_of_sample" figures)
    keeps its in-sample figure: the in-sample figure is above `break_even` and the
    out-of-sample one's part above break-even is at least `keep` times the in-sample
    one's."""
    inside, outside = table["in_sample"], table["out_of_sample"]
    gain = inside - break_even
    # NaN compares False, so an undefined figure never keeps.
    return (gain > 0) & (outside - break_even >= keep * gain)


def _unmeasured(name: str) -> str:
    """Why an objective cannot judge the check, and what to give instead."""
    choices = ", ".join(MEASURED)
    if name in NO_BREAK_EVEN:
        return (
            f"The objective {name} cannot judge this check: {NO_BREAK_EVEN[name]}; give "
            f"objective= one of {choices} to judge it."
        )
    return (
        f"The objective {name} cannot be measured on a part of the trades; give "
        f"objective= one of {choices} to judge this check."
    )


def _too_few_sessions(count: int, held_out: int, need: int, oos: float, name: str) -> str:
    return (
        f"Too few sessions to judge: {words.percent(oos, 0)} of "
        f"{words.whole(count, 'session')} puts {held_out} out of sample and "
        f"{count - held_out} in sample, and the {name} of a part needs "
        f"{words.whole(need, 'session')} in each part."
    )


def run(
    trades: TradeRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = too_few(trades, settings["min_trades"], text) or no_edge(trades, text)
    if short is not None:
        return short
    name = objective_name(objective)
    measure = part_measure(objective, trades)
    if measure is None:
        return not_judged(text, _unmeasured(name))
    count = len(trades.sessions)
    held_out = round(settings["oos"] * count)
    if not measure.min_sessions <= held_out <= count - measure.min_sessions:
        why = _too_few_sessions(count, held_out, measure.min_sessions, settings["oos"], name)
        return not_judged(text, why)
    table, first_marked = _assignments(trades, measure.figure, held_out, seed, runs)
    table["judged"] = judgeable(table)
    table["kept"] = table["judged"] & kept(table, settings["keep"], measure.break_even)
    judged = int(table["judged"].sum())
    if not judged:
        why = (
            f"No assignment has the {name} of both parts defined (a part with no trade, or "
            "whose session returns never vary), so nothing can be judged."
        )
        return not_judged(text, why)
    if judged < settings["pass_share"] * runs:
        why = (
            f"Too little to judge: {words.number(judged, 0)} of {words.whole(runs, 'assignment')} "
            f"put a trade out of sample with the {name} of both parts defined, and the rule "
            f"needs {words.percent(settings['pass_share'], 0)} of them."
        )
        summary = {"runs": float(runs), "judged_assignments": float(judged)}
        return Outcome(NOT_JUDGED, "n/a", text, why, table, summary)
    return _judged(trades, table, first_marked, settings, measure, name, text)


def judgeable(table: pd.DataFrame) -> pd.Series:
    """Whether each assignment of `table` can be judged: its out-of-sample part holds a
    closed trade and both its figures are defined."""
    defined = table["in_sample"].notna() & table["out_of_sample"].notna()
    return (table["out_of_sample_trades"] > 0) & defined


def _judged(
    trades: TradeRun,
    table: pd.DataFrame,
    first_marked: np.ndarray,
    settings: Mapping[str, Any],
    measure: Measure,
    name: str,
    text: str,
) -> Outcome:
    """The verdict, figure, note and blocks of assignments whose kept column is set."""
    runs, count, held_out = len(table), len(trades.sessions), int(first_marked.sum())
    judged = table[table["judged"]]
    share = float(judged["kept"].mean())
    verdict = PASSED if share >= settings["pass_share"] else FAILED
    summary = {
        "runs": float(runs),
        "judged_assignments": float(len(judged)),
        "kept_share": share,
        "out_of_sample_sessions": float(held_out),
        "sessions": float(count),
        "median_in_sample": float(judged["in_sample"].median()),
        "median_out_of_sample": float(judged["out_of_sample"].median()),
        "median_difference": float(judged["difference"].median()),
    }
    figure = (
        f"{words.percent(share)} of {words.whole(len(judged), 'judged assignment')} kept "
        f"{words.percent(settings['keep'], 0)} of the in-sample {name}"
    )
    judged_count, assignments = words.number(len(judged), 0), words.whole(runs, "assignment")
    note = (
        f"Each assignment puts {words.whole(held_out, 'session')} of {count} out of sample "
        f"at random; a closed trade belongs to the part of its entry session, and {name} is "
        f"measured on each part. {judged_count} of the {assignments} "
        f"put a trade out of sample with the {name} of both parts defined and are judged; "
        f"one whose in-sample figure is not above {measure.break_even:g}, its break-even, "
        "does not keep."
    )
    blocks = _blocks(trades, judged, first_marked, summary, name)
    return Outcome(verdict, figure, text, note, table, summary, blocks)


def assignment_chart(trades: TradeRun, marked: np.ndarray) -> SessionsBlock:
    """One assignment's out-of-sample sessions shaded on the price when the report has
    the run's bars (`TradeRun.prices`), else on the run's cumulative return."""
    if trades.prices is not None:
        symbol, closes = trades.prices
        return SessionsBlock(
            "One assignment",
            f"{symbol} close; shaded sessions are out of sample",
            trades.sessions,
            closes,
            marked,
            percent=False,
            line=f"{symbol}'s close at each session close, the close of its last strategy bar",
        )
    return SessionsBlock(
        "One assignment",
        "The run's cumulative return; shaded sessions are out of sample",
        trades.sessions,
        trades.cumulative,
        marked,
        percent=True,
        line=(
            "the run's cumulative return at each session close; the price is drawn when "
            "the report has the run's bars, which it loads for the checks that rerun the "
            "backtest"
        ),
    )


def _blocks(
    trades: TradeRun,
    judged: pd.DataFrame,
    marked: np.ndarray,
    summary: dict[str, float],
    name: str,
) -> tuple:
    differences = judged["difference"].to_numpy(dtype=float)
    shown = differences[np.isfinite(differences)]
    tiles = (
        Tile("Kept", words.percent(summary["kept_share"]), "of the judged assignments"),
        Tile("Median in sample", words.number(summary["median_in_sample"]), name),
        Tile("Median out of sample", words.number(summary["median_out_of_sample"]), name),
        Tile("Median difference", words.number(summary["median_difference"]), "out minus in"),
        Tile(
            "Out of sample",
            words.whole(int(summary["out_of_sample_sessions"]), "session"),
            f"of {int(summary['sessions'])}",
        ),
        Tile(
            "Judged",
            words.whole(int(summary["judged_assignments"]), "assignment"),
            f"of {words.number(summary['runs'], 0)}: a trade out of sample, both figures",
        ),
    )
    return (
        (
            assignment_chart(trades, marked),
            HistogramBlock(
                "Out of sample minus in sample",
                f"The {name} of each judged assignment",
                shown,
                False,
                (Mark(0.0, "no change", REFERENCE),),
            ),
        ),
        (TilesBlock("Summary", "Over the assignments", tiles),),
    )


__all__ = ["SETTINGS", "assignment_chart", "judgeable", "kept", "rule", "run"]

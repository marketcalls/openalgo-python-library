"""Shifted bars (design 17, section 2.7): does it depend on where the bars start?

The strategy reruns on its own bars with the boundaries of its strategy bars moved
(`_settings.RunPlan.bar_shift`, the kernel's `engine::shifted_bars`): every strategy bar
ends k base bars earlier. An intraday interval of n base bars has the n - 1 shifts
k = 1 to n - 1 (a 5-minute strategy on 1-minute bars has four); a daily strategy on
intraday base bars has days that end `minutes` (15, 30, 45 and 60) earlier, each a whole
number of base bars shorter than the shortest session of the run (other minutes are left
out, and the note says so), its decision made that much before the close with the day's
bars so far and its later bars opening the next day's bar. Such a day opens at the
session before's close less the shift, so its `bar.time` is on that session's date (its
`close_time` on its own), and the run's first day starts at its session's open: the
last minutes of the session before the run belong to no bar (the history keeps the bars
closed before the run starts, the run builds its bars from its own sessions). The note
says both (`DAYS_NOTE`).

Figures, over the shifts that ran: the share profitable (net profit of at least half a
paisa), and the median shifted objective over the run's, both measured from the
objective's break-even. Pass: every shift profitable and the ratio at least `keep` (0.5).

Not judged when the strategy bars are the base bars (there is nothing to shift: a daily
strategy on daily bars, or an intraday one on bars of its own interval), when no shift
can be made, when the run did not make money or has too few closed trades, when its
objective is undefined, has no break-even or is not above it, when the strategy was made
by `bt.from_arrays` (its arrays hold one decision per strategy bar of the run, which
shifted bars do not keep), or when no shift ran.

Charts: the equity of each shift with the run's bold; candles of the run's last
`CANDLES` strategy bars beside the same stretch with the first shift (drawn with the
package calendar); a table by shift.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .._arrays import made_by
from .._calendar_convert import interval_nanos
from .._layout import NANOS_PER_SECOND, parse_interval
from ..resample import shifted_bars
from . import _words as words
from ._bar_common import BarRun, RerunSet
from ._blocks import CandlesBlock, Tile, TilesBlock
from ._part_objective import objective_name
from ._reruns import REAL, Family
from ._setting import Setting, between
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES
from ._variant_common import (
    labelled_table,
    objective_unjudged,
    ratios,
    run_unjudged,
    variant_set,
    variants_chart,
    variants_table,
)
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: The strategy bars the candle charts show.
CANDLES = 40
#: What a daily strategy is told of its days that end earlier (module docstring).
DAYS_NOTE = (
    " A day that ends earlier opens at the close of the session before, less the shift, so "
    "its bar.time is on the date of the session before (its close_time on its own date): a "
    "rule on the date of bar.time reads the day before in these runs. The last minutes of "
    "the session before the run belong to no bar, so the run's first day starts at its "
    "session's open."
)
NANOS_PER_MINUTE = 60 * NANOS_PER_SECOND


def _minutes(value: Any) -> tuple[int, ...] | None:
    """The `minutes` setting: a non-empty list of whole minutes above 0, in order."""
    if isinstance(value, str) or not isinstance(value, list | tuple | range) or not value:
        return None
    if not all(isinstance(m, int) and not isinstance(m, bool) and m > 0 for m in value):
        return None
    return tuple(sorted(set(value)))


SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "minutes",
        (15, 30, 45, 60),
        _minutes,
        "a list of whole minutes above 0",
        "how much earlier a daily strategy's days end",
    ),
    Setting(
        "keep",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the run's objective the median shift must keep",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"every shift profitable, and the median shift keeps "
        f"{words.percent(settings['keep'], 0)} of the run's {objective_name(objective)}"
    )


def shifts(subject: BarRun, settings: Mapping[str, Any]) -> tuple[tuple[int, ...], str]:
    """(the shifts in base bars, why there are none or some minutes are left out)."""
    strategy = subject.result._record.strategy
    base = subject.bars.base
    kind = parse_interval(strategy.interval)[0]
    if base == strategy.interval or parse_interval(base)[0] != "intraday":
        return (), (
            f"The strategy bars are the base bars ({strategy.interval} on {base}), so there is "
            "nothing to shift: run the strategy on finer base bars (base='1m') to judge it."
        )
    step = interval_nanos(base)
    if kind == "intraday":
        return tuple(range(1, interval_nanos(strategy.interval) // step)), ""
    sessions = subject.bars.sessions
    shortest = min((s.close - s.open) // step for s in sessions) if sessions else 0
    made, left = [], []
    for minutes in settings["minutes"]:
        bars, rest = divmod(minutes * NANOS_PER_MINUTE, step)
        (left if rest or not 0 < bars < shortest else made).append((minutes, int(bars)))
    why = ""
    if left:
        named = ", ".join(str(minutes) for minutes, _ in left)
        why = (
            f" Days ending {named} minutes earlier are left out: a shift is a whole number "
            f"of {base} base bars, shorter than the shortest session."
        )
    return tuple(bars for _, bars in made), why


def _family(shift: int) -> Family:
    return Family(REAL, plan=(("bar_shift", shift),))


def unjudged(subject: BarRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None:
    the one source of whether the check reruns (`reruns`)."""
    text = rule(settings, objective)
    made, why = shifts(subject, settings)
    if not made:
        return not_judged(text, why.strip() or "No shift can be made.")
    short = run_unjudged(subject, settings, text)
    if short is None and made_by(subject.result._record.strategy) == "bt.from_arrays":
        short = not_judged(
            text,
            "Shifted bars cannot judge this strategy: its decisions are arrays fixed in "
            "advance (bt.from_arrays), one for each strategy bar of the run, which shifted "
            "bars do not keep.",
        )
    return short or objective_unjudged(subject, objective, text, "shifted")


def reruns(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: one run of each shift, none when it is
    not judged before any rerun (`unjudged`)."""
    if unjudged(subject, settings, objective) is not None:
        return []
    return [(_family(shift), 1) for shift in shifts(subject, settings)[0]]


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    made, left_out = shifts(subject, settings)
    set_ = variant_set(subject, [_family(shift) for shift in made])
    if not set_.ran.any():
        return not_judged(text, "No shifted run ran." + set_.failure_note())
    name = objective_name(objective)
    ratio = ratios(set_.objective, subject.real, objective)
    summary = _summary(set_, ratio)
    passed = summary["profitable_share"] == 1.0 and summary["ratio"] >= settings["keep"]
    figure = (
        f"{words.percent(summary['profitable_share'])} of "
        f"{words.whole(int(summary['shifts']), 'shift')} profitable; the median shift keeps "
        f"{words.percent(summary['ratio'])} of the run's {name} of {words.number(subject.real)}"
    )
    daily = parse_interval(subject.result._record.strategy.interval)[0] != "intraday"
    note = (
        f"Each run rebuilt the strategy bars from the run's own {subject.bars.base} bars, "
        "every bar ending that many base bars earlier, and reran the strategy on them."
        + left_out
        + (DAYS_NOTE if daily else "")
        + set_.failure_note()
    )
    labels = [_label(subject, shift) for shift in made]
    table = labelled_table(set_, ratio, shift=made, label=labels)
    blocks = _blocks(subject, set_, ratio, labels, made, summary, name)
    return Outcome(PASSED if passed else FAILED, figure, text, note, table, summary, blocks)


def _label(subject: BarRun, shift: int) -> str:
    """A shift in words: base bars earlier, and minutes too for a daily strategy."""
    base = subject.bars.base
    minutes = shift * interval_nanos(base) // NANOS_PER_MINUTE
    return f"{words.whole(shift, 'base bar')} earlier ({minutes} min)"


def _summary(set_: RerunSet, ratio: np.ndarray) -> dict[str, float]:
    ran = set_.ran
    measured = ran & np.isfinite(ratio)
    return {
        "shifts": float(ran.sum()),
        "profitable_share": float((set_.net_profit[ran] >= BREAK_EVEN_RUPEES).mean()),
        "ratio": float(np.median(ratio[measured])) if measured.any() else math.nan,
        "median_objective": float(np.median(set_.objective[set_.measured]))
        if set_.measured.any()
        else math.nan,
    }


def _candles(subject: BarRun, shift: int) -> tuple[CandlesBlock, CandlesBlock]:
    """The run's last strategy bars of its first symbol beside the same stretch with
    `shift`."""
    strategy = subject.result._record.strategy
    base, bars = subject.bars.base, subject.bars.run_part(0)
    made = [
        shifted_bars(
            bars,
            strategy.interval,
            exchange=strategy.exchange,
            source_interval=base,
            on_closed_day="drop",
            shift_nanos=k * interval_nanos(base),
        )
        for k in (0, shift)
    ]
    first = int(made[0]["ts_open"].iloc[-min(CANDLES, len(made[0]))])
    shifted = made[1][made[1]["ts_close"].to_numpy() > first]
    symbol = subject.bars.symbols[0]
    return (
        CandlesBlock(
            "The run's strategy bars", f"{symbol}, its last {CANDLES}", made[0].iloc[-CANDLES:]
        ),
        CandlesBlock("Shifted", f"{symbol}, {_label(subject, shift)}", shifted),
    )


def _blocks(
    subject: BarRun,
    set_: RerunSet,
    ratio: np.ndarray,
    labels: list[str],
    made: tuple[int, ...],
    summary: Mapping[str, float],
    name: str,
) -> tuple:
    chart = variants_chart(
        "Equity of each shift",
        "The strategy on bars ending earlier; the run bold",
        [("shifted bars", range(len(made)))],
        set_,
        subject,
    )
    table = variants_table(
        "By shift", "Each shift's run against the run's", labels, set_, ratio, name
    )
    tiles = TilesBlock(
        "Summary",
        "Over the shifts",
        (
            Tile("Profitable", words.percent(summary["profitable_share"]), "of the shifts"),
            Tile("Median kept", words.percent(summary["ratio"]), f"of the run's {name}"),
            Tile("Median", words.number(summary["median_objective"]), name),
        ),
    )
    return ((chart,), _candles(subject, made[0]), (table,), (tiles,))


__all__ = ["CANDLES", "SETTINGS", "reruns", "rule", "run", "shifts", "unjudged"]

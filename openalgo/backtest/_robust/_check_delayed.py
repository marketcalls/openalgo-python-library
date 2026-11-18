"""Delayed orders (design 17, section 2.14): does it survive entering or exiting a bar
late?

The strategy reruns on its own bars with its decided entries held 0 to `most` (2)
strategy bars and its exits 0 to `most` (`bt.backtest(delay_entries=, delay_exits=)`,
the kernel's `engine::delay`), every pair but (0, 0), which is the run itself: eight
reruns by default. The delays are later than the run: they add to the run's own
`delay_entries` and `delay_exits` (`own_delays`), and the note says so. Every decision
about an order is held, its cancellations and modifications as long as its placement, so
an order works on its old terms until a change reaches it. The engine-held stops and
targets rest with the broker and are not held, nor are the engine's own exits.

Figures: each pair's objective over the run's, measured from the objective's break-even.
Pass: the run with entries one bar late (and exits on time) profitable (net profit of at
least half a paisa) and keeping at least `keep` (half) of the run's objective.

Not judged when the run did not make money or has too few closed trades, when its
objective is undefined, has no break-even or is not above it, or when the one-bar entry
delay did not run.

Charts: the equity of each pair, a group for each entry delay, with the run's bold; a
table by pair.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _words as words
from ._bar_common import BarRun, RerunSet
from ._blocks import Tile, TilesBlock
from ._part_objective import objective_name
from ._reruns import REAL, Family
from ._setting import Setting, between, whole
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

#: The pair the pass rule judges: entries one strategy bar late, exits on time.
JUDGED = (1, 0)

SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "most",
        2,
        whole(1),
        "a whole number, 1 or more",
        "the longest delay of entries and of exits, strategy bars",
    ),
    Setting(
        "keep",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the run's objective the one-bar entry delay must keep",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        "with entries one bar late, the run is profitable and keeps "
        f"{words.percent(settings['keep'], 0)} of the run's {objective_name(objective)}"
    )


def pairs(settings: Mapping[str, Any]) -> list[tuple[int, int]]:
    """(entry delay, exit delay) of every rerun, in order: entries first, (0, 0) left
    out."""
    most = settings["most"]
    return [(e, x) for e in range(most + 1) for x in range(most + 1) if (e, x) != (0, 0)]


def own_delays(subject: BarRun) -> tuple[int, int]:
    """(entries, exits): the strategy bars the run itself held its decided orders."""
    arguments = subject.result._record.arguments
    return arguments["delay_entries"], arguments["delay_exits"]


def _family(pair: tuple[int, int], own: tuple[int, int]) -> Family:
    """The rerun with the delays of `pair` added to the run's `own`."""
    entries, exits = own[0] + pair[0], own[1] + pair[1]
    return Family(REAL, arguments=(("delay_entries", entries), ("delay_exits", exits)))


def unjudged(subject: BarRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None:
    the one source of whether the check reruns (`reruns`)."""
    text = rule(settings, objective)
    short = run_unjudged(subject, settings, text)
    return short or objective_unjudged(subject, objective, text, "delayed")


def reruns(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: one run of each pair, none when it is not
    judged before any rerun (`unjudged`)."""
    if unjudged(subject, settings, objective) is not None:
        return []
    own = own_delays(subject)
    return [(_family(pair, own), 1) for pair in pairs(settings)]


def _label(pair: tuple[int, int]) -> str:
    entries, exits = pair
    return f"entries {entries} late, exits {exits} late"


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    made, own = pairs(settings), own_delays(subject)
    set_ = variant_set(subject, [_family(pair, own) for pair in made])
    judged = made.index(JUDGED)
    if not set_.ran[judged]:
        return not_judged(
            text, "The run with entries one bar late did not run." + set_.failure_note()
        )
    name = objective_name(objective)
    ratio = ratios(set_.objective, subject.real, objective)
    profitable = bool(set_.net_profit[judged] >= BREAK_EVEN_RUPEES)
    passed = profitable and bool(ratio[judged] >= settings["keep"])
    summary = _summary(set_, ratio, judged)
    figure = (
        f"with entries one bar late the run made {words.rupees(set_.net_profit[judged])} and "
        f"keeps {words.percent(ratio[judged])} of the run's {name} of "
        f"{words.number(subject.real)}; {words.percent(summary['profitable_share'])} of "
        f"{words.whole(len(made), 'delayed run')} profitable"
    )
    note = (
        "Each run held the strategy's decided entries and exits the strategy bars named "
        "before they could fill"
        + (f", added to the run's own (entries {own[0]}, exits {own[1]})" if any(own) else "")
        + ", and their cancellations and modifications as long; the legs' stops and targets "
        "and the engine's own exits were not held, and an order held past its session's "
        "close (intraday) or the run's last bar was refused (delay_past_close)."
        + set_.failure_note()
    )
    labels = [_label(pair) for pair in made]
    table = labelled_table(
        set_, ratio, delay_entries=[e for e, _ in made], delay_exits=[x for _, x in made]
    )
    blocks = _blocks(subject, set_, ratio, labels, made, summary, name)
    return Outcome(PASSED if passed else FAILED, figure, text, note, table, summary, blocks)


def _summary(set_: RerunSet, ratio: np.ndarray, judged: int) -> dict[str, float]:
    ran = set_.ran
    return {
        "runs": float(ran.sum()),
        "profitable_share": float((set_.net_profit[ran] >= BREAK_EVEN_RUPEES).mean()),
        "entry_late_ratio": float(ratio[judged]),
        "entry_late_net_profit": float(set_.net_profit[judged]),
        "worst_ratio": float(np.nanmin(ratio[ran])) if np.isfinite(ratio[ran]).any() else np.nan,
    }


def _blocks(
    subject: BarRun,
    set_: RerunSet,
    ratio: np.ndarray,
    labels: list[str],
    made: list[tuple[int, int]],
    summary: Mapping[str, float],
    name: str,
) -> tuple:
    entry_delays = sorted({entries for entries, _ in made})
    groups = [
        (
            f"entries {entries} late" if entries else "entries on time, exits late",
            [k for k, (e, _) in enumerate(made) if e == entries],
        )
        for entries in entry_delays
    ]
    chart = variants_chart(
        "Equity with late orders",
        "A group for each entry delay; the run bold",
        groups,
        set_,
        subject,
    )
    table = variants_table(
        "By delay", "Strategy bars late; each run against the run's", labels, set_, ratio, name
    )
    tiles = TilesBlock(
        "Summary",
        "Over the delayed runs",
        (
            Tile("Entries 1 late", words.percent(summary["entry_late_ratio"]), f"of the {name}"),
            Tile("Its net P&L", words.rupees(summary["entry_late_net_profit"]), "rupees"),
            Tile("Profitable", words.percent(summary["profitable_share"]), "of the delayed runs"),
            Tile("Worst kept", words.percent(summary["worst_ratio"]), f"of the {name}"),
        ),
    )
    return ((chart,), (table,), (tiles,))


__all__ = ["JUDGED", "SETTINGS", "own_delays", "pairs", "reruns", "rule", "run", "unjudged"]

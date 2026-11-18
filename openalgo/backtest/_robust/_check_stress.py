"""Synthetic stress (design 17, section 2.9): does it hold through shocks history did not
have?

Candidates come from the noise generator (`noise_candidates`, 200, with the noise check's
default settings) and the permutation generator (`permutation_candidates`, 200), and a
candidate is kept when the stress rules hold for it (`_stress_rules`: a deeper drawdown, a
longer time to a new high, a deeper fall below the moving average, returns little
correlated with the real ones, or a function of (real, candidate) bars), "any" of them
(the default) or "all" (`combine`). The strategy is rerun on every kept candidate.

Figures per rule, over the kept candidates the rule holds for: how many, the share
profitable, the median and average net P&L in rupees. A rule is judged when it holds for
at least `min_kept` (10) kept candidates. Pass: at least `pass_share` (half) of them
profitable under every judged rule. Not judged when no rule is judged, when the run did
not make money, or when it has too few closed trades.

Draws: noise candidate j is run j of part 0 of the check's stream, permutation candidate
j run j of part 1 (`_random.part_seed`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import _random
from . import _words as words
from ._bar_common import BarRun, RerunSet, real_path, rerun_set
from ._blocks import PathGroup, PathsBlock, TableBlock
from ._generate_noisy import Noise, ranges
from ._reruns import NOISY, PERMUTED, Family
from ._setting import Setting, between, whole
from ._stress_rules import BUILT_IN, StressRule, held, prices, rules_value, stress_rules
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES, exact_mean, no_edge, too_few
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: The kept candidates a chart draws for each rule.
DRAWN = 40


def _combine(value: Any) -> str | None:
    return value if value in ("any", "all") else None


SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "noise_candidates", 200, whole(0), "a whole number, 0 or more", "noisy candidates made"
    ),
    Setting(
        "permutation_candidates",
        200,
        whole(0),
        "a whole number, 0 or more",
        "permuted candidates made",
    ),
    Setting(
        "rules",
        BUILT_IN,
        rules_value,
        f"a list of {list(BUILT_IN)} and functions of (real, candidate) bars",
        "the stress rules",
    ),
    Setting("combine", "any", _combine, "'any' or 'all'", "how the rules keep a candidate"),
    Setting(
        "average", 200, whole(2), "a whole number, 2 or more", "sessions of the moving average"
    ),
    Setting(
        "correlation",
        0.5,
        between(-1.0, 1.0, low_open=False, high_open=False),
        "a number from -1 to 1",
        "the correlation of returns below which a candidate is kept",
    ),
    Setting("min_kept", 10, whole(1), "a whole number, 1 or more", "kept candidates a rule needs"),
    Setting(
        "pass_share",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of a rule's kept candidates that must make money",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"at least {words.percent(settings['pass_share'], 0)} of the kept candidates "
        "profitable under every rule"
    )


def _families(seed: int) -> tuple[Family, Family]:
    return (
        Family(NOISY, _random.part_seed(seed, 0), Noise()),
        Family(PERMUTED, _random.part_seed(seed, 1)),
    )


def kept_candidates(
    subject: BarRun, rules: Sequence[StressRule], settings: Mapping[str, Any], seed: int
) -> list[tuple[Family, int, tuple[bool, ...]]]:
    """(family, run, whether each rule holds) of every kept candidate (module docstring)."""
    bars = subject.bars
    real = [prices(frame, bars.first_run_ns) for frame in bars.frames]
    noise_family, permuted_family = _families(seed)
    atrs = [ranges(frame, Noise().period) for frame in bars.frames]
    counts = (
        (noise_family, settings["noise_candidates"]),
        (permuted_family, settings["permutation_candidates"]),
    )
    keep = any if settings["combine"] == "any" else all
    kept = []
    for family, count in counts:
        for run in range(count):
            frames = family.frames(bars, run, atrs)
            flags = held(rules, real, [prices(frame, bars.first_run_ns) for frame in frames])
            if keep(flags):
                kept.append((family, run, flags))
    return kept


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = too_few(subject.trades, settings["min_trades"], text) or no_edge(subject.trades, text)
    if short is not None:
        return short
    rules = stress_rules(settings["rules"], settings)
    kept = kept_candidates(subject, rules, settings, seed)
    rows = []
    for family in dict.fromkeys(family for family, _, _ in kept):
        mine = [run for each, run, _ in kept if each == family]
        rows += subject.reruns.rows(family, mine)
    set_ = rerun_set(rows) if rows else None
    flags = np.array([flags for _, _, flags in kept], dtype=bool).reshape(len(kept), len(rules))
    table = _rule_table(rules, flags, set_, settings)
    judged = table[table["judged"]]
    if not len(judged):
        made = settings["noise_candidates"] + settings["permutation_candidates"]
        why = (
            f"Too few kept candidates to judge: no rule holds for {settings['min_kept']} or "
            f"more of the {words.whole(made, 'candidate')}."
        )
        return not_judged(text, why)
    lowest = judged["profitable_share"].idxmin()
    passed = bool((judged["profitable_share"] >= settings["pass_share"]).all())
    figure = (
        f"{words.whole(len(kept), 'candidate')} kept; the lowest share profitable is "
        f"{words.percent(judged.loc[lowest, 'profitable_share'])} ({lowest}) over "
        f"{words.whole(len(judged), 'judged rule')}"
    )
    note = (
        f"Candidates: {settings['noise_candidates']} noisy and "
        f"{settings['permutation_candidates']} permuted copies of the run's bars, kept when "
        f"{settings['combine']} of the rules hold, each rule judged on the session closes of "
        "every instrument against the real ones." + (set_.failure_note() if set_ else "")
    )
    summary = {"candidates": float(len(kept)), "judged_rules": float(len(judged))}
    blocks = _blocks(subject, rules, flags, set_, table)
    return Outcome(
        PASSED if passed else FAILED, figure, text, note, table.reset_index(), summary, blocks
    )


def _rule_table(
    rules: Sequence[StressRule],
    flags: np.ndarray,
    set_: RerunSet | None,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for k, stress in enumerate(rules):
        mine = flags[:, k] if len(flags) else np.zeros(0, dtype=bool)
        pnl = set_.net_profit[mine] if set_ is not None else np.zeros(0)
        pnl = pnl[np.isfinite(pnl)]
        rows.append(
            {
                "rule": stress.name,
                "meaning": stress.meaning,
                "kept": int(mine.sum()),
                "judged": len(pnl) >= settings["min_kept"],
                "profitable_share": float((pnl >= BREAK_EVEN_RUPEES).mean())
                if len(pnl)
                else np.nan,
                "median_pnl": float(np.median(pnl)) if len(pnl) else np.nan,
                "average_pnl": exact_mean(pnl) if len(pnl) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("rule")


def _blocks(
    subject: BarRun,
    rules: Sequence[StressRule],
    flags: np.ndarray,
    set_: RerunSet | None,
    table: pd.DataFrame,
) -> tuple:
    setup = subject.reruns.setup
    groups = []
    for k, stress in enumerate(rules):
        if set_ is None or not len(flags):
            break
        mine = np.flatnonzero(flags[:, k] & set_.ran)[:DRAWN]
        if len(mine):
            groups.append(PathGroup(stress.name, set_.paths[mine]))
    chart = PathsBlock(
        "Equity on the kept candidates",
        f"Up to {DRAWN} kept candidates of each rule; the run bold",
        setup.steps,
        pd.DatetimeIndex(setup.dates),
        tuple(groups),
        real_path(subject.trades, setup.steps),
    )
    rows = tuple(
        (
            str(name),
            str(row.meaning),
            words.number(row.kept, 0),
            words.percent(row.profitable_share),
            words.rupees(row.median_pnl),
            words.rupees(row.average_pnl),
            "judged" if row.judged else "too few",
        )
        for name, row in table.iterrows()
    )
    columns = ("Rule", "Keeps", "Kept", "Profitable", "Median P&L", "Average P&L", "Judged")
    return ((chart,), (TableBlock("By rule", "The kept candidates of each rule", columns, rows),))


__all__ = ["SETTINGS", "kept_candidates", "rule", "run"]

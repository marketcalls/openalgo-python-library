"""Synthetic stress (design 17, section 2.9): does it hold through shocks history did not
have?

Candidates: the noise check's first `noise_candidates` noisy copies (200, at the noise
check's default settings) and the permutation test's first `permutation_candidates`
permuted paths (200), so a report that runs those checks reruns no candidate again
(`_bar_common`). The strategy runs on every candidate. A candidate is kept as a shock when
the stress rules hold for it (`_stress_rules`: a drawdown, a time to a new high or a fall
below the moving average at least `shock` times the real one's, or a function of (real,
candidate) bars), "any" of them (the default) or "all" (`combine`).

Judging: a generator can take an edge away by itself, whatever the shocks. A permutation
takes away an edge that lives in the order of the moves (the permutation test passes a
strategy exactly when its edge does), so a strategy with such an edge makes money on
about half the permuted paths, shocks or not, and "at least half the kept series
profitable" would fail it for the edge the permutation test passes. Each rule's kept
candidates are therefore judged against the share of all the candidates of the same
generators that made money: its expected share is the average, over its kept
candidates, of the share of their generator's candidates that are profitable.

Figures per rule: the candidates kept (of each generator), the share of them profitable,
the expected share, their ratio, the median and average net P&L in rupees. A rule is
judged when it keeps `min_kept` (10) or more that ran and its expected share is above
zero. Pass: under every judged rule the kept share is at least `keep` (half) of the
expected share. When the strategy makes money on every candidate of a generator this is
the design's rule: at least half the kept ones profitable. Not judged when no rule is
judged, when the run did not make money, or when it has too few closed trades.

Permuted candidates are left out, with the reason, when the run's symbols' bars are not
at the same times (`_bar_common.grid_problem`). For a strategy made by `bt.from_arrays`
the note says what its fixed decisions leave unjudged (`_bar_common.FIXED_TRADES`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import _words as words
from ._bar_common import (
    BarRun,
    RerunSet,
    fixed_note,
    grid_problem,
    joined,
    noisy_family,
    permuted_family,
    real_path,
    rerun_set,
)
from ._blocks import PathGroup, PathsBlock, TableBlock
from ._generate_noisy import Noise, ranges
from ._reruns import NOISY, PERMUTED, Family
from ._setting import Setting, between, whole
from ._stress_rules import DEFAULT_RULES, RULES_TAKE, StressRule, held, prices, rules_value
from ._stress_rules import stress_rules as make_rules
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
    Setting("rules", DEFAULT_RULES, rules_value, RULES_TAKE, "the stress rules"),
    Setting("combine", "any", _combine, "'any' or 'all'", "how the rules keep a candidate"),
    Setting(
        "shock",
        1.5,
        between(1.0, 100.0, low_open=False, high_open=False),
        "a number from 1 to 100",
        "how many times the real figure a candidate's must reach to be a shock",
    ),
    Setting(
        "average", 200, whole(2), "a whole number, 2 or more", "sessions of the moving average"
    ),
    Setting(
        "correlation",
        0.5,
        between(-1.0, 1.0, low_open=False, high_open=False),
        "a number from -1 to 1",
        "the correlation of returns below which a noisy candidate is kept",
    ),
    Setting("min_kept", 10, whole(1), "a whole number, 1 or more", "kept candidates a rule needs"),
    Setting(
        "keep",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "how often a rule's kept candidates must make money, as a share of how often all "
        "the candidates of their generators do",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        f"under every rule that keeps {settings['min_kept']} or more candidates, the kept "
        f"ones make money at least {words.percent(settings['keep'], 0)} as often as all the "
        "candidates of their generators"
    )


@dataclass(frozen=True, eq=False)
class Candidates:
    """Every candidate of a stress check, in order: the generator of each (`NOISY` or
    `PERMUTED`), whether each rule holds for each (candidates, rules), and why a
    generator's candidates were left out ("" when none were)."""

    generators: np.ndarray
    flags: np.ndarray
    left_out: str

    def kept(self, combine: str) -> np.ndarray:
        """Whether each candidate is kept: `combine` "any" or "all" of its rules hold."""
        if not len(self.flags):
            return np.zeros(0, dtype=bool)
        return self.flags.any(axis=1) if combine == "any" else self.flags.all(axis=1)


def _families(subject: BarRun, settings: Mapping[str, Any]) -> list[tuple[Family, int]]:
    """(family, candidates) of the generators that can run (module docstring)."""
    families = [(noisy_family(subject.seed, Noise()), settings["noise_candidates"])]
    if not grid_problem(subject.bars):
        families.append((permuted_family(subject.seed), settings["permutation_candidates"]))
    return [(family, count) for family, count in families if count]


def candidates(
    subject: BarRun, rules: Sequence[StressRule], settings: Mapping[str, Any]
) -> Candidates:
    """The `Candidates` of `subject` under `rules` (module docstring)."""
    bars = subject.bars
    real = [prices(frame, bars.first_run_ns) for frame in bars.frames]
    atrs = [ranges(frame, Noise().period) for frame in bars.frames]
    generators, flags = [], []
    for family, count in _families(subject, settings):
        for run in range(count):
            frames = family.frames(bars, run, atrs)
            made = [prices(frame, bars.first_run_ns) for frame in frames]
            flags.append(held(rules, real, made, permuted=family.bars == PERMUTED))
            generators.append(family.bars)
    problem = grid_problem(bars)
    left_out = ""
    if problem and settings["permutation_candidates"]:
        left_out = f" Permuted candidates are left out: {problem}."
    table = np.array(flags, dtype=bool).reshape(len(flags), len(rules))
    return Candidates(np.array(generators, dtype=object), table, left_out)


def rule_table(
    rules: Sequence[StressRule],
    made: Candidates,
    set_: RerunSet,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """The figures of each rule over its kept candidates (module docstring), by rule
    name; `set_` holds the reruns of every candidate, in order."""
    ran, profitable = set_.ran, set_.net_profit >= BREAK_EVEN_RUPEES
    base = {
        generator: float(profitable[ran & (made.generators == generator)].mean())
        for generator in (NOISY, PERMUTED)
        if (ran & (made.generators == generator)).any()
    }
    kept = made.kept(settings["combine"])
    rows = []
    for k, stress in enumerate(rules):
        mine = kept & made.flags[:, k] & ran
        pnl = set_.net_profit[mine]
        if mine.any():
            expected = exact_mean(np.array([base[g] for g in made.generators[mine]]))
            share = float(profitable[mine].mean())
        else:
            expected = share = np.nan
        judged = bool(mine.sum() >= settings["min_kept"] and expected > 0)
        ratio = share / expected if expected > 0 else np.nan
        rows.append(
            {
                "rule": stress.name,
                "meaning": stress.meaning,
                "kept": int(mine.sum()),
                "kept_noisy": int((mine & (made.generators == NOISY)).sum()),
                "kept_permuted": int((mine & (made.generators == PERMUTED)).sum()),
                "profitable_share": share,
                "expected_share": expected,
                "ratio": ratio,
                "median_pnl": float(np.median(pnl)) if len(pnl) else np.nan,
                "average_pnl": exact_mean(pnl) if len(pnl) else np.nan,
                "judged": judged,
                "passed": judged and ratio >= settings["keep"],
            }
        )
    return pd.DataFrame(rows).set_index("rule")


def _reruns(subject: BarRun, settings: Mapping[str, Any]) -> RerunSet | None:
    """The reruns of every candidate, in the order `candidates` makes them."""
    parts = [
        rerun_set(subject.reruns.rows(family, range(count)))
        for family, count in _families(subject, settings)
    ]
    return joined(parts) if parts else None


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = too_few(subject.trades, settings["min_trades"], text) or no_edge(subject.trades, text)
    if short is not None:
        return short
    rules = make_rules(settings["rules"], settings)
    made = candidates(subject, rules, settings)
    set_ = _reruns(subject, settings)
    if set_ is None:
        return not_judged(text, "No candidate was made." + made.left_out)
    table = rule_table(rules, made, set_, settings)
    judged = table[table["judged"]]
    note = _note(made, set_, settings, subject)
    if not len(judged):
        why = (
            f"Too few kept candidates to judge: no rule keeps {settings['min_kept']} or more "
            f"of the {words.whole(len(made.generators), 'candidate')} from generators on "
            f"whose candidates the strategy makes money. {note}"
        )
        return not_judged(text, why)
    passed = bool(judged["passed"].all())
    lowest = judged["ratio"].idxmin()
    worst = judged.loc[lowest]
    kept = int((made.kept(settings["combine"]) & set_.ran).sum())
    figure = (
        f"{kept} of {words.whole(len(made.generators), 'candidate')} kept as shocks; under "
        f"{lowest}, {words.percent(worst['profitable_share'])} of its shocks are profitable "
        f"against {words.percent(worst['expected_share'])} of their generators' candidates, "
        f"{words.number(worst['ratio'])} times, over {words.whole(len(judged), 'judged rule')}"
    )
    summary = _summary(made, set_, kept, len(judged))
    blocks = _blocks(subject, rules, made, set_, table, settings)
    return Outcome(
        PASSED if passed else FAILED, figure, text, note, table.reset_index(), summary, blocks
    )


def _summary(made: Candidates, set_: RerunSet, kept: int, judged: int) -> dict[str, float]:
    """The candidates made, those kept that ran, the rules judged, and the share of each
    generator's candidates that made money (NaN for a generator with none that ran)."""
    ran, profitable = set_.ran, set_.net_profit >= BREAK_EVEN_RUPEES
    summary = {"candidates": float(len(made.generators)), "kept": float(kept)}
    summary["judged_rules"] = float(judged)
    for generator in (NOISY, PERMUTED):
        mine = ran & (made.generators == generator)
        share = float(profitable[mine].mean()) if mine.any() else np.nan
        summary[f"{generator}_profitable"] = share
    return summary


def _note(made: Candidates, set_: RerunSet, settings: Mapping[str, Any], run: BarRun) -> str:
    note = (
        f"Candidates: the noise check's first {settings['noise_candidates']} noisy copies of "
        f"the run's bars and the permutation test's first {settings['permutation_candidates']} "
        f"permuted paths; the strategy ran on each, and one is kept as a shock when "
        f"{settings['combine']} of the rules hold, each rule judged on the session closes of "
        "every instrument against the real ones. A rule's kept candidates are judged against "
        "the share of all the candidates of their generators that make money, since a "
        "permutation takes away an edge that lives in the order of the moves, shocks or not."
    )
    return note + fixed_note(run) + made.left_out + set_.failure_note()


def _blocks(
    subject: BarRun,
    rules: Sequence[StressRule],
    made: Candidates,
    set_: RerunSet,
    table: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple:
    setup = subject.reruns.setup
    kept = made.kept(settings["combine"])
    groups = []
    for k, stress in enumerate(rules):
        mine = np.flatnonzero(kept & made.flags[:, k] & set_.ran)[:DRAWN]
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
            f"{row.kept} ({row.kept_noisy} noisy)",
            words.percent(row.profitable_share),
            words.percent(row.expected_share),
            words.number(row.ratio),
            words.rupees(row.median_pnl),
            words.rupees(row.average_pnl),
            _standing(row, settings),
        )
        for name, row in table.iterrows()
    )
    columns = (
        "Rule",
        "Keeps",
        "Kept",
        "Profitable",
        "Expected",
        "Ratio",
        "Median P&L",
        "Average P&L",
        "Judged",
    )
    subtitle = "Expected: the share of all the candidates of their generators that make money"
    return ((chart,), (TableBlock("By rule", subtitle, columns, rows),))


def _standing(row: Any, settings: Mapping[str, Any]) -> str:
    """A rule's standing in the table: judged, or why not."""
    if row.judged:
        return "judged"
    if row.kept < settings["min_kept"]:
        return "too few"
    return "none profitable"


__all__ = ["SETTINGS", "Candidates", "candidates", "rule", "rule_table", "run"]

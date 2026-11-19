"""The candidates of synthetic stress and the figures of each rule over them (design 17,
section 2.9; `_check_stress` runs the check).

`Candidates` holds every candidate in the order it was made: its generator (`NOISY` or
`PERMUTED`) and whether each rule holds for it; `candidates` makes them from the noise
check's noisy copies and the permutation test's permuted paths (`_check_stress`).

`rule_table` measures each rule over the candidates it keeps that ran: how many (of each
generator), the share of them that made money, the expected share (the average, over
its kept candidates, of the share of their generator's candidates that made money), the
ratio of the two, the median and average net P&L. A rule is judged when it keeps
`min_kept` or more, at most `MOST_KEPT` (half) of the candidates of its generators that
ran, and its expected share is above zero (`why_unjudged`): a shock is a path harder than
most, so a rule that keeps nearly every candidate singles out no shock, and its ratio is
about 1 by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._bar_common import BarRun, RerunSet, grid_problem, noisy_family, permuted_family
from ._generate_noisy import Noise, ranges
from ._reruns import NOISY, PERMUTED, Family
from ._stress_rules import StressRule, held, prices
from ._trade_common import BREAK_EVEN_RUPEES, exact_mean

#: The most of the candidates of its generators that ran a rule may keep and be judged
#: (module docstring): a shock is a path harder than most (lead, 2026-11-18).
MOST_KEPT = 0.5


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


def families(subject: BarRun, settings: Mapping[str, Any]) -> list[tuple[Family, int]]:
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
    for family, count in families(subject, settings):
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


def why_unjudged(kept: int, candidates: int, expected: float, min_kept: int) -> str:
    """Why a rule that keeps `kept` of the `candidates` of its generators that ran, with
    `expected` the share of them that make money, is not judged; "" when it is
    (module docstring)."""
    if kept < min_kept:
        return f"keeps {kept} of {candidates} candidates, fewer than {min_kept}"
    if kept > MOST_KEPT * candidates:
        return f"keeps {kept} of {candidates} candidates, so it does not single out shocks"
    if not expected > 0:
        return (
            f"keeps {kept} of {candidates} candidates, of generators on whose candidates the "
            "strategy never makes money"
        )
    return ""


def rule_table(
    rules: Sequence[StressRule],
    made: Candidates,
    set_: RerunSet,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """The figures of each rule over its kept candidates (module docstring), by rule
    name; `set_` holds the reruns of every candidate, in order. `candidates` counts the
    candidates of the rule's generators that ran, `why` says why it is not judged ("" when
    it is)."""
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
        read = (NOISY, PERMUTED) if stress.permuted else (NOISY,)
        candidates = int((ran & np.isin(made.generators, read)).sum())
        pnl = set_.net_profit[mine]
        if mine.any():
            expected = exact_mean(np.array([base[g] for g in made.generators[mine]]))
            share = float(profitable[mine].mean())
        else:
            expected = share = np.nan
        why = why_unjudged(int(mine.sum()), candidates, expected, settings["min_kept"])
        ratio = share / expected if expected > 0 else np.nan
        rows.append(
            {
                "rule": stress.name,
                "meaning": stress.meaning,
                "kept": int(mine.sum()),
                "kept_noisy": int((mine & (made.generators == NOISY)).sum()),
                "kept_permuted": int((mine & (made.generators == PERMUTED)).sum()),
                "candidates": candidates,
                "profitable_share": share,
                "expected_share": expected,
                "ratio": ratio,
                "median_pnl": float(np.median(pnl)) if len(pnl) else np.nan,
                "average_pnl": exact_mean(pnl) if len(pnl) else np.nan,
                "judged": not why,
                "why": why,
                "passed": not why and ratio >= settings["keep"],
            }
        )
    return pd.DataFrame(rows).set_index("rule")


__all__ = ["MOST_KEPT", "Candidates", "candidates", "families", "rule_table", "why_unjudged"]

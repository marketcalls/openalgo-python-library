"""The rules of synthetic stress (design 17, section 2.9): which generated series show a
shock the real one did not have, each rule judged on a candidate's own prices against the
real ones.

The prices a built-in rule reads are each instrument's session closes (the close of a
session's last base bar), so a rule reads the same at any base interval: the closes of
the run's sessions, and for the moving average the history's closes before them too.

    drawdown        the candidate's maximum drawdown of its closes over the run is
                    deeper than the real one's
    recovery        its longest time from a high to a new high, in sessions (a high not
                    regained by the end runs to the end), is longer than the real one's
    below_average   its deepest close below its simple moving average of `average`
                    sessions (200 by default), the lowest close over its average less one
                    over the run, is deeper than the real one's; neither has it while the
                    average is not defined (fewer sessions than `average`)
    correlation     the correlation of its session returns (close to close) with the
                    real ones over the run is below `correlation` (0.5 by default)

A rule may also be a function of (real, candidate), the run's bars of one instrument each
(canonical layout, from the run's first bar), that returns True when it holds; its name
is the function's. A rule holds for a candidate when it holds for at least one of its
instruments, and a comparison with an undefined figure does not hold.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._ohlc import session_days

#: The built-in rules, in the order a report lists them.
BUILT_IN = ("drawdown", "recovery", "below_average", "correlation")
#: What each built-in rule keeps, in words.
MEANINGS = {
    "drawdown": "a deeper maximum drawdown of the price",
    "recovery": "a longer time from a high to a new high",
    "below_average": "a deeper fall below the moving average",
    "correlation": "returns with a low correlation to the real ones",
}


@dataclass(frozen=True, eq=False)
class Prices:
    """One instrument's prices as the rules read them: `closes` the session closes,
    history first, rupees; `in_run` whether each session is in the run; `bars` the run's
    base bars (canonical layout)."""

    closes: np.ndarray
    in_run: np.ndarray
    bars: pd.DataFrame


def prices(bars: pd.DataFrame, first_run_ns: int) -> Prices:
    """The `Prices` of `bars` (canonical layout, history first), whose run starts at the
    bar opening at `first_run_ns`, UTC nanoseconds."""
    opens = bars["ts_open"].to_numpy(dtype=np.int64)
    days = session_days(opens)
    last = np.flatnonzero(np.append(days[1:] != days[:-1], True))
    closes = bars["close"].to_numpy(dtype=float)[last]
    run = bars[opens >= first_run_ns].reset_index(drop=True)
    return Prices(closes, opens[last] >= first_run_ns, run)


def max_drawdown(closes: np.ndarray) -> float:
    """The deepest fall of `closes` below their running high, a fraction (zero or
    negative); NaN without a close."""
    if not len(closes):
        return math.nan
    return float(np.min(closes / np.maximum.accumulate(closes)) - 1.0)


def longest_recovery(closes: np.ndarray) -> int:
    """The most sessions in a row whose close is below the running high: the longest time
    from a high to a new high (to the end when it is not regained)."""
    below = closes < np.maximum.accumulate(closes)
    edges = np.diff(np.concatenate(([0], below.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def deepest_below_average(closes: np.ndarray, in_run: np.ndarray, window: int) -> float:
    """The lowest close over its simple moving average of `window` sessions, less one,
    over the run's sessions (a fraction; negative below the average); NaN while the
    average is not defined in the run."""
    average = pd.Series(closes).rolling(window).mean().to_numpy()
    ratio = closes[in_run] / average[in_run] - 1.0
    known = ratio[np.isfinite(ratio)]
    return float(known.min()) if len(known) else math.nan


def return_correlation(real: np.ndarray, candidate: np.ndarray) -> float:
    """The correlation of the session returns (close to close) of two series of closes of
    the same sessions; NaN with fewer than three closes or a series that never moves."""
    if len(real) < 3:
        return math.nan
    a, b = real[1:] / real[:-1] - 1.0, candidate[1:] / candidate[:-1] - 1.0
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _built_in(name: str, settings: dict[str, Any]) -> Callable[[Prices, Prices], bool]:
    """The test of built-in rule `name` on one instrument (module docstring)."""
    window, floor = settings["average"], settings["correlation"]

    def holds(real: Prices, candidate: Prices) -> bool:
        a, b = real.closes[real.in_run], candidate.closes[candidate.in_run]
        if name == "drawdown":
            return max_drawdown(b) < max_drawdown(a)
        if name == "recovery":
            return longest_recovery(b) > longest_recovery(a)
        if name == "below_average":
            deeper = deepest_below_average(candidate.closes, candidate.in_run, window)
            return deeper < deepest_below_average(real.closes, real.in_run, window)
        return return_correlation(a, b) < floor

    return holds


def _given(function: Callable[..., Any]) -> Callable[[Prices, Prices], bool]:
    def holds(real: Prices, candidate: Prices) -> bool:
        return bool(function(real.bars, candidate.bars))

    return holds


@dataclass(frozen=True, eq=False)
class StressRule:
    """A rule: its `name`, what it keeps in words, and `holds(real, candidate)`, whether
    it holds for one instrument's `Prices`."""

    name: str
    meaning: str
    holds: Callable[[Prices, Prices], bool]


def stress_rules(rules: Sequence[Any], settings: dict[str, Any]) -> tuple[StressRule, ...]:
    """The rules named in `rules` (built-in names or functions), with the check's
    `settings` (`average`, `correlation`)."""
    made = []
    for rule in rules:
        if isinstance(rule, str):
            made.append(StressRule(rule, MEANINGS[rule], _built_in(rule, settings)))
        else:
            name = getattr(rule, "__name__", "custom")
            made.append(StressRule(name, f"the function {name}", _given(rule)))
    return tuple(made)


def held(
    rules: Sequence[StressRule], real: Sequence[Prices], candidate: Sequence[Prices]
) -> tuple[bool, ...]:
    """Whether each rule holds for a candidate: for at least one of its instruments."""
    pairs = list(zip(real, candidate, strict=True))
    return tuple(any(rule.holds(a, b) for a, b in pairs) for rule in rules)


def rules_value(value: Any) -> tuple[Any, ...] | None:
    """The `rules` setting: a non-empty list of built-in rule names and functions, as a
    tuple; None for anything else (`_setting.Setting.rule`)."""
    if isinstance(value, str) or not isinstance(value, list | tuple) or not value:
        return None
    if all((isinstance(rule, str) and rule in BUILT_IN) or callable(rule) for rule in value):
        return tuple(value)
    return None


__all__ = [
    "BUILT_IN",
    "MEANINGS",
    "Prices",
    "StressRule",
    "deepest_below_average",
    "held",
    "longest_recovery",
    "max_drawdown",
    "prices",
    "return_correlation",
    "rules_value",
    "stress_rules",
]

"""The rules of synthetic stress (design 17, section 2.9): which generated series show a
shock the real one did not have, each rule judged on a candidate's own prices against the
real ones.

The prices a built-in rule reads are each instrument's session closes (the close of a
session's last base bar), so a rule reads the same at any base interval: the closes of
the run's sessions, and for the moving average the history's closes before them too.

A shock is a figure at least `shock` times the real one's (1.5 by default: a fall half as
deep again, a time to a new high half as long again) and worse than it, so a candidate
that differs from the real series by the ordinary variation of one path (noise of a
fifth of a bar's range moves a drawdown by a fraction of a point) is not kept. A real
figure of zero (a series that never fell) makes any fall a shock.

    drawdown        the depth of the candidate's maximum drawdown of its closes over the
                    run, a fraction of the high
    recovery        its longest time from a high to a new high, in sessions (a high not
                    regained by the end runs to the end)
    below_average   its deepest close below its simple moving average of `average`
                    sessions (200 by default) over the run, a fraction of the average (0
                    when it never closed below); neither series has it while the average
                    is not defined (fewer sessions than `average`)
    correlation     the correlation of its session returns (close to close) with the
                    real ones over the run is below `correlation` (0.5 by default); it
                    reads noisy candidates only, since a permuted path is the run's own
                    moves in another order, uncorrelated with them by construction on
                    daily bars, which is no shock. It is not among the default rules
                    (`DEFAULT_RULES`): noise of the default size keeps the correlation
                    near 1, so it would keep no candidate but those of a series whose
                    returns barely vary, where noise is all a correlation sees

A rule may also be a function of (real, candidate), the run's bars of one instrument each
(canonical layout, from the run's first bar), that returns True when it holds; its name
is the function's, and it reads candidates of both generators. A rule holds for a
candidate when it holds for at least one of its instruments, and a comparison with an
undefined figure does not hold. Each rule needs a name of its own (`rules_value`).
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
#: The rules a stress check keeps candidates by unless told otherwise (module docstring).
DEFAULT_RULES = ("drawdown", "recovery", "below_average")
#: What each built-in rule keeps, in words; {shock} and {floor} are its settings.
MEANINGS = {
    "drawdown": "a maximum drawdown of the price {shock:g} times as deep or more",
    "recovery": "a time from a high to a new high {shock:g} times as long or more",
    "below_average": "a fall below the moving average {shock:g} times as deep or more",
    "correlation": "noisy copies whose returns correlate below {floor:g} with the real ones",
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


def _severity(name: str, prices_: Prices, window: int) -> float:
    """How bad the figure of built-in rule `name` is for `prices_`, zero or more (module
    docstring); NaN when undefined."""
    closes = prices_.closes[prices_.in_run]
    if name == "drawdown":
        return -max_drawdown(closes)
    if name == "recovery":
        return float(longest_recovery(closes))
    below = deepest_below_average(prices_.closes, prices_.in_run, window)
    return max(0.0, -below) if math.isfinite(below) else math.nan


def is_shock(candidate: float, real: float, shock: float) -> bool:
    """Whether a candidate's figure is a shock against the real one's (both zero or
    more): worse, and at least `shock` times it. False when either is undefined."""
    if not (math.isfinite(candidate) and math.isfinite(real)):
        return False
    return candidate > real and candidate >= shock * real


def _built_in(name: str, settings: dict[str, Any]) -> Callable[[Prices, Prices], bool]:
    """The test of built-in rule `name` on one instrument (module docstring)."""
    window, floor, shock = settings["average"], settings["correlation"], settings["shock"]

    def holds(real: Prices, candidate: Prices) -> bool:
        if name == "correlation":
            a, b = real.closes[real.in_run], candidate.closes[candidate.in_run]
            return return_correlation(a, b) < floor
        return is_shock(_severity(name, candidate, window), _severity(name, real, window), shock)

    return holds


def _given(function: Callable[..., Any]) -> Callable[[Prices, Prices], bool]:
    def holds(real: Prices, candidate: Prices) -> bool:
        return bool(function(real.bars, candidate.bars))

    return holds


@dataclass(frozen=True, eq=False)
class StressRule:
    """A rule: its `name`, what it keeps in words, `holds(real, candidate)`, whether it
    holds for one instrument's `Prices`, and whether it reads permuted candidates
    (`permuted`; a rule that does not holds for none of them)."""

    name: str
    meaning: str
    holds: Callable[[Prices, Prices], bool]
    permuted: bool


def rule_name(rule: Any) -> str:
    """The name of a rule as given: a built-in rule's name, or a function's."""
    return rule if isinstance(rule, str) else getattr(rule, "__name__", "custom")


def stress_rules(rules: Sequence[Any], settings: dict[str, Any]) -> tuple[StressRule, ...]:
    """The rules named in `rules` (built-in names or functions), with the check's
    `settings` (`average`, `correlation`, `shock`)."""
    made = []
    for rule in rules:
        name = rule_name(rule)
        if isinstance(rule, str):
            meaning = MEANINGS[rule].format(shock=settings["shock"], floor=settings["correlation"])
            made.append(StressRule(rule, meaning, _built_in(rule, settings), rule != "correlation"))
        else:
            made.append(StressRule(name, f"the function {name}", _given(rule), True))
    return tuple(made)


def held(
    rules: Sequence[StressRule],
    real: Sequence[Prices],
    candidate: Sequence[Prices],
    *,
    permuted: bool,
) -> tuple[bool, ...]:
    """Whether each rule holds for a candidate, `permuted` or noisy: for at least one of
    its instruments, and never for a rule that does not read the candidate's generator."""
    pairs = list(zip(real, candidate, strict=True))
    return tuple(
        (rule.permuted or not permuted) and any(rule.holds(a, b) for a, b in pairs)
        for rule in rules
    )


def rules_value(value: Any) -> tuple[Any, ...] | None:
    """The `rules` setting: a non-empty list of built-in rule names and functions, each
    with a name of its own (the report's table is by rule name), as a tuple; None for
    anything else (`_setting.Setting.rule`)."""
    if isinstance(value, str) or not isinstance(value, list | tuple) or not value:
        return None
    if not all((isinstance(rule, str) and rule in BUILT_IN) or callable(rule) for rule in value):
        return None
    names = [rule_name(rule) for rule in value]
    return tuple(value) if len(set(names)) == len(names) else None


#: What the `rules` setting takes, in words.
RULES_TAKE = (
    f"a list of {list(BUILT_IN)} and functions of (real, candidate) bars, each with a name of "
    "its own (two functions need two def statements, since every lambda is named <lambda>)"
)


__all__ = [
    "BUILT_IN",
    "DEFAULT_RULES",
    "MEANINGS",
    "RULES_TAKE",
    "Prices",
    "StressRule",
    "deepest_below_average",
    "held",
    "is_shock",
    "longest_recovery",
    "max_drawdown",
    "prices",
    "return_correlation",
    "rule_name",
    "rules_value",
    "stress_rules",
]

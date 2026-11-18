"""The seventeen robustness checks, in the order a report lists them (design 17,
section 2): each one's number, name, title, the question it answers, its settings, its
procedure and what it reads.

The trade-list checks read the run's closed trades only, with no rerun: they are the
quick checks of every tearsheet. The bar checks rerun the backtest on generated bars, on
other symbols' bars, or on the run's own bars with one setting changed (`_reruns`). The
searching checks rerun it with its parameters changed, over a grid of the parameter
ranges or around the run's own, on the whole run or on windows of it
(`_search_setup`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ConfigError
from . import (
    _check_delayed,
    _check_liquidity,
    _check_noise,
    _check_noise_optimization,
    _check_other_markets,
    _check_out_of_sample,
    _check_parameters,
    _check_permutation,
    _check_random_exits,
    _check_random_oos,
    _check_resample,
    _check_reshuffle,
    _check_shifted,
    _check_stress,
    _check_variance,
    _check_vs_random,
    _check_walk_forward,
)
from ._setting import Setting

#: What a check reads: the run's trade list, its bars rerun with one thing changed, or its
#: bars rerun over parameter combinations.
TRADE_LIST, BARS, SEARCH = "trade list", "bars", "search"


@dataclass(frozen=True)
class CheckSpec:
    """One check: its `number` in the report, `name` (its keyword and `checks=` entry),
    `title`, `question`, its `settings` and its procedure `run`, what it reads (`reads`:
    `TRADE_LIST`, `BARS` or `SEARCH`), and for a check that reruns the backtest what it
    reruns (`reruns`, the families and their runs; none when it is not judged before any
    rerun), which the plan counts."""

    number: int
    name: str
    title: str
    question: str
    settings: tuple[Setting, ...]
    run: Callable
    reads: str
    reruns: Callable | None = None

    @property
    def trade_list(self) -> bool:
        """Whether the check reads the trade list only (a quick check)."""
        return self.reads == TRADE_LIST

    @property
    def searches(self) -> bool:
        """Whether the check reruns the backtest over parameter combinations."""
        return self.reads == SEARCH


def _check(number: int, name: str, title: str, question: str, module, reads: str) -> CheckSpec:
    reruns = None if reads == TRADE_LIST else module.reruns
    return CheckSpec(number, name, title, question, module.SETTINGS, module.run, reads, reruns)


CHECKS = (
    _check(
        1,
        "out_of_sample",
        "Out of sample",
        "Does it work on bars the search never saw?",
        _check_out_of_sample,
        SEARCH,
    ),
    _check(
        2,
        "random_out_of_sample",
        "Random out of sample",
        "Was the out-of-sample period just lucky?",
        _check_random_oos,
        TRADE_LIST,
    ),
    _check(
        3,
        "walk_forward",
        "Walk-forward",
        "Does it hold when re-fitted through time?",
        _check_walk_forward,
        SEARCH,
    ),
    _check(
        4,
        "vs_random",
        "Versus random",
        "Is it better than random entries with the same exposure?",
        _check_vs_random,
        BARS,
    ),
    _check(
        5,
        "permutation",
        "Permutation test",
        "Could the edge come from the price path's distribution alone?",
        _check_permutation,
        BARS,
    ),
    _check(6, "noise", "Noise", "Does it survive small changes to every bar?", _check_noise, BARS),
    _check(
        7,
        "shifted",
        "Shifted bars",
        "Does it depend on where the bars start?",
        _check_shifted,
        BARS,
    ),
    _check(
        8,
        "other_markets",
        "Other markets",
        "Does the logic hold on related symbols?",
        _check_other_markets,
        BARS,
    ),
    _check(
        9,
        "stress",
        "Synthetic stress",
        "Does it hold through shocks history did not have?",
        _check_stress,
        BARS,
    ),
    _check(
        10,
        "reshuffle",
        "Trade reshuffle",
        "How deep can the drawdown get in another order?",
        _check_reshuffle,
        TRADE_LIST,
    ),
    _check(
        11,
        "resample",
        "Trade resample",
        "How wide is the range of outcomes?",
        _check_resample,
        TRADE_LIST,
    ),
    _check(
        12, "random_exits", "Randomized exits", "Are the exits fitted?", _check_random_exits, BARS
    ),
    _check(
        13, "variance", "Variance", "Does it survive a worse win rate?", _check_variance, TRADE_LIST
    ),
    _check(
        14,
        "delayed",
        "Delayed orders",
        "Does it survive entering or exiting a bar late?",
        _check_delayed,
        BARS,
    ),
    _check(
        15, "liquidity", "Liquidity", "Does it survive worse fill prices?", _check_liquidity, BARS
    ),
    _check(
        16,
        "parameters",
        "Parameter neighbourhood",
        "A plateau, or one lucky spike?",
        _check_parameters,
        SEARCH,
    ),
    _check(
        17,
        "noise_optimization",
        "Noise-robust optimization",
        "Which settings hold across noisy histories?",
        _check_noise_optimization,
        SEARCH,
    ),
)
BY_NAME = {check.name: check for check in CHECKS}
#: The quick checks: those that read the trade list only.
QUICK = tuple(check.name for check in CHECKS if check.trade_list)
#: The checks that search parameter combinations.
SEARCHING = tuple(check.name for check in CHECKS if check.searches)


def find(name: str) -> CheckSpec:
    """The check named `name`. Raises ConfigError naming every check for another."""
    if name not in BY_NAME:
        raise ConfigError(f"there is no check {name!r}; the checks are {list(BY_NAME)}")
    return BY_NAME[name]


__all__ = [
    "BARS",
    "BY_NAME",
    "CHECKS",
    "QUICK",
    "SEARCH",
    "SEARCHING",
    "TRADE_LIST",
    "CheckSpec",
    "find",
]

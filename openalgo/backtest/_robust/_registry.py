"""The seventeen robustness checks, in the order a report lists them (design 17,
section 2): each one's number, name, title, the question it answers, the build unit
that brings it (an internal tag, never shown) and, for a check this version runs, its
settings and procedure.

A check this version does not run is listed with what it needs, in plain words
(`needs`), and asking for it raises NotImplementedError (`not_built`) saying so;
`checks="all"` runs the built checks and lists the others as not built. The trade-list
checks read the run's closed trades only, with no rerun: they are the quick checks of
every tearsheet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .._messages import not_supported
from ..errors import ConfigError
from . import _check_random_oos, _check_resample, _check_reshuffle, _check_variance
from ._setting import Setting

#: What the checks a later version brings need, by their build unit, as a reader would
#: put it. A message names what a check needs, never its internal unit (`_messages`).
NEEDS = {
    "RB2": "the bar generators (permuted, noisy and synthetic bars)",
    "RB3": "late orders, other fill prices and shifted bar boundaries in the engine",
    "RB4": "walk-forward and the parameter searches",
}


@dataclass(frozen=True)
class CheckSpec:
    """One check: its `number` in the report, `name` (its keyword and `checks=` entry),
    `title`, `question`, the build `unit` that brings it (internal), its `settings` and its
    procedure `run` (empty and None for a check not built in this version), and whether
    it reads the trade list only (`trade_list`)."""

    number: int
    name: str
    title: str
    question: str
    unit: str
    settings: tuple[Setting, ...] = ()
    run: Callable | None = None
    trade_list: bool = False

    @property
    def built(self) -> bool:
        """Whether this version runs the check."""
        return self.run is not None


def _trade_check(number: int, name: str, title: str, question: str, module) -> CheckSpec:
    return CheckSpec(number, name, title, question, "RB1", module.SETTINGS, module.run, True)


CHECKS = (
    CheckSpec(
        1, "out_of_sample", "Out of sample", "Does it work on bars the search never saw?", "RB4"
    ),
    _trade_check(
        2,
        "random_out_of_sample",
        "Random out of sample",
        "Was the out-of-sample period just lucky?",
        _check_random_oos,
    ),
    CheckSpec(
        3, "walk_forward", "Walk-forward", "Does it hold when re-fitted through time?", "RB4"
    ),
    CheckSpec(
        4,
        "vs_random",
        "Versus random",
        "Is it better than random entries with the same exposure?",
        "RB2",
    ),
    CheckSpec(
        5,
        "permutation",
        "Permutation test",
        "Could the edge come from the price path's distribution alone?",
        "RB2",
    ),
    CheckSpec(6, "noise", "Noise", "Does it survive small changes to every bar?", "RB2"),
    CheckSpec(7, "shifted", "Shifted bars", "Does it depend on where the bars start?", "RB3"),
    CheckSpec(
        8, "other_markets", "Other markets", "Does the logic hold on related symbols?", "RB2"
    ),
    CheckSpec(
        9, "stress", "Synthetic stress", "Does it hold through shocks history did not have?", "RB2"
    ),
    _trade_check(
        10,
        "reshuffle",
        "Trade reshuffle",
        "How deep can the drawdown get in another order?",
        _check_reshuffle,
    ),
    _trade_check(
        11, "resample", "Trade resample", "How wide is the range of outcomes?", _check_resample
    ),
    CheckSpec(12, "random_exits", "Randomized exits", "Are the exits fitted?", "RB3"),
    _trade_check(13, "variance", "Variance", "Does it survive a worse win rate?", _check_variance),
    CheckSpec(
        14,
        "delayed",
        "Delayed orders",
        "Does it survive entering or exiting a bar late?",
        "RB3",
    ),
    CheckSpec(15, "liquidity", "Liquidity", "Does it survive worse fill prices?", "RB3"),
    CheckSpec(16, "parameters", "Parameter neighbourhood", "A plateau, or one lucky spike?", "RB4"),
    CheckSpec(
        17,
        "noise_optimization",
        "Noise-robust optimization",
        "Which settings hold across noisy histories?",
        "RB4",
    ),
)
BY_NAME = {check.name: check for check in CHECKS}
#: The names of the checks this version runs, in report order.
BUILT = tuple(check.name for check in CHECKS if check.built)
#: The quick checks: those that read the trade list only.
QUICK = tuple(check.name for check in CHECKS if check.trade_list)


def find(name: str) -> CheckSpec:
    """The check named `name`. Raises ConfigError naming every check for another."""
    if name not in BY_NAME:
        raise ConfigError(f"there is no check {name!r}; the checks are {list(BY_NAME)}")
    return BY_NAME[name]


def needs(check: CheckSpec) -> str:
    """What `check`, which this version does not run, needs: "it needs ..."."""
    return f"it needs {NEEDS[check.unit]}"


def not_built_note(check: CheckSpec) -> str:
    """The note of `check`, which this version does not run, in a report."""
    return f"Not built in this version: {needs(check)}."


def not_built(check: CheckSpec) -> NotImplementedError:
    """The error for asking for `check`, which this version does not run: what it
    needs, and the checks it can run instead."""
    return not_supported(
        f"the {check.name} check ({check.title})",
        f"{needs(check)}. checks='all' runs the checks this version has "
        f"({', '.join(BUILT)}) and lists the rest as not built",
    )


__all__ = [
    "BUILT",
    "BY_NAME",
    "CHECKS",
    "NEEDS",
    "QUICK",
    "CheckSpec",
    "find",
    "needs",
    "not_built",
    "not_built_note",
]

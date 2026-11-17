"""The settings of a check: each has a name, a default, a rule for the values it takes
and a line saying what it means. A check declares its settings beside its procedure; a
report applies the check's dict (`noise={"runs": 500}`) over the defaults and refuses,
with a ConfigError naming what it takes, any value the rule does not allow.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError

#: A rule: the value as the check reads it, or None when the value is not allowed.
Rule = Callable[[Any], Any]


@dataclass(frozen=True)
class Setting:
    """One setting of a check: `rule` returns the value to use, or None when a value is
    not allowed, and `takes` says in words what it allows. A setting whose default is
    None (no limit, the report's runs) also takes None."""

    name: str
    default: Any
    rule: Rule
    takes: str
    meaning: str


def _real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def whole(least: int) -> Rule:
    """Whole numbers from `least` up."""

    def rule(value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            return None
        return int(value) if value >= least else None

    return rule


def between(low: float, high: float, *, low_open: bool, high_open: bool) -> Rule:
    """Real numbers from `low` to `high`, each end left out when it is open."""

    def rule(value: Any) -> Any:
        number = _real(value)
        if number is None:
            return None
        above = number > low if low_open else number >= low
        under = number < high if high_open else number <= high
        return number if above and under else None

    return rule


def resolve(check: str, settings: tuple[Setting, ...], given: Any) -> dict[str, Any]:
    """The values of `settings` with the dict `given` applied (None gives the defaults).
    Raises ConfigError for a value that is not a dict, an unknown key (naming the valid
    ones) or a value its rule does not allow."""
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise ConfigError(f"{check}= takes a dict of its settings, got {given!r}")
    names = [setting.name for setting in settings]
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise ConfigError(f"{check}= has no setting {unknown}; it takes {names}")
    values: dict[str, Any] = {}
    for setting in settings:
        if setting.name not in given:
            values[setting.name] = setting.default
            continue
        value = given[setting.name]
        # None keeps a setting whose default is None (no limit, the report's runs).
        allowed = None if value is None else setting.rule(value)
        if allowed is None and not (value is None and setting.default is None):
            raise ConfigError(f"{check}: {setting.name} takes {setting.takes}, got {value!r}")
        values[setting.name] = allowed
    return values


__all__ = ["Setting", "between", "resolve", "whole"]

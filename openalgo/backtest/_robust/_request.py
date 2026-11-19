"""What a robustness report is asked to do: the checks it lists, the objective, the runs
and the seed, the worker processes and every check's settings, all checked before any
work starts.

`checks` is "all" (every check), "quick" (the trade-list checks, which need no rerun) or
a list of check names. A keyword named after a check holds that check's settings as a
dict (`variance={"points": 10}`). Everything else `bt.robustness` takes is a
`bt.backtest` run argument, or a parameter range named after a strategy parameter
(`fast=range(5, 30, 5)`); ranges belong to the checks that search parameters
(`_registry.SEARCHING`), which must be listed to take them, as `others=`, the symbols of
the other markets check, needs that check listed.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._objective import check_objective
from ..errors import ConfigError
from . import _random
from ._check_other_markets import others_value
from ._grid import checked_ranges
from ._registry import BY_NAME, CHECKS, QUICK, SEARCHING, CheckSpec, find
from ._setting import resolve, whole

DEFAULT_OBJECTIVE = "sharpe"
DEFAULT_RUNS = 1_000
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Request:
    """A checked request: `listed` the checks the report runs, in report order;
    `objective`, `runs` and `seed`; `workers` (None for the CPU count); `settings` the
    resolved settings of each listed check; `others` the (symbol, exchange) pairs of
    `others=`, exchange None for the strategy's own; `ranges` the parameter ranges, each
    as a tuple of its values."""

    listed: tuple[CheckSpec, ...]
    objective: Any
    runs: int
    seed: int
    workers: int | None
    settings: Mapping[str, Mapping[str, Any]]
    others: tuple[tuple[str, str | None], ...] = ()
    ranges: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    @property
    def reruns(self) -> tuple[CheckSpec, ...]:
        """The listed checks that rerun the backtest (the bar and searching checks)."""
        return tuple(check for check in self.listed if not check.trade_list)

    def runs_of(self, check: CheckSpec) -> int:
        """The runs of `check`: its `runs` setting, else the report's."""
        given = self.settings[check.name].get("runs")
        return self.runs if given is None else given


def select(checks: Any) -> tuple[CheckSpec, ...]:
    """The checks `checks` names, in report order. Raises ConfigError for anything but
    "all", "quick" or a non-empty list of check names."""
    if checks == "all":
        return CHECKS
    if checks == "quick":
        return tuple(BY_NAME[name] for name in QUICK)
    if isinstance(checks, str) or not hasattr(checks, "__iter__"):
        raise ConfigError(f"checks takes 'all', 'quick' or a list of check names, got {checks!r}")
    named = [find(name) if isinstance(name, str) else None for name in checks]
    if not named or None in named:
        raise ConfigError(f"checks takes a non-empty list of check names, got {checks!r}")
    return tuple(check for check in CHECKS if check in named)


def split_keywords(strategy: type, keywords: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """(run arguments, check settings, parameter ranges) of the keywords. Raises
    ConfigError for a name that is none of them, or both a check and a parameter."""
    from ..run import backtest

    run_names = set(inspect.signature(backtest).parameters) - {"strategy", "client"}
    parameters = strategy.parameters()
    run, checks, ranges = {}, {}, {}
    for name, value in keywords.items():
        if name in BY_NAME and name in parameters:
            raise ConfigError(
                f"{name} is both a check and a parameter of {strategy.__name__}: rename the "
                "parameter to set the check, or fix the parameter with params="
            )
        if name in BY_NAME:
            checks[name] = value
        elif name in run_names:
            run[name] = value
        elif name in parameters:
            ranges[name] = value
        else:
            raise ConfigError(
                f"bt.robustness: {name!r} is neither a check ({list(BY_NAME)}), a run argument "
                f"of bt.backtest nor a parameter of {strategy.__name__} ({sorted(parameters)})"
            )
    return run, checks, ranges


def _whole(name: str, value: Any, least: int, most: int | None = None) -> int:
    number = whole(least)(value)
    if number is None or (most is not None and number > most):
        bound = "" if most is None else f" and at most {most}"
        raise ConfigError(f"{name} takes a whole number, {least} or more{bound}, got {value!r}")
    return number


def build_request(
    *,
    checks: Any,
    objective: Any,
    runs: Any,
    seed: Any,
    workers: Any,
    others: Any,
    check_settings: Mapping[str, Any],
    ranges: Mapping[str, Any],
) -> Request:
    """The checked `Request`. Raises ConfigError for unusable values (naming what each
    takes), for settings of a check `checks` does not list, for `others=` without the
    other markets check, and for parameter ranges without a check that searches them or
    with an empty range."""
    listed = select(checks)
    check_objective(objective)
    pairs = () if others is None else others_value(others)
    if pairs and BY_NAME["other_markets"] not in listed:
        raise ConfigError(
            "others= names symbols for the other_markets check, which checks= does not run"
        )
    if ranges and not any(check.searches for check in listed):
        raise ConfigError(
            f"parameter ranges ({', '.join(sorted(ranges))}) are for the checks that search "
            f"them ({', '.join(SEARCHING)}), which checks= does not run; fix the parameters "
            "with params= instead"
        )
    for name in check_settings:
        check = BY_NAME[name]
        if check not in listed:
            raise ConfigError(f"{name}= sets the {name} check, which checks= does not run")
    return Request(
        listed=listed,
        objective=objective,
        runs=_whole("runs", runs, 1),
        seed=_whole("seed", seed, 0, _random.MAX_SEED),
        workers=None if workers is None else _whole("workers", workers, 1),
        settings={
            check.name: resolve(check.name, check.settings, check_settings.get(check.name))
            for check in listed
        },
        others=pairs,
        ranges=checked_ranges(ranges),
    )


__all__ = [
    "DEFAULT_OBJECTIVE",
    "DEFAULT_RUNS",
    "DEFAULT_SEED",
    "Request",
    "build_request",
    "select",
    "split_keywords",
]

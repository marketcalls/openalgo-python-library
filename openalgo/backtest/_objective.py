"""What `bt.optimize` ranks combinations by, and the key metrics of each (07-research-and-
reporting 1).

An objective is the name of a metric of `openalgo.backtest.metrics` (`"net_profit"`, the
default: end equity minus start equity, rupees; `"sharpe"`, `"sortino"`, `"calmar"`,
`"cagr"`, `"total_return"`, `"profit_factor"`, `"expectancy"`, `"win_rate"`, ...) or a
function that takes a `Result` and returns a number. The metrics are the run's own,
`Result.metrics`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .errors import ConfigError

DEFAULT_OBJECTIVE = "net_profit"
# The metrics every row of an optimisation table carries.
KEY_METRICS = ("net_profit", "total_return", "max_drawdown", "sharpe", "trades")
# Units of the metric definitions whose values are not numbers, so cannot rank a sweep.
NOT_NUMBERS = ("timestamp", "text", "flag")


def check_objective(objective: Any) -> None:
    """Raise ConfigError unless `objective` is a callable or the name of a metric whose
    value is a number."""
    if callable(objective):
        return
    from .metrics import DEFINITIONS

    if not isinstance(objective, str) or objective not in DEFINITIONS:
        raise ConfigError(
            "objective takes a metric name of openalgo.backtest.metrics (for example "
            f"'net_profit', 'sharpe', 'calmar') or a function of a Result, got {objective!r}"
        )
    unit = DEFINITIONS[objective].unit
    if unit in NOT_NUMBERS:
        raise ConfigError(
            f"objective {objective!r} is not a number ({unit}), so it cannot rank the "
            "combinations: name a metric that is one, such as 'net_profit', 'sharpe' or "
            "'calmar'"
        )


def evaluate(result: Any, objective: str | Callable[[Any], float]) -> dict[str, float]:
    """The objective (key "objective") and the key metrics of one result. Raises
    ConfigError when a function objective gives something other than a number."""
    figures = result.metrics
    row = {name: float(figures[name]) for name in KEY_METRICS}
    # A count of trades reads as a whole number in the table (8, not 8.0).
    row["trades"] = int(figures["trades"])
    value = objective(result) if callable(objective) else figures[objective]
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ConfigError(f"the objective gave {value!r}; it must give a number")
    row["objective"] = float(value)
    return row


__all__ = ["DEFAULT_OBJECTIVE", "KEY_METRICS", "check_objective", "evaluate"]

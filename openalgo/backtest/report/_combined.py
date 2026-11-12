"""Strategy-level and combined figures: the combined equity of strategies on fixed
allocations and the correlation of their session returns."""

from collections.abc import Mapping

import pandas as pd

from .._messages import counted
from ..errors import ConfigError
from . import _periods as periods
from ._inputs import check_opening_wealth, series_columns
from ._kernel import call
from ._performance import Performance


def combine_equity(
    equities: Mapping[str, pd.Series], opening_wealth: Mapping[str, float] | None = None
) -> pd.Series:
    """Implementation of `openalgo.backtest.metrics.combine_equity` (documented there)."""
    if not isinstance(equities, Mapping) or not equities:
        raise ConfigError("pass a mapping {strategy name: equity Series} with at least one entry")
    curves = [
        series_columns(series, what=f"equity of {name!r}") for name, series in equities.items()
    ]
    capitals = None if opening_wealth is None else _capitals(equities, opening_wealth)
    ts, total = call("combine", curves, capitals)
    return pd.Series(total, index=periods.ist_times(ts), name="equity")


def _capitals(equities: Mapping[str, pd.Series], opening_wealth: object) -> list[float]:
    """Each strategy's opening wealth in rupees, in the order of `equities`. Raises
    ConfigError unless `opening_wealth` maps exactly those names to positive numbers."""
    if not isinstance(opening_wealth, Mapping):
        raise ConfigError("opening_wealth maps each strategy name to its capital in rupees")
    missing = [name for name in equities if name not in opening_wealth]
    unknown = [name for name in opening_wealth if name not in equities]
    if missing or unknown:
        raise ConfigError(
            f"opening_wealth must give the capital of every strategy and of no other: "
            f"missing {missing}, unknown {unknown}"
        )
    return [
        check_opening_wealth(opening_wealth[name], what=f"opening_wealth of {name!r}")
        for name in equities
    ]


def return_correlation(strategies: Mapping[str, Performance | pd.Series]) -> pd.DataFrame:
    """Implementation of `openalgo.backtest.metrics.return_correlation`."""
    if not isinstance(strategies, Mapping) or len(strategies) < 2:
        raise ConfigError("pass a mapping {strategy name: Performance} with two or more entries")
    returns = {}
    for name, value in strategies.items():
        series = value.session_returns if isinstance(value, Performance) else value
        if not isinstance(series, pd.Series):
            raise ConfigError(f"{name!r} must be a Performance or a Series of session returns")
        missing = int(series.isna().sum())
        if missing:
            raise ConfigError(
                f"{name!r} has {counted(missing, 'missing session return')} (NaN): a missing "
                "return is not a move of 0, so drop it or give its value"
            )
        returns[name] = series
    # A session on which a strategy has no point left its equity unchanged: return 0.
    frame = pd.DataFrame(returns).sort_index().fillna(0.0)
    return frame.corr(method="pearson")


__all__ = ["combine_equity", "return_correlation"]

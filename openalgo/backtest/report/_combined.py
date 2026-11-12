"""Strategy-level and combined figures: the combined equity of strategies on fixed
allocations and the correlation of their session returns."""

from collections.abc import Mapping

import pandas as pd

from ..errors import ConfigError
from . import _periods as periods
from ._inputs import series_columns
from ._kernel import call
from ._performance import Performance


def combine_equity(equities: Mapping[str, pd.Series]) -> pd.Series:
    """Implementation of `openalgo.backtest.metrics.combine_equity` (documented there)."""
    if not isinstance(equities, Mapping) or not equities:
        raise ConfigError("pass a mapping {strategy name: equity Series} with at least one entry")
    curves = [
        series_columns(series, what=f"equity of {name!r}") for name, series in equities.items()
    ]
    ts, total = call("combine", curves)
    return pd.Series(total, index=periods.ist_times(ts), name="equity")


def return_correlation(strategies: Mapping[str, Performance | pd.Series]) -> pd.DataFrame:
    """Implementation of `openalgo.backtest.metrics.return_correlation`."""
    if not isinstance(strategies, Mapping) or len(strategies) < 2:
        raise ConfigError("pass a mapping {strategy name: Performance} with two or more entries")
    returns = {}
    for name, value in strategies.items():
        series = value.session_returns if isinstance(value, Performance) else value
        if not isinstance(series, pd.Series):
            raise ConfigError(f"{name!r} must be a Performance or a Series of session returns")
        returns[name] = series
    # A session on which a strategy has no point left its equity unchanged: return 0.
    frame = pd.DataFrame(returns).sort_index().fillna(0.0)
    return frame.corr(method="pearson")


__all__ = ["combine_equity", "return_correlation"]

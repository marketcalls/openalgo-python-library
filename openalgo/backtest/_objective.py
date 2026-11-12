"""What `bt.optimize` ranks combinations by, and the key metrics of each (07-research-and-
reporting 1).

An objective is the name of a metric of `openalgo.backtest.metrics` (`"net_profit"`, the
default: end equity minus capital, rupees; `"sharpe"`, `"sortino"`, `"calmar"`,
`"cagr"`, `"total_return"`, `"profit_factor"`, `"expectancy"`, `"win_rate"`, ...) or a
function that takes a `Result` and returns a number. The metrics are computed by
`metrics.analyse` on the run's equity (starting from the capital) and its closed trades.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .errors import ConfigError

DEFAULT_OBJECTIVE = "net_profit"
# The metrics every row of an optimisation table carries.
KEY_METRICS = ("net_profit", "total_return", "max_drawdown", "sharpe", "trades")


def check_objective(objective: Any) -> None:
    """Raise ConfigError unless `objective` is a known metric name or a callable."""
    if callable(objective):
        return
    from .metrics import DEFINITIONS

    if not isinstance(objective, str) or objective not in DEFINITIONS:
        raise ConfigError(
            "objective takes a metric name of openalgo.backtest.metrics (for example "
            f"'net_profit', 'sharpe', 'calmar') or a function of a Result, got {objective!r}"
        )


def _trades(result: Any) -> pd.DataFrame | None:
    """The closed trades in the layout `metrics.analyse` reads, or None without any."""
    trades = result.trades
    if trades.empty:
        return None
    return pd.DataFrame(
        {
            "entry_time": trades["entry_time"],
            "exit_time": trades["exit_time"],
            "entry_price": trades["entry_price_rupees"],
            "exit_price": trades["exit_price_rupees"],
            "quantity": trades["max_units"],
            "side": np.where(trades["direction"] == "long", "BUY", "SELL"),
            "gross_pnl": trades["gross_pnl_rupees"],
            "net_pnl": trades["net_pnl_rupees"],
            "costs": trades["costs_rupees"],
            "symbol": trades["symbol"],
        }
    )


def performance(result: Any, exchange: str) -> dict[str, Any]:
    """Every metric of a result, by name: its equity from the capital (one nanosecond
    before the first close) and its closed trades."""
    from . import metrics

    equity = result.equity["equity_rupees"]
    capital = float(result.assumptions["capital"])
    start = pd.Series([capital], index=[equity.index[0] - pd.Timedelta(1, unit="ns")])
    series = pd.concat([start, equity])
    return dict(metrics.analyse(series, exchanges=exchange, trades=_trades(result)).metrics)


def evaluate(
    result: Any, exchange: str, objective: str | Callable[[Any], float]
) -> dict[str, float]:
    """The objective (key "objective") and the key metrics of one result."""
    figures = performance(result, exchange)
    row = {name: float(figures[name]) for name in KEY_METRICS}
    value = objective(result) if callable(objective) else figures[objective]
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ConfigError(f"the objective gave {value!r}; it must give a number")
    row["objective"] = float(value)
    return row


__all__ = ["DEFAULT_OBJECTIVE", "KEY_METRICS", "check_objective", "evaluate", "performance"]

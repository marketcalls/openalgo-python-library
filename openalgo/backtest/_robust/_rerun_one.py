"""One rerun of the backtest on given bars, and what a check keeps of it.

A rerun is an ordinary backtest of the deterministic kernel, run as `bt.backtest` runs
one (`_pipeline.prepare` and `execute`) with the run's own arguments and `data=` the
given bars by symbol (`_run_bars.rerun_arguments`), so it reads them, checks them and
warms its indicators up from them as it would the user's own data. It is quiet: the
notices `bt.backtest` raises as Python warnings at the end of a run (orders not placed,
a wait for the warm-up) are about the generated bars, not the user's run, and are not
raised; each rerun's own warnings stay in its result.

The indicator checks that do not depend on the bars (an indicator's look-ahead check,
its warm-up measured on the engine's reference bars) are done once, by the first rerun
on the run's own bars, and passed to the others (`Known`), as `bt.optimize` passes them
from its first combination to the rest.

What a check keeps of a rerun (`RerunRow`): the objective, the net profit in rupees,
the maximum drawdown (a fraction, zero or negative), the closed trades, the largest win
and loss in rupees, the longest win and loss streaks in trades, the cumulative return
at the fan's sessions (`path`, fractions of the capital: point 0 is the start, point k
the close of the k-th session of the run), how large each symbol's position grew
(`sizes`, `_position_sizes`), the years it ran (`years`, its elapsed days over
`DAYS_PER_YEAR`, which annualises a total such as the net profit) and, for a family that
asks for it, its `RunDetail`: its equity at each session close, its closed trades and its
figures, which the out-of-sample and walk-forward checks stitch and tabulate. A rerun
whose bars could not be made or that
the data layer refuses (a DataError, `_reruns.run_batch`) keeps NaN figures and the
reason in `failure`; any other error is the strategy's or the engine's and is raised.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._causality_checks import IndicatorChecks
from .._objective import evaluate
from .._pipeline import execute, prepare
from ._position_sizes import PositionSize, position_sizes
from ._run_bars import strategy_name

#: Days in a year, as the metrics' CAGR counts elapsed time (`metrics`, Conventions).
DAYS_PER_YEAR = 365.25
#: The figures a `RunDetail` keeps, metric keys of the run (the in-sample against
#: out-of-sample table of design 17, section 2.1).
DETAIL_FIGURES = (
    "net_profit",
    "cagr",
    "sharpe",
    "profit_factor",
    "win_rate",
    "max_drawdown",
    "max_drawdown_amount",
)


@dataclass(frozen=True, eq=False)
class RunDetail:
    """What a rerun keeps beyond its row (module docstring): `session_equity`, its equity
    in rupees at the last equity point of each IST session date, indexed by that instant;
    `trades`, its closed trades as the metrics read them (`metrics.analyse`); `opening`,
    the instant it opened with its capital; `figures`, its `DETAIL_FIGURES` and `trades`
    (closed trades) and `trades_per_year`."""

    session_equity: pd.Series
    trades: pd.DataFrame
    opening: pd.Timestamp
    figures: Mapping[str, float]


@dataclass(frozen=True)
class Known:
    """Indicator checks a rerun need not do again: the look-ahead checks done (`keys`)
    and the warm-ups measured, by label (`measured`)."""

    keys: frozenset[str] = frozenset()
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class RerunRow:
    """What a check keeps of one rerun (module docstring)."""

    objective: float
    net_profit: float
    max_drawdown: float
    trades: int
    largest_win: float
    largest_loss: float
    win_streak: int
    loss_streak: int
    path: np.ndarray
    sizes: tuple[PositionSize, ...]
    failure: str = ""
    years: float = math.nan
    detail: RunDetail | None = None


def rerun(
    strategy: type,
    arguments: dict[str, Any],
    symbols: Sequence[str],
    frames: Sequence[pd.DataFrame],
    known: Known,
    on_path: Callable[[str], Any] | None = None,
    plan: Mapping[str, Any] | None = None,
) -> tuple[Any, Known]:
    """(the `Result` of `strategy` on `frames`, the bars of `symbols`, with the run's
    `arguments`; the indicator checks known after it). `on_path` is told the evaluation
    path the rerun took; `plan` sets settings of the run's plan that no run argument
    gives (`_settings.RunPlan.bar_shift`, `exit_scales`). Raises what the run raises."""
    args = {**arguments, "data": dict(zip(symbols, frames, strict=True))}
    checks = IndicatorChecks(strategy_name(strategy), known.keys, measured=known.measured)
    prepared = prepare(strategy, None, args, checks=checks)
    if plan:
        prepared = dataclasses.replace(prepared, plan=dataclasses.replace(prepared.plan, **plan))
    result = execute(strategy, prepared, args, checks=checks, on_path=on_path)
    return result, Known(checks.keys, checks.measured())


def path_at(result: Any, dates: pd.DatetimeIndex, steps: np.ndarray) -> np.ndarray:
    """The cumulative return of `result` at `steps` of the run's sessions `dates` (module
    docstring); a session the rerun has no return for keeps the return before it."""
    returns = result.performance.session_returns
    days = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    grown = pd.Series((1.0 + returns.to_numpy(dtype=float)).cumprod() - 1.0, index=days)
    wanted = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    at_close = grown.reindex(wanted, method="ffill").fillna(0.0).to_numpy(dtype=float)
    return np.concatenate(([0.0], at_close))[steps]


def run_detail(result: Any) -> RunDetail:
    """The `RunDetail` of a rerun's `result`."""
    equity = result.equity["equity_rupees"]
    days = equity.index.normalize()
    at_close = equity[~days.duplicated(keep="last")]
    trades = result.performance.trades
    metrics = result.metrics
    years = float(metrics["elapsed_days"]) / DAYS_PER_YEAR
    figures = {name: float(metrics[name]) for name in DETAIL_FIGURES}
    figures["trades"] = float(metrics["trades"])
    figures["trades_per_year"] = figures["trades"] / years if years > 0 else math.nan
    closed = trades[trades["exit_time"].notna()].reset_index(drop=True)
    return RunDetail(at_close, closed, pd.Timestamp(metrics["start"]), figures)


def measured_row(
    result: Any,
    objective: Any,
    dates: pd.DatetimeIndex,
    steps: np.ndarray,
    detail: bool = False,
) -> RerunRow:
    """The `RerunRow` of a rerun's `result` under `objective`, with its `RunDetail`
    when `detail`."""
    metrics = result.metrics
    return RerunRow(
        objective=evaluate(result, objective)["objective"],
        net_profit=float(metrics["net_profit"]),
        max_drawdown=float(metrics["max_drawdown"]),
        trades=_count(metrics["trades"]),
        largest_win=float(metrics["largest_win"]),
        largest_loss=float(metrics["largest_loss"]),
        win_streak=_count(metrics["max_consecutive_wins"]),
        loss_streak=_count(metrics["max_consecutive_losses"]),
        path=path_at(result, dates, steps),
        sizes=position_sizes(result),
        years=float(metrics["elapsed_days"]) / DAYS_PER_YEAR,
        detail=run_detail(result) if detail else None,
    )


def _count(value: Any) -> int:
    """A count of the metrics, 0 when it is undefined (a run with no closed trade)."""
    number = float(value)
    return 0 if math.isnan(number) else int(number)


def failed_row(reason: str, points: int) -> RerunRow:
    """The row of a rerun left out for `reason` (its bars could not be made, or the data
    layer refused them), with `points` points of its path."""
    nan = math.nan
    return RerunRow(nan, nan, nan, 0, nan, nan, 0, 0, np.full(points, nan), (), reason)


__all__ = [
    "DAYS_PER_YEAR",
    "DETAIL_FIGURES",
    "Known",
    "RerunRow",
    "RunDetail",
    "failed_row",
    "measured_row",
    "path_at",
    "rerun",
    "run_detail",
]

"""Rolling statistics and the horizon summary of session returns.

Rolling: volatility, Sharpe, Sortino, session win rate and beta over trailing windows
of sessions, computed by the kernel exactly as for the whole run. The default window is
half a year of sessions of the traded calendar.

Horizons: for each of 1, 3 and 6 months and 1, 3 and 5 years, the window from the same
calendar date that many months before the last session date (the month's last day when
that date does not exist) to the end. A horizon is listed only when the run started on
or before that date. Its figures are those of the sessions after the date: total return,
CAGR over the calendar days of the window, volatility, Sharpe, the deepest drawdown at
session closes and, with a benchmark, the benchmark's return over the same sessions.
"Since start" covers the whole run, with CAGR over its elapsed time.
"""

import numbers

import numpy as np
import pandas as pd

from ..errors import ConfigError
from . import _periods as periods
from ._kernel import call

#: The horizons of the summary: (label, months).
HORIZONS = (("1M", 1), ("3M", 3), ("6M", 6), ("1Y", 12), ("3Y", 36), ("5Y", 60))
#: The label of the row that covers the whole run.
SINCE_START = "Since start"
#: Figures of the horizon summary, as `metrics_path` names them.
HORIZON_FIGURES = ("total_return", "cagr", "volatility", "sharpe", "max_drawdown")
#: Rolling statistics, as the kernel names them (beta only with a benchmark).
ROLLING_COLUMNS = ("volatility", "sharpe", "sortino", "win_rate", "beta")
#: Shortest rolling window: a standard deviation needs two sessions.
MIN_WINDOW_SESSIONS = 2


def default_window(sessions_per_year: float) -> int:
    """Half a year of sessions, rounded, and never below `MIN_WINDOW_SESSIONS`."""
    return max(MIN_WINDOW_SESSIONS, round(sessions_per_year / 2))


def check_window(value: object) -> int:
    """A rolling window as a whole number of sessions. Raises ConfigError otherwise."""
    whole = isinstance(value, numbers.Integral) and not isinstance(value, bool)
    if not whole or int(value) < MIN_WINDOW_SESSIONS:  # type: ignore[arg-type]
        raise ConfigError(
            f"rolling_window is a number of sessions, {MIN_WINDOW_SESSIONS} or more, got {value!r}"
        )
    return int(value)  # type: ignore[arg-type]


def rolling_frame(
    returns: pd.Series,
    benchmark: pd.Series | None,
    window: int,
    sessions_per_year: float,
    rate: float,
) -> pd.DataFrame:
    """Rolling statistics indexed by session date (see the module docstring)."""
    bench = None if benchmark is None else benchmark.to_numpy(dtype=float)
    out = call("rolling", returns.to_numpy(dtype=float), bench, window, sessions_per_year, rate)
    columns = {name: out[name] for name in ROLLING_COLUMNS if name in out}
    return pd.DataFrame(columns, index=returns.index)


def _row(
    label: str,
    since: pd.Timestamp,
    window: pd.Series,
    bench: pd.Series | None,
    elapsed_ns: int,
    annual: tuple[float, float],
) -> dict:
    figures = call("path", window.to_numpy(dtype=float), *annual, elapsed_ns)
    row = {"horizon": label, "from": since.date(), "sessions": len(window)}
    row |= {name: figures[name] for name in HORIZON_FIGURES}
    if bench is not None:
        growth = (1.0 + bench).to_numpy(dtype=float)
        row["benchmark_return"] = float(np.prod(growth)) - 1.0 if len(growth) else 0.0
    return row


def horizon_frame(
    returns: pd.Series,
    benchmark: pd.Series | None,
    start_ns: int,
    end_ns: int,
    sessions_per_year: float,
    rate: float,
) -> pd.DataFrame:
    """The horizon summary (see the module docstring), one row per horizon covered."""
    annual = (sessions_per_year, rate)
    start = pd.Timestamp(periods.ist_date(start_ns))
    end = pd.Timestamp(periods.ist_date(end_ns))
    rows = []
    for label, months in HORIZONS:
        since = end - pd.DateOffset(months=months)
        if start > since:
            continue
        inside = returns.index > since
        window = returns[inside]
        bench = None if benchmark is None else benchmark[inside]
        elapsed = (end - since).days * periods.NANOS_PER_DAY
        rows.append(_row(label, since, window, bench, elapsed, annual))
    rows.append(_row(SINCE_START, start, returns, benchmark, end_ns - start_ns, annual))
    return pd.DataFrame(rows).set_index("horizon")


__all__ = [
    "HORIZONS",
    "HORIZON_FIGURES",
    "MIN_WINDOW_SESSIONS",
    "ROLLING_COLUMNS",
    "SINCE_START",
    "check_window",
    "default_window",
    "horizon_frame",
    "rolling_frame",
]

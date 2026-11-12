"""Turn the inputs of `metrics.analyse` into the columns the native kernel reads.

Timestamps are accepted as time-zone-aware datetimes (any zone) or as int64 UTC
nanoseconds. Naive datetimes are refused rather than guessed: a wrong zone would move
every point by 5 hours 30 minutes and could change its IST session date.
"""

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._messages import counted
from ..costs import COMPONENTS
from ..errors import ConfigError

#: Columns a trades table must have (see `metrics.analyse`).
TRADE_COLUMNS = (
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "side",
    "gross_pnl",
    "net_pnl",
    "costs",
)
#: Trade sides: the side of the entry order, so "BUY" is a long trade.
SIDES = ("BUY", "SELL")
#: Optional per-trade cost columns summed into the cost breakdown, in rupees: the six
#: components of `openalgo.backtest.costs` plus the depository charge on delivery sells.
COST_COLUMNS = (*COMPONENTS, "dp_charge")
#: The exit time the kernel reads as "still held at the end of the run".
OPEN_EXIT_NANOS = int(np.iinfo(np.int64).max)
#: Integer timestamps before 1980-01-01 UTC are refused. Seconds, milliseconds or
#: microseconds since the epoch read as nanoseconds all land in January 1970, and no
#: Indian market data a backtest can load is that old.
EARLIEST_INTEGER_NANOS = int(pd.Timestamp("1980-01-01", tz="UTC").value)


def utc_nanos(values: object, *, what: str) -> tuple[np.ndarray, np.ndarray]:
    """(int64 UTC nanoseconds, missing mask) for an index, column or array of times.

    Missing values (NaT, NA) are 0 in the first array and True in the mask. Raises
    ConfigError for naive datetimes and for values that are not times.
    """
    series = values if isinstance(values, pd.Index | pd.Series) else pd.Index(values)
    dtype = series.dtype
    if len(series) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    if isinstance(dtype, pd.DatetimeTZDtype):
        index = pd.DatetimeIndex(series)
        missing = np.asarray(index.isna())
        nanos = index.tz_convert("UTC").as_unit("ns").asi8
        return np.where(missing, 0, nanos).astype(np.int64), missing
    if pd.api.types.is_datetime64_dtype(dtype):
        raise ConfigError(
            f"{what} carry no time zone. Localise them first, for example "
            ".tz_localize('Asia/Kolkata') for IST clock times: a guessed zone would move "
            "every point by 5 hours 30 minutes."
        )
    if pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        missing = np.asarray(pd.isna(series))
        nanos = pd.array(series, dtype="Int64").to_numpy(dtype=np.int64, na_value=0)
        if np.any(nanos[~missing] < EARLIEST_INTEGER_NANOS):
            raise ConfigError(
                f"{what} are integers before 1980 when read as UTC nanoseconds: they look "
                "like seconds, milliseconds or microseconds. Convert them to nanoseconds, "
                "or pass time-zone-aware datetimes."
            )
        return nanos, missing
    raise ConfigError(
        f"{what} must be time-zone-aware datetimes or int64 UTC nanoseconds, got {dtype}"
    )


def _floats(values: object, *, what: str) -> np.ndarray:
    try:
        return np.asarray(pd.to_numeric(pd.Series(values), errors="raise"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{what} must be numbers: {exc}") from None


def series_columns(series: object, *, what: str) -> tuple[np.ndarray, np.ndarray]:
    """(int64 UTC nanoseconds, float64 values) of a time-indexed Series.

    Raises ConfigError for anything but a pandas Series, a missing timestamp or a
    value that is not a number.
    """
    if not isinstance(series, pd.Series):
        raise ConfigError(
            f"{what} must be a pandas Series indexed by time, got {type(series).__name__}"
        )
    nanos, missing = utc_nanos(series.index, what=f"{what} timestamps")
    if missing.any():
        raise ConfigError(f"{what} has {counted(int(missing.sum()), 'missing timestamp')}")
    return nanos, _floats(series.to_numpy(), what=what)


def check_opening_wealth(value: object, *, what: str = "opening_wealth") -> float:
    """A capital before the first equity point, in rupees, as a float.

    Raises ConfigError for anything but a positive finite number (True, text and NaN
    included): a wrong opening wealth would rescale every return of the run.
    """
    real = isinstance(value, numbers.Real) and not isinstance(value, bool)
    number = float(value) if real else math.nan  # type: ignore[arg-type]
    if not (math.isfinite(number) and number > 0):
        raise ConfigError(
            f"{what} is the capital before the first equity point, a positive number of "
            f"rupees, got {value!r}"
        )
    return number


def flow_columns(cash_flows: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    """The cash flows as (timestamps, rupees), empty when there are none."""
    if cash_flows is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return series_columns(cash_flows, what="cash_flows")


def benchmark_columns(benchmark: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The benchmark as (timestamps, values). Values must be positive levels (an index
    or a price), never returns. Raises ConfigError otherwise."""
    nanos, values = series_columns(benchmark, what="benchmark")
    if len(values) == 0 or not np.all(np.isfinite(values) & (values > 0)):
        raise ConfigError(
            "benchmark must hold positive index or price levels, one per timestamp (not "
            "returns), with no missing value"
        )
    if np.any(np.diff(nanos) <= 0):
        raise ConfigError("benchmark timestamps must increase: sort it and drop duplicates")
    return nanos, values


@dataclass(frozen=True)
class TradeTable:
    """A checked trades table as columns. Times are int64 UTC nanoseconds, money is
    rupees, quantity is units. Open trades (no exit time) have `is_open` True."""

    frame: pd.DataFrame
    entry_ts: np.ndarray
    exit_ts: np.ndarray
    is_open: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray
    quantity: np.ndarray
    gross_pnl: np.ndarray
    net_pnl: np.ndarray
    costs: np.ndarray

    def closed(self, column: np.ndarray) -> np.ndarray:
        """The rows of `column` that belong to closed trades."""
        return column[~self.is_open]

    def holding_exits(self) -> np.ndarray:
        """Exit times with open trades held to the end, for the exposure kernel."""
        return np.where(self.is_open, OPEN_EXIT_NANOS, self.exit_ts).astype(np.int64)

    def cost_breakdown(self) -> dict[str, float]:
        """Rupees per cost column present in the table, summed over closed trades."""
        present = [name for name in COST_COLUMNS if name in self.frame.columns]
        closed = self.frame.loc[~self.is_open]
        return {name: float(_floats(closed[name], what=name).sum()) for name in present}


def trade_table(trades: pd.DataFrame | None) -> TradeTable:
    """Check a trades table (columns in `TRADE_COLUMNS`) and return its columns.

    Raises ConfigError for a missing column, a missing entry time, a side other than
    "BUY" or "SELL", a quantity that is not positive, or a price that is missing where
    it is needed. P&L and time order are checked by the kernel.
    """
    frame = pd.DataFrame(columns=list(TRADE_COLUMNS)) if trades is None else trades
    if not isinstance(frame, pd.DataFrame):
        raise ConfigError(f"trades must be a pandas DataFrame, got {type(frame).__name__}")
    absent = [name for name in TRADE_COLUMNS if name not in frame.columns]
    if absent:
        raise ConfigError(f"trades is missing column(s) {absent}; it needs {list(TRADE_COLUMNS)}")
    entry_ts, entry_missing = utc_nanos(frame["entry_time"], what="trades entry_time")
    exit_ts, is_open = utc_nanos(frame["exit_time"], what="trades exit_time")
    if entry_missing.any():
        raise ConfigError("trades has a row with no entry_time: every trade needs one")

    def number(name: str) -> np.ndarray:
        return _floats(frame[name], what=f"trades {name}")

    table = TradeTable(
        frame=frame,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        is_open=is_open,
        entry_price=number("entry_price"),
        exit_price=number("exit_price"),
        quantity=number("quantity"),
        gross_pnl=number("gross_pnl"),
        net_pnl=number("net_pnl"),
        costs=number("costs"),
    )
    _check_trade_values(table)
    return table


def _check_trade_values(table: TradeTable) -> None:
    sides = table.frame["side"].astype(str)
    if not sides.isin(SIDES).all():
        bad = sorted(set(sides[~sides.isin(SIDES)]))
        raise ConfigError(f"trades side must be 'BUY' (long) or 'SELL' (short), found {bad}")
    if not np.all(np.isfinite(table.quantity) & (table.quantity > 0)):
        raise ConfigError("trades quantity must be a positive number of units (never lots)")
    if not np.all(np.isfinite(table.entry_price)):
        raise ConfigError("trades entry_price is missing on some row")
    if not np.all(np.isfinite(table.closed(table.exit_price))):
        raise ConfigError("trades exit_price is missing on a trade that has an exit_time")


__all__ = [
    "COST_COLUMNS",
    "OPEN_EXIT_NANOS",
    "SIDES",
    "TRADE_COLUMNS",
    "TradeTable",
    "benchmark_columns",
    "flow_columns",
    "check_opening_wealth",
    "series_columns",
    "trade_table",
    "utc_nanos",
]

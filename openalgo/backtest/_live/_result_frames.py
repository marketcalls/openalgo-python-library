"""The tables of a live session's result (16-session-design 1.4; unit 13), as pandas
DataFrames built from the session's records (`_session_records`, `_session_close`).

Each table has its columns whether or not it has rows. An instant is an IST datetime
column beside the UTC nanoseconds it came from (`*_ns`), as the backtest's tables keep
both; one not reached (a lock still standing, a window still open) is NaT, its
nanoseconds missing. Prices are rupees, quantities units.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, fields
from typing import Any

import pandas as pd

from .._result_tables import IST
from ._fill_comparison import FillRow
from ._reconcile_findings import Reconciliation
from ._session_open import OpenPosition
from ._session_records import FillRecord, LockRecord, OrderRecord
from ._session_trader import Skipped
from ._stops_cover import Unprotected
from .bars import BarDrift

#: The reconciliations' columns of a shared instrument's group (`Reconciliation`), which
#: only a session with an instrument several strategies share has: a session of strategies
#: alone on their instruments keeps its table as it was before shared instruments.
GROUP_COLUMNS = (
    "group",
    "group_units",
    "member_units",
    "external_units",
    "allocation",
    "stops_cut",
)
#: The order of the open positions' columns: what is held, then its stop, then the note.
OPEN_POSITION_COLUMNS = (
    "strategy",
    "symbol",
    "exchange",
    "units",
    "stop_order_id",
    "stop_confirmed",
    "note",
)


def frame(
    rows: Iterable[Any], columns: Sequence[str], instants: Iterable[tuple[str, str]] = ()
) -> pd.DataFrame:
    """A table of `rows` (dataclasses) with `columns`, and for each (nanoseconds column,
    time column) of `instants` an IST time column inserted before the nanoseconds one,
    which holds whole nanoseconds or a missing value."""
    table = pd.DataFrame([asdict(row) for row in rows], columns=list(columns))
    for nanos, name in instants:
        table[nanos] = pd.Series(_nanos(table[nanos]), index=table.index, dtype="Int64")
        utc = pd.to_datetime(table[nanos], unit="ns", utc=True)
        table.insert(table.columns.get_loc(nanos), name, utc.dt.as_unit("ns").dt.tz_convert(IST))
    return table


def orders_frame(orders: Sequence[OrderRecord]) -> pd.DataFrame:
    """Every order the session placed (`_session_records.OrderRecord`), `created` its
    IST instant."""
    return frame(orders, _names(OrderRecord), [("created_ns", "created")])


def fill_comparison_frame(rows: Sequence[FillRow]) -> pd.DataFrame:
    """Each fill against a backtest of the same bars (`_fill_comparison.FillRow`),
    `live_time` and `backtest_time` the IST instants of `live_ns` and `backtest_ns` (NaT
    for a row not compared)."""
    instants = [("live_ns", "live_time"), ("backtest_ns", "backtest_time")]
    return frame(rows, _names(FillRow), instants)


def trades_frame(trades: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Every round trip the strategies closed in this start, each strategy's `Result.trades`
    (by name, in call order) one after the other with `strategy` and `entry_day` (the IST
    date of `entry_time`) first: a trade carried from an earlier day shows the day it was
    entered (20-positional-live 4; the lead's ruling N5a-4), and its fills' flags
    ("price_unverified", "missed_day") are in its `flags`."""
    tables = []
    for name, table in trades.items():
        table = table.copy()
        table.insert(0, "strategy", name)
        table.insert(1, "entry_day", table["entry_time"].dt.date)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def fills_frame(fills: Sequence[FillRecord]) -> pd.DataFrame:
    """Every fill the engine took (`_session_records.FillRecord`), `time` when it was
    heard, IST."""
    return frame(fills, _names(FillRecord), [("at_ns", "time")])


def locks_frame(locks: Sequence[LockRecord]) -> pd.DataFrame:
    """Every entry lock: its strategy and reason, `set` and `lifted` (NaT while it
    stands), IST."""
    return frame(locks, _names(LockRecord), [("set_ns", "set"), ("lifted_ns", "lifted")])


def unprotected_frame(windows: Sequence[Unprotected], ended_ns: int) -> pd.DataFrame:
    """Every window a position had no confirmed stop at the broker: the leg, `from` and
    `to` (NaT for one still open at the end), the units and why, and its `seconds` up to
    the end (`ended_ns`, UTC nanoseconds)."""
    table = frame(windows, _names(Unprotected), [("from_ns", "from"), ("to_ns", "to")])
    table["seconds"] = pd.Series([w.seconds(ended_ns) for w in windows], dtype="float64")
    return table


def drift_frame(drift: Sequence[BarDrift]) -> pd.DataFrame:
    """Each broker candle against the tick-built bar of its slice: the candle's prices
    and volume less the bar's, `time` the IST open of the bar's slice."""
    return frame(drift, _names(BarDrift), [("ts_open_ns", "time")])


def reconciliations_frame(rows: Sequence[Reconciliation], *, shared: bool) -> pd.DataFrame:
    """Each comparison with the broker's books that differed, `time` when, IST, with the
    engine's and the broker's units (missing: not read) and what was done; in a session
    with an instrument several strategies share (`shared`), the `GROUP_COLUMNS` too: of a
    member of that instrument, its group's figures, the allocation made and its stop cut
    (`_reconcile_findings.Reconciliation`; missing or "" for a strategy alone)."""
    names = _names(Reconciliation)
    if not shared:
        names = tuple(name for name in names if name not in GROUP_COLUMNS)
    table = frame(rows, names, [("at_ns", "time")])
    counts = ("broker_units", "group_units", "external_units") if shared else ("broker_units",)
    for column in counts:
        table[column] = table[column].astype("Int64")
    return table


def skipped_frame(skipped: Sequence[Skipped]) -> pd.DataFrame:
    """Each strategy bar whose `on_bar` was skipped, `decided_at` the IST decision, and
    why."""
    return frame(skipped, _names(Skipped), [("decided_at_ns", "decided_at")])


def open_positions_frame(positions: Sequence[OpenPosition]) -> pd.DataFrame:
    """Each position still open at the end, with its stop and a note (`OpenPosition`)."""
    return frame(positions, OPEN_POSITION_COLUMNS)


def _names(kind: type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(kind))


def _nanos(column: pd.Series) -> list[int | None]:
    """Nanoseconds as integers, a missing one None."""
    return [None if pd.isna(value) else int(value) for value in column]


__all__ = [
    "GROUP_COLUMNS",
    "OPEN_POSITION_COLUMNS",
    "drift_frame",
    "fill_comparison_frame",
    "fills_frame",
    "frame",
    "locks_frame",
    "open_positions_frame",
    "orders_frame",
    "reconciliations_frame",
    "skipped_frame",
    "trades_frame",
    "unprotected_frame",
]

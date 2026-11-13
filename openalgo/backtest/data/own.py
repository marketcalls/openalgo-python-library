"""Own data without a server: `Frame` for tables in memory.

Three things must be declared, and none is guessed:

- the columns: found by their canonical names (any letter case) or through
  `columns={"timestamp": "Date", "open": "Open", ...}`. An ambiguous or missing
  column is a ConfigError that lists what is available;
- the time zone: a tz-aware index or column, or `tz=` for naive timestamps. There
  is no default, except that daily and longer bars whose naive timestamps are all at
  midnight are session dates, read as IST dates (`_stamps`);
- the bar label: `label="open"` (the stamp is the instant the bar opens, the
  default and the OpenAlgo convention) or `label="close"`.

The rows then go through the same normalisation, calendar validation and quality
checks as OpenAlgo data. Rows are selected by date only after normalisation, so a
bar labelled outside the requested dates (a month stamped at its last day) is not
lost. `files.Files` reads the same tables from CSV, Parquet and Feather files.
"""

import datetime as dt
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .._calendar_convert import DateLike, format_ist
from .._layout import BAR_COLUMNS, PRICE_COLUMNS, is_intraday, parse_interval
from .._messages import counted
from .._protocols import SessionCalendar
from ..errors import ConfigError, DataError, NoDataError
from ._sessions import NANOS_PER_DAY, NowLike, day_start_ns
from ._stamps import to_utc_ns
from .normalise import LABELS, VALUE_COLUMNS, Stamped

# Canonical fields a mapping may name: the timestamp, the value columns of the
# layout, and two optional columns that select rows of a multi-instrument table.
FIELDS = ("timestamp", *VALUE_COLUMNS, "symbol", "exchange")

_INDEX = "<index>"  # the timestamp is the index of the table
_CANONICAL = "<canonical>"  # the timestamp is ts_open of a frame in the canonical layout
_TIMESTAMP_NAMES = ("timestamp", "datetime", "date", "time", "ts")
# A bar can be labelled this many days outside the requested dates and still lie
# inside them: a period stamped at its last calendar day, a zone a day away.
_LABEL_MARGIN_DAYS = {"intraday": 2, "D": 2, "W": 8, "M": 32, "Q": 93, "Y": 367}
_SHOWN_VALUES = 10

ColumnSource = str | tuple[str, ...]


def _to_table(data: object, columns: object) -> pd.DataFrame:
    """Turn in-memory data into a pandas DataFrame. The input is never mutated."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, Mapping):
        try:
            return pd.DataFrame(dict(data))
        except ValueError as exc:
            raise ConfigError(f"the columns of the dict do not line up: {exc}") from exc
    if isinstance(data, np.ndarray) and data.dtype.names:
        return pd.DataFrame.from_records(data)
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or not isinstance(columns, list | tuple):
            raise ConfigError(
                "a plain NumPy array needs columns=[...] naming its columns in order (for "
                "example ['timestamp', 'open', 'high', 'low', 'close', 'volume']); a "
                "structured array or a dict of arrays needs no names"
            )
        if len(columns) != data.shape[1]:
            raise ConfigError(
                f"columns names {counted(len(columns), 'field')} but the array has "
                f"{counted(data.shape[1], 'column')}"
            )
        return pd.DataFrame(data, columns=list(columns))
    # A dataframe of another library, by duck typing: no import, no pyarrow needed.
    if hasattr(data, "collect") and not hasattr(data, "to_dict"):
        data = data.collect()  # a lazy polars frame
    tried = ""
    if not isinstance(data, pd.Series) and hasattr(data, "to_dict") and hasattr(data, "columns"):
        try:
            return pd.DataFrame(data.to_dict(as_series=False))
        except TypeError as exc:  # not the polars signature: try to_pandas, else say why
            tried = f" (to_dict(as_series=False) failed: {exc})"
    if not isinstance(data, pd.Series) and hasattr(data, "to_pandas"):
        return data.to_pandas()
    raise ConfigError(
        f"unsupported data of type {type(data).__name__}{tried}: pass a pandas or polars "
        "DataFrame, a dict of columns or NumPy arrays"
    )


def _resolve_columns(table: pd.DataFrame, columns: object) -> dict[str, ColumnSource]:
    """Map each canonical field to the column, pair of columns, index or canonical
    `ts_open` that holds it. Raises ConfigError when a field cannot be told."""
    mapping = dict(columns) if isinstance(columns, Mapping) else {}
    available = [str(name) for name in table.columns]
    unknown = [key for key in mapping if key not in FIELDS]
    if unknown:
        raise ConfigError(
            f"columns= has unknown keys {unknown}; the keys are {list(FIELDS)} and the values "
            "are the names in your data"
        )
    by_lower_name: dict[str, list] = {}
    for name in table.columns:
        by_lower_name.setdefault(str(name).lower(), []).append(name)

    resolved = {}
    for name in FIELDS:
        if name in mapping:
            resolved[name] = _mapped_column(table, name, mapping[name], available)
        elif name != "timestamp" and name in by_lower_name:
            resolved[name] = _only(by_lower_name[name], repr(name))
    if "timestamp" not in resolved:
        if isinstance(table.index, pd.DatetimeIndex):
            resolved["timestamp"] = _INDEX
        elif all(name in table.columns for name in BAR_COLUMNS[:2]):
            resolved["timestamp"] = _CANONICAL
        else:
            candidates = [name for key in _TIMESTAMP_NAMES for name in by_lower_name.get(key, [])]
            if not candidates:
                raise ConfigError(
                    "no timestamp found: use a DatetimeIndex, a column named one of "
                    f"{list(_TIMESTAMP_NAMES)}, or columns={{'timestamp': ...}}. "
                    f"Available: {available}"
                )
            resolved["timestamp"] = _only(candidates, "the timestamp")
    missing = [name for name in PRICE_COLUMNS if name not in resolved]
    if missing:
        raise ConfigError(
            f"the data has no column for {missing}. Available: {available}. Map them with "
            "columns={'open': ..., 'high': ..., 'low': ..., 'close': ...}."
        )
    return resolved


def _only(candidates: list, what: str) -> str:
    if len(candidates) > 1:
        raise ConfigError(
            f"several columns could be {what}: {candidates}. Say which one with columns= "
            "(for the timestamp, a pair (date_column, time_column) joins two columns)."
        )
    return candidates[0]


def _mapped_column(table: pd.DataFrame, name: str, wanted: object, available: list) -> ColumnSource:
    names = list(wanted) if isinstance(wanted, list | tuple) else [wanted]
    if name != "timestamp" and len(names) != 1:
        raise ConfigError(f"columns[{name!r}] must name one column")
    is_index = name == "timestamp" and names == [table.index.name]
    for column in names:
        if column not in table.columns and not is_index:
            raise ConfigError(
                f"columns[{name!r}] names {column!r}, which is not in the data. "
                f"Available: {available}"
            )
    if len(names) > 1:
        return tuple(names)
    return names[0] if names[0] in table.columns else _INDEX


class Frame:
    """Bars from in-memory data, as a data provider.

    data: a pandas DataFrame, a dict of columns, a structured NumPy array, a 2-D
        NumPy array with `columns=[names in order]`, or a polars DataFrame (read
        through its `to_dict`, so pyarrow is not needed).
    tz: time zone of naive timestamps, for example "Asia/Kolkata". Required unless
        the timestamps are tz-aware, or the bars are daily or longer and every naive
        timestamp is at midnight (session dates, read as IST dates). No default.
    columns: mapping from canonical field to the name in the data. Fields:
        timestamp, open, high, low, close, volume, oi, symbol, exchange. The
        timestamp may be a pair `(date_column, time_column)`.
    label: "open" or "close", the instant an intraday stamp refers to.
    unit: "s", "ms", "us" or "ns" when timestamps are numbers.
    timestamp_format: strptime pattern for timestamp strings that are not ISO 8601.
    symbol, exchange, interval: what the data holds. When given, a request for
        anything else is a ConfigError; when omitted, the request is trusted.
    calendar, now: the session calendar and the current time that `bars` uses.
    regular_session_only: drop pre-open, post-close and closed-day rows (counted).

    A frame already in the canonical layout (`ts_open` and `ts_close` as int64 UTC
    nanoseconds) is accepted as it is and needs no `tz`.

    Raises ConfigError at construction for an unknown label or interval.
    """

    name = "frame"

    def __init__(
        self,
        data: object,
        *,
        tz: str | None = None,
        columns: Mapping[str, object] | list[str] | tuple[str, ...] | None = None,
        label: str = "open",
        unit: str | None = None,
        timestamp_format: str | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        interval: str | None = None,
        calendar: SessionCalendar | None = None,
        now: NowLike = None,
        regular_session_only: bool = True,
    ) -> None:
        if label not in LABELS:
            raise ConfigError(f"label must be 'open' or 'close', got {label!r}")
        if interval is not None:
            parse_interval(interval)
        if columns is not None and not isinstance(columns, Mapping | list | tuple):
            raise ConfigError("columns must be a mapping (or a list of names for a 2-D array)")
        self.capabilities = {"bars", "oi"}
        self.timestamp_convention = f"{tz if tz is not None else 'tz-aware'}/{label}"
        self._data = data
        self._tz = tz
        self._columns = columns
        self._label = label
        self._unit = unit
        self._format = timestamp_format
        self._held = {"symbol": symbol, "exchange": exchange, "interval": interval}
        self._calendar = calendar
        self._now = now
        self._regular_only = bool(regular_session_only)
        self._table: pd.DataFrame | None = None

    def bars(
        self, symbol: str, exchange: str, interval: str, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Canonical bars for the inclusive IST date range. `load_bars(data=frame)`
        returns the same bars together with their metadata, and raises the same
        errors."""
        from .loader import load_bars  # the loader is built on this module

        options = {"data": self, "calendar": self._calendar, "now": self._now}
        return load_bars(symbol, exchange, interval, start, end, **options).bars

    def stamped(
        self, symbol: str, exchange: str, interval: str, start: dt.date, end: dt.date
    ) -> Stamped:
        """The rows that can belong to the inclusive IST dates, stamped in UTC, for
        `build_bars` to normalise and select.

        Raises ConfigError for a request the data was declared not to hold, for
        undeclared time zones and for columns that cannot be found; NoDataError when
        the data holds nothing near the dates; DataError for unreadable files.
        """
        asked = {"symbol": symbol, "exchange": exchange, "interval": interval}
        for what, held in self._held.items():
            if held is not None and held != asked[what]:
                advice = " Load the base interval and resample (base=...)."
                raise ConfigError(
                    f"this {type(self).__name__} holds {what} {held!r} but {asked[what]!r} was "
                    f"requested.{advice if what == 'interval' else ''}"
                )
        table, origin = self._read_table(symbol, exchange, interval)
        fields = _resolve_columns(table, self._columns)
        table = _select_rows(table, fields, {"symbol": symbol, "exchange": exchange}, origin)
        stamps_ns, described, canonical = self._stamps(table, fields, origin, interval)
        margin_ns = _LABEL_MARGIN_DAYS[parse_interval(interval)[0]] * NANOS_PER_DAY
        low, high = day_start_ns(start) - margin_ns, day_start_ns(end) + NANOS_PER_DAY + margin_ns
        rows = np.flatnonzero((stamps_ns >= low) & (stamps_ns < high))
        if len(rows) == 0:
            held = "it is empty"
            if len(stamps_ns):
                first, last = format_ist(int(stamps_ns.min())), format_ist(int(stamps_ns.max()))
                rows = counted(len(stamps_ns), "row runs", "rows run")
                held = f"its {rows} from {first} to {last}"
            raise NoDataError(
                f"{exchange}:{symbol} {interval}: {origin} holds no rows between {start} and "
                f"{end} ({held})"
            )
        declared = f"The timestamps were read as {described}."
        return Stamped(
            stamps_ns[rows],
            {name: table[fields[name]].iloc[rows] for name in VALUE_COLUMNS if name in fields},
            label="open" if canonical else self._label,
            canonical=canonical,
            regular_session_only=self._regular_only,
            hints={
                "late_5h30": f"{declared} If the data holds IST clock times, declare "
                "tz='Asia/Kolkata'.",
                "early_5h30": f"{declared} If the data holds UTC times, declare tz='UTC'.",
                "close_labelled": f"label={self._label!r} was declared; if the stamps "
                "are bar close times, declare label='close'.",
                "other": declared,
            },
            origin=origin,
            convention="utc_ns/open (canonical layout)"
            if canonical
            else f"{described}/{self._label}",
        )

    def stamp_range(self, symbol: str, exchange: str, interval: str) -> tuple[int, int]:
        """The first and last timestamp of the rows a request for `symbol`, `exchange`
        and `interval` reads, int64 UTC nanoseconds, with the declared `tz`, `unit`,
        `columns` and `timestamp_format` (the bar label is not applied): what a run
        with no `start=` or `end=` takes its dates from.

        Raises ConfigError for an undeclared time zone or columns that cannot be found,
        NoDataError when there are no rows for the symbol."""
        table, origin = self._read_table(symbol, exchange, interval)
        fields = _resolve_columns(table, self._columns)
        table = _select_rows(table, fields, {"symbol": symbol, "exchange": exchange}, origin)
        stamps_ns, _, _ = self._stamps(table, fields, origin, interval)
        if len(stamps_ns) == 0:
            raise NoDataError(f"{exchange}:{symbol} {interval}: {origin} has no rows")
        return int(stamps_ns.min()), int(stamps_ns.max())

    def _read_table(self, symbol: str, exchange: str, interval: str) -> tuple[pd.DataFrame, str]:
        """The table to read rows from and how to name it in messages."""
        if self._table is None:
            self._table = _to_table(self._data, self._columns)
        return self._table, "the supplied data"

    def _stamps(
        self, table: pd.DataFrame, fields: dict[str, ColumnSource], origin: str, interval: str
    ) -> tuple[np.ndarray, str, bool]:
        """(UTC nanosecond stamps, the convention applied, whether already canonical).
        Daily and longer `interval`s read naive midnight stamps as IST session dates."""
        source = fields["timestamp"]
        if source == _CANONICAL:
            raw = table["ts_open"]
            if raw.dtype != np.dtype("int64"):
                raise DataError(
                    f"{origin} has ts_open and ts_close columns but ts_open is {raw.dtype}, "
                    "not int64 UTC nanoseconds"
                )
            return raw.to_numpy(), "canonical UTC nanoseconds", True
        if source == _INDEX:
            values = table.index
        elif isinstance(source, tuple):
            values = table[source[0]].astype(str)
            for name in source[1:]:
                values = values + " " + table[name].astype(str)
        else:
            values = table[source]
        what = f"the timestamps of {origin}"
        stamps_ns, described = to_utc_ns(
            values,
            tz=self._tz,
            unit=self._unit,
            fmt=self._format,
            what=what,
            session_dates=not is_intraday(interval),
        )
        return stamps_ns, described, False


def _select_rows(
    table: pd.DataFrame, fields: dict[str, ColumnSource], wanted: dict[str, str], origin: str
) -> pd.DataFrame:
    """Rows of the requested symbol and exchange, when the table has such columns."""
    for name, value in wanted.items():
        column = fields.get(name)
        if column is None:
            continue
        held = table[column].astype(str)
        matches = (held == str(value)).to_numpy()
        if not matches.any():
            seen = sorted(set(held.unique().tolist()))[:_SHOWN_VALUES]
            raise NoDataError(
                f"{origin} has no rows with {name} {value!r} (column {column!r} holds {seen})"
            )
        if not matches.all():
            table = table.loc[matches]
    return table


__all__ = ["Frame"]

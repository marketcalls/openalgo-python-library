"""Timestamp normalisation into the canonical bars layout (PRD 06 section 3).

`build_bars` is the single path every source goes through, on every load. It takes
one stamp per row, already converted to a UTC instant, and produces `ts_open` and
`ts_close`:

- Intraday: the stamp is the instant the bar opens (or closes, with label="close");
  the series is checked against the calendar (see `_placement`). Pre-open,
  post-close and closed-day bars are dropped by the calendar's placement of each
  bar and counted; a late first bar is counted.
- D: the stamp is mapped to its IST calendar date; open and close are that
  session's, from the calendar.
- W, M, Q, Y (own data only; server data is resampled from D): the stamp is mapped
  to the period that contains its IST date; open is the open of the first session of
  the period and close the close of the last one.

Rows outside the requested dates are left out before anything is counted, bars that
have not closed at `now_ns` are dropped (the partial candle rule), and an
open-interest column that is zero on every returned row becomes NaN. Every count in
the `NormaliseReport` describes the bars of this call.
"""

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._calendar_convert import from_day, interval_nanos, to_day
from .._layout import BAR_COLUMNS, PRICE_COLUMNS, parse_interval
from .._messages import counted
from .._protocols import SessionCalendar
from ..errors import ConfigError, DataError
from . import _placement
from ._sessions import Sessions, ist_days

VALUE_COLUMNS = BAR_COLUMNS[2:]  # open, high, low, close, volume, oi: float64
LABELS = ("open", "close")
# How many example dates or instants a report or an error message carries.
MAX_EXAMPLES = 5

_OPTIONAL_COLUMNS = tuple(name for name in VALUE_COLUMNS if name not in PRICE_COLUMNS)
# The date sessions are built around when there are neither rows nor requested dates.
_NO_DATE = dt.date(1970, 1, 1)


@dataclass
class NormaliseReport:
    """What normalisation removed or noticed. Counts are rows unless a name says days.

    rows_in, rows_out: rows inside the requested dates, and rows in the bars.
    duplicates_removed, duplicates_conflicting: repeated timestamps dropped (the first
        row is kept) and how many of those carried different values.
    preopen_dropped, postclose_dropped, closed_day_dropped: rows outside the regular
        session that were dropped; `closed_day_examples` lists some of the dates.
    partial_dropped: bars that had not closed at the load time.
    incomplete_period_dropped: W, M, Q or Y bars that are not wholly inside the
        requested dates and the data they were built from.
    oi_zero_to_nan: rows whose open interest of 0 was turned into NaN.
    session_days, late_start_days: session days with bars, and those whose first bar
        opens after the session open; `late_start_examples` lists the first of them.
    shift_applied_seconds: a declared timestamp shift removed before the checks.
    notes: sentences for the warnings of the load.
    """

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    duplicates_conflicting: int = 0
    preopen_dropped: int = 0
    postclose_dropped: int = 0
    closed_day_dropped: int = 0
    closed_day_examples: list[str] = field(default_factory=list)
    partial_dropped: int = 0
    incomplete_period_dropped: int = 0
    oi_zero_to_nan: int = 0
    session_days: int = 0
    late_start_days: int = 0
    late_start_examples: list[str] = field(default_factory=list)
    shift_applied_seconds: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Stamped:
    """The rows of one series as a source delivers them, before normalisation.

    stamps_ns: UTC nanoseconds, one per row. values: the value columns (see
    `build_bars`). label: "open" or "close", what an intraday stamp refers to.
    canonical: the stamps are bar opens of the canonical layout already.
    regular_session_only: drop rows outside the regular session.
    shift_seconds: a declared timestamp shift already removed from the stamps.
    hints: source-specific advice appended to a timestamp error, keyed by the
    calendar's finding ("late_5h30", "early_5h30", "close_labelled") or "other".
    origin: how the source is named in messages. convention: the declared convention
    of the stamps.
    """

    stamps_ns: np.ndarray
    values: Mapping[str, object]
    label: str = "open"
    canonical: bool = False
    regular_session_only: bool = True
    shift_seconds: int = 0
    hints: Mapping[str, str] = field(default_factory=dict)
    origin: str = ""
    convention: str = ""


@dataclass
class _Rows:
    """Stamped rows placed in time. `first_day` and `last_day` are the IST day numbers
    of the bar open and close; `in_session` marks rows the calendar has a session for."""

    stamps: np.ndarray
    ts_open: np.ndarray
    ts_close: np.ndarray
    first_day: np.ndarray
    last_day: np.ndarray
    in_session: np.ndarray
    columns: dict[str, np.ndarray]

    def take(self, keep: np.ndarray) -> "_Rows":
        """The rows where `keep` is True."""
        columns = {name: column[keep] for name, column in self.columns.items()}
        placed = (self.stamps, self.ts_open, self.ts_close, self.first_day, self.last_day)
        return _Rows(*(array[keep] for array in placed), self.in_session[keep], columns)


@dataclass(frozen=True)
class _Series:
    """The series being normalised: its name for messages, its sessions, the hints
    for timestamp errors, the report to fill and the session filter."""

    tag: str
    exchange: str
    interval: str
    sessions: Sessions
    hints: Mapping[str, str]
    report: NormaliseReport
    regular_only: bool


def build_bars(
    stamps_ns: np.ndarray,
    values: Mapping[str, object],
    *,
    symbol: str,
    exchange: str,
    interval: str,
    calendar: SessionCalendar | Sessions | None,
    label: str = "open",
    now_ns: int | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    regular_session_only: bool = True,
    canonical_input: bool = False,
    hints: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, NormaliseReport]:
    """Turn stamped rows into canonical bars. Returns (bars, report).

    stamps_ns: UTC nanoseconds, one per row, in any order.
    values: "open", "high", "low", "close" and optionally "volume" and "oi"
        (array-likes of the same length). A missing optional column becomes NaN.
    calendar: a SessionCalendar, `Sessions` bound to the exchange, or None for the
        package default calendar.
    label: "open" or "close", the instant an intraday stamp refers to.
    now_ns: bars that close after this instant are dropped. None keeps all rows.
    start, end: inclusive IST dates; a bar is kept when it opens on or after `start`
        and closes on or before `end`. None leaves that side open.
    regular_session_only: drop pre-open, post-close and closed-day rows (counted).
    canonical_input: the stamps are session opens already (a frame in the canonical
        layout), so the stamp convention of daily rows is not checked.
    hints: source-specific advice for timestamp errors (see `Stamped`).

    Raises ConfigError for an unknown interval or label, DataError for missing or
    non-numeric columns and TimestampError when the rows do not fit the calendar.
    """
    kind, _ = parse_interval(interval)
    if label not in LABELS:
        raise ConfigError(f"label must be 'open' or 'close', got {label!r}")
    report = NormaliseReport()
    stamps_ns = np.array(stamps_ns, dtype="int64").reshape(-1)
    columns = _value_columns(values, len(stamps_ns), report)
    order = np.argsort(stamps_ns, kind="stable")
    stamps_ns, columns = stamps_ns[order], {name: column[order] for name, column in columns.items()}
    series = _Series(
        tag=f"{exchange}:{symbol} {interval}",
        exchange=exchange,
        interval=interval,
        sessions=_sessions(calendar, exchange, stamps_ns, start, end),
        hints=hints or {},
        report=report,
        regular_only=regular_session_only,
    )
    if kind == "intraday":
        rows = _place_intraday(stamps_ns, columns, label, series)
    else:
        rows = _place_calendar_rows(stamps_ns, columns, kind, canonical_input, series)
    rows = _dedupe(_within(rows, start, end, kind, report), report)
    report.rows_in = len(rows.ts_open) + report.duplicates_removed
    if kind == "intraday":
        keep = _checked_intraday(rows, label, series)
    else:
        keep = _checked_calendar_rows(rows, kind, series)
    if now_ns is not None:
        still_open = keep & (rows.ts_close > int(now_ns))
        report.partial_dropped = int(still_open.sum())
        keep = keep & ~still_open
    rows = rows.take(keep)
    rows.columns["oi"] = _unknown_open_interest(rows.columns["oi"], report)
    report.rows_out = len(rows.ts_open)
    attrs = {"symbol": symbol, "exchange": exchange, "interval": interval}
    return make_frame(rows.ts_open, rows.ts_close, rows.columns, attrs), report


def make_frame(
    ts_open: np.ndarray,
    ts_close: np.ndarray,
    values: Mapping[str, np.ndarray],
    attrs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Assemble canonical bars from arrays; the caller guarantees order and content.

    ts_open, ts_close: UTC nanoseconds. values: one float array per value column.
    """
    data = {
        "ts_open": np.ascontiguousarray(ts_open, dtype="int64"),
        "ts_close": np.ascontiguousarray(ts_close, dtype="int64"),
    }
    for name in VALUE_COLUMNS:
        data[name] = np.ascontiguousarray(values[name], dtype="float64")
    frame = pd.DataFrame(data, columns=list(BAR_COLUMNS))
    frame.attrs.update(attrs or {})
    return frame


def _sessions(
    calendar: SessionCalendar | Sessions | None,
    exchange: str,
    stamps_ns: np.ndarray,
    start: dt.date | None,
    end: dt.date | None,
) -> Sessions:
    """Sessions that reach over the requested dates and over every stamp (sorted)."""
    spans = [day for day in (start, end) if day is not None]
    if len(stamps_ns):
        spans += [from_day(day) for day in ist_days(stamps_ns[[0, -1]])]
    first, last = (min(spans), max(spans)) if spans else (_NO_DATE, _NO_DATE)
    if isinstance(calendar, Sessions):
        return calendar.covering(first, last)
    return Sessions(calendar, exchange, first, last)


def _value_columns(
    values: Mapping[str, object], length: int, report: NormaliseReport
) -> dict[str, np.ndarray]:
    columns = {}
    for name in PRICE_COLUMNS:
        if values.get(name) is None:
            raise DataError(f"missing price column {name!r}")
        columns[name] = _as_float(values[name], length, name)
    for name in _OPTIONAL_COLUMNS:
        if values.get(name) is None:
            columns[name] = np.full(length, np.nan)
            if name == "volume" and length:
                report.notes.append("the source has no volume column: volume is NaN")
        else:
            columns[name] = _as_float(values[name], length, name)
    return columns


def _as_float(data: object, length: int, name: str) -> np.ndarray:
    try:
        if isinstance(data, pd.Series):
            out = data.to_numpy(dtype="float64", na_value=np.nan)
        else:
            out = np.asarray(data, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise DataError(f"column {name!r} is not numeric: {exc}") from exc
    out = np.array(out, dtype="float64").reshape(-1)
    if len(out) != length:
        raise DataError(
            f"column {name!r} has {counted(len(out), 'value')} for {counted(length, 'timestamp')}"
        )
    return out


def _within(
    rows: _Rows, start: dt.date | None, end: dt.date | None, kind: str, report: NormaliseReport
) -> _Rows:
    """The rows whose bar lies inside the requested dates. A W, M, Q or Y bar that
    only overlaps them is counted: it would be a bar of part of its period."""
    low = to_day(start) if start is not None else np.iinfo("int64").min
    high = to_day(end) if end is not None else np.iinfo("int64").max
    inside = (rows.first_day >= low) & (rows.last_day <= high)
    if kind in _placement.PERIOD_FREQUENCY:
        overlaps = (rows.last_day >= low) & (rows.first_day <= high)
        report.incomplete_period_dropped = int((overlaps & ~inside).sum())
    return rows.take(inside)


def _dedupe(rows: _Rows, report: NormaliseReport) -> _Rows:
    """Keep the first row of each timestamp, as the SDK does, and count the rest."""
    repeated = np.zeros(len(rows.stamps), dtype=bool)
    repeated[1:] = rows.stamps[1:] == rows.stamps[:-1]
    if not repeated.any():
        return rows
    differs = np.zeros(len(repeated), dtype=bool)
    for column in rows.columns.values():
        same = (column[1:] == column[:-1]) | (np.isnan(column[1:]) & np.isnan(column[:-1]))
        differs[1:] |= ~same
    report.duplicates_removed = int(repeated.sum())
    report.duplicates_conflicting = int((repeated & differs).sum())
    return rows.take(~repeated)


def _unknown_open_interest(oi: np.ndarray, report: NormaliseReport) -> np.ndarray:
    """Open interest that is 0 on every row means the source supplies none: NaN.

    A series with any non-zero value keeps its zeros, which are then real (a newly
    listed contract has no open interest yet).
    """
    known = np.isfinite(oi)
    if known.any() and not np.any(oi[known] != 0.0):
        report.oi_zero_to_nan = int(known.sum())
        return np.full(len(oi), np.nan)
    return oi


# --- intraday (the placing and the checks are in `_placement`) -----------------------


def _place_intraday(stamps_ns: np.ndarray, columns: dict, label: str, series: _Series) -> _Rows:
    """Rows at their bar opens. `ts_close` and `in_session` are set by the check."""
    step_ns = interval_nanos(series.interval)
    ts_open = _placement.place_intraday(stamps_ns, step_ns, label, series.sessions)
    days = ist_days(ts_open)
    unknown = np.zeros(len(ts_open), dtype=bool)
    return _Rows(stamps_ns, ts_open, ts_open.copy(), days, days, unknown, columns)


def _checked_intraday(rows: _Rows, label: str, series: _Series) -> np.ndarray:
    """Validate the series against the calendar, set each bar's close from the
    calendar's placement and return which rows are kept."""
    report, sessions, step_ns = series.report, series.sessions, interval_nanos(series.interval)
    if len(rows.ts_open) == 0:
        return np.ones(0, dtype=bool)
    options = {"tag": series.tag, "interval": series.interval, "hints": dict(series.hints)}
    counts = _placement.check_intraday(rows.ts_open, step_ns, label, sessions, **options)
    rows.in_session = counts["placement"] == _placement.IN_SESSION
    rows.ts_close = _placement.bar_closes(rows.ts_open, rows.in_session, step_ns, sessions)
    if counts["offset_check"] == "inconclusive":
        report.notes.append(
            f"the calendar's offset check was inconclusive: {counts['offset_reason']}"
        )
    report.session_days = counts["session_days"]
    report.late_start_days = counts["days_starting_late"]
    if counts["first_late_day"] is not None:
        report.late_start_examples = [str(counts["first_late_day"])]
    if not series.regular_only:
        outside = int((~rows.in_session).sum())
        if outside:
            kept = counted(
                outside,
                "bar outside the regular session was kept",
                "bars outside the regular session were kept",
            )
            report.notes.append(f"{kept} (regular_session_only=False)")
        return np.ones(len(rows.ts_open), dtype=bool)
    report.preopen_dropped = counts["pre_open_bars"]
    report.postclose_dropped = counts["post_close_bars"]
    report.closed_day_dropped = counts["closed_day_bars"]
    if counts["first_closed_day"] is not None:
        report.closed_day_examples = [str(counts["first_closed_day"])]
    return rows.in_session.copy()


# --- daily and longer (the placing and the checks are in `_placement`) ---------------


def _place_calendar_rows(
    stamps_ns: np.ndarray, columns: dict, unit: str, canonical_input: bool, series: _Series
) -> _Rows:
    notes = series.report.notes
    placed = _placement.place_rows(
        stamps_ns, unit, canonical_input, series.sessions, series.tag, notes
    )
    return _Rows(stamps_ns, *placed, columns)


def _checked_calendar_rows(rows: _Rows, unit: str, series: _Series) -> np.ndarray:
    """Validate daily or longer rows against the calendar. Returns which are kept."""
    options = {"tag": series.tag, "regular_only": series.regular_only}
    closed = _placement.check_rows(
        rows.first_day, rows.in_session, unit, series.sessions, **options
    )
    if closed.any():
        series.report.closed_day_dropped = int(closed.sum())
        days = rows.first_day[closed][:MAX_EXAMPLES]
        series.report.closed_day_examples = [str(from_day(day)) for day in days]
    return ~closed


__all__ = ["LABELS", "MAX_EXAMPLES", "NormaliseReport", "Stamped", "VALUE_COLUMNS", "build_bars"]

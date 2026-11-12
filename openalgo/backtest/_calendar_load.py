"""Loading holiday and session data into a native calendar.

Two shapes are understood: the built-in year documents of `calendar_data` and the
response of OpenAlgo `holidays(year)`. Each loader writes dated rules and coverage
marks into the native calendar and records names, sources, notes and problems in a
`Journal`.

A `holidays(year)` response is a dict `{"status", "year", "data": [...]}`, or its
`data` list. Each row has `date`, `description`, `holiday_type`, `closed_exchanges` and
`open_exchanges` (`exchange`, `start_time`, `end_time` in epoch milliseconds). As on
the server: on a TRADING_HOLIDAY or a SPECIAL_SESSION row the closed exchanges close,
the open exchanges trade in the given window and other exchanges keep their regular
rule; a SETTLEMENT_HOLIDAY row changes nothing. The server stores 23:55 as the MCX
close on every date although MCX closes at 23:30 while the United States is on
daylight saving time: with `mcx_close="regular"` (default) such a window closes at the
regular MCX close of its date, with "server" it is used as given.

The rule for data that is not confirmed: rows loaded as unverified (the default for an
OpenAlgo server, whose holiday tables are kept by hand) never replace data already in
the calendar. On a date whose coverage is "unverified" or "sourced" (a built-in list,
or data loaded before) the existing answer is kept, and a row that disagrees with it is
recorded in `issues`. Rows loaded as verified replace what is there. A load that had
to skip rows (`strict=False`) never marks a year "sourced".
"""

import datetime as dt
from dataclasses import dataclass, field

from ._calendar_convert import (
    COVERAGE_LEVELS,
    HOLIDAY_EXCHANGES,
    SECONDS_PER_DAY,
    call_native,
    check_exchange,
    coverage_code,
    exchange_list,
    format_ist,
    to_day,
    to_seconds,
)
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from .errors import CalendarError, ConfigError

_NANOS_PER_MILLISECOND = NANOS_PER_SECOND // 1000
_DOCUMENT_FORMAT = 1
_TEMPLATE = coverage_code("template")
_UNVERIFIED = coverage_code("unverified")
_SOURCED = coverage_code("sourced")
_MCX_CLOSE_MODES = ("regular", "server")
# The close the OpenAlgo server stores for MCX on every date, 23:55 IST in seconds
# since midnight (database/market_calendar_db.py, DEFAULT_MARKET_TIMINGS and the
# evening windows of its holiday seed, commit d8cc13f24).
_SERVER_MCX_CLOSE_SEC = 23 * 3600 + 55 * 60
_HOLIDAY_TYPES = ("TRADING_HOLIDAY", "SPECIAL_SESSION", "SETTLEMENT_HOLIDAY")


@dataclass
class Journal:
    """What a calendar remembers about the data loaded into it.

    names: holiday or session names by (exchange, IST date).
    sources: citations by year, {source id: citation}.
    notes: caveats by year.
    issues: problems met while loading.
    """

    names: dict[tuple[str, dt.date], str] = field(default_factory=dict)
    sources: dict[int, dict[str, str]] = field(default_factory=dict)
    notes: dict[int, list[str]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def copy(self) -> "Journal":
        """An independent copy, so a failed load can be thrown away."""
        return Journal(
            names=dict(self.names),
            sources={year: dict(items) for year, items in self.sources.items()},
            notes={year: list(items) for year, items in self.notes.items()},
            issues=list(self.issues),
        )


def level_of(verified: bool) -> int:
    """The native coverage code of a rule: sourced when verified, else unverified."""
    return _SOURCED if verified else _UNVERIFIED


# ---------------------------------------------------------------------------
# Built-in year documents
# ---------------------------------------------------------------------------


def load_document(cal, journal: Journal, document: dict) -> None:
    """Load one year in the layout described in `calendar_data`.

    Raises CalendarError for another layout or an entry dated outside the year.
    """
    if not isinstance(document, dict) or document.get("format") != _DOCUMENT_FORMAT:
        raise CalendarError(
            f'unsupported calendar document: expected a dict with "format": '
            f"{_DOCUMENT_FORMAT} in the layout described in openalgo.backtest.calendar_data"
        )
    year = int(document["year"])
    for entry in document.get("days", ()):
        date = dt.date.fromisoformat(entry["date"])
        if date.year != year:
            raise CalendarError(
                f"calendar document for {year} holds an entry dated {entry['date']}"
            )
        _load_document_day(cal, entry, to_day(date))
        for exchange in (*entry.get("closed", ()), *entry.get("sessions", {})):
            if entry.get("name"):
                journal.names[(exchange, date)] = entry["name"]
    for exchange, level in document.get("coverage", {}).items():
        cal.set_year_coverage(check_exchange(exchange), year, coverage_code(level))
    journal.sources.setdefault(year, {}).update(document.get("sources", {}))
    journal.notes.setdefault(year, []).extend(document.get("notes", ()))


def _load_document_day(cal, entry: dict, day: int) -> None:
    unverified = set(entry.get("unverified", ()))
    unverified_hours = set(entry.get("hours_unverified", ()))
    for exchange in entry.get("closed", ()):
        level = level_of(exchange not in unverified)
        call_native(cal.set_closed, check_exchange(exchange), day, level)
    for exchange, (open_, close) in entry.get("sessions", {}).items():
        check_exchange(exchange)
        open_sec = to_seconds(open_, what="session open")
        # A close of None is the regular close of the date, resolved when asked.
        close_sec = None if close is None else to_seconds(close, what="session close")
        flags = (exchange in unverified_hours, level_of(exchange not in unverified))
        call_native(cal.set_session, exchange, day, open_sec, close_sec, *flags)


# ---------------------------------------------------------------------------
# OpenAlgo holidays(year)
# ---------------------------------------------------------------------------


@dataclass
class _Load:
    """One holidays(year) load: its settings and what it did."""

    cal: object
    source: str
    level: int
    targets: list[str]
    mcx_close: str
    summary: dict


def load_holidays(
    cal,
    journal: Journal,
    response,
    *,
    year: int | None,
    source: str,
    verified: bool,
    exchanges,
    strict: bool,
    mcx_close: str,
) -> dict:
    """Load an OpenAlgo `holidays(year)` response. See `Calendar.add_holidays`."""
    if mcx_close not in _MCX_CLOSE_MODES:
        raise ConfigError(f"mcx_close must be one of {_MCX_CLOSE_MODES}, got {mcx_close!r}")
    rows, year = _holiday_rows(response, year)
    targets = list(HOLIDAY_EXCHANGES) if exchanges is None else exchange_list(exchanges)
    summary = {"year": year, "rows": len(rows), "closures": 0, "sessions": 0, "kept": 0}
    summary.update(coverage_marked=False, issues=[])
    if not rows:
        summary["issues"].append(
            f"{source}: no holiday rows for {year}; the year stays as it was (template-only "
            "for exchanges without other data)"
        )
        journal.issues.extend(summary["issues"])
        return summary

    load = _Load(cal, source, level_of(verified), targets, mcx_close, summary)
    skipped = False
    for row in rows:
        date = _holiday_row_date(row, year, source)
        if _holiday_row_type(row, date, source) == "SETTLEMENT_HOLIDAY":
            continue  # trading runs on its regular hours
        problems = _load_holiday_row(load, row, date)
        if problems and strict:
            raise CalendarError(
                f"{source}: {problems[0]} Correct the entry on the OpenAlgo Holidays page, "
                "or load with strict=False to skip it and keep the regular rule for that date."
            )
        skipped = skipped or bool(problems)
        summary["issues"].extend(f"{source}: {problem} Skipped." for problem in problems)
        for exchange in _row_exchanges(row, targets):
            if row.get("description"):
                journal.names.setdefault((exchange, date), str(row["description"]))

    marked = _UNVERIFIED if skipped else load.level
    for exchange in targets:
        existing = cal.year_coverage(exchange, year)
        cal.set_year_coverage(exchange, year, max(existing, marked))
    summary["coverage_marked"] = True
    journal.sources.setdefault(year, {})[source] = (
        "holidays supplied in the OpenAlgo holidays(year) shape"
        + ("" if marked == _SOURCED else " (unverified)")
    )
    journal.issues.extend(summary["issues"])
    return summary


def _row_exchanges(row: dict, targets: list[str]) -> list[str]:
    closed = list(row.get("closed_exchanges") or ())
    opened = [window.get("exchange") for window in row.get("open_exchanges") or ()]
    return [exchange for exchange in (*closed, *opened) if exchange in targets]


def _protected(load: _Load, exchange: str, day: int) -> str | None:
    """The coverage of loaded data an unverified row must not replace, else None."""
    if load.level == _SOURCED:
        return None
    existing = call_native(load.cal.coverage, exchange, day)
    return None if existing == _TEMPLATE else COVERAGE_LEVELS[existing]


def _keep(load: _Load, exchange: str, date: dt.date, kept: str, disagrees: bool, what: str):
    load.summary["kept"] += 1
    if disagrees:
        load.summary["issues"].append(
            f"{load.source}: the row {what} {exchange} on {date}, but the calendar already "
            f"holds {kept} data that differs; the existing data was kept (load with "
            "verified=True to replace it)."
        )


def _load_holiday_row(load: _Load, row: dict, date: dt.date) -> list[str]:
    """Apply one holiday row, counting into the summary. Returns the windows it could not use."""
    day = to_day(date)
    problems = []
    for exchange in row.get("closed_exchanges") or ():
        if exchange not in load.targets:
            continue
        kept = _protected(load, exchange, day)
        if kept is not None:
            disagrees = call_native(load.cal.session, exchange, day) is not None
            _keep(load, exchange, date, kept, disagrees, "closes")
            continue
        call_native(load.cal.set_closed, exchange, day, load.level)
        load.summary["closures"] += 1
    for window in row.get("open_exchanges") or ():
        if window.get("exchange") in load.targets:
            problem = _load_window(load, window, date)
            if problem is not None:
                problems.append(problem)
    return problems


def _window_seconds(window: dict, date: dt.date) -> tuple[int, int] | str:
    """(open_sec, close_sec) of an epoch-millisecond window, or the problem with it."""
    exchange = window.get("exchange")
    try:
        start_ns = int(window.get("start_time")) * _NANOS_PER_MILLISECOND
        end_ns = int(window.get("end_time")) * _NANOS_PER_MILLISECOND
    except (TypeError, ValueError):
        return f"{exchange} on {date} is marked open but has no start_time and end_time."
    midnight_ns = to_day(date) * SECONDS_PER_DAY * NANOS_PER_SECOND - IST_OFFSET_NANOS
    open_sec, open_rest = divmod(start_ns - midnight_ns, NANOS_PER_SECOND)
    close_sec, close_rest = divmod(end_ns - midnight_ns, NANOS_PER_SECOND)
    if open_rest or close_rest or not 0 <= open_sec < close_sec <= SECONDS_PER_DAY:
        return (
            f"the {exchange} window on {date} runs from {format_ist(start_ns)} to "
            f"{format_ist(end_ns)}, which is not a session inside that IST date."
        )
    return open_sec, close_sec


def _load_window(load: _Load, window: dict, date: dt.date) -> str | None:
    """Set a session from an epoch-millisecond window. Returns a problem, or None."""
    seconds = _window_seconds(window, date)
    if isinstance(seconds, str):
        return seconds
    open_sec, close_sec = seconds
    exchange, day = window["exchange"], to_day(date)
    if exchange == "MCX" and load.mcx_close == "regular" and close_sec == _SERVER_MCX_CLOSE_SEC:
        close_sec = None  # the regular MCX close of the date, which depends on the season
    kept = _protected(load, exchange, day)
    if kept is not None:
        existing = call_native(load.cal.session, exchange, day)
        midnight_ns = day * SECONDS_PER_DAY * NANOS_PER_SECOND - IST_OFFSET_NANOS
        same_open = (
            existing is not None and existing[1] == midnight_ns + open_sec * NANOS_PER_SECOND
        )
        _keep(load, exchange, date, kept, not same_open, "opens")
        return None
    call_native(load.cal.set_session, exchange, day, open_sec, close_sec, False, load.level)
    load.summary["sessions"] += 1
    return None


def _response_rows(response, name: str) -> list[dict]:
    """The `data` list of an OpenAlgo response dict, or the list itself."""
    if not isinstance(response, dict):
        return list(response or [])
    if response.get("status") == "error":
        raise CalendarError(
            f"the {name} response is an error: {response.get('message', 'no message')}. "
            "Fix the request (the server serves 2020 to 2050) or load the data from "
            "another source."
        )
    if "data" not in response:
        raise CalendarError(
            f"the {name} response has no 'data' list: pass the dict returned by the "
            f"client's {name} call, or its data list"
        )
    return list(response["data"] or [])


def _holiday_rows(response, year: int | None) -> tuple[list[dict], int]:
    rows = _response_rows(response, "holidays")
    if year is None and isinstance(response, dict):
        year = response.get("year")
    if year is None:
        years = {str(row.get("date", ""))[:4] for row in rows}
        if len(years) != 1 or not next(iter(years)).isdigit():
            raise CalendarError(
                "cannot tell which year these holidays belong to: pass year=, or the "
                "whole holidays(year) response"
            )
        year = next(iter(years))
    return rows, int(year)


def _holiday_row_date(row: dict, year: int, source: str) -> dt.date:
    try:
        date = dt.date.fromisoformat(str(row["date"])[:10])
    except (KeyError, ValueError):
        raise CalendarError(
            f"{source}: a holiday row has no usable 'date' (YYYY-MM-DD): {row!r}"
        ) from None
    if date.year != year:
        raise CalendarError(
            f"{source}: a row dated {date} was supplied for year {year}; load each year "
            "from its own holidays(year) response"
        )
    return date


def _holiday_row_type(row: dict, date: dt.date, source: str) -> str:
    kind = row.get("holiday_type", "TRADING_HOLIDAY")
    if kind not in _HOLIDAY_TYPES:
        raise CalendarError(
            f"{source}: unknown holiday_type {kind!r} on {date}; expected one of "
            f"{', '.join(_HOLIDAY_TYPES)}"
        )
    return kind


__all__ = ["Journal", "level_of", "load_document", "load_holidays"]

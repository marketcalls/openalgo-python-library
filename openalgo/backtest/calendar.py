"""Session calendars for Indian exchanges.

A `Calendar` answers, for an OpenAlgo exchange and an IST date: is the exchange open,
when does its session open and close, and how well is that answer backed by data.

    import datetime as dt
    from openalgo.backtest.calendar import default_calendar

    cal = default_calendar()
    cal.session("NSE", dt.date(2026, 11, 12))          # (open_ns, close_ns) or None
    cal.sessions("MCX", "2026-01-01", "2026-03-31")    # DataFrame, one row per session
    cal.union_grid(["NSE", "MCX"], "2026-03-02", "2026-03-06", base="1m")

How a session is decided, first match wins:

1. A dated rule: a trading holiday, a special session (Muhurat, a weekend session, an
   evening-only session on a holiday) or a per-date override. Rules come from the
   built-in files in `calendar_data`, from an OpenAlgo `holidays(year)` response, or
   from `add_session` and `add_closure`.
2. Saturday and Sunday are closed.
3. The regular hours in force on that date, from the kernel's dated and sourced table
   (`rust/oa_bt/src/market/calendar/regular_hours.csv`): for example NFO closes at
   15:30, at 15:40 from 2026-08-03 and at 15:45 from 2026-12-07.

Bars of interval N open at IST midnight plus the exchange's regular open plus whole
multiples of N on every date, special sessions included, as on the OpenAlgo server.

Nothing is assumed silently: a year with no holiday list is "template" (weekday rule
only), data not confirmed by an exchange document is "unverified", and a session whose
times are not confirmed has `hours_unverified`.

Units: instants are int64 UTC nanoseconds, dates are IST dates (`datetime.date` or ISO
string) from 1677 to 2262, clock times IST "HH:MM". `NSE_INDEX` and `BSE_INDEX` follow
NSE and BSE. Unknown exchanges and unsupported dates raise `CalendarError`, unreadable
dates and times `ConfigError`, unusable timestamps `TimestampError`.
"""

import datetime as dt
import functools
import warnings
from typing import Self

import numpy as np
import pandas as pd

from . import _calendar_load as load
from . import _calendar_series as series
from ._calendar_convert import (
    COVERAGE_LEVELS,
    EXCHANGES,
    HOLIDAY_EXCHANGES,
    DateLike,
    calendar_owner,
    call_native,
    check_exchange,
    exchange_list,
    from_day,
    interval_nanos,
    native_attr,
    to_day,
    to_seconds,
)
from ._calendar_edit import copy_sessions
from ._calendar_types import (
    SESSION_KINDS,
    CalendarWarning,
    IntradayTimes,
    SessionInfo,
    SessionsPerYear,
    grid_mask,
)
from ._protocols import SessionCalendar
from .calendar_data import builtin_documents
from .errors import CalendarError

Exchanges = str | list[str] | tuple[str, ...]
_READ_ONLY = "this calendar is shared and read-only: change a .copy() or a new Calendar()"


class Calendar:
    """Session calendar for the OpenAlgo exchanges. Implements `SessionCalendar`.

    `Calendar()` loads the built-in holiday files. `Calendar(builtin=False)` starts
    from the regular hours alone (every year is then template-only). Every loader
    changes nothing when it raises.
    """

    def __init__(self, *, builtin: bool = True) -> None:
        self._cal = call_native(native_attr("Calendar"))
        self._journal = load.Journal()
        if builtin:
            for document in builtin_documents():
                load.load_document(self._cal, self._journal, document)

    # ----- construction ----------------------------------------------------

    @classmethod
    def from_openalgo(
        cls,
        client,
        years,
        *,
        builtin: bool = True,
        strict: bool = True,
        verified: bool = False,
        mcx_close: str = "regular",
    ) -> Self:
        """Build a calendar from an OpenAlgo server's holiday tables.

        Calls `client.holidays(year)` once per year and loads each response with
        `add_holidays`. Server rows are "unverified" by default, so they fill years
        the built-in files do not cover and never replace built-in data; pass
        `verified=True` only for a server whose tables you have checked.
        """
        calendar = cls(builtin=builtin)
        options = {"verified": verified, "strict": strict, "mcx_close": mcx_close}
        for year in map(int, years):
            source = f"OpenAlgo holidays({year})"
            calendar.add_holidays(client.holidays(year), year=year, source=source, **options)
        return calendar

    @classmethod
    def from_sessions(
        cls, calendar: SessionCalendar, exchange: str, start: DateLike, end: DateLike
    ) -> Self:
        """Copy the sessions and coverage of one exchange from any `SessionCalendar`.

        Every IST date from `start` to `end` takes the session the other calendar
        returns (or a closure) and its `coverage` of that date; a calendar without
        `coverage` has its dates marked "unverified". This is how the kernel runs
        against a calendar that is not a `Calendar`. Other dates and exchanges keep
        the regular hours. Raises CalendarError for a session outside its IST date.
        """
        copied = cls(builtin=False)
        copy_sessions(copied._cal, calendar, check_exchange(exchange), to_day(start), to_day(end))
        return copied

    def copy(self) -> Self:
        """A private, editable copy of this calendar."""
        clone = type(self).__new__(type(self))
        clone._cal = self._cal.copy()
        clone._journal = self._journal.copy()
        return clone

    def _freeze(self) -> Self:
        """Make the calendar read-only, in the kernel too (used for the shared one)."""
        self._cal.freeze()
        return self

    def _editable(self) -> None:
        if self._cal.frozen:
            raise CalendarError(_READ_ONLY)

    # ----- editing ---------------------------------------------------------

    def add_holidays(
        self,
        response,
        *,
        year: int | None = None,
        source: str = "user",
        verified: bool = False,
        exchanges: Exchanges | None = None,
        strict: bool = True,
        mcx_close: str = "regular",
    ) -> dict:
        """Load holidays and special sessions in the shape OpenAlgo `holidays(year)` returns.

        `response` is the response dict or its `data` list; the row shape, the rule for
        unverified rows and `mcx_close` are described in `_calendar_load`. With
        `verified=False` (default) the rows never replace data the calendar holds, and
        each disagreement is listed in `issues`; with `verified=True` they replace it
        and count as "sourced". The `exchanges` (default: all with their own holiday
        list) are marked covered for the year at that level; a year is never lowered.

        A window outside its row's IST date raises CalendarError; with `strict=False`
        it is skipped, listed in `issues`, and the year is marked "unverified" at most.
        Returns a summary dict: year, rows, closures, sessions, kept, coverage_marked,
        issues.
        """
        options = {"year": year, "source": source, "verified": verified, "strict": strict}
        options.update(exchanges=exchanges, mcx_close=mcx_close)
        self._editable()
        cal, journal = self._cal.copy(), self._journal.copy()
        summary = load.load_holidays(cal, journal, response, **options)
        self._cal, self._journal = cal, journal
        return summary

    def add_session(
        self,
        exchange: str,
        day: DateLike,
        open: str,
        close: str | None = None,
        *,
        name: str | None = None,
        verified: bool = True,
        hours_unverified: bool = False,
    ) -> None:
        """Give an exchange a session on a date with IST times ("HH:MM").

            cal.add_session("NSE", "2020-02-01", "09:15", "15:30", name="Union Budget")

        A `close` of None closes at the exchange's regular close for that date (an
        evening session on a holiday), whatever the regular hours are when asked.
        Raises CalendarError unless open is before close and both lie on the date.
        """
        self._editable()
        exchange, number = check_exchange(exchange), to_day(day)
        open_sec = to_seconds(open, what="session open")
        close_sec = None if close is None else to_seconds(close, what="session close")
        flags = (bool(hours_unverified), load.level_of(verified))
        call_native(self._cal.set_session, exchange, number, open_sec, close_sec, *flags)
        self._name([exchange], number, name)

    def add_closure(
        self, exchanges: Exchanges, day: DateLike, *, name: str | None = None, verified: bool = True
    ) -> None:
        """Mark one exchange, or several, closed on a date."""
        self._editable()
        codes, number = exchange_list(exchanges), to_day(day)
        for exchange in codes:
            call_native(self._cal.set_closed, exchange, number, load.level_of(verified))
        self._name(codes, number, name)

    def _name(self, exchanges: list[str], day: int, name: str | None) -> None:
        for exchange in exchanges:
            if name:
                self._journal.names[(calendar_owner(exchange), from_day(day))] = str(name)

    def set_regular_hours(
        self, exchange: str, open: str, close: str, *, effective_from: DateLike | None = None
    ) -> None:
        """Set the regular session times of an exchange, as IST "HH:MM".

        Without `effective_from` the hours replace the built-in regular hours on every
        date; with it they apply from that IST date onward. Hours set here count as
        verified. Evening sessions that close at the regular close follow them.
        """
        self._editable()
        start = None if effective_from is None else to_day(effective_from)
        open_sec = to_seconds(open, what="session open")
        close_sec = to_seconds(close, what="session close")
        call_native(
            self._cal.set_regular_hours, check_exchange(exchange), start, open_sec, close_sec
        )

    # ----- sessions --------------------------------------------------------

    def session(self, exchange: str, day: DateLike) -> tuple[int, int] | None:
        """(open, close) of the session as UTC nanoseconds, or None when closed."""
        info = self.session_info(exchange, day)
        return None if info is None else (info.open, info.close)

    def session_info(self, exchange: str, day: DateLike) -> SessionInfo | None:
        """The session with its kind, coverage and `hours_unverified`, or None when closed."""
        exchange = check_exchange(exchange)
        row = call_native(self._cal.session, exchange, to_day(day))
        if row is None:
            return None
        number, open_, close, kind, coverage, unverified = row
        kind, coverage = SESSION_KINDS[kind], COVERAGE_LEVELS[coverage]
        return SessionInfo(exchange, from_day(number), open_, close, kind, coverage, unverified)

    def is_open(self, exchange: str, day: DateLike) -> bool:
        """Whether the exchange has a session on the IST date."""
        return self.session_info(exchange, day) is not None

    def sessions(self, exchange: str, start: DateLike, end: DateLike) -> pd.DataFrame:
        """Every session from `start` to `end` (IST dates, inclusive), one row each.

        Columns: date (datetime.date), open, close (int64 UTC nanoseconds), kind,
        coverage (strings) and hours_unverified (bool). Raises CalendarError when
        `start` is after `end`.
        """
        exchange = check_exchange(exchange)
        columns = call_native(self._cal.sessions, exchange, to_day(start), to_day(end))
        return series.sessions_frame(columns, exchange)

    def sessions_per_year(
        self, exchanges: Exchanges, start: DateLike, end: DateLike
    ) -> SessionsPerYear:
        """Average sessions per year from `start` to `end`: the annualisation factor.

        With several exchanges the union calendar is used (dates on which any of them
        trades), scaled by 365.25 days per year. The window must span at least 28
        calendar days, so weekdays and weekends count in proportion (CalendarError
        otherwise). Warns with CalendarWarning when a year of the window has no
        holiday list: the value is then overstated by that year's holidays.
        """
        args = (exchange_list(exchanges), to_day(start), to_day(end))
        result = series.annualisation(call_native(self._cal.sessions_per_year, *args))
        if result.template_only:
            years = ", ".join(f"{exchange} {year}" for exchange, year in result.template_only)
            message = f"sessions_per_year counts every weekday of {years} as a session: no "
            message += "holiday list is loaded for it, so the value is overstated"
            warnings.warn(message, CalendarWarning, stacklevel=2)
        return result

    def intraday_times(self, exchange: str, day: DateLike) -> IntradayTimes | None:
        """Default `entry_until` and `square_off` of an exchange on a date, or None
        when it is closed. The values are dated rows of the kernel's table
        `intraday_defaults.csv`; MCX and special sessions take them before that day's
        close (`relative_to_close`)."""
        row = call_native(self._cal.intraday_times, check_exchange(exchange), to_day(day))
        return None if row is None else IntradayTimes(*row)

    # ----- coverage and provenance -----------------------------------------

    def coverage(self, exchange: str, when: int | DateLike) -> str:
        """How well a date, or a year given as an int, is backed by holiday data.

        "template": no holiday list is loaded, weekday rule only. "unverified": data is
        loaded but not confirmed. "sourced": data from a cited exchange or regulator
        document, or supplied by the user as verified.
        """
        exchange = check_exchange(exchange)
        if isinstance(when, int | np.integer) and not isinstance(when, bool):
            return COVERAGE_LEVELS[self._cal.year_coverage(exchange, int(when))]
        return COVERAGE_LEVELS[call_native(self._cal.coverage, exchange, to_day(when))]

    def holidays(self, exchange: str, year: int) -> pd.DataFrame:
        """The dated rules of an exchange in a year, one row each.

        Columns: date, name, open, close (IST "HH:MM"; both None when closed, close
        None for the regular close), coverage and hours_unverified.
        """
        exchange = check_exchange(exchange)
        first, last = to_day(dt.date(int(year), 1, 1)), to_day(dt.date(int(year), 12, 31))
        rules = call_native(self._cal.rules, exchange, first, last)
        owner = calendar_owner(exchange)
        names = {date: name for (code, date), name in self._journal.names.items() if code == owner}
        return series.holidays_frame(rules, names)

    def sources(self, year: int) -> dict[str, str]:
        """Citations of the data loaded for a year: {source id: citation}."""
        return dict(self._journal.sources.get(int(year), {}))

    def notes(self, year: int) -> list[str]:
        """Caveats that came with the data loaded for a year."""
        return list(self._journal.notes.get(int(year), ()))

    @property
    def issues(self) -> list[str]:
        """Problems met while loading: skipped rows, kept data, years with no rows."""
        return list(self._journal.issues)

    # ----- grid and series -------------------------------------------------

    def union_grid(
        self, exchanges: Exchanges, start: DateLike, end: DateLike, base: str = "1m"
    ) -> pd.DataFrame:
        """The union-calendar grid of base-interval slices the kernel iterates.

        One row per instant at which any of the exchanges has a base bar opening, from
        `start` to `end` (IST dates, inclusive); with `base="D"` one per session. The
        columns, and why decisions must follow the close columns rather than row
        order, are described in `_calendar_series`. Raises CalendarError for a reversed
        range or a grid above 50 million slices, ConfigError for a base that is neither
        intraday nor "D".
        """
        codes = list(dict.fromkeys(exchange_list(exchanges)))
        args = (codes, to_day(start), to_day(end), interval_nanos(base))
        return series.grid_frame(call_native(self._cal.union_grid, *args), codes, base)

    def check_intraday(
        self, bars, *, exchange: str | None = None, interval: str | None = None
    ) -> dict:
        """Count how an intraday series sits against the calendar. Never judges data.

        `bars` is a canonical bars frame or an array of `ts_open` (int64 UTC
        nanoseconds), sorted ascending; `exchange` and `interval` default to
        `bars.attrs`. The keys of the dict are listed in `_calendar_series`. Raises
        TimestampError for unsorted input or a missing timestamp (NaT).
        """
        exchange, ts_open, nanos = series.series_arguments(bars, exchange, interval)
        return series.check_report(call_native(self._cal.check_intraday, exchange, ts_open, nanos))

    def validate_intraday(
        self,
        bars,
        *,
        exchange: str | None = None,
        interval: str | None = None,
        allow_closed_days: bool = False,
    ) -> dict:
        """Check an intraday series and raise TimestampError on wrong timestamps.

        Errors, in this order: timestamps 5 hours 30 minutes off, bars stamped at
        their close, bars in a session that do not open at the regular open plus
        whole intervals, and bars on a date with no session (unless
        `allow_closed_days`). Each message names the likely cause. Pre-open,
        post-close and missing bars are counts in the returned dict (the one
        `check_intraday` returns), not errors. When a 5 hour 30 minute shift fits
        better but not decisively, a CalendarWarning gives the reason.
        """
        exchange, ts_open, nanos = series.series_arguments(bars, exchange, interval)
        args = (exchange, ts_open, nanos, bool(allow_closed_days))
        report = series.check_report(call_native(self._cal.validate_intraday, *args))
        if report["offset_check"] == "inconclusive":
            warnings.warn(f"{exchange}: {report['offset_reason']}", CalendarWarning, stacklevel=2)
        return report


@functools.cache
def default_calendar() -> Calendar:
    """The shared calendar with the built-in holiday data. It is read-only.

    Use `default_calendar().copy()` or `Calendar()` for one you can add to.
    """
    return Calendar(builtin=True)._freeze()


#: The `close_<EXCH>` value of a grid row where that exchange has no bar.
NO_BAR: int = int(native_attr("NO_BAR"))

__all__ = [
    "COVERAGE_LEVELS",
    "EXCHANGES",
    "HOLIDAY_EXCHANGES",
    "NO_BAR",
    "SESSION_KINDS",
    "Calendar",
    "CalendarWarning",
    "IntradayTimes",
    "SessionInfo",
    "SessionsPerYear",
    "default_calendar",
    "grid_mask",
]

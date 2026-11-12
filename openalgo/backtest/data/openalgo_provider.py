"""The OpenAlgo data provider: broker history, Historify and the "auto" choice (FR-DATA-01, 02).

`source="api"` asks the broker through the OpenAlgo server, `source="db"` reads
Historify, and `source="auto"` probes Historify and uses it for the whole range if
it is complete, otherwise the broker for the whole range. The two are never
stitched together, because their bars are aggregated differently.

The probe behind "auto" (PRD 06 section 1, PRD 14 section 6). An API key cannot
query Historify's catalog, so coverage is decided with history requests. Historify
is complete for a range when it returns a daily bar for every session of the range
that has closed and, for an intraday interval, bars inside the session of the first
and of the last of those days. The probe runs on every load: a cached series never
decides the source. It cannot see intraday gaps between those two days; a load that
uses Historify for intraday bars says so in its warnings.

The provider only ever calls the read-only `history` and `ping` endpoints. It hands
raw rows to `loader`, which normalises them (see `normalise`).
"""

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._calendar_convert import DateLike, to_day
from .._layout import NANOS_PER_SECOND, is_intraday
from .._messages import counted
from .._protocols import SessionCalendar
from ..errors import ConfigError, DataError
from ._sessions import IST_OFFSET_SECONDS, NowLike, Sessions, resolve_now, to_date
from .brokers import DB_PROFILE, broker_profile
from .fetching import (
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_BACKOFF_MAX_SECONDS,
    DEFAULT_MAX_RETRIES,
    DateRange,
    FetchReport,
    HistoryFetcher,
    Series,
    Transport,
    bucket_days,
)

SOURCES = ("auto", "api", "db")

# Intraday stamps are the UTC instant of the bar open; D rows are stamped at 00:00
# UTC of the IST session date (PRD 06 section 3).
TIMESTAMP_CONVENTION = "utc_open/ist_date"

_HISTORIFY_HINT = (
    "Source: Historify (source='db'). It stores what the broker adapter delivered, and a "
    "file imported with date-time strings and no zone is read as UTC, which puts IST clock "
    "times 5 hours 30 minutes late."
)
_DECLARE_SHIFT = (
    " If the server really delivers shifted epochs, declare it with "
    "OpenAlgoProvider(intraday_shift_seconds={seconds}); the calendar check must then pass."
)


@dataclass
class Probe:
    """How `source="auto"` decided.

    decision: "db" (Historify) or "api" (the broker). reason: why, in words.
    failed: a probe request failed, so Historify could not be checked.
    daily_expected, daily_found: completed sessions of the range and how many of
    them have a daily bar in Historify. first_missing: the first that has none (ISO
    date). intraday_days_checked: the days whose intraday bars were asked for.
    """

    decision: str = "api"
    reason: str = ""
    failed: bool = False
    daily_expected: int = 0
    daily_found: int = 0
    first_missing: str | None = None
    intraday_days_checked: list[str] = field(default_factory=list)


class OpenAlgoProvider:
    """Bars from an OpenAlgo server, as a data provider.

    client: an `openalgo.api` instance (duck-typed, see `fetching`).
    source: "auto" (default), "api" (broker) or "db" (Historify).
    broker: broker name for the chunk window table. None asks the server once
        through `ping`; if that fails the conservative default is used and said so.
    calendar: a SessionCalendar for `bars` and `probe`; None is the package default.
    pacing: least seconds between two requests. None uses the table (0.35 s for a
        broker unless it needs more, 0.1 s for Historify).
    chunk_days: override the chunk window of the table, in days.
    max_retries, backoff_base, backoff_max: HTTP 429 handling. The wait doubles
        from `backoff_base` up to `backoff_max` seconds and honours Retry-After.
    intraday_shift_seconds: declare that intraday stamps are this many seconds ahead
        of the true UTC bar open (19800 for IST clock time delivered as UTC). None
        uses the per-broker flag of the table for source="api" and 0 for Historify.
        The calendar check must pass after the shift is removed.
    regular_session_only: drop pre-open, post-close and closed-day bars (counted).
    now: the current time for `bars` and `probe`, injectable for tests.
    sleep, clock: replacements for `time.sleep` and `time.monotonic`, for tests.

    Raises ConfigError for an unknown source, a negative pacing, a chunk window
    below one day or a client that cannot be used.
    """

    name = "openalgo"
    timestamp_convention = TIMESTAMP_CONVENTION

    def __init__(
        self,
        client: object,
        *,
        source: str = "auto",
        broker: str | None = None,
        calendar: SessionCalendar | None = None,
        pacing: float | None = None,
        chunk_days: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max: float = DEFAULT_BACKOFF_MAX_SECONDS,
        intraday_shift_seconds: int | None = None,
        regular_session_only: bool = True,
        now: NowLike = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if source not in SOURCES:
            raise ConfigError(f"source must be one of {SOURCES}, got {source!r}")
        if pacing is not None and pacing < 0:
            raise ConfigError("pacing must be zero or more seconds")
        if chunk_days is not None and chunk_days < 1:
            raise ConfigError("chunk_days must be at least 1")
        self.capabilities = {"bars", "oi"}
        self.source = source
        self.regular_session_only = bool(regular_session_only)
        timing = {name: value for name, value in (("sleep", sleep), ("clock", clock)) if value}
        self.fetcher = HistoryFetcher(
            Transport(client), int(max_retries), float(backoff_base), float(backoff_max), **timing
        )
        self._broker = str(broker).strip().lower() if broker else None
        self._broker_note: str | None = None
        self._calendar = calendar
        self._pacing = pacing
        self._chunk_days = chunk_days
        self._shift = intraday_shift_seconds
        self._now = now

    def bars(
        self, symbol: str, exchange: str, interval: str, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Canonical bars for the inclusive IST date range, from the configured
        source, without the local cache. `load_bars(source=provider)` returns the
        same bars together with their metadata, and raises the same errors.
        """
        from .loader import load_bars  # the loader is built on this module

        options = {"source": self, "calendar": self._calendar, "now": self._now, "cache": False}
        return load_bars(symbol, exchange, interval, start, end, **options).bars

    def broker_name(self) -> str | None:
        """The broker behind the server, asked once through `ping`; None if unknown.

        Raises DataError when the server cannot be reached.
        """
        if self._broker is None:
            reply = self.fetcher.request("ping", {}, self._spacing("db"))
            data = reply.body.get("data") if reply.body else None
            broker = data.get("broker") if isinstance(data, dict) else None
            if reply.succeeded() and broker:
                self._broker = str(broker).strip().lower()
            else:
                self._broker = ""
                self._broker_note = (
                    f"the broker could not be identified (ping answered HTTP {reply.status}: "
                    f"{reply.message()}); the conservative chunk window and pacing were used "
                    "and the local cache was not"
                )
        return self._broker or None

    def probe(
        self, symbol: str, exchange: str, interval: str, start: DateLike, end: DateLike
    ) -> Probe:
        """The decision `source="auto"` would take for the range.

        Raises ConfigError for bad dates or an unknown interval. A probe request that
        fails does not raise: the decision is then the broker and `failed` is True.
        """
        start, end = to_date(start, "start"), to_date(end, "end")
        sessions = Sessions(self._calendar, exchange, start, end)
        return self.choose_source(
            (symbol, exchange, interval), start, end, sessions, resolve_now(self._now), "auto"
        )[1]

    def choose_source(
        self,
        series: Series,
        start: dt.date,
        end: dt.date,
        sessions: Sessions,
        now_ns: int,
        requested: str | None = None,
    ) -> tuple[str, Probe | None]:
        """The source to read ("db" or "api") and, for "auto", the probe that chose it."""
        requested = requested or self.source
        if requested != "auto":
            return requested, None
        probe = self._probe(series, start, end, sessions, now_ns)
        return probe.decision, probe

    def _probe(
        self, series: Series, start: dt.date, end: dt.date, sessions: Sessions, now_ns: int
    ) -> Probe:
        symbol, exchange, interval = series
        probe = Probe()
        last = min(end, sessions.last_complete_day(now_ns))
        required = sessions.days(start, last) if start <= last else []
        probe.daily_expected = len(required)
        if not required:
            probe.reason = (
                "the range holds no completed session, so Historify coverage cannot be confirmed"
            )
            return probe
        spacing = self._spacing("db")
        try:
            daily = self.fetcher.history((symbol, exchange, "D"), (start, last), "db", spacing)
            found = set(bucket_days(daily).tolist())
            missing = [day for day in required if to_day(day) not in found]
            probe.daily_found = len(required) - len(missing)
            if missing:
                probe.first_missing = missing[0].isoformat()
                probe.reason = (
                    f"Historify lacks a daily bar for {len(missing)} of "
                    f"{counted(len(required), 'session')} (first missing: {probe.first_missing})"
                )
                return probe
            edge_days = dict.fromkeys((required[0], required[-1])) if is_intraday(interval) else ()
            for day in edge_days:
                probe.intraday_days_checked.append(day.isoformat())
                stamps_ns = self.fetcher.history(series, (day, day), "db", spacing)["timestamp"]
                session_open, session_close = sessions.session(day)
                stamps_ns = stamps_ns * NANOS_PER_SECOND
                if not np.any((stamps_ns >= session_open) & (stamps_ns < session_close)):
                    probe.reason = (
                        f"Historify has daily bars but no {interval} bars inside the session "
                        f"of {day}"
                    )
                    return probe
        except DataError as exc:
            probe.failed = True
            probe.reason = f"the Historify probe failed: {exc}"
            return probe
        probe.decision = "db"
        probe.reason = (
            f"Historify holds a daily bar for all {counted(len(required), 'session')} of the range"
        )
        if is_intraday(interval):
            probe.reason += f" and {interval} bars on the first and last of them"
        return probe

    def collect(
        self,
        series: Series,
        ranges: Sequence[DateRange],
        source: str,
        sessions: Sessions,
        now_ns: int,
    ) -> tuple[np.ndarray, FetchReport]:
        """The raw rows of the inclusive IST date `ranges` from `source` ("db" or
        "api"), with a report of the fetch (see `HistoryFetcher.collect`).

        Raises DataError when a request fails or is rate limited beyond the retries.
        """
        profile = DB_PROFILE if source == "db" else broker_profile(self.broker_name())
        window_days = self._chunk_days or profile.window_days(series[2])

        def completed_sessions(first: dt.date, last: dt.date) -> int:
            """Sessions from `first` to `last` that had closed at the load time."""
            closes = (sessions.session(day)[1] for day in sessions.days(first, last))
            return sum(1 for close_ns in closes if close_ns <= now_ns)

        return self.fetcher.collect(
            series, ranges, source, window_days, self._spacing(source), completed_sessions
        )

    def shift_seconds(self, source: str, interval: str) -> int:
        """The declared shift of intraday stamps for a source, in seconds."""
        if not is_intraday(interval):
            return 0
        if self._shift is not None:
            return int(self._shift)
        if source == "api":
            return broker_profile(self.broker_name()).intraday_shift_seconds
        return 0

    def notes(self, source: str) -> list[str]:
        """Warnings about the broker behind a source: unidentified, or unverified."""
        if source != "api":
            return []
        broker = self.broker_name()
        notes = [self._broker_note] if self._broker_note else []
        if broker and not broker_profile(broker).timestamps_verified:
            notes.append(
                f"the timestamp convention of broker {broker!r} is not verified in the broker "
                "table; this series passed the session calendar check"
            )
        return notes

    def hints(self, source: str, shift_seconds: int) -> dict[str, str]:
        """Source-specific advice for a timestamp error, keyed by the validator's
        offset code (see `build_bars`)."""
        where = _HISTORIFY_HINT
        if source == "api":
            broker = self.broker_name()
            profile = broker_profile(broker)
            where = f"Source: OpenAlgo source='api', broker {broker or 'unknown'}."
            if broker and not profile.timestamps_verified:
                where += f" {profile.notes}"
        if shift_seconds:
            where += f" A shift of {shift_seconds} s was declared and removed before this check."
        late = _DECLARE_SHIFT.format(seconds=shift_seconds + IST_OFFSET_SECONDS)
        early = _DECLARE_SHIFT.format(seconds=shift_seconds - IST_OFFSET_SECONDS)
        return {"late_5h30": where + late, "early_5h30": where + early, "other": where}

    def _spacing(self, source: str) -> float:
        if self._pacing is not None:
            return self._pacing
        if source == "db":
            return DB_PROFILE.spacing_seconds
        return broker_profile(self.broker_name()).spacing_seconds


__all__ = ["SOURCES", "TIMESTAMP_CONVENTION", "OpenAlgoProvider", "Probe"]

"""`load_bars`: the one entry point the run functions use to get bars.

A load obtains the rows of the base interval from own data, a registered provider or
an OpenAlgo server, then ends with the steps every load shares (`_finish`). This
module chooses the provider and holds the server-source path: the "auto" choice, the
local cache of raw rows and the fetch of the sessions it lacks or has not settled.

W, M, Q and Y bars of a server source are always built from daily bars; the server's
own aggregation of those periods is never used, because it aggregates whatever days
a request happens to cover.
"""

import datetime as dt
import os
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .._calendar_convert import DateLike, from_day, to_day
from .._layout import NANOS_PER_SECOND, parse_interval
from .._messages import counted
from .._protocols import SessionCalendar
from ..errors import ConfigError, DataError, NoDataError
from ._finish import PERIOD_UNITS, BarsResult, LoadMetadata, LoadRequest, finish
from ._sessions import NowLike, Sessions, ist_days, resolve_now, to_date
from .cache import RawCache, SeriesKey, is_settled
from .corporate_actions import CorporateAction, declared
from .fetching import DateRange, bucket_days
from .files import Files
from .normalise import VALUE_COLUMNS, Stamped
from .openalgo_provider import SOURCES, OpenAlgoProvider, Probe
from .own import Frame
from .providers import DataProvider, get_provider, validate_layout, validate_provider
from .splits import require_share

# The arguments of load_bars that describe own data, with the value that means
# "not given".
_OWN_DATA_DEFAULTS = {
    "tz": None,
    "columns": None,
    "label": "open",
    "unit": None,
    "timestamp_format": None,
}
_PROBE_LIMIT = (
    "source='auto' chose Historify from its daily bars and the intraday bars of the first "
    "and the last session; the probe does not check the intraday bars of the days in "
    "between (see the missing_bars count of the quality report)"
)
_NO_DATA_ADVICE = {
    "db": "Download the series in Historify first, or use source='api' for recent data.",
    "api": (
        "A broker serves only instruments in today's master contract, so expired contracts "
        "and very old ranges need Historify (source='db')."
    ),
}


def load_bars(
    symbol: str,
    exchange: str,
    interval: str,
    start: DateLike,
    end: DateLike,
    *,
    source: str | DataProvider | None = "auto",
    client: object = None,
    data: object = None,
    calendar: SessionCalendar | None = None,
    cache: bool = True,
    cache_dir: str | os.PathLike | None = None,
    refresh: bool = False,
    now: NowLike = None,
    base: str | None = None,
    strict: bool = True,
    tz: str | None = None,
    columns: Mapping[str, object] | list[str] | None = None,
    label: str = "open",
    unit: str | None = None,
    timestamp_format: str | None = None,
    broker: str | None = None,
    provider_options: Mapping[str, object] | None = None,
    corporate_actions: Iterable[CorporateAction] | None = None,
) -> BarsResult:
    """Load bars for one instrument in the canonical layout.

    symbol, exchange, interval: OpenAlgo strings ("SBIN", "NSE", "5m" or "D").
    start, end: inclusive IST dates (ISO strings, dates or datetimes). A bar is
        returned when it opens on or after `start` and closes on or before `end`.
    source: "auto" (Historify if it covers the whole range, else the broker; never
        both), "db", "api", the name of a registered provider, or a provider object.
    client: an `openalgo.api` instance; needed for "auto", "db" and "api".
    data: own data instead of a server: a pandas or polars DataFrame, a dict of
        columns, NumPy arrays, a file path (CSV; Parquet and Feather with pyarrow)
        or a provider object. `tz`, `columns`, `label`, `unit` and
        `timestamp_format` describe it (see `Frame`). Naive timestamps need `tz`.
    calendar: a SessionCalendar. None uses the package default calendar.
    cache, cache_dir, refresh: the raw rows of an OpenAlgo source are kept in the
        local cache (see `cache`). Later calls read it and fetch only the sessions
        that are missing or not settled. `refresh=True` fetches the range again.
    now: the current time, injectable for tests (see `resolve_now`). Bars that have
        not closed by then are dropped (the partial candle of today).
    base: the interval to obtain and resample from. The default is `interval`,
        except that W, M, Q and Y bars of a server source are built from "D".
    strict: stop when a quality check of error severity fails. With strict=False
        the bars are returned and `meta.quality` carries the errors.
    broker: broker name for the chunk window table. None asks the server.
    provider_options: keyword arguments for the provider (for OpenAlgo: pacing,
        chunk_days, max_retries, backoff_base, backoff_max, intraday_shift_seconds,
        regular_session_only, sleep, clock).
    corporate_actions: the splits, bonuses and checked dates declared for a share on
        NSE or BSE (a list of `CorporateAction`). A gap that looks like an unadjusted
        split or bonus is an error unless declared (see `splits`).

    Returns a `BarsResult` with `.bars` (the DataFrame) and `.meta` (`LoadMetadata`).

    Raises ConfigError for arguments that do not fit together (corporate actions that
    are not CorporateAction objects, conflict, or are given for an index or a
    derivative contract among them), NoDataError when the
    source has nothing for the range, TimestampError when bars do not fit the
    session calendar, CalendarError for a calendar that cannot answer,
    DataQualityError (with strict) for failed quality checks and DataError for
    server failures and a cache that cannot be repaired.
    """
    start, end = to_date(start, "start"), to_date(end, "end")
    if start > end:
        raise ConfigError(f"start {start} is after end {end}")
    actions = declared(corporate_actions)
    require_share(exchange, actions)
    own = dict(zip(_OWN_DATA_DEFAULTS, (tz, columns, label, unit, timestamp_format), strict=True))
    options = dict(provider_options or {})
    now_ns = resolve_now(now)
    try:
        provider = _provider(source or "auto", client, data, own, options, broker, calendar, now_ns)
    except TypeError as exc:
        raise ConfigError(
            f"provider_options {sorted(options)} do not fit the provider: {exc}"
        ) from exc
    server = isinstance(provider, OpenAlgoProvider)
    kind = parse_interval(interval)[0]
    if base is not None:
        parse_interval(base)
    request = LoadRequest(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        base=base or ("D" if server and kind in PERIOD_UNITS else interval),
        start=start,
        end=end,
        now_ns=now_ns,
        sessions=Sessions(calendar, exchange, start, end),
        corporate_actions=actions,
    )
    if server:
        stamped, meta = _server_rows(request, provider, cache, cache_dir, refresh)
    else:
        stamped, meta = _provider_rows(request, provider)
    return finish(request, stamped, meta, strict)


# --- which provider ------------------------------------------------------------------


def _is_provider(data: object) -> bool:
    """True for `data=` given as a provider object (anything but a pandas DataFrame with
    a `bars` method), False for a table of own data."""
    return not isinstance(data, pd.DataFrame) and callable(getattr(data, "bars", None))


def own_reader(data: object, tz: str | None) -> Frame | None:
    """The own data of one `data=` entry as `load_bars` reads it, for what needs it
    before a load (a run's dates when `start=` or `end=` is omitted): `Files` for a file
    path, `Frame` for a table, both with naive timestamps in `tz`, the object itself for
    a `Frame` or `Files`, and None for a provider that is not own data."""
    if isinstance(data, Frame):
        return data
    if isinstance(data, str | os.PathLike):
        return Files(data, tz=tz)
    return None if _is_provider(data) else Frame(data, tz=tz)


def _provider(
    source: str | DataProvider,
    client: object,
    data: object,
    own: dict,
    options: dict,
    broker: str | None,
    calendar: SessionCalendar | None,
    now_ns: int,
) -> DataProvider:
    """The provider a load reads from: own data, an OpenAlgo server or a plugin."""
    own_given = sorted(name for name, value in own.items() if value != _OWN_DATA_DEFAULTS[name])
    if data is not None:
        if source != "auto":
            raise ConfigError(
                f"data= and source={source!r} are alternatives: own data needs no source"
            )
        if isinstance(data, str | os.PathLike):
            return Files(data, **own, **options)
        if not _is_provider(data):
            return Frame(data, **own, **options)
        source = data
    if own_given:
        raise ConfigError(
            f"{', '.join(own_given)} describe own data (data=...) and have no meaning for "
            f"{getattr(source, 'name', source)!r}"
        )
    if not isinstance(source, str):
        if options:
            raise ConfigError(
                f"provider_options {sorted(options)} cannot be given with a provider object; "
                "give them to the provider"
            )
        return validate_provider(source)
    if source == "openalgo":
        # The registry name of the server sources: the provider's own arguments may
        # then come in provider_options. The calendar and the time of the load win.
        client, source = options.pop("client", client), options.pop("source", "auto")
        options = {
            name: value for name, value in options.items() if name not in ("calendar", "now")
        }
    if source not in SOURCES:
        return get_provider(source, **options)
    if client is None:
        raise ConfigError(
            f"source={source!r} reads from an OpenAlgo server: pass "
            "client=openalgo.api(api_key=..., host=...), or pass data= to use your own data"
        )
    return OpenAlgoProvider(
        client, source=source, broker=broker, calendar=calendar, now=now_ns, **options
    )


def _provider_rows(request: LoadRequest, provider: DataProvider) -> tuple[Stamped, LoadMetadata]:
    """The rows of own data or of a plugin, with what is known about their source."""
    name = provider.name
    if isinstance(provider, Frame):
        stamped = provider.stamped(*request.series, request.start, request.end)
    else:
        raw = provider.bars(*request.series, request.start, request.end)
        if not isinstance(raw, pd.DataFrame) and hasattr(raw, "to_pandas"):
            raw = raw.to_pandas()  # an Arrow table from a provider that has pyarrow
        validate_layout(raw, what=f"bars from data provider {name!r}")
        # A plugin's frame goes through the same calendar validation as every other
        # source; its ts_open values are the stamps.
        stamped = Stamped(
            raw["ts_open"].to_numpy(),
            {column: raw[column] for column in VALUE_COLUMNS},
            canonical=True,
            origin=f"the data provider {name!r}",
            convention=provider.timestamp_convention,
        )
    meta = request.meta(name, name, name, timestamp_convention=stamped.convention)
    return stamped, meta


# --- OpenAlgo: the server and the cache of raw rows ----------------------------------


def _server_rows(
    request: LoadRequest,
    provider: OpenAlgoProvider,
    use_cache: bool,
    cache_dir: str | os.PathLike | None,
    refresh: bool,
) -> tuple[Stamped, LoadMetadata]:
    """The raw rows of an OpenAlgo source, through the local cache when it applies."""
    sessions, now_ns = request.sessions, request.now_ns
    today = from_day(int(ist_days(now_ns)))
    if request.start > today:
        raise NoDataError(
            f"{request.tag}: the requested range {request.start} to {request.end} starts "
            f"after today ({today}); there is no history to load yet"
        )
    source, probe = provider.choose_source(
        request.series, request.start, request.end, sessions, now_ns
    )
    broker = provider.broker_name() if source == "api" else None
    shift = provider.shift_seconds(source, request.base)
    meta = request.meta(
        provider.name,
        source,
        provider.source,
        broker=broker,
        timestamp_convention=provider.timestamp_convention,
        auto_probe=probe,
        warnings=_probe_notes(probe, request.base) + provider.notes(source),
    )
    whole = [(request.start, min(request.end, today))]
    if use_cache and shift:
        meta.warnings.append(
            f"a timestamp shift of {shift} s is declared for this source: the local cache, "
            "whose day partitions follow the stamps as delivered, was not used"
        )
    if use_cache and not shift and (source == "db" or broker):
        records = _through_cache(request, provider, meta, RawCache(cache_dir), refresh)
    else:
        meta.cache = "bypassed" if use_cache else "off"
        records, meta.fetch = provider.collect(request.series, whole, source, sessions, now_ns)
    if len(records) == 0:
        chosen = f" source='auto' chose {source!r} because {probe.reason}." if probe else ""
        raise NoDataError(
            f"{request.tag}: source={source!r} returned no bars for {request.start} to "
            f"{whole[0][1]}.{chosen} {_NO_DATA_ADVICE[source]}"
        )
    present = {name: records[name] for name in VALUE_COLUMNS if not np.isnan(records[name]).all()}
    stamped = Stamped(
        (records["timestamp"] - shift) * NANOS_PER_SECOND,
        present,
        regular_session_only=provider.regular_session_only,
        shift_seconds=shift,
        hints=provider.hints(source, shift),
        origin=f"the broker ({broker or 'unknown'}, source='api')"
        if source == "api"
        else "Historify (source='db')",
        convention=provider.timestamp_convention,
    )
    return stamped, meta


def _probe_notes(probe: Probe | None, interval: str) -> list[str]:
    """What a load says about how `auto` chose its source."""
    if probe is None:
        return []
    if probe.failed:
        return [
            f"source='auto' could not check Historify ({probe.reason}); the broker was used "
            "for the whole range"
        ]
    if probe.decision == "db" and parse_interval(interval)[0] == "intraday":
        return [_PROBE_LIMIT]
    return []


def _through_cache(
    request: LoadRequest,
    provider: OpenAlgoProvider,
    meta: LoadMetadata,
    store: RawCache,
    refresh: bool,
) -> np.ndarray:
    """The raw rows of the requested dates from the cache, after fetching every
    session that is missing from it or not settled (all of them with `refresh`)."""
    sessions, now_ns, source = request.sessions, request.now_ns, meta.source
    namespace = f"openalgo-{source}" + (f"-{meta.broker}" if source == "api" else "")
    key = SeriesKey(namespace, request.exchange, request.symbol, request.base)
    cached = store.read(key, request.start, request.end)
    meta.cache, meta.cache_dir = "hit", str(store.root)
    meta.warnings.extend(cached.notes)
    today = from_day(int(ist_days(now_ns)))
    wanted = [
        day
        for day in sessions.days(request.start, min(request.end, today))
        if sessions.session(day)[0] < now_ns  # a session that has not opened has no bars
    ]
    unsettled = [
        day
        for day in wanted
        if refresh
        or not is_settled(day, cached.fetched_at.get(day), sessions.session(day)[1], source)
    ]
    ranges = _runs(unsettled, wanted)
    if not ranges:
        return cached.records
    meta.cache = "refresh" if refresh else ("top-up" if len(cached.records) else "miss")
    try:
        records, meta.fetch = provider.collect(request.series, ranges, source, sessions, now_ns)
    except DataError as exc:
        if not cached.notes:
            raise
        raise DataError(f"{' '.join(cached.notes)}. Fetching them again failed: {exc}") from exc
    served = set(bucket_days(records).tolist())
    kept = [day for day in unsettled if day in cached.fetched_at and to_day(day) not in served]
    if kept:
        meta.warnings.append(
            "the source returned no rows for "
            f"{counted(len(kept), 'session that is', 'sessions that are')} in the cache "
            f"({kept[0]} to {kept[-1]}); the cached rows were kept"
        )
    fetched_at_s = now_ns // NANOS_PER_SECOND
    store.write(key, records, ranges, fetched_at_s, source=source, broker=meta.broker)
    return store.read(key, request.start, request.end).records


def _runs(days: list[dt.date], among: list[dt.date]) -> list[DateRange]:
    """Group `days` into ranges of days that follow each other in `among`."""
    position = {day: index for index, day in enumerate(among)}
    ranges: list[DateRange] = []
    for day in days:
        if ranges and position[day] == position[ranges[-1][1]] + 1:
            ranges[-1] = (ranges[-1][0], day)
        else:
            ranges.append((day, day))
    return ranges


__all__ = ["BarsResult", "LoadMetadata", "load_bars", "own_reader"]

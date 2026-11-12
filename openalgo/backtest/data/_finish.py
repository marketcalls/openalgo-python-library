"""The steps every load ends with, and what a load returns.

`finish` takes the rows a source delivered and:

1. normalises them into the canonical layout against the session calendar;
2. resamples when the requested interval is coarser than the base, and drops any bar
   that has not closed or is built from part of its period;
3. describes what was obtained against what was asked, runs the quality checks and
   hashes the bars for the run manifest.

`LoadMetadata` is the one record of what a load knows about its bars.
"""

import dataclasses
import datetime as dt
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._calendar_convert import from_day, to_day
from .._layout import BAR_COLUMNS, IST_OFFSET_NANOS, parse_interval
from ..errors import NoDataError
from ..resample import resample_bars
from ._sessions import Sessions, ist_days
from .corporate_actions import CorporateAction
from .fetching import FetchReport
from .normalise import VALUE_COLUMNS, NormaliseReport, Stamped, build_bars
from .openalgo_provider import OpenAlgoProvider, Probe
from .quality import QualityReport, check_bars

PERIOD_UNITS = ("W", "M", "Q", "Y")
# Days by which a W, M, Q or Y bar may start after, or end before, the sessions asked
# for without the series counting as short: the bar that holds the first or the last
# requested day is left out when its period reaches beyond the range.
_PERIOD_SLACK_DAYS = {"W": 7, "M": 31, "Q": 92, "Y": 366}
# Mixed into every manifest hash, so that a change of the hashed form changes them.
_HASH_HEADER = b"openalgo.backtest bars v1\n"
_ADVICE = {
    "api": (
        "A broker keeps only a limited history for this interval. Download the series "
        "into Historify and use source='db' for a longer range."
    ),
    "db": "Historify does not hold the rest: download the missing range in Historify.",
}
_OWN_ADVICE = "The supplied data does not cover the rest of the requested range."


@dataclass
class LoadMetadata:
    """Everything `load_bars` knows about the bars it returned.

    provider: "openalgo", "frame", "files" or the name of a plugin.
    source: the source actually used: "db", "api" or the provider name.
    source_requested: what was asked for ("auto", "db", "api" or a provider name).
    broker: the broker name for source "api", else None.
    timestamp_convention: the declared convention of the raw timestamps.
    cache: "hit", "top-up", "miss", "refresh", "off", "bypassed" or "not-applicable".
    cache_dir: the cache directory used, or None.
    auto_probe: how source="auto" decided, when it was asked.
    fetch: how the fetch of this load went, when there was one.
    warnings: sentences for the user; the short-history sentence comes first.
    requested_start, requested_end: the requested IST dates, ISO strings.
    obtained_start, obtained_end: IST dates of the first and last bar obtained at
        the base interval, ISO strings.
    rows: the number of bars returned.
    short_history: the source returned less than the requested range.
    base_interval: the interval the bars were resampled from, if any.
    normalisation: what normalisation and resampling dropped in this load.
    resample_dropped: the base bars the resampler left out, when the result was
        resampled: pre_open_bars, post_close_bars, closed_day_bars, empty_buckets.
    quality: the QualityReport of the returned bars; base_quality: that of the base
        bars when the result was resampled.
    partitions: the returned bars hashed per IST month, each entry with "key",
        "month", "sha256" and "rows".
    """

    symbol: str
    exchange: str
    interval: str
    provider: str
    source: str
    source_requested: str
    broker: str | None = None
    timestamp_convention: str = ""
    cache: str = "not-applicable"
    cache_dir: str | None = None
    auto_probe: Probe | None = None
    fetch: FetchReport | None = None
    warnings: list[str] = field(default_factory=list)
    requested_start: str = ""
    requested_end: str = ""
    obtained_start: str | None = None
    obtained_end: str | None = None
    rows: int = 0
    short_history: bool = False
    base_interval: str | None = None
    normalisation: NormaliseReport | None = None
    resample_dropped: dict[str, int] | None = None
    quality: QualityReport | None = None
    base_quality: QualityReport | None = None
    partitions: list[dict] = field(default_factory=list)

    def partition_hashes(self) -> dict[str, str]:
        """{partition key: content hash} of the bars returned, for the run manifest."""
        return {entry["key"]: entry["sha256"] for entry in self.partitions}

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form, with the partition hashes under "partition_hashes"."""
        out = dataclasses.asdict(self)
        out["quality"]["ok"] = self.quality.ok
        out["partition_hashes"] = self.partition_hashes()
        return out


@dataclass
class BarsResult:
    """Canonical bars plus their metadata. Unpacks as `bars, meta = load_bars(...)`."""

    bars: pd.DataFrame
    meta: LoadMetadata

    def __iter__(self):
        yield self.bars
        yield self.meta


def partition_hashes(result: BarsResult | LoadMetadata) -> dict[str, str]:
    """{partition key: content hash} of the bars a load returned. This is what a run
    manifest records about its data, so that a result can be reproduced or flagged
    stale when the data changed."""
    meta = result.meta if isinstance(result, BarsResult) else result
    return meta.partition_hashes()


@dataclass(frozen=True)
class LoadRequest:
    """One load: the instrument, the interval to return and the base interval to
    obtain, the inclusive IST dates, the load time, the sessions of the exchange and
    the corporate actions declared for the share."""

    symbol: str
    exchange: str
    interval: str
    base: str
    start: dt.date
    end: dt.date
    now_ns: int
    sessions: Sessions
    corporate_actions: tuple[CorporateAction, ...] = ()

    @property
    def series(self) -> tuple[str, str, str]:
        """(symbol, exchange, base interval): the series a source is asked for."""
        return self.symbol, self.exchange, self.base

    @property
    def tag(self) -> str:
        """The series as messages name it, for example 'NSE:SBIN 1m'."""
        return f"{self.exchange}:{self.symbol} {self.base}"

    def meta(self, provider: str, source: str, requested: str, **facts: object) -> LoadMetadata:
        """A LoadMetadata for this request; `facts` fill its other fields."""
        return LoadMetadata(
            self.symbol, self.exchange, self.interval, provider, source, requested, **facts
        )


def finish(request: LoadRequest, stamped: Stamped, meta: LoadMetadata, strict: bool) -> BarsResult:
    """Normalise, resample, describe and check the rows of a load."""
    sessions = request.sessions
    bars, report = build_bars(
        stamped.stamps_ns,
        stamped.values,
        symbol=request.symbol,
        exchange=request.exchange,
        interval=request.base,
        calendar=sessions,
        label=stamped.label,
        now_ns=request.now_ns,
        start=request.start,
        end=request.end,
        regular_session_only=stamped.regular_session_only,
        canonical_input=stamped.canonical,
        hints=stamped.hints,
    )
    report.shift_applied_seconds = stamped.shift_seconds
    advice = _ADVICE.get(meta.source, _OWN_ADVICE)
    _describe_coverage(bars, request, meta, stamped.origin, advice)
    declared = request.corporate_actions
    base_bars = None
    if request.base != request.interval:
        meta.base_interval = request.base
        meta.base_quality = check_bars(
            bars, calendar=sessions, now_ns=request.now_ns, corporate_actions=declared
        )
        base_bars = bars
        bars, meta.resample_dropped = _resampled(bars, request, report)
        left_out = {name: count for name, count in meta.resample_dropped.items() if count}
        if left_out:
            meta.warnings.append(f"resampling to {request.interval} left out base bars: {left_out}")
    if len(bars) == 0:
        raise NoDataError(
            f"{request.tag}: {stamped.origin} has no completed bars between {request.start} and "
            f"{request.end} ({report.rows_in} rows in range; pre-open {report.preopen_dropped}, "
            f"post-close {report.postclose_dropped}, closed days {report.closed_day_dropped}, "
            f"partial {report.partial_dropped}, incomplete periods "
            f"{report.incomplete_period_dropped}). {advice}"
        )
    label = meta.provider
    if meta.provider == OpenAlgoProvider.name:  # one manifest key per source and broker
        label += f"-{meta.source}" + (f"-{meta.broker}" if meta.broker else "")
    bars.attrs = {
        "symbol": request.symbol,
        "exchange": request.exchange,
        "interval": request.interval,
        "provider": meta.provider,
        "source": meta.source,
        "broker": meta.broker,
        "timestamp_convention": meta.timestamp_convention,
    }
    meta.warnings.extend(note for note in report.notes if note not in meta.warnings)
    meta.rows = len(bars)
    meta.normalisation = report
    meta.quality = check_bars(
        bars,
        calendar=sessions,
        normalisation=report,
        now_ns=request.now_ns,
        corporate_actions=declared,
        base_bars=base_bars,
    )
    key = f"{label}/{request.exchange}/{request.symbol}/{request.interval}"
    meta.partitions = _hash_by_month(bars, key)
    if strict:
        meta.quality.raise_for_errors()
    return BarsResult(bars, meta)


def _describe_coverage(
    bars: pd.DataFrame, request: LoadRequest, meta: LoadMetadata, origin: str, advice: str
) -> None:
    """Record what the base bars cover of the requested dates. A source that returns
    less than was asked for is never silent (PRD 06 section 4): the shortfall goes
    first in the warnings. Only sessions that have closed are expected, so a range
    that ends today is not short during the day."""
    sessions, start, end = request.sessions, request.start, request.end
    meta.requested_start, meta.requested_end = start.isoformat(), end.isoformat()
    last_expected = min(end, sessions.last_complete_day(request.now_ns))
    expected = sessions.days(start, last_expected) if start <= last_expected else []
    if len(bars) == 0 or not expected:
        return
    first = from_day(int(ist_days(bars["ts_open"].iloc[0])))
    last = from_day(int(ist_days(bars["ts_close"].iloc[-1])))
    meta.obtained_start, meta.obtained_end = first.isoformat(), last.isoformat()
    slack_days = _PERIOD_SLACK_DAYS.get(parse_interval(request.base)[0], 0)
    slack = dt.timedelta(days=slack_days)
    meta.short_history = first > expected[0] + slack or last < expected[-1] - slack
    if not meta.short_history:
        return
    counted = ""
    if not slack_days:
        with_bars = set(ist_days(bars["ts_open"].to_numpy()).tolist())
        obtained = sum(1 for day in expected if to_day(day) in with_bars)
        counted = f", {obtained} of {len(expected)} sessions"
    stopped = meta.fetch.stopped if meta.fetch else None
    because = f" The fetch stopped going back because {stopped}." if stopped else ""
    meta.warnings.insert(
        0,
        f"Short history: {start} to {end} was requested but {origin} returned {first} to "
        f"{last}{counted}. {advice}{because}",
    )


def _resampled(
    bars: pd.DataFrame, request: LoadRequest, report: NormaliseReport
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resample the base bars to the requested interval, then apply the rules every
    returned bar obeys: it has closed at the load time, and a W, M, Q or Y bar lies
    wholly inside the days the base bars cover, so it is never built from part of
    its period. What these rules drop is counted in `report`; what the resampler
    left out (base bars outside the sessions, kept with regular_session_only=False,
    and buckets without a price) is returned with the bars."""
    if len(bars) == 0:
        return bars, {}
    out = resample_bars(
        bars,
        request.interval,
        exchange=request.exchange,
        calendar=request.sessions.native,
        source_interval=request.base,
        on_closed_day="drop",
    )
    dropped = {name: int(count) for name, count in out.attrs["dropped"].items()}
    keep = out["ts_close"].to_numpy() <= request.now_ns
    report.partial_dropped += int((~keep).sum())
    if parse_interval(request.interval)[0] in PERIOD_UNITS:
        first_base_day = ist_days(bars["ts_open"].iloc[0])
        last_base_day = ist_days(bars["ts_close"].iloc[-1])
        whole = (ist_days(out["ts_open"].to_numpy()) >= first_base_day) & (
            ist_days(out["ts_close"].to_numpy()) <= last_base_day
        )
        report.incomplete_period_dropped += int((keep & ~whole).sum())
        keep &= whole
    return out.loc[keep].reset_index(drop=True), dropped


def _hash_by_month(bars: pd.DataFrame, key: str) -> list[dict]:
    """The returned bars hashed per IST month of `ts_open`: what the run used.

    The hash is the SHA-256 of the rows as little-endian int64 times and float64
    values, with one bit pattern for NaN and zero for negative zero, so that equal
    bars hash equally on every platform.
    """
    dtype = [(name, "<i8") for name in BAR_COLUMNS[:2]] + [(name, "<f8") for name in VALUE_COLUMNS]
    records = np.empty(len(bars), dtype=dtype)
    for name in BAR_COLUMNS[:2]:
        records[name] = bars[name].to_numpy()
    for name in VALUE_COLUMNS:
        column = bars[name].to_numpy(dtype="float64") + 0.0  # a copy; -0.0 becomes 0.0
        column[np.isnan(column)] = np.nan
        records[name] = column
    local = (records["ts_open"] + IST_OFFSET_NANOS).astype("datetime64[ns]")
    months = np.datetime_as_string(local.astype("datetime64[M]"), unit="M")
    entries = []
    for month in dict.fromkeys(months.tolist()):
        rows = records[months == month]
        digest = hashlib.sha256(_HASH_HEADER + rows.tobytes()).hexdigest()
        entries.append(
            {
                "key": f"{key}/{month}",
                "month": month,
                "sha256": f"sha256:{digest}",
                "rows": len(rows),
            }
        )
    return entries


__all__ = [
    "PERIOD_UNITS",
    "BarsResult",
    "LoadMetadata",
    "LoadRequest",
    "finish",
    "partition_hashes",
]

"""Raw history from an OpenAlgo server: transport, chunking, pacing and backoff.

Why not `client.history()`: the SDK turns the response into a DataFrame whose
index is tz-aware IST for intraday intervals, naive for D, W and M and shifted by
5 hours 30 minutes for Q and Y, and it flattens HTTP errors into a message string.
The data layer needs the raw epoch seconds and the HTTP status (429 must be backed
off), so it posts to the same endpoint itself, through the pooled `httpx.Client`
that every REST mixin of `openalgo.api` shares as `client.client` (with
`client.base_url`, `client.headers`, `client.timeout` and `client.api_key`). No
second pool is opened. A client without those attributes but with the SDK's
`_post(endpoint, payload)` helper is used through that helper instead.

`HistoryFetcher` sends the requests, never closer together than the pacing, backs
off on HTTP 429 and walks a date range window by window. The window sizes and the
pacing per broker are data in `brokers.py`.
"""

import datetime as dt
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._layout import BAR_COLUMNS, NANOS_PER_SECOND, PRICE_COLUMNS
from .._messages import counted
from ..errors import ConfigError, DataError, SymbolNotFoundError, TimestampError
from ._sessions import ist_days

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
# How the server's history service words its HTTP 400 for a symbol its master contract
# does not hold: "Symbol 'X' not found for exchange 'NSE'. Please verify ..."
# (`validate_symbol_exchange` in services/history_service.py of the OpenAlgo server).
UNKNOWN_SYMBOL_MARK = "not found for exchange"

# A raw history row as the server returns it: epoch seconds and the bar values.
# Volume and open interest are NaN when the server sent none.
RAW_DTYPE = np.dtype([("timestamp", "<i8")] + [(name, "<f8") for name in BAR_COLUMNS[2:]])

# Defaults for HTTP 429. One second, doubling, is how the server's own broker
# adapters retry a rate-limited chunk. Five retries wait 31 seconds in total, well
# past the 10 seconds after which the server refuses a queued data request with 429
# (utils/broker_backpressure.py, BROKER_MAX_QUEUE_WAIT_SECONDS).
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_MAX_SECONDS = 60.0

# Epoch seconds reach 1e11 in the year 5138; anything at or above is not seconds
# (epoch milliseconds are about 1.7e12 today).
_EPOCH_SECONDS_LIMIT = 1e11
# Longest server message quoted in an error or a warning.
_MESSAGE_LIMIT = 300
# `error_type` values the SDK's `_post` returns when no HTTP response arrived.
_SDK_TRANSPORT_FAILURES = ("timeout_error", "connection_error", "http_error", "unknown_error")

Series = tuple[str, str, str]  # (symbol, exchange, interval) as OpenAlgo strings
DateRange = tuple[dt.date, dt.date]  # (first, last) IST dates, both inclusive


def bucket_days(records: np.ndarray) -> np.ndarray:
    """Day number (days since 1970-01-01) of the IST date each raw row is stamped on."""
    return ist_days(records["timestamp"] * NANOS_PER_SECOND)


@dataclass(frozen=True)
class Reply:
    """One HTTP answer: the status, the JSON body when it is an object, the raw
    text otherwise, and the Retry-After header in seconds when the server sent one."""

    status: int
    body: dict | None
    text: str = ""
    retry_after_seconds: float | None = None

    def message(self) -> str:
        """The server's message, cut to a length that fits an error message."""
        if self.body is not None and self.body.get("message") is not None:
            return str(self.body["message"])[:_MESSAGE_LIMIT]
        return (self.text or "no message")[:_MESSAGE_LIMIT]

    def succeeded(self) -> bool:
        """True for HTTP 200 with `"status": "success"` in the body."""
        return self.status == HTTP_OK and (self.body or {}).get("status") == "success"


class Transport:
    """Posts JSON to the OpenAlgo REST API through an SDK client (duck-typed).

    Raises ConfigError at construction when the client offers neither the pooled
    HTTP client attributes nor the `_post` helper, or has no API key.
    """

    def __init__(self, client: object) -> None:
        if client is None:
            raise ConfigError(
                "an OpenAlgo client is required: pass client=openalgo.api(api_key=..., host=...)"
            )
        pool = getattr(client, "client", None)
        base_url = getattr(client, "base_url", None)
        self._through_pool = bool(callable(getattr(pool, "post", None)) and base_url)
        if not self._through_pool and not callable(getattr(client, "_post", None)):
            raise ConfigError(
                "client must be an openalgo.api instance, or an object with the same "
                "api_key, base_url, headers and pooled HTTP client attributes: the provider "
                "reads raw history rows through the client's own connection pool"
            )
        if not getattr(client, "api_key", None):
            raise ConfigError("client has no api_key")
        self._client = client

    def post(self, endpoint: str, payload: dict) -> Reply:
        """POST `payload` (plus the API key) to `endpoint` and return the Reply.

        Raises DataError when no HTTP response arrives. The API key is never part
        of an error message.
        """
        client = self._client
        body = {"apikey": client.api_key, **payload}
        if not self._through_pool:
            return self._post_through_sdk(endpoint, body)
        options = {"json": body}
        # Passed only when the client defines them: for httpx an explicit
        # timeout=None would switch the timeout off.
        for name in ("headers", "timeout"):
            if getattr(client, name, None) is not None:
                options[name] = getattr(client, name)
        try:
            response = client.client.post(client.base_url + endpoint, **options)
        except Exception as exc:
            # Any failure of the HTTP library (timeout, refused connection, TLS)
            # means the same to the caller. Its message names the URL, never the
            # request body, so the API key cannot leak through it.
            raise DataError(
                f"could not reach the OpenAlgo server for '{endpoint}': {type(exc).__name__}: {exc}"
            ) from exc
        try:
            data = response.json()
        except ValueError:
            data = None  # not JSON (a proxy error page): the status and text still count
        try:
            retry_after = float(response.headers.get("Retry-After"))
        except (AttributeError, TypeError, ValueError):
            retry_after = None  # absent, or given as a date
        text = str(getattr(response, "text", "") or "")
        return Reply(
            int(response.status_code), data if isinstance(data, dict) else None, text, retry_after
        )

    def _post_through_sdk(self, endpoint: str, body: dict) -> Reply:
        data = self._client._post(endpoint, body)
        if not isinstance(data, dict):
            raise DataError(f"unexpected reply from the client for '{endpoint}'")
        if "code" in data:
            return Reply(int(data["code"]), data, str(data.get("raw_response") or ""))
        if data.get("status") != "success" and data.get("error_type") in _SDK_TRANSPORT_FAILURES:
            raise DataError(
                f"could not reach the OpenAlgo server for '{endpoint}': {data.get('message')}"
            )
        return Reply(HTTP_OK, data)


def plan_chunks(start: dt.date, end: dt.date, window_days: int) -> list[DateRange]:
    """Split the inclusive date range into windows of at most `window_days` days.

    Returns [(first_day, last_day), ...] oldest first. Windows are anchored at
    `end`, so the newest one is full-sized and only the oldest can be shorter. They
    do not overlap and leave no gap.

    Raises ConfigError when `window_days` is below 1 or `start` is after `end`.
    """
    if window_days < 1:
        raise ConfigError(f"chunk window must be at least 1 day, got {window_days}")
    if start > end:
        raise ConfigError(f"start {start} is after end {end}")
    chunks = []
    high = end
    while high >= start:
        low = max(start, high - dt.timedelta(days=window_days - 1))
        chunks.append((low, high))
        high = low - dt.timedelta(days=1)
    chunks.reverse()
    return chunks


def rows_to_records(rows: Sequence[dict], tag: str = "history") -> np.ndarray:
    """Read raw `/history` records into an array of RAW_DTYPE.

    The `timestamp` field must hold epoch seconds, which OpenAlgo uses for both
    sources. `tag` names the series in error messages.

    Raises DataError when a required field is missing or a value is not a number,
    and TimestampError when the timestamps are strings, fractions, missing or not
    in the range of epoch seconds: a convention is never guessed.
    """
    try:
        table = pd.DataFrame.from_records(rows)
    except (TypeError, ValueError, AttributeError) as exc:
        raise DataError(f"{tag}: unexpected history payload ({exc})") from exc
    missing = [name for name in ("timestamp", *PRICE_COLUMNS) if name not in table.columns]
    if missing:
        raise DataError(f"{tag}: history rows lack the fields {missing}")
    raw = table["timestamp"]
    if pd.api.types.is_bool_dtype(raw.dtype) or not pd.api.types.is_numeric_dtype(raw.dtype):
        raise TimestampError(
            f"{tag}: expected epoch seconds in the 'timestamp' field, got {raw.iloc[0]!r}. "
            "The provider reads raw epochs only and does not parse date strings."
        )
    epoch_seconds = raw.to_numpy(dtype="float64", na_value=np.nan)
    if not np.all(np.isfinite(epoch_seconds)):
        absent = int((~np.isfinite(epoch_seconds)).sum())
        raise TimestampError(f"{tag}: {counted(absent, 'row has', 'rows have')} no timestamp")
    if np.any(epoch_seconds != np.floor(epoch_seconds)):
        raise TimestampError(f"{tag}: timestamps are not whole epoch seconds")
    if epoch_seconds.min() <= 0 or epoch_seconds.max() >= _EPOCH_SECONDS_LIMIT:
        raise TimestampError(
            f"{tag}: timestamps are outside the range of epoch seconds (for example "
            f"{epoch_seconds.max():.0f}); they may be milliseconds."
        )
    records = np.zeros(len(table), dtype=RAW_DTYPE)
    records["timestamp"] = epoch_seconds.astype("int64")
    try:
        for name in RAW_DTYPE.names[1:]:
            present = name in table.columns
            records[name] = (
                table[name].to_numpy(dtype="float64", na_value=np.nan) if present else np.nan
            )
    except (TypeError, ValueError) as exc:
        raise DataError(f"{tag}: a history value is not a number ({exc})") from exc
    return records


@dataclass
class FetchReport:
    """How a fetch went.

    window_days: the chunk window used. windows_planned, windows_fetched: windows
    the range was split into and windows that were answered. stopped: why a broker
    walk stopped before the oldest window (None when every window was read).
    requests, retries: totals of the fetcher after this fetch. answered: the
    (first, last) IST dates of each answered window as ISO strings, in the order
    asked (strings keep the report JSON-ready, as `LoadMetadata.to_dict` needs).
    """

    window_days: int
    windows_planned: int
    windows_fetched: int = 0
    stopped: str | None = None
    requests: int = 0
    retries: int = 0
    answered: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class HistoryFetcher:
    """Paced, rate-limit aware history requests through one transport.

    max_retries, backoff_base_seconds, backoff_max_seconds: HTTP 429 is retried after
    `backoff_base_seconds`, doubling per attempt up to `backoff_max_seconds`; a
    Retry-After header lengthens a wait but never beyond the maximum.
    sleep and clock replace `time.sleep` and `time.monotonic`, so tests wait for
    nothing. requests, retries and waited_seconds count what was done.
    """

    transport: Transport
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    backoff_max_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    requests: int = 0
    retries: int = 0
    waited_seconds: float = 0.0
    _last_request_at: float | None = field(default=None, init=False, repr=False)

    def request(self, endpoint: str, payload: dict, spacing_seconds: float) -> Reply:
        """Send one request, at least `spacing_seconds` after the previous one.

        HTTP 429 is retried after a backoff wait. Raises DataError when the server
        cannot be reached or keeps answering 429 after the allowed retries.
        """
        attempt = 0
        while True:
            now = self.clock()
            if self._last_request_at is not None:
                remaining = self._last_request_at + spacing_seconds - now
                if remaining > 0:
                    self._wait(remaining)
                    now = self.clock()
            self._last_request_at = now
            reply = self.transport.post(endpoint, payload)
            self.requests += 1
            if reply.status != HTTP_TOO_MANY_REQUESTS:
                return reply
            if attempt >= self.max_retries:
                raise DataError(
                    f"the OpenAlgo server kept answering HTTP 429 (rate limited) for "
                    f"'{endpoint}' after {counted(attempt, 'retry', 'retries')}: "
                    f"{reply.message()} Raise pacing= "
                    "or try again later."
                )
            wait = self.backoff_base_seconds * (2**attempt)
            if reply.retry_after_seconds is not None and reply.retry_after_seconds > 0:
                wait = max(wait, reply.retry_after_seconds)
            self.retries += 1
            self._wait(min(self.backoff_max_seconds, wait))
            attempt += 1

    def _wait(self, seconds: float) -> None:
        self.waited_seconds += float(seconds)
        self.sleep(seconds)

    def history(
        self, series: Series, window: DateRange, source: str, spacing_seconds: float
    ) -> np.ndarray:
        """The raw rows of one history request for the inclusive IST date window.

        An empty array means the source has no rows for the window (Historify says
        so with a JSON 404).

        Raises SymbolNotFoundError when the server answers that its master contract
        does not hold the symbol, DataError for every other answer that is not a
        success, naming the HTTP status, and for transport and rate-limit failures.
        """
        symbol, exchange, interval = series
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start_date": window[0].isoformat(),
            "end_date": window[1].isoformat(),
            "source": source,
        }
        tag = f"{exchange}:{symbol} {interval}"
        reply = self.request("history", payload, spacing_seconds)
        rows = (reply.body or {}).get("data") or []
        if reply.succeeded() and isinstance(rows, list):
            return rows_to_records(rows, tag) if rows else np.zeros(0, dtype=RAW_DTYPE)
        if reply.status == HTTP_NOT_FOUND and source == "db" and reply.body is not None:
            return np.zeros(0, dtype=RAW_DTYPE)
        detail = reply.message()
        if reply.body is None:
            detail = "the body is not JSON; check the host of the client"
        message = (
            f"{tag}: the OpenAlgo history request for {window[0]} to {window[1]} "
            f"(source={source!r}) failed (HTTP {reply.status}: {detail})"
        )
        if reply.status == HTTP_BAD_REQUEST and UNKNOWN_SYMBOL_MARK in detail:
            raise SymbolNotFoundError(message, symbol, exchange)
        raise DataError(message)

    def collect(
        self,
        series: Series,
        ranges: Sequence[DateRange],
        source: str,
        window_days: int,
        spacing_seconds: float,
        completed_sessions: Callable[[dt.date, dt.date], int],
    ) -> tuple[np.ndarray, FetchReport]:
        """Fetch the inclusive IST date `ranges` window by window.

        Historify ("db") is read oldest first, every window. A broker ("api") is
        read newest first and the walk stops at the first window of the oldest range
        that holds nothing although `completed_sessions(first, last)` says a session
        has closed in it: that is where the broker's history ends. A window without
        such a session (a weekend, a holiday, today before the close) never ends the
        walk, and neither does one in a later range: sessions the cache has settled
        lie before it, so it is a gap (a holiday the calendar does not know), not the
        start of the broker's history.

        Returns (rows sorted by timestamp, FetchReport). Raises DataError when any
        window fails: a failure is never taken for the end of history.
        """
        windows = [
            (number, window)
            for number, date_range in enumerate(sorted(ranges))
            for window in plan_chunks(*date_range, window_days)
        ]
        report = FetchReport(window_days, len(windows))
        parts = []
        for number, window in windows if source == "db" else reversed(windows):
            records = self.history(series, window, source, spacing_seconds)
            report.windows_fetched += 1
            report.answered.append((window[0].isoformat(), window[1].isoformat()))
            if len(records):
                parts.append(records)
            elif source == "api" and number == 0 and completed_sessions(*window):
                report.stopped = (
                    f"the broker returned no bars for {window[0]} to {window[1]}, so the "
                    f"history is taken to start after {window[1]}; nothing older was asked for"
                )
                break
        report.requests, report.retries = self.requests, self.retries
        records = np.concatenate(parts) if parts else np.zeros(0, dtype=RAW_DTYPE)
        return records[np.argsort(records["timestamp"], kind="stable")], report


__all__ = [
    "DEFAULT_BACKOFF_BASE_SECONDS",
    "DEFAULT_BACKOFF_MAX_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "RAW_DTYPE",
    "DateRange",
    "FetchReport",
    "HistoryFetcher",
    "Reply",
    "Series",
    "Transport",
    "bucket_days",
    "plan_chunks",
    "rows_to_records",
]

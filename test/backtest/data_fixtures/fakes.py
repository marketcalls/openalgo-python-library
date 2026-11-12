"""Fakes for the data layer tests: a session calendar, an OpenAlgo client and a clock.

The fake client answers the two read-only endpoints the provider uses (`history`
and `ping`) from in-memory rows, in the raw shapes the OpenAlgo server produces for
Historify (`source="db"`) and for a broker (`source="api"`). It never opens a
socket; its base URL is not resolvable on purpose.
"""

import datetime as dt

NS = 1_000_000_000
IST_SECONDS = 19_800
DAY_SECONDS = 86_400

SESSION_HOURS = {
    "NSE": (9 * 3600 + 15 * 60, 15 * 3600 + 30 * 60),
    "MCX": (9 * 3600, 23 * 3600 + 30 * 60),
}


def ist_midnight_seconds(day):
    """Epoch seconds of 00:00 IST on `day`."""
    return (day - dt.date(1970, 1, 1)).days * DAY_SECONDS - IST_SECONDS


def ist_date_of(epoch_seconds):
    return dt.date(1970, 1, 1) + dt.timedelta(days=(epoch_seconds + IST_SECONDS) // DAY_SECONDS)


def at_ist(day, hour, minute=0, second=0):
    """UTC nanoseconds of an IST clock time on `day`."""
    return (ist_midnight_seconds(day) + hour * 3600 + minute * 60 + second) * NS


def days_between(first, last):
    return [first + dt.timedelta(days=n) for n in range((last - first).days + 1)]


class FakeCalendar:
    """A SessionCalendar: weekday sessions with fixed hours per exchange, optional
    holidays and specials. Every date is defined by the test, so its coverage is
    "sourced"."""

    def __init__(self, holidays=(), special=None):
        self.holidays = set(holidays)
        self.special = dict(special or {})
        self.asked = []  # every (exchange, day) the code under test looked up

    def coverage(self, exchange, day):
        return "sourced"

    def session(self, exchange, day):
        self.asked.append((exchange, day))
        if (exchange, day) in self.special:
            return self.special[(exchange, day)]
        if day.weekday() >= 5 or day in self.holidays:
            return None
        opens, closes = SESSION_HOURS[exchange]
        base = ist_midnight_seconds(day)
        return (base + opens) * NS, (base + closes) * NS

    def session_days(self, exchange, first, last):
        return [day for day in days_between(first, last) if self.session(exchange, day)]


def truth_rows(calendar, exchange, days, step=60):
    """The reference series: one bar per `step` seconds of every session in `days`.

    `timestamp` is the true UTC epoch second of the bar open. Prices are a fixed
    function of the timestamp, so two builds of the same range are identical.
    """
    rows = []
    for day in days:
        session = calendar.session(exchange, day)
        if session is None:
            continue
        for stamp in range(session[0] // NS, session[1] // NS, step):
            k = (stamp // step) % 97
            base = 100.0 + k * 0.25
            rows.append(
                {
                    "timestamp": stamp,
                    "open": base,
                    "high": base + 1.0,
                    "low": base - 0.5,
                    "close": base + 0.5,
                    "volume": 1000 + k,
                }
            )
    return rows


def daily_rows(calendar, exchange, days):
    """Daily rows in the OpenAlgo convention: 00:00 UTC of the IST session date."""
    rows = []
    for day in days:
        if calendar.session(exchange, day) is None:
            continue
        ordinal = (day - dt.date(1970, 1, 1)).days
        base = 200.0 + (ordinal % 50)
        rows.append(
            {
                "timestamp": ordinal * DAY_SECONDS,
                "open": base,
                "high": base + 5.0,
                "low": base - 5.0,
                "close": base + 1.0,
                "volume": 500_000 + ordinal,
            }
        )
    return rows


def historify_shape(rows, calendar=None, exchange=None, preopen_minutes=0):
    """Rows as Historify returns them: integer epochs, oi stored as 0, and (when
    asked) the pre-open one-minute bars that its REST reads keep."""
    out = []
    seen_days = set()
    for row in rows:
        day = ist_date_of(row["timestamp"])
        if preopen_minutes and day not in seen_days:
            seen_days.add(day)
            session_open = calendar.session(exchange, day)[0] // NS
            for minute in range(preopen_minutes):
                stamp = session_open - 15 * 60 + minute * 60
                out.append(
                    {
                        "timestamp": stamp,
                        "open": 99.0,
                        "high": 99.5,
                        "low": 98.5,
                        "close": 99.0,
                        "volume": 10,
                        "oi": 0,
                    }
                )
        out.append(
            {
                "timestamp": int(row["timestamp"]),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "oi": 0,
            }
        )
    return out


def broker_shape(rows):
    """Rows as a broker adapter returns them through the server: another field
    order, whole prices as integers, oi added as 0 by the server."""

    def number(value):
        return int(value) if float(value).is_integer() else value

    return [
        {
            "close": number(row["close"]),
            "high": number(row["high"]),
            "low": number(row["low"]),
            "open": number(row["open"]),
            "timestamp": int(row["timestamp"]),
            "volume": int(row["volume"]),
            "oi": 0,
        }
        for row in rows
    ]


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = dict(headers or {})

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    @property
    def text(self):
        return "" if self._body is None else str(self._body)


class _FakeHTTP:
    def __init__(self, owner):
        self._owner = owner

    def post(self, url, json=None, headers=None, timeout=None):
        return self._owner.handle(url, json, headers, timeout)


class FakeClient:
    """Looks like `openalgo.api` to the provider: api_key, base_url, headers, timeout
    and a pooled HTTP client with `post`. Serves history from dicts of raw rows.

    db, api: {(symbol, exchange, interval): [raw rows]}
    api_fail_before: a date; broker requests that end before it answer HTTP 500
        (a broker that errors for ranges it no longer holds).
    aggregates_periods: answer W and M requests to Historify by aggregating the
        stored D rows of the requested dates, stamped at the period start, as the
        server's SQL does. The data layer must never rely on that aggregation.
    """

    api_key = "fake-key-for-tests"
    base_url = "http://openalgo.invalid/api/v1/"
    headers = {"Content-Type": "application/json"}
    timeout = 5.0

    def __init__(
        self,
        *,
        broker="zerodha",
        db=None,
        api=None,
        api_fail_before=None,
        clock=None,
        aggregates_periods=False,
    ):
        self.client = _FakeHTTP(self)
        self.broker = broker
        self.db = dict(db or {})
        self.api = dict(api or {})
        self.api_fail_before = api_fail_before
        self.aggregates_periods = aggregates_periods
        self.calls = []
        self.call_times = []
        self.script = []
        self.keys_seen = set()
        self.options_seen = []
        self._clock = clock

    # --- what the tests inspect -------------------------------------------------------

    def history_calls(self, source=None):
        return [
            (p["source"], p["interval"], p["start_date"], p["end_date"])
            for endpoint, p in self.calls
            if endpoint == "history" and (source is None or p["source"] == source)
        ]

    def endpoints(self):
        return {endpoint for endpoint, _ in self.calls}

    def queue(self, status, body=None, headers=None):
        """Answer the next request with this instead of the normal behaviour."""
        self.script.append((status, body, headers))

    # --- the server side --------------------------------------------------------------

    def handle(self, url, payload, headers, timeout):
        assert url.startswith(self.base_url), url
        endpoint = url[len(self.base_url) :]
        payload = dict(payload)
        self.keys_seen.add(payload.pop("apikey", None))
        self.options_seen.append((headers, timeout))
        self.calls.append((endpoint, payload))
        if self._clock is not None:
            self.call_times.append(self._clock())
        if self.script:
            status, body, extra = self.script.pop(0)
            return FakeResponse(status, body, extra)
        if endpoint == "ping":
            if self.broker is None:
                return FakeResponse(
                    500, {"status": "error", "message": "An unexpected error occurred"}
                )
            return FakeResponse(
                200, {"status": "success", "data": {"message": "pong", "broker": self.broker}}
            )
        if endpoint == "history":
            return self._history(payload)
        return FakeResponse(404, None)

    def _history(self, payload):
        key = (payload["symbol"], payload["exchange"], payload["interval"])
        first = dt.date.fromisoformat(payload["start_date"])
        last = dt.date.fromisoformat(payload["end_date"])
        source = payload["source"]
        held = (self.db if source == "db" else self.api).get(key, [])
        if source == "db" and self.aggregates_periods and key[2] in ("W", "M"):
            daily = self.db.get((key[0], key[1], "D"), [])
            held = _server_period_rows(daily, key[2], first, last)
        if held and not isinstance(held[0].get("timestamp"), (int, float)):
            rows = list(held)  # rows with malformed stamps are served as they are
        else:
            rows = [row for row in held if first <= ist_date_of(int(row["timestamp"])) <= last]
        if source == "db":
            if not rows:
                return FakeResponse(
                    404,
                    {
                        "status": "error",
                        "message": f"No data found for {key[0]}:{key[1]} interval {key[2]} in "
                        "local database. Download data first using Historify.",
                    },
                )
            return FakeResponse(200, {"status": "success", "data": rows})
        if self.api_fail_before is not None and last < self.api_fail_before:
            return FakeResponse(
                500,
                {
                    "status": "error",
                    "message": "Error fetching historical data: range not available",
                },
            )
        return FakeResponse(200, {"status": "success", "data": rows})


def _server_period_rows(daily, unit, first, last):
    """W or M rows as Historify builds them: the D rows of the requested dates are
    grouped by week (Monday) or month and stamped at 00:00 UTC of the period start."""
    groups = {}
    for row in daily:
        day = ist_date_of(int(row["timestamp"]))
        if first <= day <= last:
            start = day - dt.timedelta(days=day.weekday()) if unit == "W" else day.replace(day=1)
            groups.setdefault(start, []).append(row)
    return [
        {
            "timestamp": (start - dt.date(1970, 1, 1)).days * DAY_SECONDS,
            "open": rows[0]["open"],
            "high": max(row["high"] for row in rows),
            "low": min(row["low"] for row in rows),
            "close": rows[-1]["close"],
            "volume": sum(row["volume"] for row in rows),
            "oi": 0,
        }
        for start, rows in sorted(groups.items())
    ]


class SdkOnlyClient:
    """A client that exposes only the SDK's `_post` helper (no public HTTP client)."""

    api_key = "fake-key-for-tests"

    def __init__(self, inner):
        self._inner = inner

    def _post(self, endpoint, payload):
        response = self._inner.handle(self._inner.base_url + endpoint, payload, None, None)
        try:
            data = response.json()
        except ValueError:
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}",
                "code": response.status_code,
                "error_type": "http_error",
            }
        if response.status_code != 200:
            data.setdefault("status", "error")
            data["code"] = response.status_code
        return data


class FakeClock:
    """A monotonic clock that only moves when `sleep` is called. No real waiting."""

    def __init__(self, start=1000.0):
        self.now = start
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


# --- helpers shared by the test modules ---------------------------------------------


def shifted(rows, seconds):
    """The same rows with every epoch-second stamp moved by `seconds`."""
    return [dict(row, timestamp=row["timestamp"] + seconds) for row in rows]


def open_ns(rows):
    """The UTC nanosecond stamp of every row, as a list."""
    return [row["timestamp"] * NS for row in rows]


def make_provider(client, calendar, now_ns, clock=None, **options):
    """An OpenAlgoProvider wired to the fakes: injected calendar, clock and sleep."""
    from openalgo.backtest.data import OpenAlgoProvider

    clock = clock or FakeClock()
    return OpenAlgoProvider(
        client, calendar=calendar, now=now_ns, sleep=clock.sleep, clock=clock.clock, **options
    )


def server_load(
    client,
    interval,
    start,
    end,
    *,
    calendar,
    now_ns,
    symbol="X",
    exchange="NSE",
    clock=None,
    **options,
):
    """`load_bars` from a fake server with an injected calendar, clock and sleep."""
    from openalgo.backtest.data import load_bars

    clock = clock or FakeClock()
    provider_options = {"sleep": clock.sleep, "clock": clock.clock}
    provider_options.update(options.pop("provider_options", {}))
    return load_bars(
        symbol,
        exchange,
        interval,
        start,
        end,
        client=client,
        calendar=calendar,
        now=now_ns,
        provider_options=provider_options,
        **options,
    )


def normalise(rows, interval="1m", exchange="NSE", calendar=None, **options):
    """Run raw history rows through `build_bars`; returns (bars, report)."""
    import numpy as np

    from openalgo.backtest.data.normalise import build_bars

    stamps_ns = np.array([int(row["timestamp"]) for row in rows], dtype="int64") * NS
    names = [name for name in ("open", "high", "low", "close", "volume", "oi") if name in rows[0]]
    values = {name: [row.get(name) for row in rows] for name in names}
    return build_bars(
        stamps_ns,
        values,
        symbol="X",
        exchange=exchange,
        interval=interval,
        calendar=calendar or FakeCalendar(),
        **options,
    )


def real_calendar(holidays=()):
    """The engine's own `Calendar` with the regular hours and the given closures on
    NSE and MCX, for tests that must not rest on the fake calendar."""
    from openalgo.backtest.calendar import Calendar

    calendar = Calendar(builtin=False)
    for day in holidays:
        calendar.add_closure(["NSE", "MCX"], day)
    return calendar


def canonical_frame(rows, step_seconds=60):
    """Rows as a DataFrame in the canonical bars layout, built without the data
    layer: `ts_open` is the row stamp and every bar is `step_seconds` long."""
    import numpy as np
    import pandas as pd

    stamps = np.array([row["timestamp"] for row in rows], dtype="int64") * NS
    columns = {"ts_open": stamps, "ts_close": stamps + step_seconds * NS}
    for name in ("open", "high", "low", "close", "volume"):
        columns[name] = np.array([row[name] for row in rows], dtype="float64")
    columns["oi"] = np.full(len(rows), np.nan)
    return pd.DataFrame(columns)


def canonical_daily_frame(calendar, exchange, days):
    """Daily bars in the canonical layout, built without the data layer: one bar per
    session of `days`, from the session open to the session close of the calendar."""
    import numpy as np
    import pandas as pd

    rows = daily_rows(calendar, exchange, days)
    sessions = [calendar.session(exchange, day) for day in days]
    sessions = [session for session in sessions if session is not None]
    columns = {
        "ts_open": np.array([session[0] for session in sessions], dtype="int64"),
        "ts_close": np.array([session[1] for session in sessions], dtype="int64"),
    }
    for name in ("open", "high", "low", "close", "volume"):
        columns[name] = np.array([row[name] for row in rows], dtype="float64")
    columns["oi"] = np.full(len(rows), np.nan)
    return pd.DataFrame(columns)


def ist_frame(rows):
    """Rows as a pandas DataFrame with a tz-aware IST DatetimeIndex (own data)."""
    import pandas as pd

    index = pd.to_datetime([row["timestamp"] for row in rows], unit="s", utc=True)
    frame = pd.DataFrame(
        {name: [row[name] for row in rows] for name in ("open", "high", "low", "close", "volume")},
        index=index.tz_convert("Asia/Kolkata"),
    )
    frame.index.name = "timestamp"
    return frame

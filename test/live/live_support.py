"""Helpers shared by the opt-in live suite (see conftest.py).

Nothing here reads the API key into a variable: `new_client` passes the environment
value straight to the client, so no traceback or log line of this suite can show it.
"""

import datetime as dt
import os
import time
from collections.abc import Callable
from typing import Any

import pytest

from openalgo import api
from openalgo.backtest.calendar import default_calendar

API_KEY_ENV = "OPENALGO_API_KEY"
HOST_ENV = "OPENALGO_HOST"
WS_URL_ENV = "OPENALGO_WS_URL"
DEFAULT_HOST = "http://127.0.0.1:5000"
LIVE = bool(os.environ.get(API_KEY_ENV))
live_only = pytest.mark.skipif(
    not LIVE, reason=f"set {API_KEY_ENV} to run the live suite against an OpenAlgo server"
)

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NANOS_PER_SECOND = 1_000_000_000
# The server writes a contract's expiry as DD-MMM-YY, for example 19-NOV-26.
SERVER_EXPIRY_FORMAT = "%d-%b-%y"
# A contract this close to expiry is skipped: its last day trades differently.
MIN_DAYS_TO_EXPIRY = 2
# Keep clear of the session edges, where the server may not quote or fill yet.
SESSION_EDGE_SECONDS = 120


def new_client() -> Any:
    """An `openalgo.api` client for the server named by the environment."""
    options = {"host": os.environ.get(HOST_ENV, DEFAULT_HOST)}
    if os.environ.get(WS_URL_ENV):
        options["ws_url"] = os.environ[WS_URL_ENV]
    return api(api_key=os.environ[API_KEY_ENV], **options)


def today_ist() -> dt.date:
    """Today's date in India."""
    return dt.datetime.now(IST).date()


def server_expiry(row: dict) -> dt.date:
    """The expiry date of a master-contract row."""
    return dt.datetime.strptime(row["expiry"], SERVER_EXPIRY_FORMAT).date()


def near_month_future(client: Any, name: str, exchange: str) -> dict:
    """The master-contract row of the future of `name` with the nearest expiry at least
    MIN_DAYS_TO_EXPIRY days away. Skips the test when the server lists none."""
    reply = client.search(query=name, exchange=exchange)
    rows = reply.get("data") or [] if isinstance(reply, dict) else []
    first_day = today_ist() + dt.timedelta(days=MIN_DAYS_TO_EXPIRY)
    futures = [
        row
        for row in rows
        if row.get("name") == name
        and str(row.get("instrumenttype", "")).upper().startswith("FUT")
        and server_expiry(row) >= first_day
    ]
    if not futures:
        pytest.skip(f"the server lists no {exchange} future of {name} expiring after {first_day}")
    return min(futures, key=server_expiry)


def session_is_open(exchange: str, until_ns: Callable[[Any], int] | None = None) -> bool:
    """Whether the exchange's session of today is open now, away from its edges.

    until_ns picks the end of the usable window from the calendar's intraday times
    (for example the last entry time); the default is the session close.
    """
    calendar, day = default_calendar(), today_ist()
    session = calendar.session(exchange, day)
    if session is None:
        return False
    end = session[1] if until_ns is None else until_ns(calendar.intraday_times(exchange, day))
    edge = SESSION_EDGE_SECONDS * NANOS_PER_SECOND
    return session[0] + edge <= time.time_ns() < end - edge


def wait_until(condition: Callable[[], bool], timeout_seconds: float, poll: float = 0.2) -> bool:
    """Poll `condition` until it holds or the timeout passes; returns its last value."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll)
    return condition()


def require_sandbox(client: Any) -> None:
    """Fail unless the server is in analyzer (sandbox) mode. Call it before every order."""
    status = client.analyzerstatus()
    data = status.get("data") or {} if isinstance(status, dict) else {}
    assert data.get("analyze_mode") is True and data.get("mode") == "analyze", (
        f"the server is not in analyzer mode ({status}); no order was placed"
    )

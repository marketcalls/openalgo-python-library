"""The SDK clients of one live session (16-session-design 2.2): five `openalgo.api`
objects derived from the user's client, each with a timeout fit for the thread that
uses it, so that a hung call on one thread never holds the work of another.

| Client | REST timeout | Used by |
|---|---|---|
| `orders` | the user's (the SDK's 120 s by default) | the order worker's calls and reads |
| `guard` | `GUARD_CLIENT_TIMEOUT_S`, 10 s | the mode guard's reads, on the order worker |
| `data` | `DATA_CLIENT_TIMEOUT_S`, 15 s | preflight, warm-up history, candles, quotes |
| `alerts` | `ALERTS_CLIENT_TIMEOUT_S`, 10 s | the alert sender's Telegram posts |
| `feed` | the user's (no REST call) | the feed: the session's data and order callbacks |

The orders client keeps the user's timeout because the order path's windows assume the
server's own 120 s wait on the broker: a shorter timeout only turns more answers into
unknowns. A guard read that times out is a failed read, which the guard counts.

Every derived client carries the user's API key, host, API version, WebSocket URL and
port, has its feed's automatic reconnect on, and never prints (the live layer logs). The
user's client is left alone: none of its settings, callbacks or connections changes, so
a session never takes the data callback a user's own code set (the SDK keeps one per
mode). `SessionClients` holds the five and closes them at the end; built directly from
other clients it is the session's test seam (`run_session(..., clients=...)`).

The API key is never logged, printed or put in an error message.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ... import api
from ..errors import ConfigError, LiveError
from ._preflight_plan import positive_amount
from ._types import LOGGER_NAME

#: Seconds the mode guard's client waits for an answer: the guard reads `analyzerstatus`
#: every 5 s and before each entry on the order worker, between order calls, and a hung
#: read must not hold a stop for the orders client's 120 s (16-session-design 2.2).
GUARD_CLIENT_TIMEOUT_S = 10.0
#: Seconds the data client waits: the preflight, warm-up history, candle checks and
#: quotes; a hung read must not hold the next (16-session-design 2.2).
DATA_CLIENT_TIMEOUT_S = 15.0
#: Seconds the alerts client waits: a hung post must not hold urgent alerts for 120 s
#: (merge notes, alerts round 4).
ALERTS_CLIENT_TIMEOUT_S = 10.0

#: The five clients, in the order they are built and closed.
ROLES = ("orders", "guard", "data", "alerts", "feed")

# `BaseAPI` builds its REST address as f"{host}/api/{version}/" (openalgo/base.py). The
# pattern takes the last "/api/" as the SDK's, so a host with a path of its own keeps it.
_BASE_URL = re.compile(r"(?P<host>.+)/api/(?P<version>[^/]+)/")
_PASS_A_CLIENT = "pass the openalgo.api client the session trades through"

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class ClientSettings:
    """What an `openalgo.api` client was made with: `host` (its REST address without
    `/api/<version>/`), the API `version`, `timeout_s` (seconds a REST call waits for an
    answer), the WebSocket `ws_url` and `ws_port`, and the `api_key`, which its repr
    never shows."""

    host: str
    version: str
    timeout_s: float
    ws_url: str
    ws_port: Any
    api_key: str = field(repr=False)

    def build(self, *, timeout_s: float) -> api:
        """A new `openalgo.api` with these settings and `timeout_s` seconds for each REST
        call, its feed's automatic reconnect on and its printing off. It opens no
        connection until it is used."""
        return api(
            api_key=self.api_key,
            host=self.host,
            version=self.version,
            timeout=timeout_s,
            ws_port=self.ws_port,
            ws_url=self.ws_url,
            verbose=False,
            auto_reconnect=True,
        )


def settings_of(client: Any) -> ClientSettings:
    """The settings `client` was made with, read from an `openalgo.api` client.

    Raises ConfigError, naming what is missing and never showing the key, for an object
    without a non-empty `api_key`, a `base_url` of the form `<host>/api/<version>/`, a
    `timeout` that is a positive number of seconds (a client that waits for ever could
    hold every stop of the session), or a WebSocket `ws_url` and `ws_port`.
    """
    key = getattr(client, "api_key", None)
    if not isinstance(key, str) or not key.strip():
        raise ConfigError(f"the client given to the session has no API key: {_PASS_A_CLIENT}")
    base_url = getattr(client, "base_url", None)
    match = _BASE_URL.fullmatch(base_url) if isinstance(base_url, str) else None
    if match is None:
        raise ConfigError(
            f"the client's base_url {base_url!r} is not the SDK's <host>/api/<version>/: "
            f"{_PASS_A_CLIENT}"
        )
    timeout = getattr(client, "timeout", None)
    if positive_amount(timeout) is None:
        raise ConfigError(
            f"the client's timeout must be a positive number of seconds (the SDK's default "
            f"is 120), got {timeout!r}: a call that waits for ever would hold every stop"
        )
    ws_url = getattr(client, "ws_url", None)
    if not isinstance(ws_url, str) or not ws_url or not hasattr(client, "ws_port"):
        raise ConfigError(f"the client has no WebSocket feed settings: {_PASS_A_CLIENT}")
    return ClientSettings(
        host=match["host"],
        version=match["version"],
        timeout_s=float(timeout),
        ws_url=ws_url,
        ws_port=client.ws_port,
        api_key=key,
    )


@dataclass(frozen=True)
class SessionClients:
    """The five SDK clients of one live session (see the module docstring for what each
    is for). `derive` builds them from the user's client; a test builds one directly from
    clients of the fake OpenAlgo. A context manager that closes them."""

    orders: Any
    guard: Any
    data: Any
    alerts: Any
    feed: Any

    def close(self) -> None:
        """Disconnect the feed client's WebSocket (which stops its automatic reconnect),
        then close every client's pooled HTTP connections. Every client is closed even
        when one fails; then raises LiveError naming each failure, chained from the first.
        Safe to call twice."""
        failures: list[tuple[str, Exception]] = []
        try:
            self.feed.disconnect()
        # A failure here must not leave the REST clients open: it is raised below.
        except Exception as error:
            failures.append(("the feed client's disconnect", error))
        failures += _close_each((role, getattr(self, role)) for role in ROLES)
        if failures:
            named = "; ".join(f"{what}: {type(err).__name__}: {err}" for what, err in failures)
            first = failures[0][1]
            raise LiveError(f"the session's clients did not all close ({named})") from first

    def __enter__(self) -> SessionClients:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def derive(client: Any) -> SessionClients:
    """The session's five clients, built from the settings of the user's `client` with
    the timeouts of the module docstring. The user's client is only read.

    Raises ConfigError for a `client` that is not an `openalgo.api` client (see
    `settings_of`). Should building one fail, the ones built are closed and the error is
    raised.
    """
    settings = settings_of(client)
    timeouts = {
        "orders": settings.timeout_s,
        "guard": GUARD_CLIENT_TIMEOUT_S,
        "data": DATA_CLIENT_TIMEOUT_S,
        "alerts": ALERTS_CLIENT_TIMEOUT_S,
        "feed": settings.timeout_s,
    }
    built: dict[str, Any] = {}
    try:
        for role in ROLES:
            built[role] = settings.build(timeout_s=timeouts[role])
    except BaseException:
        for what, error in _close_each(built.items()):
            logger.error("%s could not be closed after a failed build: %s", what, error)
        raise
    return SessionClients(**built)


def _close_each(clients: Iterable[tuple[str, Any]]) -> list[tuple[str, Exception]]:
    """Close every client given as (role, client); the failures, as (what, error)."""
    failures = []
    for role, client in clients:
        try:
            client.close()
        # Every client is closed whatever one does; the caller reports the failures.
        except Exception as error:
            failures.append((f"the {role} client", error))
    return failures


__all__ = [
    "ALERTS_CLIENT_TIMEOUT_S",
    "DATA_CLIENT_TIMEOUT_S",
    "GUARD_CLIENT_TIMEOUT_S",
    "ROLES",
    "ClientSettings",
    "SessionClients",
    "derive",
    "settings_of",
]

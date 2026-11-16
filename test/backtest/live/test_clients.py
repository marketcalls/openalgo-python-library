"""The SDK clients of a live session (16-session-design 2.2): five `openalgo.api` objects
derived from the user's client, each with the user's API key, host, API version and
WebSocket settings and a timeout of its own, the user's client left alone, a non-client
refused, and every client closed at the end.

No test makes a REST call: the SDK's HTTP pools are stood in for by `LightPool`, which
builds no TLS context and records its close. The feed tests connect derived clients to
the fake OpenAlgo's in-process WebSocket proxy, which the fixture installs.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from fake_openalgo import API_KEY
from fake_openalgo.fixtures import *  # noqa: F403
from feed_fakes import wait_for

from openalgo import api
from openalgo import base as base_module
from openalgo.backtest._live import clients as clients_module
from openalgo.backtest._live.clients import (
    ALERTS_CLIENT_TIMEOUT_S,
    DATA_CLIENT_TIMEOUT_S,
    GUARD_CLIENT_TIMEOUT_S,
    ROLES,
    SessionClients,
    derive,
    settings_of,
)
from openalgo.backtest.errors import ConfigError, LiveError

KEY = "PLACEHOLDER-KEY-NOT-A-REAL-ONE"
HOST = "http://openalgo.invalid:5000"
SBIN = {"exchange": "NSE", "symbol": "SBIN"}


class LightPool:
    """Stands in for the SDK's `httpx.Client`: builds nothing and records its close."""

    def __init__(self, *args, **kwargs):
        self.is_closed = False

    def close(self):
        self.is_closed = True


class LightHttp:
    """Stands in for the `httpx` module inside `openalgo.base`."""

    Client = LightPool

    @staticmethod
    def Limits(**kwargs):
        """Named as in httpx, because the SDK calls it by that name."""
        return None


@pytest.fixture(autouse=True)
def light_http(monkeypatch):
    """Every client built in these tests gets a `LightPool` for its REST calls."""
    monkeypatch.setattr(base_module, "httpx", LightHttp)


def user_client(**options):
    """The user's client: an `openalgo.api` that no test lets reach a network."""
    options.setdefault("api_key", KEY)
    options.setdefault("host", HOST)
    return api(**options)


@pytest.fixture
def derived():
    """The session's clients derived from a user's client made with a 45 s timeout."""
    clients = derive(user_client(timeout=45.0))
    yield clients
    clients.close()


# -- what each client carries --------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_each_client_carries_the_users_key_and_rest_address(derived, role):
    client = getattr(derived, role)

    assert (client.api_key, client.base_url) == (KEY, f"{HOST}/api/v1/")


@pytest.mark.parametrize(
    ("role", "timeout_s"),
    [
        ("orders", 45.0),
        ("guard", GUARD_CLIENT_TIMEOUT_S),
        ("data", DATA_CLIENT_TIMEOUT_S),
        ("alerts", ALERTS_CLIENT_TIMEOUT_S),
        ("feed", 45.0),
    ],
)
def test_each_client_has_its_own_timeout(derived, role, timeout_s):
    assert getattr(derived, role).timeout == timeout_s


def test_the_guard_and_alerts_wait_10_seconds_and_the_data_client_15():
    # 16-session-design 2.2: a hung read or post must not hold the next for 120 s.
    assert (GUARD_CLIENT_TIMEOUT_S, ALERTS_CLIENT_TIMEOUT_S, DATA_CLIENT_TIMEOUT_S) == (10, 10, 15)


def test_an_api_version_other_than_v1_is_carried():
    with derive(user_client(version="v2")) as clients:
        assert clients.orders.base_url == f"{HOST}/api/v2/"


def test_a_host_with_a_path_of_its_own_keeps_it():
    host = "https://example.invalid/openalgo"

    with derive(user_client(host=host)) as clients:
        assert clients.data.base_url == f"{host}/api/v1/"


@pytest.mark.parametrize(
    ("options", "ws_url"),
    [
        ({}, "ws://openalgo.invalid:8765"),
        ({"ws_port": 9001}, "ws://openalgo.invalid:9001"),
        ({"host": "https://algo.example.invalid"}, "wss://algo.example.invalid/ws"),
        ({"ws_url": "wss://feed.example.invalid/stream"}, "wss://feed.example.invalid/stream"),
    ],
    ids=["derived from the host", "a port chosen", "a hosted server", "a URL of its own"],
)
def test_the_feed_client_connects_where_the_users_client_would(options, ws_url):
    user = user_client(**options)

    with derive(user) as clients:
        assert (clients.feed.ws_url, clients.feed.ws_port) == (ws_url, user.ws_port)


def test_the_feed_client_reconnects_by_itself_and_never_prints():
    with derive(user_client(verbose=2, auto_reconnect=False)) as clients:
        assert (clients.feed.auto_reconnect, clients.feed.verbose) == (True, 0)


def test_the_five_clients_are_new_objects_each():
    user = user_client()

    with derive(user) as clients:
        made = [getattr(clients, role) for role in ROLES]
        assert len({id(client) for client in [user, *made]}) == 1 + len(ROLES)


def test_the_session_clients_hold_exactly_the_five_roles():
    assert tuple(f.name for f in dataclasses.fields(SessionClients)) == ROLES


def test_the_settings_never_show_the_api_key():
    settings = settings_of(user_client())

    assert KEY not in repr(settings)
    assert settings.api_key == KEY


# -- a non-client is refused ----------------------------------------------------------------


class NotAClient:
    """Anything with some of a client's attributes, each set by keyword."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def client_like(**changes):
    attributes = {
        "api_key": KEY,
        "base_url": f"{HOST}/api/v1/",
        "timeout": 120.0,
        "ws_url": "ws://openalgo.invalid:8765",
        "ws_port": 8765,
    }
    attributes.update(changes)
    return NotAClient(**{name: value for name, value in attributes.items() if value is not ...})


@pytest.mark.parametrize(
    ("client", "says"),
    [
        (None, "no API key"),
        ("http://127.0.0.1:5000", "no API key"),
        (client_like(api_key=...), "no API key"),
        (client_like(api_key="  "), "no API key"),
        (client_like(base_url=...), "base_url None"),
        (client_like(base_url="http://127.0.0.1:5000/"), "not the SDK's <host>/api/<version>/"),
        (client_like(timeout=None), "positive number of seconds"),
        (client_like(timeout=0), "positive number of seconds"),
        (client_like(timeout=math.inf), "positive number of seconds"),
        (client_like(timeout=True), "positive number of seconds"),
        (client_like(ws_url=...), "no WebSocket feed settings"),
        (client_like(ws_port=...), "no WebSocket feed settings"),
    ],
    ids=[
        "None",
        "a host text",
        "no key",
        "a blank key",
        "no base_url",
        "a base_url without the API path",
        "no timeout",
        "a zero timeout",
        "an endless timeout",
        "a bool timeout",
        "no ws_url",
        "no ws_port",
    ],
)
def test_a_non_client_is_refused_saying_what_is_missing(client, says):
    with pytest.raises(ConfigError, match="openalgo.api client|seconds") as raised:
        derive(client)

    assert says in str(raised.value)
    assert KEY not in str(raised.value)


def test_a_client_that_fails_to_build_closes_the_ones_built_first(monkeypatch):
    built = []

    def build_or_fail(**options):
        if len(built) == 2:
            raise RuntimeError("no more clients")
        client = api(**options)
        built.append(client)
        return client

    monkeypatch.setattr(clients_module, "api", build_or_fail)

    with pytest.raises(RuntimeError, match="no more clients"):
        derive(user_client())

    assert [client.client.is_closed for client in built] == [True, True]


# -- closing ------------------------------------------------------------------------------------


def test_close_closes_every_clients_http_pool():
    clients = derive(user_client())

    clients.close()

    assert [getattr(clients, role).client.is_closed for role in ROLES] == [True] * len(ROLES)


def test_closing_twice_is_harmless():
    clients = derive(user_client())
    clients.close()

    clients.close()

    assert clients.orders.client.is_closed


def test_the_clients_close_when_their_with_block_ends():
    with derive(user_client()) as clients:
        pass

    assert clients.alerts.client.is_closed


class FailingClose:
    """A client whose close fails; its feed disconnect works."""

    def __init__(self):
        self.closed = False

    def disconnect(self):
        """The feed's disconnect."""

    def close(self):
        self.closed = True
        raise OSError("the pool would not close")


class Closing(FailingClose):
    def close(self):
        self.closed = True


def test_one_failing_close_still_closes_the_others_then_says_which_failed():
    clients = SessionClients(
        orders=Closing(), guard=FailingClose(), data=Closing(), alerts=Closing(), feed=Closing()
    )

    with pytest.raises(LiveError, match="the guard client: OSError: the pool would not close"):
        clients.close()

    assert [getattr(clients, role).closed for role in ROLES] == [True] * len(ROLES)


# -- the feed client on the fake OpenAlgo's proxy -------------------------------------------------


def test_close_disconnects_the_feed_so_it_never_reconnects(fake_openalgo):
    clients = derive(fake_openalgo.client())
    try:
        assert clients.feed.connect() is True
    finally:
        clients.close()

    assert clients.feed.connected is False
    assert wait_for(lambda: fake_openalgo.feed_server.live_socket() is None)
    assert clients.feed.auto_reconnect is True


def test_the_users_data_callback_keeps_its_ticks_beside_the_sessions(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    user = fake_openalgo.client()
    user_ticks, session_ticks = [], []
    assert user.connect() is True
    user.subscribe_quote([SBIN], user_ticks.append)
    clients = derive(user)
    try:
        assert clients.feed.api_key == API_KEY
        assert clients.feed.connect() is True
        clients.feed.subscribe_quote([SBIN], session_ticks.append)

        fake_openalgo.trade("SBIN", "NSE", 812.0)
    finally:
        clients.close()

    assert user.quote_callback == user_ticks.append
    assert [len(user_ticks), len(session_ticks)] == [1, 1]

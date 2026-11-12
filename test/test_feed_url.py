"""The WebSocket URL of the feed: what an explicit ws_url, an http host, an https
host and a chosen ws_port each give, for FeedAPI and for the unified api class.

Nothing here opens a socket: the clients are only constructed.
"""

import pytest
from feed_fakes import API_KEY, CLIENT_CLASSES
from feed_fixtures import *  # noqa: F403

import openalgo
from openalgo._feed_url import derive_ws_url
from openalgo.feed import FeedAPI

# Every row: what the caller configures, and the WebSocket URL that follows.
# A key that is missing was not passed to the constructor.
CONFIGURATIONS = [
    # Self-hosted, local or on a LAN: plain ws:// on the WebSocket port.
    ({}, "ws://127.0.0.1:8765"),
    ({"host": "http://127.0.0.1:5000"}, "ws://127.0.0.1:8765"),
    ({"host": "http://127.0.0.1:5000", "ws_port": 9001}, "ws://127.0.0.1:9001"),
    ({"host": "http://localhost:5000"}, "ws://localhost:8765"),
    ({"host": "http://localhost:5000", "ws_port": 9001}, "ws://localhost:9001"),
    ({"host": "http://192.168.1.10:5000"}, "ws://192.168.1.10:8765"),
    ({"host": "http://192.168.1.10:5000", "ws_port": 8766}, "ws://192.168.1.10:8766"),
    ({"host": "http://example.org:5000/openalgo/v2"}, "ws://example.org:8765"),
    # Hosted on a domain behind TLS: wss:// on the path the installers serve.
    ({"host": "https://trade.example.org"}, "wss://trade.example.org/ws"),
    ({"host": "https://trade.example.org/"}, "wss://trade.example.org/ws"),
    ({"host": "https://trade.example.org:8443"}, "wss://trade.example.org/ws"),
    ({"host": "https://trade.example.org/openalgo"}, "wss://trade.example.org/ws"),
    ({"host": "https://trade.example.org:8443/openalgo"}, "wss://trade.example.org/ws"),
    ({"host": "HTTPS://Trade.Example.org"}, "wss://Trade.Example.org/ws"),
    ({"host": "https://[2001:db8::1]:5000/x"}, "wss://[2001:db8::1]/ws"),
    # An https host with a chosen port keeps the URL it had in 2.0.5.
    ({"host": "https://trade.example.org", "ws_port": 8765}, "ws://trade.example.org:8765"),
    ({"host": "https://trade.example.org", "ws_port": 9443}, "ws://trade.example.org:9443"),
    ({"host": "https://trade.example.org:8443/x", "ws_port": 9443}, "ws://trade.example.org:9443"),
    # A host without a scheme, an IPv6 literal, a scheme in upper case.
    ({"host": "example.org:5000/path"}, "ws://example.org:8765"),
    ({"host": "localhost", "ws_port": 9001}, "ws://localhost:9001"),
    ({"host": "http://[::1]:5000"}, "ws://[::1]:8765"),
    ({"host": "[::1]:5000", "ws_port": 9001}, "ws://[::1]:9001"),
    ({"host": "HTTP://Example.org:5000"}, "ws://Example.org:8765"),
    # An explicit ws_url wins, whatever the host and the port are.
    ({"ws_url": "ws://10.0.0.5:9999/feed"}, "ws://10.0.0.5:9999/feed"),
    ({"host": "http://127.0.0.1:5000", "ws_url": "wss://a.example/ws"}, "wss://a.example/ws"),
    ({"host": "https://trade.example.org", "ws_url": "ws://10.0.0.5:9999"}, "ws://10.0.0.5:9999"),
    (
        {"host": "https://trade.example.org", "ws_url": "wss://b.example/feed"},
        "wss://b.example/feed",
    ),
    (
        {"host": "https://trade.example.org", "ws_port": 9443, "ws_url": "wss://c.example/ws"},
        "wss://c.example/ws",
    ),
]
ROW_IDS = [
    " ".join(f"{name}={value}" for name, value in config.items()) or "defaults"
    for config, _ in CONFIGURATIONS
]


@pytest.fixture
def build():
    """Construct clients (no connection is made) and close them afterwards."""
    built = []

    def construct(cls, **config):
        built.append(cls(API_KEY, **config))
        return built[-1]

    yield construct
    for client in built:
        client.close()


@CLIENT_CLASSES
@pytest.mark.parametrize("config, expected", CONFIGURATIONS, ids=ROW_IDS)
def test_ws_url_for_each_configuration(build, cls, config, expected):
    assert build(cls, **config).ws_url == expected


@pytest.mark.parametrize("config, expected", CONFIGURATIONS, ids=ROW_IDS)
def test_unified_api_and_feed_api_derive_the_same_url(build, config, expected):
    assert build(openalgo.api, **config).ws_url == build(FeedAPI, **config).ws_url == expected


@CLIENT_CLASSES
@pytest.mark.parametrize("host", ["http://127.0.0.1:5000", "https://trade.example.org"])
def test_ws_port_attribute_is_8765_when_no_port_was_passed(build, cls, host):
    assert build(cls, host=host).ws_port == 8765


@CLIENT_CLASSES
def test_ws_port_attribute_is_the_port_that_was_passed(build, cls):
    assert build(cls, ws_port=9001).ws_port == 9001


# -- against the formula of 2.0.5 ---------------------------------------------------


def derived_by_2_0_5(host, ws_port=8765):
    """The derivation exactly as openalgo 2.0.5 wrote it (8765 was the default port)."""
    if host.startswith("http://"):
        ws_host = host[7:]
    elif host.startswith("https://"):
        ws_host = host[8:]
    else:
        ws_host = host
    ws_host = ws_host.split("/")[0].split(":")[0]
    return ws_host, f"ws://{ws_host}:{ws_port}"


# Every shape of host that gave a usable URL in 2.0.5.
HTTPS_HOSTS = [
    "https://example.org",
    "https://example.org:8443",
    "https://example.org:8443/openalgo",
    "https://Trade.Example.org/",
]
OTHER_HOSTS = [
    "http://127.0.0.1:5000",
    "http://example.org",
    "http://example.org:5000",
    "http://example.org:5000/openalgo/v2",
    "http://example.org/",
    "http://Example.ORG:5000",
    "example.org",
    "example.org:5000",
    "example.org:5000/path",
    "localhost",
    "192.168.1.10:5000",
]


@pytest.mark.parametrize("host", OTHER_HOSTS)
def test_host_that_is_not_https_gives_the_2_0_5_result_when_no_port_is_passed(host):
    assert derive_ws_url(host) == derived_by_2_0_5(host)


@pytest.mark.parametrize("host", OTHER_HOSTS + HTTPS_HOSTS)
@pytest.mark.parametrize("ws_port", [8765, 9001])
def test_every_host_gives_the_2_0_5_result_when_a_port_is_passed(host, ws_port):
    assert derive_ws_url(host, ws_port) == derived_by_2_0_5(host, ws_port)


@pytest.mark.parametrize("host", HTTPS_HOSTS)
def test_https_host_is_the_one_case_that_differs_from_2_0_5_when_no_port_is_passed(host):
    ws_host, ws_url = derive_ws_url(host)
    assert (ws_host, ws_url) == (derived_by_2_0_5(host)[0], f"wss://{ws_host}/ws")


@pytest.mark.parametrize("host", ["http://[::1", "", "http://", "://", "[", "http://exa mple.org"])
def test_text_that_is_not_a_url_never_raises(host):
    ws_host, ws_url = derive_ws_url(host)
    assert ws_url == f"ws://{ws_host}:8765"

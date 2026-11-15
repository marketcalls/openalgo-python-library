"""The fake OpenAlgo's tests never reach a network: the `offline` fixture refuses and
records every connection and host-name lookup, and the feed finds the real WebSocket
module nowhere. A refused attempt is cleared at the end of the test that makes it on
purpose; any other fails its test.

The addresses tried here are in 192.0.2.0/24, a block kept for documentation that no
host answers on (RFC 5737), never the OpenAlgo server on this computer.
"""

from __future__ import annotations

import socket

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from feed_fakes import NoRealSockets

from openalgo import api
from openalgo import feed as feed_module

NOWHERE = ("192.0.2.1", 9)


def test_opening_a_connection_is_refused_and_recorded(offline):
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(NOWHERE, timeout=1)

    assert offline.attempts == [repr(NOWHERE)]
    offline.attempts.clear()


def test_connecting_a_raw_socket_is_refused_and_recorded(offline):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(ConnectionRefusedError):
            sock.connect(NOWHERE)
    finally:
        sock.close()

    assert offline.attempts == [repr(NOWHERE)]
    offline.attempts.clear()


def test_a_host_name_lookup_is_refused_and_recorded(offline):
    with pytest.raises(ConnectionRefusedError):
        socket.getaddrinfo("openalgo.invalid", 80)

    assert offline.attempts == [repr(("openalgo.invalid", 80))]
    offline.attempts.clear()


@pytest.mark.parametrize(
    ("lookup", "argument"),
    [
        ("gethostbyname", "openalgo.invalid"),
        ("gethostbyname_ex", "openalgo.invalid"),
        ("gethostbyaddr", NOWHERE[0]),
    ],
)
def test_the_older_name_lookups_are_refused_and_recorded(offline, lookup, argument):
    with pytest.raises(ConnectionRefusedError):
        getattr(socket, lookup)(argument)

    assert offline.attempts == [repr(argument)]
    offline.attempts.clear()


def test_an_sdk_client_off_the_fakes_transport_reaches_nothing(offline):
    client = api(api_key="placeholder-key", host=f"http://{NOWHERE[0]}:{NOWHERE[1]}", timeout=1.0)
    try:
        answer = client.ping()
    finally:
        client.close()

    assert answer["error_type"] == "connection_error"
    assert len(offline.attempts) == 1
    offline.attempts.clear()


def test_the_fakes_clients_open_no_connection(fake_openalgo, offline):
    client = fake_openalgo.client()

    client.ping()
    assert client.connect() is True

    assert offline.attempts == []


def test_outside_the_fake_the_feed_finds_no_websocket_module():
    assert feed_module.websocket is NoRealSockets


def test_the_fake_installs_its_proxy_as_the_feeds_websocket_module(fake_openalgo):
    assert feed_module.websocket is fake_openalgo.feed_server

"""Fixtures of the feed client tests. A test module gets them with

    from feed_fixtures import *  # noqa: F403

because pytest finds a fixture by its name in the module that uses it. (They
are not in test/conftest.py, which belongs to the other suites of this folder.)
"""

import pytest
from feed_fakes import (
    API_KEY,
    FakeServer,
    Recorder,
    no_real_sockets,
    no_rest_client,
    retire,
    shorten,
    wait_for,
)

from openalgo import feed as feed_module
from openalgo.feed import FeedAPI

__all__ = [
    "no_real_sockets",
    "no_rest_client",
    "server",
    "make_client",
    "client",
    "recorder",
    "connected",
]


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(feed_module, "websocket", fake)
    yield fake
    # Let every fake socket thread finish.
    fake.answer_close = True
    fake.gate.set()
    for sock in fake.sockets:
        sock.close()


@pytest.fixture
def make_client(server):
    """Build FeedAPI (or openalgo.api) clients wired to the fake server."""
    created = []

    def factory(cls=FeedAPI, key=API_KEY, **kwargs):
        client = cls(key, **kwargs)
        shorten(client, poll=0.002, backoff=(0.01,), backoff_slice=0.005)
        created.append(client)
        return client

    yield factory
    # Nothing may keep a socket thread waiting once the test is over.
    server.answer_close = True
    server.gate.set()
    for client in created:
        retire(client)


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def recorder(client):
    return Recorder(client)


@pytest.fixture
def connected(server, client, recorder):
    """The client, connected and authenticated, with the recorder attached."""
    assert client.connect() is True
    assert wait_for(lambda: recorder.names() == ["connect"])
    return client

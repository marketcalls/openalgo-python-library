"""The pytest fixtures of the fake OpenAlgo. A test module takes them with

    from fake_openalgo.fixtures import *  # noqa: F403

since pytest finds a fixture by its name in the module that uses it.

- `offline` (automatic): any attempt to open a network connection or to look a host
  name up fails the test, both where it is made and at the end of the test (the SDK
  turns the error into an answer, which would hide it).
- `no_real_sockets` (automatic, for the session, from the feed tests): the real
  WebSocket module is never reachable from `openalgo.feed`.
- `fake_openalgo`: a `FakeOpenAlgo` on Monday 2026-11-16 10:15:00 IST in live mode, its
  proxy installed for `openalgo.feed`; `make_fake_openalgo(**options)` builds one with
  other options. At the end the fake's clients are retired and the test fails if the
  fake logged a problem.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from feed_fakes import no_real_sockets

from openalgo import feed as feed_module

from .clock import ist_ns
from .fake import FakeOpenAlgo

#: The fixture's start: Monday 2026-11-16, 10:15:00 IST, a trading day of NSE and NFO.
SESSION_START_NS = ist_ns(2026, 11, 16, 10, 15)


class ConnectionGuard:
    """Stands in for opening a connection: records the attempt and refuses it."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def refuse(self, target: object) -> None:
        """Record an attempt to reach `target` and raise ConnectionRefusedError."""
        self.attempts.append(repr(target))
        raise ConnectionRefusedError(f"a test tried to open a network connection to {target!r}")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[ConnectionGuard]:
    """Refuse every network connection for the test; fail it if one was tried."""
    guard = ConnectionGuard()

    def connect(sock: socket.socket, address: object) -> None:
        guard.refuse(address)

    def create_connection(address: object, *args: Any, **kwargs: Any) -> socket.socket:
        guard.refuse(address)
        raise AssertionError("unreachable: the guard raised")

    def getaddrinfo(host: object, port: object, *args: Any, **kwargs: Any) -> list[Any]:
        # A name lookup would send a DNS query off this computer.
        guard.refuse((host, port))
        raise AssertionError("unreachable: the guard raised")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield guard
    assert guard.attempts == [], f"the test tried to open network connections: {guard.attempts}"


@pytest.fixture
def make_fake_openalgo(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., FakeOpenAlgo]]:
    """Build a `FakeOpenAlgo` (options as its constructor's, `start_ns` defaulting to
    `SESSION_START_NS`) and install its proxy as `openalgo.feed`'s WebSocket module.
    One per test: the last one built owns the feed."""
    built: list[FakeOpenAlgo] = []

    def build(**options: Any) -> FakeOpenAlgo:
        options.setdefault("start_ns", SESSION_START_NS)
        fake = FakeOpenAlgo(**options)
        monkeypatch.setattr(feed_module, "websocket", fake.feed_server)
        built.append(fake)
        return fake

    yield build
    for fake in built:
        fake.close()
    problems = [entry for fake in built for entry in fake.log.problems()]
    assert problems == [], f"the fake OpenAlgo logged problems: {problems}"


@pytest.fixture
def fake_openalgo(make_fake_openalgo: Callable[..., FakeOpenAlgo]) -> FakeOpenAlgo:
    """A `FakeOpenAlgo` in live mode at `SESSION_START_NS`."""
    return make_fake_openalgo()


__all__ = [
    "SESSION_START_NS",
    "ConnectionGuard",
    "fake_openalgo",
    "make_fake_openalgo",
    "no_real_sockets",
    "offline",
]

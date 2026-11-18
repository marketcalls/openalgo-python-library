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
  other options. Either keeps the test offline even in a module that took no other
  fixture. At the end each fake's end checks run (`FakeOpenAlgo.at_test_end`: the
  session invariants of `session_invariants`), its clients are retired, and the test
  fails if a check failed or the fake logged a problem.
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


def guard_connections(monkeypatch: pytest.MonkeyPatch) -> ConnectionGuard:
    """Refuse, for the rest of the test, every connection and every host-name lookup
    (a lookup would send a DNS query off this computer). Returns the guard, which keeps
    the attempts."""
    guard = ConnectionGuard()

    def connect(sock: socket.socket, address: object) -> None:
        guard.refuse(address)

    def refused(target: object, *args: Any, **kwargs: Any) -> Any:
        guard.refuse(target)
        raise AssertionError("unreachable: the guard raised")

    def getaddrinfo(host: object, port: object, *args: Any, **kwargs: Any) -> list[Any]:
        guard.refuse((host, port))
        raise AssertionError("unreachable: the guard raised")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    for lookup in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, lookup, refused)
    return guard


def _check(guard: ConnectionGuard) -> None:
    assert guard.attempts == [], f"the test tried to open network connections: {guard.attempts}"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[ConnectionGuard]:
    """Refuse every network connection and lookup for the test; fail it if one was
    tried."""
    guard = guard_connections(monkeypatch)
    yield guard
    _check(guard)


@pytest.fixture
def make_fake_openalgo(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Callable[..., FakeOpenAlgo]]:
    """Build a `FakeOpenAlgo` (options as its constructor's, `start_ns` defaulting to
    `SESSION_START_NS`) and install its proxy as `openalgo.feed`'s WebSocket module.
    One per test: the last one built owns the feed. The test is offline either way:
    through `offline` when its module took it, else through a guard of its own (a
    fixture is found by name only in a module that imported it)."""
    own_guard = None if "offline" in request.fixturenames else guard_connections(monkeypatch)
    built: list[FakeOpenAlgo] = []

    def build(**options: Any) -> FakeOpenAlgo:
        options.setdefault("start_ns", SESSION_START_NS)
        fake = FakeOpenAlgo(**options)
        monkeypatch.setattr(feed_module, "websocket", fake.feed_server)
        built.append(fake)
        return fake

    yield build
    try:
        for fake in built:
            for check in fake.end_checks:
                check()
    finally:
        for fake in built:
            fake.close()
    problems = [entry for fake in built for entry in fake.log.problems()]
    assert problems == [], f"the fake OpenAlgo logged problems: {problems}"
    if own_guard is not None:
        _check(own_guard)


@pytest.fixture
def fake_openalgo(make_fake_openalgo: Callable[..., FakeOpenAlgo]) -> FakeOpenAlgo:
    """A `FakeOpenAlgo` in live mode at `SESSION_START_NS`."""
    return make_fake_openalgo()


__all__ = [
    "SESSION_START_NS",
    "ConnectionGuard",
    "fake_openalgo",
    "guard_connections",
    "make_fake_openalgo",
    "no_real_sockets",
    "offline",
]

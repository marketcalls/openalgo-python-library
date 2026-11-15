"""A test module that takes only the fake's fixture, not `offline`, is offline all the
same: `make_fake_openalgo` installs the connection guard itself. Nothing here tries a
connection; it checks that the guard stands in for each way of making one.
"""

from __future__ import annotations

import socket

from fake_openalgo.fixtures import fake_openalgo, make_fake_openalgo  # noqa: F401

#: The socket functions as they were before any test ran (this module is imported at
#: collection, when no fixture is active).
UNGUARDED = (
    (socket.socket, "connect", socket.socket.connect),
    (socket.socket, "connect_ex", socket.socket.connect_ex),
    (socket, "create_connection", socket.create_connection),
    (socket, "getaddrinfo", socket.getaddrinfo),
    (socket, "gethostbyname", socket.gethostbyname),
    (socket, "gethostbyname_ex", socket.gethostbyname_ex),
    (socket, "gethostbyaddr", socket.gethostbyaddr),
)


def test_the_fake_alone_guards_every_way_of_connecting(fake_openalgo):  # noqa: F811
    fake_openalgo.client().ping()

    unguarded = [name for owner, name, real in UNGUARDED if getattr(owner, name) is real]
    assert unguarded == []


def test_the_guard_is_gone_after_the_test():
    unguarded = [name for owner, name, real in UNGUARDED if getattr(owner, name) is real]

    assert len(unguarded) == len(UNGUARDED)

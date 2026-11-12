"""disconnect() and reconnect attempts that overlap with something else: a
close the peer never answers, a hook, a data callback, a drop, the
reconnect loop, and a socket that was replaced.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import threading
import time

import pytest
from feed_fakes import (
    AUTH_OK,
    QUIET,
    SBIN,
    WAIT,
    Recorder,
    go_down,
    shorten,
    tick,
    wait_for,
)
from feed_fixtures import *  # noqa: F403

# ---------------------------------------------------------------------------
# disconnect() against a slow peer, a hook, a callback, a drop and the loop
# ---------------------------------------------------------------------------


def test_close_that_the_peer_never_answers_is_still_reported_once_as_clean(
    server, connected, recorder
):
    shorten(connected, disconnect=0.05)
    server.answer_close = False
    server.close_timeout = 0.05
    sock = server.last
    connected.disconnect()
    assert sock.finished.wait(WAIT)  # websocket-client reports the lost socket afterwards
    assert [info["error"] for info in recorder.infos("disconnect")] == [None]


def test_disconnect_called_from_a_hook_does_not_wait_for_its_own_thread(server, make_client):
    client = make_client()
    shorten(client, disconnect=30, connect=0.05, authenticate=0.05)
    recorder = Recorder(client)
    client.on_connect(lambda info: client.disconnect())
    started = time.time()
    client.connect()
    assert wait_for(lambda: recorder.names() == ["connect", "disconnect"])
    assert time.time() - started < WAIT


def test_disconnect_called_from_a_data_callback_fires_the_hook_once(server, connected, recorder):
    connected.subscribe_ltp([SBIN], lambda message: connected.disconnect())
    server.last.deliver(tick(SBIN, 100.0))
    time.sleep(QUIET)
    assert recorder.names() == ["connect", "disconnect"]


def test_disconnect_called_from_a_data_callback_closes_the_socket(server, connected):
    connected.subscribe_ltp([SBIN], lambda message: connected.disconnect())
    server.last.deliver(tick(SBIN, 100.0))
    assert server.last.finished.wait(WAIT)


def test_disconnect_during_a_reconnect_attempt_abandons_that_attempt(server, connected, recorder):
    go_down(server, connected)
    assert wait_for(lambda: len(server.sockets) == 2)  # the attempt is held before it opens
    connected.disconnect()
    server.gate.set()
    assert server.last.finished.wait(WAIT)
    assert (connected.connected, recorder.names()) == (False, ["connect", "disconnect"])


def test_reconnect_attempt_that_was_already_decided_is_refused_after_disconnect(
    server, connected, recorder
):
    at_the_attempt = threading.Event()
    go_on = threading.Event()
    do_connect = connected._do_connect

    def held_on_the_reconnect_thread():
        if threading.current_thread() is not threading.main_thread():
            at_the_attempt.set()
            go_on.wait(WAIT)
        return do_connect()

    connected._do_connect = held_on_the_reconnect_thread
    server.last.drop()
    assert at_the_attempt.wait(WAIT)  # the loop has passed its checks
    connected.disconnect()
    go_on.set()
    time.sleep(QUIET)
    assert (connected.connected, len(server.sockets)) == (False, 1)


def test_disconnect_that_starts_while_a_drop_is_handled_gives_one_consistent_report(
    server, connected, recorder
):
    shorten(connected, disconnect=0.05)
    write_log = connected._log

    def disconnect_while_the_close_is_handled(level, category, message):
        # The close handler prints this line on the WebSocket thread. Run a
        # whole disconnect() on another thread at exactly that point.
        if message.startswith("Disconnected from"):
            other = threading.Thread(target=connected.disconnect)
            other.start()
            other.join(WAIT)
        write_log(level, category, message)

    connected._log = disconnect_while_the_close_is_handled
    server.last.drop()
    outcomes = [(i["requested"], i["will_reconnect"]) for i in recorder.infos("disconnect")]
    assert outcomes in ([(True, False)], [(False, True)])


def test_disconnect_racing_a_drop_is_reported_once_and_consistently(server, make_client):
    for _ in range(10):
        client = make_client()
        recorder = Recorder(client)
        assert client.connect() is True
        dropper = threading.Thread(target=server.last.drop)
        dropper.start()
        client.disconnect()
        dropper.join(WAIT)
        outcomes = [(i["requested"], i["will_reconnect"]) for i in recorder.infos("disconnect")]
        assert outcomes in ([(True, False)], [(False, True)])


# ---------------------------------------------------------------------------
# A socket that was replaced cannot affect the current connection
# ---------------------------------------------------------------------------


@pytest.fixture
def replaced_socket(server, make_client):
    """A client whose first socket opened but never authenticated, then was
    replaced by a second connect() that succeeded. Returns (client, recorder,
    the first socket)."""
    server.auth_reply = None
    client = make_client(auto_reconnect=False)
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    stale = server.last
    server.auth_reply = AUTH_OK
    assert client.connect() is True
    assert wait_for(lambda: recorder.names() == ["connect"])
    return client, recorder, stale


def test_replaced_socket_is_closed(replaced_socket):
    _, _, stale = replaced_socket
    assert stale.finished.wait(WAIT)


def test_close_of_a_replaced_socket_leaves_the_feed_connected(replaced_socket):
    client, _, stale = replaced_socket
    assert stale.finished.wait(WAIT)
    assert (client.connected, client.authenticated) == (True, True)


def test_close_of_a_replaced_socket_fires_no_disconnect_hook(replaced_socket):
    _, recorder, stale = replaced_socket
    assert stale.finished.wait(WAIT)
    assert recorder.names() == ["connect"]


def test_acknowledgement_on_a_replaced_socket_does_not_authenticate_the_feed(server, make_client):
    server.auth_reply = None
    server.answer_close = False  # the replaced socket stays open for a while
    server.close_timeout = 0.2
    client = make_client(auto_reconnect=False)
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    stale = server.last
    server.gate.clear()  # the second attempt hangs before it opens
    second_connect = threading.Thread(target=client.connect)
    second_connect.start()
    assert wait_for(lambda: len(server.sockets) == 2)

    stale.deliver(AUTH_OK)

    assert (client.authenticated, recorder.names()) == (False, [])
    server.gate.set()
    second_connect.join(WAIT)

"""How hooks are delivered: one at a time and in order, on which thread,
outside every SDK lock, and how listeners are registered and removed.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import threading
import time

import pytest
from feed_fakes import (
    QUIET,
    SBIN,
    WAIT,
    Recorder,
    drop_and_recover,
    raise_boom,
    reconnecting_once,
    shorten,
    subscribe_frame,
    tick,
    wait_for,
)
from feed_fixtures import *  # noqa: F403

# ---------------------------------------------------------------------------
# One ordered delivery path
# ---------------------------------------------------------------------------


def numbered(seen, name):
    """A listener that records the event name with its connection number."""
    return lambda info: seen.append((name, info["seq"]))


def test_connection_number_grows_with_every_connection(server, connected, recorder):
    drop_and_recover(server, recorder)
    assert [info["seq"] for _, info in recorder.events] == [1, 1, 2, 2]


def test_listener_that_reconnects_does_not_reorder_events_for_later_listeners(server, make_client):
    client = make_client(auto_reconnect=False)
    seen = []
    client.on_disconnect(reconnecting_once(client))
    client.on_connect(numbered(seen, "connect"))
    client.on_disconnect(numbered(seen, "disconnect"))
    assert client.connect() is True
    server.last.drop()
    assert wait_for(lambda: len(seen) == 3)
    time.sleep(QUIET)
    assert seen == [("connect", 1), ("disconnect", 1), ("connect", 2)]


def test_listener_that_reconnects_keeps_the_order_with_auto_reconnect_on(server, make_client):
    client = make_client()
    seen = []
    client.on_disconnect(reconnecting_once(client))
    client.on_connect(numbered(seen, "connect"))
    client.on_reconnect(numbered(seen, "reconnect"))
    client.on_disconnect(numbered(seen, "disconnect"))
    assert client.connect() is True
    server.last.drop()
    assert wait_for(lambda: len(seen) == 4)
    time.sleep(QUIET)
    assert seen == [("connect", 1), ("disconnect", 1), ("connect", 2), ("reconnect", 2)]


def test_connect_listener_is_not_called_for_a_connection_an_earlier_listener_ended(
    server, make_client
):
    client = make_client(auto_reconnect=False)
    shorten(client, connect=0.05, authenticate=0.05)  # connect() sees the socket go again
    seen = []
    client.on_connect(lambda info: client.disconnect())
    client.on_connect(numbered(seen, "connect"))
    client.on_disconnect(numbered(seen, "disconnect"))
    client.connect()
    assert wait_for(lambda: seen == [("disconnect", 1)])
    time.sleep(QUIET)
    assert seen == [("disconnect", 1)]


def test_disconnect_during_a_slow_connect_hook_is_delivered_after_that_hook(server, make_client):
    client = make_client(auto_reconnect=False)
    shorten(client, disconnect=0.05)
    server.close_timeout = 0.05
    seen = []
    release = threading.Event()

    def slow(info):
        seen.append("connect, first listener starts")
        release.wait(WAIT)
        seen.append("connect, first listener ends")

    client.on_connect(slow)
    client.on_connect(lambda info: seen.append("connect, second listener"))
    client.on_disconnect(lambda info: seen.append("disconnect"))
    client.connect()
    assert wait_for(lambda: len(seen) == 1)
    client.disconnect()
    release.set()
    assert wait_for(lambda: "disconnect" in seen)
    assert seen == [
        "connect, first listener starts",
        "connect, first listener ends",
        "disconnect",
    ]


def test_hooks_never_run_at_the_same_time(server, make_client):
    client = make_client()
    running = []
    overlaps = []

    def hook(*args):
        running.append(1)
        overlaps.append(len(running))
        time.sleep(0.01)
        running.pop()

    recorder = Recorder(client)
    for register in (client.on_connect, client.on_disconnect, client.on_reconnect):
        register(hook)
    client.on_error(hook)
    assert client.connect() is True
    client.subscribe_ltp([SBIN], raise_boom)
    for price in range(3):
        server.last.deliver(tick(SBIN, float(price)))
    drop_and_recover(server, recorder)
    client.disconnect()
    assert set(overlaps) == {1}


# ---------------------------------------------------------------------------
# Threads, locks and registration
# ---------------------------------------------------------------------------


def test_each_hook_runs_on_the_websocket_thread_of_its_connection(server, connected, recorder):
    first = server.last
    drop_and_recover(server, recorder)
    second = server.last
    assert recorder.threads == [
        ("connect", first.thread),
        ("disconnect", first.thread),
        ("connect", second.thread),
        ("reconnect", second.thread),
    ]


def test_ws_thread_is_the_thread_of_the_current_connection(server, connected, recorder):
    drop_and_recover(server, recorder)
    assert connected.ws_thread is server.last.thread


def sdk_locks(client):
    return {
        "lock": client.lock,
        "_state_lock": client._state_lock,
        "_reconnect_lock": client._reconnect_lock,
        "hook registry": client._hooks._lock,
        "error reporter": client._errors._lock,
    }


def test_no_sdk_lock_is_held_while_a_hook_or_callback_runs(server, client):
    held = []

    def probe(*args):
        # Another thread can take a lock only if the thread running this hook
        # does not hold it (one of the locks is re-entrant, so trying from
        # this thread would prove nothing).
        def try_locks():
            for name, lock in sdk_locks(client).items():
                if lock.acquire(blocking=False):
                    lock.release()
                else:
                    held.append(name)

        helper = threading.Thread(target=try_locks)
        helper.start()
        helper.join(WAIT)

    recorder = Recorder(client)
    for register in (client.on_connect, client.on_disconnect, client.on_reconnect):
        register(probe)
    client.on_error(probe)
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN], probe) is True
    assert client.subscribe_quote([SBIN], raise_boom) is True
    server.last.deliver(tick(SBIN, 100.0, 1))
    server.last.deliver(tick(SBIN, 100.0, 2))
    drop_and_recover(server, recorder)  # the hooks of a drop and of a recovery
    client.disconnect()  # the hook path that disconnect() starts
    assert (held, recorder.count("disconnect")) == ([], 2)


def test_a_hook_may_subscribe(server, client):
    results = []
    client.on_connect(lambda info: results.append(client.subscribe_ltp([SBIN])))
    assert client.connect() is True
    assert wait_for(lambda: results == [True])
    assert server.last.frames("subscribe") == [subscribe_frame(1, SBIN)]


def test_listeners_run_in_registration_order(client):
    calls = []
    client.on_connect(lambda info: calls.append("first"))
    client.on_connect(lambda info: calls.append("second"))
    assert client.connect() is True
    assert wait_for(lambda: calls == ["first", "second"])


def test_reconnect_listeners_do_not_share_the_subscription_lists(server, connected, recorder):
    seen = []
    connected.on_reconnect(lambda info: info["subscriptions"]["ltp"].clear())
    connected.on_reconnect(lambda info: seen.append(info["subscriptions"]["ltp"]))
    connected.subscribe_ltp([SBIN])
    drop_and_recover(server, recorder)
    assert wait_for(lambda: seen == [[SBIN]])


@pytest.mark.parametrize("register", ["on_connect", "on_disconnect", "on_reconnect", "on_error"])
def test_registration_returns_the_callback_so_it_works_as_a_decorator(client, register):
    def callback(*args):
        return None

    assert getattr(client, register)(callback) is callback


@pytest.mark.parametrize("register", ["on_connect", "on_disconnect", "on_reconnect", "on_error"])
def test_registering_something_that_is_not_callable_raises_type_error(client, register):
    with pytest.raises(TypeError):
        getattr(client, register)(None)


def test_removed_hook_no_longer_fires(client):
    calls = []

    def hook(info):
        calls.append(info["event"])

    client.on_connect(hook)
    client.remove_hook(hook)
    assert client.connect() is True
    time.sleep(QUIET)
    assert calls == []


def test_remove_hook_tells_whether_the_callback_was_registered(client):
    def hook(info):
        return None

    client.on_disconnect(hook)
    assert (client.remove_hook(hook), client.remove_hook(hook)) == (True, False)


def test_exception_in_a_hook_reaches_the_error_hook(client, recorder):
    client.on_connect(raise_boom)
    assert client.connect() is True
    assert wait_for(lambda: len(recorder.errors) == 1)
    exc, context = recorder.errors[0]
    assert (str(exc), context["kind"], context["callback"], context["message"]["event"]) == (
        "boom",
        "hook",
        "on_connect",
        "connect",
    )


def test_exception_in_a_hook_does_not_stop_the_listeners_after_it(client):
    calls = []
    client.on_connect(raise_boom)
    client.on_connect(lambda info: calls.append("after"))
    client.on_error(lambda exc, context: None)
    assert client.connect() is True
    assert wait_for(lambda: calls == ["after"])

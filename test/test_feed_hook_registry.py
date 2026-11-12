"""Hook listeners and their ordered delivery (openalgo._feed_hooks). No sockets."""

import os
import sys
import threading

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openalgo._feed_hooks import HookRegistry, fresh_copy  # noqa: E402

WAIT = 5.0


class Failures:
    """Collects what the registry reports when a listener raises."""

    def __init__(self):
        self.seen = []

    def __call__(self, event, exc, args):
        self.seen.append((event, exc, args))


@pytest.fixture
def failures():
    return Failures()


@pytest.fixture
def hooks(failures):
    return HookRegistry(("up", "down"), failures)


def emit(hooks, event, *args, **kwargs):
    hooks.post(event, *args, **kwargs)
    hooks.drain()


def test_add_returns_the_callback(hooks):
    def listener(payload):
        return None

    assert hooks.add("up", listener) is listener


def test_add_rejects_something_that_is_not_callable(hooks):
    with pytest.raises(TypeError):
        hooks.add("up", None)


def test_has_tells_whether_an_event_has_listeners(hooks):
    hooks.add("up", print)
    assert (hooks.has("up"), hooks.has("down")) == (True, False)


def test_listeners_are_called_in_registration_order(hooks):
    calls = []
    hooks.add("up", lambda payload: calls.append("first"))
    hooks.add("up", lambda payload: calls.append("second"))
    emit(hooks, "up", {})
    assert calls == ["first", "second"]


def test_nothing_is_delivered_before_drain(hooks):
    calls = []
    hooks.add("up", calls.append)
    hooks.post("up", {})
    assert calls == []


def test_events_are_delivered_in_the_order_they_were_posted(hooks):
    calls = []
    hooks.add("up", lambda payload: calls.append("up"))
    hooks.add("down", lambda payload: calls.append("down"))
    hooks.post("down", {})
    hooks.post("up", {})
    hooks.drain()
    assert calls == ["down", "up"]


def test_event_caused_by_a_listener_waits_until_every_listener_saw_the_current_one(hooks):
    calls = []

    def causes_down(payload):
        calls.append("up seen by the first listener")
        emit(hooks, "down", {})

    hooks.add("up", causes_down)
    hooks.add("up", lambda payload: calls.append("up seen by the second listener"))
    hooks.add("down", lambda payload: calls.append("down"))
    emit(hooks, "up", {})
    assert calls == [
        "up seen by the first listener",
        "up seen by the second listener",
        "down",
    ]


def test_event_posted_by_another_thread_is_delivered_by_the_thread_already_delivering(hooks):
    delivered_on = []
    in_listener = threading.Event()
    release = threading.Event()

    def slow(payload):
        in_listener.set()
        release.wait(WAIT)

    hooks.add("up", slow)
    hooks.add("down", lambda payload: delivered_on.append(threading.current_thread()))
    deliverer = threading.Thread(target=emit, args=(hooks, "up", {}))
    deliverer.start()
    assert in_listener.wait(WAIT)
    emit(hooks, "down", {})  # returns at once: the other thread is delivering
    release.set()
    deliverer.join(WAIT)
    assert delivered_on == [deliverer]


def test_drain_returns_at_once_while_another_thread_is_delivering(hooks):
    in_listener = threading.Event()
    release = threading.Event()
    calls = []

    def slow(payload):
        in_listener.set()
        release.wait(WAIT)

    hooks.add("up", slow)
    hooks.add("down", calls.append)
    deliverer = threading.Thread(target=emit, args=(hooks, "up", {}))
    deliverer.start()
    assert in_listener.wait(WAIT)
    emit(hooks, "down", {})
    delivered_before_the_slow_listener_returned = list(calls)
    release.set()
    deliverer.join(WAIT)
    assert delivered_before_the_slow_listener_returned == []


def test_listener_is_skipped_when_the_event_is_no_longer_current(hooks):
    calls = []
    current = [True]

    def ends_it(payload):
        calls.append("first")
        current[0] = False

    hooks.add("up", ends_it)
    hooks.add("up", lambda payload: calls.append("second"))
    emit(hooks, "up", {}, still_current=lambda: current[0])
    assert calls == ["first"]


def test_event_without_a_currency_check_reaches_every_listener(hooks):
    calls = []
    hooks.add("down", lambda payload: calls.append("first"))
    hooks.add("down", lambda payload: calls.append("second"))
    emit(hooks, "down", {})
    assert calls == ["first", "second"]


def test_each_listener_gets_its_own_copy_of_nested_data(hooks):
    seen = []

    def clears(payload):
        payload["subscriptions"]["ltp"].clear()

    hooks.add("up", clears)
    hooks.add("up", lambda payload: seen.append(payload["subscriptions"]["ltp"]))
    emit(hooks, "up", {"subscriptions": {"ltp": [{"symbol": "SBIN"}]}})
    assert seen == [[{"symbol": "SBIN"}]]


def test_objects_that_are_not_containers_are_passed_as_they_are(hooks):
    error = ConnectionResetError("reset")
    seen = []
    hooks.add("down", lambda payload: seen.append(payload["error"]))
    emit(hooks, "down", {"error": error})
    assert seen[0] is error


def test_fresh_copy_copies_dicts_lists_and_tuples_all_the_way_down():
    original = {"a": [{"b": (1, [2])}]}
    copy = fresh_copy(original)
    copy["a"][0]["b"][1].append(3)
    assert original == {"a": [{"b": (1, [2])}]}


def test_event_with_several_arguments_passes_them_all(hooks):
    seen = []
    hooks.add("down", lambda exc, context: seen.append((exc, context)))
    error = ValueError("boom")
    emit(hooks, "down", error, {"kind": "callback"})
    assert seen == [(error, {"kind": "callback"})]


def test_exception_in_a_listener_is_reported_with_the_event_and_its_arguments(hooks, failures):
    def raises(payload):
        raise ValueError("boom")

    hooks.add("up", raises)
    emit(hooks, "up", {"seq": 1})
    event, exc, args = failures.seen[0]
    assert (event, str(exc), args) == ("up", "boom", ({"seq": 1},))


def test_exception_in_a_listener_does_not_stop_the_listeners_after_it(hooks):
    calls = []

    def raises(payload):
        raise ValueError("boom")

    hooks.add("up", raises)
    hooks.add("up", lambda payload: calls.append("after"))
    emit(hooks, "up", {})
    assert calls == ["after"]


def test_delivery_continues_after_a_listener_raised_a_base_exception(hooks):
    calls = []

    def interrupts(payload):
        raise KeyboardInterrupt

    hooks.add("up", interrupts)
    hooks.add("down", calls.append)
    with pytest.raises(KeyboardInterrupt):
        emit(hooks, "up", {})
    emit(hooks, "down", {"seq": 2})
    assert calls == [{"seq": 2}]


def test_removed_listener_is_not_called(hooks):
    calls = []
    hooks.add("up", calls.append)
    hooks.remove(calls.append)
    emit(hooks, "up", {})
    assert calls == []


def test_remove_tells_whether_the_callback_was_registered(hooks):
    def listener(payload):
        return None

    hooks.add("down", listener)
    assert (hooks.remove(listener), hooks.remove(listener)) == (True, False)


def test_listener_registered_twice_is_called_twice(hooks):
    calls = []
    hooks.add("up", calls.append)
    hooks.add("up", calls.append)
    emit(hooks, "up", {})
    assert len(calls) == 2


def test_registry_lock_is_free_while_a_listener_runs(hooks):
    held = []

    def probe(payload):
        def try_lock():
            if hooks._lock.acquire(blocking=False):
                hooks._lock.release()
            else:
                held.append("registry lock")

        helper = threading.Thread(target=try_lock)
        helper.start()
        helper.join(WAIT)

    hooks.add("up", probe)
    emit(hooks, "up", {})
    assert held == []

"""The connection hooks: when on_connect, on_disconnect and on_reconnect fire
and what they receive.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import threading
import time

from feed_fakes import (
    AUTH_OK,
    INFY,
    NOTHING_REQUESTED,
    QUIET,
    SBIN,
    WAIT,
    Recorder,
    drop_and_recover,
    shorten,
    subscribe_frame,
    wait_for,
)
from feed_fixtures import *  # noqa: F403

# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


def test_connect_hook_fires_once_when_the_server_accepts_the_key(connected, recorder):
    time.sleep(QUIET)
    assert recorder.names() == ["connect"]


def test_connect_hook_receives_the_connection_number_url_broker_and_user(connected, recorder):
    assert recorder.infos("connect") == [
        {
            "event": "connect",
            "seq": 1,
            "url": connected.ws_url,
            "reconnect": False,
            "broker": "fakebroker",
            "user_id": "tester",
        }
    ]


def test_connect_hook_runs_with_the_feed_already_authenticated(connected, recorder):
    assert recorder.flags == [("connect", True, True)]


def test_repeated_acknowledgement_does_not_fire_the_connect_hook_again(server, connected, recorder):
    server.last.deliver(AUTH_OK)
    assert recorder.names() == ["connect"]


def test_connect_hook_waits_for_an_acknowledgement_that_arrives_late(server, make_client):
    server.auth_reply = None
    client = make_client(auto_reconnect=False)
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    assert recorder.names() == []
    server.last.deliver(AUTH_OK)
    assert recorder.names() == ["connect"]


def test_refused_connection_fires_no_hook(server, make_client):
    server.accept = False
    client = make_client(auto_reconnect=False)
    shorten(client, connect=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    assert server.last.finished.wait(WAIT)
    assert recorder.names() == []


def test_first_connection_after_a_refused_connect_is_a_connect_not_a_reconnect(server, make_client):
    server.accept = False
    client = make_client()  # auto_reconnect left on: the SDK keeps trying
    shorten(client, connect=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    server.accept = True
    assert wait_for(lambda: client.authenticated)
    time.sleep(QUIET)
    assert [(name, info["reconnect"]) for name, info in recorder.events] == [("connect", False)]


def test_drop_before_the_acknowledgement_fires_no_disconnect_hook(server, make_client):
    server.auth_reply = None
    client = make_client()
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    assert client.connect() is False  # the socket is open, the key not yet accepted
    server.gate.clear()
    server.last.drop()
    assert (client.connected, recorder.names()) == (False, [])


def test_connection_that_follows_a_drop_before_the_acknowledgement_is_a_connect(
    server, make_client
):
    server.auth_reply = None
    client = make_client()
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    server.auth_reply = AUTH_OK
    server.last.drop()
    assert wait_for(lambda: client.authenticated)
    time.sleep(QUIET)
    assert [(name, info["reconnect"]) for name, info in recorder.events] == [("connect", False)]


def test_connect_while_connected_returns_true_without_a_second_socket(server, connected):
    assert connected.connect() is True
    time.sleep(QUIET)
    assert len(server.sockets) == 1


def test_connect_while_connected_fires_no_hook(connected, recorder):
    connected.connect()
    time.sleep(QUIET)
    assert recorder.names() == ["connect"]


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


def test_drop_fires_the_disconnect_hook_once(server, connected, recorder):
    server.gate.clear()  # keep the reconnect from completing during the test
    server.last.drop()
    time.sleep(QUIET)
    assert recorder.names() == ["connect", "disconnect"]


def test_disconnect_hook_reports_the_transport_error_of_a_drop(server, connected, recorder):
    server.gate.clear()
    error = ConnectionResetError("fake reset")
    server.last.drop(error)
    assert recorder.infos("disconnect") == [
        {
            "event": "disconnect",
            "seq": 1,
            "url": connected.ws_url,
            "code": None,
            "reason": "fake reset",
            "error": error,
            "requested": False,
            "will_reconnect": True,
        }
    ]


def test_disconnect_hook_reports_the_close_code_and_reason_from_the_server(
    server, connected, recorder
):
    server.gate.clear()
    server.last.server_close(1001, "going away")
    info = recorder.infos("disconnect")[0]
    assert (info["code"], info["reason"]) == (1001, "going away")


def test_clean_close_by_the_server_is_not_reported_as_an_error(server, connected, recorder):
    server.gate.clear()
    server.last.server_close(1000, None)  # websocket-client still calls on_error for it
    assert recorder.infos("disconnect")[0]["error"] is None


def test_disconnect_hook_runs_with_the_feed_already_marked_down(server, connected, recorder):
    server.gate.clear()
    server.last.drop()
    assert recorder.flags[-1] == ("disconnect", False, False)


def test_disconnect_call_fires_the_hook_before_it_returns(connected, recorder):
    connected.disconnect()
    assert recorder.names() == ["connect", "disconnect"]


def test_disconnect_call_is_reported_as_requested_and_clean(connected, recorder):
    connected.disconnect()
    assert recorder.infos("disconnect") == [
        {
            "event": "disconnect",
            "seq": 1,
            "url": connected.ws_url,
            "code": None,
            "reason": "disconnect() called",
            "error": None,
            "requested": True,
            "will_reconnect": False,
        }
    ]


def test_disconnect_call_runs_the_hook_on_the_calling_thread(connected, recorder):
    connected.disconnect()
    assert recorder.threads[-1] == ("disconnect", threading.current_thread())


def test_disconnect_call_stops_the_automatic_reconnect(server, connected):
    connected.disconnect()
    time.sleep(QUIET)
    assert len(server.sockets) == 1


def test_disconnect_call_leaves_the_socket_closed(server, connected):
    connected.disconnect()
    assert server.last.finished.is_set()


def test_second_disconnect_call_fires_nothing(connected, recorder):
    connected.disconnect()
    connected.disconnect()
    assert recorder.names() == ["connect", "disconnect"]


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------


def test_recovery_fires_connect_then_reconnect_once_each(server, connected, recorder):
    drop_and_recover(server, recorder)
    time.sleep(QUIET)
    assert recorder.names() == ["connect", "disconnect", "connect", "reconnect"]


def test_connect_hook_of_a_recovery_is_marked_as_a_reconnect(server, connected, recorder):
    drop_and_recover(server, recorder)
    assert [info["reconnect"] for info in recorder.infos("connect")] == [False, True]


def test_reconnect_hook_lists_the_subscriptions_the_server_accepted(server, connected, recorder):
    assert connected.subscribe_ltp([SBIN]) is True
    assert connected.subscribe_orders() is True
    drop_and_recover(server, recorder)
    assert recorder.infos("reconnect")[0]["subscriptions"] == {
        "ltp": [SBIN],
        "quote": [],
        "depth": [],
        "orders": True,
    }


def test_reconnect_hook_leaves_out_an_instrument_the_server_refused(server, connected, recorder):
    connected.subscribe_ltp([SBIN, INFY])
    server.refuse = [INFY]
    drop_and_recover(server, recorder)
    assert recorder.infos("reconnect")[0]["subscriptions"]["ltp"] == [SBIN]


def test_instrument_refused_in_a_replay_reaches_the_error_hook(server, connected, recorder):
    connected.subscribe_ltp([SBIN, INFY])
    server.refuse = [INFY]
    drop_and_recover(server, recorder)
    exc, context = recorder.errors[0]
    assert (exc.code, [entry["symbol"] for entry in context["rejected"]]) == (
        "SUBSCRIBE_REJECTED",
        ["INFY"],
    )


def test_reconnect_hook_fires_when_the_server_does_not_answer_the_replay(
    server, connected, recorder
):
    shorten(connected, replay_ack=0.05)
    connected.subscribe_ltp([SBIN])
    server.answer_subscribe = False
    drop_and_recover(server, recorder)
    assert recorder.infos("reconnect")[0]["subscriptions"] == NOTHING_REQUESTED


def test_replay_the_server_did_not_answer_reaches_the_error_hook(server, connected, recorder):
    shorten(connected, replay_ack=0.05)
    connected.subscribe_ltp([SBIN])
    server.answer_subscribe = False
    drop_and_recover(server, recorder)
    exc, context = recorder.errors[0]
    assert (exc.code, context["unanswered"]["ltp"]) == ("REPLAY_NOT_ACKNOWLEDGED", [SBIN])


def test_reconnect_hook_runs_after_the_replay_was_sent(server, connected, recorder):
    replayed_when_hook_ran = []
    connected.on_reconnect(
        lambda info: replayed_when_hook_ran.extend(server.last.frames("subscribe"))
    )
    assert connected.subscribe_ltp([SBIN]) is True
    drop_and_recover(server, recorder)
    assert replayed_when_hook_ran == [subscribe_frame(1, SBIN)]


def test_three_drops_are_each_reported_exactly_once(server, connected, recorder):
    for _ in range(3):
        drop_and_recover(server, recorder)
    time.sleep(QUIET)
    assert recorder.names() == ["connect"] + ["disconnect", "connect", "reconnect"] * 3


def test_failed_reconnect_attempts_fire_no_hooks(server, connected, recorder):
    shorten(connected, connect=0.02)
    server.accept = False
    server.last.drop()
    assert wait_for(lambda: len(server.sockets) >= 4)  # several refused attempts
    assert recorder.names() == ["connect", "disconnect"]


def test_feed_recovers_once_the_server_accepts_connections_again(server, connected, recorder):
    shorten(connected, connect=0.02)
    server.accept = False
    server.last.drop()
    assert wait_for(lambda: len(server.sockets) >= 4)
    server.accept = True
    assert wait_for(lambda: recorder.names() == ["connect", "disconnect", "connect", "reconnect"])

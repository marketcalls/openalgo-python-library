"""Shared helpers of the feed client tests: an in-process fake of the part of
websocket-client that FeedAPI uses, a scripted fake server, builders for server
messages, and the two fixtures that keep every test off the network.

Nothing here opens a socket or contacts a server, and the API key is a
placeholder string. The fixtures that build clients are in feed_fixtures.py.
"""

import json
import os
import queue
import sys
import threading
import time

import pytest
from websocket import WebSocketConnectionClosedException

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import openalgo  # noqa: E402
from openalgo import base as base_module  # noqa: E402
from openalgo import feed as feed_module  # noqa: E402
from openalgo.feed import FeedAPI  # noqa: E402

API_KEY = "PLACEHOLDER-API-KEY-NOT-A-REAL-KEY"
AUTHENTICATE = {"action": "authenticate", "api_key": API_KEY}
AUTH_OK = {"type": "auth", "status": "success", "broker": "fakebroker", "user_id": "tester"}
KEY_REFUSED = {"status": "error", "code": "AUTHENTICATION_ERROR", "message": "Invalid API key"}
BROKER_ERROR = {"status": "error", "code": "BROKER_ERROR", "message": "Broker adapter not found"}

# Upper bound for every wait. The awaited condition normally holds within a
# few milliseconds; the bound only matters when a test is about to fail.
WAIT = 5.0
# Pause used before asserting that nothing more happened.
QUIET = 0.05

SBIN = {"exchange": "NSE", "symbol": "SBIN"}
INFY = {"exchange": "NSE", "symbol": "INFY"}
TCS = {"exchange": "NSE", "symbol": "TCS"}
NIFTY = {"exchange": "NSE_INDEX", "symbol": "NIFTY"}

NOTHING_REQUESTED = {"ltp": [], "quote": [], "depth": [], "orders": False}
KINDS = ["ltp", "quote", "depth", "order"]
MODE_LABELS = {1: "LTP", 2: "Quote", 3: "Depth"}
CLIENT_CLASSES = pytest.mark.parametrize("cls", [FeedAPI, openalgo.api], ids=["FeedAPI", "api"])


# ---------------------------------------------------------------------------
# The shared fake: the part of websocket-client 1.9 that FeedAPI uses
# ---------------------------------------------------------------------------


def closed_by_peer(code, reason):
    """The exception websocket-client hands to on_error when the peer sends a
    close frame. It does this for a clean close too."""
    text = (
        "Connection closed normally (code 1000)"
        if code == 1000
        else f"Connection closed (code {code})"
    )
    error = WebSocketConnectionClosedException(f"{text}: {reason}" if reason else text)
    error.status_code = code
    error.reason = reason
    return error


class FakeSocket:
    """Stand-in for websocket.WebSocketApp. One instance per connection attempt."""

    def __init__(self, server, url, on_message, on_error, on_open, on_close):
        self.server = server
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self.on_open = on_open
        self.on_close = on_close
        self.sent = []  # decoded frames the SDK sent on this socket
        self.run_kwargs = None  # keyword arguments given to run_forever
        self.thread = None  # the thread that ran run_forever
        self.finished = threading.Event()  # run_forever returned, on_close included
        self._inbox = queue.Queue()
        self._alive = False
        self._close_answered = threading.Event()

    # -- the websocket-client surface -------------------------------------

    def run_forever(self, **kwargs):
        self.run_kwargs = kwargs
        self.thread = threading.current_thread()
        self.server.sockets.append(self)
        code = reason = None
        try:
            self.server.gate.wait()
            if not self.server.accept:
                self.on_error(self, ConnectionRefusedError("fake: connection refused"))
                return
            self._alive = True
            self.on_open(self)
            while True:
                kind, payload, done = self._inbox.get()
                if kind == "message":
                    self.on_message(self, payload)
                    done.set()
                    continue
                self._alive = False
                if kind == "drop":
                    self.on_error(self, payload)
                elif kind == "peer close":
                    code, reason = payload
                    self.on_error(self, closed_by_peer(code, reason))
                else:  # this side called close() and the peer answered
                    code = 1000
                return
        finally:
            self._alive = False
            self._close_answered.set()  # nothing is left to wait for
            self.on_close(self, code, reason)
            self.finished.set()

    def send(self, data):
        if not self._alive:
            raise WebSocketConnectionClosedException("Connection is already closed.")
        frame = json.loads(data)
        self.sent.append(frame)
        reply = self.server.reply_to(frame)
        if reply is not None:
            self._inbox.put(("message", json.dumps(reply), threading.Event()))

    def close(self, **kwargs):
        """Like websocket-client: send a close frame and wait for the peer's
        answer, for at most the close timeout. Called on the socket's own
        thread (from a callback) it cannot wait for that thread."""
        if not self._alive:
            return
        if not self.server.answer_close:
            # No answer ever comes: websocket-client then shuts the socket down.
            time.sleep(self.server.close_timeout)
            lost = WebSocketConnectionClosedException("Connection to remote host was lost.")
            self._inbox.put(("drop", lost, threading.Event()))
            return
        self._inbox.put(("local close", None, threading.Event()))
        if threading.current_thread() is not self.thread:
            self._close_answered.wait(self.server.close_timeout)

    # -- what a test drives -----------------------------------------------

    def deliver(self, message):
        """Push one server message and return after the SDK has processed it."""
        self.deliver_raw(json.dumps(message))

    def deliver_raw(self, text):
        done = threading.Event()
        self._inbox.put(("message", text, done))
        assert done.wait(WAIT), "the fake socket did not process the message"

    def drop(self, error=None):
        """Transport failure: on_error, then on_close without a close code."""
        error = error or ConnectionResetError("fake: connection reset")
        self._inbox.put(("drop", error, threading.Event()))
        assert self.finished.wait(WAIT), "the fake socket did not close"

    def server_close(self, code, reason):
        """The server sends a close frame with a code and a reason."""
        self._inbox.put(("peer close", (code, reason), threading.Event()))
        assert self.finished.wait(WAIT), "the fake socket did not close"

    def break_transport(self):
        """The transport is gone but websocket-client has not noticed yet:
        send() fails while no close event has been delivered."""
        self._alive = False

    def frames(self, action):
        return [frame for frame in self.sent if frame.get("action") == action]


class FakeServer:
    """Scripted OpenAlgo WebSocket proxy.

    Installed in place of the websocket module inside openalgo.feed, so every
    websocket.WebSocketApp(...) call in the SDK builds a FakeSocket. It answers
    frames the way websocket_proxy/server.py does.
    """

    def __init__(self):
        self.sockets = []  # every connection attempt that was started, oldest first
        self.accept = True  # False: new connections are refused
        self.auth_reply = AUTH_OK  # None: the authenticate frame gets no reply
        self.answer_subscribe = True  # False: subscribe frames get no reply
        self.refuse = []  # instruments the server does not accept
        self.answer_close = True  # False: the peer never answers a close frame
        self.close_timeout = 1.0  # how long close() waits for that answer
        self.gate = threading.Event()  # cleared: new connections hang before opening
        self.gate.set()

    def WebSocketApp(
        self, url, on_message=None, on_error=None, on_open=None, on_close=None, **kwargs
    ):
        """Named as in websocket-client, because the SDK calls it by that name."""
        return FakeSocket(self, url, on_message, on_error, on_open, on_close)

    @property
    def last(self):
        return self.sockets[-1]

    def reply_to(self, frame):
        action = frame.get("action")
        if action == "authenticate":
            return self.auth_reply
        if not self.answer_subscribe:
            return None
        if action == "subscribe":
            entries = [self._entry(instrument, frame["mode"]) for instrument in frame["symbols"]]
            partial = any(entry["status"] == "error" for entry in entries)
            return {
                "type": "subscribe",
                "status": "partial" if partial else "success",
                "subscriptions": entries,
                "message": "Subscription processing complete",
                "broker": "fakebroker",
            }
        if action == "unsubscribe":
            return {
                "type": "unsubscribe",
                "status": "success",
                "message": "Unsubscription processing complete",
                "successful": [dict(symbol, status="success") for symbol in frame["symbols"]],
                "failed": [],
                "broker": "fakebroker",
            }
        if action in ("subscribe_orders", "unsubscribe_orders"):
            return {"type": action, "status": "success", "message": "ok"}
        return None

    def _entry(self, instrument, mode):
        base = {"symbol": instrument["symbol"], "exchange": instrument["exchange"]}
        if base in [{"symbol": r["symbol"], "exchange": r["exchange"]} for r in self.refuse]:
            return dict(base, status="error", message="Subscription failed", broker="fakebroker")
        return dict(base, status="success", mode=MODE_LABELS[mode], depth=5, broker="fakebroker")


class NoRealSockets:
    """What openalgo.feed finds as its websocket module whenever no test has
    its fake installed. A thread that outlives its test can then never open a
    real socket: the attempt fails inside the SDK instead."""

    attempts = []  # the name of every thread that tried

    @classmethod
    def WebSocketApp(cls, *args, **kwargs):
        """Named as in websocket-client, because the SDK calls it by that name."""
        cls.attempts.append(threading.current_thread().name)
        raise AssertionError("a real WebSocket was about to be opened outside a test")


class NoHttpClient:
    """Stands in for httpx.Client: a REST call from a test fails at once."""

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def post(self, *args, **kwargs):
        raise AssertionError("these tests must not make REST calls")

    get = post


class NoHttp:
    """Stands in for the httpx module inside openalgo.base."""

    Client = NoHttpClient

    @staticmethod
    def Limits(**kwargs):
        """Named as in httpx, because BaseAPI calls it by that name."""
        return None


class Recorder:
    """Registers every hook on a client and records what fires, in order."""

    def __init__(self, client):
        self.client = client
        self.events = []  # (name, info)
        self.errors = []  # (exception, context)
        self.threads = []  # (name, thread the hook ran on)
        self.flags = []  # (name, connected, authenticated) as seen inside the hook
        client.on_connect(self._recording("connect"))
        client.on_disconnect(self._recording("disconnect"))
        client.on_reconnect(self._recording("reconnect"))
        client.on_error(self._record_error)

    def _recording(self, name):
        def hook(info):
            self.threads.append((name, threading.current_thread()))
            self.flags.append((name, self.client.connected, self.client.authenticated))
            self.events.append((name, info))

        return hook

    def _record_error(self, exc, context):
        self.threads.append(("error", threading.current_thread()))
        self.errors.append((exc, context))

    def names(self):
        return [name for name, _ in self.events]

    def infos(self, name):
        return [info for event, info in self.events if event == name]

    def count(self, name):
        return self.names().count(name)


def wait_for(predicate, timeout=WAIT):
    """Poll until predicate() is true. Returns at once when it already is."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def shorten(client, **seconds):
    """Give the client shorter waiting times, through its one test seam."""
    client._timing = client._timing._replace(**seconds)


def retire(client):
    """End a client for good when its test is over: it must not reconnect by
    itself, and none of its threads may outlive the test."""
    client.auto_reconnect = False
    client.disconnect()
    for thread in (client._reconnect_thread, client.ws_thread):
        if thread is not None:
            thread.join(WAIT)
            assert not thread.is_alive(), "a feed thread outlived its test"
    client.close()
    assert not client.connected, "a listener brought the client back during teardown"


def reconnecting_once(client):
    """A disconnect listener that calls connect() again, the first time only
    (so that the disconnect at the end of the test stays a disconnect)."""
    calls = []

    def listener(info):
        calls.append(info)
        if len(calls) == 1:
            client.connect()

    return listener


def tick(instrument, ltp, mode=1):
    """A market_data message as the server sends it."""
    data = {"ltp": ltp, "timestamp": 1700000000000}
    if mode == 2:
        data.update({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10})
    if mode == 3:
        data["depth"] = {
            "buy": [{"price": ltp - 1, "quantity": 5, "orders": 1}],
            "sell": [{"price": ltp + 1, "quantity": 7, "orders": 2}],
        }
    return {
        "type": "market_data",
        "exchange": instrument["exchange"],
        "symbol": instrument["symbol"],
        "mode": mode,
        "data": data,
    }


def order_update(orderid):
    return {
        "type": "order_update",
        "orderid": orderid,
        "symbol": "SBIN",
        "exchange": "NSE",
        "order_status": "complete",
        "quantity": 1,
        "filled_quantity": 1,
        "average_price": 100.0,
    }


def subscribe_frame(mode, *instruments):
    return {"action": "subscribe", "symbols": list(instruments), "mode": mode, "depth": 5}


def subscribe_kind(client, kind, callback=None):
    """Subscribe SBIN (or the order stream) for one of the four data callbacks."""
    if kind == "order":
        return client.subscribe_orders(callback)
    return getattr(client, f"subscribe_{kind}")([SBIN], callback)


def message_for(kind, value):
    """A server message that reaches the callback of the given kind."""
    if kind == "order":
        return order_update(str(value))
    return tick(SBIN, float(value), KINDS.index(kind) + 1)


def go_down(server, client):
    """Drop the current socket and hold the automatic reconnect before it opens."""
    server.gate.clear()
    server.last.drop()
    assert client.connected is False


def come_back(server, recorder):
    """Release the held reconnect and wait until the feed has been restored."""
    restored = recorder.count("reconnect") + 1
    server.gate.set()
    assert wait_for(lambda: recorder.count("reconnect") == restored)


def drop_and_recover(server, recorder):
    """One full cycle: the socket drops and the automatic reconnect restores it."""
    restored = recorder.count("reconnect") + 1
    server.last.drop()
    assert wait_for(lambda: recorder.count("reconnect") == restored)


def raise_boom(message):
    raise ValueError("boom")


def feed_log(caplog):
    return [record for record in caplog.records if record.name == "openalgo.feed"]


# ---------------------------------------------------------------------------
# Fixtures that keep the tests off the network (see feed_fixtures.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def no_real_sockets():
    """Take the real websocket module away from openalgo.feed for the rest of
    the test session. It is deliberately not put back: the session ends right
    after, and a stray thread must not find it again. At the end, fail if any
    thread outlived its test far enough to try."""
    feed_module.websocket = NoRealSockets
    yield
    assert NoRealSockets.attempts == [], "a feed thread outlived its test"


@pytest.fixture(autouse=True)
def no_rest_client(monkeypatch):
    """BaseAPI builds an httpx.Client for REST. Replace it, so no test can reach
    the network and no time goes into building TLS contexts."""
    monkeypatch.setattr(base_module, "httpx", NoHttp)

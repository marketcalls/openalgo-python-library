"""The WebSocket feed against a running OpenAlgo server (read-only, needs the MCX session).

One feed session runs once for the module (`feed_run`): connect with every hook
registered, subscribe LTP and quote for three near-month MCX futures, collect ticks,
unsubscribe one of them, then disconnect. The tests read what it recorded.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from live_support import live_only, near_month_future, new_client, session_is_open

from openalgo import api

pytestmark = live_only

UNDERLYINGS = ("CRUDEOIL", "GOLDM", "NATURALGAS")
TICK_WINDOW_SECONDS = 30
# Frames already on the wire when the unsubscribe is sent may still arrive.
UNSUBSCRIBE_GRACE_SECONDS = 2
SILENCE_WINDOW_SECONDS = 10
DISCONNECT_SETTLE_SECONDS = 1


class _Records(logging.Handler):
    """Keeps the records of the openalgo.feed logger."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass
class FeedRun:
    """What one feed session recorded."""

    instruments: list[dict]
    silenced: dict
    events: dict[str, list] = field(default_factory=lambda: {"connect": [], "disconnect": []})
    errors: list = field(default_factory=list)
    ticks: dict[tuple[str, str], list[tuple[float, dict]]] = field(default_factory=dict)
    snapshots: dict[str, Any] = field(default_factory=dict)
    connected: bool = False
    unsubscribed_at: float = 0.0
    disconnected_at: float = 0.0
    log_records: list = field(default_factory=list)

    def tick_times(self, mode: str, symbol: str, after: float = 0.0) -> list[float]:
        return [stamp for stamp, _ in self.ticks.get((mode, symbol), []) if stamp > after]


def _record_ticks(run: FeedRun, mode: str, lock: threading.Lock):
    def on_data(message: dict) -> None:
        with lock:
            run.ticks.setdefault((mode, message["symbol"]), []).append((time.time(), message))

    return on_data


def _run_session(feed: Any, run: FeedRun) -> None:
    lock = threading.Lock()
    feed.on_connect(run.events["connect"].append)
    feed.on_disconnect(run.events["disconnect"].append)
    feed.on_error(lambda error, context: run.errors.append((error, context)))
    run.connected = feed.connect()
    if not run.connected:
        return
    feed.subscribe_ltp(run.instruments, _record_ticks(run, "ltp", lock))
    feed.subscribe_quote(run.instruments, _record_ticks(run, "quote", lock))
    run.snapshots["subscribed"] = feed.get_subscriptions()
    time.sleep(TICK_WINDOW_SECONDS)
    run.snapshots["ltp"] = feed.get_ltp()
    run.snapshots["quote"] = feed.get_quotes()
    run.unsubscribed_at = time.time()
    feed.unsubscribe_ltp([run.silenced])
    feed.unsubscribe_quote([run.silenced])
    run.snapshots["after_unsubscribe"] = feed.get_subscriptions()
    run.snapshots["silenced_ltp"] = feed.get_ltp(run.silenced["exchange"], run.silenced["symbol"])
    time.sleep(UNSUBSCRIBE_GRACE_SECONDS + SILENCE_WINDOW_SECONDS)
    run.disconnected_at = time.time()
    feed.disconnect()
    time.sleep(DISCONNECT_SETTLE_SECONDS)
    run.snapshots["after_disconnect"] = feed.get_subscriptions()


@pytest.fixture(scope="module")
def feed_run(client) -> FeedRun:
    if not session_is_open("MCX"):
        pytest.skip("the MCX session is not open: no ticks to receive")
    rows = [near_month_future(client, name, "MCX") for name in UNDERLYINGS]
    instruments = [{"exchange": "MCX", "symbol": row["symbol"]} for row in rows]
    run = FeedRun(instruments=instruments, silenced=instruments[-1])
    handler = _Records()
    logger = logging.getLogger("openalgo.feed")
    logger.addHandler(handler)
    feed = new_client()
    try:
        _run_session(feed, run)
    finally:
        logger.removeHandler(handler)
        feed.disconnect()
        feed.close()
    run.log_records = handler.records
    assert run.connected, "connect() did not authenticate with the feed"
    return run


def test_the_feed_url_of_a_local_host_is_the_default_port():
    local = api(api_key="not-a-key", host="http://127.0.0.1:5000")
    try:
        assert local.ws_url == "ws://127.0.0.1:8765"
    finally:
        local.close()


def test_connect_fires_once_with_the_broker_and_the_user(feed_run):
    (event,) = feed_run.events["connect"]
    assert event["event"] == "connect"
    assert event["reconnect"] is False
    assert event["broker"]
    assert event["user_id"]


@pytest.mark.parametrize("mode", ["ltp", "quote"])
def test_every_subscribed_future_ticks(feed_run, mode):
    silent = [
        entry["symbol"]
        for entry in feed_run.instruments
        if not feed_run.tick_times(mode, entry["symbol"])
    ]
    assert silent == []


def test_ltp_ticks_carry_a_price_and_a_timestamp(feed_run):
    symbol = feed_run.instruments[0]["symbol"]
    _, message = feed_run.ticks[("ltp", symbol)][-1]
    assert message["mode"] == 1
    assert message["data"]["ltp"] > 0
    assert message["data"]["timestamp"] > 0


@pytest.mark.parametrize("snapshot", ["ltp", "quote"])
def test_the_polled_snapshot_holds_every_subscribed_future(feed_run, snapshot):
    polled = feed_run.snapshots[snapshot][snapshot].get("MCX", {})
    assert sorted(polled) == sorted(entry["symbol"] for entry in feed_run.instruments)


def test_get_subscriptions_lists_what_was_requested(feed_run):
    requested = feed_run.snapshots["subscribed"]
    expected = sorted(entry["symbol"] for entry in feed_run.instruments)
    assert sorted(entry["symbol"] for entry in requested["ltp"]) == expected
    assert sorted(entry["symbol"] for entry in requested["quote"]) == expected
    assert requested["depth"] == []
    assert requested["orders"] is False


def test_an_unsubscribed_future_leaves_the_requested_set(feed_run):
    requested = feed_run.snapshots["after_unsubscribe"]
    silenced = feed_run.silenced
    assert silenced not in requested["ltp"]
    assert silenced not in requested["quote"]
    assert feed_run.snapshots["silenced_ltp"] == {"ltp": {}}


@pytest.mark.parametrize("mode", ["ltp", "quote"])
def test_an_unsubscribed_future_stops_ticking_while_the_others_go_on(feed_run, mode):
    quiet_from = feed_run.unsubscribed_at + UNSUBSCRIBE_GRACE_SECONDS
    still_subscribed = feed_run.instruments[0]["symbol"]
    assert feed_run.tick_times(mode, feed_run.silenced["symbol"], after=quiet_from) == []
    assert feed_run.tick_times(mode, still_subscribed, after=quiet_from)


def test_disconnect_fires_once_as_requested(feed_run):
    (event,) = feed_run.events["disconnect"]
    assert event["requested"] is True
    assert event["will_reconnect"] is False
    assert event["error"] is None


def test_disconnect_forgets_the_subscriptions_and_stops_the_ticks(feed_run):
    assert feed_run.snapshots["after_disconnect"] == {
        "ltp": [],
        "quote": [],
        "depth": [],
        "orders": False,
    }
    late = [
        stamp
        for ticks in feed_run.ticks.values()
        for stamp, _ in ticks
        if stamp > feed_run.disconnected_at + DISCONNECT_SETTLE_SECONDS
    ]
    assert late == []


def test_no_error_reached_the_error_hooks_or_the_feed_logger(feed_run):
    assert feed_run.errors == []
    assert [record.getMessage() for record in feed_run.log_records] == []


def test_quote_ticks_carry_the_average_traded_price(feed_run):
    symbol = feed_run.instruments[0]["symbol"]
    _, message = feed_run.ticks[("quote", symbol)][-1]
    assert message["data"]["avg_trade_price"] > 0

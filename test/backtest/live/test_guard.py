"""The mode guard of a live session: the mode adopted at start, the checks of every order
response and order update, the polled and pre-entry reads of OpenAlgo's analyzer switch,
failed reads, the breach and its alert, the banner, and semi-auto (08-live-trading
section 3; D21, D30).

The fake OpenAlgo below answers `analyzerstatus` as the server does
(services/analyzer_service.py) and fails the test if the guard calls anything else; the
responses are shaped as the server's order services and the SDK's own errors give them.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import FrozenInstanceError

import pytest

from openalgo.backtest._live._types import AlertKind, Mode, OrderStatus, OrderUpdate
from openalgo.backtest._live.guard import ModeBreach, ModeGuard, SeenIn
from openalgo.backtest.errors import ConfigError, LiveError, ModeChangedError

# 2026-11-16 09:15:00 IST is 03:45:00 UTC.
SESSION_OPEN_NS = int(dt.datetime(2026, 11, 16, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
SECOND_NS = 10**9

# The SDK's answer when the server cannot be reached (openalgo/account.py).
UNREACHABLE = {
    "status": "error",
    "message": "Failed to connect to the server. Please check if the server is running.",
    "error_type": "connection_error",
}
# Order responses as the server's services give them.
SANDBOX_PLACED = {"status": "success", "orderid": "26111600000001", "mode": "analyze"}
LIVE_PLACED = {"status": "success", "orderid": "1100000012345"}
SANDBOX_REJECTED = {"status": "error", "message": "Insufficient funds", "mode": "analyze"}
QUEUED_IN_ACTION_CENTER = {
    "status": "success",
    "message": "Order queued for approval in Action Center",
    "mode": "semi_auto",
    "pending_order_id": 41,
}
TIMED_OUT = {
    "status": "error",
    "message": "Request timed out. The server took too long to respond.",
    "error_type": "timeout_error",
}


class ManualClock:
    """The present in UTC nanoseconds, moved by hand."""

    def __init__(self) -> None:
        self.ns = SESSION_OPEN_NS

    def __call__(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += round(seconds * SECOND_NS)


class FakeOpenAlgo:
    """OpenAlgo's analyzer switch. `analyze_mode` is the switch; `down` makes the read
    answer as the SDK does when the server cannot be reached; `answer` overrides it."""

    def __init__(self, analyze_mode: bool) -> None:
        self.analyze_mode = analyze_mode
        self.down = False
        self.answer: dict | None = None
        self.reads = 0

    def analyzerstatus(self) -> dict:
        self.reads += 1
        if self.answer is not None:
            return self.answer
        if self.down:
            return UNREACHABLE
        return {
            "status": "success",
            "data": {
                "mode": "analyze" if self.analyze_mode else "live",
                "analyze_mode": self.analyze_mode,
                "total_logs": 7,
            },
        }

    def __getattr__(self, name: str):
        raise AssertionError(f"the mode guard called {name}; it may call only analyzerstatus")


class RecordedAlerts:
    """The alerts sent, as (kind, text)."""

    def __init__(self) -> None:
        self.sent: list[tuple[AlertKind, str]] = []

    def send(self, kind: AlertKind, text: str) -> None:
        self.sent.append((kind, text))

    def of(self, kind: AlertKind) -> list[str]:
        return [text for sent_kind, text in self.sent if sent_kind is kind]


def started_guard(session_mode: Mode, poll_seconds: float = 5.0):
    """A guard started with the switch in `session_mode`, and its fakes."""
    server = FakeOpenAlgo(analyze_mode=session_mode is Mode.SANDBOX)
    clock = ManualClock()
    alerts = RecordedAlerts()
    guard = ModeGuard(server, now=clock, alerts=alerts, poll_seconds=poll_seconds)
    guard.start()
    return guard, server, clock, alerts


def update(order_id: str, mode: Mode) -> OrderUpdate:
    """An order update of a filled 10-share buy, handled in `mode`."""
    return OrderUpdate(
        order_id=order_id,
        symbol="SBIN",
        exchange="NSE",
        action="BUY",
        quantity=10,
        filled_quantity=10,
        pending_quantity=0,
        average_price=812.5,
        status=OrderStatus.COMPLETE,
        rejection_reason="",
        mode=mode,
        ts_recv_ns=SESSION_OPEN_NS,
        source="stream",
    )


def flip(server: FakeOpenAlgo) -> None:
    """The user flips the analyzer switch."""
    server.analyze_mode = not server.analyze_mode


# Start: the mode is read from the switch, never chosen in code.


@pytest.mark.parametrize(("analyze_mode", "mode"), [(True, Mode.SANDBOX), (False, Mode.LIVE)])
def test_start_adopts_the_mode_of_the_analyzer_switch(analyze_mode, mode):
    guard = ModeGuard(FakeOpenAlgo(analyze_mode), now=ManualClock())
    assert guard.start() is mode


def test_start_raises_live_error_telling_to_check_openalgo_when_it_cannot_be_reached():
    server = FakeOpenAlgo(analyze_mode=True)
    server.down = True
    with pytest.raises(LiveError, match="Check that OpenAlgo is running"):
        ModeGuard(server, now=ManualClock()).start()


@pytest.mark.parametrize(
    "answer",
    [
        {"status": "error", "message": "Invalid openalgo apikey"},
        {"status": "success"},
        {"status": "success", "data": {"mode": "live"}},
        {"status": "success", "data": {"analyze_mode": "false"}},
    ],
    ids=["rejected key", "no data", "no analyze_mode", "analyze_mode not a boolean"],
)
def test_start_raises_live_error_when_the_answer_does_not_give_the_mode(answer):
    server = FakeOpenAlgo(analyze_mode=False)
    server.answer = answer
    with pytest.raises(LiveError, match="did not report its mode"):
        ModeGuard(server, now=ManualClock()).start()


def test_start_twice_raises_live_error():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    with pytest.raises(LiveError, match="started already"):
        guard.start()


@pytest.mark.parametrize(
    "use",
    [
        lambda guard: guard.poll(),
        lambda guard: guard.before_entry(),
        lambda guard: guard.check_response(LIVE_PLACED),
        lambda guard: guard.check_update(update("1", Mode.LIVE)),
        lambda guard: guard.note_order_sent("1"),
        lambda guard: guard.banner("zerodha"),
    ],
    ids=["poll", "before_entry", "check_response", "check_update", "note_order_sent", "banner"],
)
def test_the_guard_refuses_to_be_used_before_start(use):
    guard = ModeGuard(FakeOpenAlgo(analyze_mode=True), now=ManualClock())
    with pytest.raises(LiveError, match="call start"):
        use(guard)


@pytest.mark.parametrize("poll_seconds", [0, -5.0, math.nan, math.inf])
def test_a_poll_interval_that_is_not_a_positive_number_of_seconds_raises_config_error(
    poll_seconds,
):
    with pytest.raises(ConfigError, match="poll_seconds"):
        ModeGuard(FakeOpenAlgo(analyze_mode=True), now=ManualClock(), poll_seconds=poll_seconds)


# The banner.


def test_the_sandbox_banner_says_orders_are_simulated():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    assert guard.banner("zerodha") == "OpenAlgo is in SANDBOX mode: orders are simulated"


def test_the_live_banner_names_the_broker_the_orders_go_to():
    guard, _, _, _ = started_guard(Mode.LIVE)
    assert guard.banner("zerodha") == "OpenAlgo is in LIVE mode: orders go to zerodha"


def test_a_live_banner_without_a_broker_raises_config_error():
    guard, _, _, _ = started_guard(Mode.LIVE)
    with pytest.raises(ConfigError, match="broker"):
        guard.banner(" ")


# Order responses.


@pytest.mark.parametrize(
    ("session_mode", "response"),
    [
        (Mode.SANDBOX, SANDBOX_PLACED),
        (Mode.SANDBOX, SANDBOX_REJECTED),
        (Mode.LIVE, LIVE_PLACED),
    ],
    ids=["sandbox order", "sandbox reject", "live order"],
)
def test_a_response_in_the_session_mode_leaves_the_guard_open(session_mode, response):
    guard, _, _, alerts = started_guard(session_mode)
    guard.check_response(response)
    assert not guard.locked
    assert alerts.sent == []


@pytest.mark.parametrize(
    "response", [SANDBOX_PLACED, SANDBOX_REJECTED], ids=["sandbox order", "sandbox reject"]
)
def test_a_response_handled_in_the_sandbox_locks_a_live_session(response):
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.check_response(response)
    assert guard.locked
    assert guard.breach.seen_mode is Mode.SANDBOX


def test_a_successful_order_without_the_sandbox_marker_locks_a_sandbox_session():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.check_response(LIVE_PLACED)
    assert guard.locked
    assert guard.breach.seen_mode is Mode.LIVE


@pytest.mark.parametrize(
    "response",
    [TIMED_OUT, UNREACHABLE, {"status": "error", "message": "Invalid openalgo apikey"}],
    ids=["timeout", "unreachable", "rejected key"],
)
def test_an_error_without_a_mode_does_not_lock_a_sandbox_session(response):
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.check_response(response)
    assert not guard.locked


def test_a_response_that_locks_names_the_order_it_placed():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.check_response(LIVE_PLACED)
    assert guard.breach.orders_after == ("1100000012345",)


def test_a_response_that_locks_records_where_and_when_it_was_seen():
    guard, _, clock, _ = started_guard(Mode.LIVE)
    clock.advance(42)
    guard.check_response(SANDBOX_PLACED)
    assert guard.breach.seen_in is SeenIn.RESPONSE
    assert guard.breach.seen_ns == SESSION_OPEN_NS + 42 * SECOND_NS
    assert guard.breach.session_mode is Mode.LIVE


# Order updates.


def test_an_update_of_an_order_of_the_session_in_the_other_mode_locks():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.note_order_sent("26111600000001")
    guard.check_update(update("26111600000001", Mode.LIVE))
    assert guard.locked
    assert guard.breach.seen_in is SeenIn.UPDATE
    assert guard.breach.seen_mode is Mode.LIVE


def test_an_update_of_an_order_of_the_session_in_its_mode_leaves_the_guard_open():
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.note_order_sent("1100000012345")
    guard.check_update(update("1100000012345", Mode.LIVE))
    assert not guard.locked


def test_an_update_of_an_order_the_session_did_not_send_is_not_judged():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.check_update(update("manual-at-the-broker", Mode.LIVE))
    assert not guard.locked


def test_an_update_of_an_order_known_from_its_response_is_judged():
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.check_response(LIVE_PLACED)
    guard.check_update(update("1100000012345", Mode.SANDBOX))
    assert guard.locked


# Polling cadence, on the manual clock.


def test_poll_reads_nothing_before_the_interval_after_start_has_passed():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    clock.advance(4.999)
    guard.poll()
    assert server.reads == 1


def test_poll_reads_the_switch_once_the_interval_has_passed():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    clock.advance(5)
    guard.poll()
    assert server.reads == 2


def test_poll_reads_once_per_interval_however_often_it_is_called():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    for _ in range(100):
        clock.advance(0.25)
        guard.poll()
    assert server.reads == 1 + 5


def test_poll_follows_the_given_interval():
    guard, server, clock, _ = started_guard(Mode.SANDBOX, poll_seconds=2.0)
    for _ in range(10):
        clock.advance(1)
        guard.poll()
    assert server.reads == 1 + 5


def test_a_switch_flipped_mid_session_locks_at_the_next_poll():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    flip(server)
    clock.advance(5)
    guard.poll()
    assert guard.locked
    assert guard.breach.seen_in is SeenIn.POLL
    assert guard.breach.seen_mode is Mode.LIVE
    assert guard.breach.seen_ns == SESSION_OPEN_NS + 5 * SECOND_NS


def test_a_switch_flipped_back_before_the_next_poll_is_not_seen_by_polling():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    flip(server)
    clock.advance(2)
    flip(server)
    clock.advance(3)
    guard.poll()
    assert not guard.locked


def test_poll_reads_nothing_once_locked():
    guard, server, clock, _ = started_guard(Mode.LIVE)
    guard.check_response(SANDBOX_PLACED)
    clock.advance(60)
    guard.poll()
    assert server.reads == 1


# Failed polls.


def test_a_failed_poll_is_not_a_mode_change():
    guard, server, clock, alerts = started_guard(Mode.SANDBOX)
    server.down = True
    clock.advance(5)
    guard.poll()
    assert not guard.locked
    assert alerts.of(AlertKind.MODE_BREACH) == []


def test_consecutive_failed_polls_are_counted():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    server.down = True
    for _ in range(4):
        clock.advance(5)
        guard.poll()
    assert guard.failed_polls == 4


def test_a_good_read_sets_the_failed_count_back_to_zero():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    server.down = True
    clock.advance(5)
    guard.poll()
    server.down = False
    clock.advance(5)
    guard.poll()
    assert guard.failed_polls == 0


def test_the_third_failed_poll_in_a_row_sends_one_alert():
    guard, server, clock, alerts = started_guard(Mode.SANDBOX)
    server.down = True
    sent_after_each = []
    for _ in range(5):
        clock.advance(5)
        guard.poll()
        sent_after_each.append(len(alerts.of(AlertKind.ERROR)))
    assert sent_after_each == [0, 0, 1, 1, 1]


def test_the_failed_read_alert_tells_to_check_openalgo():
    guard, server, clock, alerts = started_guard(Mode.SANDBOX)
    server.down = True
    for _ in range(3):
        clock.advance(5)
        guard.poll()
    assert "Check that OpenAlgo is running" in alerts.of(AlertKind.ERROR)[0]


def test_a_new_run_of_failed_polls_after_a_good_read_alerts_again():
    guard, server, clock, alerts = started_guard(Mode.SANDBOX)
    for down in [True] * 3 + [False] + [True] * 3:
        server.down = down
        clock.advance(5)
        guard.poll()
    assert len(alerts.of(AlertKind.ERROR)) == 2


def test_a_failed_poll_waits_the_interval_before_the_next_read():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    server.down = True
    clock.advance(5)
    guard.poll()
    clock.advance(1)
    guard.poll()
    assert server.reads == 2


# The read before an entry.


def test_before_entry_reads_the_switch_even_inside_the_poll_interval():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    clock.advance(1)
    guard.before_entry()
    assert server.reads == 2


def test_before_entry_returns_when_the_mode_is_unchanged():
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.before_entry()
    assert not guard.locked


def test_before_entry_raises_mode_changed_error_when_the_switch_was_flipped():
    guard, server, clock, _ = started_guard(Mode.LIVE)
    flip(server)
    clock.advance(1)
    with pytest.raises(ModeChangedError):
        guard.before_entry()
    assert guard.breach.seen_in is SeenIn.BEFORE_ENTRY


def test_before_entry_raises_mode_changed_error_without_reading_once_locked():
    guard, server, _, _ = started_guard(Mode.LIVE)
    guard.check_response(SANDBOX_PLACED)
    with pytest.raises(ModeChangedError):
        guard.before_entry()
    assert server.reads == 1


def test_before_entry_raises_live_error_not_a_mode_change_when_openalgo_does_not_answer():
    guard, server, _, _ = started_guard(Mode.SANDBOX)
    server.down = True
    with pytest.raises(LiveError, match="the entry is not sent") as raised:
        guard.before_entry()
    assert not isinstance(raised.value, ModeChangedError)
    assert not guard.locked


def test_a_failed_read_before_an_entry_counts_as_a_failed_poll():
    guard, server, _, _ = started_guard(Mode.SANDBOX)
    server.down = True
    with pytest.raises(LiveError):
        guard.before_entry()
    assert guard.failed_polls == 1


def test_the_read_before_an_entry_restarts_the_poll_interval():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    clock.advance(4)
    guard.before_entry()
    clock.advance(4)
    guard.poll()
    assert server.reads == 2


# The breach and its alert.


def test_a_lock_sends_one_mode_breach_alert_however_often_the_change_is_seen():
    guard, server, clock, alerts = started_guard(Mode.LIVE)
    guard.check_response(SANDBOX_PLACED)
    guard.check_response(SANDBOX_REJECTED)
    flip(server)
    clock.advance(5)
    guard.poll()
    with pytest.raises(ModeChangedError):
        guard.before_entry()
    assert len(alerts.of(AlertKind.MODE_BREACH)) == 1


def test_the_breach_alert_names_the_orders_sent_after_the_change():
    guard, server, clock, alerts = started_guard(Mode.SANDBOX)
    flip(server)
    guard.note_order_sent("1100000012345")
    guard.note_order_sent("1100000012346")
    clock.advance(5)
    guard.poll()
    assert "1100000012345, 1100000012346" in alerts.of(AlertKind.MODE_BREACH)[0]


def test_the_breach_alert_says_the_session_sends_nothing_more_and_must_be_started_again():
    guard, _, _, alerts = started_guard(Mode.LIVE)
    guard.check_response(SANDBOX_PLACED)
    text = alerts.of(AlertKind.MODE_BREACH)[0]
    assert "sends nothing more" in text
    assert "again in the mode you want" in text


def test_the_breach_alert_names_both_modes_and_the_ist_time():
    guard, _, clock, alerts = started_guard(Mode.LIVE)
    clock.advance(5 * 60)
    guard.check_response(SANDBOX_PLACED)
    text = alerts.of(AlertKind.MODE_BREACH)[0]
    assert "started in LIVE mode and SANDBOX mode was seen" in text
    assert "2026-11-16 09:20 IST" in text


def test_the_mode_changed_error_carries_the_breach_alert_text():
    guard, server, _, alerts = started_guard(Mode.SANDBOX)
    flip(server)
    with pytest.raises(ModeChangedError) as raised:
        guard.before_entry()
    assert str(raised.value) == alerts.of(AlertKind.MODE_BREACH)[0]


def test_the_breach_lists_orders_sent_since_the_switch_was_last_read_in_the_session_mode():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    guard.note_order_sent("before-the-good-read")
    clock.advance(5)
    guard.poll()
    guard.note_order_sent("after-the-good-read")
    flip(server)
    clock.advance(5)
    guard.poll()
    assert guard.breach.orders_after == ("after-the-good-read",)


def test_orders_confirmed_by_their_own_response_are_not_listed():
    guard, server, clock, _ = started_guard(Mode.LIVE)
    guard.check_response(LIVE_PLACED)
    guard.note_order_sent("1100000012345")
    flip(server)
    clock.advance(5)
    guard.poll()
    assert guard.breach.orders_after == ()


def test_orders_confirmed_by_their_own_update_are_not_listed():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    guard.note_order_sent("26111600000001")
    guard.check_update(update("26111600000001", Mode.SANDBOX))
    flip(server)
    clock.advance(5)
    guard.poll()
    assert guard.breach.orders_after == ()


def test_an_order_noted_after_the_lock_is_added_to_the_breach_without_a_second_alert():
    guard, _, _, alerts = started_guard(Mode.SANDBOX)
    guard.check_response(LIVE_PLACED)
    guard.note_order_sent("1100000012399")
    assert guard.breach.orders_after == ("1100000012345", "1100000012399")
    assert len(alerts.of(AlertKind.MODE_BREACH)) == 1


def test_noting_the_order_that_caused_the_lock_does_not_list_it_twice():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.check_response(LIVE_PLACED)
    guard.note_order_sent("1100000012345")
    assert guard.breach.orders_after == ("1100000012345",)


def test_an_empty_order_id_raises_config_error():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    with pytest.raises(ConfigError, match="order ID"):
        guard.note_order_sent("")


def test_the_breach_is_frozen():
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.check_response(SANDBOX_PLACED)
    with pytest.raises(FrozenInstanceError):
        guard.breach.orders_after = ()


def test_a_guard_without_alerts_still_locks():
    server = FakeOpenAlgo(analyze_mode=False)
    guard = ModeGuard(server, now=ManualClock())
    guard.start()
    guard.check_response(SANDBOX_PLACED)
    assert guard.locked


def test_the_guard_never_calls_anything_but_analyzerstatus():
    guard, server, clock, _ = started_guard(Mode.SANDBOX)
    guard.before_entry()
    guard.check_response(SANDBOX_PLACED)
    guard.note_order_sent("26111600000001")
    flip(server)
    clock.advance(5)
    guard.poll()
    assert guard.locked
    assert server.reads == 3


# Semi-auto (the Action Center).


@pytest.mark.parametrize("session_mode", [Mode.LIVE, Mode.SANDBOX])
def test_a_response_with_a_pending_order_id_locks_the_guard_as_semi_auto(session_mode):
    guard, _, _, _ = started_guard(session_mode)
    guard.check_response(QUEUED_IN_ACTION_CENTER)
    assert guard.locked
    assert guard.breach.semi_auto
    assert guard.breach.pending_order_id == "41"


def test_a_semi_auto_breach_has_no_mode_seen():
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.check_response(QUEUED_IN_ACTION_CENTER)
    assert guard.breach.seen_mode is None
    assert guard.breach.session_mode is Mode.LIVE


def test_a_pending_order_id_alone_locks_the_guard_as_semi_auto():
    guard, _, _, _ = started_guard(Mode.LIVE)
    guard.check_response({"status": "success", "pending_order_id": 7})
    assert guard.breach.semi_auto


def test_the_semi_auto_alert_names_the_pending_order_and_says_to_reject_it_and_turn_it_off():
    guard, _, _, alerts = started_guard(Mode.LIVE)
    guard.check_response(QUEUED_IN_ACTION_CENTER)
    text = alerts.of(AlertKind.MODE_BREACH)[0]
    assert "pending order 41" in text
    assert "reject it in the Action Center" in text
    assert "order mode back to auto" in text
    assert "sends nothing more" in text


def test_a_semi_auto_lock_raises_mode_changed_error_before_the_next_entry():
    guard, _, _, _ = started_guard(Mode.SANDBOX)
    guard.check_response(QUEUED_IN_ACTION_CENTER)
    with pytest.raises(ModeChangedError, match="semi-auto"):
        guard.before_entry()


def test_a_mode_breach_is_not_semi_auto():
    breach = ModeBreach(
        session_mode=Mode.SANDBOX,
        seen_mode=Mode.LIVE,
        seen_ns=SESSION_OPEN_NS,
        seen_in=SeenIn.POLL,
        orders_after=(),
    )
    assert not breach.semi_auto

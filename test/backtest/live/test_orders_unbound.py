"""Unbound intents (a 502, a timeout, a lost connection) settled against OpenAlgo's order
book: bound to their order, marked not placed after two reads that miss it, or left for
the operator with an alert."""

import datetime as dt

import pytest
from orders_fakes import (
    CONNECTION_LOST,
    DROPPED,
    NS,
    PROXY_504,
    SERVER_502,
    START_IST,
    START_NS,
    TIMEOUT,
    Clock,
    FakeJournal,
    book,
    book_row,
    http_error,
    make_intent,
    make_rig,
    placed,
    server_error,
    status_answer,
    update,
)

from openalgo.backtest._live._order_book import book_time_ns
from openalgo.backtest._live._order_follow import MAX_UNCLAIMED_UPDATES
from openalgo.backtest._live._types import AlertKind, OrderStatus
from openalgo.backtest._live.orders import BROKER_WINDOW_NS, Resolution, Resolved

SEMI_AUTO = {
    "status": "success",
    "message": "Order queued for approval in Action Center",
    "mode": "semi_auto",
    "pending_order_id": 17,
}


def unbound_rig(*rows, intent=None, reads=2, **answers):
    """A rig whose intent (i-1 unless given) was sent and answered with OpenAlgo's own
    502, with an order book holding `rows` for `reads` reads."""
    rig = make_rig(placeorder=[SERVER_502, *answers.pop("placeorder", [])], **answers)
    rig.client.answers["orderbook"] = [book(*rows) for _ in range(reads)]
    rig.path.send(intent or make_intent("i-1"))
    return rig


def settle(rig):
    """Read the order book 5 and 6 seconds after now, and return every resolution."""
    rig.clock.advance(5)
    resolutions = rig.path.resolve_unbound()
    rig.clock.advance(1)
    return resolutions + rig.path.resolve_unbound()


def test_an_unbound_intent_found_in_the_order_book_is_bound_to_its_order():
    rig = unbound_rig(book_row("2611160001"))
    intent = rig.journal.intents["i-1"]

    resolutions = rig.path.resolve_unbound()

    assert resolutions == [Resolution(intent, Resolved.BOUND, order_id="2611160001")]
    assert rig.journal.order_of == {"i-1": "2611160001"}
    assert rig.guard.sent == ["2611160001"]


def test_an_unbound_intent_is_not_placed_after_two_reads_that_miss_it():
    rig = unbound_rig()
    rig.clock.advance(5)

    assert rig.path.resolve_unbound() == []
    assert rig.journal.not_placed == {}
    rig.clock.advance(1)
    (resolution,) = rig.path.resolve_unbound()

    assert resolution.kind is Resolved.NOT_PLACED
    assert "order book" in rig.journal.not_placed["i-1"]
    assert rig.journal.unbound() == []


@pytest.mark.parametrize(
    "answer",
    [
        SERVER_502,
        http_error(500, server_error("Failed to place order due to internal error")),
        http_error(503, server_error("Service unavailable")),
        {"status": "success"},
        CONNECTION_LOST,
    ],
    ids=["openalgo 502", "openalgo 500", "openalgo 503", "success, no ID", "connection refused"],
)
def test_when_the_server_has_finished_misses_count_from_five_seconds(answer):
    rig = make_rig(placeorder=[answer], orderbook=[book(), book()])
    rig.path.send(make_intent("i-1"))

    (resolution,) = settle(rig)

    assert resolution.kind is Resolved.NOT_PLACED


@pytest.mark.parametrize(
    "answer",
    [
        TIMEOUT,
        DROPPED,
        PROXY_504,
        http_error(502),
        http_error(504, {"message": "Endpoint request timed out"}),
        http_error(502, {"message": "An invalid response was received from the upstream"}),
    ],
    ids=["timeout", "dropped connection", "proxy page", "502 page", "gateway 504", "gateway 502"],
)
def test_otherwise_misses_count_only_after_the_servers_broker_window(answer):
    rig = make_rig(placeorder=[answer], orderbook=[book()] * 3)
    rig.path.send(make_intent("i-1"))

    rig.clock.advance(BROKER_WINDOW_NS / NS - 1)
    before_the_window = rig.path.resolve_unbound()
    rig.clock.advance(1)
    first_miss = rig.path.resolve_unbound()
    rig.clock.advance(1)
    second_miss = rig.path.resolve_unbound()

    assert (before_the_window, first_miss) == ([], [])
    assert [resolution.kind for resolution in second_miss] == [Resolved.NOT_PLACED]


def test_the_broker_window_runs_from_the_send_not_from_the_answer():
    rig = make_rig(placeorder=[TIMEOUT], orderbook=[book()] * 2)
    rig.client.before_call = lambda name, kwargs: name == "placeorder" and rig.clock.advance(10)
    rig.path.send(make_intent("i-1"))

    rig.clock.advance(BROKER_WINDOW_NS / NS - 10)
    rig.path.resolve_unbound()
    rig.clock.advance(1)

    assert [resolution.kind for resolution in rig.path.resolve_unbound()] == [Resolved.NOT_PLACED]


def test_an_intent_left_unbound_before_a_restart_waits_the_whole_broker_window():
    journal = FakeJournal()
    journal.record_intent(make_intent("i-1"))
    journal.mark_unbound("i-1", "Request timed out.")
    rig = make_rig(journal=journal, clock=Clock(START_NS + 30 * NS), orderbook=[book()] * 4)

    early = settle(rig)
    rig.clock.advance(BROKER_WINDOW_NS / NS)
    late = settle(rig)

    assert early == []
    assert [resolution.kind for resolution in late] == [Resolved.NOT_PLACED]


def test_a_read_sooner_than_five_seconds_after_the_send_does_not_count_as_a_miss():
    rig = unbound_rig(reads=3)
    for seconds in (1, 3.5, 1):
        rig.clock.advance(seconds)
        assert rig.path.resolve_unbound() == []

    assert rig.journal.not_placed == {}


def test_the_order_book_is_read_at_most_once_a_second():
    rig = unbound_rig()

    for _ in range(10):
        rig.clock.advance(0.1)
        rig.path.resolve_unbound()

    assert len(rig.client.calls_to("orderbook")) == 1


def test_orders_are_sent_again_once_the_unbound_intent_is_settled():
    rig = unbound_rig(book_row("2611160001"), placeorder=[placed("2611160002")])
    rig.path.resolve_unbound()

    outcome = rig.path.send(make_intent("i-2"))

    assert outcome.order_id == "2611160002"


@pytest.mark.parametrize(
    "difference",
    [
        {"strategy": "other_strategy"},
        {"symbol": "INFY"},
        {"exchange": "BSE"},
        {"action": "SELL"},
        {"quantity": 20},
        {"pricetype": "LIMIT"},
        {"product": "CNC"},
    ],
    ids=lambda difference: next(iter(difference)),
)
def test_a_row_that_differs_in_any_field_is_not_the_intents_order(difference):
    rig = unbound_rig(book_row("2611160001", **difference))

    (resolution,) = settle(rig)

    assert resolution.kind is Resolved.NOT_PLACED


def test_a_broker_row_that_names_no_strategy_matches_on_the_rest():
    row = book_row("2611160001", quantity="10", action="buy")
    del row["strategy"]
    rig = unbound_rig(row)

    (resolution,) = rig.path.resolve_unbound()

    assert resolution.order_id == "2611160001"


def test_the_sandbox_orderstatus_spelling_of_the_price_type_is_read():
    row = book_row("2611160001")
    row["price_type"] = row.pop("pricetype")
    rig = unbound_rig(row)

    (resolution,) = rig.path.resolve_unbound()

    assert resolution.kind is Resolved.BOUND


@pytest.mark.parametrize(
    ("placed_at", "kind"),
    [
        ("2026-11-16 10:15:20", Resolved.BOUND),
        ("2026-11-16 10:15:18", Resolved.BOUND),
        ("2026-11-16 10:15:17", Resolved.NOT_PLACED),
        ("2026-11-16 09:30:00", Resolved.NOT_PLACED),
    ],
    ids=["same second", "within the skew limit", "past the skew limit", "earlier order"],
)
def test_an_order_placed_before_the_intent_beyond_the_clock_skew_limit_is_not_its_order(
    placed_at, kind
):
    intent = make_intent("i-1", created_ns=START_NS + 400_000_000)
    rig = unbound_rig(book_row("2611160001", timestamp=placed_at), intent=intent)

    (resolution,) = settle(rig)

    assert resolution.kind is kind


def test_two_matching_orders_leave_the_intent_unbound_for_the_operator():
    rig = unbound_rig(book_row("2611160001"), book_row("2611160002"))

    (resolution,) = rig.path.resolve_unbound()

    assert resolution.kind is Resolved.AMBIGUOUS
    assert resolution.candidates == ("2611160001", "2611160002")
    assert [intent.intent_id for intent in rig.journal.unbound()] == ["i-1"]


def test_an_ambiguous_intent_sends_one_error_alert_naming_it():
    rig = unbound_rig(book_row("2611160001"), book_row("2611160002"))

    settle(rig)

    ((kind, text),) = rig.alerts.sent
    assert kind is AlertKind.ERROR
    assert "i-1" in text
    assert "Nothing more is sent for SBIN on NSE (MIS) until it is resolved" in text


def test_while_openalgo_is_silent_the_stuck_alert_says_nothing_is_sent_for_any_instrument():
    rig = make_rig(placeorder=[TIMEOUT], orderbook=[TIMEOUT])
    rig.path.send(make_intent("i-1"))
    rig.clock.advance(1)

    rig.path.resolve_unbound()

    ((kind, text),) = rig.alerts.sent
    assert kind is AlertKind.ERROR
    assert "Nothing is sent for any instrument until OpenAlgo answers again" in text


def test_a_matching_row_whose_time_cannot_be_read_leaves_the_intent_ambiguous():
    rig = unbound_rig(book_row("2611160001", timestamp="16/11/26 10.15"))

    (resolution,) = rig.path.resolve_unbound()

    assert (resolution.kind, resolution.candidates) == (Resolved.AMBIGUOUS, ("2611160001",))
    assert rig.journal.order_of == {}
    assert rig.journal.not_placed == {}


def test_an_order_book_that_cannot_be_read_keeps_the_intent_and_sends_one_alert():
    rig = unbound_rig(reads=0)
    rig.client.answers["orderbook"] = [TIMEOUT, TIMEOUT]

    assert settle(rig) == []
    assert [intent.intent_id for intent in rig.journal.unbound()] == ["i-1"]
    assert [kind for kind, _ in rig.alerts.sent] == [AlertKind.ERROR]


def test_with_nothing_unbound_the_order_book_is_not_read():
    rig = make_rig(placeorder=[placed("1")])
    rig.path.send(make_intent("i-1"))

    assert rig.path.resolve_unbound() == []
    assert rig.client.calls_to("orderbook") == []


def test_nothing_is_resolved_while_the_guard_is_locked():
    rig = unbound_rig()
    rig.guard.locked = True

    assert settle(rig) == []
    assert rig.client.calls_to("orderbook") == []


def test_a_book_of_the_other_mode_settles_nothing():
    rig = unbound_rig(reads=0)
    sandbox_book = {**book(), "mode": "analyze"}
    rig.client.answers["orderbook"] = [sandbox_book, sandbox_book]

    assert settle(rig) == []
    assert rig.guard.locked
    assert rig.guard.responses[-1] is sandbox_book
    assert rig.journal.not_placed == {}


def test_an_order_this_session_bound_is_not_matched_to_another_intent():
    rig = make_rig(placeorder=[placed("2611160001"), SERVER_502])
    rig.client.answers["orderbook"] = [book(book_row("2611160001"))] * 2
    rig.path.send(make_intent("i-1"))
    rig.path.send(make_intent("i-2"))

    (resolution,) = settle(rig)

    assert (resolution.intent.intent_id, resolution.kind) == ("i-2", Resolved.NOT_PLACED)


def test_an_order_another_intent_owns_in_the_journal_is_never_bound_after_a_restart():
    journal = FakeJournal()
    journal.record_intent(make_intent("i-A"))
    journal.bind("i-A", "111")
    journal.record_intent(make_intent("i-B", created_ns=START_NS + NS))
    journal.mark_unbound("i-B", "timeout")
    rig = make_rig(journal=journal, clock=Clock(START_NS + 200 * NS))
    rig.client.answers["orderbook"] = [book(book_row("111", timestamp=START_IST))] * 2

    resolutions = rig.path.resolve_unbound()
    rig.clock.advance(1)
    resolutions += rig.path.resolve_unbound()

    assert [(r.intent.intent_id, r.kind) for r in resolutions] == [("i-B", Resolved.NOT_PLACED)]
    assert journal.order_of == {"i-A": "111"}


def test_an_update_of_an_order_not_bound_is_neither_returned_nor_journaled():
    rig = unbound_rig(book_row("555"))

    assert rig.path.on_update(update("555", OrderStatus.COMPLETE, 10)) is None
    assert rig.journal.updates == []
    assert rig.guard.updates == []


def test_the_resolution_carries_the_latest_update_of_the_order_it_binds():
    rig = unbound_rig(book_row("555"), orderstatus=[status_answer("555", "complete")])
    rig.path.on_update(update("555", OrderStatus.OPEN, 0))
    done = update("555", OrderStatus.COMPLETE, 10)
    rig.path.on_update(done)

    (resolution,) = rig.path.resolve_unbound()

    assert (resolution.kind, resolution.update) == (Resolved.BOUND, done)
    assert rig.journal.updates == [done]
    rig.clock.advance(5)
    assert rig.path.due_polls() == []
    assert rig.client.calls_to("orderstatus") == []


def test_only_the_latest_updates_of_so_many_orders_not_bound_are_kept():
    rig = unbound_rig(book_row("555"))
    rig.path.on_update(update("555", OrderStatus.COMPLETE, 10))
    for n in range(MAX_UNCLAIMED_UPDATES):
        rig.path.on_update(update(f"manual-{n}", OrderStatus.COMPLETE, 10))

    (resolution,) = rig.path.resolve_unbound()

    assert (resolution.kind, resolution.update) == (Resolved.BOUND, None)


def test_an_order_found_in_the_book_is_polled_like_one_sent():
    rig = unbound_rig(
        book_row("2611160001"), orderstatus=[status_answer("2611160001", "trigger pending")]
    )
    rig.clock.advance(3)
    rig.path.resolve_unbound()

    rig.path.due_polls()

    assert [kwargs["order_id"] for kwargs in rig.client.calls_to("orderstatus")] == ["2611160001"]


def test_a_sandbox_order_book_error_reaches_the_guard_as_the_server_wrote_it():
    rig = unbound_rig(reads=0)
    rig.client.answers["orderbook"] = [
        http_error(500, server_error("Error getting orderbook", mode="analyze"))
    ]

    rig.path.resolve_unbound()

    assert rig.guard.responses[-1]["mode"] == "analyze"


@pytest.mark.parametrize(
    "written",
    [
        "2026-11-16 10:15:20",
        "2026-11-16T10:15:20",
        "2026-11-16 10:15:20.750000",
        "16-Nov-2026 10:15:20",
        "16-11-2026 10:15:20",
        "10:15:20 16-11-2026",
        "2026-11-16T04:45:20+00:00",
        "2026-11-16T10:15:20+05:30",
    ],
)
def test_order_book_times_are_read_in_the_brokers_formats_as_ist(written):
    expected = int(dt.datetime(2026, 11, 16, 4, 45, 20, tzinfo=dt.UTC).timestamp()) * NS

    assert book_time_ns(written) == expected


@pytest.mark.parametrize("written", ["", None, "yesterday", "16/11/26 10.15"])
def test_an_order_book_time_that_cannot_be_read_is_none(written):
    assert book_time_ns(written) is None

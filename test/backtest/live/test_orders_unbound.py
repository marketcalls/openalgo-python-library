"""Unbound intents (a 502, a timeout, a lost connection) settled against OpenAlgo's order
book: bound to their order, marked not placed, or left for the operator."""

import datetime as dt

import pytest
from orders_fakes import (
    NS,
    START_NS,
    TIMEOUT,
    book,
    book_row,
    http_error,
    make_intent,
    make_rig,
    placed,
    status_answer,
)

from openalgo.backtest._live._order_book import book_time_ns
from openalgo.backtest._live.orders import Resolution, Resolved
from openalgo.backtest.errors import LiveError


def unbound_rig(*rows, intent=None, **answers):
    """A rig whose intent (i-1 unless given) was sent and answered 502, with an order
    book holding `rows`."""
    rig = make_rig(placeorder=[http_error(502), *answers.pop("placeorder", [])], **answers)
    rig.client.answers["orderbook"] = [book(*rows)]
    rig.path.send(intent or make_intent("i-1"))
    return rig


def test_an_unbound_intent_found_in_the_order_book_is_bound_to_its_order():
    rig = unbound_rig(book_row("2611160001"))
    intent = rig.journal.intents["i-1"]

    resolutions = rig.path.resolve_unbound()

    assert resolutions == [Resolution(intent, Resolved.BOUND, order_id="2611160001")]
    assert rig.journal.order_of == {"i-1": "2611160001"}
    assert rig.guard.sent == ["2611160001"]


def test_an_unbound_intent_missing_from_the_order_book_is_marked_not_placed():
    rig = unbound_rig()

    (resolution,) = rig.path.resolve_unbound()

    assert resolution.kind is Resolved.NOT_PLACED
    assert "order book" in rig.journal.not_placed["i-1"]
    assert rig.journal.unbound() == []


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
    ],
    ids=lambda difference: next(iter(difference)),
)
def test_a_row_that_differs_in_any_field_is_not_the_intents_order(difference):
    rig = unbound_rig(book_row("2611160001", **difference))

    (resolution,) = rig.path.resolve_unbound()

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

    (resolution,) = rig.path.resolve_unbound()

    assert resolution.kind is kind


def test_two_matching_orders_leave_the_intent_unbound_for_the_operator():
    rig = unbound_rig(book_row("2611160001"), book_row("2611160002"))

    (resolution,) = rig.path.resolve_unbound()

    assert resolution.kind is Resolved.AMBIGUOUS
    assert resolution.candidates == ("2611160001", "2611160002")
    assert [intent.intent_id for intent in rig.journal.unbound()] == ["i-1"]


def test_an_order_this_session_bound_is_not_matched_to_another_intent():
    rig = make_rig(placeorder=[placed("2611160001"), TIMEOUT])
    rig.client.answers["orderbook"] = [book(book_row("2611160001"))]
    rig.path.send(make_intent("i-1"))
    rig.path.send(make_intent("i-2"))

    (resolution,) = rig.path.resolve_unbound()

    assert (resolution.intent.intent_id, resolution.kind) == ("i-2", Resolved.NOT_PLACED)


def test_a_matching_row_whose_time_cannot_be_read_raises_and_changes_nothing():
    rig = unbound_rig(book_row("2611160001", timestamp="16/11/26 10.15"))

    with pytest.raises(LiveError, match="'16/11/26 10.15' cannot be read"):
        rig.path.resolve_unbound()

    assert rig.journal.order_of == {}
    assert rig.journal.not_placed == {}


def test_an_order_book_that_cannot_be_read_resolves_nothing():
    rig = unbound_rig()
    rig.client.answers["orderbook"] = [TIMEOUT]

    assert rig.path.resolve_unbound() == []
    assert [intent.intent_id for intent in rig.journal.unbound()] == ["i-1"]


def test_with_nothing_unbound_the_order_book_is_not_read():
    rig = make_rig(placeorder=[placed("1")])
    rig.path.send(make_intent("i-1"))

    assert rig.path.resolve_unbound() == []
    assert rig.client.calls_to("orderbook") == []


def test_an_order_found_in_the_book_is_polled_like_one_sent():
    rig = unbound_rig(
        book_row("2611160001"), orderstatus=[status_answer("2611160001", "trigger pending")]
    )
    rig.clock.advance(3)
    rig.path.resolve_unbound()

    rig.path.due_polls()

    assert [kwargs["order_id"] for kwargs in rig.client.calls_to("orderstatus")] == ["2611160001"]


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

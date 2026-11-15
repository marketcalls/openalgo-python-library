"""Sending orders: the intent journaled before the HTTP call, each request sent once, the
answer read as accepted, rejected, unanswered or queued, the mode guard's part, and the
rate budget."""

import pytest
from orders_fakes import (
    CONNECTION_LOST,
    NS,
    START_NS,
    STRATEGY,
    TIMEOUT,
    Clock,
    FakeGuard,
    book,
    http_error,
    make_intent,
    make_rig,
    placed,
    server_error,
    status_answer,
)

from openalgo.backtest._live._order_answers import server_answer
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live.orders import Outcome, RateBudget, SendOutcome
from openalgo.backtest.errors import LiveError, ModeChangedError

#: The server's message for an order whose broker answer was lost (utils/httpx_client.py).
UNANSWERED = (
    "The connection to the broker dropped after the order was sent, so it may have been "
    "placed. Check the order book before placing it again."
)
MODIFY_FIELDS = {
    "symbol": "SBIN",
    "action": "SELL",
    "exchange": "NSE",
    "price_type": "SL-M",
    "product": "MIS",
    "quantity": 10,
    "price": 0,
    "trigger_price": 806.5,
}


def placed_rig(**answers):
    """A rig whose first placement, intent i-1, is accepted as order 1."""
    rig = make_rig(placeorder=[placed("1"), *answers.pop("placeorder", [])], **answers)
    rig.path.send(make_intent("i-1"))
    return rig


def test_the_intent_is_journaled_before_placeorder_is_called():
    rig = make_rig(placeorder=[placed("1")])
    journaled_at_call = []
    rig.client.before_call = lambda name, kwargs: journaled_at_call.append(
        list(rig.journal.intents)
    )

    rig.path.send(make_intent("i-1"))

    assert journaled_at_call == [["i-1"]]


def test_placeorder_is_sent_the_intents_order_with_its_strategy():
    rig = make_rig(placeorder=[placed("1")])

    rig.path.send(make_intent(price_type="LIMIT", price=811.5))

    assert rig.client.calls_to("placeorder") == [
        {
            "strategy": STRATEGY,
            "symbol": "SBIN",
            "action": "BUY",
            "exchange": "NSE",
            "price_type": "LIMIT",
            "product": "MIS",
            "quantity": 10,
            "price": 811.5,
            "trigger_price": 0.0,
        }
    ]


def test_an_accepted_placement_binds_the_order_id_and_tells_the_guard():
    rig = make_rig(placeorder=[placed("2611160001")])

    outcome = rig.path.send(make_intent("i-1"))

    assert outcome == SendOutcome(Outcome.ACCEPTED, order_id="2611160001")
    assert rig.journal.order_of == {"i-1": "2611160001"}
    assert rig.guard.sent == ["2611160001"]


def test_the_order_id_is_journaled_before_the_guard_reads_the_answer():
    rig = make_rig(placeorder=[placed("2611160001")])
    bound_when_read = []
    check_response = rig.guard.check_response

    def reading(response):
        bound_when_read.append(dict(rig.journal.order_of))
        check_response(response)

    rig.guard.check_response = reading

    rig.path.send(make_intent("i-1"))

    assert bound_when_read == [{"i-1": "2611160001"}]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (
            {
                "status": "error",
                "message": "Insufficient funds",
                "code": 200,
                "error_type": "api_error",
            },
            "Insufficient funds",
        ),
        (http_error(400, server_error("Invalid symbol")), "Invalid symbol"),
        (http_error(403, server_error("Invalid openalgo apikey")), "Invalid openalgo apikey"),
        (http_error(429, "Too Many Requests"), "HTTP 429: Too Many Requests"),
        (
            http_error(400, server_error("Insufficient funds", mode="analyze")),
            "Insufficient funds",
        ),
    ],
    ids=["api error", "400", "403", "429", "sandbox 400"],
)
def test_an_error_answer_is_a_rejection_marked_not_placed_with_the_servers_message(answer, reason):
    rig = make_rig(placeorder=[answer])

    outcome = rig.path.send(make_intent("i-1"))

    assert outcome == SendOutcome(Outcome.REJECTED, reason=reason)
    assert rig.journal.not_placed == {"i-1": reason}
    assert rig.journal.order_of == {}
    assert rig.journal.unbound() == []


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (http_error(502, server_error(UNANSWERED)), UNANSWERED),
        (
            http_error(500, server_error("Failed to place order due to internal error")),
            "Failed to place order due to internal error",
        ),
        (http_error(504, "<html>Gateway Timeout</html>"), "HTTP 504: <html>Gateway Timeout</html>"),
        (TIMEOUT, TIMEOUT["message"]),
        (CONNECTION_LOST, CONNECTION_LOST["message"]),
        (
            {"status": "error", "message": "HTTP error occurred: ", "error_type": "http_error"},
            "HTTP error occurred: ",
        ),
        (
            {"status": "error", "message": "Invalid JSON response", "error_type": "json_error"},
            "Invalid JSON response",
        ),
        (
            {"status": "error", "message": "An unexpected error", "error_type": "unknown_error"},
            "An unexpected error",
        ),
    ],
    ids=["502", "500", "504 page", "timeout", "connection", "broken", "json", "unknown"],
)
def test_a_transport_failure_leaves_the_intent_unbound(answer, reason):
    rig = make_rig(placeorder=[answer])
    intent = make_intent("i-1")

    outcome = rig.path.send(intent)

    assert outcome == SendOutcome(Outcome.UNANSWERED, reason=reason)
    assert rig.journal.unbound_reasons == {"i-1": reason}
    assert rig.journal.unbound() == [intent]


@pytest.mark.parametrize(
    "answer",
    [http_error(502, server_error(UNANSWERED)), TIMEOUT, CONNECTION_LOST, {"status": "success"}],
    ids=["502", "timeout", "connection", "no order ID"],
)
def test_an_intent_left_unbound_is_noted_in_doubt_with_the_guard(answer):
    rig = make_rig(placeorder=[answer])

    rig.path.send(make_intent("i-1"))

    assert rig.guard.in_doubt == ["i-1"]


@pytest.mark.parametrize(
    "answer", [placed("1"), http_error(400, server_error("Invalid symbol"))], ids=["placed", "400"]
)
def test_a_placed_or_refused_order_is_not_in_doubt(answer):
    rig = make_rig(placeorder=[answer])

    rig.path.send(make_intent("i-1"))

    assert rig.guard.in_doubt == []


def test_a_sandbox_rejection_wrapped_by_the_sdk_reaches_the_guard_in_sandbox_mode():
    wrapped = {
        "status": "error",
        "message": 'HTTP 400: {"message":"Insufficient funds","mode":"analyze","status":"error"}\n',
        "code": 400,
        "error_type": "http_error",
    }
    rig = make_rig(placeorder=[wrapped])

    rig.path.send(make_intent("i-1"))

    assert rig.guard.responses == [
        {"status": "error", "message": "Insufficient funds", "mode": "analyze", "code": 400}
    ]


def test_the_semi_auto_refusal_of_a_cancel_reaches_the_guard_as_the_server_wrote_it():
    refusal = (
        "Cancel order operation is not allowed in Semi-Auto mode. Please switch to Auto "
        "mode to cancel orders."
    )
    rig = placed_rig(cancelorder=[http_error(403, server_error(refusal))])

    outcome = rig.path.cancel("1")

    assert rig.guard.responses[-1] == {"status": "error", "message": refusal, "code": 403}
    assert outcome == SendOutcome(Outcome.REJECTED, reason=refusal)


def test_a_sandbox_orderstatus_error_reaches_the_guard_in_sandbox_mode():
    not_found = http_error(404, server_error("Order 1 not found", mode="analyze"))
    rig = placed_rig(orderstatus=[not_found])
    rig.clock.advance(2)

    rig.path.due_polls()

    assert rig.guard.responses[-1]["mode"] == "analyze"


@pytest.mark.parametrize(
    "answer",
    [
        placed("1"),
        TIMEOUT,
        CONNECTION_LOST,
        {"status": "error", "message": "HTTP error occurred: ", "error_type": "http_error"},
        http_error(504, "<html>Gateway Timeout</html>"),
        http_error(400, '["not", "an", "object"]'),
    ],
    ids=["success", "timeout", "connection", "broken", "page", "json list"],
)
def test_server_answer_returns_an_answer_without_a_server_body_unchanged(answer):
    assert server_answer(answer) is answer


@pytest.mark.parametrize(
    "answer", [{}, {"message": "accepted"}, {"status": "queued"}], ids=["empty", "no status", "odd"]
)
def test_an_answer_with_no_status_of_success_or_error_leaves_the_intent_unbound(answer):
    rig = make_rig(placeorder=[answer])

    outcome = rig.path.send(make_intent("i-1"))

    assert outcome.kind is Outcome.UNANSWERED
    assert rig.journal.not_placed == {}


def test_the_mode_is_read_after_the_budget_wait_just_before_the_call():
    rig = placed_rig(placeorder=[placed(str(n)) for n in range(2, 7)])
    for n in range(2, 6):
        rig.path.send(make_intent(f"i-{n}", purpose=Purpose.EXIT))
    events = []
    before_entry = rig.guard.before_entry

    def reading():
        events.append(("mode read", rig.clock.now_ns - START_NS))
        before_entry()

    rig.guard.before_entry = reading
    rig.client.before_call = lambda name, kwargs: events.append((name, rig.clock.now_ns - START_NS))

    rig.path.send(make_intent("i-6"))

    assert events == [("mode read", NS), ("placeorder", NS)]


def test_a_success_without_an_order_id_leaves_the_intent_unbound():
    rig = make_rig(placeorder=[{"status": "success"}])

    outcome = rig.path.send(make_intent("i-1"))

    assert outcome.kind is Outcome.UNANSWERED
    assert "i-1" in rig.journal.unbound_reasons


def test_a_502_is_never_retried():
    rig = make_rig(placeorder=[http_error(502)])

    rig.path.send(make_intent("i-1"))

    assert len(rig.client.calls_to("placeorder")) == 1


@pytest.mark.parametrize(
    "request_while_unbound",
    [
        lambda path: path.send(make_intent("i-2")),
        lambda path: path.modify("1", **MODIFY_FIELDS),
        lambda path: path.cancel("1"),
    ],
    ids=["send", "modify", "cancel"],
)
def test_nothing_is_sent_for_the_instrument_of_an_unbound_intent(request_while_unbound):
    rig = placed_rig(placeorder=[TIMEOUT])
    rig.path.send(make_intent("i-unbound"))
    calls = len(rig.client.calls)

    with pytest.raises(LiveError, match=r"i-unbound of SBIN on NSE \(MIS\) is unbound"):
        request_while_unbound(rig.path)

    assert len(rig.client.calls) == calls


def test_a_trailing_modify_of_another_instruments_stop_is_not_held_back():
    stop = make_intent("i-stop", symbol="INFY", purpose=Purpose.STOP, action="SELL")
    rig = make_rig(placeorder=[placed("1"), TIMEOUT], modifyorder=[placed("1")])
    rig.path.send(stop)
    rig.path.send(make_intent("i-unbound"))

    outcome = rig.path.modify("1", **{**MODIFY_FIELDS, "symbol": "INFY"})

    assert outcome.kind is Outcome.ACCEPTED


@pytest.mark.parametrize(
    "other",
    [{"symbol": "INFY"}, {"exchange": "BSE"}, {"product": "CNC"}],
    ids=["symbol", "exchange", "product"],
)
def test_a_send_for_another_instrument_or_product_goes_ahead(other):
    rig = make_rig(placeorder=[TIMEOUT, placed("2")])
    rig.path.send(make_intent("i-unbound"))

    outcome = rig.path.send(make_intent("i-2", **other))

    assert outcome.order_id == "2"


@pytest.mark.parametrize(
    "request_when_locked",
    [
        lambda path: path.send(make_intent("i-2", purpose=Purpose.EXIT)),
        lambda path: path.modify("1", **MODIFY_FIELDS),
        lambda path: path.cancel("1"),
    ],
    ids=["send", "modify", "cancel"],
)
def test_a_locked_guard_refuses_every_request(request_when_locked):
    rig = placed_rig()
    rig.guard.locked = True

    with pytest.raises(ModeChangedError, match="locked"):
        request_when_locked(rig.path)

    assert len(rig.client.calls) == 1
    assert list(rig.journal.intents) == ["i-1"]


@pytest.mark.parametrize(
    ("purpose", "reads"),
    [
        (Purpose.ENTRY, 1),
        (Purpose.ADD, 1),
        (Purpose.STOP, 0),
        (Purpose.EXIT, 0),
        (Purpose.RISK_EXIT, 0),
        (Purpose.SQUARE_OFF, 0),
    ],
)
def test_only_an_entry_or_an_add_reads_the_mode_just_before_it_is_sent(purpose, reads):
    rig = make_rig(placeorder=[placed("1")])

    rig.path.send(make_intent(purpose=purpose))

    assert rig.guard.entry_checks == reads


@pytest.mark.parametrize(
    "error",
    [ModeChangedError("the analyzer switch changed"), LiveError("analyzerstatus failed")],
    ids=["mode changed", "mode unread"],
)
def test_an_entry_whose_mode_check_fails_is_not_sent_and_the_error_propagates(error):
    rig = make_rig(guard=FakeGuard(entry_error=error))

    with pytest.raises(type(error), match=str(error)):
        rig.path.send(make_intent())

    assert rig.client.calls == []
    assert rig.journal.intents == {}


def test_a_semi_auto_answer_is_queued_and_not_bound():
    answer = {
        "status": "success",
        "message": "Order queued for approval in Action Center",
        "mode": "semi_auto",
        "pending_order_id": 17,
    }
    rig = make_rig(placeorder=[answer])

    outcome = rig.path.send(make_intent("i-1"))

    assert (outcome.kind, outcome.order_id, outcome.pending_order_id) == (Outcome.QUEUED, "", "17")
    assert rig.journal.order_of == {}
    assert rig.journal.queued_as == {"i-1": "17"}
    assert rig.journal.unbound() == []
    assert rig.guard.responses == [answer]
    assert rig.guard.sent == []
    assert rig.guard.in_doubt == ["i-1"]
    assert rig.guard.locked


def test_modify_sends_the_fields_with_the_orders_strategy():
    rig = placed_rig(modifyorder=[placed("1")])

    outcome = rig.path.modify("1", **MODIFY_FIELDS)

    assert outcome == SendOutcome(Outcome.ACCEPTED, order_id="1")
    assert rig.client.calls_to("modifyorder") == [
        {"order_id": "1", "strategy": STRATEGY, **MODIFY_FIELDS}
    ]


@pytest.mark.parametrize(
    "fields",
    [
        {key: value for key, value in MODIFY_FIELDS.items() if key != "trigger_price"},
        {**MODIFY_FIELDS, "validity": "DAY"},
    ],
    ids=["missing", "unknown"],
)
def test_modify_with_a_missing_or_unknown_field_raises_and_sends_nothing(fields):
    rig = placed_rig()

    with pytest.raises(LiveError, match="modify takes"):
        rig.path.modify("1", **fields)

    assert rig.client.calls_to("modifyorder") == []


def test_cancel_sends_the_order_id_with_its_strategy():
    rig = placed_rig(cancelorder=[placed("1")])

    outcome = rig.path.cancel("1")

    assert outcome == SendOutcome(Outcome.ACCEPTED, order_id="1")
    assert rig.client.calls_to("cancelorder") == [{"order_id": "1", "strategy": STRATEGY}]


@pytest.mark.parametrize(
    "request_on_unknown",
    [lambda path: path.modify("9", **MODIFY_FIELDS), lambda path: path.cancel("9")],
    ids=["modify", "cancel"],
)
def test_an_order_this_session_did_not_send_is_not_changed(request_on_unknown):
    rig = placed_rig()

    with pytest.raises(LiveError, match="order 9 was not sent by this session"):
        request_on_unknown(rig.path)

    assert len(rig.client.calls) == 1


@pytest.mark.parametrize("answer", [http_error(502), TIMEOUT], ids=["502", "timeout"])
def test_an_unanswered_cancel_is_not_retried(answer):
    rig = placed_rig(cancelorder=[answer])

    outcome = rig.path.cancel("1")

    assert outcome.kind is Outcome.UNANSWERED
    assert len(rig.client.calls_to("cancelorder")) == 1


def test_a_rejected_modify_returns_the_servers_message():
    rig = placed_rig(modifyorder=[http_error(400, server_error("Order cannot be modified"))])

    outcome = rig.path.modify("1", **MODIFY_FIELDS)

    assert outcome == SendOutcome(Outcome.REJECTED, reason="Order cannot be modified")


def test_every_answer_the_analyzer_switch_routes_reaches_the_guard():
    # Orders and books alike: the switch routes the reads too.
    answers = {
        "placeorder": [placed("1"), http_error(502)],
        "modifyorder": [http_error(400, "Order cannot be modified")],
        "cancelorder": [placed("1")],
        "orderstatus": [status_answer("1", "cancelled")],
        "tradebook": [{"status": "success", "data": []}],
        "orderbook": [book()],
    }
    expected = [
        answers["placeorder"][0],
        answers["modifyorder"][0],
        answers["cancelorder"][0],
        answers["orderstatus"][0],
        answers["tradebook"][0],
        answers["placeorder"][1],
        answers["orderbook"][0],
    ]
    rig = make_rig(**answers)

    rig.path.send(make_intent("i-1"))
    rig.path.modify("1", **MODIFY_FIELDS)
    rig.path.cancel("1")
    rig.clock.advance(2)
    rig.path.due_polls()
    rig.path.send(make_intent("i-2", purpose=Purpose.EXIT))
    rig.path.resolve_unbound()

    assert len(rig.guard.responses) == len(expected)
    assert all(seen is sent for seen, sent in zip(rig.guard.responses, expected, strict=True))


def test_place_modify_and_cancel_share_one_budget_of_five_a_second():
    rig = placed_rig(
        placeorder=[placed("2"), placed("3")],
        modifyorder=[placed("1")],
        cancelorder=[placed("2"), placed("3")],
    )
    sent_at = []
    rig.client.before_call = lambda name, kwargs: sent_at.append(rig.clock.now_ns - START_NS)

    rig.path.send(make_intent("i-2", purpose=Purpose.EXIT))
    rig.path.modify("1", **MODIFY_FIELDS)
    rig.path.cancel("2")
    rig.path.send(make_intent("i-3", purpose=Purpose.EXIT))
    rig.path.cancel("3")

    assert sent_at == [0, 0, 0, 0, NS]
    assert rig.clock.slept == [1.0]


def test_the_budget_allows_five_requests_in_a_second():
    clock = Clock()
    budget = RateBudget(clock.now)

    for _ in range(5):
        assert budget.wait_seconds() == 0.0
        budget.take()

    assert budget.wait_seconds() == 1.0


def test_a_request_waits_until_the_oldest_of_the_five_is_a_second_old():
    clock = Clock()
    budget = RateBudget(clock.now)
    for _ in range(5):
        budget.take()
        clock.advance(0.1)

    assert budget.wait_seconds() == pytest.approx(0.5)
    clock.advance(0.5)
    assert budget.wait_seconds() == 0.0


def test_take_raises_when_the_budget_is_spent():
    clock = Clock()
    budget = RateBudget(clock.now, per_second=1)
    budget.take()

    with pytest.raises(LiveError, match="spent"):
        budget.take()


@pytest.mark.parametrize("per_second", [0, 6, 10])
def test_a_budget_outside_one_to_half_of_openalgos_raises(per_second):
    with pytest.raises(LiveError, match="at most half of OpenAlgo's 10"):
        RateBudget(Clock().now, per_second=per_second)

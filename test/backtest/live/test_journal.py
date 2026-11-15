"""The live journal's order record (08-live-trading 6.1 and 7): intents written before
their orders are sent, bindings and order updates (the trading day and the mode are in
`test_journal_days.py`)."""

import inspect

import pytest
from journal_samples import (
    BROKER,
    DAY,
    DAY_NS,
    LEG,
    NEXT_DAY,
    a_bound_order,
    a_stop_order,
    an_intent,
    an_update,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest._live._types import BoundOrder, JournalLike, Mode, OrderStatus
from openalgo.backtest._live.journal import Journal
from openalgo.backtest.errors import JournalError


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


@pytest.fixture
def journal(path):
    with started_journal(path) as handle:
        yield handle


@pytest.mark.parametrize("name", [name for name in vars(JournalLike) if not name.startswith("_")])
def test_the_journal_has_every_method_of_the_contract_with_its_parameters(name):
    wanted = list(inspect.signature(getattr(JournalLike, name)).parameters.values())
    offered = list(inspect.signature(getattr(Journal, name)).parameters.values())
    assert [p.name for p in offered[: len(wanted)]] == [p.name for p in wanted]
    extra = offered[len(wanted) :]
    assert all(p.kind is p.KEYWORD_ONLY and p.default is not p.empty for p in extra)


# Intents and bindings


def test_a_recorded_intent_is_found_unbound_after_a_reopen(path):
    with started_journal(path) as journal:
        journal.record_intent(an_intent())
    with started_journal(path) as reopened:
        assert reopened.unbound() == [an_intent()]


def record_an_intent(journal):
    journal.record_intent(an_intent())


def bind_an_intent(journal):
    a_bound_order(journal, "i1", "o1")


def mark_an_intent_not_placed(journal):
    journal.record_intent(an_intent())
    journal.mark_not_placed("i1", "insufficient funds")


def set_a_stop(journal):
    a_stop_order(journal, "o9")
    journal.set_stop(*LEG, "o9")


COMMITTED_WRITES = [
    pytest.param(record_an_intent, lambda j: j.unbound(day=DAY) == [an_intent()], id="intent"),
    pytest.param(bind_an_intent, lambda j: j.order_id_of("i1") == "o1", id="binding"),
    pytest.param(
        mark_an_intent_not_placed,
        lambda j: j.intent("i1") is not None and j.unbound(day=DAY) == [],
        id="not placed",
    ),
    pytest.param(
        lambda j: j.record_update(an_update()),
        lambda j: j.last_update("o1", day=DAY) == an_update(),
        id="update",
    ),
    pytest.param(set_a_stop, lambda j: j.stop_of(*LEG, day=DAY) == "o9", id="stop"),
    pytest.param(
        lambda j: j.put_rms_state("ema", {"peak": 1.5}),
        lambda j: j.rms_state("ema", day=DAY) == {"peak": 1.5},
        id="risk state",
    ),
    pytest.param(
        lambda j: j.put_user_state("ema", {"entries": 1}),
        lambda j: getattr(j.user_state("ema"), "state", None) == {"entries": 1},
        id="user state",
    ),
    pytest.param(
        lambda j: j.start_session(NEXT_DAY, Mode.SANDBOX, BROKER),
        lambda j: j.session(NEXT_DAY) is not None,
        id="session",
    ),
]


@pytest.mark.parametrize(("write", "is_visible"), COMMITTED_WRITES)
def test_every_write_is_committed_before_it_returns(path, write, is_visible):
    with started_journal(path) as writer, open_journal(path) as reader:
        assert not is_visible(reader)
        write(writer)
        assert is_visible(reader)


def test_unbound_intents_come_oldest_first(journal):
    for intent_id in ("i3", "i1", "i2"):
        journal.record_intent(an_intent(intent_id))
    assert [intent.intent_id for intent in journal.unbound()] == ["i3", "i1", "i2"]


def test_a_bound_intent_leaves_the_unbound_list(journal):
    a_bound_order(journal, "i1", "o1")
    assert journal.unbound() == []


def test_an_intent_marked_unbound_stays_in_the_unbound_list(journal):
    journal.record_intent(an_intent())
    journal.mark_unbound("i1", "HTTP 502 on placeorder")
    assert journal.unbound() == [an_intent()]


def test_an_intent_marked_not_placed_leaves_the_unbound_list(journal):
    journal.record_intent(an_intent())
    journal.mark_not_placed("i1", "not in the order book")
    assert journal.unbound() == []


def test_a_queued_intent_leaves_the_unbound_list_for_the_queued_list(journal):
    journal.record_intent(an_intent())
    journal.mark_queued("i1", "17")
    assert (journal.unbound(), journal.queued()) == ([], [an_intent()])


def test_a_queued_intent_is_found_queued_after_a_reopen(path):
    with started_journal(path) as journal:
        journal.record_intent(an_intent())
        journal.mark_queued("i1", "17")
    with started_journal(path) as reopened:
        assert reopened.queued() == [an_intent()]


def test_a_queued_intent_whose_order_is_approved_is_bound_and_leaves_the_queued_list(journal):
    journal.record_intent(an_intent())
    journal.mark_queued("i1", "17")
    journal.bind("i1", "o1")
    assert (journal.queued(), journal.order_id_of("i1")) == ([], "o1")


def test_the_queued_intents_are_of_the_sessions_day(journal):
    journal.record_intent(an_intent())
    journal.mark_queued("i1", "17")
    assert journal.queued(day=NEXT_DAY) == []


def test_an_unbound_intent_found_in_the_order_book_is_bound(journal):
    journal.record_intent(an_intent())
    journal.mark_unbound("i1", "timeout")
    journal.bind("i1", "o1")
    assert (journal.unbound(), journal.order_id_of("i1")) == ([], "o1")


def test_a_bound_intent_maps_to_its_order_and_back(journal):
    a_bound_order(journal, "i1", "o1")
    assert (journal.order_id_of("i1"), journal.intent_of("o1")) == ("o1", an_intent())


def test_an_intent_reads_back_as_it_was_recorded(journal):
    stop = an_intent("s1", action="SELL", price_type="SL", price=95.25, trigger_price=100.25)
    journal.record_intent(stop)
    assert journal.intent("s1") == stop


def test_recording_an_intent_twice_is_refused(journal):
    journal.record_intent(an_intent())
    with pytest.raises(JournalError, match="'i1' is already in the journal"):
        journal.record_intent(an_intent())


def test_binding_an_unknown_intent_is_refused(journal):
    with pytest.raises(JournalError, match="'i9' is not in the sandbox ledger"):
        journal.bind("i9", "o1")


def test_binding_an_intent_to_a_second_order_is_refused(journal):
    a_bound_order(journal, "i1", "o1")
    with pytest.raises(JournalError, match="already bound to order 'o1'"):
        journal.bind("i1", "o2")


def test_binding_an_order_to_a_second_intent_on_its_day_is_refused(journal):
    a_bound_order(journal, "i1", "o1")
    journal.record_intent(an_intent("i2"))
    with pytest.raises(JournalError, match="order 'o1' is already bound to intent 'i1'"):
        journal.bind("i2", "o1")


def test_binding_the_same_pair_twice_changes_nothing(journal):
    a_bound_order(journal, "i1", "o1")
    journal.bind("i1", "o1")
    assert journal.order_id_of("i1") == "o1"


def test_binding_to_a_blank_order_id_is_refused(journal):
    journal.record_intent(an_intent())
    with pytest.raises(JournalError, match="order_id must be a non-empty string"):
        journal.bind("i1", "")


@pytest.mark.parametrize(
    "mark", [Journal.mark_unbound, Journal.mark_not_placed, Journal.mark_queued]
)
def test_a_bound_intent_cannot_be_marked(journal, mark):
    a_bound_order(journal, "i1", "o1")
    with pytest.raises(JournalError, match="bound to order 'o1', so it reached the broker"):
        mark(journal, "i1", "timeout")


@pytest.mark.parametrize(
    "mark", [Journal.mark_unbound, Journal.mark_not_placed, Journal.mark_queued]
)
def test_marking_an_unknown_intent_is_refused(journal, mark):
    with pytest.raises(JournalError, match="'i9' is not in the sandbox ledger"):
        mark(journal, "i9", "timeout")


def test_lookups_of_what_the_journal_does_not_hold_return_none(journal):
    journal.record_intent(an_intent())
    found = (
        journal.intent("i9"),
        journal.order_id_of("i1"),
        journal.intent_of("o9"),
        journal.last_update("o9"),
        journal.stop_of(*LEG),
        journal.rms_state("ema"),
        journal.user_state("ema"),
        journal.session(NEXT_DAY),
    )
    assert found == (None,) * 8


# Order updates


def test_every_update_is_kept_in_arrival_order(journal):
    arrived = [
        an_update(filled_quantity=4, pending_quantity=6),
        an_update(filled_quantity=2, pending_quantity=8),
        an_update(status=OrderStatus.COMPLETE, filled_quantity=10, pending_quantity=0),
    ]
    for update in arrived:
        journal.record_update(update)
    assert journal.updates("o1") == arrived


def test_the_last_update_is_the_latest_to_arrive(journal):
    partly_filled = an_update(filled_quantity=4, pending_quantity=6, average_price=99.5)
    journal.record_update(an_update())
    journal.record_update(partly_filled)
    assert journal.last_update("o1") == partly_filled


def test_an_update_arriving_after_a_terminal_one_is_stale(journal):
    journal.record_update(an_update(status=OrderStatus.CANCELLED))
    journal.record_update(an_update(status=OrderStatus.OPEN, ts_recv_ns=DAY_NS + 9_000))
    assert journal.last_update("o1") == an_update(status=OrderStatus.CANCELLED)


FILLED = {"status": OrderStatus.COMPLETE, "filled_quantity": 10, "pending_quantity": 0}
LATER_NS = DAY_NS + 9_000


def test_the_first_terminal_update_wins_over_a_later_terminal_one(journal):
    streamed = an_update(**FILLED, average_price=812.4, source="stream")
    polled = an_update(**FILLED, average_price=0.0, source="poll", ts_recv_ns=LATER_NS)
    journal.record_update(streamed)
    journal.record_update(polled)
    assert journal.last_update("o1") == streamed


def test_a_priced_fill_replaces_an_earlier_fill_reported_at_price_0(journal):
    polled = an_update(**FILLED, average_price=0.0, source="poll")
    streamed = an_update(**FILLED, average_price=812.4, source="stream", ts_recv_ns=LATER_NS)
    journal.record_update(polled)
    journal.record_update(streamed)
    assert journal.last_update("o1") == streamed


@pytest.mark.parametrize(
    "later",
    [
        an_update(**FILLED | {"filled_quantity": 6}, average_price=812.4, ts_recv_ns=LATER_NS),
        an_update(status=OrderStatus.CANCELLED, average_price=0.0, ts_recv_ns=LATER_NS),
    ],
    ids=["fewer units filled", "no price either"],
)
def test_a_fill_at_price_0_stays_final_against_a_later_update_that_gives_no_better(journal, later):
    polled = an_update(**FILLED, average_price=0.0, source="poll")
    journal.record_update(polled)
    journal.record_update(later)
    assert journal.last_update("o1") == polled


def test_an_update_may_arrive_before_its_order_is_bound(journal):
    journal.record_intent(an_intent())
    journal.record_update(an_update())
    journal.bind("i1", "o1")
    assert journal.open_orders() == [BoundOrder(an_intent(), "o1", an_update())]


def test_open_orders_are_the_bound_orders_with_no_terminal_update(journal):
    for number in range(1, 5):
        a_bound_order(journal, f"i{number}", f"o{number}")
    journal.record_update(an_update("o2", OrderStatus.TRIGGER_PENDING))
    journal.record_update(an_update("o3", OrderStatus.COMPLETE))
    journal.record_update(an_update("o4", OrderStatus.OPEN))
    journal.record_update(an_update("o4", OrderStatus.REJECTED))
    journal.record_update(an_update("o5", OrderStatus.OPEN))  # an order the engine did not send
    assert journal.open_orders() == [
        BoundOrder(an_intent("i1"), "o1", None),
        BoundOrder(an_intent("i2"), "o2", an_update("o2", OrderStatus.TRIGGER_PENDING)),
    ]


def test_a_filled_order_stays_in_the_bound_orders(journal):
    filled = an_update(status=OrderStatus.COMPLETE, filled_quantity=10, average_price=812.4)
    a_bound_order(journal, "i1", "o1")
    journal.record_update(filled)
    assert journal.bound_orders() == [BoundOrder(an_intent(), "o1", filled)]
    assert journal.open_orders() == []

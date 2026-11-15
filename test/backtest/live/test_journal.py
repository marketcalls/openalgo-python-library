"""The live journal's order record (08-live-trading 6.1 and 7): intents written before
their orders are sent, bindings, order updates, the values each field keeps, the trading
day each intent belongs to, and one ledger per mode."""

from datetime import timedelta

import pytest
from journal_samples import (
    BROKER,
    DAY,
    DAY_NS,
    LEG,
    NEXT_DAY,
    ManualClock,
    a_bound_order,
    an_intent,
    an_update,
    ist_ns,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest._live._types import JournalLike, Mode, OrderStatus, Purpose
from openalgo.backtest._live.journal import Journal, JournalOrder
from openalgo.backtest.errors import JournalError


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


@pytest.fixture
def journal(path):
    with started_journal(path) as handle:
        yield handle


def test_the_journal_offers_what_the_order_path_needs():
    needed = [name for name in vars(JournalLike) if not name.startswith("_")]
    assert "mark_not_placed" in needed
    assert all(callable(getattr(Journal, name, None)) for name in needed)


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
    a_bound_order(journal, "s1", "o9", purpose=Purpose.STOP)
    journal.set_stop(*LEG, "o9")


COMMITTED_WRITES = [
    pytest.param(record_an_intent, lambda j: j.unbound(day=DAY) == [an_intent()], id="intent"),
    pytest.param(bind_an_intent, lambda j: j.order_id_of("i1") == "o1", id="binding"),
    pytest.param(mark_an_intent_not_placed, lambda j: j.unbound(day=DAY) == [], id="not placed"),
    pytest.param(
        lambda j: j.record_update(an_update()),
        lambda j: j.last_update("o1") == an_update(),
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
        lambda j: j.user_state("ema").state == {"entries": 1},
        id="user state",
    ),
    pytest.param(
        lambda j: j.start_session(DAY, Mode.SANDBOX, BROKER),
        lambda j: j.session(DAY) is not None,
        id="session",
    ),
]


@pytest.mark.parametrize(("write", "is_visible"), COMMITTED_WRITES)
def test_every_write_is_committed_before_it_returns(path, write, is_visible):
    with started_journal(path) as writer, open_journal(path) as reader:
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


def test_an_unbound_intent_found_in_the_order_book_is_bound(journal):
    journal.record_intent(an_intent())
    journal.mark_unbound("i1", "timeout")
    journal.bind("i1", "o1")
    assert (journal.unbound(), journal.order_id_of("i1")) == ([], "o1")


def test_a_bound_intent_maps_to_its_order_and_back(journal):
    a_bound_order(journal, "i1", "o1")
    assert (journal.order_id_of("i1"), journal.intent_of("o1")) == ("o1", an_intent())


def test_an_intent_reads_back_as_it_was_recorded(journal):
    stop = an_intent(
        "s1",
        action="SELL",
        price_type="SL",
        purpose=Purpose.STOP,
        price=95.25,
        trigger_price=100.25,
    )
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


def test_binding_an_order_to_a_second_intent_is_refused(journal):
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


@pytest.mark.parametrize("mark", [Journal.mark_unbound, Journal.mark_not_placed])
def test_a_bound_intent_cannot_be_marked(journal, mark):
    a_bound_order(journal, "i1", "o1")
    with pytest.raises(JournalError, match="bound to order 'o1', so it reached the broker"):
        mark(journal, "i1", "timeout")


@pytest.mark.parametrize("mark", [Journal.mark_unbound, Journal.mark_not_placed])
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


def test_an_update_may_arrive_before_its_order_is_bound(journal):
    journal.record_intent(an_intent())
    journal.record_update(an_update())
    journal.bind("i1", "o1")
    assert journal.open_orders() == [JournalOrder("o1", an_intent(), an_update())]


def test_an_update_from_another_mode_is_refused(journal):
    with pytest.raises(JournalError, match="updated in live mode, and this is the sandbox ledger"):
        journal.record_update(an_update(mode=Mode.LIVE))
    assert journal.updates("o1") == []


def test_open_orders_are_the_bound_orders_with_no_terminal_update(journal):
    for number in range(1, 5):
        a_bound_order(journal, f"i{number}", f"o{number}")
    journal.record_update(an_update("o2", OrderStatus.TRIGGER_PENDING))
    journal.record_update(an_update("o3", OrderStatus.COMPLETE))
    journal.record_update(an_update("o4", OrderStatus.OPEN))
    journal.record_update(an_update("o4", OrderStatus.REJECTED))
    journal.record_update(an_update("o5", OrderStatus.OPEN))  # an order the engine did not send
    assert journal.open_orders() == [
        JournalOrder("o1", an_intent("i1"), None),
        JournalOrder("o2", an_intent("i2"), an_update("o2", OrderStatus.TRIGGER_PENDING)),
    ]


def test_a_filled_order_stays_in_the_bound_orders(journal):
    filled = an_update(status=OrderStatus.COMPLETE, filled_quantity=10, average_price=812.4)
    a_bound_order(journal, "i1", "o1")
    journal.record_update(filled)
    assert journal.bound_orders() == [JournalOrder("o1", an_intent(), filled)]
    assert journal.open_orders() == []


# The trading day


def test_an_intent_belongs_to_the_ist_trading_day_it_was_created_on(path):
    # 00:30 IST on DAY is 19:00 UTC the day before: the IST day is the one that counts.
    just_after_midnight = an_intent(created_ns=ist_ns(DAY, 0, 30))
    with open_journal(path) as journal:
        journal.record_intent(just_after_midnight)
        assert journal.unbound(day=DAY) == [just_after_midnight]
        assert journal.unbound(day=DAY - timedelta(days=1)) == []


def test_the_unbound_intents_are_those_of_the_sessions_day(journal):
    journal.record_intent(an_intent("today"))
    journal.record_intent(an_intent("tomorrow", created_ns=ist_ns(NEXT_DAY, 10)))
    assert [intent.intent_id for intent in journal.unbound()] == ["today"]


def test_the_orders_are_those_of_the_sessions_day(path):
    with started_journal(path) as journal:
        a_bound_order(journal, "i1", "o1")
    with started_journal(path, day=NEXT_DAY, now=ManualClock(ist_ns(NEXT_DAY, 9))) as next_day:
        a_bound_order(next_day, "i2", "o2", created_ns=ist_ns(NEXT_DAY, 10))
        assert next_day.open_orders() == [
            JournalOrder("o2", an_intent("i2", created_ns=ist_ns(NEXT_DAY, 10)), None)
        ]
        assert next_day.bound_orders(day=DAY) == [JournalOrder("o1", an_intent("i1"), None)]


@pytest.mark.parametrize(
    "read",
    [
        Journal.unbound,
        Journal.open_orders,
        Journal.bound_orders,
        Journal.snapshot,
        lambda j: j.stop_of(*LEG),
        lambda j: j.clear_stop(*LEG),
        lambda j: j.rms_state("ema"),
        lambda j: j.put_rms_state("ema", {}),
    ],
    ids=[
        "unbound",
        "open_orders",
        "bound_orders",
        "snapshot",
        "stop_of",
        "clear_stop",
        "rms_state",
        "put_rms_state",
    ],
)
def test_what_belongs_to_a_day_needs_the_session_or_a_day(path, read):
    with (
        open_journal(path) as journal,
        pytest.raises(JournalError, match="no trading day: call start_session first, or pass day="),
    ):
        read(journal)


# Modes


def test_an_intent_id_is_unique_across_both_ledgers(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        sandbox.record_intent(an_intent())
    with (
        open_journal(path, mode=Mode.LIVE) as live,
        pytest.raises(JournalError, match="'i1' is already in the journal"),
    ):
        live.record_intent(an_intent())


def test_an_order_of_one_ledger_is_unknown_in_the_other(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        a_bound_order(sandbox, "i1", "o1")
    with open_journal(path, mode=Mode.LIVE) as live:
        assert (live.intent("i1"), live.intent_of("o1"), live.bound_orders(day=DAY)) == (
            None,
            None,
            [],
        )

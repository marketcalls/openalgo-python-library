"""The trading day and the mode in the live journal: an intent is recorded only on the
IST day of the session started on the handle, what a restart reads is one day's, an
order ID given again on a later day is a new order, and each mode keeps its own ledger."""

from datetime import timedelta

import pytest
from journal_samples import (
    DAY,
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

from openalgo.backtest._live._types import BoundOrder, Mode, OrderStatus
from openalgo.backtest._live.journal import Journal
from openalgo.backtest.errors import JournalError

NEXT_DAY_NS = ist_ns(NEXT_DAY, 10)


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


@pytest.fixture
def journal(path):
    with started_journal(path) as handle:
        yield handle


def next_day_journal(path):
    return started_journal(path, day=NEXT_DAY, now=ManualClock(NEXT_DAY_NS))


def test_an_intent_is_filed_under_the_ist_day_it_was_created_on(journal):
    # 00:30 IST on DAY is 19:00 UTC the day before: the IST day is the one that counts.
    just_after_midnight = an_intent(created_ns=ist_ns(DAY, 0, 30))
    journal.record_intent(just_after_midnight)
    assert journal.unbound() == [just_after_midnight]


@pytest.mark.parametrize(
    "created_ns",
    [ist_ns(DAY - timedelta(days=1), 23, 30), NEXT_DAY_NS],
    ids=["the evening before", "the next day"],
)
def test_an_intent_of_another_day_than_the_sessions_is_refused(journal, created_ns):
    with pytest.raises(JournalError, match="start the session for the trading day it trades"):
        journal.record_intent(an_intent(created_ns=created_ns))
    assert journal.intent("i1") is None


def test_the_unbound_intents_are_those_of_the_sessions_day(path):
    with started_journal(path) as journal:
        journal.record_intent(an_intent("today"))
    with next_day_journal(path) as next_day:
        next_day.record_intent(an_intent("tomorrow", created_ns=NEXT_DAY_NS))
        assert [intent.intent_id for intent in next_day.unbound()] == ["tomorrow"]
        assert [intent.intent_id for intent in next_day.unbound(day=DAY)] == ["today"]


def test_the_orders_are_those_of_the_sessions_day(path):
    with started_journal(path) as journal:
        a_bound_order(journal, "i1", "o1")
    with next_day_journal(path) as next_day:
        a_bound_order(next_day, "i2", "o2", created_ns=NEXT_DAY_NS)
        assert next_day.open_orders() == [
            BoundOrder(an_intent("i2", created_ns=NEXT_DAY_NS), "o2", None)
        ]
        assert next_day.bound_orders(day=DAY) == [BoundOrder(an_intent("i1"), "o1", None)]


ONE_SECOND_NS = 1_000_000_000
NEXT_DAY_MIDNIGHT_NS = ist_ns(NEXT_DAY, 0)


def test_an_update_after_midnight_of_an_evening_order_stays_with_its_session(path):
    # An MCX order bound at 23:50 IST, cancelled by the exchange a second after midnight.
    crude = {"exchange": "MCX", "symbol": "CRUDEOIL"}
    evening = an_intent(**crude, created_ns=ist_ns(DAY, 23, 50))
    after_midnight = NEXT_DAY_MIDNIGHT_NS + ONE_SECOND_NS
    cancel = an_update(**crude, status=OrderStatus.CANCELLED, ts_recv_ns=after_midnight)
    with started_journal(path, now=ManualClock(ist_ns(DAY, 23, 50))) as journal:
        journal.record_intent(evening)
        journal.bind("i1", "o1")
        journal.record_update(cancel)
        assert (journal.open_orders(), journal.last_update("o1")) == ([], cancel)


def test_an_update_of_an_order_never_bound_is_kept_under_the_day_it_arrived(journal):
    arrived = an_update("o9", ts_recv_ns=NEXT_DAY_MIDNIGHT_NS + ONE_SECOND_NS)
    journal.record_update(arrived)
    assert (journal.updates("o9"), journal.updates("o9", day=NEXT_DAY)) == ([], [arrived])


def test_an_order_id_given_again_on_a_later_day_is_bound_again(path):
    with started_journal(path) as journal:
        a_bound_order(journal, "i1", "o1")
        journal.record_update(an_update(status=OrderStatus.COMPLETE, filled_quantity=10))
    with next_day_journal(path) as next_day:
        a_bound_order(next_day, "i2", "o1", created_ns=NEXT_DAY_NS)
        assert next_day.bound_orders() == [
            BoundOrder(an_intent("i2", created_ns=NEXT_DAY_NS), "o1", None)
        ]
        assert (next_day.last_update("o1"), next_day.updates("o1")) == (None, [])
        assert next_day.intent_of("o1", day=DAY) == an_intent("i1")


@pytest.mark.parametrize(
    "read",
    [
        Journal.unbound,
        Journal.open_orders,
        Journal.bound_orders,
        Journal.snapshot,
        lambda j: j.record_intent(an_intent()),
        lambda j: j.intent_of("o1"),
        lambda j: j.last_update("o1"),
        lambda j: j.updates("o1"),
        lambda j: j.set_stop(*LEG, "o1"),
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
        "record_intent",
        "intent_of",
        "last_update",
        "updates",
        "set_stop",
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
    with started_journal(path, mode=Mode.SANDBOX) as sandbox:
        sandbox.record_intent(an_intent())
    with (
        started_journal(path, mode=Mode.LIVE) as live,
        pytest.raises(JournalError, match="'i1' is already in the journal"),
    ):
        live.record_intent(an_intent())


def test_an_order_of_one_ledger_is_unknown_in_the_other(path):
    with started_journal(path, mode=Mode.SANDBOX) as sandbox:
        a_bound_order(sandbox, "i1", "o1")
    with started_journal(path, mode=Mode.LIVE) as live:
        found = (live.intent("i1"), live.intent_of("o1"), live.bound_orders())
        assert found == (None, None, [])


def test_an_update_of_the_other_mode_is_kept_in_its_own_ledger(path):
    live_fill = an_update(status=OrderStatus.COMPLETE, mode=Mode.LIVE, filled_quantity=10)
    with started_journal(path, mode=Mode.SANDBOX) as sandbox:
        sandbox.record_update(live_fill)
        assert sandbox.updates("o1") == []
    with started_journal(path, mode=Mode.LIVE) as live:
        assert live.updates("o1") == [live_fill]


def test_an_update_of_the_other_mode_never_changes_this_ledgers_order(journal):
    a_bound_order(journal, "i1", "o1")
    journal.record_update(an_update(status=OrderStatus.CANCELLED, mode=Mode.LIVE))
    assert journal.open_orders() == [BoundOrder(an_intent(), "o1", None)]

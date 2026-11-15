"""What the live journal keeps besides orders (08-live-trading 6.2 and 7): each leg's
broker-held stop, the risk rules' state of the day, the strategy's `self.state`, the
sessions, and the snapshot a restart restores."""

import math
import re
from datetime import date, datetime

import numpy as np
import pytest
from journal_samples import (
    BROKER,
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

from openalgo.backtest._live._types import Mode, OrderStatus, Purpose
from openalgo.backtest._live.journal import (
    JournalOrder,
    JournalSession,
    JournalSnapshot,
    SavedState,
)
from openalgo.backtest.errors import JournalError


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


@pytest.fixture
def journal(path):
    with started_journal(path) as handle:
        yield handle


def a_stop_order(journal, order_id, **changes):
    a_bound_order(journal, f"intent-of-{order_id}", order_id, purpose=Purpose.STOP, **changes)


# Stops


def test_a_stop_is_recorded_read_and_cleared(journal):
    a_stop_order(journal, "s1")
    journal.set_stop(*LEG, "s1")
    assert journal.stop_of(*LEG) == "s1"
    journal.clear_stop(*LEG)
    assert journal.stop_of(*LEG) is None


def test_a_stop_needs_its_order_bound_in_this_ledger(journal):
    with pytest.raises(JournalError, match="stop order 's1' is not bound to an intent"):
        journal.set_stop(*LEG, "s1")


def test_a_stop_bound_in_the_other_ledger_is_refused(path):
    with started_journal(path, mode=Mode.LIVE) as live:
        a_stop_order(live, "s1")
    with (
        started_journal(path, mode=Mode.SANDBOX) as sandbox,
        pytest.raises(JournalError, match="not bound to an intent in the sandbox ledger"),
    ):
        sandbox.set_stop(*LEG, "s1")


def test_a_second_stop_on_a_leg_is_refused(journal):
    a_stop_order(journal, "s1")
    a_stop_order(journal, "s2")
    journal.set_stop(*LEG, "s1")
    with pytest.raises(JournalError, match="already has stop order 's1': a leg has one exit owner"):
        journal.set_stop(*LEG, "s2")


def test_recording_the_same_stop_twice_changes_nothing(journal):
    a_stop_order(journal, "s1")
    journal.set_stop(*LEG, "s1")
    journal.set_stop(*LEG, "s1")
    assert journal.stop_of(*LEG) == "s1"


@pytest.mark.parametrize(
    "other_leg", [("orb", "SBIN", "NSE"), ("ema", "INFY", "NSE"), ("ema", "SBIN", "BSE")]
)
def test_each_leg_has_its_own_stop(journal, other_leg):
    a_stop_order(journal, "s1")
    a_stop_order(journal, "s2")
    journal.set_stop(*LEG, "s1")
    journal.set_stop(*other_leg, "s2")
    assert (journal.stop_of(*LEG), journal.stop_of(*other_leg)) == ("s1", "s2")


def test_clearing_a_leg_with_no_stop_changes_nothing(journal):
    journal.clear_stop(*LEG)
    assert journal.stop_of(*LEG) is None


def test_a_stop_belongs_to_the_trading_day_of_its_order(path):
    with started_journal(path) as journal:
        a_stop_order(journal, "s1")
        journal.set_stop(*LEG, "s1")
    with started_journal(path, day=NEXT_DAY) as next_day:
        a_stop_order(next_day, "s2", created_ns=ist_ns(NEXT_DAY, 10))
        next_day.set_stop(*LEG, "s2")
        assert (next_day.stop_of(*LEG), next_day.stop_of(*LEG, day=DAY)) == ("s2", "s1")


# Saved states

JSON_VALUES = {
    "count": 3,
    "peak": 101.25,
    "name": "ORB",
    "armed": True,
    "last_exit": None,
    "levels": [99.5, [100, 101]],
    "by_symbol": {"SBIN": {"entries": []}},
}


def test_the_risk_state_reads_back_after_a_reopen(path):
    with started_journal(path) as journal:
        journal.put_rms_state("ema", JSON_VALUES)
    with started_journal(path) as reopened:
        assert reopened.rms_state("ema") == JSON_VALUES


def test_self_state_reads_back_after_a_reopen_with_when_it_was_saved(path):
    with started_journal(path, now=ManualClock(ist_ns(DAY, 15, 20))) as journal:
        journal.put_user_state("ema", JSON_VALUES)
    with open_journal(path) as reopened:
        assert reopened.user_state("ema") == SavedState(JSON_VALUES, ist_ns(DAY, 15, 20))


def test_the_risk_state_does_not_carry_into_the_next_day(path):
    with started_journal(path) as journal:
        journal.put_rms_state("ema", {"daily_realized": -1_250.0})
    with started_journal(path, day=NEXT_DAY) as next_day:
        assert next_day.rms_state("ema") is None
        assert next_day.rms_state("ema", day=DAY) == {"daily_realized": -1_250.0}


def test_self_state_carries_across_days(path):
    with started_journal(path) as journal:
        journal.put_user_state("ema", {"entries": 2})
    with started_journal(path, day=NEXT_DAY) as next_day:
        assert next_day.user_state("ema").state == {"entries": 2}


def test_saving_a_state_replaces_the_last(journal):
    journal.put_user_state("ema", {"entries": 1, "armed": True})
    journal.put_user_state("ema", {"entries": 2})
    assert journal.user_state("ema").state == {"entries": 2}


@pytest.mark.parametrize(
    ("state", "named"),
    [
        ({"levels": (1, 2)}, "self.state['levels'] has the type tuple"),
        ({"seen": {"SBIN"}}, "self.state['seen'] has the type set"),
        ({"by_strike": {100: "CE"}}, "self.state['by_strike'] has the key 100"),
        ({"peak": math.nan}, "self.state['peak'] is nan"),
        ({"peak": math.inf}, "self.state['peak'] is inf"),
        ({"when": datetime(2026, 11, 16, 9, 15)}, "self.state['when'] has the type datetime"),
        ({"count": np.int64(3)}, "self.state['count'] has the type int64"),
        ({"legs": [{}, {"expiry": date(2026, 11, 26)}]}, "self.state['legs'][1]['expiry']"),
    ],
    ids=["tuple", "set", "int key", "nan", "inf", "datetime", "numpy int", "nested date"],
)
def test_a_value_json_does_not_keep_is_refused_naming_its_key(journal, state, named):
    with pytest.raises(JournalError, match=re.escape(named)):
        journal.put_user_state("ema", state)


def test_a_risk_state_json_does_not_keep_is_refused_naming_its_key(journal):
    with pytest.raises(JournalError, match=re.escape("rms_state['trail'] has the type tuple")):
        journal.put_rms_state("ema", {"trail": (99.0, 1_000)})


def test_a_state_that_holds_itself_is_refused(journal):
    state = {"levels": []}
    state["levels"].append(state)
    with pytest.raises(JournalError, match=re.escape("self.state['levels'][0] holds itself")):
        journal.put_user_state("ema", state)


def test_a_state_that_is_not_a_dict_is_refused(journal):
    with pytest.raises(JournalError, match="must be a dict, not a list"):
        journal.put_user_state("ema", [1, 2])


def test_a_refused_state_leaves_the_last_saved_one(journal):
    journal.put_user_state("ema", {"entries": 1})
    with pytest.raises(JournalError):
        journal.put_user_state("ema", {"entries": {2}})
    assert journal.user_state("ema").state == {"entries": 1}


# Sessions


def test_the_first_start_of_a_day_records_its_session(path):
    with open_journal(path, now=ManualClock(5_000)) as journal:
        started = journal.start_session(DAY, Mode.SANDBOX, BROKER)
        assert (
            started
            == journal.session(DAY)
            == JournalSession(DAY, Mode.SANDBOX, BROKER, 5_000, 5_000, 1)
        )


def test_a_restart_counts_one_more_start_and_keeps_the_first(path):
    clock = ManualClock(5_000)
    with open_journal(path, now=clock) as journal:
        journal.start_session(DAY, Mode.SANDBOX, BROKER)
    clock.ns = 9_000
    with open_journal(path, now=clock) as restarted:
        assert restarted.start_session(DAY, Mode.SANDBOX, BROKER) == JournalSession(
            DAY, Mode.SANDBOX, BROKER, 5_000, 9_000, 2
        )


def test_each_day_has_its_own_session(journal):
    assert journal.session(NEXT_DAY) is None


@pytest.mark.parametrize(
    ("mode", "broker"), [(Mode.LIVE, BROKER), (Mode.SANDBOX, "dhan")], ids=["mode", "broker"]
)
def test_a_session_unlike_the_journal_is_refused(path, mode, broker):
    with (
        open_journal(path) as journal,
        pytest.raises(JournalError, match="this journal was opened for the sandbox ledger"),
    ):
        journal.start_session(DAY, mode, broker)


def test_a_session_day_must_be_a_date(path):
    with (
        open_journal(path) as journal,
        pytest.raises(JournalError, match="day must be a datetime.date"),
    ):
        journal.start_session(datetime(2026, 11, 16, 9, 15), Mode.SANDBOX, BROKER)


# Snapshots


def fill_for_a_restart(journal):
    journal.record_intent(an_intent("i1"))
    journal.mark_unbound("i1", "HTTP 502 on placeorder")
    a_bound_order(journal, "i2", "o2")
    journal.record_update(an_update("o2", OrderStatus.TRIGGER_PENDING))
    a_bound_order(journal, "i3", "o3")
    journal.record_update(an_update("o3", OrderStatus.COMPLETE, filled_quantity=10))
    journal.record_intent(an_intent("i4"))
    journal.set_stop(*LEG, "o2")
    journal.put_rms_state("ema", {"peak": 104.5, "daily_realized": -250.0})
    journal.put_user_state("ema", {"entries": 1})


def test_a_snapshot_holds_what_a_restart_restores(path):
    with started_journal(path, now=ManualClock(ist_ns(DAY, 11))) as journal:
        fill_for_a_restart(journal)
        working = JournalOrder("o2", an_intent("i2"), an_update("o2", OrderStatus.TRIGGER_PENDING))
        filled = JournalOrder(
            "o3", an_intent("i3"), an_update("o3", OrderStatus.COMPLETE, filled_quantity=10)
        )
        assert journal.snapshot() == JournalSnapshot(
            day=DAY,
            unbound=(an_intent("i1"), an_intent("i4")),
            open_orders=(working,),
            bound_orders=(working, filled),
            stops={LEG: "o2"},
            rms_states={"ema": {"peak": 104.5, "daily_realized": -250.0}},
            user_states={"ema": SavedState({"entries": 1}, ist_ns(DAY, 11))},
        )


def test_a_snapshot_after_close_and_reopen_equals_the_one_before(path):
    with started_journal(path) as journal:
        fill_for_a_restart(journal)
        before = journal.snapshot()
    with open_journal(path) as reopened:
        assert reopened.snapshot(day=DAY) == before


def test_a_snapshot_is_of_one_trading_day(path):
    with started_journal(path, now=ManualClock(ist_ns(DAY, 11))) as journal:
        fill_for_a_restart(journal)
    with started_journal(path, day=NEXT_DAY) as next_day:
        assert next_day.snapshot() == JournalSnapshot(
            day=NEXT_DAY,
            unbound=(),
            open_orders=(),
            bound_orders=(),
            stops={},
            rms_states={},
            user_states={"ema": SavedState({"entries": 1}, ist_ns(DAY, 11))},
        )


def test_each_mode_keeps_its_own_ledger(path):
    with started_journal(path, mode=Mode.SANDBOX) as sandbox:
        fill_for_a_restart(sandbox)
    with started_journal(path, mode=Mode.LIVE) as live:
        assert live.snapshot() == JournalSnapshot(DAY, (), (), (), {}, {}, {})


def test_a_snapshot_names_the_filled_order_a_crash_left_without_a_stop(path):
    with started_journal(path) as journal:
        a_bound_order(journal, "entry", "o1")
        journal.record_update(an_update(status=OrderStatus.COMPLETE, filled_quantity=10))
    with open_journal(path) as restarted:
        (order,) = restarted.snapshot(day=DAY).bound_orders
        assert (order.order_id, order.terminal, order.last_update.filled_quantity) == (
            "o1",
            True,
            10,
        )

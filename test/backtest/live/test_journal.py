"""The live journal (08-live-trading 6.1 and 7): intents written before their orders are
sent, bindings, order updates, stops, saved states and sessions, all kept across a
restart, in one ledger per mode, in a file bound to its account and broker."""

import math
import re
import sqlite3
from datetime import date, datetime

import numpy as np
import pytest

from openalgo.backtest._live._types import (
    JournalLike,
    Mode,
    OrderIntent,
    OrderStatus,
    OrderUpdate,
    Purpose,
)
from openalgo.backtest._live.journal import (
    Journal,
    JournalOrder,
    JournalSession,
    JournalSnapshot,
    default_journal_path,
)
from openalgo.backtest.errors import JournalError

ACCOUNT = "trader1"
BROKER = "zerodha"
DAY = date(2026, 11, 16)
LEG = ("ema", "SBIN", "NSE")


class ManualClock:
    """UTC nanoseconds that move only when a test sets them."""

    def __init__(self, ns: int = 1_000) -> None:
        self.ns = ns

    def __call__(self) -> int:
        return self.ns


def open_journal(path, *, broker=BROKER, mode=Mode.SANDBOX, account=ACCOUNT, now=None):
    return Journal.open(path, account=account, broker=broker, mode=mode, now=now or ManualClock())


@pytest.fixture
def path(tmp_path):
    return tmp_path / "live" / ACCOUNT / "journal.sqlite"


@pytest.fixture
def journal(path):
    with open_journal(path) as handle:
        yield handle


def an_intent(intent_id="i1", **changes):
    values = {
        "intent_id": intent_id,
        "strategy": "ema",
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "price_type": "MARKET",
        "product": "MIS",
        "purpose": Purpose.ENTRY,
        "created_ns": 1_000,
    }
    return OrderIntent(**values | changes)


def an_update(order_id="o1", status=OrderStatus.OPEN, **changes):
    values = {
        "order_id": order_id,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "filled_quantity": 0,
        "pending_quantity": 10,
        "average_price": 0.0,
        "status": status,
        "rejection_reason": "",
        "mode": Mode.SANDBOX,
        "ts_recv_ns": 2_000,
        "source": "stream",
    }
    return OrderUpdate(**values | changes)


def a_bound_order(journal, intent_id, order_id):
    journal.record_intent(an_intent(intent_id))
    journal.bind(intent_id, order_id)


def set_user_version(path, version):
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()


# Intents and bindings


def test_the_journal_offers_what_the_order_path_needs():
    needed = [name for name in vars(JournalLike) if not name.startswith("_")]
    assert needed and all(callable(getattr(Journal, name, None)) for name in needed)


def test_a_recorded_intent_is_found_unbound_after_a_reopen(path):
    with open_journal(path) as journal:
        journal.record_intent(an_intent())
    with open_journal(path) as reopened:
        assert reopened.unbound() == [an_intent()]


def record_an_intent(journal):
    journal.record_intent(an_intent())


def bind_an_intent(journal):
    a_bound_order(journal, "i1", "o1")


def mark_an_intent_not_placed(journal):
    journal.record_intent(an_intent())
    journal.mark_not_placed("i1", "insufficient funds")


COMMITTED_WRITES = [
    pytest.param(record_an_intent, lambda j: j.unbound() == [an_intent()], id="intent"),
    pytest.param(bind_an_intent, lambda j: j.order_id_of("i1") == "o1", id="binding"),
    pytest.param(mark_an_intent_not_placed, lambda j: j.unbound() == [], id="not placed"),
    pytest.param(
        lambda j: j.record_update(an_update()),
        lambda j: j.last_update("o1") == an_update(),
        id="update",
    ),
    pytest.param(lambda j: j.set_stop(*LEG, "s1"), lambda j: j.stop_of(*LEG) == "s1", id="stop"),
    pytest.param(
        lambda j: j.put_rms_state("ema", {"peak": 1.5}),
        lambda j: j.rms_state("ema") == {"peak": 1.5},
        id="risk state",
    ),
    pytest.param(
        lambda j: j.put_user_state("ema", {"entries": 1}),
        lambda j: j.user_state("ema") == {"entries": 1},
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
    with open_journal(path) as writer, open_journal(path) as reader:
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
        journal.session(DAY),
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
    journal.record_update(an_update(status=OrderStatus.OPEN, ts_recv_ns=3_000))
    assert journal.last_update("o1") == an_update(status=OrderStatus.CANCELLED)


def test_an_update_may_arrive_before_its_order_is_bound(journal):
    journal.record_intent(an_intent())
    journal.record_update(an_update())
    journal.bind("i1", "o1")
    assert journal.open_orders() == [JournalOrder("o1", an_intent(), an_update())]


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


# Stops


def test_a_stop_is_recorded_read_and_cleared(journal):
    journal.set_stop(*LEG, "s1")
    assert journal.stop_of(*LEG) == "s1"
    journal.clear_stop(*LEG)
    assert journal.stop_of(*LEG) is None


def test_a_second_stop_on_a_leg_is_refused(journal):
    journal.set_stop(*LEG, "s1")
    with pytest.raises(JournalError, match="already has stop order 's1': a leg has one exit owner"):
        journal.set_stop(*LEG, "s2")


def test_recording_the_same_stop_twice_changes_nothing(journal):
    journal.set_stop(*LEG, "s1")
    journal.set_stop(*LEG, "s1")
    assert journal.stop_of(*LEG) == "s1"


@pytest.mark.parametrize(
    "other_leg", [("orb", "SBIN", "NSE"), ("ema", "INFY", "NSE"), ("ema", "SBIN", "BSE")]
)
def test_each_leg_has_its_own_stop(journal, other_leg):
    journal.set_stop(*LEG, "s1")
    journal.set_stop(*other_leg, "s2")
    assert (journal.stop_of(*LEG), journal.stop_of(*other_leg)) == ("s1", "s2")


def test_clearing_a_leg_with_no_stop_changes_nothing(journal):
    journal.clear_stop(*LEG)
    assert journal.stop_of(*LEG) is None


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


@pytest.mark.parametrize(
    ("put", "read"),
    [(Journal.put_rms_state, Journal.rms_state), (Journal.put_user_state, Journal.user_state)],
    ids=["risk state", "user state"],
)
def test_a_saved_state_reads_back_after_a_reopen(path, put, read):
    with open_journal(path) as journal:
        put(journal, "ema", JSON_VALUES)
    with open_journal(path) as reopened:
        assert read(reopened, "ema") == JSON_VALUES


def test_saving_a_state_replaces_the_last(journal):
    journal.put_user_state("ema", {"entries": 1, "armed": True})
    journal.put_user_state("ema", {"entries": 2})
    assert journal.user_state("ema") == {"entries": 2}


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
    assert journal.user_state("ema") == {"entries": 1}


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
    journal.start_session(DAY, Mode.SANDBOX, BROKER)
    assert journal.session(date(2026, 11, 17)) is None


@pytest.mark.parametrize(
    ("mode", "broker"), [(Mode.LIVE, BROKER), (Mode.SANDBOX, "dhan")], ids=["mode", "broker"]
)
def test_a_session_unlike_the_journal_is_refused(journal, mode, broker):
    with pytest.raises(JournalError, match="this journal was opened for the sandbox ledger"):
        journal.start_session(DAY, mode, broker)


def test_a_session_day_must_be_a_date(journal):
    with pytest.raises(JournalError, match="day must be a datetime.date"):
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


def test_a_snapshot_holds_what_a_restart_restores(journal):
    fill_for_a_restart(journal)
    assert journal.snapshot() == JournalSnapshot(
        unbound=(an_intent("i1"), an_intent("i4")),
        open_orders=(
            JournalOrder("o2", an_intent("i2"), an_update("o2", OrderStatus.TRIGGER_PENDING)),
        ),
        stops={LEG: "o2"},
        rms_states={"ema": {"peak": 104.5, "daily_realized": -250.0}},
        user_states={"ema": {"entries": 1}},
    )


def test_a_snapshot_after_close_and_reopen_equals_the_one_before(path):
    with open_journal(path) as journal:
        fill_for_a_restart(journal)
        before = journal.snapshot()
    with open_journal(path) as reopened:
        assert reopened.snapshot() == before


# Modes


def test_each_mode_keeps_its_own_ledger(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        fill_for_a_restart(sandbox)
        sandbox.start_session(DAY, Mode.SANDBOX, BROKER)
    with open_journal(path, mode=Mode.LIVE) as live:
        empty = JournalSnapshot((), (), {}, {}, {})
        assert (live.snapshot(), live.intent_of("o2"), live.session(DAY)) == (empty, None, None)


def test_an_intent_id_is_unique_across_both_ledgers(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        sandbox.record_intent(an_intent())
    with (
        open_journal(path, mode=Mode.LIVE) as live,
        pytest.raises(JournalError, match="'i1' is already in the journal"),
    ):
        live.record_intent(an_intent())


# Opening the file


def test_a_new_journal_has_schema_version_1_and_a_write_ahead_log(path):
    open_journal(path).close()
    connection = sqlite3.connect(path)
    found = [
        connection.execute(f"PRAGMA {name}").fetchone()[0]
        for name in ("user_version", "journal_mode")
    ]
    connection.close()
    assert found == [1, "wal"]


def test_opening_creates_the_missing_folders(tmp_path):
    path = tmp_path / "a" / "b" / "journal.sqlite"
    open_journal(path).close()
    assert path.is_file()


def test_opening_with_another_broker_is_refused(path):
    open_journal(path).close()
    with pytest.raises(JournalError, match="bound to broker 'zerodha'.*would open a new position"):
        open_journal(path, broker="dhan")


def test_opening_with_another_account_is_refused(path):
    open_journal(path).close()
    with pytest.raises(JournalError, match="belongs to account 'trader1', not 'trader2'"):
        open_journal(path, account="trader2")


def test_a_journal_from_a_newer_version_is_refused(path):
    open_journal(path).close()
    set_user_version(path, 2)
    with pytest.raises(JournalError, match="written by a newer openalgo.*pip install -U openalgo"):
        open_journal(path)


def test_an_unknown_schema_version_is_refused(path):
    open_journal(path).close()
    set_user_version(path, -1)
    with pytest.raises(JournalError, match="unknown schema version -1"):
        open_journal(path)


def test_a_sqlite_file_that_is_not_a_journal_is_refused(path):
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE prices (symbol TEXT)")
    connection.close()
    with pytest.raises(JournalError, match="not an OpenAlgo live journal"):
        open_journal(path)


def test_a_file_that_is_not_sqlite_is_refused(path):
    path.parent.mkdir(parents=True)
    path.write_bytes(b"symbol,close\nSBIN,812.5\n" * 20)
    with pytest.raises(JournalError, match="file is not a database"):
        open_journal(path)


def test_a_folder_in_place_of_the_file_is_refused(path):
    path.mkdir(parents=True)
    with pytest.raises(JournalError, match="cannot open the live journal"):
        open_journal(path)


@pytest.mark.parametrize(
    ("argument", "value"), [("account", ""), ("broker", " "), ("broker", None)]
)
def test_a_blank_account_or_broker_is_refused(path, argument, value):
    with pytest.raises(JournalError, match=f"{argument} must be a non-empty string"):
        open_journal(path, **{argument: value})


def test_a_mode_that_is_not_a_mode_is_refused(path):
    with pytest.raises(JournalError, match="mode must be Mode.SANDBOX or Mode.LIVE"):
        open_journal(path, mode="live")


def test_an_sqlite_error_becomes_a_journal_error_that_says_what_to_do(journal):
    with pytest.raises(
        JournalError, match="cannot record the intent.*check that the disk"
    ) as raised:
        journal.record_intent(an_intent(symbol=None))
    assert isinstance(raised.value.__cause__, sqlite3.IntegrityError)


# Closing


@pytest.mark.parametrize(
    "call",
    [
        lambda j: j.record_intent(an_intent()),
        lambda j: j.unbound(),
        lambda j: j.record_update(an_update()),
        lambda j: j.set_stop(*LEG, "s1"),
        lambda j: j.put_user_state("ema", {}),
        lambda j: j.session(DAY),
        lambda j: j.snapshot(),
    ],
    ids=[
        "record_intent",
        "unbound",
        "record_update",
        "set_stop",
        "put_user_state",
        "session",
        "snapshot",
    ],
)
def test_a_closed_journal_refuses_every_call(path, call):
    journal = open_journal(path)
    journal.close()
    with pytest.raises(JournalError, match="is closed; open it again with Journal.open"):
        call(journal)


def test_closing_twice_is_harmless(path):
    journal = open_journal(path)
    journal.close()
    journal.close()


def test_leaving_the_with_block_closes_the_journal(path):
    with open_journal(path) as journal:
        pass
    with pytest.raises(JournalError, match="is closed"):
        journal.unbound()


# Where the journal lives


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALGO_LIVE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_the_default_journal_is_under_the_home_folder(home):
    expected = home / ".openalgo" / "live" / "trader1" / "journal.sqlite"
    assert default_journal_path("trader1") == expected


def test_the_live_dir_variable_moves_the_journals(home, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", str(home / "journals"))
    assert default_journal_path("trader1") == home / "journals" / "trader1" / "journal.sqlite"


def test_a_blank_live_dir_variable_counts_as_unset(home, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", " ")
    expected = home / ".openalgo" / "live" / "trader1" / "journal.sqlite"
    assert default_journal_path("trader1") == expected


@pytest.mark.parametrize("account", ["", "a/b", "a\\b", "C:x", ".", ".."])
def test_an_account_that_is_not_a_plain_folder_name_is_refused(account):
    with pytest.raises(JournalError, match=r"account.* must be"):
        default_journal_path(account)

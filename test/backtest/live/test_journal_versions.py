"""The journal's schema version 2 (18-multi-strategy 12.1; unit M11): each start's version of
each strategy (`versions`), the version id on every intent, and the upgrade in place of a
version 1 file, the layout openalgo 2.3.0 wrote."""

import sqlite3

import pytest
from journal_samples import (
    ACCOUNT,
    BROKER,
    DAY,
    DAY_NS,
    NEXT_DAY,
    ManualClock,
    an_intent,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest._live._types import Mode
from openalgo.backtest.errors import JournalError

#: The layout of a version 1 journal, as openalgo 2.3.0 created it.
VERSION_1 = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT",
    """CREATE TABLE intents (
        seq INTEGER PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE, ledger TEXT NOT NULL,
        day TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
        exchange TEXT NOT NULL, action TEXT NOT NULL, quantity INTEGER NOT NULL,
        price_type TEXT NOT NULL, product TEXT NOT NULL, purpose TEXT NOT NULL,
        created_ns INTEGER NOT NULL, price REAL NOT NULL, trigger_price REAL NOT NULL,
        state TEXT NOT NULL, order_id TEXT, reason TEXT NOT NULL,
        changed_ns INTEGER NOT NULL) STRICT""",
    "CREATE UNIQUE INDEX intents_by_order ON intents (ledger, day, order_id)",
    "CREATE INDEX intents_by_day ON intents (ledger, day, state, seq)",
    """CREATE TABLE updates (
        seq INTEGER PRIMARY KEY, ledger TEXT NOT NULL, day TEXT NOT NULL,
        order_id TEXT NOT NULL, symbol TEXT NOT NULL, exchange TEXT NOT NULL,
        action TEXT NOT NULL, quantity INTEGER NOT NULL, filled_quantity INTEGER NOT NULL,
        pending_quantity INTEGER NOT NULL, average_price REAL NOT NULL,
        status TEXT NOT NULL, terminal INTEGER NOT NULL, rejection_reason TEXT NOT NULL,
        mode TEXT NOT NULL, ts_recv_ns INTEGER NOT NULL, source TEXT NOT NULL) STRICT""",
    "CREATE INDEX updates_by_order ON updates (ledger, day, order_id, terminal, seq)",
    """CREATE TABLE stops (
        ledger TEXT NOT NULL, day TEXT NOT NULL, strategy TEXT NOT NULL,
        symbol TEXT NOT NULL, exchange TEXT NOT NULL, order_id TEXT NOT NULL,
        set_ns INTEGER NOT NULL, PRIMARY KEY (ledger, day, strategy, symbol, exchange)) STRICT""",
    """CREATE TABLE risk_states (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, day TEXT NOT NULL,
        state TEXT NOT NULL, saved_ns INTEGER NOT NULL,
        PRIMARY KEY (ledger, strategy, day)) STRICT""",
    """CREATE TABLE user_states (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, state TEXT NOT NULL,
        saved_ns INTEGER NOT NULL, PRIMARY KEY (ledger, strategy)) STRICT""",
    """CREATE TABLE sessions (
        ledger TEXT NOT NULL, day TEXT NOT NULL, broker TEXT NOT NULL,
        first_started_ns INTEGER NOT NULL, last_started_ns INTEGER NOT NULL,
        starts INTEGER NOT NULL, PRIMARY KEY (ledger, day)) STRICT""",
)
#: A manifest as a session writes one (`_session_versions`), cut to what these tests read.
MANIFEST = {"strategy": {"name": "ema", "parameters": {"fast": 9}}, "engine_version": "x"}


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


def version_1_file(path) -> None:
    """A version 1 journal of ACCOUNT at BROKER holding one bound intent of DAY."""
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    for statement in VERSION_1:
        connection.execute(statement)
    connection.executemany(
        "INSERT INTO meta VALUES (?, ?)", (("account", ACCOUNT), ("broker", BROKER))
    )
    connection.execute(
        "INSERT INTO intents (intent_id, ledger, day, strategy, symbol, exchange, action, "
        "quantity, price_type, product, purpose, created_ns, price, trigger_price, state, "
        "order_id, reason, changed_ns) VALUES ('old', 'sandbox', '2026-11-16', 'ema', 'SBIN', "
        f"'NSE', 'BUY', 10, 'MARKET', 'MIS', 'entry', {DAY_NS}, 0.0, 0.0, 'bound', '9', '', "
        f"{DAY_NS})"
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()


def user_version(path) -> int:
    connection = sqlite3.connect(path)
    value = connection.execute("PRAGMA user_version").fetchone()[0]
    connection.close()
    return value


# ----- the version id on every intent ----------------------------------------------------------


def test_an_intent_keeps_the_version_id_it_was_recorded_with(path):
    with started_journal(path) as journal:
        journal.record_intent(an_intent("i1", version_id="0a1b2c3d4e5f"))
        assert journal.intent("i1").version_id == "0a1b2c3d4e5f"


def test_an_intent_made_without_a_version_id_keeps_none(path):
    with started_journal(path) as journal:
        journal.record_intent(an_intent("i1"))
        assert journal.intent("i1").version_id == ""


# ----- each start's versions -------------------------------------------------------------------


def test_each_start_records_each_strategys_version_in_order(path):
    clock = ManualClock()
    with started_journal(path, now=clock) as journal:
        journal.record_version("ema", "0a1b2c3d4e5f", MANIFEST)
        clock.ns += 1_000
        journal.record_version("ema", "ffeeddccbbaa", MANIFEST)
        rows = journal.versions()
    assert [(row.strategy, row.version_id, row.started_ns) for row in rows] == [
        ("ema", "0a1b2c3d4e5f", DAY_NS),
        ("ema", "ffeeddccbbaa", DAY_NS + 1_000),
    ]
    assert rows[0].manifest == MANIFEST


def test_the_versions_of_a_day_are_that_days_and_that_ledgers_only(path):
    with started_journal(path) as journal:
        journal.record_version("ema", "0a1b2c3d4e5f", MANIFEST)
        assert journal.versions(day=NEXT_DAY) == []
    with open_journal(path, mode=Mode.LIVE) as live:
        assert live.versions(day=DAY) == []


def test_a_manifest_that_json_cannot_keep_is_refused(path):
    with (
        started_journal(path) as journal,
        pytest.raises(JournalError, match="manifest of strategy 'ema' cannot be saved"),
    ):
        journal.record_version("ema", "0a1b2c3d4e5f", {"when": DAY})


def test_a_version_needs_the_session_day(path):
    with (
        open_journal(path) as journal,
        pytest.raises(JournalError, match="call start_session first, or pass day="),
    ):
        journal.record_version("ema", "0a1b2c3d4e5f", MANIFEST)


# ----- the upgrade of a version 1 file ---------------------------------------------------------


def test_a_version_1_journal_is_upgraded_to_version_2_in_place(path):
    version_1_file(path)
    open_journal(path).close()
    assert user_version(path) == 2


def test_an_upgraded_journal_keeps_its_intents_with_no_version_id(path):
    version_1_file(path)
    with started_journal(path) as journal:
        old = journal.intent("old")
        assert (old.strategy, old.quantity, old.version_id) == ("ema", 10, "")
        assert journal.order_id_of("old") == "9"


def test_an_upgraded_journal_takes_versions_and_new_intents(path):
    version_1_file(path)
    with started_journal(path) as journal:
        journal.record_version("ema", "0a1b2c3d4e5f", MANIFEST)
        journal.record_intent(an_intent("new", version_id="0a1b2c3d4e5f"))
        assert [row.version_id for row in journal.versions()] == ["0a1b2c3d4e5f"]
        assert journal.intent("new").version_id == "0a1b2c3d4e5f"


def test_opening_an_upgraded_journal_again_changes_nothing(path):
    version_1_file(path)
    open_journal(path).close()
    with started_journal(path) as journal:
        assert journal.intent("old").version_id == ""
    assert user_version(path) == 2

"""The upgrade in place of a schema version 2 journal, the layout openalgo 2.4 writes, to
version 3 (20-positional-live, the lead's ruling N2-2): in the transaction that opens it,
its rows untouched, its version row updated; the tables of positions held overnight are
then there to use. A file of a later schema is refused (`test_journal_file.py`).
"""

from __future__ import annotations

import sqlite3

import pytest
from journal_samples import ACCOUNT, BROKER, DAY, DAY_NS, journal_path, open_journal
from journal_samples import started_journal as started

from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live.journal import GttTerms
from openalgo.backtest.errors import JournalError

#: The layout of a version 2 journal as openalgo 2.4 creates it (`_journal_file` of 2.4.1).
VERSION_2 = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT",
    """CREATE TABLE intents (
        seq INTEGER PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE, ledger TEXT NOT NULL,
        day TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
        exchange TEXT NOT NULL, action TEXT NOT NULL, quantity INTEGER NOT NULL,
        price_type TEXT NOT NULL, product TEXT NOT NULL, purpose TEXT NOT NULL,
        created_ns INTEGER NOT NULL, price REAL NOT NULL, trigger_price REAL NOT NULL,
        state TEXT NOT NULL, order_id TEXT, reason TEXT NOT NULL,
        changed_ns INTEGER NOT NULL, version_id TEXT NOT NULL DEFAULT '') STRICT""",
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
    """CREATE TABLE versions (
        ledger TEXT NOT NULL, day TEXT NOT NULL, strategy TEXT NOT NULL,
        version_id TEXT NOT NULL, manifest TEXT NOT NULL, started_ns INTEGER NOT NULL) STRICT""",
    "CREATE INDEX versions_by_day ON versions (ledger, day, strategy)",
)
#: One row in each table of version 2, as a session of 2026-11-16 left them.
ROWS = {
    "meta": [("account", ACCOUNT), ("broker", BROKER)],
    "intents": [
        (1, "i1", "sandbox", "2026-11-16", "ema", "SBIN", "NSE", "BUY", 10, "MARKET", "MIS",
         "entry", DAY_NS, 0.0, 0.0, "bound", "9", "", DAY_NS, "0a1b2c3d4e5f"),
    ],
    "updates": [
        (1, "sandbox", "2026-11-16", "9", "SBIN", "NSE", "BUY", 10, 10, 0, 812.5, "complete",
         1, "", "sandbox", DAY_NS + 1, "stream"),
    ],
    "stops": [("sandbox", "2026-11-16", "ema", "SBIN", "NSE", "10", DAY_NS + 2)],
    "risk_states": [("sandbox", "ema", "2026-11-16", '{"trail": 805.0}', DAY_NS + 3)],
    "user_states": [("sandbox", "ema", '{"count": 4}', DAY_NS + 4)],
    "sessions": [("sandbox", "2026-11-16", BROKER, DAY_NS, DAY_NS, 1)],
    "versions": [("sandbox", "2026-11-16", "ema", "0a1b2c3d4e5f", "{}", DAY_NS)],
}  # fmt: skip
#: The tables version 3 adds.
POSITIONAL_TABLES = {
    "day_closes",
    "decided_bars",
    "gtt_children",
    "gtt_states",
    "gtts",
    "kept_refusals",
}


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


def version_2_file(path, *also: str) -> None:
    """A version 2 journal of ACCOUNT at BROKER holding `ROWS`, with the statements `also`
    run in it after its layout."""
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    for statement in (*VERSION_2, *also):
        connection.execute(statement)
    for table, rows in ROWS.items():
        marks = ", ".join("?" * len(rows[0]))
        connection.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()


def read_file(path) -> tuple[int, set[str], dict[str, list[tuple]]]:
    """The file's schema version, its tables, and the rows of version 2's tables."""
    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        name
        for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    rows = {
        table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in ROWS
    }
    connection.close()
    return version, tables, rows


def test_a_version_2_journal_is_upgraded_to_version_3_with_its_rows_untouched(path):
    version_2_file(path)
    *_, before = read_file(path)

    open_journal(path).close()

    version, tables, after = read_file(path)
    assert version == 3
    assert tables >= POSITIONAL_TABLES
    assert after == before


def test_an_upgraded_journal_reads_its_old_rows_and_takes_the_new_ones(path):
    version_2_file(path)

    with started(path) as journal:
        assert journal.order_id_of("i1") == "9"
        assert journal.user_state("ema").state == {"count": 4}
        terms = GttTerms("SELL", 790.0, 780.0, 10)
        seq = journal.record_gtt("ema", 1, 7, "SBIN", "NSE", "CNC", terms, purpose=Purpose.STOP)
        journal.put_day_close("ema", {"positions": []}, {"count": 4}, "0a1b2c3d4e5f")

        assert journal.gtt(seq).terms.quantity == 10
        assert journal.day_close("ema").day == DAY


def test_an_upgrade_that_fails_part_way_leaves_the_file_at_version_2(path):
    # A table of version 3's name that is already there makes its step fail after the
    # step's first tables were made: the transaction undoes them all.
    version_2_file(path, "CREATE TABLE decided_bars (note TEXT)")

    with pytest.raises(JournalError, match="table decided_bars already exists"):
        open_journal(path)

    version, tables, _ = read_file(path)
    assert (version, "gtts" in tables) == (2, False)

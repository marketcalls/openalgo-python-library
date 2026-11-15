"""The live journal's file (08-live-trading 7): where it lives, its layout and version,
the account and broker it is bound to, closing it, the thread it belongs to, and the
error each failure raises."""

import sqlite3
import threading
from pathlib import Path

import pytest
from journal_samples import (
    ACCOUNT,
    DAY,
    LEG,
    a_bound_order,
    an_intent,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest._live._journal_file import failure
from openalgo.backtest._live.journal import default_journal_path
from openalgo.backtest.errors import JournalError


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


def run_sql(path, *statements):
    connection = sqlite3.connect(path)
    for statement in statements:
        connection.execute(statement)
    connection.commit()
    connection.close()


def read_pragma(path, name):
    connection = sqlite3.connect(path)
    value = connection.execute(f"PRAGMA {name}").fetchone()[0]
    connection.close()
    return value


# Opening the file


def test_a_new_journal_has_schema_version_1_and_a_write_ahead_log(path):
    open_journal(path).close()
    assert (read_pragma(path, "user_version"), read_pragma(path, "journal_mode")) == (1, "wal")


def test_every_table_is_strict(path):
    open_journal(path).close()
    connection = sqlite3.connect(path)
    tables = connection.execute(
        "SELECT name, strict FROM pragma_table_list WHERE schema = 'main' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    connection.close()
    assert len(tables) == 7
    assert {strict for _, strict in tables} == {1}


def test_opening_creates_the_missing_folders(tmp_path):
    path = tmp_path / "a" / "b" / "journal.sqlite"
    open_journal(path).close()
    assert path.is_file()


def test_opening_with_another_broker_is_refused(path):
    open_journal(path).close()
    with pytest.raises(JournalError, match="bound to broker 'zerodha'.*would open a new position"):
        open_journal(path, broker="dhan")


def test_the_broker_binding_error_suggests_a_journal_folder_for_the_new_broker(path):
    open_journal(path).close()
    with pytest.raises(JournalError, match="give 'dhan' a journal of its own .*OPENALGO_LIVE_DIR"):
        open_journal(path, broker="dhan")


def test_opening_with_another_account_is_refused(path):
    open_journal(path).close()
    with pytest.raises(JournalError, match="belongs to account 'trader1', not 'trader2'"):
        open_journal(path, account="trader2")


def test_a_journal_from_a_newer_version_is_refused(path):
    open_journal(path).close()
    run_sql(path, "PRAGMA user_version = 2")
    with pytest.raises(JournalError, match="written by a newer openalgo.*pip install -U openalgo"):
        open_journal(path)


def test_an_unknown_schema_version_is_refused(path):
    open_journal(path).close()
    run_sql(path, "PRAGMA user_version = -1")
    with pytest.raises(JournalError, match="unknown schema version -1"):
        open_journal(path)


def test_a_sqlite_file_that_is_not_a_journal_is_refused_and_left_as_it_was(path):
    path.parent.mkdir(parents=True)
    run_sql(path, "CREATE TABLE prices (symbol TEXT)")
    with pytest.raises(JournalError, match="not an OpenAlgo live journal"):
        open_journal(path)
    assert (read_pragma(path, "journal_mode"), read_pragma(path, "user_version")) == ("delete", 0)


def test_a_file_that_is_not_sqlite_is_refused_as_damaged(path):
    path.parent.mkdir(parents=True)
    path.write_bytes(b"symbol,close\nSBIN,812.5\n" * 20)
    with pytest.raises(JournalError, match="file is not a database.*damaged or is not a journal"):
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


# What a failure says


def test_a_value_sqlite_refuses_names_the_field(path):
    with started_journal(path) as journal:
        a_bound_order(journal, "s1", "o1")
        with pytest.raises(JournalError, match="stops.strategy.*a value was refused") as raised:
            journal.set_stop(None, "SBIN", "NSE", "o1")
    assert isinstance(raised.value.__cause__, sqlite3.IntegrityError)


@pytest.mark.parametrize(
    ("error", "advice"),
    [
        (sqlite3.OperationalError("database is locked"), "no other program holds the file"),
        (sqlite3.OperationalError("database or disk is full"), "check that the disk has space"),
        (sqlite3.IntegrityError("NOT NULL constraint failed: stops.symbol"), "a value was refused"),
    ],
    ids=["locked", "disk full", "refused value"],
)
def test_a_sqlite_error_says_what_happened_and_what_to_do(error, advice):
    assert advice in str(failure(error, "record the stop", Path("journal.sqlite")))


@pytest.mark.parametrize(
    ("change", "read"),
    [
        ("UPDATE intents SET purpose = 'hedge'", lambda j: j.intent("i1")),
        ("UPDATE intents SET purpose = 'hedge'", lambda j: j.unbound()),
        (
            "INSERT INTO user_states VALUES ('sandbox', 'ema', '{', 0)",
            lambda j: j.user_state("ema"),
        ),
    ],
    ids=["intent", "unbound", "self.state"],
)
def test_a_row_changed_outside_openalgo_is_reported_unreadable(path, change, read):
    with started_journal(path) as journal:
        journal.record_intent(an_intent())
    run_sql(path, change)
    with (
        started_journal(path) as journal,
        pytest.raises(JournalError, match="cannot read .*changed outside openalgo"),
    ):
        read(journal)


def test_a_call_from_another_thread_says_the_journal_belongs_to_the_sessions_thread(path):
    raised = []

    def read_from_the_feed_thread():
        try:
            journal.unbound()
        except JournalError as error:
            raised.append(str(error))

    with started_journal(path) as journal:
        feed = threading.Thread(target=read_from_the_feed_thread)
        feed.start()
        feed.join()
    assert len(raised) == 1 and "belongs to the session's thread" in raised[0]


# Closing


@pytest.mark.parametrize(
    "call",
    [
        lambda j: j.record_intent(an_intent()),
        lambda j: j.unbound(day=DAY),
        lambda j: j.set_stop(*LEG, "s1"),
        lambda j: j.put_user_state("ema", {}),
        lambda j: j.session(DAY),
        lambda j: j.snapshot(day=DAY),
    ],
    ids=["record_intent", "unbound", "set_stop", "put_user_state", "session", "snapshot"],
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
        journal.session(DAY)


# Where the journal lives


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALGO_LIVE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_the_default_journal_is_under_the_home_folder(home):
    expected = home / ".openalgo" / "live" / ACCOUNT / "journal.sqlite"
    assert default_journal_path(ACCOUNT) == expected


def test_the_live_dir_variable_moves_the_journals(home, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", str(home / "journals"))
    assert default_journal_path(ACCOUNT) == home / "journals" / ACCOUNT / "journal.sqlite"


def test_a_blank_live_dir_variable_counts_as_unset(home, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", " ")
    expected = home / ".openalgo" / "live" / ACCOUNT / "journal.sqlite"
    assert default_journal_path(ACCOUNT) == expected


def test_a_relative_live_dir_variable_is_refused(home, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", "journals")
    with pytest.raises(JournalError, match="OPENALGO_LIVE_DIR must name an absolute folder"):
        default_journal_path(ACCOUNT)


@pytest.mark.parametrize(
    "account",
    [
        "",
        "a/b",
        "a\\b",
        "C:x",
        "a<b",
        "a>b",
        'a"b',
        "a|b",
        "a?b",
        "a*b",
        "a\tb",
        ".",
        "..",
        "trader.",
        "trader ",
    ],
)
def test_an_account_that_is_not_a_plain_folder_name_is_refused(home, account):
    with pytest.raises(JournalError, match=r"account.* must be"):
        default_journal_path(account)


@pytest.mark.parametrize("account", ["CON", "con", "Nul", "aux.journal", "COM1", "LPT9", "prn.txt"])
def test_an_account_named_like_a_windows_device_is_refused(home, account):
    with pytest.raises(JournalError, match="is a device name on Windows"):
        default_journal_path(account)


@pytest.mark.parametrize("account", ["trader1", "CONSOLE", "COM10", "my-acct_2.zerodha"])
def test_an_ordinary_account_name_is_a_folder(home, account):
    assert default_journal_path(account).parent.name == account

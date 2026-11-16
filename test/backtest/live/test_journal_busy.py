"""How long the live journal waits for a lock another connection holds (`busy_timeout_s`,
16-session-design 2.4 and 3.1 step 8): a session writes one file through two handles,
and a file still locked past the wait is reported with what to do."""

import math
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from journal_samples import ACCOUNT, BROKER, DAY, ManualClock, an_intent, journal_path

from openalgo.backtest._live._journal_file import failure
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.journal import Journal
from openalgo.backtest.errors import JournalError

#: A wait short enough for a test that waits for real.
SHORT_WAIT_S = 0.05


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


def open_journal(path, **options):
    return Journal.open(
        path, account=ACCOUNT, broker=BROKER, mode=Mode.SANDBOX, now=ManualClock(), **options
    )


def busy_timeout_ms(journal):
    # A setting of the handle's own connection, so only it can show it.
    return journal._connection.execute("PRAGMA busy_timeout").fetchone()[0]


def write_locked(path):
    """Another program's connection holding the journal's write lock."""
    blocker = sqlite3.connect(path, isolation_level=None, timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    return blocker


def test_a_journal_waits_five_seconds_for_a_lock_by_default(path):
    with open_journal(path) as journal:
        assert busy_timeout_ms(journal) == 5_000


def test_a_journal_waits_the_busy_timeout_given(path):
    with open_journal(path, busy_timeout_s=0.25) as journal:
        assert busy_timeout_ms(journal) == 250


def test_a_write_that_finds_the_file_locked_past_the_busy_timeout_says_what_to_do(path):
    with open_journal(path, busy_timeout_s=SHORT_WAIT_S) as journal:
        journal.start_session(DAY, Mode.SANDBOX, BROKER)
        with closing(write_locked(path)), pytest.raises(JournalError) as raised:
            journal.record_intent(an_intent())

    message = str(raised.value)
    assert "another program holds the file" in message
    assert f"past the {SHORT_WAIT_S:g} seconds a write waits (busy_timeout_s)" in message
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)


def test_opening_a_file_locked_past_the_busy_timeout_says_what_to_do(path):
    open_journal(path).close()

    with closing(write_locked(path)), pytest.raises(JournalError, match="another program holds"):
        open_journal(path, busy_timeout_s=SHORT_WAIT_S)


def test_two_handles_of_one_session_write_the_file_in_turn(path):
    with open_journal(path) as worker, open_journal(path) as loop:
        worker.start_session(DAY, Mode.SANDBOX, BROKER)
        worker.record_intent(an_intent("i1"))
        loop.put_rms_state("ema", {"trail": 805.0}, day=DAY)
        worker.record_intent(an_intent("i2"))

        assert [intent.intent_id for intent in loop.unbound(day=DAY)] == ["i1", "i2"]
        assert worker.rms_state("ema") == {"trail": 805.0}


@pytest.mark.parametrize("seconds", [0, -1, math.nan, math.inf, True, "5"], ids=str)
def test_a_busy_timeout_that_is_not_a_positive_number_of_seconds_is_refused(path, seconds):
    with pytest.raises(JournalError, match="busy_timeout_s is how many seconds a write waits"):
        open_journal(path, busy_timeout_s=seconds)
    assert not path.exists()


def test_the_advice_for_a_busy_file_names_the_wait():
    error = sqlite3.OperationalError("database is locked")
    error.sqlite_errorname = "SQLITE_BUSY"

    message = str(failure(error, "record the stop", Path("journal.sqlite"), 2.5))

    assert "kept it locked past the 2.5 seconds a write waits" in message

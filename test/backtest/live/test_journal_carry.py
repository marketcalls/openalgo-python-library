"""What the live journal carries from one trading day's session to the next
(`_live/_journal_carry.py`; 20-positional-live 3.5, 4 and 8, the lead's ruling N2-2): the
day-close record of each strategy and the latest one read back, kept refusals and their
clearing, and the last bar each strategy decided. Each row is the ledger's of the handle's
mode: a live handle never reads a sandbox record back as its own.

The journal is on the test's folder with a manual clock; expected values are the ones
written.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest
from journal_samples import (
    DAY,
    DAY_NS,
    NEXT_DAY,
    ManualClock,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest._live._types import Mode
from openalgo.backtest.errors import JournalError

#: The day after NEXT_DAY (Wednesday 2026-11-18).
THIRD_DAY = date(2026, 11, 18)
#: A strategy's carried state and `self.state`, cut to what these tests read.
CARRIED = {"positions": [{"instrument": 0, "product": "CNC", "quantity_units": 40.0}]}
USER_STATE = {"entries": 3}
VERSION = "0a1b2c3d4e5f"
#: A broker's refusal of a stop GTT that no price explains (20-positional-live 3.5).
REFUSAL = "GTT is not allowed for this segment"


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


def close_day(journal, strategy: str, day: date, units: float) -> None:
    carried = {"positions": [{"instrument": 0, "product": "CNC", "quantity_units": units}]}
    journal.put_day_close(strategy, carried, USER_STATE, VERSION, day=day)


# ----- the day-close record ------------------------------------------------------------------


def test_a_day_close_record_reads_back_whole(path):
    with started_journal(path) as journal:
        journal.put_day_close("hold", CARRIED, USER_STATE, VERSION)
        record = journal.day_close("hold")

    assert (record.strategy, record.day, record.mode) == ("hold", DAY, Mode.SANDBOX)
    assert (record.carried, record.user_state) == (CARRIED, USER_STATE)
    assert (record.version_id, record.saved_ns) == (VERSION, DAY_NS)


def test_the_latest_record_across_three_days_and_two_ledgers_is_the_handles_own(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        for day, units in ((DAY, 10.0), (NEXT_DAY, 20.0), (THIRD_DAY, 30.0)):
            close_day(sandbox, "hold", day, units)
        close_day(sandbox, "swing", DAY, 5.0)
    with open_journal(path, mode=Mode.LIVE) as live:
        close_day(live, "hold", DAY, 40.0)
        close_day(live, "hold", NEXT_DAY, 50.0)

        latest = live.day_close("hold")
        closes = live.day_closes()

    assert (latest.day, latest.mode, latest.carried["positions"][0]["quantity_units"]) == (
        NEXT_DAY,
        Mode.LIVE,
        50.0,
    )
    assert list(closes) == ["hold"]
    assert closes["hold"].day == NEXT_DAY


def test_a_live_handle_never_reads_a_sandbox_record_as_its_own(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        close_day(sandbox, "hold", NEXT_DAY, 10.0)
    with open_journal(path, mode=Mode.LIVE) as live:
        assert (live.day_close("hold"), live.day_closes()) == (None, {})


def test_the_other_ledgers_latest_records_are_read_for_naming_them(path):
    with open_journal(path, mode=Mode.SANDBOX) as sandbox:
        close_day(sandbox, "hold", DAY, 10.0)
        close_day(sandbox, "hold", NEXT_DAY, 20.0)
        close_day(sandbox, "swing", DAY, 5.0)
    with open_journal(path, mode=Mode.LIVE) as live:
        held = live.other_ledger_day_closes()

    assert {name: (r.day, r.mode) for name, r in held.items()} == {
        "hold": (NEXT_DAY, Mode.SANDBOX),
        "swing": (DAY, Mode.SANDBOX),
    }


def test_a_later_record_of_the_same_day_replaces_the_earlier(path):
    clock = ManualClock()
    with started_journal(path, now=clock) as journal:
        close_day(journal, "hold", DAY, 10.0)
        clock.ns += 1
        close_day(journal, "hold", DAY, 0.0)
        record = journal.day_close("hold")

    assert (record.carried["positions"][0]["quantity_units"], record.saved_ns) == (0.0, DAY_NS + 1)


def test_a_day_close_record_json_cannot_keep_is_refused(path):
    with (
        started_journal(path) as journal,
        pytest.raises(JournalError, match="the carried state of strategy 'hold' cannot be saved"),
    ):
        journal.put_day_close("hold", {"day": DAY}, USER_STATE, VERSION)


def test_a_day_close_record_needs_a_day(path):
    with open_journal(path) as journal, pytest.raises(JournalError, match="call start_session"):
        journal.put_day_close("hold", CARRIED, USER_STATE, VERSION)


# ----- kept refusals -------------------------------------------------------------------------


def test_a_kept_refusal_reads_back_with_its_leg_message_and_day(path):
    with started_journal(path) as journal:
        journal.keep_refusal("hold", "SBIN", "NSE", "CNC", REFUSAL)
        (kept,) = journal.kept_refusals()

    assert (kept.strategy, kept.symbol, kept.exchange, kept.product) == (
        "hold",
        "SBIN",
        "NSE",
        "CNC",
    )
    assert (kept.message, kept.day, kept.kept_ns) == (REFUSAL, DAY, DAY_NS)


def test_a_refusal_is_kept_across_days_until_it_is_cleared(path):
    with started_journal(path) as journal:
        journal.keep_refusal("hold", "SBIN", "NSE", "CNC", REFUSAL)
        journal.keep_refusal("swing", "INFY", "NSE", "CNC", REFUSAL)
    with started_journal(path, day=NEXT_DAY) as journal:
        kept_next_day = [kept.strategy for kept in journal.kept_refusals()]
        journal.clear_refusals("hold")
        left = [kept.strategy for kept in journal.kept_refusals()]

    assert (kept_next_day, left) == (["hold", "swing"], ["swing"])


def test_a_live_handle_never_reads_a_sandbox_refusal(path):
    with started_journal(path) as journal:
        journal.keep_refusal("hold", "SBIN", "NSE", "CNC", REFUSAL)
    with open_journal(path, mode=Mode.LIVE) as live:
        assert live.kept_refusals() == []


def test_a_refusal_with_no_message_is_refused(path):
    with started_journal(path) as journal, pytest.raises(JournalError, match="message"):
        journal.keep_refusal("hold", "SBIN", "NSE", "CNC", " ")


# ----- the last decided bar -----------------------------------------------------------------


def test_the_last_bar_each_strategy_decided_reads_back(path):
    with started_journal(path) as journal:
        journal.put_decided_bar("daily", DAY_NS)
        journal.put_decided_bar("daily", DAY_NS)
        journal.put_decided_bar("weekly", DAY_NS - 1)

        assert (journal.decided_bar("daily"), journal.decided_bar("weekly")) == (DAY_NS, DAY_NS - 1)
        assert journal.decided_bar("swing") is None


def test_the_same_bar_decided_again_changes_nothing_not_even_when_it_was_noted(path):
    clock = ManualClock()
    with started_journal(path, now=clock) as journal:
        journal.put_decided_bar("daily", DAY_NS)
        clock.ns += 5
        journal.put_decided_bar("daily", DAY_NS)

    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT bar_ns, decided_ns FROM decided_bars").fetchall()
    assert rows == [(DAY_NS, DAY_NS)]


def test_a_bar_before_the_one_decided_is_refused(path):
    with started_journal(path) as journal:
        journal.put_decided_bar("daily", DAY_NS)
        with pytest.raises(JournalError, match="decides its bars in order, each once"):
            journal.put_decided_bar("daily", DAY_NS - 1)


def test_a_live_handle_never_reads_a_sandbox_decided_bar(path):
    with started_journal(path) as journal:
        journal.put_decided_bar("daily", DAY_NS)
    with open_journal(path, mode=Mode.LIVE) as live:
        assert live.decided_bar("daily") is None

"""The GTTs in the live journal (`_live/_journal_gtt.py`; 20-positional-live 3, the lead's
ruling N2-2): a GTT written before its place call, bound to the broker's trigger ID, each
state it takes and when, the terms of a confirmed modify, the GTTs not ended across days,
and the child order a fired GTT placed. Each row is the ledger's of the handle's mode.

The journal is on the test's folder with a manual clock; expected values are the ones
written.
"""

from __future__ import annotations

import pytest
from journal_samples import (
    DAY,
    DAY_NS,
    NEXT_DAY,
    ManualClock,
    a_bound_order,
    ist_ns,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.journal import GttState, GttTerms
from openalgo.backtest.errors import JournalError

#: A long CNC leg's stop: sell 10 units when SBIN trades at 790.00, the child's limit 780.30.
STOP = GttTerms("SELL", 790.0, 780.3, 10)


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


def recorded(journal, gtt_id=1, request_id=7, terms=STOP, product="CNC") -> int:
    return journal.record_gtt("ema", gtt_id, request_id, "SBIN", "NSE", product, terms)


# ----- recording and binding --------------------------------------------------------------


def test_a_gtt_written_before_its_place_call_reads_back_as_requested(path):
    with started_journal(path) as journal:
        seq = recorded(journal)
        gtt = journal.gtt(seq)

    assert (gtt.day, gtt.strategy, gtt.gtt_id, gtt.request_id) == (DAY, "ema", 1, 7)
    assert (gtt.symbol, gtt.exchange, gtt.product, gtt.terms) == ("SBIN", "NSE", "CNC", STOP)
    assert (gtt.state, gtt.trigger_id, gtt.recorded_ns) == (GttState.REQUESTED, None, DAY_NS)


def test_each_gtt_has_its_own_number_in_order(path):
    with started_journal(path) as journal:
        assert [recorded(journal, gtt_id) for gtt_id in (1, 2, 3)] == [1, 2, 3]


def test_the_places_answer_binds_the_brokers_trigger_id(path):
    with started_journal(path) as journal:
        seq = recorded(journal)
        journal.bind_gtt(seq, "T-901")
        journal.bind_gtt(seq, "T-901")

        assert (journal.gtt(seq).state, journal.gtt_by_trigger("T-901").seq) == (
            GttState.PLACED,
            seq,
        )


@pytest.mark.parametrize(
    "rebind", ["another trigger", "a trigger of another gtt"], ids=lambda words: words
)
def test_a_gtt_is_bound_to_one_trigger_and_a_trigger_to_one_gtt(path, rebind):
    with started_journal(path) as journal:
        first, second = recorded(journal, 1), recorded(journal, 2)
        journal.bind_gtt(first, "T-901")
        with pytest.raises(JournalError, match="one GTT"):
            if rebind == "another trigger":
                journal.bind_gtt(first, "T-902")
            else:
                journal.bind_gtt(second, "T-901")


@pytest.mark.parametrize(
    ("product", "terms", "words"),
    [
        ("MIS", STOP, "CNC or NRML, not 'MIS'"),
        ("CNC", GttTerms("SHORT", 790.0, 780.3, 10), "BUY or SELL"),
        ("CNC", GttTerms("SELL", 0.0, 780.3, 10), "a trigger above 0"),
        ("CNC", GttTerms("SELL", 790.0, -1.0, 10), "a limit of 0 rupees or more"),
        ("CNC", GttTerms("SELL", 790.0, 780.3, 0), "units above 0"),
        ("CNC", GttTerms("SELL", 790.0, 780.3, 2.5), "quantity must be a whole number"),
        ("CNC", GttTerms("SELL", float("nan"), 780.3, 10), "trigger_price must be a finite"),
    ],
    ids=["mis", "side", "trigger", "limit", "no units", "part units", "trigger not a number"],
)
def test_a_gtt_the_broker_could_not_hold_is_refused(path, product, terms, words):
    with started_journal(path) as journal, pytest.raises(JournalError, match=words):
        recorded(journal, terms=terms, product=product)


def test_a_gtt_needs_the_sessions_day(path):
    with open_journal(path) as journal, pytest.raises(JournalError, match="call start_session"):
        recorded(journal)


# ----- states and terms ---------------------------------------------------------------------


def test_each_state_of_a_gtt_is_kept_with_its_note_and_time(path):
    clock = ManualClock()
    with started_journal(path, now=clock) as journal:
        seq = recorded(journal)
        clock.ns += 1
        journal.bind_gtt(seq, "T-901")
        clock.ns += 1
        journal.set_gtt_state(seq, GttState.ACTIVE)
        clock.ns += 1
        journal.set_gtt_state(seq, GttState.ENDED, note="cancelled by hand")

        assert journal.gtt_states(seq) == [
            (GttState.REQUESTED, "", DAY_NS),
            (GttState.PLACED, "", DAY_NS + 1),
            (GttState.ACTIVE, "", DAY_NS + 2),
            (GttState.ENDED, "cancelled by hand", DAY_NS + 3),
        ]
        assert (journal.gtt(seq).state, journal.gtt(seq).changed_ns) == (GttState.ENDED, DAY_NS + 3)


@pytest.mark.parametrize("ended", [state for state in GttState if state.terminal])
def test_an_ended_gtt_changes_no_more(path, ended):
    with started_journal(path) as journal:
        seq = recorded(journal)
        if ended is not GttState.NOT_PLACED:
            journal.bind_gtt(seq, "T-901")
        journal.set_gtt_state(seq, ended)
        with pytest.raises(JournalError, match="an ended GTT changes no more"):
            journal.set_gtt_state(seq, GttState.ACTIVE)


def test_a_gtt_whose_place_has_not_answered_can_only_be_marked_not_placed(path):
    with started_journal(path) as journal:
        seq = recorded(journal)
        with pytest.raises(JournalError, match="place has not answered"):
            journal.set_gtt_state(seq, GttState.ACTIVE)
        journal.set_gtt_state(seq, GttState.NOT_PLACED, note="no such GTT in the book")

        assert journal.gtt(seq).state is GttState.NOT_PLACED


def test_a_confirmed_modify_keeps_the_terms_the_broker_holds(path):
    trailed = GttTerms("SELL", 795.5, 785.75, 6)
    with started_journal(path) as journal:
        seq = recorded(journal)
        journal.bind_gtt(seq, "T-901")
        journal.set_gtt_terms(seq, trailed)

        assert journal.gtt_by_trigger("T-901").terms == trailed


def test_a_gtt_not_at_the_broker_is_not_modified(path):
    with started_journal(path) as journal:
        seq = recorded(journal)
        with pytest.raises(JournalError, match="only a GTT the broker holds is modified"):
            journal.set_gtt_terms(seq, STOP)


def test_an_unknown_gtt_is_none_and_cannot_change(path):
    with started_journal(path) as journal:
        assert (journal.gtt(5), journal.gtt_by_trigger("T-5")) == (None, None)
        with pytest.raises(JournalError, match="GTT 5 is not in the sandbox ledger"):
            journal.bind_gtt(5, "T-5")


# ----- across days and ledgers ----------------------------------------------------------------


def test_the_gtts_not_ended_are_read_on_a_later_day_oldest_first(path):
    with started_journal(path) as journal:
        carried, cancelled = recorded(journal, 1), recorded(journal, 2)
        for seq, trigger in ((carried, "T-1"), (cancelled, "T-2")):
            journal.bind_gtt(seq, trigger)
        journal.set_gtt_state(cancelled, GttState.CANCELLED)
    clock = ManualClock(ist_ns(NEXT_DAY, 9))
    with started_journal(path, day=NEXT_DAY, now=clock) as journal:
        today = recorded(journal, 1, 3)

        assert [(gtt.seq, gtt.day) for gtt in journal.open_gtts()] == [
            (carried, DAY),
            (today, NEXT_DAY),
        ]


def test_a_handle_never_sees_the_other_ledgers_gtts(path):
    with started_journal(path) as journal:
        journal.bind_gtt(recorded(journal), "T-901")
    with open_journal(path, mode=Mode.LIVE) as live:
        assert (live.open_gtts(), live.gtt(1), live.gtt_by_trigger("T-901")) == ([], None, None)


# ----- the child of a fired GTT ---------------------------------------------------------------


def fired(journal) -> int:
    seq = recorded(journal)
    journal.bind_gtt(seq, "T-901")
    journal.set_gtt_state(seq, GttState.TRIGGERED)
    return seq


def test_a_fired_gtts_child_is_bound_by_its_broker_and_kernel_orders(path):
    with started_journal(path) as journal:
        seq = fired(journal)
        journal.bind_gtt_child(seq, "B-77", 4)
        journal.bind_gtt_child(seq, "B-77", 4)
        child = journal.gtt_child(seq)

        assert (child.order_id, child.kernel_order, child.day) == ("B-77", 4, DAY)
        assert journal.gtt_of_child("B-77").seq == seq


def test_a_gtt_has_one_child(path):
    with started_journal(path) as journal:
        seq = fired(journal)
        journal.bind_gtt_child(seq, "B-77", 4)
        with pytest.raises(JournalError, match="places one child"):
            journal.bind_gtt_child(seq, "B-78", 5)


def test_an_order_is_the_child_of_one_gtt(path):
    with started_journal(path) as journal:
        first, second = fired(journal), recorded(journal, 2)
        journal.bind_gtt(second, "T-902")
        journal.bind_gtt_child(first, "B-77", 4)
        with pytest.raises(JournalError, match="child of GTT 1 already"):
            journal.bind_gtt_child(second, "B-77", 5)


def test_an_order_the_session_placed_itself_is_no_gtts_child(path):
    with started_journal(path) as journal:
        a_bound_order(journal, "i1", "B-77")
        with pytest.raises(JournalError, match="placed by the session itself"):
            journal.bind_gtt_child(fired(journal), "B-77", 4)


def test_a_gtt_that_never_reached_the_broker_has_no_child(path):
    with started_journal(path) as journal:
        seq = recorded(journal)
        with pytest.raises(JournalError, match="never reached the broker"):
            journal.bind_gtt_child(seq, "B-77", 4)


def test_a_gtt_with_no_child_bound_answers_none(path):
    with started_journal(path) as journal:
        assert (journal.gtt_child(fired(journal)), journal.gtt_of_child("B-77")) == (None, None)

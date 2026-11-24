"""The order worker's GTT calls (`_live/_gtt_send.py`, `_gtt_calls.py`, `_gtt_confirm.py`;
20-positional-live 3.1, 3.5, 3.6; the lead's ruling N3a-1): each call's payload, journal
first, each call confirmed by a read of the GTT book made at once, the unanswered calls
resolved against the book by shape, refusals answered with the broker's words and kept
across starts when they repeat on a later day, and each call counted under the order rate.

The worker runs on the fake OpenAlgo (`gtt_samples`): SBIN at 800.00, a long CNC leg's
stop sells 10 at a trigger of 790.00 with a child limit of 780.00. Expected payloads are
the SDK's (openalgo/gtt.py), IDs the fake broker's (`fake_openalgo.gtts`).
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from gtt_samples import (
    FIRST_TRIGGER,
    LIMIT,
    NEXT_DAY,
    SECOND_TRIGGER,
    TRIGGER,
    UNITS,
    GttRig,
    answers,
    broker_with_sbin,
    calls,
)
from worker_samples import STRATEGY

from openalgo.backtest._live._gtt_send import NOT_FOUND_NOTE, REPLACED_NOTE
from openalgo.backtest._live._gtt_types import Direction, GttAction, GttAnswerKind
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live.inbox import NotSent, NotSentReason
from openalgo.backtest._live.journal import GttState, GttTerms

STOP_TERMS = GttTerms("SELL", TRIGGER, LIMIT, UNITS)
#: The modify of the tests: the trigger raised to 795.00 and the units cut to 6.
MODIFIED_TERMS = GttTerms("SELL", 795.0, LIMIT, 6)
REFUSAL = "Trigger price is too close to the last price"


@pytest.fixture
def broker(fake_openalgo):
    return broker_with_sbin(fake_openalgo)


@pytest.fixture
def rig(broker, tmp_path):
    made = GttRig(broker, tmp_path)
    made.messages()
    yield made
    made.worker.close()


def placed(rig) -> int:
    """Place the stop and return its journal number (confirmed: the tests start there)."""
    (answer,) = answers(rig.send(rig.gtt()))
    assert answer.kind is GttAnswerKind.PLACED
    return answer.seq


def journal_gtt(rig, seq):
    with rig.journal() as journal:
        return journal.gtt(seq)


def modify(rig, seq):
    return rig.gtt(GttAction.MODIFY, seq=seq, trigger=795.0, units=6)


# ----- the payloads ---------------------------------------------------------------------------


def test_a_longs_stop_is_sent_in_triggerprice_sl_with_its_limit_and_units(rig, broker):
    rig.send(rig.gtt())

    (call,) = broker.log.calls("placegttorder")
    assert call.detail["request"] == {
        "strategy": STRATEGY,
        "trigger_type": "SINGLE",
        "exchange": "NSE",
        "symbol": "SBIN",
        "action": "SELL",
        "product": "CNC",
        "quantity": 10,
        "pricetype": "LIMIT",
        "price": 780.0,
        "triggerprice_sl": 790.0,
        "triggerprice_tg": 0.0,
        "stoploss": None,
        "target": None,
    }


def test_a_shorts_stop_is_sent_in_triggerprice_tg(rig, broker):
    short_stop = rig.gtt(
        side="BUY", trigger=810.0, limit=820.0, direction=Direction.ABOVE, product="NRML"
    )

    rig.send(short_stop)

    (call,) = broker.log.calls("placegttorder")
    sent = call.detail["request"]
    assert (sent["action"], sent["product"], sent["price"]) == ("BUY", "NRML", 820.0)
    assert (sent["triggerprice_sl"], sent["triggerprice_tg"]) == (0.0, 810.0)


# ----- confirmations ---------------------------------------------------------------------------


def test_a_place_is_confirmed_by_the_gtt_resting_in_a_read_made_at_once(rig, broker):
    request = rig.gtt()

    (answer,) = answers(rig.send(request))

    assert (answer.request, answer.kind, answer.seq, answer.trigger_id) == (
        request,
        GttAnswerKind.PLACED,
        1,
        FIRST_TRIGGER,
    )
    # The leg's units are read before the GTT book: the baseline of a GTT gone later.
    assert calls(broker)[-4:] == ["placegttorder", "positionbook", "holdings", "gttorderbook"]
    assert journal_gtt(rig, 1).state is GttState.ACTIVE


def test_the_journal_holds_the_gtt_before_its_place_call(rig, broker, monkeypatch):
    seen = []
    original = rig.orders._client.placegttorder

    def spy(**fields):
        with rig.journal() as journal:
            seen.append(journal.gtt(1).state)
        return original(**fields)

    monkeypatch.setattr(rig.orders._client, "placegttorder", spy)

    rig.send(rig.gtt())

    assert seen == [GttState.REQUESTED]


def test_a_modify_is_confirmed_by_the_read_showing_its_trigger_and_units(rig):
    seq = placed(rig)

    (answer,) = answers(rig.send(modify(rig, seq)))

    assert (answer.kind, answer.seq, answer.trigger_id) == (
        GttAnswerKind.MODIFIED,
        seq,
        FIRST_TRIGGER,
    )
    gtt = journal_gtt(rig, seq)
    assert (gtt.terms, gtt.state) == (MODIFIED_TERMS, GttState.ACTIVE)


def test_a_cancel_is_confirmed_by_the_gtt_ended_with_no_child_in_the_order_book(rig, broker):
    seq = placed(rig)

    (answer,) = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))

    assert (answer.kind, answer.seq) == (GttAnswerKind.CANCELLED, seq)
    assert calls(broker)[-5:] == [
        "cancelgttorder",
        "positionbook",
        "holdings",
        "gttorderbook",
        "orderbook",
    ]
    assert journal_gtt(rig, seq).state is GttState.CANCELLED


# ----- unanswered calls ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "fault",
    [
        lambda faults: faults.timeout("placegttorder", reached_broker=True),
        lambda faults: faults.lose_answer("placegttorder"),
    ],
    ids=["a timeout", "an error after the gtt was made"],
)
def test_an_unanswered_place_found_by_its_shape_is_bound(rig, broker, fault):
    fault(broker.faults)

    (answer,) = answers(rig.send(rig.gtt()))

    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 1, FIRST_TRIGGER)
    assert journal_gtt(rig, 1).trigger_id == FIRST_TRIGGER


def test_an_unanswered_place_not_found_in_two_reads_2_seconds_apart_is_placed_again(rig, broker):
    broker.faults.timeout("placegttorder")

    first = answers(rig.send(rig.gtt()))
    later = answers(rig.step_for(2.0))

    assert first == []
    (answer,) = later
    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 2, FIRST_TRIGGER)
    assert len(broker.log.calls("placegttorder")) == 2
    with rig.journal() as journal:
        assert journal.gtt(1).state is GttState.NOT_PLACED
        assert journal.gtt_states(1)[-1][1] == NOT_FOUND_NOTE


def test_a_place_unanswered_twice_and_not_found_is_unresolved(rig, broker):
    broker.faults.timeout("placegttorder")
    broker.faults.timeout("placegttorder")

    rig.send(rig.gtt())
    news = answers(rig.step_for(2.0)) + answers(rig.step_for(2.0))

    (answer,) = news
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert "unanswered twice" in answer.message
    states = [journal_gtt(rig, seq).state for seq in (1, 2)]
    assert states == [GttState.NOT_PLACED, GttState.NOT_PLACED]
    assert broker.gtts() == []


def test_nothing_more_is_sent_for_a_gtt_while_its_unanswered_place_is_looked_for(rig, broker):
    broker.faults.timeout("placegttorder")
    rig.send(rig.gtt())

    rig.send(rig.gtt(GttAction.CANCEL, seq=1))

    assert broker.log.calls("cancelgttorder") == []


def test_an_unanswered_modify_the_book_shows_made_is_modified(rig, broker):
    seq = placed(rig)
    broker.faults.timeout("modifygttorder", reached_broker=True)

    (answer,) = answers(rig.send(modify(rig, seq)))

    assert (answer.kind, journal_gtt(rig, seq).terms) == (GttAnswerKind.MODIFIED, MODIFIED_TERMS)


def test_an_unanswered_modify_the_book_shows_not_made_in_two_reads_is_refused(rig, broker):
    seq = placed(rig)
    broker.faults.timeout("modifygttorder")

    first = answers(rig.send(modify(rig, seq)))
    (answer,) = answers(rig.step_for(2.0))

    assert first == []
    assert answer.kind is GttAnswerKind.MODIFY_REFUSED
    gtt = journal_gtt(rig, seq)
    assert (gtt.terms, gtt.state) == (STOP_TERMS, GttState.ACTIVE)


def test_a_modify_applied_in_part_is_cancelled_and_placed_again(rig, broker):
    seq = placed(rig)
    broker.half_apply_next_gtt_modify()

    (answer,) = answers(rig.send(modify(rig, seq)))

    assert (answer.kind, answer.seq, answer.trigger_id) == (
        GttAnswerKind.MODIFIED,
        2,
        SECOND_TRIGGER,
    )
    assert [gtt.status for gtt in broker.gtts()] == ["cancelled", "active"]
    with rig.journal() as journal:
        old, new = journal.gtt(seq), journal.gtt(2)
        assert (old.state, journal.gtt_states(seq)[-1][1]) == (GttState.CANCELLED, REPLACED_NOTE)
        assert (new.terms, new.state, new.gtt_id) == (MODIFIED_TERMS, GttState.ACTIVE, 1)


def test_an_unanswered_cancel_whose_gtt_ended_with_no_child_is_cancelled(rig, broker):
    seq = placed(rig)
    broker.faults.timeout("cancelgttorder", reached_broker=True)

    (answer,) = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))

    assert answer.kind is GttAnswerKind.CANCELLED


def test_an_unanswered_cancel_whose_gtt_rests_in_two_reads_is_refused(rig, broker):
    seq = placed(rig)
    broker.faults.timeout("cancelgttorder")

    first = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))
    (answer,) = answers(rig.step_for(2.0))

    assert (first, answer.kind) == ([], GttAnswerKind.CANCEL_REFUSED)
    assert journal_gtt(rig, seq).state is GttState.ACTIVE


# ----- refusals --------------------------------------------------------------------------------


def test_a_refused_place_answers_the_brokers_message_and_is_not_kept_on_its_first_day(rig, broker):
    broker.refuse_next_gtt(REFUSAL)

    (answer,) = answers(rig.send(rig.gtt()))

    assert (answer.kind, answer.message, answer.kept) == (GttAnswerKind.REFUSED, REFUSAL, False)
    with rig.journal() as journal:
        assert (journal.gtt(1).state, journal.kept_refusals()) == (GttState.NOT_PLACED, [])


def test_a_stop_refused_twice_on_one_day_is_the_days_alone(rig, broker):
    for _ in range(2):
        broker.refuse_next_gtt(REFUSAL)
        (answer,) = answers(rig.send(rig.gtt()))

    assert answer.kept is False


def refused_on_two_days(broker, folder, purpose):
    """A stop GTT (or a strategy's GTT entry) the broker refuses on the fixture's day and
    on the next: the second day's answer, and the journal's kept refusals."""
    for day in (None, NEXT_DAY):
        if day is not None:
            broker.clock.advance(24 * 60 * 60)
        made = GttRig(broker, folder) if day is None else GttRig(broker, folder, day=day)
        broker.refuse_next_gtt(REFUSAL)
        (answer,) = answers(made.send(made.gtt(purpose=purpose)))
        with made.journal() as journal:
            kept = journal.kept_refusals()
        made.worker.close()
    return answer, kept


def test_a_stop_refused_again_on_a_later_day_is_kept_with_the_brokers_message(broker, tmp_path):
    answer, kept = refused_on_two_days(broker, tmp_path, Purpose.STOP)

    assert (answer.kind, answer.kept) == (GttAnswerKind.REFUSED, True)
    (refusal,) = kept
    assert (refusal.strategy, refusal.symbol, refusal.exchange, refusal.product) == (
        STRATEGY,
        "SBIN",
        "NSE",
        "CNC",
    )
    assert (refusal.message, refusal.day) == (REFUSAL, NEXT_DAY)


def test_a_gtt_entry_refused_again_on_a_later_day_is_not_kept(broker, tmp_path):
    answer, kept = refused_on_two_days(broker, tmp_path, Purpose.ENTRY)

    assert (answer.kept, kept) == (False, [])


# ----- the order rate and the holds --------------------------------------------------------


def test_each_gtt_call_counts_as_an_order_call_under_the_order_rate(broker, tmp_path):
    rig = GttRig(broker, tmp_path, orders_per_second=1)
    rig.messages()
    rig.worker.put(rig.gtt(gtt_id=1))
    rig.worker.put(rig.gtt(gtt_id=2))

    rig.worker.step()
    sent_at_once = len(broker.log.calls("placegttorder"))
    rig.clock.advance(1.0)
    rig.worker.step()

    assert (sent_at_once, len(broker.log.calls("placegttorder"))) == (1, 2)
    rig.worker.close()


def test_a_gtt_request_after_the_guard_locked_is_not_sent(rig, broker):
    # The analyzer switch changed: the next answer the guard checks carries the sandbox's
    # mode, which locks it.
    rig.guard.check_response({"status": "success", "mode": "analyze"})

    messages = rig.run(rig.gtt())

    (not_sent,) = [m for m in messages if isinstance(m, NotSent)]
    assert not_sent.reason is NotSentReason.LOCKED
    assert broker.log.calls("placegttorder") == []


def test_a_gtt_request_while_openalgo_is_silent_is_put_back(rig, broker):
    # The order path's last call timed out: nothing is sent until OpenAlgo answers again.
    timed_out = {"status": "error", "message": "timed out", "error_type": "timeout_error"}
    rig.worker._started.path.note_answer(timed_out)

    messages = rig.run(rig.gtt())

    (not_sent,) = [m for m in messages if isinstance(m, NotSent)]
    assert not_sent.reason is NotSentReason.SILENT
    assert (broker.log.calls("placegttorder"), rig.worker.unfinished) == ([], 1)

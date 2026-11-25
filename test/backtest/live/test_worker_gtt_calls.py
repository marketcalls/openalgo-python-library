"""The order worker's GTT calls (`_live/_gtt_send.py`, `_gtt_calls.py`, `_gtt_confirm.py`;
20-positional-live 3.1, 3.5, 3.6; the lead's ruling N3a-1): each call's payload, journal
first, each call confirmed by a read of the GTT book made at once, the unanswered calls
resolved against the book by shape, refusals answered with the broker's words and kept
across starts when they repeat on a later day, and each call counted under the order rate.

The worker runs on the fake OpenAlgo (`gtt_samples`): SBIN at 800.00, a long CNC leg's
stop sells 10 at a trigger of 790.00 with a child limit of 780.00. An unanswered place is
placed again only against a reference of its leg from before it was sent (the lead's
ruling M3 on N3a's review round 5): a test of one gives the session's start read
(`GttRig.start_read`). Expected payloads are the SDK's (openalgo/gtt.py), IDs the fake
broker's (`fake_openalgo.gtts`).
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
    UNREADABLE_ROW,
    GttRig,
    answers,
    bought,
    broker_with_sbin,
    calls,
    outcomes,
    with_rows,
)
from worker_samples import STRATEGY

from openalgo.backtest._live._gtt_child import CHILD_SEARCH_S
from openalgo.backtest._live._gtt_reads import UNITS_WAIT_S
from openalgo.backtest._live._gtt_send import NOT_FOUND_NOTE, REPLACED_NOTE
from openalgo.backtest._live._gtt_types import (
    Direction,
    GttAction,
    GttAnswer,
    GttAnswerKind,
    GttEvent,
    ReadGtts,
)
from openalgo.backtest._live._own_fills import POSITION_BOOK_LAG_S
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live.inbox import NotSent, NotSentReason
from openalgo.backtest._live.journal import GttState, GttTerms

STOP_TERMS = GttTerms("SELL", TRIGGER, LIMIT, UNITS)
#: The modify of the tests: the trigger raised to 795.00 and the units cut to 6.
MODIFIED_TERMS = GttTerms("SELL", 795.0, LIMIT, 6)
REFUSAL = "Trigger price is too close to the last price"


#: Both shapes of a broker's GTT book (20-positional-live 3.1, review H2): one that keeps
#: the GTTs that ended (history rows), one that lists the active ones only. A test marked
#: so runs on each; any other on the book with history rows.
SHAPES = pytest.mark.parametrize("history", [True, False], ids=["history rows", "active only"])


@pytest.fixture
def history():
    """Whether the broker's GTT book keeps history rows (`SHAPES` runs both)."""
    return True


@pytest.fixture
def broker(make_fake_openalgo, history):
    return broker_with_sbin(make_fake_openalgo(gtt_history=history))


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
@SHAPES
def test_an_unanswered_place_found_by_its_shape_is_bound(rig, broker, fault):
    fault(broker.faults)

    (answer,) = answers(rig.send(rig.gtt()))

    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 1, FIRST_TRIGGER)
    assert journal_gtt(rig, 1).trigger_id == FIRST_TRIGGER


#: Seconds from an unanswered call's answer to its decision when the server may still be
#: working on it: misses count from its send plus 150 s (the order path's BROKER_WINDOW_NS),
#: 30 s after the client's 120 s timeout answered, and two reads 2 s apart decide.
DECIDED_AFTER_TIMEOUT_S = 32.0
#: The same when the server has finished with it (a refused connection): misses count from
#: 5 s after the answer (NOT_PLACED_AFTER_NS), so the reads 6 s and 8 s after it decide.
DECIDED_AFTER_FINISHED_S = 8.0
#: Seconds from an unanswered call's send in which the server may still be working on it
#: (the order path's BROKER_WINDOW_NS): with a client timeout shorter than 150 s, a place
#: is decided once they pass.
BROKER_WINDOW_S = 150.0
UNANSWERED_PLACES = pytest.mark.parametrize(
    ("fault", "decided_after_s"),
    [
        (lambda faults: faults.timeout("placegttorder"), DECIDED_AFTER_TIMEOUT_S),
        (lambda faults: faults.refuse_connection("placegttorder"), DECIDED_AFTER_FINISHED_S),
    ],
    ids=["a timeout", "a refused connection"],
)


@UNANSWERED_PLACES
@SHAPES
def test_an_unanswered_place_not_found_once_misses_count_is_placed_again(
    rig, broker, fault, decided_after_s
):
    rig.start_read(0)
    fault(broker.faults)

    first = answers(rig.send(rig.gtt()))
    early = answers(rig.step_for(decided_after_s - 2.0, every_s=2.0))
    later = answers(rig.step_for(2.0, every_s=2.0))

    assert (first, early) == ([], [])
    (answer,) = later
    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 2, FIRST_TRIGGER)
    assert len(broker.log.calls("placegttorder")) == 2
    with rig.journal() as journal:
        assert journal.gtt(1).state is GttState.NOT_PLACED
        assert journal.gtt_states(1)[-1][1] == NOT_FOUND_NOTE


@SHAPES
def test_an_unanswered_place_not_found_with_no_reference_from_before_it_is_unresolved(rig, broker):
    broker.faults.refuse_connection("placegttorder")

    news = answers(rig.send(rig.gtt())) + answers(rig.step_for(DECIDED_AFTER_FINISHED_S, 2.0))

    # No settled read of the leg was made before the place: had it landed and fired, nothing
    # could tell, so it is never placed again.
    (answer,) = news
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert "no settled read of its leg's units" in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


@SHAPES
def test_an_unanswered_place_the_server_finishes_late_is_bound_and_never_placed_again(rig, broker):
    # The server places it 20 s after the client's timeout: within the 150 s from its send
    # in which it may still be working on it.
    broker.faults.late_landing("placegttorder", seconds=20.0)

    news = answers(rig.send(rig.gtt())) + answers(rig.step_for(DECIDED_AFTER_TIMEOUT_S, 2.0))

    (answer,) = news
    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 1, FIRST_TRIGGER)
    assert len(broker.log.calls("placegttorder")) == 1
    assert [gtt.trigger_id for gtt in broker.gtts()] == [FIRST_TRIGGER]


@SHAPES
def test_a_place_unanswered_twice_and_not_found_is_unresolved(rig, broker):
    rig.start_read(0)
    broker.faults.timeout("placegttorder")
    broker.faults.timeout("placegttorder")

    rig.send(rig.gtt())
    news = answers(rig.step_for(DECIDED_AFTER_TIMEOUT_S, 2.0))
    news += answers(rig.step_for(DECIDED_AFTER_TIMEOUT_S, 2.0))

    (answer,) = news
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert "unanswered twice" in answer.message
    states = [journal_gtt(rig, seq).state for seq in (1, 2)]
    assert states == [GttState.NOT_PLACED, GttState.NOT_PLACED]
    assert broker.gtts() == []


def a_gtt_by_hand(broker, status):
    """Another client places a GTT of the stop's shape, which then ends with `status`
    (or fires, for "triggered") with no child: FIRST_TRIGGER, unbound in the journal."""
    hand = broker.client()
    fields = {"symbol": "SBIN", "exchange": "NSE", "product": "CNC", "action": "SELL"}
    terms = {"quantity": UNITS, "price": LIMIT, "triggerprice_sl": TRIGGER}
    trigger_id = hand.placegttorder(strategy="manual", **fields, **terms)["trigger_id"]
    broker.set_gtt_status(trigger_id, status)
    return trigger_id


@pytest.mark.parametrize("status", ["cancelled", "expired", "rejected"])
@SHAPES
def test_an_unanswered_place_is_not_bound_to_a_gtt_of_its_shape_that_ended(rig, broker, status):
    rig.start_read(0)
    a_gtt_by_hand(broker, status)
    broker.faults.timeout("placegttorder")

    news = answers(rig.send(rig.gtt())) + answers(rig.step_for(DECIDED_AFTER_TIMEOUT_S, 2.0))

    # Not found once misses count, it is placed again: the hand GTT is never the session's.
    (answer,) = news
    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 2, SECOND_TRIGGER)
    with rig.journal() as journal:
        assert journal.gtt_by_trigger(FIRST_TRIGGER) is None


def test_an_unanswered_place_is_not_placed_again_when_a_history_row_of_its_shape_fired(rig, broker):
    fired = a_gtt_by_hand(broker, "triggered")
    broker.faults.timeout("placegttorder")

    news = answers(rig.send(rig.gtt())) + answers(rig.step_for(DECIDED_AFTER_TIMEOUT_S, 2.0))

    # It may be this place, landed and fired: placing it again could sell twice.
    (answer,) = news
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert fired in answer.message
    assert [gtt.trigger_id for gtt in broker.gtts()] == [fired]


def fallen_once_placed(rig, broker, monkeypatch, price=789.0):
    """SBIN trades at `price` just after each place call ends: a GTT the broker holds by
    then fires at once, and its child sells at 789.00 (above its 780.00 limit), or rests
    unfilled at a price below the limit."""
    original = rig.orders._client.placegttorder

    def placed_then_fallen(**fields):
        answer = original(**fields)
        broker.trade("SBIN", "NSE", price)
        return answer

    monkeypatch.setattr(rig.orders._client, "placegttorder", placed_then_fallen)


@SHAPES
def test_an_unanswered_place_whose_gtt_fired_before_its_first_read_is_not_placed_again(
    rig, broker, monkeypatch, history
):
    bought(broker, product="NRML")
    rig.start_read(UNITS, "NRML")
    broker.faults.timeout("placegttorder", reached_broker=True)
    fallen_once_placed(rig, broker, monkeypatch)

    news = rig.send(rig.gtt(product="NRML"))
    news += rig.step_for(DECIDED_AFTER_TIMEOUT_S, every_s=2.0)

    # It reached the broker and fired before the first read. A book with history rows shows
    # it fired; an active-only book shows nothing, but the position fell by its 10 from the
    # start read: placed again, a second stop would sell 10 more.
    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "the position fell by its units since"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1
    assert broker.position("SBIN", "NSE") == 0


@SHAPES
def test_an_unanswered_place_whose_child_rests_in_the_order_book_is_not_placed_again(
    rig, broker, monkeypatch, history
):
    bought(broker, product="NRML")
    rig.start_read(UNITS, "NRML")
    broker.faults.timeout("placegttorder", reached_broker=True)
    # SBIN gaps to 775.00: the GTT fires, and its child rests unfilled at its 780.00 limit.
    fallen_once_placed(rig, broker, monkeypatch, price=775.0)

    news = rig.send(rig.gtt(product="NRML"))
    news += rig.step_for(DECIDED_AFTER_TIMEOUT_S, every_s=2.0)

    # The position has not moved, but the order book holds an order of its child's shape
    # placed since its send: it may have landed and fired.
    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "of its child's shape were placed since"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


@SHAPES
def test_an_unanswered_place_that_landed_late_and_fired_is_not_placed_again(rig, broker, history):
    bought(broker, product="NRML")
    rig.start_read(UNITS, "NRML")
    broker.faults.late_landing("placegttorder", seconds=1.0)
    news = rig.send(rig.gtt(product="NRML"))
    # The server places it a second after the client gave up (the next call it answers
    # lands it), it fires at once, and the order book never lists its child: on a book
    # without history rows only the position's fall by its units from the start read
    # tells it fired.
    broker.clock.advance(1.0)
    broker.client().ping()
    broker.trade("SBIN", "NSE", 789.0)
    broker.hide_order(broker.orders()[-1].order_id, reads=100)

    news += rig.step_for(DECIDED_AFTER_TIMEOUT_S, every_s=2.0)

    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "the position fell by its units since"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


def test_an_unanswered_place_after_an_own_exit_a_lagging_book_hid_is_placed_again(rig, broker):
    bought(broker)
    rig.start_read(UNITS)
    # The session's partial exit sells 5 while the position book lags it (16-session-design
    # F78); the stop for the 5 left is placed, and its connection is refused.
    broker.lag_positionbook()
    rig.run(rig.own_exit(5))
    broker.faults.refuse_connection("placegttorder")
    first = answers(rig.send(rig.gtt(units=5)))
    broker.catch_up_positionbook()

    news = answers(rig.step_for(POSITION_BOOK_LAG_S + DECIDED_AFTER_FINISHED_S, 2.0))

    # The reads made while the exit had not settled judge nothing. Once it settles, the 5
    # the book shows are the start read's 10 moved by the exit's 5: nothing fired, and the
    # place, refused before it reached the server, is placed again.
    (answer,) = news
    assert (first, answer.kind, answer.seq) == ([], GttAnswerKind.PLACED, 2)
    assert len(broker.log.calls("placegttorder")) == 2


@SHAPES
def test_an_unanswered_place_just_after_an_own_entry_that_landed_late_and_fired_is_not_placed_again(
    rig, broker, history
):
    # The session's start read holds no SBIN; its own entry buys 10 SBIN NRML, and its stop
    # is placed a second later with a client timeout of 10 s. The server places it a second
    # after the client gave up, it fires at once, and the order book never lists its child.
    rig.start_read(0, "NRML")
    rig.run(rig.own_entry(UNITS, product="NRML"))
    rig.step_for(1.0)
    rig.orders._client.timeout = 10.0
    broker.faults.late_landing("placegttorder", seconds=1.0)
    news = rig.send(rig.gtt(product="NRML"))
    broker.clock.advance(1.0)
    broker.client().ping()
    broker.trade("SBIN", "NSE", 789.0)
    broker.hide_order(broker.orders()[-1].order_id, reads=1000)

    news += rig.step_for(BROKER_WINDOW_S, every_s=2.0)

    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "the position fell by its units since"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1
    assert [gtt.status for gtt in broker.gtts()] == ["triggered"]


def test_an_unanswered_place_while_an_own_order_works_on_its_leg_is_unresolved_after_the_wait(
    rig, broker
):
    bought(broker)
    rig.start_read(UNITS)
    # The session's exit of 5 rests at a limit of 900.00, far above SBIN's 800.00: a fill
    # of it could show in the position book before its word, so the leg never settles.
    rig.run(rig.own_exit(5, price_type="LIMIT", price=900.0))
    broker.faults.refuse_connection("placegttorder")

    early = answers(rig.send(rig.gtt()))
    early += answers(rig.step_for(DECIDED_AFTER_FINISHED_S + UNITS_WAIT_S - 2.0, 2.0))
    later = answers(rig.step_for(2.0, every_s=2.0))

    # Misses decide 8 s after the answer; the leg's units are waited for 90 s more, then
    # it is never placed again on units that cannot be trusted.
    kinds = [answer.kind for answer in early], [answer.kind for answer in later]
    assert kinds == ([], [GttAnswerKind.UNRESOLVED])
    assert "not settled" in later[0].message
    assert len(broker.log.calls("placegttorder")) == 1


def test_an_unanswered_place_is_placed_again_beside_a_row_of_another_symbol_it_cannot_read(
    rig, broker, monkeypatch
):
    rig.start_read(0)
    with_rows(rig, monkeypatch, lambda rows: [*rows, UNREADABLE_ROW])
    broker.faults.refuse_connection("placegttorder")

    news = answers(rig.send(rig.gtt())) + answers(rig.step_for(DECIDED_AFTER_FINISHED_S, 2.0))

    # A user's TCS row with no leg cannot be this SBIN place: it holds nothing back.
    (answer,) = news
    assert (answer.kind, answer.seq, answer.trigger_id) == (GttAnswerKind.PLACED, 2, FIRST_TRIGGER)


def test_an_unanswered_place_beside_a_row_of_its_symbol_it_cannot_read_is_unresolved(
    rig, broker, monkeypatch
):
    blank = {**UNREADABLE_ROW, "symbol": "SBIN", "exchange": "NSE"}
    with_rows(rig, monkeypatch, lambda rows: [*rows, blank])
    broker.faults.refuse_connection("placegttorder")

    news = answers(rig.send(rig.gtt())) + answers(rig.step_for(DECIDED_AFTER_FINISHED_S, 2.0))

    # An SBIN row the engine cannot read may be this place: it is never placed again beside
    # it, and not waited for without end either.
    (answer,) = news
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert "990001" in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


def test_an_answered_place_a_lagging_book_shows_from_its_third_read_is_placed(rig, broker):
    broker.hide_gtt(FIRST_TRIGGER, reads=2)

    news = rig.send(rig.gtt()) + rig.step_for(4.0, every_s=2.0)

    # A broker's book may show a new GTT a few seconds late: no read is a miss until 5 s
    # after the answer, so the third read (4 s after it) confirms it.
    (answer,) = news
    assert (answer.kind, answer.trigger_id) == (GttAnswerKind.PLACED, FIRST_TRIGGER)
    assert journal_gtt(rig, 1).state is GttState.ACTIVE


@pytest.mark.parametrize("history", [True, False], ids=["history rows", "active only"])
def test_a_place_that_fired_before_its_read_is_answered_before_the_followers_outcomes(
    make_fake_openalgo, tmp_path, monkeypatch, history
):
    fake = broker_with_sbin(make_fake_openalgo(gtt_history=history))
    bought(fake)
    made = GttRig(fake, tmp_path)
    made.messages()
    original = made.orders._client.placegttorder

    def fired_at_once(**fields):
        answer = original(**fields)
        fake.trade("SBIN", "NSE", 789.0)
        return answer

    monkeypatch.setattr(made.orders._client, "placegttorder", fired_at_once)

    news = made.send(made.gtt()) + made.step_for(10.0, every_s=2.0)

    # The session learns the GTT was placed before it learns it fired (GttNews' order).
    found = [item.kind if isinstance(item, GttAnswer) else item.event for item in news]
    assert found == [GttAnswerKind.PLACED, GttEvent.IN_FLIGHT, GttEvent.TRIGGERED]
    made.worker.close()


@SHAPES
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


def test_an_unanswered_modify_the_server_may_still_make_is_cancelled_and_placed_again(rig, broker):
    seq = placed(rig)
    broker.faults.timeout("modifygttorder")

    first = answers(rig.send(modify(rig, seq)))
    (answer,) = answers(rig.step_for(2.0))

    # Two reads show the old terms while the server may still apply the modify: its
    # result cannot be told, so the GTT is cancelled and placed again with its terms.
    assert first == []
    assert (answer.kind, answer.seq, answer.trigger_id) == (
        GttAnswerKind.MODIFIED,
        2,
        SECOND_TRIGGER,
    )
    assert [gtt.status for gtt in broker.gtts()] == ["cancelled", "active"]


def test_an_unanswered_modify_that_never_reached_the_server_is_refused_once_misses_count(
    rig, broker
):
    seq = placed(rig)
    broker.faults.refuse_connection("modifygttorder")

    first = answers(rig.send(modify(rig, seq)))
    early = answers(rig.step_for(DECIDED_AFTER_FINISHED_S - 2.0, every_s=2.0))
    (answer,) = answers(rig.step_for(2.0, every_s=2.0))

    assert (first, early) == ([], [])
    assert answer.kind is GttAnswerKind.MODIFY_REFUSED
    gtt = journal_gtt(rig, seq)
    assert (gtt.terms, gtt.state) == (STOP_TERMS, GttState.ACTIVE)


@SHAPES
def test_an_answered_modify_whose_row_a_read_leaves_out_is_confirmed_by_a_later_read(rig, broker):
    seq = placed(rig)
    broker.hide_gtt(FIRST_TRIGGER, reads=1)

    first = answers(rig.send(modify(rig, seq)))
    later = rig.step_for(2.0, every_s=2.0)

    # A broker's book can drop a row from one read: a modify waits for its misses, as a
    # place does, before it is unresolved.
    assert first == []
    (answer,) = later
    assert (answer.kind, journal_gtt(rig, seq).terms) == (GttAnswerKind.MODIFIED, MODIFIED_TERMS)


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


def test_an_unanswered_cancel_whose_gtt_rests_once_misses_count_is_refused(rig, broker):
    seq = placed(rig)
    broker.faults.timeout("cancelgttorder")

    first = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))
    early = answers(rig.step_for(DECIDED_AFTER_TIMEOUT_S - 2.0, every_s=2.0))
    (answer,) = answers(rig.step_for(2.0, every_s=2.0))

    # Until 150 s from its send the server may still cancel it: no read is a miss before.
    assert (first, early, answer.kind) == ([], [], GttAnswerKind.CANCEL_REFUSED)
    assert journal_gtt(rig, seq).state is GttState.ACTIVE


@pytest.fixture
def fired_unseen(make_fake_openalgo, tmp_path):
    """An active-only book; the account's 10 SBIN CNC, its stop resting. A trade at
    `price` fires the stop before its cancel, which times out (it never reached the
    broker), and the order book leaves the child out of its next read (a broker's book
    lists a new order late): (fake, rig, the stop's journal number, the child's order ID)."""

    def made(price):
        fake = broker_with_sbin(make_fake_openalgo(gtt_history=False))
        bought(fake)
        rig = GttRig(fake, tmp_path)
        rig.messages()
        (answer,) = answers(rig.send(rig.gtt()))
        fake.trade("SBIN", "NSE", price)
        child = fake.orders()[-1].order_id
        fake.hide_order(child, reads=1)
        fake.faults.timeout("cancelgttorder")
        return fake, rig, answer.seq, child

    return made


def test_a_cancelled_gtt_gone_with_the_position_fallen_by_its_units_fired(fired_unseen):
    # The child sells the 10 at once (789.00 is above its 780.00 limit).
    fake, rig, seq, _ = fired_unseen(789.0)

    (answer,) = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))

    assert (answer.kind, fake.position("SBIN", "NSE")) == (GttAnswerKind.CANCEL_REFUSED, 0)
    rig.worker.close()


def test_a_cancelled_gtt_gone_with_no_child_found_is_looked_for_once_more(fired_unseen):
    # The price gaps through the child's 780.00 limit: the child rests unfilled.
    fake, rig, seq, child = fired_unseen(775.0)

    first = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))
    (answer,) = answers(rig.step_for(2.0, every_s=2.0))

    assert (first, answer.kind) == ([], GttAnswerKind.CANCEL_REFUSED)
    assert child in answer.message
    rig.worker.close()


@SHAPES
def test_a_cancel_soon_after_a_partial_exit_is_cancelled(rig, broker):
    bought(broker)
    seq = placed(rig)
    # The partial exit (20-positional-live 2): the stop modified to the 5 units left, then
    # the session's own exit sells 5. The GTT's baseline moves with the session's own fill.
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    rig.run(rig.own_exit(5))
    rig.clock.advance(2.0)

    news = rig.send(rig.gtt(GttAction.CANCEL, seq=seq))
    news += rig.step_for(CHILD_SEARCH_S, every_s=2.0)

    (answer,) = answers(news)
    assert (answer.kind, outcomes(news)) == (GttAnswerKind.CANCELLED, [])
    assert broker.position("SBIN", "NSE") == 5
    assert journal_gtt(rig, seq).state is GttState.CANCELLED


def cut_for_a_partial_exit(rig, broker) -> int:
    """The account's 10 SBIN CNC and its stop, modified to the 5 units a partial exit will
    leave (20-positional-live 2); the exit is not sent yet. The stop's journal number."""
    bought(broker)
    seq = placed(rig)
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    return seq


@SHAPES
def test_a_cancel_sent_while_the_exits_word_is_on_its_way_is_cancelled(rig, broker):
    seq = cut_for_a_partial_exit(rig, broker)
    # The position book shows the exit's sale before the order path hears of it: the order
    # stream holds its word, which the order path's poll 2 s after the send would bring.
    broker.faults.hold_order_updates()
    rig.run(rig.own_exit(5))

    news = rig.send(rig.gtt(GttAction.CANCEL, seq=seq))
    broker.faults.release_order_updates()
    news += rig.step_for(CHILD_SEARCH_S, every_s=2.0)

    (answer,) = answers(news)
    assert (answer.kind, outcomes(news)) == (GttAnswerKind.CANCELLED, [])
    assert journal_gtt(rig, seq).state is GttState.CANCELLED


@SHAPES
def test_a_cancel_after_the_position_book_caught_up_with_the_exit_is_cancelled(rig, broker):
    seq = cut_for_a_partial_exit(rig, broker)
    # The position book lags the exit's fill (16-session-design F78) through a read of the
    # GTT book, then catches up.
    broker.lag_positionbook()
    rig.run(rig.own_exit(5))
    rig.step_for(2.0)
    rig.send(ReadGtts())
    broker.catch_up_positionbook()
    rig.step_for(4.0, every_s=2.0)

    news = rig.send(rig.gtt(GttAction.CANCEL, seq=seq))
    news += rig.step_for(CHILD_SEARCH_S, every_s=2.0)

    (answer,) = answers(news)
    assert (answer.kind, outcomes(news)) == (GttAnswerKind.CANCELLED, [])
    assert broker.position("SBIN", "NSE") == 5


@pytest.mark.parametrize(
    ("hidden_reads", "sign"),
    [(3, "may be its child"), (1000, "the position moved by its units")],
    ids=["its child listed late", "its child never listed"],
)
@SHAPES
def test_a_cancel_of_a_stop_that_fired_just_after_an_own_exit_is_refused(
    rig, broker, history, hidden_reads, sign
):
    # Every fill is the session's own: its entry of 10, settled before a read gives the
    # stop its baseline, then the partial exit (the stop modified to the 5 left, then the
    # exit sells 5).
    rig.run(rig.own_entry(UNITS))
    seq = placed(rig)
    rig.step_for(POSITION_BOOK_LAG_S, every_s=2.0)
    rig.send(ReadGtts())
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    rig.run(rig.own_exit(5))
    # 4 s later the stop fires and its child sells the 5 left; the order book lists the
    # child late or never. The cancel is answered HTTP 502 and never reaches the broker.
    rig.step_for(4.0)
    broker.trade("SBIN", "NSE", 789.0)
    broker.hide_order(broker.orders()[-1].order_id, reads=hidden_reads)
    broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(rig.gtt(GttAction.CANCEL, seq=seq))
    news += rig.step_for(UNITS_WAIT_S, every_s=2.0)

    # The leg fell by the stop's 5 while the exit had not settled: its units are waited
    # for until the child shows, or the exit settles and they show the stop fired. A
    # cancel answered CANCELLED would send a second exit beside the child.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    assert (sign if not history else "it fired") in answer.message
    assert broker.position("SBIN", "NSE") == 0


@pytest.mark.parametrize("history", [False], ids=["active only"])
def test_a_cancel_of_a_fired_stop_while_an_own_order_works_on_its_leg_is_refused_after_the_wait(
    rig, broker
):
    bought(broker)
    seq = placed(rig)
    # The session's exit of 5 rests at a limit of 900.00, far above SBIN's 800.00: the leg
    # never settles. The stop fires and sells the 10; the order book never lists its child.
    rig.run(rig.own_exit(5, price_type="LIMIT", price=900.0))
    broker.trade("SBIN", "NSE", 789.0)
    broker.hide_order(broker.orders()[-1].order_id, reads=1000)
    broker.faults.gateway_error("cancelgttorder", status=502)

    early = answers(rig.send(rig.gtt(GttAction.CANCEL, seq=seq)))
    early += answers(rig.step_for(UNITS_WAIT_S - 2.0, every_s=2.0))
    later = answers(rig.step_for(2.0, every_s=2.0))

    # On a book of active GTTs only the stop is simply gone: the units are waited for
    # 90 s, then the answer that sends no second exit.
    kinds = [answer.kind for answer in early], [answer.kind for answer in later]
    assert kinds == ([], [GttAnswerKind.CANCEL_REFUSED])
    assert "not settled" in later[0].message


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


def refused_on_two_days(broker, folder, purpose, first=None):
    """A stop GTT (or a strategy's GTT entry) the broker refuses on the fixture's day (one
    made for `first` when given) and on the next: the second day's answer, and the
    journal's kept refusals."""
    for day, made_for in ((None, first or purpose), (NEXT_DAY, purpose)):
        if day is not None:
            broker.clock.advance(24 * 60 * 60)
        made = GttRig(broker, folder) if day is None else GttRig(broker, folder, day=day)
        broker.refuse_next_gtt(REFUSAL)
        (answer,) = answers(made.send(made.gtt(purpose=made_for)))
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


def test_a_stops_first_refusal_after_an_entrys_on_an_earlier_day_is_not_kept(broker, tmp_path):
    answer, kept = refused_on_two_days(broker, tmp_path, Purpose.STOP, first=Purpose.ENTRY)

    # The journal keeps what each GTT was for: only a stop's refusal of an earlier day
    # makes a stop's refusal one that repeats (20-positional-live 3.5).
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

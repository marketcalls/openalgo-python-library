"""A GTT's leg's units judged only against a reference (`_live/_gtt_units.py`, the GTT
sender and follower; the lead's rulings M1 to M4 on N3a's review round 5,
20-positional-live 3.1 and 3.3): a settled read of the leg's units made when the GTT could
not yet have fired, moved by the session's own fills since. A read made while the
session's own fills on the leg had not settled judges nothing, and with no reference the
answer is the one that can send no second exit and no second GTT: the cancel of a GTT gone
from the book is refused, an unanswered place is never placed again, a gone GTT is a
mismatch.

The scenarios are the review's probes, over both shapes of a broker's GTT book (one that
keeps the GTTs that ended, one that lists the active ones only): a stop placed at once
after the session's own entry, fired with its child listed late or never, then cancelled;
a move seen before the leg settled, then position books that answer as before it; a
position book that lags the session's own exit; an unanswered place that landed late and
fired while the position book lagged the entry. Where a test says so, the session's start
read (`GttRig.start_read`, as unit N3b's session gives it) is the reference from before
the entry.

The worker runs on the fake OpenAlgo (`gtt_samples`): SBIN at 800.00, a long leg's stop
sells 10 at a trigger of 790.00 with a child limit of 780.00; a trade at 789.00 fires it
and its child sells at once.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from gtt_samples import UNITS, GttRig, answers, bought, broker_with_sbin, outcomes

from openalgo.backtest._live._gtt_child import CHILD_SEARCH_S
from openalgo.backtest._live._gtt_reads import UNITS_WAIT_S
from openalgo.backtest._live._gtt_types import GttAction, GttAnswerKind, GttEvent, ReadGtts
from openalgo.backtest._live._own_fills import POSITION_BOOK_LAG_S

SHAPES = pytest.mark.parametrize("history", [True, False], ids=["history rows", "active only"])
PRODUCTS = pytest.mark.parametrize("product", ["CNC", "NRML"])
CHILD_LISTED = pytest.mark.parametrize(
    "hidden_reads", [3, 1000], ids=["its child listed late", "its child never listed"]
)
#: The words of an answer given for want of a reference.
NO_REFERENCE = "no settled read of its leg's units"
#: Seconds the worker is stepped after a call: past every wait and search of the tests.
LONG_ENOUGH_S = 200.0


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


def placed(rig, product="CNC") -> int:
    """Place the stop in `product` and return its journal number."""
    (answer,) = answers(rig.send(rig.gtt(product=product)))
    assert answer.kind is GttAnswerKind.PLACED
    return answer.seq


def fire(broker, hidden_reads):
    """SBIN trades at 789.00: the stop fires and its child sells at once; the order book
    leaves the child out of its next `hidden_reads` reads."""
    broker.trade("SBIN", "NSE", 789.0)
    broker.hide_order(broker.orders()[-1].order_id, reads=hidden_reads)


def cancel(rig, seq, product="CNC"):
    return rig.gtt(GttAction.CANCEL, seq=seq, product=product)


def stop_right_after_an_own_entry(rig, product):
    """The session's own entry buys 10 SBIN in `product` and its stop is placed at once,
    while the entry's fill has not settled: the stop's journal number."""
    rig.run(rig.own_entry(UNITS, product=product))
    return placed(rig, product)


# ----- M2: the cancel of a stop placed at once after the session's own entry (P6, P6b) -----


@pytest.mark.parametrize("after_s", [4.0, 31.0], ids=["4 s after the entry", "31 s after it"])
@CHILD_LISTED
@PRODUCTS
@SHAPES
def test_the_cancel_of_a_fired_stop_with_no_reference_is_refused(
    rig, broker, history, product, hidden_reads, after_s
):
    seq = stop_right_after_an_own_entry(rig, product)
    rig.step_for(after_s)
    fire(broker, hidden_reads)
    broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(cancel(rig, seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # No read of the leg settled before the stop could fire, and the session gave none: a
    # cancel answered CANCELLED would send a second exit beside the child.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    assert ("it fired" if history else NO_REFERENCE) in answer.message
    assert broker.position("SBIN", "NSE") == 0


@pytest.mark.parametrize("history", [False], ids=["active only"])
def test_the_cancel_of_a_stop_with_no_reference_gone_from_the_book_is_refused_though_none_fired(
    rig, broker
):
    seq = stop_right_after_an_own_entry(rig, "CNC")
    rig.step_for(4.0)

    news = rig.send(cancel(rig, seq)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # The broker cancelled it, but a book of active GTTs only shows a fired GTT the same
    # way: with nothing to judge the units by, the cancel is never confirmed.
    (answer,) = answers(news)
    assert (answer.kind, broker.position("SBIN", "NSE")) == (GttAnswerKind.CANCEL_REFUSED, UNITS)
    assert NO_REFERENCE in answer.message


@pytest.mark.parametrize(
    ("fires", "kind"),
    [(False, GttAnswerKind.CANCELLED), (True, GttAnswerKind.CANCEL_REFUSED)],
    ids=["nothing fired", "it fired"],
)
@PRODUCTS
@SHAPES
def test_a_cancel_4_s_after_an_own_entry_is_judged_once_the_leg_settles_against_the_start_read(
    rig, broker, history, product, fires, kind
):
    rig.start_read(0, product)
    seq = stop_right_after_an_own_entry(rig, product)
    rig.step_for(4.0)
    if fires:
        fire(broker, hidden_reads=1000)
        broker.faults.gateway_error("cancelgttorder", status=502)

    early = answers(rig.send(cancel(rig, seq, product)))
    early += answers(rig.step_for(POSITION_BOOK_LAG_S - 6.0, every_s=2.0))
    later = answers(rig.step_for(LONG_ENOUGH_S, every_s=2.0))

    # A book with history rows shows the stop cancelled or fired at once. On one of active
    # GTTs only it is simply gone: the entry's fill is waited for until it settles (30 s
    # after it was heard), then the units against the start read's 0 moved by the entry's
    # 10 say whether it fired.
    (answer,) = early + later
    assert answer.kind is kind
    assert len(early) == (1 if history else 0)


# ----- M2: a move seen before the leg settled, then stale position books (P8) --------------


@pytest.mark.parametrize("stale_reads", [0, 3], ids=["no stale read", "three stale reads"])
@pytest.mark.parametrize("history", [False], ids=["active only"])
def test_a_move_seen_before_the_leg_settled_is_judged_only_on_a_settled_read(
    rig, broker, monkeypatch, stale_reads
):
    bought(broker)
    seq = placed(rig)
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    client = rig.orders._client
    before_the_exit = client.positionbook()
    rig.run(rig.own_exit(5))
    rig.step_for(4.0)
    fire(broker, hidden_reads=1000)
    broker.faults.gateway_error("cancelgttorder", status=502)
    original, reads = client.positionbook, []

    def lagging(**fields):
        # The cancel's first read shows the stop's sale; the next `stale_reads` answer as
        # before the exit (a position book that lags, 16-session-design F78).
        reads.append(1)
        return before_the_exit if 2 <= len(reads) < 2 + stale_reads else original(**fields)

    monkeypatch.setattr(client, "positionbook", lagging)

    news = rig.send(cancel(rig, seq)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # A read made before the exit settled judges nothing, whatever its units show: the
    # settled read shows the 5 left sold by the stop.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    assert "the position moved by its units" in answer.message


# ----- M2: a position book that lags the session's own exit (P1) ---------------------------


@pytest.mark.parametrize("lag_s", [8.0, 20.0], ids=["caught up 8 s on", "caught up 20 s on"])
@CHILD_LISTED
@SHAPES
def test_a_cancel_while_the_position_book_lags_an_own_exit_waits_for_the_exit_to_settle(
    rig, broker, history, hidden_reads, lag_s
):
    bought(broker)
    seq = placed(rig)
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    broker.lag_positionbook()
    rig.run(rig.own_exit(5))
    rig.step_for(4.0)
    fire(broker, hidden_reads)
    broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(cancel(rig, seq)) + rig.step_for(lag_s - 4.0, every_s=2.0)
    broker.catch_up_positionbook()
    news += rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # The lagging book shows neither the exit nor the stop's sale: it judges nothing until
    # the exit settles, 30 s after its fill was heard, by when the book has caught up.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    if history:
        assert "it fired" in answer.message
    else:
        sign = "may be its child" if hidden_reads == 3 else "the position moved by its units"
        assert sign in answer.message


# ----- M3: an unanswered place just after the session's own entry (P2) ---------------------


def landed_late_and_fired(rig, broker, *, lag_s):
    """The session's own entry buys 10 SBIN NRML, the position book lagging it when `lag_s`
    is above 0; the stop is placed a second later with a client timeout of 10 s. The server
    places it a second after the client gave up, it fires at once, and the order book
    never lists its child; the position book catches up `lag_s` after the entry. The news
    of the place."""
    if lag_s:
        broker.lag_positionbook()
    rig.run(rig.own_entry(UNITS, product="NRML"))
    rig.step_for(1.0)
    rig.orders._client.timeout = 10.0
    broker.faults.late_landing("placegttorder", seconds=1.0)
    news = rig.send(rig.gtt(product="NRML"))
    broker.clock.advance(1.0)
    broker.client().ping()
    fire(broker, hidden_reads=1000)
    if lag_s:
        news += rig.step_for(lag_s - 12.0)
        broker.catch_up_positionbook()
    return news + rig.step_for(LONG_ENOUGH_S, every_s=2.0)


@pytest.mark.parametrize("lag_s", [0.0, 15.0, 25.0], ids=["no lag", "lag 15 s", "lag 25 s"])
@SHAPES
def test_an_unanswered_place_with_no_reference_from_before_it_is_never_placed_again(
    rig, broker, history, lag_s
):
    news = landed_late_and_fired(rig, broker, lag_s=lag_s)

    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert ("of its shape has fired" if history else NO_REFERENCE) in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


@pytest.mark.parametrize("lag_s", [15.0, 25.0], ids=["lag 15 s", "lag 25 s"])
@SHAPES
def test_an_unanswered_place_that_fired_while_the_book_lagged_the_entry_is_unresolved(
    rig, broker, history, lag_s
):
    rig.start_read(0, "NRML")

    news = landed_late_and_fired(rig, broker, lag_s=lag_s)

    # Against the start read's 0 moved by the entry's 10, the settled read's 0 is the stop's
    # fall: placed again, a second stop would rest beside the one that fired.
    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "the position fell by its units since"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


@SHAPES
def test_an_unanswered_place_just_after_an_own_entry_that_never_landed_is_placed_again(rig, broker):
    rig.start_read(0, "NRML")
    rig.run(rig.own_entry(UNITS, product="NRML"))
    rig.step_for(1.0)
    broker.faults.refuse_connection("placegttorder")

    early = answers(rig.send(rig.gtt(product="NRML")))
    early += answers(rig.step_for(POSITION_BOOK_LAG_S - 2.0, every_s=2.0))
    later = answers(rig.step_for(2.0, every_s=2.0))

    # Misses decide 8 s after the refusal; the entry's fill is waited for until it settles
    # (30 s after it was heard), then the units match the start read's 0 moved by the
    # entry's 10: nothing fired, and it is placed again.
    (answer,) = later
    assert (early, answer.kind, answer.seq) == ([], GttAnswerKind.PLACED, 2)
    assert len(broker.log.calls("placegttorder")) == 2


# ----- M4: a stop gone from the book with no child found -----------------------------------


@pytest.mark.parametrize(
    ("start_read", "decided"),
    [(True, GttEvent.CHILD_MISSED), (False, GttEvent.MISMATCH)],
    ids=["a start read", "no start read"],
)
@SHAPES
def test_a_stop_gone_soon_after_an_own_entry_is_judged_against_the_start_read(
    rig, broker, start_read, decided
):
    if start_read:
        rig.start_read(0, "CNC")
    stop_right_after_an_own_entry(rig, "CNC")
    rig.step_for(4.0)
    fire(broker, hidden_reads=1000)

    news = rig.send(ReadGtts()) + rig.step_for(CHILD_SEARCH_S + UNITS_WAIT_S, every_s=2.0)

    # No child is ever listed: after the search the units decide, against a reference from
    # before the stop could fire, or not at all.
    assert [outcome.event for outcome in outcomes(news)] == [GttEvent.IN_FLIGHT, decided]
    if not start_read:
        assert NO_REFERENCE in outcomes(news)[-1].message

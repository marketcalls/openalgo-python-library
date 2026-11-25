"""A GTT's leg's units judged only against a reference (`_live/_gtt_units.py`, the GTT
sender and follower; the lead's rulings M1 to M4 on N3a's review round 5 and R6-1, R6-2 on
round 6, 20-positional-live 3.1 and 3.3): a settled read of the leg's units made when the
GTT could not yet have fired, moved by the session's own fills since. A read made while the
session's own fills on the leg had not settled judges nothing, and with no reference, or
units moved by another amount than the GTT's, the answer is the one that can send no
second exit and no second GTT: the cancel of a GTT gone from the book is refused, an
unanswered place is never placed again, a gone GTT is a mismatch.

The scenarios are the reviews' probes, each over both shapes of a broker's GTT book (one
that keeps the GTTs that ended, one that lists the active ones only) and both positional
products: a stop placed at once after the session's own entry, fired with its child
listed late or never, then cancelled; the session's reads given as references, after a
same-day restart too; a move seen before the leg settled, then position books that answer
as before it; a position book that lags the session's own exit; an unanswered place that
landed late and fired while the position book lagged the entry; a child part filled; a
modify that cannot be told whose stop fired. Where a test says so, the session's start
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
#: Order-book reads that leave the child out: 3 (listed late) or every one (never listed).
LATE, NEVER = 3, 1000
CHILD_LISTED = pytest.mark.parametrize(
    "hidden_reads", [LATE, NEVER], ids=["its child listed late", "its child never listed"]
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


def placed(rig, product) -> int:
    """Place the stop in `product` and return its journal number."""
    (answer,) = answers(rig.send(rig.gtt(product=product)))
    assert answer.kind is GttAnswerKind.PLACED
    return answer.seq


def fire(broker, hidden_reads):
    """SBIN trades at 789.00: the stop fires and its child sells at once; the order book
    leaves the child out of its next `hidden_reads` reads."""
    broker.trade("SBIN", "NSE", 789.0)
    broker.hide_order(broker.orders()[-1].order_id, reads=hidden_reads)


def cancel(rig, seq, product):
    return rig.gtt(GttAction.CANCEL, seq=seq, product=product)


def stop_right_after_an_own_entry(rig, product):
    """The session's own entry buys 10 SBIN in `product` and its stop is placed at once,
    while the entry's fill has not settled: the stop's journal number."""
    rig.run(rig.own_entry(UNITS, product=product))
    return placed(rig, product)


def partly_exited(rig, broker, product):
    """Another client of the account bought 10 SBIN in `product`; the stop is placed, cut
    to the 5 units a partial exit leaves, and the session's own exit sells 5 (a test may
    lag the position book first). The stop's journal number."""
    bought(broker, product=product)
    seq = placed(rig, product)
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5, product=product))
    return seq


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


@pytest.mark.parametrize(
    ("history", "kind"),
    [(True, GttAnswerKind.CANCELLED), (False, GttAnswerKind.CANCEL_REFUSED)],
    ids=["history rows", "active only"],
)
@PRODUCTS
def test_the_cancel_of_a_stop_with_no_reference_that_did_not_fire_is_judged_by_the_book_shape(
    rig, broker, product, kind
):
    seq = stop_right_after_an_own_entry(rig, product)
    rig.step_for(4.0)

    news = rig.send(cancel(rig, seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # The broker cancelled it. A book with history rows shows it cancelled; one of active
    # GTTs only shows a fired GTT the same way, gone: with nothing to judge the units by,
    # that cancel is never confirmed.
    (answer,) = answers(news)
    assert (answer.kind, broker.position("SBIN", "NSE")) == (kind, UNITS)
    if kind is GttAnswerKind.CANCEL_REFUSED:
        assert NO_REFERENCE in answer.message


def cancel_4_s_after_an_own_entry(rig, broker, product, hidden_reads):
    """The session's start read holds no SBIN; its own entry buys 10 and the stop is placed
    at once. 4 s later the stop fires (unless `hidden_reads` is None), the order book
    leaving its child out of `hidden_reads` reads, and the session cancels it, the cancel
    answered 502 when it fired. The answers before the entry settles (within
    `POSITION_BOOK_LAG_S` of its fill) and after."""
    rig.start_read(0, product)
    seq = stop_right_after_an_own_entry(rig, product)
    rig.step_for(4.0)
    if hidden_reads is not None:
        fire(broker, hidden_reads)
        broker.faults.gateway_error("cancelgttorder", status=502)
    early = answers(rig.send(cancel(rig, seq, product)))
    early += answers(rig.step_for(POSITION_BOOK_LAG_S - 6.0, every_s=2.0))
    return early, answers(rig.step_for(LONG_ENOUGH_S, every_s=2.0))


@pytest.mark.parametrize(
    ("history", "hidden_reads", "kind"),
    [
        (True, None, GttAnswerKind.CANCELLED),
        (True, LATE, GttAnswerKind.CANCEL_REFUSED),
        (True, NEVER, GttAnswerKind.CANCEL_REFUSED),
        (False, LATE, GttAnswerKind.CANCEL_REFUSED),
    ],
    ids=[
        "history rows, nothing fired",
        "history rows, it fired, its child listed late",
        "history rows, it fired, its child never listed",
        "active only, it fired, its child listed late",
    ],
)
@PRODUCTS
def test_a_cancel_4_s_after_an_own_entry_is_answered_by_its_row_or_its_child_before_the_leg_settles(
    rig, broker, product, hidden_reads, kind
):
    early, later = cancel_4_s_after_an_own_entry(rig, broker, product, hidden_reads)

    # A book with history rows shows the stop cancelled or fired at once, and a child the
    # order book lists answers as soon as it is found: neither waits for the entry's units.
    (answer,) = early
    assert (answer.kind, later) == (kind, [])


@pytest.mark.parametrize(
    ("hidden_reads", "kind"),
    [(None, GttAnswerKind.CANCELLED), (NEVER, GttAnswerKind.CANCEL_REFUSED)],
    ids=["nothing fired", "it fired, its child never listed"],
)
@pytest.mark.parametrize("history", [False], ids=["active only"])
@PRODUCTS
def test_a_cancel_4_s_after_an_own_entry_of_a_stop_gone_waits_for_the_leg_to_settle(
    rig, broker, product, hidden_reads, kind
):
    early, later = cancel_4_s_after_an_own_entry(rig, broker, product, hidden_reads)

    # The stop is simply gone and no child is listed: the entry's fill is waited for until it
    # settles (30 s after it was heard), then the units against the start read's 0 moved by
    # the entry's 10 say whether it fired.
    (answer,) = later
    assert (early, answer.kind) == ([], kind)


@pytest.mark.parametrize(
    ("fires", "kind"),
    [(False, GttAnswerKind.CANCELLED), (True, GttAnswerKind.CANCEL_REFUSED)],
    ids=["nothing fired", "it fired, its child never listed"],
)
@PRODUCTS
@SHAPES
def test_a_stop_with_no_reference_takes_one_from_a_settled_read_that_shows_it_resting(
    rig, broker, history, product, fires, kind
):
    seq = stop_right_after_an_own_entry(rig, product)
    # 31 s after the entry a read of the GTT book (one on the clock, as unit N3b's session
    # makes every 10 s) finds the entry settled and the stop resting: 10 units, 10 of them
    # the entry's, is the stop's reference.
    rig.step_for(POSITION_BOOK_LAG_S + 1.0)
    rig.send(ReadGtts())
    rig.step_for(4.0)
    if fires:
        fire(broker, hidden_reads=NEVER)
        broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(cancel(rig, seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    (answer,) = answers(news)
    assert answer.kind is kind
    if fires:
        sign = "it fired" if history else "the position moved by its units"
        assert sign in answer.message


# ----- R6-1: the session's reads given as references (`GttFollower.reference`) -------------


@PRODUCTS
@SHAPES
def test_a_start_read_after_a_same_day_restart_counts_the_own_fills_the_order_path_took_back(
    rig, broker, history, product
):
    # The session's own entry bought 10 SBIN; 40 s later the worker restarts on the same day,
    # the order path takes the entry back, and the session gives its start read of 10.
    rig.run(rig.own_entry(UNITS, product=product))
    rig.step_for(POSITION_BOOK_LAG_S + 10.0)
    rig.restart()
    rig.messages()
    rig.start_read(UNITS, product)
    # A new own entry buys 10 more, and the stop for the 20 is placed at once.
    rig.run(rig.own_entry(UNITS, product=product))
    (answer,) = answers(rig.send(rig.gtt(units=2 * UNITS, product=product)))
    rig.step_for(4.0)
    fire(broker, hidden_reads=NEVER)
    broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(cancel(rig, answer.seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # Against the start read's 10 (the day's own 10 counted in it) moved by the new entry's
    # 10, the settled 0 is the stop's fall of 20: a CANCELLED would sell a second time.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    assert ("it fired" if history else "the position moved by its units") in answer.message
    assert broker.position("SBIN", "NSE") == 0


@pytest.mark.parametrize("history", [False], ids=["active only"])
@PRODUCTS
def test_a_read_given_before_the_sessions_own_entry_settled_is_no_reference(rig, product):
    rig.run(rig.own_entry(UNITS, product=product))

    taken = rig.start_read(UNITS, product)
    seq = placed(rig, product)
    rig.step_for(4.0)
    news = rig.send(cancel(rig, seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # The position book may not show the entry yet: the read is ignored, so the stop has no
    # reference and its cancel, once it is gone from the book, is never confirmed.
    (answer,) = answers(news)
    assert (taken, answer.kind) == (False, GttAnswerKind.CANCEL_REFUSED)
    assert NO_REFERENCE in answer.message


@PRODUCTS
def test_a_read_given_after_an_own_fill_heard_since_it_was_made_is_no_reference(rig, product):
    read_ns = rig.clock.now()
    rig.run(rig.own_entry(UNITS, product=product))
    rig.step_for(POSITION_BOOK_LAG_S + 10.0)

    # The session read 0 units before its entry filled, and gives that read once the entry
    # has settled: the 10 the order path counts now are not in it.
    taken = rig.start_read(0, product, read_ns=read_ns)

    assert taken is False


# ----- M2: a move seen before the leg settled, then stale position books (P8) --------------


@pytest.mark.parametrize("stale_reads", [0, 3], ids=["no stale read", "three stale reads"])
@CHILD_LISTED
@PRODUCTS
@SHAPES
def test_a_move_seen_before_the_leg_settled_is_judged_only_on_a_settled_read(
    rig, broker, monkeypatch, history, product, hidden_reads, stale_reads
):
    seq = partly_exited(rig, broker, product)
    client = rig.orders._client
    before_the_exit = client.positionbook()
    rig.run(rig.own_exit(5, product=product))
    rig.step_for(4.0)
    fire(broker, hidden_reads)
    broker.faults.gateway_error("cancelgttorder", status=502)
    original, reads = client.positionbook, []

    def lagging(**fields):
        # The cancel's first read shows the stop's sale; the next `stale_reads` answer as
        # before the exit (a position book that lags, 16-session-design F78).
        reads.append(1)
        return before_the_exit if 2 <= len(reads) < 2 + stale_reads else original(**fields)

    monkeypatch.setattr(client, "positionbook", lagging)

    news = rig.send(cancel(rig, seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # A read made before the exit settled judges nothing, whatever its units show: the
    # child once listed, or the settled read that shows the 5 left sold by the stop, says
    # it fired.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    if history:
        assert "it fired" in answer.message
    else:
        sign = "may be its child" if hidden_reads == LATE else "the position moved by its units"
        assert sign in answer.message


# ----- M2: a position book that lags the session's own exit (P1) ---------------------------


@pytest.mark.parametrize("lag_s", [8.0, 20.0], ids=["caught up 8 s on", "caught up 20 s on"])
@CHILD_LISTED
@PRODUCTS
@SHAPES
def test_a_cancel_while_the_position_book_lags_an_own_exit_waits_for_the_exit_to_settle(
    rig, broker, history, product, hidden_reads, lag_s
):
    seq = partly_exited(rig, broker, product)
    broker.lag_positionbook()
    rig.run(rig.own_exit(5, product=product))
    rig.step_for(4.0)
    fire(broker, hidden_reads)
    broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(cancel(rig, seq, product)) + rig.step_for(lag_s - 4.0, every_s=2.0)
    broker.catch_up_positionbook()
    news += rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # The lagging book shows neither the exit nor the stop's sale: it judges nothing until
    # the exit settles, 30 s after its fill was heard, by when the book has caught up.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    if history:
        assert "it fired" in answer.message
    else:
        sign = "may be its child" if hidden_reads == LATE else "the position moved by its units"
        assert sign in answer.message


# ----- M3: an unanswered place just after the session's own entry (P2) ---------------------


def landed_late_and_fired(rig, broker, product, *, lag_s):
    """The session's own entry buys 10 SBIN in `product`, the position book lagging it when
    `lag_s` is above 0; the stop is placed a second later with a client timeout of 10 s.
    The server places it a second after the client gave up, it fires at once, and the order
    book never lists its child; the position book catches up `lag_s` after the entry. The
    news of the place."""
    if lag_s:
        broker.lag_positionbook()
    rig.run(rig.own_entry(UNITS, product=product))
    rig.step_for(1.0)
    rig.orders._client.timeout = 10.0
    broker.faults.late_landing("placegttorder", seconds=1.0)
    news = rig.send(rig.gtt(product=product))
    broker.clock.advance(1.0)
    broker.client().ping()
    fire(broker, hidden_reads=NEVER)
    if lag_s:
        news += rig.step_for(lag_s - 12.0)
        broker.catch_up_positionbook()
    return news + rig.step_for(LONG_ENOUGH_S, every_s=2.0)


LAGS = pytest.mark.parametrize("lag_s", [0.0, 15.0, 25.0], ids=["no lag", "lag 15 s", "lag 25 s"])


@LAGS
@PRODUCTS
@SHAPES
def test_an_unanswered_place_with_no_reference_from_before_it_is_never_placed_again(
    rig, broker, history, product, lag_s
):
    news = landed_late_and_fired(rig, broker, product, lag_s=lag_s)

    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    assert ("of its shape has fired" if history else NO_REFERENCE) in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


@LAGS
@PRODUCTS
@SHAPES
def test_an_unanswered_place_just_after_an_own_entry_that_fired_is_unresolved(
    rig, broker, history, product, lag_s
):
    rig.start_read(0, product)

    news = landed_late_and_fired(rig, broker, product, lag_s=lag_s)

    # Against the start read's 0 moved by the entry's 10, the settled read's 0 is the stop's
    # fall: placed again, a second stop would rest beside the one that fired.
    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "the position fell by its units since"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


@PRODUCTS
@SHAPES
def test_an_unanswered_place_just_after_an_own_entry_that_never_landed_is_placed_again(
    rig, broker, product
):
    rig.start_read(0, product)
    rig.run(rig.own_entry(UNITS, product=product))
    rig.step_for(1.0)
    broker.faults.refuse_connection("placegttorder")

    early = answers(rig.send(rig.gtt(product=product)))
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
@PRODUCTS
@SHAPES
def test_a_stop_gone_soon_after_an_own_entry_is_judged_against_the_start_read(
    rig, broker, product, start_read, decided
):
    if start_read:
        rig.start_read(0, product)
    stop_right_after_an_own_entry(rig, product)
    rig.step_for(4.0)
    fire(broker, hidden_reads=NEVER)

    news = rig.send(ReadGtts()) + rig.step_for(CHILD_SEARCH_S + UNITS_WAIT_S, every_s=2.0)

    # No child is ever listed: after the search the units decide, against a reference from
    # before the stop could fire, or not at all.
    assert [outcome.event for outcome in outcomes(news)] == [GttEvent.IN_FLIGHT, decided]
    if not start_read:
        assert NO_REFERENCE in outcomes(news)[-1].message


# ----- R6-2: units moved by another amount are taken as "it may have fired" ----------------


def part_filled(broker, hidden_reads):
    """The stop fires and its child sells 4 of its 10 at once, the 6 left resting at its
    limit; the order book leaves the child out of its next `hidden_reads` reads."""
    broker.split_next_fill(4)
    fire(broker, hidden_reads)


@CHILD_LISTED
@PRODUCTS
@SHAPES
def test_the_cancel_of_a_stop_gone_while_its_child_part_filled_is_refused(
    rig, broker, history, product, hidden_reads
):
    bought(broker, product=product)
    rig.start_read(UNITS, product)
    seq = placed(rig, product)
    rig.step_for(4.0)
    part_filled(broker, hidden_reads)
    broker.faults.gateway_error("cancelgttorder", status=502)

    news = rig.send(cancel(rig, seq, product)) + rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # 6 units where 10 were expected: not the stop's 10, but its child part filled may be
    # why. A CANCELLED would send an exit beside the child's 6 left.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.CANCEL_REFUSED
    assert ("it fired" if history else "the position went from 10 to 6 units") in answer.message


@PRODUCTS
@SHAPES
def test_an_unanswered_place_whose_child_may_have_part_filled_is_never_placed_again(
    rig, broker, history, product
):
    bought(broker, product=product)
    rig.start_read(UNITS, product)
    # The server places it a second after the client gave up; it fires at once, its child
    # sells 4 of the 10, and the order book never lists the child.
    broker.faults.late_landing("placegttorder", seconds=1.0)
    news = rig.send(rig.gtt(product=product))
    broker.clock.advance(1.0)
    broker.client().ping()
    part_filled(broker, hidden_reads=NEVER)

    news += rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    (answer,) = answers(news)
    assert (answer.kind, answer.seq) == (GttAnswerKind.UNRESOLVED, None)
    sign = "of its shape has fired" if history else "the position went from 10 to 6 units"
    assert sign in answer.message
    assert len(broker.log.calls("placegttorder")) == 1


def old_units_until_the_cancel(rig, broker, monkeypatch):
    """The GTT book shows every GTT for 10 units, its units before the modify, until the
    worker cancels it to place it again; just before that cancel the stop (now 5 units at
    the broker) fires, its child is never listed, and the cancel is answered 502 without
    reaching the broker."""
    client = rig.orders._client
    book, cancel_call, stale = client.gttorderbook, client.cancelgttorder, [True]

    def old_units(**fields):
        answer = book(**fields)
        if not stale[0]:
            return answer
        rows = [
            {**row, "legs": [{**leg, "quantity": UNITS} for leg in row.get("legs", [])]}
            for row in answer["data"]
        ]
        return {**answer, "data": rows}

    def fired_first(**fields):
        stale[0] = False
        fire(broker, hidden_reads=NEVER)
        broker.faults.gateway_error("cancelgttorder", status=502)
        return cancel_call(**fields)

    monkeypatch.setattr(client, "gttorderbook", old_units)
    monkeypatch.setattr(client, "cancelgttorder", fired_first)


@PRODUCTS
@SHAPES
def test_a_modify_that_cannot_be_told_is_not_placed_again_when_its_stop_fired_for_its_units(
    rig, broker, monkeypatch, history, product
):
    bought(broker, product=product)
    rig.start_read(UNITS, product)
    seq = placed(rig, product)
    rig.run(rig.own_exit(5, product=product))
    rig.step_for(POSITION_BOOK_LAG_S + 10.0)
    rig.send(ReadGtts())
    old_units_until_the_cancel(rig, broker, monkeypatch)

    news = rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5, product=product))
    news += rig.step_for(LONG_ENOUGH_S, every_s=2.0)

    # The broker applied the cut to 5, but its book showed the old 10: the stop is cancelled
    # to be placed again. It fired for the modify's 5 first: placed again, a second stop
    # would rest on the flat leg.
    (answer,) = answers(news)
    assert answer.kind is GttAnswerKind.UNRESOLVED
    assert ("it fired" if history else "the position moved by its units") in answer.message
    assert [gtt.status for gtt in broker.gtts()] == ["triggered"]
    assert broker.position("SBIN", "NSE") == 0

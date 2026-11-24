"""The GTT follower (`_live/_gtt_follow.py`, `_gtt_child.py`, `_gtt_book.py`;
20-positional-live 3.1 to 3.3; the lead's ruling N3a-2): each status of the GTT book read as
3.1's table says, with "gone" (absent from a read that succeeded) its own status; the child
of a GTT that fired found by 3.2's rule; a GTT gone with no child resolved by the leg's units
(3.3); a strategy's GTT entry resolved with the position rising. Table-driven over the two
shapes of a broker's book: one that keeps the GTTs that ended (history rows), one that lists
the active ones only, where they are simply gone.

The worker runs on the fake OpenAlgo (`gtt_samples`): the account bought 10 SBIN CNC at
800.00, and the stop GTT (sell 10 at a trigger of 790.00, child limit 780.00) rests with that
position as its baseline. Expected prices and units are worked by hand from the scenario.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from gtt_samples import (
    FIRST_TRIGGER,
    LIMIT,
    NEXT_DAY,
    PRICE,
    TRIGGER,
    UNITS,
    UNREADABLE_ROW,
    GttRig,
    bought,
    broker_with_sbin,
    calls,
    gtt_news,
    outcomes,
    with_rows,
)
from worker_samples import STRATEGY

from openalgo.backtest._live._gtt_follow import CHILD_SEARCH_S, GTT_BOOK_EVERY_S
from openalgo.backtest._live._gtt_types import (
    PRICE_UNVERIFIED,
    BindGttChild,
    Direction,
    GttAction,
    GttEvent,
    ReadGtts,
)
from openalgo.backtest._live._own_fills import POSITION_BOOK_LAG_S
from openalgo.backtest._live._types import AlertKind, Purpose
from openalgo.backtest._live.journal import GttState

SHAPES = pytest.mark.parametrize("history", [True, False], ids=["history rows", "active only"])


def world(make_fake_openalgo, folder, *, history=True, analyzer=False, **rig_options):
    """The fake (with or without history), the account's 10 SBIN CNC, and a rig whose stop
    GTT rests: (fake, rig, the stop's journal number)."""
    fake = broker_with_sbin(make_fake_openalgo(gtt_history=history, analyzer=analyzer))
    bought(fake)
    rig = GttRig(fake, folder, **rig_options)
    rig.messages()
    rig.send(rig.gtt())
    return fake, rig, 1


def events(news):
    return [outcome.event for outcome in outcomes(news)]


def sell_by_hand(fake, *lots):
    """Another client sells SBIN CNC at the market, one order per (units, price): SBIN
    trades at the price first."""
    other = fake.client()
    for units, price in lots:
        fake.trade("SBIN", "NSE", price)
        other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="CNC",
            quantity=units,
        )


def until_settled(rig):
    """Step the worker every 2 seconds through the child search; the news."""
    return rig.step_for(CHILD_SEARCH_S, every_s=2.0)


# ----- 3.1: the statuses -----------------------------------------------------------------------


@SHAPES
@pytest.mark.parametrize(
    ("status", "kept"),
    [
        ("cancelled", GttEvent.ENDED_OUTSIDE),
        ("deleted", GttEvent.ENDED_OUTSIDE),
        ("disabled", GttEvent.ENDED_OUTSIDE),
        ("expired", GttEvent.EXPIRED),
        ("rejected", GttEvent.REJECTED),
    ],
)
def test_a_gtt_that_ended_outside_the_session_is_told_by_its_row_or_gone_by_its_units(
    make_fake_openalgo, tmp_path, history, status, kept
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    fake.set_gtt_status(FIRST_TRIGGER, status)

    news = rig.send(ReadGtts()) + until_settled(rig)

    # A book with history rows shows how it ended; on one without, it is gone with the
    # position unchanged after the child search: ended outside the session (3.3).
    expected = [kept] if history else [GttEvent.IN_FLIGHT, GttEvent.ENDED_OUTSIDE]
    assert events(news) == expected
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ENDED
    rig.worker.close()


@pytest.mark.parametrize("status", ["active", "transit"])
def test_a_resting_gtt_has_no_outcome(make_fake_openalgo, tmp_path, status):
    fake, rig, seq = world(make_fake_openalgo, tmp_path)
    if status != "active":
        fake.set_gtt_status(FIRST_TRIGGER, status)

    news = rig.send(ReadGtts())

    assert outcomes(news) == []
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ACTIVE
    rig.worker.close()


def test_an_unknown_status_is_taken_as_resting_with_one_alert(make_fake_openalgo, tmp_path):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    fake.set_gtt_status(FIRST_TRIGGER, "paused")

    news = rig.send(ReadGtts()) + rig.send(ReadGtts())

    assert outcomes(news) == []
    (alert,) = [text for kind, text in rig.alerts.sent if kind is AlertKind.ERROR]
    assert "shows status 'paused', which the engine does not know" in alert
    rig.worker.close()


def test_a_row_the_engine_cannot_read_does_not_hide_the_rows_it_can(
    make_fake_openalgo, tmp_path, monkeypatch
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path)
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")
    with_rows(rig, monkeypatch, lambda rows: [*rows, UNREADABLE_ROW])

    news = rig.send(ReadGtts())

    assert events(news) == [GttEvent.ENDED_OUTSIDE]
    rig.worker.close()


def test_a_gtt_whose_own_row_cannot_be_read_is_not_judged_by_that_read(
    make_fake_openalgo, tmp_path, monkeypatch
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path)
    blank = {"trigger_id": FIRST_TRIGGER, "status": "cancelled", "legs": []}
    with_rows(rig, monkeypatch, lambda rows: [blank])

    news = rig.send(ReadGtts())

    assert outcomes(news) == []
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ACTIVE
    rig.worker.close()


def modified_by_hand(fake, trigger, units):
    """The user changes the stop GTT in OpenAlgo to `trigger` (rupees) and `units`."""
    fake.client().modifygttorder(
        trigger_id=FIRST_TRIGGER,
        strategy="manual",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        product="CNC",
        quantity=units,
        price=LIMIT,
        triggerprice_sl=trigger,
    )


@SHAPES
def test_a_resting_gtt_that_shows_other_terms_than_the_journals_is_reported_once(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    modified_by_hand(fake, 795.0, 6)

    news = rig.send(ReadGtts()) + rig.send(ReadGtts())

    (found,) = outcomes(news)
    assert (found.seq, found.event) == (seq, GttEvent.OTHER_TERMS)
    assert found.message == (
        "the broker holds a trigger of 795.00 for 6 units, the journal 790.00 for 10 units"
    )
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ACTIVE
    rig.worker.close()


def test_a_gtt_reported_with_other_terms_is_still_followed(make_fake_openalgo, tmp_path):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    modified_by_hand(fake, 795.0, UNITS)
    rig.send(ReadGtts())
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")

    news = rig.send(ReadGtts())

    assert events(news) == [GttEvent.ENDED_OUTSIDE]
    rig.worker.close()


def test_a_gtt_that_shows_the_journals_terms_again_is_reported_when_they_differ_again(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    news = []
    for trigger in (795.0, TRIGGER, 795.0):
        modified_by_hand(fake, trigger, UNITS)
        news += rig.send(ReadGtts())

    assert events(news) == [GttEvent.OTHER_TERMS, GttEvent.OTHER_TERMS]
    rig.worker.close()


def test_a_failed_read_changes_nothing(make_fake_openalgo, tmp_path):
    fake, rig, seq = world(make_fake_openalgo, tmp_path)
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")
    fake.faults.gateway_error("gttorderbook", status=503)

    news = rig.send(ReadGtts())

    assert outcomes(news) == []
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ACTIVE
    rig.worker.close()


# ----- 3.2: the child ---------------------------------------------------------------------------


@SHAPES
def test_a_stop_that_fired_has_its_one_child_found_in_the_order_book(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    fake.trade("SBIN", "NSE", 790.0)

    news = rig.send(ReadGtts())

    child = fake.orders()[-1].order_id
    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.TRIGGERED]
    found = outcomes(news)[-1]
    assert (found.seq, found.strategy, found.trigger_id, found.order_id) == (
        seq,
        STRATEGY,
        FIRST_TRIGGER,
        child,
    )
    with rig.journal() as journal:
        # Fired, its child not bound until the session follows it as the leg's exit.
        assert journal.gtt(seq).state is GttState.TRIGGERED
    rig.worker.close()


def test_the_sandboxs_child_is_the_one_its_book_names(make_fake_openalgo, tmp_path):
    fake, rig, _ = world(make_fake_openalgo, tmp_path, analyzer=True)
    fake.trade("SBIN", "NSE", 790.0)

    news = rig.send(ReadGtts())

    assert outcomes(news)[-1].order_id == fake.orders()[-1].order_id
    assert calls(fake)[-1] == "gttorderbook"
    rig.worker.close()


def test_a_child_not_found_at_once_is_looked_for_again_2_seconds_later(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    fake.trade("SBIN", "NSE", 790.0)
    fake.faults.gateway_error("orderbook", status=503)

    first = rig.send(ReadGtts())
    later = rig.step_for(2.0)

    assert (events(first), events(later)) == ([GttEvent.IN_FLIGHT], [GttEvent.TRIGGERED])
    rig.worker.close()


def test_two_orders_that_could_each_be_the_child_are_a_mismatch_naming_both(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    fake.clock.advance(1)
    resting = fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="LIMIT",
        product="CNC",
        quantity=UNITS,
        price=900.0,
    )["orderid"]
    fake.trade("SBIN", "NSE", 790.0)

    news = rig.send(ReadGtts())

    child = fake.orders()[-1].order_id
    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.MISMATCH]
    assert set(outcomes(news)[-1].candidates) == {resting, child}
    rig.worker.close()


def test_an_order_placed_before_the_last_read_that_showed_the_gtt_resting_is_no_child(
    make_fake_openalgo, tmp_path
):
    fake = broker_with_sbin(make_fake_openalgo())
    bought(fake)
    fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="LIMIT",
        product="CNC",
        quantity=UNITS,
        price=900.0,
    )
    fake.clock.advance(5)
    rig = GttRig(fake, tmp_path)
    rig.messages()
    rig.send(rig.gtt())
    fake.trade("SBIN", "NSE", 790.0)

    news = rig.send(ReadGtts())

    assert outcomes(news)[-1].order_id == fake.orders()[-1].order_id
    rig.worker.close()


def test_the_session_binds_the_child_and_the_gtt_is_fired(make_fake_openalgo, tmp_path):
    fake, rig, seq = world(make_fake_openalgo, tmp_path)
    fake.trade("SBIN", "NSE", 790.0)
    child = outcomes(rig.send(ReadGtts()))[-1].order_id

    rig.send(BindGttChild(seq, child, kernel_order=7))

    with rig.journal() as journal:
        assert (journal.gtt(seq).state, journal.gtt_child(seq).order_id) == (GttState.FIRED, child)
    rig.worker.close()


# ----- 3.3: a GTT gone with no child ------------------------------------------------------------


@SHAPES
def test_a_gtt_gone_with_the_position_fallen_by_its_units_fired_priced_by_the_trade_book(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    sell_by_hand(fake, (5, 795.0), (5, 793.0))
    # The broker shows it triggered, or (with no history) not at all, and no child of it.
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")

    news = rig.send(ReadGtts()) + until_settled(rig)

    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.CHILD_MISSED]
    missed = outcomes(news)[-1]
    # (5 * 795.00 + 5 * 793.00) / 10 = 794.00.
    assert (missed.units, missed.price, missed.flags) == (UNITS, 794.0, ())
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.FIRED
    rig.worker.close()


def test_a_missed_child_with_no_trade_in_the_trade_book_is_priced_at_its_limit_flagged(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    sell_by_hand(fake, (5, 795.0), (5, 793.0))
    fake.end_day()
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")

    news = rig.send(ReadGtts()) + until_settled(rig)

    missed = outcomes(news)[-1]
    assert (missed.event, missed.price, missed.flags) == (
        GttEvent.CHILD_MISSED,
        LIMIT,
        (PRICE_UNVERIFIED,),
    )
    rig.worker.close()


@SHAPES
def test_a_gtt_gone_with_the_position_unchanged_ended_outside_the_session(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path, history=history)
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")

    news = rig.send(ReadGtts()) + until_settled(rig)

    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.ENDED_OUTSIDE]
    rig.worker.close()


def test_a_gtt_gone_with_the_position_moved_by_other_units_is_a_mismatch(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    sell_by_hand(fake, (4, 795.0))
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")

    news = rig.send(ReadGtts()) + until_settled(rig)

    found = outcomes(news)[-1]
    assert (found.event, found.message) == (
        GttEvent.MISMATCH,
        "the position went from 10 to 6 units",
    )
    rig.worker.close()


def test_a_gtt_first_seen_gone_with_no_baseline_cannot_be_told(make_fake_openalgo, tmp_path):
    fake = broker_with_sbin(make_fake_openalgo())
    bought(fake)
    # The position book cannot be read when the place is confirmed: no baseline is kept.
    fake.faults.gateway_error("positionbook", status=503)
    rig = GttRig(fake, tmp_path)
    rig.messages()
    rig.send(rig.gtt())
    rig.worker.close()
    sell_by_hand(fake, (5, 795.0), (5, 793.0))
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")
    restarted = GttRig(fake, tmp_path)

    news = restarted.step_for(2.0, every_s=2.0) + until_settled(restarted)

    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.MISMATCH]
    assert "no read showed it resting with its position" in outcomes(news)[-1].message
    restarted.worker.close()


def test_a_restart_judges_a_gtt_gone_unseen_by_the_baseline_the_journal_kept(
    make_fake_openalgo, tmp_path
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=False)
    rig.worker.close()
    # After the session's close the stop fires and its child sells the 10 (789.00 is
    # above its 780.00 limit); the next day the order and trade books hold neither.
    fake.trade("SBIN", "NSE", 789.0)
    fake.end_day()
    fake.clock.advance(24 * 60 * 60)
    restarted = GttRig(fake, tmp_path, day=NEXT_DAY)

    news = restarted.step_for(2.0, every_s=2.0) + until_settled(restarted)

    # The day-1 baseline (10) against the units now (0): it fired with its child missed,
    # priced at the child's limit and flagged (20-positional-live 3.3, 4.3).
    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.CHILD_MISSED]
    missed = outcomes(news)[-1]
    assert (missed.units, missed.price, missed.flags) == (UNITS, LIMIT, (PRICE_UNVERIFIED,))
    with restarted.journal() as journal:
        assert journal.gtt(seq).state is GttState.FIRED
    restarted.worker.close()


@SHAPES
def test_a_gtt_one_read_left_out_rests_again_when_the_next_read_shows_it(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    fake.hide_gtt(FIRST_TRIGGER, reads=1)

    news = rig.send(ReadGtts()) + rig.send(ReadGtts())

    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.RESTING]
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ACTIVE
    rig.worker.close()


@SHAPES
def test_a_gone_gtt_is_read_once_more_before_its_units_decide(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    fake.hide_gtt(FIRST_TRIGGER, reads=1)

    news = rig.send(ReadGtts()) + until_settled(rig)

    # Its units unchanged would end it outside the session, and the session would place a
    # second stop beside it: the read made before deciding shows it resting.
    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.RESTING]
    with rig.journal() as journal:
        assert journal.gtt(seq).state is GttState.ACTIVE
    rig.worker.close()


def partly_exited(rig, seq, left):
    """The session's partial exit (20-positional-live 2): the stop GTT modified to the
    `left` units, then the session's own exit sells the rest at the market."""
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=left))
    rig.run(rig.own_exit(UNITS - left))


@SHAPES
def test_a_gtt_cancelled_by_hand_after_a_partial_exit_ended_outside_the_session(
    make_fake_openalgo, tmp_path, history
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    partly_exited(rig, seq, 5)
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")

    news = rig.send(ReadGtts()) + until_settled(rig)

    # The position fell from 10 to 5 by the session's own exit, not by the GTT: its
    # baseline moves with the session's own fills.
    assert fake.position("SBIN", "NSE") == 5
    expected = [GttEvent.ENDED_OUTSIDE] if history else [GttEvent.IN_FLIGHT, GttEvent.ENDED_OUTSIDE]
    assert events(news) == expected
    rig.worker.close()


def test_a_gtt_that_fired_after_a_partial_exit_fired_with_its_child_missed(
    make_fake_openalgo, tmp_path
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=False)
    partly_exited(rig, seq, 5)
    # The broker shows it gone and no child of it; the 5 left were sold in two lots.
    sell_by_hand(fake, (2, 795.0), (3, 793.0))
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")

    news = rig.send(ReadGtts()) + until_settled(rig)

    missed = outcomes(news)[-1]
    # (2 * 795.00 + 3 * 793.00) / 5 = 793.80, the session's own exit at 800.00 left out.
    assert (missed.event, missed.units, missed.price) == (GttEvent.CHILD_MISSED, 5, 793.8)
    rig.worker.close()


def test_the_baseline_a_partial_exit_moved_is_kept_for_a_restart(make_fake_openalgo, tmp_path):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=False)
    partly_exited(rig, seq, 5)
    # The worker takes the exit's fill at its next step, with no read of the GTT book.
    rig.worker.step()
    rig.worker.close()
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")
    restarted = GttRig(fake, tmp_path)

    news = restarted.step_for(2.0, every_s=2.0) + until_settled(restarted)

    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.ENDED_OUTSIDE]
    with restarted.journal() as journal:
        assert journal.gtt(seq).baseline_units == 5
    restarted.worker.close()


def test_the_child_is_looked_for_for_60_seconds_before_the_units_decide(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")
    rig.send(ReadGtts())

    early = rig.step_for(CHILD_SEARCH_S - 2.0, every_s=2.0)
    last = rig.step_for(2.0, every_s=2.0)

    assert (events(early), events(last)) == ([], [GttEvent.ENDED_OUTSIDE])
    rig.worker.close()


# ----- the session's own fills on the leg, not settled yet (16-session-design 5.3, F78) ---------


def baseline(rig, seq):
    with rig.journal() as journal:
        return journal.gtt(seq).baseline_units


def test_a_read_while_the_exits_word_is_on_its_way_counts_the_exit_once(
    make_fake_openalgo, tmp_path
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path)
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    fake.faults.hold_order_updates()
    rig.run(rig.own_exit(5))
    # The position book shows the exit's sale; the order path has not heard of it yet.
    rig.send(ReadGtts())
    fake.faults.release_order_updates()

    rig.step_for(2.0)

    assert baseline(rig, seq) == 5
    rig.worker.close()


def lagged_through_a_read(fake, rig, seq, *, on_the_clock=False):
    """The stop modified to 5, then the session's exit sells 5 while the position book lags
    it (16-session-design F78) through a read of the GTT book: one asked for 2 s later, or
    the read on the clock 10 s after the modify's (a rig whose market is open); then the
    position book catches up."""
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    fake.lag_positionbook()
    rig.run(rig.own_exit(5))
    if on_the_clock:
        rig.step_for(GTT_BOOK_EVERY_S)
    else:
        rig.step_for(2.0)
        rig.send(ReadGtts())
    fake.catch_up_positionbook()
    rig.step_for(4.0, every_s=2.0)


@pytest.mark.parametrize("on_the_clock", [False, True], ids=["a read asked for", "on the clock"])
def test_a_read_while_the_position_book_lags_the_exit_keeps_the_baseline_the_exit_moved(
    make_fake_openalgo, tmp_path, on_the_clock
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, market_open=lambda now_ns: on_the_clock)
    reads = gtt_reads(fake)

    lagged_through_a_read(fake, rig, seq, on_the_clock=on_the_clock)

    # The modify's read and the read during the lag (asked for, or on the clock).
    assert gtt_reads(fake) - reads == 2

    # 10 units less the exit's 5, never the 10 the lagging book still showed.
    assert baseline(rig, seq) == 5
    rig.worker.close()


@pytest.mark.parametrize(
    ("ending", "decided"),
    [("cancelled", GttEvent.ENDED_OUTSIDE), ("fired", GttEvent.CHILD_MISSED)],
    ids=["cancelled by hand", "fired, its child never listed"],
)
@SHAPES
def test_a_gtt_that_ends_after_a_read_of_a_lagging_book_is_judged_by_the_baseline_the_exit_moved(
    make_fake_openalgo, tmp_path, history, ending, decided
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=history)
    lagged_through_a_read(fake, rig, seq)
    if ending == "cancelled":
        fake.set_gtt_status(FIRST_TRIGGER, "cancelled")
    else:
        # The stop fires and its child sells the 5 left at 789.00, but the order book
        # never lists the child.
        fake.trade("SBIN", "NSE", 789.0)
        fake.hide_order(fake.orders()[-1].order_id, reads=1000)

    news = rig.send(ReadGtts()) + until_settled(rig)

    assert events(news)[-1] is decided
    rig.worker.close()


def test_a_gone_gtt_is_decided_once_the_sessions_own_fill_on_its_leg_has_settled(
    make_fake_openalgo, tmp_path
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=False)
    rig.send(rig.gtt(GttAction.MODIFY, seq=seq, units=5))
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")
    rig.send(ReadGtts())
    rig.step_for(CHILD_SEARCH_S - 4.0, every_s=2.0)
    # Just before the child search ends, the session's exit sells 5 while the position book
    # lags it; the book catches up 10 s later.
    fake.lag_positionbook()
    rig.run(rig.own_exit(5))
    early = rig.step_for(10.0, every_s=2.0)
    fake.catch_up_positionbook()

    later = rig.step_for(POSITION_BOOK_LAG_S, every_s=2.0)

    assert (events(early), events(later)) == ([], [GttEvent.ENDED_OUTSIDE])
    rig.worker.close()


def test_a_gone_gtt_whose_leg_an_own_order_still_works_on_is_a_mismatch_after_the_wait(
    make_fake_openalgo, tmp_path
):
    fake, rig, seq = world(make_fake_openalgo, tmp_path, history=False)
    # The session's exit of 5 rests at a limit of 900.00, far above SBIN's 800.00: a fill of
    # it could show in the position book before its word.
    rig.run(rig.own_exit(5, price_type="LIMIT", price=900.0))
    fake.set_gtt_status(FIRST_TRIGGER, "cancelled")

    searched = rig.send(ReadGtts()) + until_settled(rig)
    waited = rig.step_for(POSITION_BOOK_LAG_S, every_s=2.0)

    assert (events(searched), events(waited)) == ([GttEvent.IN_FLIGHT], [GttEvent.MISMATCH])
    assert "not settled" in outcomes(waited)[-1].message
    rig.worker.close()


# ----- a strategy's GTT entry (review M4) ------------------------------------------------------


@pytest.fixture
def entry_world(make_fake_openalgo, tmp_path):
    """No position; a strategy's GTT entry rests: buy 10 SBIN CNC when it trades at 810.00
    or above, the child's limit 820.00."""
    fake = broker_with_sbin(make_fake_openalgo())
    rig = GttRig(fake, tmp_path)
    rig.messages()
    entry = rig.gtt(
        side="BUY", trigger=810.0, limit=820.0, direction=Direction.ABOVE, purpose=Purpose.ENTRY
    )
    rig.send(entry)
    yield fake, rig
    rig.worker.close()


def test_a_gtt_entry_that_fired_has_its_child_found(entry_world):
    fake, rig = entry_world
    fake.trade("SBIN", "NSE", 810.0)

    news = rig.send(ReadGtts())

    assert outcomes(news)[-1].order_id == fake.orders()[-1].order_id


def test_a_gtt_entry_gone_with_the_position_risen_by_its_units_fired(entry_world):
    fake, rig = entry_world
    other = fake.client()
    for price in (805.0, 807.0):
        fake.trade("SBIN", "NSE", price)
        other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="CNC",
            quantity=5,
        )
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")

    news = rig.send(ReadGtts()) + until_settled(rig)

    missed = outcomes(news)[-1]
    # (5 * 805.00 + 5 * 807.00) / 10 = 806.00.
    assert (missed.event, missed.units, missed.price) == (GttEvent.CHILD_MISSED, UNITS, 806.0)


# ----- 3.1: when the book is read --------------------------------------------------------------


def gtt_reads(fake) -> int:
    return len(fake.log.calls("gttorderbook"))


@pytest.mark.parametrize(("open_", "reads"), [(True, 1), (False, 0)], ids=["open", "closed"])
def test_the_book_is_read_every_10_seconds_while_the_market_is_open(
    make_fake_openalgo, tmp_path, open_, reads
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path, market_open=lambda now_ns: open_)
    before = gtt_reads(fake)

    rig.step_for(GTT_BOOK_EVERY_S - 1.0)
    held = gtt_reads(fake)
    rig.step_for(1.0)

    assert (held - before, gtt_reads(fake) - before) == (0, reads)
    rig.worker.close()


def test_an_order_the_journal_does_not_know_on_the_gtts_symbol_brings_a_read_at_once(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    before = gtt_reads(fake)
    sell_by_hand(fake, (1, PRICE))

    rig.worker.step()

    assert gtt_reads(fake) - before == 1
    rig.worker.close()


def test_a_session_holding_no_gtt_reads_no_gtt_book(fake_openalgo, tmp_path):
    broker_with_sbin(fake_openalgo)
    rig = GttRig(fake_openalgo, tmp_path, market_open=lambda now_ns: True)
    rig.messages()

    rig.step_for(30.0)
    sell_by_hand(fake_openalgo, (1, PRICE))
    rig.worker.step()

    assert gtt_reads(fake_openalgo) == 0
    rig.worker.close()


def test_a_restart_follows_the_journals_gtts_and_reads_the_book_at_its_first_step(
    make_fake_openalgo, tmp_path
):
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    rig.worker.close()
    fake.trade("SBIN", "NSE", 790.0)
    restarted = GttRig(fake, tmp_path)
    before = gtt_reads(fake)

    restarted.worker.step()

    news = gtt_news(restarted.messages())
    assert gtt_reads(fake) - before == 1
    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.TRIGGERED]
    restarted.worker.close()


def test_an_order_placed_before_a_restored_gtt_was_last_confirmed_is_no_child(
    make_fake_openalgo, tmp_path
):
    fake = broker_with_sbin(make_fake_openalgo())
    bought(fake)
    sell_by_hand(fake, (UNITS, PRICE))
    fake.clock.advance(5)
    rig = GttRig(fake, tmp_path)
    rig.messages()
    rig.send(rig.gtt())
    rig.worker.close()
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")
    restarted = GttRig(fake, tmp_path)

    news = restarted.step_for(2.0, every_s=2.0)

    # The sale by hand of 10 has the child's shape but came before the GTT rested.
    assert events(news) == [GttEvent.IN_FLIGHT]
    restarted.worker.close()

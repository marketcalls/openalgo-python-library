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
    PRICE,
    UNITS,
    GttRig,
    bought,
    broker_with_sbin,
    calls,
    gtt_news,
    outcomes,
)
from worker_samples import STRATEGY

from openalgo.backtest._live._gtt_follow import CHILD_SEARCH_S, GTT_BOOK_EVERY_S
from openalgo.backtest._live._gtt_types import (
    PRICE_UNVERIFIED,
    BindGttChild,
    Direction,
    GttEvent,
    ReadGtts,
)
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
    fake, rig, _ = world(make_fake_openalgo, tmp_path)
    rig.worker.close()
    sell_by_hand(fake, (5, 795.0), (5, 793.0))
    fake.set_gtt_status(FIRST_TRIGGER, "triggered")
    restarted = GttRig(fake, tmp_path)

    news = restarted.step_for(2.0, every_s=2.0) + until_settled(restarted)

    assert events(news) == [GttEvent.IN_FLIGHT, GttEvent.MISMATCH]
    assert "no read showed it resting with its position" in outcomes(news)[-1].message
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

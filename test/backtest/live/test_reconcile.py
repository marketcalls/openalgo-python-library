"""Reconciliation in a running session, on the fake OpenAlgo (`_live/reconcile.py`,
`_live/_reconcile_act.py`, `_live/_reconcile_guard.py`; 16-session-design 5.1 to 5.3 and
7.2 rows F48, F49 and F78, unit 12): the broker's books read on a schedule and compared
leg by leg, trades attributed to the session's own orders before anything is adopted, an
explained difference adopted, an unexplained one locked and alerted, and nothing ever
sent to make the broker match.

The market is `stops_samples`': SBIN on NSE from 14:40:05 IST, trading 10 units every
10 s at 500.00. `StopHold` buys 10 SBIN at 14:42:01 with an SL-M stop at 497.00; `Hold`
buys 10 with no stop. Another client of the account (`manual`) stands for an order placed
by hand. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import Hold, alerts, ist
from stops_samples import STOP_TRIGGER, StopHold, check_invariants, events, stop_world
from test_session_threads import alone, passing, run_threaded  # noqa: F401

import openalgo.account as account
from openalgo.backtest._live._reconcile_act import Corrections
from openalgo.backtest._live._reconcile_findings import Findings, Sighting
from openalgo.backtest._live._reconcile_rules import Kind, LegBooks
from openalgo.backtest._live._types import LOGGER_NAME, AlertKind
from openalgo.backtest._live.reconcile import RECONCILE_EVERY_S

#: When the other client's order goes in, and when the session is stopped.
BY_HAND = ist(14, 43)
STOPPED = ist(14, 46)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOPPED)
    return world


def by_hand(world, action: str, quantity: int, at_ns: int = BY_HAND) -> None:
    """Another client of the account places a MARKET order of SBIN at `at_ns`."""
    other = world.fake.client()
    world.script.at(
        at_ns,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action=action,
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=quantity,
        ),
    )


def stop_of(world):
    return next(o for o in world.orders() if o.price_type == "SL-M")


def strategy_orders(world, name: str) -> list:
    return [o for o in world.orders() if o.strategy == name]


def mismatches(outcome, kind: str) -> list:
    return [row for row in outcome.reconciliations if row.kind == kind]


# ----- the schedule ---------------------------------------------------------------------


def test_the_books_are_read_when_trading_begins_and_every_30_seconds(world):
    outcome = world.run(Hold, on_stop="keep")
    reads = [e["ts_ns"] for e in events(outcome, "request") if e["type"] == "ReadBooks"]
    assert reads[0] <= ist(14, 40, 16)
    gaps = [b - a for a, b in zip(reads, reads[1:], strict=False)]
    assert max(gaps) <= RECONCILE_EVERY_S * 10**9


def test_a_fill_brings_a_read_3_seconds_after_it_is_heard(world):
    outcome = world.run(Hold, on_stop="keep")
    (fill,) = [f.ts_ns for f in world.fake.fills()]
    reads = [e["ts_ns"] for e in events(outcome, "request") if e["type"] == "ReadBooks"]
    assert fill + 3 * 10**9 in reads


def test_a_session_whose_books_agree_is_clean(world):
    outcome = world.run(StopHold, on_stop="keep")
    assert outcome.reconciliations == ()
    found = {e["found"] for e in events(outcome, "reconcile")}
    assert found == {"agree"}


# ----- F48: an order by hand on the traded symbol ---------------------------------------


def test_F48_a_buy_by_hand_locks_the_strategys_entries_after_two_reads(world, caplog):
    by_hand(world, "BUY", 10)
    outcome = world.run(Hold, on_stop="keep")
    rows = mismatches(outcome, "more")
    assert [row.action for row in rows[:2]] == ["noted", "locked"]
    assert [e["reason"] for e in events(outcome, "lock")] == ["reconciliation"]
    [text] = alerts(caplog, "reconcile_mismatch")
    assert "the engine holds 10 units, OpenAlgo's books show 20" in text
    assert "the excess is not the engine's" in text


def test_a_difference_that_lasts_is_read_again_only_on_the_30_second_schedule(world):
    # The excess is seen at 14:43:05 and confirmed by the read 3 s later; from then on
    # the books are read every 30 s, as when they agree.
    by_hand(world, "BUY", 10)
    outcome = world.run(Hold, on_stop="keep")
    reads = [e["ts_ns"] for e in events(outcome, "request") if e["type"] == "ReadBooks"]
    later = [instant for instant in reads if ist(14, 43, 30) <= instant < STOPPED]
    gaps = {b - a for a, b in zip(later, later[1:], strict=False)}
    assert gaps == {round(RECONCILE_EVERY_S * 10**9)}


def test_a_difference_that_lasts_is_listed_when_seen_and_when_acted_on_only(world):
    by_hand(world, "BUY", 10)
    outcome = world.run(Hold, on_stop="keep")
    assert [(row.kind, row.action) for row in outcome.reconciliations] == [
        ("more", "noted"),
        ("more", "locked"),
    ]


def test_F48_nothing_is_sent_for_the_excess_bought_by_hand(world):
    by_hand(world, "BUY", 10)
    world.run(Hold)
    sells = [o for o in strategy_orders(world, "Hold") if o.action == "SELL"]
    # The safe stop's flatten sells the engine's own 10 only.
    assert [(o.quantity, o.filled) for o in sells] == [(10, 10)]
    assert world.fake.position("SBIN", "NSE") == 10


# ----- F49: the stop's news lost, its trade in the trade book ---------------------------


@pytest.fixture
def stop_filled_unheard(world):
    """From 14:42:30 the order stream is silent and every orderstatus read fails; at 14:43
    the broker fills the stop at its trigger."""

    def silence():
        world.fake.faults.silence_order_updates()
        for _ in range(300):
            world.fake.faults.gateway_error("orderstatus", status=503)

    world.script.at(ist(14, 42, 30), silence)
    world.script.at(BY_HAND, lambda: world.fake.fill(stop_of(world).order_id, price=STOP_TRIGGER))
    return world


def test_F49_the_stops_trade_is_attributed_to_the_stop_and_nothing_is_adopted(
    stop_filled_unheard,
):
    outcome = stop_filled_unheard.run(StopHold, on_stop="keep")
    world = stop_filled_unheard
    attributed = events(outcome, "attributed")
    assert [(e["order_id"], e["filled"]) for e in attributed] == [(stop_of(world).order_id, 10)]
    assert events(outcome, "adopted") == []
    assert outcome.open_positions == ()
    assert [o.action for o in strategy_orders(world, "StopHold")] == ["BUY", "SELL"]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="StopHold")


def test_F49_a_stop_filled_in_two_parts_is_attributed_part_by_part(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # SBIN stops trading at 14:43, so the stop's rest fills only when the broker fills it.
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 43))
    world.stop_at(STOPPED)

    def silence():
        world.fake.faults.silence_order_updates()
        for _ in range(300):
            world.fake.faults.gateway_error("orderstatus", status=503)

    world.script.at(ist(14, 42, 30), silence)
    world.script.at(BY_HAND, lambda: world.fake.fill(stop_of(world).order_id, 4))
    world.script.at(
        BY_HAND + 40 * 10**9, lambda: world.fake.fill(stop_of(world).order_id, price=496.0)
    )
    outcome = world.run(StopHold, on_stop="keep")
    assert [e["filled"] for e in events(outcome, "attributed")] == [4, 10]
    assert events(outcome, "adopted") == []
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()


# ----- explained differences: adopted --------------------------------------------------


def test_a_sale_by_hand_of_part_is_adopted_and_the_stop_cut_down_to_what_is_left(world):
    by_hand(world, "SELL", 6)
    outcome = world.run(StopHold, on_stop="keep")
    assert [(e["side"], e["units"]) for e in events(outcome, "adopted")] == [("SELL", 6)]
    stop = stop_of(world)
    assert (stop.quantity, stop.status) == (4, "trigger pending")
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (4, True)
    assert [row.action for row in mismatches(outcome, "explained_fewer")][:2] == [
        "noted",
        "adopted",
    ]


def test_a_sale_by_hand_of_all_is_adopted_and_the_stop_cancelled(world):
    by_hand(world, "SELL", 10)
    outcome = world.run(StopHold, on_stop="keep")
    assert [(e["side"], e["units"]) for e in events(outcome, "adopted")] == [("SELL", 10)]
    assert stop_of(world).status == "cancelled"
    assert outcome.open_positions == ()
    assert [o.action for o in strategy_orders(world, "StopHold")] == ["BUY", "SELL"]
    assert world.fake.position("SBIN", "NSE") == 0


def test_an_adoption_locks_the_strategys_entries_for_the_session(world):
    by_hand(world, "SELL", 6)
    outcome = world.run(StopHold, on_stop="keep")
    assert [e["reason"] for e in events(outcome, "lock")] == ["reconciliation"]


# ----- the other side: the stop cancelled, nothing else ---------------------------------


def test_a_broker_position_on_the_other_side_cancels_the_stop_and_sends_nothing_else(world, caplog):
    by_hand(world, "SELL", 20)
    outcome = world.run(StopHold, on_stop="keep")
    assert stop_of(world).status == "cancelled"
    assert [o.action for o in strategy_orders(world, "StopHold")] == ["BUY", "SELL"]
    assert world.fake.position("SBIN", "NSE") == -10
    assert any("close it by hand" in text for text in alerts(caplog, "error"))
    assert mismatches(outcome, "other_side")[1].action == "stop cancelled"


# ----- F78: a lagging position book --------------------------------------------------


@pytest.fixture
def lagging(world):
    """The position book stops showing new fills at 14:41:30, before the entry: it shows
    SBIN flat while the stop works and the trade book shows the entry."""
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    return world


def test_F78_a_lagging_position_book_is_alerted_as_a_lag_and_nothing_adopted(lagging, caplog):
    outcome = lagging.run(StopHold, on_stop="keep")
    assert events(outcome, "adopted") == []
    [text] = alerts(caplog, "reconcile_mismatch")
    assert "a lag of the broker's books" in text
    assert [row.action for row in mismatches(outcome, "unexplained")][:2] == ["noted", "locked"]


def test_F78_the_stop_is_not_cancelled_while_the_book_lags(lagging):
    lagging.run(StopHold, on_stop="keep")
    assert stop_of(lagging).status == "trigger pending"
    assert [o.action for o in strategy_orders(lagging, "StopHold")] == ["BUY", "SELL"]


def test_F78_once_the_book_catches_up_the_held_exits_go(lagging):
    # The safe stop's flatten at 14:46 is held back while the book shows the leg flat;
    # the book catches up at 14:46:30 and the exit goes.
    lagging.script.at(ist(14, 46, 30), lagging.fake.catch_up_positionbook)
    outcome = lagging.run(StopHold)
    assert [e["strategy"] for e in events(outcome, "exit_held")][:1] == ["StopHold"]
    (lifted,) = events(outcome, "exit_guard_lifted")
    exit_ = [o for o in strategy_orders(lagging, "StopHold") if o.price_type == "MARKET"][-1]
    assert exit_.placed_ns >= lifted["ts_ns"] >= ist(14, 46, 30)
    assert lagging.fake.position("SBIN", "NSE") == 0


def test_F78_a_lag_that_lasts_past_the_in_flight_limit_is_alerted_to_check_the_account(
    lagging, caplog
):
    lagging.run(StopHold, on_stop="keep")
    texts = [text for text in alerts(caplog, "error") if "check the account" in text]
    assert len(texts) == 1


def test_F78_the_square_off_sends_no_exit_beyond_what_the_book_shows(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    world.stop_at(ist(15, 1))
    world.stop_at(ist(15, 1, 10))
    outcome = world.run(Hold, on_stop="keep")
    held = [e for e in events(outcome, "exit_held") if e["ts_ns"] >= ist(15, 0)]
    assert held
    assert [o.action for o in strategy_orders(world, "Hold")] == ["BUY"]
    assert any("square off" in text and "held back" in text for text in alerts(caplog, "error"))


def test_F78_a_stop_placed_again_is_held_while_the_book_shows_fewer_units(lagging, caplog):
    # Another client cancels the stop at 14:43: the engine places it again for its 10
    # units (R3), which the position book does not show, so the placement is held back.
    other = lagging.fake.client()
    lagging.script.at(
        BY_HAND,
        lambda: other.cancelorder(order_id=stop_of(lagging).order_id, strategy="StopHold"),
    )
    outcome = lagging.run(StopHold, on_stop="keep")
    assert [o.status for o in lagging.orders() if o.price_type == "SL-M"] == ["cancelled"]
    assert [e["strategy"] for e in events(outcome, "exit_held")][:1] == ["StopHold"]
    assert any(
        "stop of SBIN NSE" in text and "held back" in text for text in alerts(caplog, "error")
    )


# ----- a fill of the leg's own stop that lands inside a read ------------------------------


def stop_fills_inside_the_next_read(world: Any, monkeypatch: Any) -> None:
    """From 14:43 the next position-book call answers, then SBIN trades at 496.00 through
    the stop's trigger before the read's order and trade books are read: that read's trade
    book carries the stop's fill while its position book still shows the 10 units."""
    answer_book = account.AccountAPI.positionbook
    armed: list[bool] = []

    def position_book(api: Any, **kwargs: Any) -> Any:
        answer = answer_book(api, **kwargs)
        if armed and armed[0]:
            armed[0] = False
            world.fake.trade("SBIN", "NSE", 496.0, volume=10)
        return answer

    monkeypatch.setattr(account.AccountAPI, "positionbook", position_book)
    world.script.at(BY_HAND, lambda: armed.append(True))


@pytest.fixture(params=["inline", "threads"])
def filled_inside_a_read(request, monkeypatch):
    """`StopHold` (on_stop="keep", stopped at 14:46) on the seam the parameter names, its
    stop filling inside the first read after 14:43; before the run."""
    if request.param == "inline":
        made = request.getfixturevalue("world")
    else:
        made = request.getfixturevalue("alone")
        made.script.at(STOPPED, made.stop.request)
    made.seam = request.param
    stop_fills_inside_the_next_read(made, monkeypatch)
    return made


def run_on_seam(world: Any) -> Any:
    """Run `StopHold` with on_stop="keep" on the world's seam."""
    if world.seam == "inline":
        return world.run(StopHold, on_stop="keep")
    return run_threaded(world, passing(world), "orders", (StopHold,), on_stop="keep")


def test_a_read_inside_which_the_stop_fills_records_no_difference_and_alerts_nothing(
    filled_inside_a_read, caplog
):
    world = filled_inside_a_read
    outcome = run_on_seam(world)
    assert stop_of(world).status == "complete"
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.reconciliations == ()
    assert alerts(caplog, "reconcile_mismatch") == []


def test_the_next_read_after_a_fill_inside_a_read_judges_the_leg_flat(filled_inside_a_read):
    outcome = run_on_seam(filled_inside_a_read)
    after = [e for e in events(outcome, "reconcile") if e["ts_ns"] >= BY_HAND]
    skipped, judged = after[0], after[1]
    assert (skipped["found"], skipped["engine_units"], skipped["broker_units"]) == (
        "fill_during_read",
        0,
        10,
    )
    assert (judged["found"], judged["engine_units"], judged["broker_units"]) == ("agree", 0, 0)
    assert outcome.open_positions == ()


# ----- a late comparison (the leg in flight past the limit) -------------------------------

#: The leg the comparisons below are of, as the alerts name it; its entries lock.
LEG = SimpleNamespace(
    name="Hold", symbol="SBIN", exchange="NSE", open_units=lambda: 0.0, lock=lambda reason: True
)


class Judging:
    """The reconciler's findings over a stand-in session that keeps the alerts sent and
    the mismatches counted, with the real corrections."""

    def __init__(self) -> None:
        self.sent: list[tuple[AlertKind, str]] = []
        session = SimpleNamespace(
            now=lambda: 0,
            event=lambda *args, **fields: None,
            alert=lambda kind, text: self.sent.append((kind, text)),
        )
        session.mismatches = 0
        self.session = session
        self.findings = Findings(session, Corrections(session, {}, set()), 180.0)

    def late(self, kind: Kind, engine: int, broker: int) -> None:
        """A late comparison of the leg that found `kind`, the engine's and OpenAlgo's
        units."""
        books = LegBooks(broker, (), ())
        self.findings.judged(LEG, {(kind, engine, broker)}, books, frozenset(), late=True)


#: A position difference of each kind: (kind, the engine's units, OpenAlgo's).
DIFFERENCES = [
    (Kind.MORE, 0, 10),
    (Kind.EXPLAINED_FLAT, 10, 0),
    (Kind.EXPLAINED_FEWER, 10, 6),
    (Kind.UNEXPLAINED, 10, 0),
    (Kind.OTHER_SIDE, 10, -10),
]
DIFFERENCE_IDS = ["more", "explained flat", "explained fewer", "unexplained", "other side"]


@pytest.mark.parametrize(("kind", "engine", "broker"), DIFFERENCES, ids=DIFFERENCE_IDS)
def test_a_late_comparison_alerts_a_difference_at_once_when_first_seen(kind, engine, broker):
    judging = Judging()
    judging.late(kind, engine, broker)
    rows = [(row.kind, row.action) for row in judging.findings.rows]
    assert rows == [(kind.value, "alerted")]
    [(alert_kind, text)] = judging.sent
    assert alert_kind is AlertKind.ERROR
    assert f"the engine holds {engine} units, OpenAlgo's books show {broker}" in text


@pytest.mark.parametrize(("kind", "engine", "broker"), DIFFERENCES, ids=DIFFERENCE_IDS)
def test_a_late_comparison_takes_no_sighting_locks_nothing_and_is_kept_for_the_close(
    kind, engine, broker
):
    judging = Judging()
    judging.late(kind, engine, broker)
    assert judging.findings.unconfirmed(LEG.name) is None
    assert judging.session.mismatches == 0
    assert judging.findings.late(LEG.name) == Sighting(engine, broker)


def test_a_difference_seen_late_twice_is_alerted_once():
    judging = Judging()
    judging.late(Kind.MORE, 0, 10)
    judging.late(Kind.MORE, 0, 10)
    assert len(judging.sent) == 1
    assert [row.action for row in judging.findings.rows] == ["alerted", "alerted"]


def test_a_difference_alerted_late_is_still_acted_on_once_compared_in_time():
    judging = Judging()
    judging.late(Kind.MORE, 0, 10)
    books = LegBooks(10, (), ())
    judging.findings.judged(LEG, {(Kind.MORE, 0, 10)}, books, frozenset(), late=False)
    assert [row.action for row in judging.findings.rows] == ["alerted", "locked"]
    assert judging.findings.late(LEG.name) is None


def test_a_difference_acted_on_in_time_is_not_kept_for_the_close_when_then_seen_late():
    # Seen in two comparisons in time it was acted on and counted: a late comparison that
    # sees it again names nothing new at the close.
    judging = Judging()
    books = LegBooks(10, (), ())
    for _ in range(2):
        judging.findings.judged(LEG, {(Kind.MORE, 0, 10)}, books, frozenset(), late=False)
    judging.late(Kind.MORE, 0, 10)
    assert judging.session.mismatches == 1
    assert judging.findings.late(LEG.name) is None


def test_a_difference_seen_late_twice_is_still_kept_for_the_close():
    judging = Judging()
    judging.late(Kind.MORE, 0, 10)
    judging.late(Kind.MORE, 0, 10)
    assert judging.findings.late(LEG.name) == Sighting(0, 10)

"""The fault rows of a shared instrument on its members' stops and exits, on the fake
OpenAlgo (18-multi-strategy 5.2 rows F97 and F118, 4.9 and 4.14 as the lead decided it in
11.0a; unit M8): after the square-off the session ends once each member's kernel is flat,
and the close's final read verifies the group's net against the members' sum, a
difference going into `reconciliations` (5.1 I7); a member's closing order the broker
refuses while a sale by hand has left it fewer units is held, while the reconciler's rows
share the sale out.

The market and the strategies are `shared_samples`': SBIN at 500.00, A first in call
order; `SbinTrend` (100 units, target 502.00) and `SbinBreakout` (50 units) buy at
14:42:01 IST, `LateTrend` (100 units) at 14:43:01, each with a stop 3 points below (an SL-M
at 497.00). Every test runs the session invariants, those of groups and those restated per
member among them. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from session_trace import WITHIN
from shared_samples import A_UNITS, B_UNITS, LateTrend, SbinBreakout, SbinTrend, sbin_orders, terms
from stops_samples import stop_world

from openalgo.backtest._live._margin_reject import MARGIN_LOCK
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._session_news import RECONCILE_LOCK
from openalgo.backtest._live._stops_cover import REFUSED_FOR_MARGIN
from openalgo.backtest._live._types import LOGGER_NAME

#: The square-off of NSE's MIS positions, and the read of the books 30 s after it
#: (16-session-design 3.4, `SQUARE_OFF_VERIFY_AFTER_S`).
SQUARE_OFF = ist(15, 0)
VERIFY_AT = SQUARE_OFF + 30 * NS
#: The broker's words of the refusal (18-multi-strategy F118).
MARGIN_WORDS = "RMS: margin exceeds"
#: The purchase by hand of F97's variant, in units, and the instrument's row of the two.
BOUGHT = 10
BOTH = "SbinTrend and SbinBreakout"
#: The sale by hand of F118: 120 of the members' 150, so the broker holds 30.
SOLD = 120
SOLD_AT = ist(14, 44)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def buy_by_hand(world, units: int, at_ns: int) -> None:
    """Another client buys `units` SBIN at market at `at_ns`."""
    other = world.fake.client()
    world.script.at(
        at_ns,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=units,
        ),
    )


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]


# ----- F97: the square-off of a shared instrument --------------------------------------------


def test_F97_the_session_ends_with_each_member_flat_and_the_closes_read_finds_the_group_flat(
    world, caplog
):
    outcome = world.run([SbinTrend, SbinBreakout])

    # Each member's square-off exit fills as it is placed. The session ends (3.4) once each
    # member's kernel is flat; the close's final read, taken after both exits filled,
    # compares the group's net with the members' sum and finds both flat.
    exits = [sbin_orders(world, name)[-1] for name in ("SbinTrend", "SbinBreakout")]
    assert [(o.action, o.price_type, o.status) for o in exits] == [
        ("SELL", "MARKET", "complete")
    ] * 2
    both_filled = max(o.placed_ns for o in exits)
    reads = [c.ts_ns for c in world.fake.log.calls("positionbook") if c.ts_ns > both_filled]
    assert reads and reads[0] < VERIFY_AT
    assert [t for t in alerts(caplog, "error") if "still not flat after square-off" in t] == []
    assert (outcome.stopped_by, world.fake.position("SBIN", "NSE")) == ("session_end", 0)
    assert (outcome.summary.flat, outcome.summary.reconcile_mismatches) == (True, 0)


def test_F97_a_buy_by_hand_at_the_square_off_is_one_difference_of_the_group_at_the_close(world):
    # Another client buys 10 SBIN just before the members' square-off exits go. The
    # session still ends once each member's kernel is flat: 3.4's end on the broker's books
    # too is not built, for a strategy alone either. The close's final read finds the
    # group's net 10 against the members' 0 and names it once, for the group (5.1 I7).
    buy_by_hand(world, BOUGHT, SQUARE_OFF + 2 * NS)
    outcome = world.run([SbinTrend, SbinBreakout])

    assert (outcome.stopped_by, world.fake.position("SBIN", "NSE")) == ("session_end", BOUGHT)
    # Each member's row carries the group's figures: the broker's net, the members' sum
    # and the external trade; nothing is sent once the session has ended.
    table = session_result(outcome).reconciliations
    columns = ["strategy", "kind", "broker_units", "action", "group", "group_units"]
    group = "SbinTrend, SbinBreakout"
    assert table[[*columns, "external_units"]].astype(object).values.tolist() == [
        ["SbinTrend", "more", BOUGHT, "noted", group, 0, BOUGHT],
        ["SbinBreakout", "more", BOUGHT, "noted", group, 0, BOUGHT],
    ]
    summary = outcome.summary
    assert (summary.reconcile_mismatches, summary.flat) == (1, False)
    assert summary.differing == (
        f"SBIN NSE {BOUGHT} units of SBIN for {BOTH} together in OpenAlgo's books, their 0",
    )
    (row,) = outcome.open_positions
    assert (row.strategy, row.units, row.stop_confirmed) == (BOTH, BOUGHT, False)


# ----- F118: a closing order refused while a sale by hand left fewer units -------------------


def sale_then_bs_stop_refused(world) -> None:
    """Another client sells 120 at `SOLD_AT`, then cancels B's stop: the kernel places it
    again (R3), and the broker refuses that one for margin right after accepting it, the
    broker holding 30 of the members' 150."""
    other = world.fake.client()

    def sell_then_cancel():
        other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=SOLD,
        )
        world.fake.reject_next(MARGIN_WORDS, price_type="SL-M")
        stop = next(o for o in sbin_orders(world, "SbinBreakout") if o.price_type == "SL-M")
        other.cancelorder(order_id=stop.order_id, strategy="SbinBreakout")

    world.script.at(SOLD_AT, sell_then_cancel)


def run_f118(world):
    """F118 to Ctrl+C at 14:46 with the positions kept: A (`LateTrend`) opened last, so
    the allocation of 4.7 takes A's 100 first and 20 of B's 50, leaving B 30."""
    sale_then_bs_stop_refused(world)
    world.trace.waive(
        WITHIN,
        "18-multi-strategy F118: B's stop is placed again (R3) after the sale by hand and "
        "before a comparison sees it, for more than the broker holds (16-session-design 5.4 "
        "'As built': the exit guard opens on a comparison); the broker refuses it",
    )
    world.stop_at(ist(14, 46))
    return world.run([LateTrend, SbinBreakout], on_stop="keep")


def test_F118_the_refused_stop_is_one_request_none_repeated_and_no_exit_is_sent(world):
    run_f118(world)

    b_orders = sbin_orders(world, "SbinBreakout")
    sent_after = [o for o in b_orders if o.placed_ns >= SOLD_AT]
    # Nothing is sent again for the refusal: no market exit, no exit retry and no stop
    # placed again, until the safe stop's protect at 14:46 places one for the 30 units
    # the allocation left B, once a read has agreed with the group (4.14a item 2).
    assert terms(sent_after) == [
        ("SELL", B_UNITS, "SL-M", "rejected"),
        ("SELL", A_UNITS + B_UNITS - SOLD, "SL-M", "trigger pending"),
    ]
    assert sent_after[1].placed_ns >= ist(14, 46)


def test_F118_b_is_locked_alerted_and_a_read_is_asked_at_once(world, caplog):
    outcome = run_f118(world)

    (held,) = events(outcome, "margin_reject")
    assert (held["strategy"], held["reason"]) == ("SbinBreakout", MARGIN_WORDS)
    locks = [(e["strategy"], e["reason"]) for e in events(outcome, "lock")]
    assert ("SbinBreakout", MARGIN_LOCK) in locks
    reads = [e["ts_ns"] for e in events(outcome, "request") if e["type"] == "ReadBooks"]
    assert any(held["ts_ns"] <= read <= held["ts_ns"] + NS for read in reads)
    words = [t for t in alerts(caplog, "error") if "refused for margin by the broker" in t]
    assert len(words) == 1 and MARGIN_WORDS in words[0]


def test_F118_the_reconcilers_rows_then_share_the_sale_out(world):
    outcome = run_f118(world)

    adopted = {e["strategy"]: (e["side"], e["units"]) for e in events(outcome, "adopted")}
    assert adopted == {"LateTrend": ("SELL", A_UNITS), "SbinBreakout": ("SELL", SOLD - A_UNITS)}
    locks = {(e["strategy"], e["reason"]) for e in events(outcome, "lock")}
    assert {("LateTrend", RECONCILE_LOCK), ("SbinBreakout", RECONCILE_LOCK)} <= locks
    assert world.fake.position("SBIN", "NSE") == A_UNITS + B_UNITS - SOLD


def test_F118_b_is_counted_unprotected_for_the_refusal_and_listed_with_what_is_left(world):
    outcome = run_f118(world)

    windows = [w for w in outcome.unprotected if w.strategy == "SbinBreakout"]
    (refused,) = [w for w in windows if w.reason == REFUSED_FOR_MARGIN]
    # From the refusal to the stop the safe stop places at 14:46.
    assert SOLD_AT <= refused.from_ns < SOLD_AT + 5 * NS
    assert ist(14, 46) <= refused.to_ns < ist(14, 46, 5)
    rows = {p.strategy: p for p in outcome.open_positions}
    assert rows["SbinBreakout"].units == A_UNITS + B_UNITS - SOLD
    assert "LateTrend" not in rows

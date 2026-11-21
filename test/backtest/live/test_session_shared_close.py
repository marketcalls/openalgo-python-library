"""What the close says of an instrument two strategies of the session share
(`_live/_session_named.py`, `_live/_session_close.py`, `_live/_alerts_summary.py`;
18-multi-strategy 4.11, unit M7).

OpenAlgo's books show one net position of the instrument for both members, so the close
gives it a row of its own in `open_positions` after the members' rows, when that net is a
position: the members' names, OpenAlgo's net, and whether it equals the members' sum. The
DAILY_SUMMARY gets one "Shared" line per shared instrument with the same figures, and a
difference of the group the close names is named and counted once.

The market and the strategies are `shared_samples`': SBIN at 500.00; A (`SbinTrend`) buys
100 at 14:42:01 IST and B 50 (`SbinBreakout` at once, `LateBreakout` at 14:43:01), each
with an SL-M stop 3 points below (497.00). Expected values come from the fake's books and
the scenario.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import Hold, ist
from shared_samples import A_UNITS, B_UNITS, LateBreakout, NeverB, SbinBreakout, SbinTrend
from stops_samples import stop_world

from openalgo.backtest._live import reconcile as reconcile_module
from openalgo.backtest._live._alerts_summary import summary_text
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._session_named import CARRIED, SharedPosition
from openalgo.backtest._live._types import LOGGER_NAME

HELD = A_UNITS + B_UNITS
#: The instrument's row's name, and the start of its summary line.
BOTH = "SbinTrend and SbinBreakout"
LINE = f"SBIN NSE ({BOTH}): OpenAlgo's net"
#: How the close names a net its group's last read saw late, as that read saw it.
LATE_WORDS = (
    "at a late read, while an order of a member was in flight past the limit, neither "
    "carried by their fills since nor confirmed by a later read"
)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def trade_by_hand(world, action: str, units: int, at_ns: int) -> None:
    """Another client trades `units` SBIN at market at `at_ns`."""
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
            quantity=units,
        ),
    )


def kept(world, members, stop_ns: int = ist(14, 45)):
    """`members` run until Ctrl+C at `stop_ns`, keeping their positions: the outcome."""
    world.stop_at(stop_ns)
    return world.run(members, on_stop="keep")


def row_of(outcome, strategy: str):
    (row,) = [p for p in outcome.open_positions if p.strategy == strategy]
    return row


# ----- the instrument's row in open_positions ---------------------------------------------


def test_a_shared_instrument_has_a_row_with_openalgos_net_after_the_members_rows(world):
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert [p.strategy for p in outcome.open_positions] == ["SbinTrend", "SbinBreakout", BOTH]
    row = row_of(outcome, BOTH)
    assert (row.symbol, row.exchange, row.units, row.stop_order_id) == ("SBIN", "NSE", HELD, "")
    assert row.note == (
        f"OpenAlgo's net position of SBIN for {BOTH} together, by the last read: it equals "
        "their sum; each holds its own stop on its row"
    )


def test_the_instruments_row_is_covered_when_the_net_agrees_and_each_members_stop_is(world):
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert row_of(outcome, "SbinTrend").stop_confirmed
    assert row_of(outcome, "SbinBreakout").stop_confirmed
    assert row_of(outcome, BOTH).stop_confirmed


def test_after_a_sale_by_hand_shared_out_the_instruments_row_shows_the_net_left(world):
    trade_by_hand(world, "SELL", 60, ist(14, 43, 30))
    outcome = kept(world, [SbinTrend, LateBreakout])
    row = row_of(outcome, "SbinTrend and LateBreakout")
    assert (row.units, row.stop_confirmed) == (HELD - 60, True)
    assert "it equals their sum" in row.note


def test_an_excess_on_a_flat_group_is_the_instruments_row_and_the_day_is_not_flat(world):
    # Neither member trades; another client buys 10: the broker holds what no member does.
    trade_by_hand(world, "BUY", 10, ist(14, 43, 30))
    never_a = type("NeverA", (NeverB,), {})
    outcome = kept(world, [never_a, NeverB])
    (row,) = outcome.open_positions
    assert (row.strategy, row.units, row.stop_confirmed) == ("NeverA and NeverB", 10, False)
    assert "their sum is 0, so they differ; check the position book in OpenAlgo" in row.note
    assert not outcome.summary.flat
    assert outcome.summary.reconcile_mismatches == 1


def test_a_net_openalgo_shows_flat_against_the_members_units_is_a_row_saying_they_differ(world):
    # The position book lags from before the entries: it shows SBIN flat, while each
    # member holds its units on its own row.
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert [p.strategy for p in outcome.open_positions] == ["SbinTrend", "SbinBreakout", BOTH]
    row = row_of(outcome, BOTH)
    assert (row.units, row.stop_confirmed) == (0, False)
    assert f"their sum is {HELD}, so they differ; check the position book in OpenAlgo" in row.note


def test_the_results_warning_names_each_member_and_the_instruments_net_apart(world):
    result = session_result(kept(world, [SbinTrend, SbinBreakout]))
    (left,) = [w for w in result.warnings if w.startswith("positions were still open")]
    assert left == (
        f"positions were still open when the session ended (SbinTrend {A_UNITS} SBIN, "
        f"SbinBreakout {B_UNITS} SBIN; OpenAlgo's net {HELD} SBIN for {BOTH} together): "
        "result.open_positions says what to do with each"
    )


@pytest.mark.parametrize("lagging", [False, True], ids=["no difference", "a difference"])
def test_a_shared_sessions_reconciliations_have_the_groups_columns_rows_or_not(world, lagging):
    # A strategy alone keeps the table's columns of 2.3.0 (test_live_result.py).
    if lagging:
        world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    table = session_result(kept(world, [SbinTrend, SbinBreakout])).reconciliations
    assert bool(len(table)) is lagging
    assert list(table.columns)[-6:] == [
        "group",
        "group_units",
        "member_units",
        "external_units",
        "allocation",
        "stops_cut",
    ]
    assert [str(table[c].dtype) for c in ("group_units", "external_units")] == ["Int64"] * 2


def test_the_results_repr_counts_the_members_positions_not_the_instruments_row(world):
    result = session_result(kept(world, [SbinTrend, SbinBreakout]))
    assert len(result.open_positions) == 3
    assert ", 2 positions open" in repr(result)


def test_a_late_read_names_the_net_as_it_saw_it_not_carried_by_fills_heard_after(
    world, monkeypatch
):
    # SbinTrend is held in flight from 14:42:30 (a test seam, as in
    # test_session_close_late_acted.py), so every read from 14:45:31 compares the group
    # late. SbinBreakout's stop sells its 50 at the broker at 14:45:40, its word held; the
    # close's final read (Ctrl+C at 14:45:41) sees OpenAlgo's net 100 against the members'
    # 150, and the fill reaches SbinBreakout's engine after it, by the poll the read asks
    # for. The net is 100, as read: carrying it by that fill would count the fill twice.
    in_flight = reconcile_module.in_flight

    def held_in_flight(session, trader) -> bool:
        held = trader.name == "SbinTrend" and session.now() >= ist(14, 42, 30)
        return held or in_flight(session, trader)

    monkeypatch.setattr(reconcile_module, "in_flight", held_in_flight)
    world.script.at(ist(14, 45, 40), lambda: b_stop_fills_unheard(world))
    outcome = kept(world, [SbinTrend, SbinBreakout], ist(14, 45, 41))
    assert world.fake.position("SBIN", "NSE") == A_UNITS
    assert outcome.summary.shared == (
        f"{LINE} {A_UNITS} {LATE_WORDS}; their sum {A_UNITS}: not confirmed",
    )
    row = row_of(outcome, BOTH)
    assert (row.units, row.stop_confirmed) == (A_UNITS, False)
    assert f"{LATE_WORDS}: their sum is {A_UNITS}; check the position book" in row.note


def b_stop_fills_unheard(world) -> None:
    """SbinBreakout's stop fills whole at the broker now, its word held on the stream."""
    world.fake.faults.hold_order_updates()
    (stop,) = [o for o in world.fake.orders() if o.strategy == "SbinBreakout" and o.trigger_price]
    world.fake.fill(stop.order_id, price=497.0)


def test_a_strategy_alone_has_no_instruments_row_and_no_shared_line(world):
    outcome = kept(world, [Hold])
    assert [p.strategy for p in outcome.open_positions] == ["Hold"]
    assert outcome.summary.shared == ()
    assert "Shared" not in summary_text(outcome.summary)


# ----- the DAILY_SUMMARY ------------------------------------------------------------------


def test_the_summary_has_a_line_per_shared_instrument_after_the_reconcile_line(world):
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert outcome.summary.shared == (
        f"{LINE} {HELD} by the last read, their sum {HELD}: they agree",
    )
    lines = summary_text(outcome.summary).splitlines()
    shared = next(i for i, line in enumerate(lines) if line.startswith("Shared"))
    assert lines[shared - 1].startswith("Reconcile")
    assert (
        lines[shared]
        == f"Shared      : {LINE} {HELD} by the last read, their sum {HELD}: they agree"
    )


def test_the_line_of_a_difference_no_trade_explains_says_the_net_and_the_sum_differ(world):
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert outcome.summary.shared == (f"{LINE} 0 by the last read, their sum {HELD}: they differ",)


def test_a_session_ended_flat_says_the_net_and_the_sum_agree_at_nothing(world):
    outcome = world.run([SbinTrend, SbinBreakout])
    assert outcome.open_positions == ()
    assert outcome.summary.shared == (f"{LINE} 0 by the last read, their sum 0: they agree",)


def test_a_difference_first_seen_by_the_closes_read_is_named_and_counted_once(world):
    # Ctrl+C at 14:45, two seconds after another client sold 60: the close's final read is
    # the first to see it. Both members' sightings hold it; the summary names the group once.
    for member in ("SbinTrend", "SbinBreakout"):
        world.trace.kept_stop_by_decision(member, "3.3 step 5 and 5.4 As built")
    trade_by_hand(world, "SELL", 60, ist(14, 44, 58))
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert outcome.summary.differing == (
        f"SBIN NSE {HELD - 60} units of SBIN for {BOTH} together in OpenAlgo's books, their {HELD}",
    )
    assert outcome.summary.reconcile_mismatches == 1
    assert outcome.summary.shared == (
        f"{LINE} {HELD - 60} by the last read, their sum {HELD}: they differ",
    )


# ----- the instrument's words on their own ------------------------------------------------


@pytest.mark.parametrize(
    ("net", "held", "late", "members_open", "listed", "agrees"),
    [
        (150, 150, False, True, True, True),
        (90, 150, False, True, True, False),
        (10, 0, False, False, True, False),
        (0, 150, False, True, True, False),
        (0, 150, False, False, False, False),
        (0, 0, False, False, False, True),
        (None, 150, False, True, False, False),
        (150, 150, True, True, True, False),
        (0, 0, True, False, False, False),
    ],
    ids=[
        "agreeing",
        "differing",
        "an excess no member holds",
        "flat at OpenAlgo, a member listed open",
        "flat at OpenAlgo, no member listed open",
        "flat everywhere",
        "not read",
        "read late: not confirmed",
        "read late, flat everywhere",
    ],
)
def test_an_instrument_is_listed_for_a_net_position_or_a_flat_net_its_listed_members_differ_from(
    net, held, late, members_open, listed, agrees
):
    shared = SharedPosition("SBIN", "NSE", ("A", "B"), net, held, late=late)
    assert (shared.listed(members_open), shared.agrees) == (listed, agrees)


def test_a_net_carried_by_the_members_fills_since_the_read_says_so():
    shared = SharedPosition("SBIN", "NSE", ("A", "B"), 50, 50, carried=True)
    assert shared.line() == (
        f"SBIN NSE (A and B): OpenAlgo's net 50 by the last read{CARRIED}, their sum 50: they agree"
    )
    assert f"by the last read{CARRIED}: it equals their sum" in shared.note()


def test_a_net_no_read_saw_is_named_as_not_read():
    shared = SharedPosition("SBIN", "NSE", ("A", "B"), None, 150)
    assert shared.line() == "SBIN NSE (A and B): OpenAlgo's net not read, their sum 150"

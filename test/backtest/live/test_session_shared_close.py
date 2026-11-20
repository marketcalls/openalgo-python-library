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

from openalgo.backtest._live._alerts_summary import summary_text
from openalgo.backtest._live._session_named import CARRIED, SharedPosition
from openalgo.backtest._live._types import LOGGER_NAME

HELD = A_UNITS + B_UNITS
#: The instrument's row's name, and the start of its summary line.
BOTH = "SbinTrend and SbinBreakout"
LINE = f"SBIN NSE ({BOTH}): OpenAlgo's net"


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


def test_a_net_openalgo_shows_flat_gives_the_instrument_no_row(world):
    # The position book lags from before the entries: it shows SBIN flat, while each
    # member holds its units on its own row.
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    outcome = kept(world, [SbinTrend, SbinBreakout])
    assert [p.strategy for p in outcome.open_positions] == ["SbinTrend", "SbinBreakout"]


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
    ("net", "held", "is_open", "agrees"),
    [
        (150, 150, True, True),
        (90, 150, True, False),
        (0, 150, False, False),
        (None, 150, False, False),
    ],
    ids=["agreeing", "differing", "flat at OpenAlgo", "not read"],
)
def test_an_instrument_is_open_when_openalgos_net_is_a_position(net, held, is_open, agrees):
    shared = SharedPosition("SBIN", "NSE", ("A", "B"), net, held)
    assert (shared.open, shared.agrees) == (is_open, agrees)


def test_a_net_carried_by_the_members_fills_since_the_read_says_so():
    shared = SharedPosition("SBIN", "NSE", ("A", "B"), 50, 50, carried=True)
    assert shared.line() == (
        f"SBIN NSE (A and B): OpenAlgo's net 50 by the last read{CARRIED}, their sum 50: they agree"
    )
    assert f"by the last read{CARRIED}: it equals their sum" in shared.note()


def test_a_net_no_read_saw_is_named_as_not_read():
    shared = SharedPosition("SBIN", "NSE", ("A", "B"), None, 150)
    assert shared.line() == "SBIN NSE (A and B): OpenAlgo's net not read, their sum 150"

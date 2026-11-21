"""The account's loss limit and Ctrl+C on an instrument two strategies of the session share,
on the fake OpenAlgo (`_live/_kill_switch.py`, `_live/_session_stop.py`,
`_live/_reconcile_guard.py`; 18-multi-strategy 4.8 and 4.9, rows F106 and F108 of 5.2,
unit M7).

The kill switch exits each member once a read shows the broker holding the group's sum on
its side; the safe stop's flatten waits, for each member, for a read begun after the
Ctrl+C that agrees with the group. A second Ctrl+C before that read ends the wait: nothing
is sent, each member keeps its confirmed stop, and the last lines name each member.

The market and the strategies are `shared_samples`': SBIN at 500.00; A (`SbinTrend`) buys
100 at 14:42:01 IST, B (`SbinBreakout`) 50, each with an SL-M stop at 497.00. Expected
values come from the fake's books and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, ist
from shared_samples import SbinBreakout, SbinTrend, sbin_orders
from stops_samples import at_price, stop_world

from openalgo.backtest._live._types import LOGGER_NAME

#: The account's daily loss limit, and the price that takes the members past it: A's 100
#: and B's 50 at 497.40 lose 2.60 a unit, Rs 390 together.
ACCOUNT_LIMIT = 300
FALLEN = 497.4
#: When SBIN falls, and when Ctrl+C is pressed.
FALL_AT = ist(14, 44)
CTRL_C = ist(14, 43, 30)
MEMBERS = ("SbinTrend", "SbinBreakout")


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`, in order."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    return [event for event in lines if event["kind"] == kind]


def exits(world, name: str) -> list:
    return [o for o in sbin_orders(world, name) if o.price_type == "MARKET" and o.action == "SELL"]


# ----- F106: the account's loss on a shared instrument ------------------------------------


def account_loss(world):
    at_price(world, FALL_AT, FALLEN)
    world.stop_at(ist(14, 46))
    return world.run([SbinTrend, SbinBreakout], account_max_daily_loss=ACCOUNT_LIMIT)


def test_F106_the_kill_switch_exits_each_member_once_the_group_agrees(world):
    outcome = account_loss(world)
    assert {e["strategy"] for e in events(outcome, "kill_switch")} == set(MEMBERS)
    assert {e["strategy"]: e["units"] for e in events(outcome, "kill_exit")} == {
        "SbinTrend": 100,
        "SbinBreakout": 50,
    }
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()


def test_F106_the_trades_record_the_account_loss_limit_and_entries_stay_locked(world):
    outcome = account_loss(world)
    for name in MEMBERS:
        assert outcome.outputs[name]["trades"]["exit_reason"] == ["account_loss_limit"]
    assert [e for e in events(outcome, "unlock") if e["reason"] == "account_loss"] == []
    assert outcome.summary.shared == (
        "SBIN NSE (SbinTrend and SbinBreakout): OpenAlgo's net 0 by the last read, their sum "
        "0: they agree",
    )


# ----- F108: Ctrl+C with a shared instrument ----------------------------------------------


def test_F108_the_flatten_waits_for_a_read_agreeing_with_the_group_then_ends_flat(world):
    world.stop_at(CTRL_C)
    outcome = world.run([SbinTrend, SbinBreakout])
    assert outcome.stopped_by == "ctrl_c"
    awaited = {e["strategy"] for e in events(outcome, "exit_awaits_read") if e["ts_ns"] >= CTRL_C}
    assert awaited == set(MEMBERS)
    first_read = min(c.ts_ns for c in world.fake.log.calls("positionbook") if c.ts_ns >= CTRL_C)
    for name in MEMBERS:
        (exit_,) = exits(world, name)
        assert exit_.placed_ns >= first_read
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()


def test_F108_variant_a_second_ctrl_c_before_the_read_leaves_each_member_a_stop(world):
    # The second press comes before the flatten's read agrees: no exit is sent, and each
    # member ends with a confirmed stop at the broker for its own units (protect by state,
    # 16-session-design 3.3 step 4), each named by a line of its own.
    world.stop_at(CTRL_C)
    world.stop_at(CTRL_C + NS // 10)
    outcome = world.run([SbinTrend, SbinBreakout])
    assert outcome.stopped_by == "ctrl_c"
    assert [exits(world, name) for name in MEMBERS] == [[], []]
    assert world.fake.position("SBIN", "NSE") == 150
    rows = {p.strategy: p for p in outcome.open_positions}
    for name, units in zip(MEMBERS, (100, 50), strict=True):
        (stop,) = [
            o
            for o in sbin_orders(world, name)
            if o.price_type == "SL-M" and o.status == "trigger pending"
        ]
        assert stop.quantity == units
        assert (rows[name].stop_order_id, rows[name].stop_confirmed) == (stop.order_id, True)
        assert any(line.startswith(f"{name} SBIN NSE {units} units: stop") for line in world.lines)

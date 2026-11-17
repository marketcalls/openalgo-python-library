"""What journal L keeps of each strategy for a restart (`_live/_session_saved.py`,
`_live/_venue_saved.py`; 16-session-design 2.5 E and F, 3.6, unit 12): the venue map's
record of each working order survives a JSON round trip, and a saved state that cannot be
read is refused, never guessed at. No session, no server.
"""

from __future__ import annotations

import json

import pytest

from openalgo.backtest._live._session_saved import DAY_LOCKS, read_saved
from openalgo.backtest._live._types import OrderIntent, Purpose
from openalgo.backtest._live._venue_requests import OrderTerms
from openalgo.backtest._live._venue_saved import SavedOrder, order_record, read_records
from openalgo.backtest.errors import LiveError

STOP = SavedOrder(
    intent=OrderIntent(
        intent_id="20261116-live-1-StopHold-o2",
        strategy="StopHold",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        price_type="SL-M",
        product="MIS",
        purpose=Purpose.STOP,
        created_ns=1_794_820_323_000_000_000,
        trigger_price=497.0,
    ),
    terms=OrderTerms(10, "SL-M", 0.0, 497.0),
    order_id="261116000000002",
    filled_units=0,
    filled_value=0.0,
    modify=(OrderTerms(10, "SL-M", 0.0, 498.0), OrderTerms(10, "SL-M", 0.0, 497.0)),
    cancel=False,
    deferred=({"command": "cancel", "order_id": 2},),
)


def test_a_saved_order_survives_a_json_round_trip():
    text = json.dumps([order_record(2, STOP)], allow_nan=False)
    assert read_records(json.loads(text)) == {2: STOP}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("order", "2", "order"),
        ("filled_units", 1.5, "filled_units"),
        ("terms", {"quantity": 10}, "o2"),
        ("intent", {"purpose": "hedge"}, "o2"),
    ],
)
def test_a_saved_order_that_cannot_be_read_is_refused(field, value, message):
    record = order_record(2, STOP) | {field: value}
    with pytest.raises(LiveError, match=message):
        read_records([record])


def live_state(units: float, locks: list[str]) -> dict:
    """The parts of a kernel's saved state the restart reads."""
    return {
        "slice": 329,
        "positions": [{"instrument": 0, "product": "MIS", "quantity_units": units}],
        "orders": [{"order": 2, "role": "leg_stop", "sent": True}],
        "desk": {"entry_locks": locks},
    }


def test_the_saved_position_stop_and_day_locks_are_read_back():
    state = {
        "live_state": live_state(10.0, ["reconciliation", "feed_dead", "starting"]),
        "venue_map": [order_record(2, STOP)],
        "adopted": -6,
    }
    saved = read_saved(json.loads(json.dumps(state)), "StopHold")
    assert (saved.units, saved.adopted, saved.sent) == (10, -6, {2: True})
    assert saved.stop == STOP
    assert saved.locks == {"reconciliation"}


def test_only_the_locks_that_last_the_day_are_set_again():
    assert {"reconciliation", "exit_failed", "account_loss"} == DAY_LOCKS


def test_nothing_saved_reads_as_nothing():
    assert read_saved(None, "StopHold") is None


def test_a_saved_state_that_cannot_be_read_refuses_the_restart_naming_the_strategy():
    with pytest.raises(LiveError, match="StopHold"):
        read_saved({"live_state": live_state(10.0, []), "venue_map": "nothing"}, "StopHold")

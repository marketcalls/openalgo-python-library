"""A member of an instrument group through the binding (`Engine.session`;
rust/oa_py/src/backtest/engine/session.rs and venue.rs; 18-multi-strategy 4.13 and 4.14):
the side the other members hold or are entering (`set_group_side`, reject
"group_other_side"), the market orders only of a shared instrument (`set_shared`, reject
"shared_resting"), and a closing order's margin rejection (`apply_margin_reject`), after
which the kernel sends nothing that closes the position by itself until the session asks:
no exit at market, no stop placed again, no exit at later opens.

No server is involved: the test plays the venue. The sessions are `test_kernel_session`'s
(SBIN in MIS, bars flat at 500.00, a buy of 100 filled at 500.00 in minute 1, paused at
the close of minute 1), the same with every position held at most 1 bar, and
`test_kernel_legs`' (a buy of 10, order 1, with a stop 3 points below, order 2, waiting
before minute 1).
"""

from __future__ import annotations

from typing import Any

import pytest
from test_kernel_legs import STOP, holding_with_stop, placed
from test_kernel_session import (
    MINUTE_NS,
    NO_BARS,
    OPEN_NS,
    bar,
    flat,
    holding,
    new_backtest,
    order,
    places,
    release_minute,
    sbin,
    session_config,
)

from openalgo.backtest._calendar_convert import native_attr
from openalgo.backtest._reject_reasons import REASONS
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import VenueRefusalError

RESTING_BUY = order("BUY", 10.0, order_type="LIMIT", price=495.0)


@pytest.mark.parametrize(
    ("side", "request_", "code"),
    [
        ("long", order("SELL", 10.0), "group_other_side"),
        ("long", order("BUY", 10.0), None),
        ("short", order("BUY", 10.0), "group_other_side"),
        (None, order("SELL", 10.0), None),
    ],
    ids=["long: a sale", "long: a buy", "short: a buy", "none: a sale"],
)
def test_an_entry_on_the_other_side_of_the_group_is_rejected(side, request_, code):
    engine = flat()
    assert engine.set_group_side(0, "MIS", side) is None
    outcome, _, reject_code = engine.submit(request_)
    assert (outcome, reject_code) == (("rejected", code) if code else ("submitted", None))


def test_an_exit_is_never_held_by_the_group_side():
    engine = holding(100.0)
    engine.set_group_side(0, "MIS", "short")
    assert engine.submit(order("SELL", 40.0))[0:3:2] == ("submitted", None)


def test_an_unknown_group_side_is_an_input_error():
    with pytest.raises(native_attr("NativeEngineInputError"), match="unknown group side"):
        flat().set_group_side(0, "MIS", "BUY")


def test_a_resting_order_on_a_shared_instrument_is_rejected_and_a_market_order_is_not():
    engine = flat()
    assert engine.set_shared(0, "MIS", True) is None
    assert engine.submit(RESTING_BUY)[0::2] == ("rejected", "shared_resting")
    assert engine.submit(order("BUY", 10.0))[0::2] == ("submitted", None)


def test_an_instrument_no_longer_shared_takes_resting_orders_again():
    engine = flat()
    engine.set_shared(0, "MIS", True)
    engine.set_shared(0, "MIS", False)
    assert engine.submit(RESTING_BUY)[0::2] == ("submitted", None)


@pytest.mark.parametrize(
    "control",
    [
        lambda engine: engine.set_group_side(0, "MIS", "long"),
        lambda engine: engine.set_shared(0, "MIS", True),
    ],
    ids=["set_group_side", "set_shared"],
)
def test_the_group_controls_are_refused_in_a_backtest(control):
    with pytest.raises(native_attr("NativeEngineError"), match="a session's controls") as raised:
        control(new_backtest())
    assert not isinstance(raised.value, native_attr("NativeEngineInputError"))


def test_a_margin_rejection_is_refused_in_a_backtest_as_any_venue_result_is():
    with pytest.raises(VenueRefusalError) as raised:
        new_backtest().apply_margin_reject(1)
    assert raised.value.kind == "not_external"


def test_a_stop_rejected_for_margin_is_neither_exited_at_market_nor_placed_again():
    engine = holding_with_stop()
    slice_index, _, step = engine.apply_margin_reject(STOP)
    assert (slice_index, step) == (1, 3)
    assert engine.take_commands() == []
    assert engine.order(STOP)["status"] == "Rejected"
    assert engine.order(STOP)["reject_code"] == "venue"


def test_protect_places_a_stop_rejected_for_margin_again():
    engine = holding_with_stop()
    engine.apply_margin_reject(STOP)
    assert engine.protect(0, "MIS") == "placed"
    (stop,) = placed(engine)
    assert (stop["role"], stop["trigger_price"], stop["quantity_units"]) == (
        "leg_stop",
        497.0,
        10.0,
    )


def test_an_entry_takes_no_margin_rejection():
    engine = flat()
    outcome, order_id, _ = engine.submit(order("BUY", 10.0))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    with pytest.raises(native_attr("NativeEngineInputError"), match="closing order"):
        engine.apply_margin_reject(order_id)
    assert engine.order(order_id)["status"] == "Submitted"


def released(engine: Any, minute: int) -> str:
    """Hand the waiting session the bar of `minute`, release it and return its next pause."""
    engine.add_bar(0, bar(minute))
    engine.release(minute + 1)
    return engine.advance()[0]


def holding_one_bar_at_most() -> Any:
    """`holding(100.0)` with every position held at most 1 bar: its time exit (order 2) is
    sent as the session reaches the open of minute 2, and the session waits before it."""
    config = session_config()
    config["risk"] = {**config["risk"], "max_hold_bars": 1}
    engine = native_attr("Engine").session(default_calendar()._cal, config, [sbin()], [NO_BARS])
    assert release_minute(engine, 0)[0] == "bar_close"
    outcome, order_id, _ = engine.submit(order("BUY", 100.0))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.apply_fill(order_id, 100.0, 500.0, OPEN_NS + MINUTE_NS)
    engine.take_commands()
    assert released(engine, 1) == "bar_close"
    assert engine.advance()[0] == "waiting"
    assert places(engine) == [(2, "SELL", 100.0, "risk_exit", "time_exit")]
    return engine


@pytest.mark.parametrize(
    ("ending", "sent_again"),
    [
        (
            lambda engine: engine.apply_reject(2, "venue"),
            [(3, "SELL", 100.0, "risk_exit", "time_exit")],
        ),
        (lambda engine: engine.apply_margin_reject(2), []),
    ],
    ids=["rejected: sent again", "rejected for margin: not sent again"],
)
def test_an_engine_exit_rejected_for_margin_is_not_sent_again_at_the_next_open(ending, sent_again):
    engine = holding_one_bar_at_most()
    ending(engine)
    assert engine.take_commands() == []
    assert engine.order(2)["status"] == "Rejected"
    assert released(engine, 2) == "bar_close"
    assert engine.advance()[0] == "waiting"
    assert places(engine) == sent_again


def test_the_group_side_vocabulary_is_the_kernels():
    assert native_attr("GROUP_SIDES") == ("long", "short")


def test_the_group_gates_say_what_happened_and_what_to_do():
    for code in ("group_other_side", "shared_resting"):
        meaning, advice = REASONS[code]
        assert "live session" in meaning and advice

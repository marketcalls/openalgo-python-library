"""A CNC leg on an instrument held for the day (16-session-design 4.9; the lead's ruling
N3b-4: no square-off and no square-off alert for a CNC or NRML leg; review finding 2 of
round 3 of unit N3b): the MIS square-off's review says nothing of it, and the note of the
leg left open names the order that holds the instrument, with no MIS advice (no broker
square-off backstop, no stop to cancel before a sale by hand): the leg is carried.

The market and the strategies are `positional_samples`': `CncThenAdd` buys 10 SBIN CNC at
14:42:01 with its stop GTT (sell 10 at 497.00), then at 14:43:01 rests an add below the
market, a LIMIT buy of 10 at 490.00. Half a second later another client places the same
order under the same strategy name (the sandbox's book names it) and the session's answer
is lost after it reached the broker, so the order book cannot tell the two apart and
nothing more can be sent for SBIN today. The session runs past NSE's MIS square-off
(15:00) to a Ctrl+C at 15:05.
"""

from __future__ import annotations

import logging

from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import MODES, CncHold, calls, gtt_world
from session_samples import NS, alerts, ist

from openalgo.backtest._live._types import LOGGER_NAME

#: The add's LIMIT price, below every price of the market.
ADD_LIMIT = 490.0


class CncThenAdd(CncHold):
    """CncHold that rests a LIMIT add of 10 at `ADD_LIMIT` at its second decision."""

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(self.qty, **type(self).rules)
        elif self.calls == 2:
            self.buy(self.qty, price=ADD_LIMIT)


@MODES
def test_a_cnc_leg_held_for_the_day_has_no_square_off_alert_and_no_mis_advice(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(15, 32)
    )
    world.trace.waive("protected", "the add cannot be told apart in the order book")
    world.trace.waive("books agree", "the add cannot be told apart in the order book")
    other = world.fake.client()

    def the_same_add_by_another_client_then_lost():
        other.placeorder(
            strategy="CncThenAdd",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="LIMIT",
            product="CNC",
            quantity=10,
            price=ADD_LIMIT,
        )
        world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)

    world.script.at(ist(14, 43, 1) + NS // 2, the_same_add_by_another_client_then_lost)
    world.stop_at(ist(15, 5))
    world.stop_at(ist(15, 6))

    outcome = world.run(CncThenAdd)

    assert not [a for a in alerts(caplog, "error") if "no square-off is sent" in a]
    (position,) = outcome.open_positions
    held = "SBIN NSE is held for the day by CncThenAdd's add order"
    assert held in position.note
    assert "the position is carried, never squared off" in position.note
    assert "MIS square-off" not in position.note
    assert "cancel its stop" not in position.note
    assert calls(world, "cancelgttorder", "cancelorder") == []

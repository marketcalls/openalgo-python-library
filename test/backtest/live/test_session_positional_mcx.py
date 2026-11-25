"""An MCX NRML leg carried past the close of a live session, on the fake OpenAlgo
(20-positional-live 7.3; the lead's ruling N3b-4): the session ends at MCX's own close
(23:55 IST on Monday 2026-11-16, by the calendar's dated rows), not NSE's, once a read of
the GTT book after it has confirmed the leg's stop GTT, and the leg is carried.

The instrument is `test_session_mcx_charges`' GOLDM05DEC26FUT (100 units a lot, tick Re 1)
at 1,20,000.00 all day; the session starts at 23:40:05 IST and the strategy buys one lot
NRML at its first decision with a stop 500 points below. GOLDM trades 100 units every 10 s
until 23:57.
"""

from __future__ import annotations

import functools

from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import DAY, HISTORY_DAYS, MINUTE_NS
from positional_samples import calls, gtts
from session_samples import NS, TRADE_EVERY_S, ist, rig

import openalgo.backtest as bt
from openalgo.backtest.calendar import default_calendar

GOLDM = "GOLDM05DEC26FUT"
LOT_UNITS = 100
PRICE = 120_000.0
START = (23, 40, 5)
#: MCX's close on the module's day, UTC nanoseconds (the calendar's dated rows).
MCX_CLOSE = ist(23, 55)


class GoldHold(bt.Strategy):
    """Buys one lot of GOLDM NRML at its first decision with a stop 500 points below."""

    symbol, exchange, interval, product = GOLDM, "MCX", "1m", "NRML"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(LOT_UNITS, stop=bt.points(500))


def flat_candles() -> list[tuple[int, float, float, float, float, float]]:
    """1-minute candles at `PRICE` over the MCX sessions of the two days before the live
    day and of the live day up to the session's start."""
    calendar = default_calendar()
    rows = []
    for day in (*HISTORY_DAYS, DAY):
        opens, closes = calendar.session("MCX", day)
        end = ist(*START[:2]) if day == DAY else closes
        rows += [(m, PRICE, PRICE, PRICE, PRICE, 1_000.0) for m in range(opens, end, MINUTE_NS)]
    return rows


def test_an_mcx_nrml_leg_is_carried_and_the_session_ends_at_the_mcx_close(
    make_fake_openalgo, monkeypatch, tmp_path
):
    build = functools.partial(make_fake_openalgo, gtt_history=True)
    world = rig(build, monkeypatch, tmp_path, start=START, trades_until=None)
    fake = world.fake
    fake.add_instrument(
        GOLDM,
        "MCX",
        lot_size=LOT_UNITS,
        tick_size=1.0,
        freeze_qty=10_000,
        instrument_type="FUT",
        expiry="05-DEC-26",
    )
    fake.seed_candles(GOLDM, "MCX", flat_candles())
    fake.trade(GOLDM, "MCX", PRICE, volume=LOT_UNITS)
    for instant in range(ist(23, 40, 15), ist(23, 57), TRADE_EVERY_S * NS):
        world.script.at(instant, lambda: fake.trade(GOLDM, "MCX", PRICE, volume=LOT_UNITS))

    outcome = world.run(GoldHold, capital=250_000)

    assert outcome.stopped_by == "session_end"
    assert fake.position(GOLDM, "MCX") == LOT_UNITS
    assert [(status, units) for status, units, *_ in gtts(world)] == [("active", LOT_UNITS)]
    after_close = [c.ts_ns for c in calls(world, "gttorderbook") if c.ts_ns >= MCX_CLOSE]
    assert after_close
    assert world.fake.clock.now() < MCX_CLOSE + 60 * NS

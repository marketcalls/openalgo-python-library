"""An MCX NRML leg carried past the close of a live session, on the fake OpenAlgo
(20-positional-live 7.3; the lead's ruling N3b-4): the session ends at MCX's own close
(23:55 IST on Monday 2026-11-16, by the calendar's dated rows), not NSE's, once a read of
the GTT book after it has confirmed the leg's stop GTT, and the leg is carried.

The instrument is `test_session_mcx_charges`' GOLDM05DEC26FUT (100 units a lot, tick Re 1)
at 1,20,000.00 all day; the session starts at 23:40:05 IST and the strategy buys one lot
NRML at its first decision with a stop 500 points below. GOLDM trades 100 units every 10 s
until 23:57.

Beside an NSE MIS strategy (`MisHold`, SBIN from 14:40:05 IST), the latest close is still
MCX's: the MIS leg is squared off at 15:00 and the session runs on to 23:55 with the NRML
leg carried (review finding 6 d of unit N3b).
"""

from __future__ import annotations

import functools

from fake_openalgo.fake import LIVE_CASH
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import DAY, HISTORY_DAYS, MINUTE_NS
from positional_samples import MisHold, calls, gtts
from session_samples import NS, TRADE_EVERY_S, ist, rig

import openalgo.backtest as bt
from openalgo.backtest.calendar import default_calendar

GOLDM = "GOLDM05DEC26FUT"
LOT_UNITS = 100
PRICE = 120_000.0
START = (23, 40, 5)
#: The capital of each strategy, rupees: the GOLDM lot's margin fits in the first.
GOLD_CAPITAL, MIS_CAPITAL = 250_000, 20_000
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


def flat_candles(
    start: tuple[int, ...] = START,
) -> list[tuple[int, float, float, float, float, float]]:
    """1-minute candles at `PRICE` over the MCX sessions of the two days before the live
    day and of the live day up to the session's `start` (IST hour, minute, second)."""
    calendar = default_calendar()
    rows = []
    for day in (*HISTORY_DAYS, DAY):
        opens, closes = calendar.session("MCX", day)
        end = ist(*start[:2]) if day == DAY else closes
        rows += [(m, PRICE, PRICE, PRICE, PRICE, 1_000.0) for m in range(opens, end, MINUTE_NS)]
    return rows


def gold_world(make_fake_openalgo, monkeypatch, tmp_path, start, *, live_cash=LIVE_CASH, **options):
    """A rig started at `start` (IST) whose GOLDM trades 100 units every 10 s at `PRICE`
    from 10 s after it until 23:57, its broker account holding `live_cash` rupees."""
    build = functools.partial(make_fake_openalgo, gtt_history=True, live_cash=live_cash)
    world = rig(build, monkeypatch, tmp_path, start=start, **options)
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
    fake.seed_candles(GOLDM, "MCX", flat_candles(start))
    fake.trade(GOLDM, "MCX", PRICE, volume=LOT_UNITS)
    for instant in range(
        ist(*start[:2], start[2] + TRADE_EVERY_S), ist(23, 57), TRADE_EVERY_S * NS
    ):
        world.script.at(instant, lambda: fake.trade(GOLDM, "MCX", PRICE, volume=LOT_UNITS))
    return world


def test_an_mcx_nrml_leg_is_carried_and_the_session_ends_at_the_mcx_close(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gold_world(make_fake_openalgo, monkeypatch, tmp_path, START, trades_until=None)
    fake = world.fake

    outcome = world.run(GoldHold, capital=GOLD_CAPITAL)

    assert outcome.stopped_by == "session_end"
    assert fake.position(GOLDM, "MCX") == LOT_UNITS
    assert [(status, units) for status, units, *_ in gtts(world)] == [("active", LOT_UNITS)]
    after_close = [c.ts_ns for c in calls(world, "gttorderbook") if c.ts_ns >= MCX_CLOSE]
    assert after_close
    assert world.fake.clock.now() < MCX_CLOSE + 60 * NS


def test_an_nse_mis_leg_is_squared_off_and_an_mcx_nrml_leg_carried_to_the_mcx_close(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The latest close among the session's exchanges is MCX's: the NSE MIS leg is squared
    # off at 15:00, and the session runs on until MCX closes with the NRML leg carried.
    world = gold_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        (14, 40, 5),
        live_cash=LIVE_CASH + MIS_CAPITAL,
        trades_until=(15, 32),
    )
    fake = world.fake

    capital = {"MisHold": MIS_CAPITAL, "GoldHold": GOLD_CAPITAL}
    outcome = world.run([MisHold, GoldHold], capital=capital)

    assert outcome.stopped_by == "session_end"
    assert fake.position("SBIN", "NSE") == 0
    assert fake.position(GOLDM, "MCX") == LOT_UNITS
    square_off = [o for o in world.orders() if o.symbol == "SBIN" and o.action == "SELL"]
    assert [(o.price_type, o.quantity, o.status) for o in square_off][-1:] == [
        ("MARKET", 10, "complete")
    ]
    assert [(status, units) for status, units, *_ in gtts(world)] == [("active", LOT_UNITS)]
    assert [c for c in calls(world, "gttorderbook") if c.ts_ns >= MCX_CLOSE]
    assert MCX_CLOSE <= world.fake.clock.now() < MCX_CLOSE + 60 * NS

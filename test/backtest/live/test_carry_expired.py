"""A carried position on a contract whose last trading day has passed (20-positional-live 9
and 4.2 step 4; the lead's rulings N5a-1 and N5a-3): its expiry is resolved before any lookup
in today's master contract, which no longer lists it, so the preflight never refuses the
start for it; the restart rules take its journal position as none (the exchange settled
it); and the restore's words name it.

The contract is November's NIFTY future, `NIFTY23NOV26FUT` (65 units a lot, its last trading
day Monday 2026-11-23: the month's last Tuesday, 2026-11-24, is a holiday), carried by
`IdxKept` from Monday 2026-11-16 with 65 units, its last close of the day 120.00; today is
Wednesday 2026-11-25. Expected values are worked by hand from these figures.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from preflight_fakes import NOW_NS, FakeOpenAlgo, make_plan

from openalgo.backtest._live._carry_restore import expired_words
from openalgo.backtest._live._carry_start import Carried, CarryAtStart, preflight_terms
from openalgo.backtest._live._journal_carry import DayClose
from openalgo.backtest._live._preflight_plan import InstrumentFacts
from openalgo.backtest._live._restart_carried import carried_units
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.preflight import check_preflight
from openalgo.backtest.reference import Instrument

CONTRACT = "NIFTY23NOV26FUT"
MONDAY, LAST_DAY, TODAY = dt.date(2026, 11, 16), dt.date(2026, 11, 23), dt.date(2026, 11, 25)
NOVEMBER = Instrument(
    symbol=CONTRACT,
    exchange="NFO",
    kind="future",
    lot_size=65,
    tick_size=0.05,
    freeze_qty=1800,
    expiry=LAST_DAY,
    underlying="NIFTY",
    settlement="cash",
    last_trading_day=LAST_DAY,
)
#: IdxKept's day-close record of Monday: 65 units, the contract's last close 120.00.
CARRIED = {
    "instruments": [{"symbol": CONTRACT, "exchange": "NFO"}],
    "positions": [
        {"instrument": 0, "product": "NRML", "quantity_units": 65.0, "margin_rupees": 1560.0}
    ],
    "orders": [],
    "gtts": [],
    "last_closes": [{"instrument": 0, "close_rupees": 120.0}],
    "account": {"capital_rupees": 100000.0},
}
RECORD = DayClose("IdxKept", MONDAY, Mode.LIVE, CARRIED, {}, "v1", 0)
CARRY = CarryAtStart(TODAY, Mode.LIVE, {"IdxKept": Carried(RECORD, MONDAY)}, {})
SPEC = SimpleNamespace(
    name="IdxKept",
    symbol=CONTRACT,
    exchange="NFO",
    product="NRML",
    instruments=(NOVEMBER,),
    capital=100000.0,
)


def test_an_expired_carried_contract_is_not_looked_up_in_todays_master_contract():
    terms = preflight_terms(CARRY, SimpleNamespace(strategies=(SPEC,)))
    # Settled by the exchange: it holds none of the account's cash any more.
    assert terms["carried"] == ()
    assert terms["expired"] == (InstrumentFacts(CONTRACT, "NFO", 65, 0.05, 1800),)
    client = FakeOpenAlgo(symbols={})
    plan = make_plan(instruments=[("SBIN", "NSE"), (CONTRACT, "NFO")], **terms)

    outcome = check_preflight(client, plan, now=lambda: NOW_NS)

    assert ("symbol", CONTRACT, "NFO") not in client.calls
    assert outcome.check(f"symbol:NFO:{CONTRACT}").ok


def test_the_restart_rules_take_an_expired_carried_position_as_none():
    assert carried_units(CARRY, SPEC) == 0


def test_the_restores_words_name_the_contract_its_last_close_and_the_settlement():
    trader = SimpleNamespace(name="IdxKept", part=SimpleNamespace(spec=SPEC))

    words = expired_words(trader, CARRY.carried["IdxKept"], 0)

    assert words == (
        f"IdxKept: 65 {CONTRACT} NFO carried from {MONDAY} expired on {LAST_DAY}: the exchange "
        "settled it, and the engine records it exited at its last close, 120.00; the session "
        "does not trade it"
    )

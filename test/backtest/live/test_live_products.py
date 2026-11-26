"""The product a live strategy trades, carried through each part of a session that holds it
(20-positional-live 7.1, review H6; the lead's ruling N3a-4, the remaining lows of unit N2's
verification): a strategy's spec trades its plan's product, its `Trader` and its venue map
the spec's, the preflight plan checks the strategies' product, and the close names an MIS
and a CNC group on one symbol apart. Each is tested with a product other than MIS, which
every session test trades.

The session's parts are in-process doubles holding what each part reads. Nothing contacts a
server.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live import _session_start
from openalgo.backtest._live._carry_start import CarryAtStart
from openalgo.backtest._live._group_rules import GroupFigure
from openalgo.backtest._live._journal_reads import JournalSession
from openalgo.backtest._live._locked_entries import LockedEntries
from openalgo.backtest._live._plan_strategies import StrategySpec
from openalgo.backtest._live._reconcile_findings import Sighting
from openalgo.backtest._live._session_named import named_legs
from openalgo.backtest._live._session_start import StartSettings, _preflight
from openalgo.backtest._live._session_traders import build_traders
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.plan import SessionSpec
from openalgo.backtest._products import CNC, MIS, NRML
from openalgo.backtest.reference import Instrument

DAY = dt.date(2026, 11, 16)
#: 11:30 IST on DAY, UTC nanoseconds.
NOW_NS = 1_794_808_800 * 10**9
PRODUCTS = pytest.mark.parametrize("product", [MIS, CNC, NRML])


def a_spec(product: str) -> StrategySpec:
    """A strategy's spec whose plan resolved `product`; nothing else of it is read."""
    return StrategySpec(
        strategy=object,  # type: ignore[arg-type]
        name="SbinHold",
        symbol="SBIN",
        exchange="NSE",
        interval="5m",
        capital=100_000.0,
        max_daily_loss=5_000.0,
        plan=SimpleNamespace(product=product, square_off=None),  # type: ignore[arg-type]
        arguments={},
        instruments=(),
        needs=(),
        measured={},
        version=SimpleNamespace(version_id="0a1b2c3d4e5f"),  # type: ignore[arg-type]
    )


@PRODUCTS
def test_a_strategy_spec_trades_the_product_its_plan_resolved(product):
    assert a_spec(product).product == product


# ----- the traders and their venue maps -------------------------------------------------------


def traders_of(product: str) -> tuple[Any, ...]:
    """The session's traders (`build_traders`) of one strategy trading SBIN in `product`."""
    engine = SimpleNamespace(cap_sizing_equity=lambda rupees: None)
    part = SimpleNamespace(
        engine=engine,
        spec=a_spec(product),
        setup=SimpleNamespace(instruments=[Instrument.cash_equity("SBIN", "NSE")]),
        slices=[],
        strategy=SimpleNamespace(state={}),
    )
    parts = SimpleNamespace(
        plan=SimpleNamespace(strategies=[part]),
        spec=SimpleNamespace(day=DAY),
        journal_session=JournalSession(DAY, Mode.LIVE, "zerodha", NOW_NS, NOW_NS, 1),
        alerts=SimpleNamespace(send=lambda kind, text: None),
        logs=SimpleNamespace(decision=lambda *args, **fields: None),
        worker=SimpleNamespace(locked_entries=LockedEntries()),
    )
    session = SimpleNamespace(
        parts=parts,
        now=lambda: NOW_NS,
        event=lambda *args, **fields: None,
        warn=lambda text: None,
    )
    return build_traders(session, 100_000.0)


#: The kernel's placement of a CNC entry: buy 10 SBIN at the market (the binding's dict).
CNC_ENTRY = {
    "command": "place",
    "order_id": 1,
    "instrument": 0,
    "product": CNC,
    "side": "BUY",
    "role": "entry",
    "quantity_units": 10.0,
    "order_type": "MARKET",
    "price": None,
    "trigger_price": None,
}


def test_a_trader_trades_the_product_of_its_strategys_spec():
    (trader,) = traders_of(CNC)

    assert trader.product == CNC


def test_a_cnc_traders_venue_map_sends_its_placement_in_cnc():
    (trader,) = traders_of(CNC)

    (request,) = trader.venue.requests([CNC_ENTRY])

    assert (request.intent.product, request.intent.purpose.value) == (CNC, "entry")


# ----- the preflight plan --------------------------------------------------------------------


@PRODUCTS
def test_the_preflight_plan_checks_the_product_of_the_sessions_strategies(monkeypatch, product):
    checked = []

    def check(client, plan, *, now):
        checked.append(plan.product)
        return SimpleNamespace(waits_for_login=False, report=lambda: "checked")

    monkeypatch.setattr(_session_start, "check_preflight", check)
    strategy = SimpleNamespace(
        name="Sample",
        symbol="SBIN",
        exchange="NSE",
        interval="5m",
        product=product,
        capital=100_000.0,
        max_daily_loss=5_000.0,
        instruments=(),
        stop_gtt=True,
    )
    spec = SessionSpec(SimpleNamespace(account_max_daily_loss=None), DAY, (strategy,))
    settings = StartSettings(
        now=lambda: NOW_NS,
        sleep=lambda seconds: None,
        threads=False,
        emit=lambda line: None,
        stop=None,  # type: ignore[arg-type]
        wait_max_s=1.0,
    )

    opened = SimpleNamespace(clients=SimpleNamespace(data=None), mode=Mode.LIVE, shared=None)

    _preflight(spec, opened, settings, CarryAtStart.none(DAY, Mode.LIVE))

    assert checked == [product]


# ----- the close's named legs -------------------------------------------------------------------


class Member:
    """A strategy of a group on SBIN NSE in `product`, holding `units`."""

    symbol, exchange = "SBIN", "NSE"

    def __init__(self, name: str, product: str, units: int) -> None:
        self.name, self.product, self._units = name, product, units

    def open_units(self) -> float:
        return float(self._units)


def test_the_close_names_an_mis_and_a_cnc_group_on_one_symbol_apart():
    day_a, day_b = Member("DayA", MIS, 10), Member("DayB", MIS, 10)
    hold_a, hold_b = Member("HoldA", CNC, 15), Member("HoldB", CNC, 15)
    # Each group's last comparison was late and saw its net differ from its members' sum.
    late = {
        "DayA": Sighting(10, 15, GroupFigure(("DayA", "DayB"), 20)),
        "DayB": Sighting(10, 15, GroupFigure(("DayA", "DayB"), 20)),
        "HoldA": Sighting(15, 25, GroupFigure(("HoldA", "HoldB"), 30)),
        "HoldB": Sighting(15, 25, GroupFigure(("HoldA", "HoldB"), 30)),
    }
    reconcile = SimpleNamespace(
        late=late.get,
        late_counted=lambda name: False,
        unconfirmed=lambda name: None,
        unjudged=lambda name: False,
    )
    session = SimpleNamespace(traders=(day_a, day_b, hold_a, hold_b), reconcile=reconcile)

    named = named_legs(session)

    assert len(named.differs) == 2
    assert "15 units of SBIN for DayA and DayB together against their 20" in named.differs[0]
    assert "25 units of SBIN for HoldA and HoldB together against their 30" in named.differs[1]

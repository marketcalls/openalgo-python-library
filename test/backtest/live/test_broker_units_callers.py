"""Every reader of the broker's units of a leg reads them through `_broker_units` with the
leg's product (20-positional-live 7.1; the lead's ruling N2-1): the kill switch, the held
cancels, the reconciler and its exit guard, a shared instrument's group comparison and the
restart. Each test runs an MIS and a CNC strategy on SBIN, which are two groups.

The account's books are the same in every test: SBIN in MIS 10 units, CNC 7 bought today
and 40 held from earlier days, so the CNC leg's units are 47 (worked by hand: 40 + 7) and
the MIS leg's 10. A caller that read the MIS line for the CNC leg would see 10, and one
that left the holdings out 7. The sessions here are in-process doubles of the parts each
caller reads; nothing contacts a server.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live._groups import Groups
from openalgo.backtest._live._kill_switch import ACCOUNT_LOSS, KillSwitch
from openalgo.backtest._live._margin_reject import MarginHolds
from openalgo.backtest._live._order_state import HeldFill
from openalgo.backtest._live._restart_books import StartBooks, judge_start
from openalgo.backtest._live._stops_held import HELD_CANCEL_DECIDE_AFTER_S, HeldWords
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live._worker_queue import ReadBooks, ReleaseCancel
from openalgo.backtest._live.inbox import Books, PositionRows
from openalgo.backtest._live.reconcile import Reconciler
from openalgo.backtest.errors import LiveError

NS = 10**9
#: 14:44:01 IST on Monday 2026-11-16, UTC nanoseconds: when each test begins.
START_NS = 1_794_820_441 * NS
MIS_UNITS, CNC_UNITS = 10, 47

POSITIONS = {
    "status": "success",
    "data": [
        {"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": MIS_UNITS},
        {"symbol": "SBIN", "exchange": "NSE", "product": "CNC", "quantity": 7},
    ],
}
HOLDINGS = {
    "status": "success",
    "data": {"holdings": [{"symbol": "SBIN", "exchange": "NSE", "quantity": 40}]},
}
NO_ORDERS = {"status": "success", "data": {"orders": []}}
NO_TRADES = {"status": "success", "data": []}
#: Funds that cannot be read: the reconciler then caps no strategy's sizing.
NO_FUNDS = {"status": "error", "message": "funds not read in these tests"}


class Leg:
    """A strategy holding `units` of SBIN NSE in `product` in its kernel, with no working
    order (its kernel position read in its product only); `exits` the exits it was asked
    for. A stop of `stop_units` units is kernel order 2, broker order "S1"."""

    symbol, exchange, finished, square_off_ns = "SBIN", "NSE", False, 0

    def __init__(self, name: str, product: str, units: int) -> None:
        self.name, self.product, self.units = name, product, units
        self.exits: list[str] = []
        self.engine = self
        self.venue = SimpleNamespace(
            bound=lambda: {},
            changing=lambda kernel_order: False,
            kernel_order_of=lambda order_id: 2 if order_id == "S1" else None,
            order_id_of=lambda kernel_order: "S1" if kernel_order == 2 else "",
        )

    def position(self, instrument: int, product: str) -> dict[str, Any]:
        assert product == self.product, "the kernel's position is read in the leg's product"
        return {"quantity_units": float(self.units), "tranches": [(1, self.units, 500.0, 0)]}

    def order(self, kernel_order: int) -> dict[str, Any]:
        """The leg's stop: as many units as the leg holds, none filled."""
        units = float(self.units)
        return {"quantity_units": units, "filled_units": 0, "trigger_rupees": 497.0}

    def open_units(self) -> float:
        return float(self.units)

    def working(self) -> list[int]:
        return []

    def entry_orders(self) -> list[int]:
        return []

    def lock(self, reason: str) -> bool:
        return True

    def request_exit(self, reason: str) -> str:
        self.exits.append(reason)
        return "sent"


class Loop:
    """The parts of a session loop the readers call: its clock, its strategies and their
    groups, what it hands the order worker, its events and alerts."""

    trading, locked = True, False

    def __init__(self, *legs: Leg) -> None:
        self.traders = legs
        self.clock = START_NS
        self.groups = Groups(legs, event=self.event)
        self.handed: list[Any] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.alerts: list[str] = []
        self.mismatches = 0
        self.last_tick: dict[Any, Any] = {}
        self.parts = SimpleNamespace(
            spec=SimpleNamespace(arguments=SimpleNamespace(on_stop="keep")),
            worker=SimpleNamespace(silent_since_ns=None),
            mode=Mode.LIVE,
        )
        self.margin = MarginHolds(self)
        idle = SimpleNamespace(stop=None, unbound_add_units=0, orders={})
        self.stops = SimpleNamespace(
            leg=lambda trader: idle, held=SimpleNamespace(holds=lambda trader: False)
        )
        self.positions_read: list[PositionRows] = []
        self.risk = SimpleNamespace(positions=self.positions_read.append)

    def now(self) -> int:
        return self.clock

    def trader_named(self, name: str) -> Leg:
        return next(leg for leg in self.traders if leg.name == name)

    def hand_over(self, requests: list[Any]) -> None:
        self.handed += requests

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))

    def alert(self, kind: Any, text: str) -> None:
        self.alerts.append(text)


def mis_and_cnc() -> tuple[Leg, Leg]:
    return Leg("SbinDay", "MIS", MIS_UNITS), Leg("SbinHold", "CNC", CNC_UNITS)


# ----- the kill switch ---------------------------------------------------------------------


def test_the_kill_switch_exits_each_strategy_on_its_own_products_units():
    day, hold = mis_and_cnc()
    loop = Loop(day, hold)
    kill = KillSwitch(loop)
    for leg in (day, hold):
        kill.fire(leg, ACCOUNT_LOSS)
    kill.duties(loop.clock)
    loop.clock += NS

    kill.positions(PositionRows(POSITIONS, loop.clock, holdings=HOLDINGS))
    kill.duties(loop.clock)

    assert (day.exits, hold.exits) == ([ACCOUNT_LOSS], [ACCOUNT_LOSS])


def test_the_kill_switch_exits_no_cnc_leg_whose_holdings_were_not_read():
    day, hold = mis_and_cnc()
    loop = Loop(day, hold)
    kill = KillSwitch(loop)
    kill.fire(hold, ACCOUNT_LOSS)
    kill.duties(loop.clock)
    loop.clock += NS

    kill.positions(PositionRows(POSITIONS, loop.clock))
    kill.duties(loop.clock)

    assert hold.exits == []


# ----- the held cancels ----------------------------------------------------------------------


def test_a_held_cancel_of_a_cnc_leg_is_decided_on_its_holdings_and_cnc_line():
    day, hold = mis_and_cnc()
    loop = Loop(hold, day)
    held = HeldWords(loop)
    held.update((HeldFill("S1", 0, START_NS),))
    decide_ns = START_NS + round(HELD_CANCEL_DECIDE_AFTER_S * NS)
    held.due(decide_ns)

    held.positions(PositionRows(POSITIONS, decide_ns, holdings=HOLDINGS))

    # 47 at the broker, 47 in the kernel: the stop filled nothing before its cancel.
    (release,) = [r for r in loop.handed if isinstance(r, ReleaseCancel)]
    assert (release.order_id, release.filled_quantity) == ("S1", 0)


# ----- the reconciler and its exit guard -----------------------------------------------------


def read(loop: Loop, reconciler: Reconciler, positions: dict[str, Any]) -> None:
    """Ask for a read of every leg now and answer it with `positions` and `HOLDINGS`."""
    reconciler.ask_of(None)
    (request,) = [r for r in loop.handed if isinstance(r, ReadBooks)]
    loop.clock += NS
    books = Books(
        request.generation,
        positions,
        NO_ORDERS,
        NO_TRADES,
        NO_FUNDS,
        loop.clock,
        loop.clock,
        holdings=HOLDINGS,
    )
    reconciler.books(books)


def compared(loop: Loop) -> dict[str, tuple[str, int | None]]:
    """What each comparison found, by strategy: the kind and the broker's units."""
    return {
        fields["strategy"]: (fields["found"], fields["broker_units"])
        for kind, fields in loop.events
        if kind == "reconcile"
    }


def reconciling(*legs: Leg) -> tuple[Loop, Reconciler]:
    loop = Loop(*legs)
    loop.reconcile = Reconciler(loop)
    return loop, loop.reconcile


def test_the_reconciler_compares_each_leg_with_its_products_units():
    loop, reconciler = reconciling(*mis_and_cnc())

    read(loop, reconciler, POSITIONS)

    assert compared(loop) == {"SbinDay": ("agree", MIS_UNITS), "SbinHold": ("agree", CNC_UNITS)}


def test_the_reconcilers_read_counts_for_the_kill_switch_with_its_holdings():
    loop, reconciler = reconciling(*mis_and_cnc())

    read(loop, reconciler, POSITIONS)

    (rows,) = loop.positions_read
    assert rows.holdings is HOLDINGS


def test_the_exit_guard_holds_the_cnc_leg_its_products_units_fall_short_of():
    day, hold = mis_and_cnc()
    hold.units = 60
    loop, reconciler = reconciling(day, hold)

    read(loop, reconciler, POSITIONS)

    # The broker holds 47 of the CNC leg's 60, and no trade explains the 13.
    assert compared(loop)["SbinHold"] == ("unexplained", CNC_UNITS)
    assert (reconciler.holds(day), reconciler.holds(hold)) == (False, True)


def test_a_shared_instrument_is_compared_on_its_own_products_units():
    first, second = Leg("HoldA", "CNC", 30), Leg("HoldB", "CNC", 17)
    loop, reconciler = reconciling(first, second, Leg("SbinDay", "MIS", MIS_UNITS))

    read(loop, reconciler, POSITIONS)

    # HoldA and HoldB hold 30 + 17 = 47 together, the CNC units at the broker.
    assert compared(loop) == {
        "HoldA": ("agree", CNC_UNITS),
        "HoldB": ("agree", CNC_UNITS),
        "SbinDay": ("agree", MIS_UNITS),
    }


# ----- the restart -------------------------------------------------------------------------


class Journal:
    """A journal holding nothing of the day: no order, no saved state."""

    def bound_orders(self, *, day: Any) -> list[Any]:
        return []

    def rms_state(self, strategy: str, *, day: Any) -> None:
        return None


def start_of(*legs: Leg) -> tuple[Any, StartBooks]:
    strategies = tuple(
        SimpleNamespace(
            name=leg.name, symbol=leg.symbol, exchange=leg.exchange, product=leg.product
        )
        for leg in legs
    )
    spec = SimpleNamespace(day=None, strategies=strategies)
    books = Books(0, POSITIONS, NO_ORDERS, NO_TRADES, NO_FUNDS, 0, 0, holdings=HOLDINGS)
    return spec, StartBooks(books)


def test_a_start_judges_each_strategy_against_its_own_products_units():
    spec, start = start_of(*mis_and_cnc())

    with pytest.raises(LiveError) as refused:
        judge_start(spec, Journal(), start, SimpleNamespace(send=lambda kind, text: None))

    # Neither has a journal position; each is refused for its own product's units.
    assert "show 10 units of SBIN NSE, which SbinDay trades" in str(refused.value)
    assert "show 47 units of SBIN NSE, which SbinHold trades" in str(refused.value)

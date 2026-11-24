"""Every reader of the broker's units of a leg reads them through `_broker_units` with the
leg's product (20-positional-live 7.1; the lead's ruling N2-1): the kill switch, the held
cancels, the reconciler and its exit guard, a shared instrument's group comparison and the
restart. Each test runs an MIS and a CNC strategy on SBIN, which are two groups. Every
kernel call a leg's product goes into is made in the leg's product (the table of
`test_every_kernel_call_of_a_cnc_leg_is_made_in_cnc`).

The account's books are the same in every test: SBIN in MIS 10 units, CNC 7 bought today
and 40 held from earlier days, so the CNC leg's units are 47 (worked by hand: 40 + 7) and
the MIS leg's 10. A caller that read the MIS line for the CNC leg would see 10, and one
that left the holdings out 7. A strategy here is the session's own `TraderOrders` (its
position, exits) on a kernel double that holds the leg in its product only and fails a
call in any other; the rest of each session is an in-process double of the parts each
caller reads. Nothing contacts a server.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live._groups import Groups
from openalgo.backtest._live._kill_switch import ACCOUNT_LOSS, KillSwitch
from openalgo.backtest._live._margin_reject import MarginHolds
from openalgo.backtest._live._order_state import HeldFill
from openalgo.backtest._live._reconcile_act import Corrections
from openalgo.backtest._live._reconcile_rules import Kind, leg_books
from openalgo.backtest._live._restart_books import StartBooks, judge_start, read_start
from openalgo.backtest._live._session_orders import TraderOrders
from openalgo.backtest._live._stops_held import HELD_CANCEL_DECIDE_AFTER_S, HeldWords
from openalgo.backtest._live._stops_kernel import protect
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live._worker_queue import ReadBooks, ReleaseCancel
from openalgo.backtest._live.inbox import Books, Envelope, PositionRows
from openalgo.backtest._live.reconcile import Reconciler
from openalgo.backtest.errors import LiveError

NS = 10**9
#: 14:44:01 IST on Monday 2026-11-16, UTC nanoseconds: when each test begins.
START_NS = 1_794_820_441 * NS
MIS_UNITS, CNC_UNITS = 10, 47
#: A price of another client's SBIN trade, rupees.
OTHER_PRICE = 498.0

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


class Kernel:
    """The kernel session of a leg holding `units` of SBIN NSE in `product`: one tranche
    opened at `opened_ns` (`START_NS` by default), no working order, and a stop (kernel order
    2) of as many units with none filled. Every call that names a product is recorded
    (`calls`, as (call, product)) and fails the test unless it names the leg's product, the
    only one the kernel holds the leg in; `exits` are the reasons of the exits asked for."""

    def __init__(self, product: str, units: int, opened_ns: int = START_NS) -> None:
        self.product, self.units, self.opened_ns = product, units, opened_ns
        self.calls: list[tuple[str, str]] = []
        self.exits: list[str] = []

    def _named(self, call: str, product: str) -> None:
        self.calls.append((call, product))
        assert product == self.product, f"the kernel's {call} is called in the leg's product"

    def position(self, instrument: int, product: str) -> dict[str, Any]:
        self._named("position", product)
        units = float(self.units)
        return {"quantity_units": units, "tranches": [(1, units, 500.0, self.opened_ns)]}

    def order(self, kernel_order: int) -> dict[str, Any]:
        """The leg's stop: as many units as the leg holds, none filled."""
        units = float(self.units)
        return {"quantity_units": units, "filled_units": 0, "trigger_rupees": 497.0}

    def working_orders(self) -> list[int]:
        return []

    def request_exit(self, instrument: int, product: str, reason: str, *limit: float) -> str:
        self._named("request_exit", product)
        self.exits.append(reason)
        return "sent"

    def protect(self, instrument: int, product: str) -> str:
        self._named("protect", product)
        return "placed"

    def apply_external_fill(
        self, instrument: int, product: str, side: str, units: float, *rest: Any
    ) -> None:
        self._named("apply_external_fill", product)
        self.units += round(units) if side == "BUY" else -round(units)

    def set_shared(self, instrument: int, product: str, shared: bool) -> None:
        self._named("set_shared", product)

    def set_group_side(self, instrument: int, product: str, side: str | None) -> None:
        self._named("set_group_side", product)


class Leg(TraderOrders):
    """A strategy holding `units` of SBIN NSE in `product`, opened at `opened_ns`: the
    session's own position and exit calls (`TraderOrders`) on a `Kernel`. A stop of as many
    units is kernel order 2, broker order "S1"."""

    symbol, exchange, finished, square_off_ns = "SBIN", "NSE", False, 0

    def __init__(self, name: str, product: str, units: int, opened_ns: int = START_NS) -> None:
        self.name, self.product = name, product
        self.engine = Kernel(product, units, opened_ns)
        self.venue = SimpleNamespace(
            bound=lambda: {},
            changing=lambda kernel_order: False,
            kernel_order_of=lambda order_id: 2 if order_id == "S1" else None,
            order_id_of=lambda kernel_order: "S1" if kernel_order == 2 else "",
        )
        self.records = SimpleNamespace(filled=lambda before, after: None)

    @property
    def exits(self) -> list[str]:
        """The reasons of the exits its kernel was asked for."""
        return self.engine.exits

    def lock(self, reason: str) -> bool:
        return True

    def unlock(self, reason: str) -> bool:
        return True


class Loop:
    """The parts of a session loop the readers call: its clock, its strategies and their
    groups, what it hands the order worker, its events, alerts and fills."""

    trading, locked = True, False

    def __init__(self, *legs: Leg) -> None:
        self.traders = legs
        self.clock = START_NS
        self.groups = Groups(legs, event=self.event)
        self.handed: list[Any] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.alerts: list[str] = []
        self.fills: list[Any] = []
        self.mismatches = 0
        self.last_tick: dict[Any, Any] = {}
        self.parts = SimpleNamespace(
            spec=SimpleNamespace(arguments=SimpleNamespace(on_stop="keep")),
            worker=SimpleNamespace(silent_since_ns=None),
            mode=Mode.LIVE,
        )
        self.margin = MarginHolds(self)
        idle = SimpleNamespace(stop=None, unbound_add_units=0, orders={})
        nothing = SimpleNamespace(holds=lambda trader: False)
        self.stops = SimpleNamespace(leg=lambda trader: idle, held=nothing, protection=nothing)
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


def read(
    loop: Loop,
    reconciler: Reconciler,
    positions: dict[str, Any],
    trades: dict[str, Any] = NO_TRADES,
) -> None:
    """Ask for a read of every leg now and answer it with `positions`, `trades` and
    `HOLDINGS`."""
    reconciler.ask_of(None)
    request = [r for r in loop.handed if isinstance(r, ReadBooks)][-1]
    loop.clock += NS
    books = Books(
        request.generation,
        positions,
        NO_ORDERS,
        trades,
        NO_FUNDS,
        loop.clock,
        loop.clock,
        holdings=HOLDINGS,
    )
    reconciler.books(books)


def other_client_sold(units: int) -> dict[str, Any]:
    """A trade book holding one trade of another client's order: `units` of SBIN NSE sold
    in CNC at `OTHER_PRICE`."""
    trade = {
        "orderid": "X9",
        "symbol": "SBIN",
        "exchange": "NSE",
        "product": "CNC",
        "action": "SELL",
        "quantity": units,
        "average_price": OTHER_PRICE,
    }
    return {"status": "success", "data": [trade]}


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
    hold.engine.units = 60
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


def test_a_shared_cnc_difference_is_judged_on_the_cnc_units():
    first, second = Leg("HoldA", "CNC", 30), Leg("HoldB", "CNC", 20)
    loop, reconciler = reconciling(first, second)

    read(loop, reconciler, POSITIONS, other_client_sold(3))

    # 30 + 20 = 50 held, 47 at the broker, and another client's sale of 3 explains it.
    assert compared(loop) == {
        "HoldA": ("explained_shared", CNC_UNITS),
        "HoldB": ("explained_shared", CNC_UNITS),
    }


#: When HoldA's and HoldB's positions opened in each case, UTC nanoseconds: a minute apart.
EARLIER, LATER = START_NS - 60 * NS, START_NS


@pytest.mark.parametrize(
    ("opened", "after"),
    [((EARLIER, LATER), (30, 17)), ((LATER, EARLIER), (27, 20))],
    ids=["HoldB opened last", "HoldA opened last"],
)
def test_a_shared_cnc_sale_is_taken_from_the_cnc_position_opened_last(opened, after):
    first, second = Leg("HoldA", "CNC", 30, opened[0]), Leg("HoldB", "CNC", 20, opened[1])
    loop, reconciler = reconciling(first, second)

    for _ in range(2):
        read(loop, reconciler, POSITIONS, other_client_sold(3))

    # Seen twice, the sale of 3 is shared out (18-multi-strategy 4.7): the member whose CNC
    # position opened most recently takes it, read from the kernel in CNC: 20 - 3 = 17, or
    # 30 - 3 = 27.
    assert (first.engine.units, second.engine.units) == after
    assert ("position", "CNC") in first.engine.calls


# ----- every kernel call of a CNC leg --------------------------------------------------------


def a_cnc_leg(name: str = "SbinHold") -> Leg:
    return Leg(name, "CNC", CNC_UNITS)


def placed_again(hold: Leg) -> None:
    protect(hold)


def exited_at_a_limit(hold: Leg) -> None:
    Loop(hold).margin.request_exit(hold, ACCOUNT_LOSS, 495.0)


def exited_at_market(hold: Leg) -> None:
    Loop(hold).margin.request_exit(hold, ACCOUNT_LOSS)


def units_read(hold: Leg) -> None:
    hold.open_units()


def sale_adopted(hold: Leg) -> None:
    """The broker holds 40 of the leg's 47 after another client sold 7: adopted."""
    loop, _ = reconciling(hold)
    positions = {"status": "success", "data": []}
    books = leg_books(
        positions, other_client_sold(7), NO_ORDERS, "SBIN", "NSE", product="CNC", holdings=HOLDINGS
    )
    Corrections(loop, {}, set()).act(
        hold, (Kind.EXPLAINED_FEWER, CNC_UNITS, 40), books, frozenset()
    )


def gates_set(hold: Leg) -> None:
    """`hold` shares SBIN in CNC with another strategy: its entry gates are set."""
    Loop(hold, a_cnc_leg("HoldB")).groups.before_decisions(hold)


@pytest.mark.parametrize(
    ("act", "calls"),
    [
        (placed_again, {"protect"}),
        (exited_at_a_limit, {"request_exit"}),
        (exited_at_market, {"request_exit"}),
        (units_read, {"position"}),
        (sale_adopted, {"apply_external_fill"}),
        (gates_set, {"set_shared", "set_group_side"}),
    ],
    ids=[
        "stop placed again",
        "exit at a limit through the margin gate",
        "exit at market through the margin gate",
        "the kernel's position",
        "another client's sale adopted",
        "a shared instrument's gates",
    ],
)
def test_every_kernel_call_of_a_cnc_leg_is_made_in_cnc(act: Callable[[Leg], None], calls: set[str]):
    hold = a_cnc_leg()

    act(hold)

    made = hold.engine.calls
    assert calls <= {call for call, _ in made}
    assert {product for _, product in made} == {"CNC"}


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


#: An answer of a book the broker could not give (the server's 500 for an expired login).
BOOK_FAILED = {"status": "error", "message": "HTTP 500: the broker refused", "code": 500}


class StartWorker:
    """The order worker at a start under the test seam (no threads): each `ReadBooks` put
    is answered on the next step with a read whose position book and holdings are the
    ones given, posted to the inbox. `clock` moves on with each sleep of the start."""

    def __init__(self, positions: dict[str, Any], holdings: dict[str, Any]) -> None:
        self.positions, self.holdings = positions, holdings
        self.asked: list[ReadBooks] = []
        self._waiting: list[ReadBooks] = []
        self._inbox: list[Envelope] = []
        self.clock = START_NS
        self.threads = False

    # The order worker.
    def put(self, request: ReadBooks) -> None:
        self.asked.append(request)
        self._waiting.append(request)

    def step(self) -> None:
        for request in self._waiting:
            books = Books(
                request.generation,
                self.positions,
                NO_ORDERS,
                NO_TRADES,
                NO_FUNDS,
                self.clock,
                self.clock,
                holdings=self.holdings,
            )
            self._inbox.append(Envelope(self.clock, books))
        self._waiting = []

    # The inbox.
    def take(self) -> Envelope | None:
        return self._inbox.pop(0) if self._inbox else None

    # The start's settings.
    def now(self) -> int:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += round(seconds * NS)


def started(positions: dict[str, Any], holdings: dict[str, Any]) -> tuple[StartWorker, Any]:
    """The start's book reads (`read_start`) against a `StartWorker`: the worker, and the
    start's books or the LiveError it raised."""
    worker = StartWorker(positions, holdings)
    journal = SimpleNamespace(unbound=lambda *, day: [])
    try:
        outcome: Any = read_start(worker, worker, worker, journal, None, None)
    except LiveError as error:
        outcome = error
    return worker, outcome


@pytest.mark.parametrize(
    ("positions", "holdings", "unread"),
    [
        (POSITIONS, BOOK_FAILED, "holdings"),
        (BOOK_FAILED, HOLDINGS, "position book"),
        (BOOK_FAILED, BOOK_FAILED, "position book and holdings"),
    ],
    ids=["holdings", "position book", "both"],
)
def test_a_start_whose_books_stay_unread_is_refused_naming_them(positions, holdings, unread):
    worker, refused = started(positions, holdings)

    assert isinstance(refused, LiveError)
    assert str(refused).startswith(f"OpenAlgo's {unread} could not be read at the start, so")
    # Read again every START_BOOKS_RETRY_S until START_BOOKS_WAIT_S ran out.
    assert len(worker.asked) > 1


def test_a_start_whose_holdings_are_read_takes_its_books():
    worker, start = started(POSITIONS, HOLDINGS)

    assert isinstance(start, StartBooks)
    assert (start.books.holdings, len(worker.asked)) == (HOLDINGS, 1)

"""Strategies with leg rules and the fake's order timeline, shared by the stops
supervisor's tests (`test_stops*.py`; `_live/stops.py`, 16-session-design 4 and 7, unit 10).

The market is `session_samples`': SBIN on NSE on Monday 2026-11-16, the session started at
14:40:05 IST, SBIN trading 10 units every 10 seconds at the script's price (500.00 unless a
test moves it). `StopHold` buys 10 SBIN at its first decision (14:42:01, filled at once at
500.00) with a stop 3 points below (an SL-M at 497.00) and the test's other leg rules.

The option is NIFTY's call of 29 December 2026 at 24000 on NFO (65 units a lot, tick
0.05, freeze quantity 1800), trading 65 units every 10 seconds at 120.00 unless a test
moves it: its stop is an SL with the limit 5 percent beyond the trigger.

`Timeline` records every change of an order at the fake broker as it happens (the fake's
own book, not the stream, which a fault may hold or lose), and `check_invariants` replays
it against the invariants of 16-session-design 7.1. Expected values come from the fake's
books and the scenario, never from the session's code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from plan_samples import candles
from session_samples import NS, TRADE_EVERY_S, ist, rig

import openalgo.backtest as bt
from openalgo.backtest._live._reconcile_guard import ExitGuard
from openalgo.backtest._live._types import Mode

#: The stop's distance below the entry, points, and the stop's trigger, rupees.
STOP_POINTS = 3.0
STOP_TRIGGER = 497.0
#: The entry's decision and the instant its fill reaches the session.
ENTRY_DECIDED = ist(14, 42, 1)
#: The option and its market.
OPTION = "NIFTY29DEC2624000CE"
OPTION_LOT = 65
OPTION_PRICE = 120.0
OPTION_FREEZE = 1800
#: The price types of a stop at the broker.
STOP_TYPES = frozenset({"SL", "SL-M"})


class StopHold(bt.Strategy):
    """Buys `qty` SBIN at its first decision with `rules` (a stop 3 points below unless a
    subclass says otherwise), then never trades again."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    qty = 10
    rules: dict[str, Any] = {"stop": bt.points(STOP_POINTS)}

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(self.qty, **type(self).rules)


class TargetHold(StopHold):
    """StopHold with a target 2 points above (502.00)."""

    rules = {"stop": bt.points(STOP_POINTS), "target": bt.points(2)}


class OptionHold(bt.Strategy):
    """Buys one lot of the option at its first decision with a stop 10 points below."""

    symbol, exchange, interval = OPTION, "NFO", "1m"
    rules: dict[str, Any] = {"stop": bt.points(10)}

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(OPTION_LOT, **type(self).rules)


@dataclass(frozen=True)
class Change:
    """An order of the live book as it stood right after a change at `ts_ns`, and the
    units and price of the fill the change made (0 for none)."""

    ts_ns: int
    order_id: str
    strategy: str
    symbol: str
    action: str
    price_type: str
    status: str
    quantity: int
    filled: int
    fill_units: int
    fill_price: float

    @property
    def working(self) -> bool:
        return self.status in ("open", "trigger pending")


class Timeline:
    """Every change of an order of the fake's live book, in order (module docstring)."""

    def __init__(self, fake: Any) -> None:
        self.changes: list[Change] = []
        venue = fake._venue
        publish = venue._publish
        clock = fake.clock

        def spy(book: Any, change: Any) -> None:
            if book.mode is Mode.LIVE:
                order, fill = change.order, change.fill
                self.changes.append(
                    Change(
                        clock.now(),
                        order.order_id,
                        order.strategy,
                        order.symbol,
                        order.action,
                        order.price_type,
                        order.status,
                        order.quantity,
                        order.filled,
                        0 if fill is None else fill.quantity,
                        0.0 if fill is None else fill.price,
                    )
                )
            publish(book, change)

        venue._publish = spy

    def states(self):
        """Each moment's orders by ID and net position by symbol, after each change."""
        orders: dict[str, Change] = {}
        positions: dict[str, int] = {}
        for change in self.changes:
            orders[change.order_id] = change
            sign = 1 if change.action == "BUY" else -1
            positions[change.symbol] = positions.get(change.symbol, 0) + sign * change.fill_units
            yield change, orders, positions

    def stops_working(self, symbol: str = "SBIN") -> list[Change]:
        """The stops of `symbol` working at the end."""
        last: dict[str, Change] = {}
        for change in self.changes:
            last[change.order_id] = change
        return [
            c
            for c in last.values()
            if c.symbol == symbol and c.working and c.price_type in STOP_TYPES
        ]


def check_invariants(timeline: Timeline, *, strategy: str, unprotected_at_end: bool = False):
    """The invariants of 16-session-design 7.1 for a long-only `strategy`, at every change
    of an order: at most one working stop, the working closing orders never above the
    position, never short, every fill priced; at the end, a position open has a working
    stop covering it (unless `unprotected_at_end`)."""
    position = 0
    working: dict[str, Change] = {}
    for change, orders, positions in timeline.states():
        assert change.fill_units == 0 or change.fill_price > 0, change
        own = [c for c in orders.values() if c.strategy == strategy and c.working]
        for symbol in {c.symbol for c in orders.values() if c.strategy == strategy}:
            held = positions.get(symbol, 0)
            stops = [c for c in own if c.symbol == symbol and c.price_type in STOP_TYPES]
            assert len(stops) <= 1, (change, stops)
            closing = sum(
                c.quantity - c.filled for c in own if c.symbol == symbol and c.action == "SELL"
            )
            assert closing <= max(held, 0), (change, closing, held)
            assert held >= 0, (change, held)
        working = {c.order_id: c for c in own}
        position = sum(positions.values())
    if position and not unprotected_at_end:
        covered = sum(c.quantity - c.filled for c in working.values() if c.price_type in STOP_TYPES)
        assert covered >= position, (position, working)


def events(outcome: Any, kind: str) -> list[dict[str, Any]]:
    """The session's event-stream lines of `kind`, oldest first."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]


def after_the_final_read(outcome: Any) -> list[dict[str, Any]]:
    """The session's event-stream lines written after the close asked for its final book
    read (its last `ReadBooks` request), oldest first."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        written = [json.loads(line) for line in stream]
    last = max(
        i
        for i, event in enumerate(written)
        if event["kind"] == "request" and event["type"] == "ReadBooks"
    )
    return written[last + 1 :]


def handed_after_the_final_read(outcome: Any) -> list[tuple[str, str | None]]:
    """What the session handed to the order worker after the close asked for its final
    book read: (type, purpose) of each request."""
    return [
        (event["type"], event.get("purpose"))
        for event in after_the_final_read(outcome)
        if event["kind"] == "request"
    ]


def held_after_the_final_read(outcome: Any) -> list[int]:
    """The kernel orders whose requests the exit guard held back after the close asked
    for its final book read: a request the engine made then."""
    return [
        event["kernel_order"]
        for event in after_the_final_read(outcome)
        if event["kind"] == "exit_held"
    ]


def ctrl_c_at_the_first_sighting(world: Any, monkeypatch: Any, *, broker_below: int) -> None:
    """Ask the session to stop (Ctrl+C) as the first comparison that shows the broker
    holding fewer than `broker_below` signed units of the leg opens the exit guard on it:
    the close's final read is then the second comparison to show it, the one that acts."""
    opened = ExitGuard.open

    def open_then_press(guard: ExitGuard, name: str, broker_units: int) -> None:
        if broker_units < broker_below and not world.stop.count:
            world.stop.request()
        opened(guard, name, broker_units)

    monkeypatch.setattr(ExitGuard, "open", open_then_press)


def sold_by_hand(world: Any, quantity: int, at_ns: int) -> None:
    """Another client of the account sells `quantity` SBIN at market at `at_ns`."""
    other = world.fake.client()
    world.script.at(
        at_ns,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=quantity,
        ),
    )


def stop_world(make_fake_openalgo: Any, monkeypatch: Any, tmp_path: Any, **options: Any) -> Any:
    """A `session_samples.rig` with the fake's order timeline recorded (`world.timeline`)."""
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, **options)
    world.timeline = Timeline(world.fake)
    return world


def option_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    trades_until: tuple[int, int] = (15, 12),
    price: float = OPTION_PRICE,
) -> Any:
    """A rig whose fake also trades the option (module docstring) from 14:40:15 to the IST
    time `trades_until`, at `world.option_price` (`price` at first, which a test may
    change); SBIN does not trade."""
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=None)
    fake = world.fake
    fake.add_instrument(
        OPTION,
        "NFO",
        lot_size=OPTION_LOT,
        tick_size=0.05,
        freeze_qty=OPTION_FREEZE,
        instrument_type="CE",
        expiry="29-DEC-26",
        strike=24000.0,
    )
    fake.seed_candles(OPTION, "NFO", [_quarter(row) for row in candles(today_until=(14, 40))])
    world.option_price = price
    fake.trade(OPTION, "NFO", price, volume=OPTION_LOT)
    for instant in range(ist(14, 40, 15), ist(*trades_until), TRADE_EVERY_S * NS):
        world.script.at(instant, lambda: fake.trade(OPTION, "NFO", world.option_price, volume=65))
    return world


def _quarter(row: tuple) -> tuple:
    """A SBIN candle priced at a quarter: an option's level, about 125 rupees."""
    start, open_, high, low, close, volume = row
    return (start, open_ / 4, high / 4, low / 4, close / 4, volume)


def at_price(world: Any, instant_ns: int, price: float) -> None:
    """SBIN trades at `price` from `instant_ns` on."""
    before = world.script.price
    world.script.price = lambda now_ns: price if now_ns >= instant_ns else before(now_ns)


def first_order(world: Any, price_type: str, action: str = "SELL") -> Any:
    """The fake's first live order of `price_type` and `action`."""
    return next(o for o in world.orders() if (o.price_type, o.action) == (price_type, action))


__all__ = [
    "ENTRY_DECIDED",
    "OPTION",
    "OPTION_FREEZE",
    "OPTION_LOT",
    "OPTION_PRICE",
    "STOP_POINTS",
    "STOP_TRIGGER",
    "Change",
    "OptionHold",
    "StopHold",
    "TargetHold",
    "Timeline",
    "after_the_final_read",
    "at_price",
    "check_invariants",
    "ctrl_c_at_the_first_sighting",
    "events",
    "first_order",
    "handed_after_the_final_read",
    "held_after_the_final_read",
    "option_world",
    "sold_by_hand",
    "stop_world",
]

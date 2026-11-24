"""A kernel session, its commands and the worker's answers, shared by the venue map's and
the fill ledger's tests (`test_venue_*.py`, `test_fill_ledger.py`).

The session is the NSE day 2025-07-01 on 1-minute slices from 09:15 IST, one instrument
(SBIN, tick 0.05), a 1-minute strategy, a capital of 10,00,000 and bars flat at 500.00;
the test plays the venue. Nothing here contacts a server.
"""

from __future__ import annotations

import datetime as dt
import itertools
from typing import Any

import numpy as np

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest._live._journal_reads import JournalSession
from openalgo.backtest._live._order_answers import Outcome, SendOutcome
from openalgo.backtest._live._types import AlertKind, Mode, OrderStatus, OrderUpdate
from openalgo.backtest._live.venue_map import VenueMap
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.reference import Instrument

DAY = dt.date(2025, 7, 1)
STRATEGY = "ema_cross"
MINUTE_NS = 60 * 1_000_000_000


def open_ns(day: dt.date) -> int:
    """09:15 IST (03:45 UTC) of `day`, UTC nanoseconds."""
    return int(dt.datetime(day.year, day.month, day.day, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9


OPEN_NS = open_ns(DAY)
SBIN = Instrument.cash_equity("SBIN", "NSE")
_NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))


def journal_session(mode: Mode = Mode.LIVE, starts: int = 1) -> JournalSession:
    """The journal session of DAY in `mode`, started `starts` times."""
    return JournalSession(DAY, mode, "zerodha", OPEN_NS, OPEN_NS, starts)


def _config(day: dt.date) -> dict[str, Any]:
    return {
        "first_day": to_day(day),
        "last_day": to_day(day),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": 1_000_000.0,
        "costs": None,
        "slippage": ("zero", 0.0, 0),
        "risk": {
            "risk_basis": "gross",
            "after_risk_exit": "lock_session",
            "round_levels": True,
            "strict_config": True,
        },
        "deliver_events": False,
    }


def new_session(day: dt.date = DAY) -> Any:
    """A kernel session of SBIN on `day`, before its first slice."""
    instrument = {
        "symbol": "SBIN",
        "exchange": "NSE",
        "kind": "EQ",
        "lot_size": 1,
        "tick_size": 0.05,
        "multiplier": 1.0,
        "freeze_qty": None,
        "last_trading_day": None,
    }
    engine_class = native_attr("Engine")
    return engine_class.session(default_calendar()._cal, _config(day), [instrument], [_NO_BARS])


def release_minute(engine: Any, minute: int, *, day: dt.date = DAY, price: float = 500.0) -> str:
    """Hand the session of `day` the bar of `minute` after 09:15, flat at `price`, release
    it, return the pause."""
    assert engine.advance()[0] == "waiting"
    bar_open = open_ns(day) + minute * MINUTE_NS
    engine.add_bar(0, (bar_open, price, price, price, price, 1_000.0))
    engine.release(minute + 1)
    return engine.advance()[0]


def order(side: str, units: float, **fields: Any) -> dict[str, Any]:
    """An order request of SBIN in MIS, a MARKET order unless `fields` say otherwise."""
    return {
        "instrument": 0,
        "side": side,
        "quantity": {"kind": "units", "value": units},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        **fields,
    }


class Alerts:
    """Keeps every alert sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[AlertKind, str]] = []

    def send(self, kind: AlertKind, text: str) -> None:
        self.sent.append((kind, text))

    def test(self) -> bool:
        return True


class Rig:
    """A kernel session and its venue map, on a clock the test sets (`now_ns`)."""

    def __init__(self, mode: Mode = Mode.LIVE, starts: int = 1) -> None:
        self.engine = new_session()
        self.alerts = Alerts()
        self.now_ns = OPEN_NS
        self.map = VenueMap(
            self.engine,
            strategy=STRATEGY,
            instruments=[SBIN],
            product="MIS",
            session=journal_session(mode, starts),
            request_ids=itertools.count(1),
            alerts=self.alerts,
            now=lambda: self.now_ns,
        )
        assert release_minute(self.engine, 0) == "bar_close"
        self.minute = 0

    def submit(self, request: dict[str, Any]) -> int:
        """Submit `request` at the current decision point; its kernel order."""
        outcome, kernel_order, code = self.engine.submit(request)
        assert outcome == "submitted", code
        return kernel_order

    def wait(self) -> list[Any]:
        """Leave the decision point; the requests of the commands it made."""
        assert self.engine.advance()[0] == "waiting"
        return self.map.requests(self.engine.take_commands())

    def next_close(self) -> None:
        """Run to the close of the next minute (a decision point)."""
        self.minute += 1
        assert release_minute(self.engine, self.minute) == "bar_close"

    def status(self, kernel_order: int) -> str:
        return self.engine.order(kernel_order)["status"]

    def units(self, kernel_order: int) -> float:
        return self.engine.order(kernel_order)["quantity_units"]


def accepted(order_id: str) -> SendOutcome:
    return SendOutcome(Outcome.ACCEPTED, order_id=order_id)


def rejected(reason: str = "RMS: margin exceeds") -> SendOutcome:
    return SendOutcome(Outcome.REJECTED, reason=reason)


def unanswered() -> SendOutcome:
    return SendOutcome(Outcome.UNANSWERED, reason="timeout")


def update(
    order_id: str,
    status: OrderStatus = OrderStatus.OPEN,
    *,
    quantity: int = 100,
    filled: int = 0,
    average: float = 0.0,
    source: str = "stream",
    action: str = "BUY",
) -> OrderUpdate:
    """A cumulative word of SBIN order `order_id`: `filled` of `quantity` at `average`."""
    return OrderUpdate(
        order_id=order_id,
        symbol="SBIN",
        exchange="NSE",
        action=action,
        quantity=quantity,
        filled_quantity=filled,
        pending_quantity=0 if status.terminal else quantity - filled,
        average_price=average,
        status=status,
        rejection_reason="RMS: margin exceeds" if status is OrderStatus.REJECTED else "",
        mode=Mode.LIVE,
        ts_recv_ns=OPEN_NS + 90 * 1_000_000_000,
        source=source,
    )

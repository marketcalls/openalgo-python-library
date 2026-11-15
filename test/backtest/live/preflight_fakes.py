"""In-process fakes for the preflight's tests: an OpenAlgo client that answers `ping`,
`funds`, `symbol` and `intervals` as the SDK returns the server's answers, an alert
channel whose test alert is scripted, and a plan to change one field at a time. Nothing
here contacts a server.

The answers are shaped as the server writes them (services/ping_service.py,
services/funds_service.py and the brokers' get_margin_data, sandbox/fund_manager.py,
services/symbol_service.py, services/intervals_service.py), and an answer other than
HTTP 200 as the SDK wraps it ("HTTP <code>: <body>", openalgo/account.py and
openalgo/data.py `_handle_response`).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from typing import Any

from openalgo.backtest._live._types import AlertKind, Mode
from openalgo.backtest._live.preflight import PreflightPlan

NS = 1_000_000_000


def ist_ns(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    """UTC nanoseconds of an IST wall-clock time."""
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    moment = dt.datetime(year, month, day, hour, minute, second, tzinfo=ist)
    return int(moment.timestamp()) * NS


#: Monday 2026-11-16, 10:15:20 IST: NSE and NFO trade that day.
NOW_NS = ist_ns(2026, 11, 16, 10, 15, 20)


def sdk_http_error(code: int, body: dict[str, Any]) -> dict[str, Any]:
    """A non-200 answer as the SDK returns it: the server's body flattened into the
    message."""
    return {
        "status": "error",
        "message": f"HTTP {code}: {json.dumps(body)}",
        "code": code,
        "error_type": "http_error",
    }


#: The SDK's own error when OpenAlgo cannot be reached.
UNREACHABLE = {
    "status": "error",
    "message": "Failed to connect to the server. Please check if the server is running.",
    "error_type": "connection_error",
}
KEY_REFUSED = sdk_http_error(403, {"status": "error", "message": "Invalid openalgo apikey"})


def pong(broker: str = "zerodha") -> dict[str, Any]:
    return {"status": "success", "data": {"message": "pong", "broker": broker}}


def live_funds(available: str = "250000.00") -> dict[str, Any]:
    """A broker's funds, its amounts as text."""
    data = {
        "availablecash": available,
        "collateral": "0.00",
        "m2mrealized": "0.00",
        "m2munrealized": "0.00",
        "utiliseddebits": "0.00",
    }
    return {"status": "success", "data": data}


def sandbox_funds(available: float = 10_000_000.0) -> dict[str, Any]:
    """The sandbox's funds, its amounts as numbers, answered in analyzer mode."""
    data = {"availablecash": available, "collateral": 0.0, "m2munrealized": 0.0}
    return {"status": "success", "data": data, "mode": "analyze"}


#: What a broker whose login has expired answers: success, with no data.
EXPIRED_FUNDS = {"status": "success", "data": {}}


def symbol_found(
    symbol: str, exchange: str, *, lotsize: Any = 1, tick_size: Any = 0.05, freeze_qty: Any = 0
) -> dict[str, Any]:
    data = {
        "symbol": symbol,
        "exchange": exchange,
        "lotsize": lotsize,
        "tick_size": tick_size,
        "freeze_qty": freeze_qty,
        "instrumenttype": "EQ",
    }
    return {"status": "success", "data": data}


def symbol_missing(symbol: str, exchange: str) -> dict[str, Any]:
    message = f"Symbol {symbol} not found in exchange {exchange}"
    return sdk_http_error(404, {"status": "error", "message": message})


#: The intervals of a broker that offers minute and daily candles.
INTERVALS = {
    "status": "success",
    "data": {
        "seconds": [],
        "minutes": ["1m", "3m", "5m", "15m", "30m", "60m"],
        "hours": [],
        "days": ["D"],
        "weeks": [],
        "months": [],
    },
}


class FakeOpenAlgo:
    """Answers the preflight's calls with the answers given; `symbols` maps (symbol,
    exchange) to an answer, and an instrument not in it is not found. `calls` lists every
    call in order. Any other call fails the test."""

    def __init__(
        self,
        *,
        ping: dict[str, Any] | None = None,
        funds: dict[str, Any] | None = None,
        symbols: dict[tuple[str, str], dict[str, Any]] | None = None,
        intervals: dict[str, Any] | None = None,
    ) -> None:
        self._ping = pong() if ping is None else ping
        self._funds = live_funds() if funds is None else funds
        self._symbols = (
            {("SBIN", "NSE"): symbol_found("SBIN", "NSE")} if symbols is None else symbols
        )
        self._intervals = INTERVALS if intervals is None else intervals
        self.calls: list[tuple[str, ...]] = []

    def ping(self) -> dict[str, Any]:
        self.calls.append(("ping",))
        return self._ping

    def funds(self) -> dict[str, Any]:
        self.calls.append(("funds",))
        return self._funds

    def symbol(self, *, symbol: str, exchange: str) -> dict[str, Any]:
        self.calls.append(("symbol", symbol, exchange))
        return self._symbols.get((symbol, exchange), symbol_missing(symbol, exchange))

    def intervals(self) -> dict[str, Any]:
        self.calls.append(("intervals",))
        return self._intervals

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the preflight must not call {name}")


class FakeAlerts:
    """An alert channel whose test alert is `delivered` or not; counts the tests."""

    def __init__(self, delivered: bool = True) -> None:
        self.delivered = delivered
        self.tests = 0
        self.sent: list[tuple[AlertKind, str]] = []

    def send(self, kind: AlertKind, text: str) -> None:
        self.sent.append((kind, text))

    def test(self) -> bool:
        self.tests += 1
        return self.delivered


def make_plan(**changes: Any) -> PreflightPlan:
    """A LIVE plan for SBIN on NSE, 5-minute bars, MIS, that passes every check against
    `FakeOpenAlgo()` at `NOW_NS`, with `changes` applied."""
    plan = PreflightPlan(
        instruments=[("SBIN", "NSE")],
        interval="5m",
        product="MIS",
        capital=100_000,
        max_daily_loss=2_000,
        mode=Mode.LIVE,
        alerts=FakeAlerts(),
    )
    return dataclasses.replace(plan, **changes)

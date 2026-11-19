"""The preflight checks that read OpenAlgo's answers (`preflight`, 08-live-trading section
4): the OpenAlgo session (`ping` and `funds`), the broker, the mode funds was answered
in, the funds against the capital, each instrument's facts (`symbol`; an index is
refused, since it cannot be traded), the broker's login (one `quotes` call), and the
intervals (`intervals`): the strategy's, and the 1-minute candles the live bars are
checked against (D27).

The answers are read as the SDK returns them (openalgo/account.py, openalgo/data.py): the
server's JSON on HTTP 200, else the SDK's own error, whose message "HTTP <code>: <body>"
holds the server's answer (`server_answer` takes it back out). Funds do not prove the
broker's login: a broker whose login has expired may still answer them with success and
no data, or with zeros (broker/*/api/funds.py on the server), and in SANDBOX mode they
are the sandbox's own and never reach the broker. A quote does, in both modes: the
server asks the broker for it whatever its analyzer switch (services/quotes_service.py).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .._layout import NANOS_PER_MINUTE, parse_interval
from ..errors import ConfigError
from ..reference.symbols import INDEX_EXCHANGES
from ._answers import server_answer
from ._guard_answers import SUCCESS
from ._order_trades import UNSUPPORTED_BROKERS, UNVERIFIED_BROKERS
from ._preflight_plan import WARNING, Check, InstrumentFacts, PreflightPlan, positive_amount
from ._types import Mode, mode_of
from .bars import DEFAULT_INTERVAL_NS

#: The interval of the broker candles the live bars are checked against (D27), as
#: OpenAlgo writes it: "1m".
CANDLE_INTERVAL = f"{DEFAULT_INTERVAL_NS // NANOS_PER_MINUTE}m"
#: The most characters of an answer quoted in a message: a gateway's error page in front
#: of OpenAlgo can run to kilobytes of HTML.
_MAX_QUOTED_CHARS = 200


class PreflightClient(Protocol):
    """What the preflight needs of an `openalgo.api` client: five of its REST calls, and
    for the clock check its pooled HTTP client and the URL of its API."""

    #: The SDK's pooled HTTP client (`BaseAPI.client`).
    client: httpx.Client
    #: The URL of the API, "<host>/api/v1/" (`BaseAPI.base_url`).
    base_url: str

    def ping(self) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/ping`."""

    def funds(self) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/funds`."""

    def symbol(self, *, symbol: str, exchange: str) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/symbol`."""

    def quotes(self, *, symbol: str, exchange: str) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/quotes`."""

    def intervals(self) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/intervals`."""


@dataclass(frozen=True)
class ServerFindings:
    """What the server checks found: their checks in order, the broker `ping` named
    (None when ping did not answer), the available cash funds reported (rupees, None when
    funds did not answer) and the facts of every instrument found, in the plan's order."""

    checks: tuple[Check, ...]
    broker: str | None
    funds_available: float | None
    instruments: tuple[InstrumentFacts, ...]


def read_server(client: PreflightClient, plan: PreflightPlan) -> ServerFindings:
    """Run the server checks of `plan`. When ping fails, nothing else is asked:
    OpenAlgo is down or refuses the API key, and every other answer would say the same."""
    ping = client.ping()
    broker = _broker_of(ping)
    if broker is None:
        return ServerFindings((_ping_failed(ping),), None, None, ())
    funds = client.funds()
    available = _available_cash(funds)
    checks = [_session_check(broker, funds, available), _broker_check(broker)]
    if available is not None:
        checks.append(_mode_check(funds, plan.mode))
        capital = positive_amount(plan.capital)
        # Without a usable capital there is nothing to compare: its own check fails.
        if capital is not None:
            checks.append(_funds_check(available, capital, plan.mode))
    found = [_symbol_check(client, symbol, exchange) for symbol, exchange in plan.instruments]
    checks += [check for check, _ in found]
    facts = tuple(facts for _, facts in found if facts is not None)
    # One quote shows the broker answers. With no instrument found there is none to ask
    # for, and their own checks have failed.
    if facts:
        checks.append(_quotes_check(client, facts[0]))
    checks.append(_interval_check(client.intervals(), plan.bar_intervals))
    return ServerFindings(tuple(checks), broker, available, facts)


def _broker_of(ping: Mapping[str, Any]) -> str | None:
    data = ping.get("data") if ping.get("status") == SUCCESS else None
    broker = data.get("broker") if isinstance(data, Mapping) else None
    return broker.strip() if isinstance(broker, str) and broker.strip() else None


#: The HTTP status OpenAlgo answers `ping` with when it refuses the API key: a wrong key,
#: or a key whose OpenAlgo session ended at the daily login reset (database/auth_db.py of
#: the server). Only this ping failure may be fixed by logging in; a refused connection,
#: a timeout, a server error or a wrong path fails the preflight at once.
_KEY_REFUSED_STATUS = 403


def _ping_failed(ping: Mapping[str, Any]) -> Check:
    if ping.get("status") == SUCCESS:
        reason = f"it answered without naming the broker: {_quoted(ping)}"
    else:
        reason = _reason(ping)
    return Check(
        "broker_session",
        False,
        f"OpenAlgo's ping failed ({reason}): check that OpenAlgo is running at the client's "
        "host, that the API key is right and that you are logged in to your broker in "
        "OpenAlgo. The funds, symbol, quote, interval and clock checks and the test alert "
        "wait for ping",
        login=ping.get("code") == _KEY_REFUSED_STATUS,
    )


def _available_cash(funds: Mapping[str, Any]) -> float | None:
    data = funds.get("data") if funds.get("status") == SUCCESS else None
    return _number(data.get("availablecash")) if isinstance(data, Mapping) else None


def _session_check(broker: str, funds: Mapping[str, Any], available: float | None) -> Check:
    if available is not None:
        return Check("broker_session", True, f"OpenAlgo is logged in to {broker}; funds answered")
    if funds.get("status") == SUCCESS:
        reason = f"it answered without the available cash: {_quoted(funds)}"
    else:
        reason = _reason(funds)
    return Check(
        "broker_session",
        False,
        f"OpenAlgo is logged in to {broker}, but the funds call failed ({reason}): the "
        "broker's login may have expired (it ends around 03:00 IST each day). Log in to "
        f"{broker} again in OpenAlgo, then start the session again",
        login=True,
    )


def _broker_check(broker: str) -> Check:
    why = UNSUPPORTED_BROKERS.get(broker.lower())
    if why is None and broker.lower() in UNVERIFIED_BROKERS:
        return Check(
            "broker_supported",
            True,
            f"{WARNING}bt.live trades through {broker}, but the session matches the trade "
            "book's trades to its orders by order ID, and that the two IDs agree has not been "
            f"checked for {broker}: watch the RECONCILE alerts, which name a fill the session "
            "could not match to its order",
        )
    if why is None:
        return Check("broker_supported", True, f"bt.live can trade through {broker}")
    return Check(
        "broker_supported",
        False,
        f"bt.live cannot trade through {broker}: {why}. Log OpenAlgo in to another broker "
        "to trade live",
    )


def _mode_check(funds: Mapping[str, Any], mode: Mode) -> Check:
    answered = mode_of(funds)
    if answered is mode:
        return Check("mode", True, f"OpenAlgo answered in {mode.name} mode, the session's")
    return Check(
        "mode",
        False,
        f"OpenAlgo answered funds in {answered.name} mode, but the session runs in "
        f"{mode.name} mode: its analyzer switch changed after the session read it. Set the "
        "switch in OpenAlgo to the mode you want, then start the session again",
    )


def _funds_check(available: float, capital: float, mode: Mode) -> Check:
    if available >= capital:
        return Check(
            "funds",
            True,
            f"{available:,.2f} rupees of available cash cover the capital of {capital:,.2f} rupees",
        )
    where = "the sandbox's funds in OpenAlgo" if mode is Mode.SANDBOX else "the broker account"
    # Some brokers answer an expired login with zeros instead of an error.
    expired = (
        ". No cash at all is also what some brokers report for an expired login: if the "
        "account holds funds, log in to OpenAlgo again"
        if available == 0
        else ""
    )
    return Check(
        "funds",
        False,
        f"OpenAlgo reports {available:,.2f} rupees of available cash (collateral not "
        f"counted), less than the capital of {capital:,.2f} rupees: lower capital, or add "
        f"funds to {where}{expired}",
        # The sandbox's funds never reach the broker: only a broker's zero hints at it.
        login=available == 0 and mode is Mode.LIVE,
    )


def _symbol_check(
    client: PreflightClient, symbol: str, exchange: str
) -> tuple[Check, InstrumentFacts | None]:
    name = f"symbol:{exchange}:{symbol}"
    if exchange in INDEX_EXCHANGES:
        return Check(
            name,
            False,
            f"{exchange}:{symbol} is an index, which cannot be traded: trade its futures or "
            "options instead",
        ), None
    answer = client.symbol(symbol=symbol, exchange=exchange)
    data = answer.get("data") if answer.get("status") == SUCCESS else None
    if not isinstance(data, Mapping):
        reason = _reason(answer) if answer.get("status") != SUCCESS else _quoted(answer)
        return Check(
            name,
            False,
            f"{exchange}:{symbol} was not found ({reason}): write the symbol as OpenAlgo "
            "does (SBIN on NSE, say) and check that the broker's master contract is "
            "downloaded in OpenAlgo",
        ), None
    answered = (str(data.get("symbol")).upper(), str(data.get("exchange")).upper())
    if answered != (symbol.upper(), exchange.upper()):
        return Check(
            name,
            False,
            f"OpenAlgo answered the lookup of {exchange}:{symbol} with "
            f"{data.get('exchange')}:{data.get('symbol')}: check that the symbol is written as "
            "OpenAlgo writes it, and download the broker's master contract again in OpenAlgo",
        ), None
    facts = _facts(symbol, exchange, data)
    if isinstance(facts, str):
        return Check(
            name,
            False,
            f"OpenAlgo gives {exchange}:{symbol} {facts}: download the broker's master "
            "contract again in OpenAlgo",
        ), None
    freeze = facts.freeze_quantity
    limit = "no freeze limit known" if freeze is None else f"freeze quantity {freeze} units"
    return Check(
        name,
        True,
        f"{exchange}:{symbol}: lot size {facts.lot_size} units, tick size {facts.tick_size} "
        f"rupees, {limit}",
    ), facts


def _facts(symbol: str, exchange: str, data: Mapping[str, Any]) -> InstrumentFacts | str:
    """The instrument's facts, or what is wrong with them. OpenAlgo gives a freeze
    quantity of 0, or none at all on an older server, when it knows no limit."""
    lot_size = _number(data.get("lotsize"))
    if lot_size is None or lot_size < 1 or not lot_size.is_integer():
        return f"no usable lot size ({data.get('lotsize')!r})"
    tick_size = _number(data.get("tick_size"))
    if tick_size is None or tick_size <= 0:
        return f"no usable tick size ({data.get('tick_size')!r})"
    given = data.get("freeze_qty")
    freeze = 0.0 if given is None else _number(given)
    if freeze is None or freeze < 0 or not freeze.is_integer():
        return f"no usable freeze quantity ({given!r})"
    return InstrumentFacts(symbol, exchange, int(lot_size), tick_size, int(freeze) or None)


def _quotes_check(client: PreflightClient, facts: InstrumentFacts) -> Check:
    """Passes when the broker quotes the instrument at a positive last traded price."""
    instrument = f"{facts.exchange}:{facts.symbol}"
    answer = client.quotes(symbol=facts.symbol, exchange=facts.exchange)
    data = answer.get("data") if answer.get("status") == SUCCESS else None
    ltp = _number(data.get("ltp")) if isinstance(data, Mapping) else None
    if ltp is not None and ltp > 0:
        return Check("broker_quotes", True, f"the broker quoted {instrument} at {ltp} rupees")
    if answer.get("status") != SUCCESS:
        reason = _reason(answer)
    else:
        reason = f"it answered without a positive price: {_quoted(answer)}"
    return Check(
        "broker_quotes",
        False,
        f"the broker gave no quote for {instrument} ({reason}): its login may have expired "
        "(it ends around 03:00 IST each day), which funds do not show in SANDBOX mode. Log "
        "in to OpenAlgo again, then start the session again",
        login=True,
    )


def _interval_check(answer: Mapping[str, Any], intervals: tuple[Any, ...]) -> Check:
    """Passes when the broker offers candles of each of `intervals` (the strategies') and
    the 1-minute candles."""
    wrong = []
    for interval in intervals:
        try:
            parse_interval(interval)
        except ConfigError as error:
            wrong.append(str(error))
    if wrong:
        return Check("interval", False, "; ".join(wrong))
    data = answer.get("data") if answer.get("status") == SUCCESS else None
    if not isinstance(data, Mapping):
        reason = _reason(answer) if answer.get("status") != SUCCESS else _quoted(answer)
        return Check(
            "interval",
            False,
            f"OpenAlgo did not list the broker's intervals ({reason}): check that OpenAlgo "
            "is logged in to the broker, then start the session again",
        )
    offered = [item for group in data.values() if isinstance(group, list) for item in group]
    needed = list(dict.fromkeys((*intervals, CANDLE_INTERVAL)))
    missing = [step for step in needed if step not in offered]
    one = len(intervals) == 1
    if not missing:
        whose = "the strategy's interval" if one else "the strategies' intervals"
        return Check(
            "interval",
            True,
            f"the broker offers {' and '.join(needed)} candles ({whose}, and "
            f"{CANDLE_INTERVAL} to check the live bars against)",
        )
    advice = []
    if any(interval in missing for interval in intervals):
        advice.append(f"choose one of those as {'the' if one else 'each'} strategy's interval")
    if CANDLE_INTERVAL in missing:
        advice.append(
            f"the session checks its tick-built bars against the broker's {CANDLE_INTERVAL} "
            "candles, so trade live through a broker that offers them"
        )
    return Check(
        "interval",
        False,
        f"the broker offers no {' and no '.join(missing)} candles (it offers "
        f"{', '.join(offered) or 'none'}): {'; '.join(advice)}",
    )


def _number(value: Any) -> float | None:
    """A finite number from a number or a numeric string (brokers give amounts as text,
    the sandbox as numbers), else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reason(answer: Mapping[str, Any]) -> str:
    """What an error answer says: the server's own message and HTTP status where the SDK
    wrapped them as "HTTP <code>: <body>", else the SDK's message."""
    own = server_answer(answer)
    message = own.get("message")
    text = _clip(str(message)) if message else "an error without a message"
    # `server_answer` gives a new answer only when it took the SDK's wrapper apart.
    return f"HTTP {own['code']}: {text}" if own is not answer else text


def _quoted(answer: Mapping[str, Any]) -> str:
    return _clip(repr(dict(answer)))


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_QUOTED_CHARS else f"{text[:_MAX_QUOTED_CHARS]}..."


__all__ = ["CANDLE_INTERVAL", "PreflightClient", "ServerFindings", "read_server"]

"""The preflight checks that need no answer from OpenAlgo's REST API (`preflight`): the risk
limits, the product, each exchange's trading day, the clock and the alert channel
(08-live-trading section 4; the alert rule of 15-phase2-plan section 6).

Amounts are rupees; instants UTC nanoseconds, shown as IST.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

from .._series import ist_time
from ..calendar import Calendar
from ..errors import CalendarError
from ._preflight_plan import WARNING, Check, positive_amount
from ._types import AlertsLike, Mode

#: The one product this release trades live: intraday MIS (Tier 1, 08-live-trading
#: section 1). CNC and NRML wait for overnight stops and holdings reconciliation.
LIVE_PRODUCT = "MIS"
#: The most the local clock may differ from the exchange's, in seconds (08-live-trading
#: section 4).
MAX_CLOCK_SKEW_SECONDS = 2
# The calendar's coverage of a date it holds no holiday list for: the weekday rule only.
_TEMPLATE_COVERAGE = "template"


def limit_checks(
    capital: Any, max_daily_loss: Any, account_max_daily_loss: Any
) -> tuple[Check, Check, Check]:
    """The checks of the risk limits: `capital` and `max_daily_loss` given and positive
    (both mandatory for `bt.live`), `account_max_daily_loss` positive when given."""
    return (
        _required("capital", capital, "the rupees the session may trade with"),
        _required(
            "max_daily_loss",
            max_daily_loss,
            "the loss in rupees after which the strategy stops entering for the day",
        ),
        _account_limit(account_max_daily_loss),
    )


def product_check(product: Any) -> Check:
    """Passes for the MIS product only."""
    if product == LIVE_PRODUCT:
        return Check("product", True, "the product is MIS (intraday)")
    return Check(
        "product",
        False,
        f"product {product!r} cannot trade live in this release, which trades intraday MIS "
        "only (CNC and NRML come later): set the strategy's product to 'MIS'",
    )


def trading_day_checks(exchanges: Iterable[str], now_ns: int, calendar: Calendar) -> list[Check]:
    """One check per exchange (first seen first): the exchange trades on the IST date of
    `now_ns` (UTC nanoseconds) and its session has not closed yet. Before the open is
    fine, the session waits for it. A pass warns when the calendar holds no holiday list
    for the year, since a holiday would then go unseen."""
    day = ist_time(now_ns).date()
    return [_trading_day(exchange, day, now_ns, calendar) for exchange in dict.fromkeys(exchanges)]


def clock_check() -> Check:
    """The clock check, which passes with a warning: OpenAlgo's REST answers carry no time
    of the server's (ping, funds, symbol, intervals, analyzerstatus, quotes; the SDK keeps
    no response header), so the local clock cannot be compared with it."""
    return Check(
        "clock",
        True,
        f"{WARNING}the local clock could not be compared with the server's, as none of "
        "OpenAlgo's answers carries the server's time: keep this computer's clock "
        "synchronised with an internet time server, since the session needs it within "
        f"{MAX_CLOCK_SKEW_SECONDS} seconds of the exchange's",
    )


def alerts_check(alerts: AlertsLike | None, mode: Mode) -> Check:
    """Sends a test alert through `alerts`. In LIVE mode it must be delivered; in SANDBOX
    mode a missing or failing channel passes with a warning, so a sandbox session can run
    before alerts are set up."""
    if alerts is None:
        problem = "the session has no alert channel"
    elif alerts.test() is True:
        return Check("alerts", True, "a test alert was delivered")
    else:
        problem = "the test alert was not delivered"
    if mode is Mode.LIVE:
        return Check(
            "alerts",
            False,
            f"{problem}, and a LIVE session needs its alerts: set up the alerts through "
            "OpenAlgo's Telegram bot, check that a message reaches you, then start the "
            "session again",
        )
    return Check(
        "alerts",
        True,
        f"{WARNING}{problem}: this SANDBOX session runs without alerts, and a LIVE "
        "session will need them",
    )


def _required(name: str, value: Any, meaning: str) -> Check:
    amount = positive_amount(value)
    if amount is None:
        return Check(
            name,
            False,
            f"{name} is required and must be a positive number of rupees, not {value!r}: "
            f"pass {name}=..., {meaning}",
        )
    return Check(name, True, f"{name} is {amount:,.2f} rupees")


def _account_limit(value: Any) -> Check:
    name = "account_max_daily_loss"
    if value is None:
        return Check(name, True, f"{name} is not set: no loss limit for the account as a whole")
    amount = positive_amount(value)
    if amount is None:
        return Check(
            name,
            False,
            f"{name} must be a positive number of rupees when given, not {value!r}: pass "
            "the daily loss limit of all the account's strategies together, or leave it out",
        )
    return Check(name, True, f"{name} is {amount:,.2f} rupees, over all the account's strategies")


def _trading_day(exchange: str, day: dt.date, now_ns: int, calendar: Calendar) -> Check:
    name = f"trading_day:{exchange}"
    try:
        session = calendar.session(exchange, day)
    except CalendarError as error:
        return Check(name, False, f"the trading day of {exchange} cannot be told: {error}")
    if session is None:
        return Check(
            name,
            False,
            f"{exchange} does not trade today, {day} (IST): start the session on a trading day",
        )
    open_ns, close_ns = session
    if now_ns >= close_ns:
        return Check(
            name,
            False,
            f"the {exchange} session of {day} closed at {_ist_clock(close_ns)} IST: start the "
            "session on a trading day, before the close",
        )
    message = f"{exchange} trades today, {day}, from {_ist_clock(open_ns)} to "
    message += f"{_ist_clock(close_ns)} IST"
    if calendar.coverage(exchange, day) == _TEMPLATE_COVERAGE:
        message = (
            f"{WARNING}{message}, but the calendar holds no holiday list for {day.year}: "
            f"check that today is not a holiday of {exchange}"
        )
    return Check(name, True, message)


def _ist_clock(ts_ns: int) -> str:
    return ist_time(ts_ns).strftime("%H:%M")


__all__ = [
    "LIVE_PRODUCT",
    "MAX_CLOCK_SKEW_SECONDS",
    "alerts_check",
    "clock_check",
    "limit_checks",
    "product_check",
    "trading_day_checks",
]

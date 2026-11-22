"""The preflight checks that need no answer from OpenAlgo's REST API (`preflight`): the risk
limits, the product, each exchange's trading day and the alert channel (08-live-trading
section 4; the alert rule of 15-phase2-plan section 6).

Amounts are rupees; instants UTC nanoseconds, shown as IST.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

from .._protocols import COVERAGE_LEVELS
from .._series import ist_time
from ..calendar import Calendar, IntradayTimes, SessionInfo
from ..errors import CalendarError
from ..report._format import indian_digits
from ._preflight_plan import WARNING, Check, positive_amount
from ._types import AlertsLike, Mode

#: The one product this release trades live: intraday MIS (Tier 1, 08-live-trading
#: section 1). CNC and NRML wait for overnight stops and holdings reconciliation.
LIVE_PRODUCT = "MIS"
# The calendar's coverage levels of a date that need a note: no holiday list (the weekday
# rule only), and holiday data not confirmed. The last level, "sourced", needs none.
_TEMPLATE_COVERAGE, _UNVERIFIED_COVERAGE, _ = COVERAGE_LEVELS


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
    fine, the session waits for it. A pass warns after the day's MIS `entry_until` (no new
    entries, the session only manages and squares off positions) or square-off, and when
    the calendar does not confirm the day's holiday data or session hours."""
    day = ist_time(now_ns).date()
    return [_trading_day(exchange, day, now_ns, calendar) for exchange in dict.fromkeys(exchanges)]


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
    return Check(name, True, f"{name} is {indian_digits(amount)} rupees")


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
    words = f"{name} is {indian_digits(amount)} rupees, over all the account's strategies"
    return Check(name, True, words)


def _trading_day(exchange: str, day: dt.date, now_ns: int, calendar: Calendar) -> Check:
    name = f"trading_day:{exchange}"
    try:
        info = calendar.session_info(exchange, day)
        times = calendar.intraday_times(exchange, day)
    except CalendarError as error:
        return Check(name, False, f"the trading day of {exchange} cannot be told: {error}")
    if info is None or times is None:
        return Check(
            name,
            False,
            f"{exchange} does not trade today, {day} (IST): start the session on a trading day",
        )
    if now_ns >= info.close:
        return Check(
            name,
            False,
            f"the {exchange} session of {day} closed at {_ist_clock(info.close)} IST: start "
            "the session on a trading day, before the close",
        )
    message = f"{exchange} trades today, {day}, from {_ist_clock(info.open)} to "
    message += f"{_ist_clock(info.close)} IST"
    notes = [*_mis_notes(times, now_ns), *_calendar_notes(info)]
    if notes:
        message = f"{WARNING}{message}, but {'; '.join(notes)}"
    return Check(name, True, message)


def _mis_notes(times: IntradayTimes, now_ns: int) -> list[str]:
    """What an MIS session started at `now_ns` can still do today (Tier 1 trades MIS
    only): after `entry_until` no new entry, after the square-off nothing but closing."""
    square_off = _ist_clock(times.square_off)
    if now_ns >= times.square_off:
        return [
            f"the MIS square-off time, {square_off} IST, has passed: the session takes no new "
            "entries today and only squares off what is still open"
        ]
    if now_ns >= times.entry_until:
        return [
            f"MIS entries stop at {_ist_clock(times.entry_until)} IST: the session takes no "
            f"new entries today and only manages its positions and squares them off at "
            f"{square_off} IST"
        ]
    return []


def _calendar_notes(info: SessionInfo) -> list[str]:
    """What the calendar does not confirm of the day: its holiday data or its hours."""
    notes = []
    if info.coverage == _TEMPLATE_COVERAGE:
        notes.append(
            f"the calendar holds no holiday list for {info.date.year}: check that today is "
            f"not a holiday of {info.exchange}"
        )
    elif info.coverage == _UNVERIFIED_COVERAGE:
        notes.append(
            f"the calendar's holiday data for {info.date.year} is not confirmed by an "
            f"exchange document: check that today is not a holiday of {info.exchange}"
        )
    if info.hours_unverified:
        notes.append(
            f"today's session hours of {info.exchange} are not confirmed by an exchange "
            "document: check them"
        )
    return notes


def _ist_clock(ts_ns: int) -> str:
    return ist_time(ts_ns).strftime("%H:%M")


__all__ = [
    "LIVE_PRODUCT",
    "alerts_check",
    "limit_checks",
    "product_check",
    "trading_day_checks",
]

"""The checks before a live session (08-live-trading section 4, D28 to D31): the account
and server locks, the preflight, and the countdown before the first order.

`bt.live` runs the same preflight in SANDBOX and in LIVE mode, and fails closed: every
check runs (those that need an answer from OpenAlgo once `ping` has succeeded), and any
failure stops the session with one `PreflightError` whose `failures` say what to fix,
each of them, not only the first. The checks, by name:

- `broker_session`: `ping` names the broker OpenAlgo is logged in to, and `funds`
  answers with the available cash. Neither proves the broker's login: ping only checks
  the API key, and in SANDBOX mode funds are the sandbox's own;
- `broker_supported`: the order path can trade through that broker (`UNSUPPORTED_BROKERS`
  of `orders`, refused with the reason);
- `mode`: `funds` was answered in the session's mode, so the analyzer switch has not
  moved since the mode guard read it;
- `funds`: the available cash covers `capital`; no cash at all, with ping working, is
  what some brokers answer for an expired login, so it is a login failure as well;
- `symbol:<EXCHANGE>:<SYMBOL>`: each instrument found, with its lot size, tick size and
  freeze quantity, in an answer for that instrument; an instrument on an index exchange
  is refused, as an index cannot be traded;
- `broker_quotes`: the broker quotes the first instrument found at a positive price,
  which shows its login works in both modes (OpenAlgo asks the broker for a quote
  whatever its analyzer switch);
- `interval`: the broker offers candles of each strategy's interval and the 1-minute
  candles the live bars are checked against (`intervals`, D27);
- `capital`, `max_daily_loss`: given and positive (both mandatory for `bt.live`), and
  `account_max_daily_loss` positive when given;
- `product`: MIS, the only product of this release (Tier 1);
- `trading_day:<EXCHANGE>`: the exchange trades today and its session has not closed
  (the package calendar); a warning after the day's MIS entry cut-off, and when the
  calendar does not confirm the day;
- `expiry:<EXCHANGE>:<SYMBOL>`: each futures and options contract's expiry in OpenAlgo's
  master contract agrees with the reference tables (`_preflight_expiry`); a warning when
  OpenAlgo gives none;
- `holidays`: OpenAlgo's market holidays (`holidays`, one call) agree with the package
  calendar for the session's exchanges from today to the year's end; a disagreement about
  today fails, one about a later day is a warning, its days kept in the report
  (`_preflight_calendar`);
- `clock`: the local clock within 2 seconds of the server's, read from the `Date`
  header of one HEAD request to the server's root (`_preflight_clock`); it fails only
  when a clock is certainly off, and warns when it cannot tell, as for a server on this
  computer;
- `alerts`: a test alert delivered; required in LIVE mode, a warning in SANDBOX mode.
  It is sent once a session: a later run while the broker's login is awaited reports
  the first run's check (`PreflightPlan.tested_alerts`).

The failures of `broker_session` and `broker_quotes`, and of `funds` with no cash at all,
are login failures (`Check.login`): a session started before the open runs the
preflight again while they are its only failures (`PreflightOutcome.waits_for_login`).

Not checked: whether OpenAlgo streams order updates. No API call reports it
(`ORDER_UPDATES_ENABLED` is a server setting), and the order path needs no check of it:
it polls `orderstatus` whenever the stream is silent for an order. The causality and
window-stability checks of the strategy belong to the strategy layer, which runs them
before the session builds its plan. Amounts are rupees, quantities units
(never lots), instants UTC nanoseconds. Nothing here prints: the session passes its
printer to `countdown`, and the checks are logged to the live layer's logger.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Callable

from ..calendar import Calendar, default_calendar
from ..errors import ConfigError
from ._preflight_calendar import holiday_check
from ._preflight_clock import clock_check
from ._preflight_local import alerts_check, limit_checks, product_check, trading_day_checks
from ._preflight_lock import LOCK_FILE_NAME, AccountLock
from ._preflight_plan import (
    WARNING,
    Check,
    InstrumentFacts,
    PreflightOutcome,
    PreflightPlan,
    PreflightReport,
)
from ._preflight_server import PreflightClient, read_server
from ._types import LOGGER_NAME, Now, Sleep

#: Seconds `bt.live` counts down before the first order (08-live-trading section 4).
COUNTDOWN_SECONDS = 10
# One line of the countdown a second.
_COUNTDOWN_STEP_SECONDS = 1.0

logger = logging.getLogger(LOGGER_NAME)


def run_preflight(
    client: PreflightClient, plan: PreflightPlan, *, now: Now, calendar: Calendar | None = None
) -> PreflightReport:
    """Run every check of `plan` (see `check_preflight`).

    Returns the report of a preflight in which every check passed. Raises PreflightError
    when any failed, its `failures` holding the message of each failed check in order,
    and ConfigError when `plan` is not a `PreflightPlan`.
    """
    return check_preflight(client, plan, now=now, calendar=calendar).report()


def check_preflight(
    client: PreflightClient, plan: PreflightPlan, *, now: Now, calendar: Calendar | None = None
) -> PreflightOutcome:
    """Run every check of `plan` (see the module docstring) against `client`, an
    `openalgo.api` client of which only `ping`, `funds`, `symbol`, `quotes` (once),
    `intervals` and `holidays` (once) are called, and its pooled HTTP client for one
    HEAD request to the server's root (the clock check, not sent to a server on this
    computer); the plan's
    alert channel sends one test alert once `ping` has succeeded, unless the plan carries
    the alerts check of an earlier run (`tested_alerts`), which is reported again. `now`
    gives the present in UTC nanoseconds, and `calendar` the sessions (the package's
    default calendar when None).

    Returns every check, passed or failed (`PreflightOutcome`): its `waits_for_login`
    tells a session started before the open to run the preflight again, and its
    `report()` gives the `PreflightReport` or raises PreflightError. Raises ConfigError
    when `plan` is not a `PreflightPlan`.
    """
    if not isinstance(plan, PreflightPlan):
        raise ConfigError(f"plan must be a PreflightPlan, not {type(plan).__name__}")
    calendar = default_calendar() if calendar is None else calendar
    now_ns = now()
    server = read_server(client, plan)
    checks = [
        *server.checks,
        *limit_checks(plan.capital, plan.max_daily_loss, plan.account_max_daily_loss),
        product_check(plan.product),
        *trading_day_checks((exchange for _, exchange in plan.instruments), now_ns, calendar),
    ]
    # The clock and the holidays are read from OpenAlgo's server and the test alert goes
    # through it, so they wait for ping to succeed; a missing alert channel is known
    # without it.
    disagrees: tuple[str, ...] = ()
    if server.broker is not None:
        exchanges = (exchange for _, exchange in plan.instruments)
        holidays, disagrees = holiday_check(client, exchanges, now_ns, calendar)
        checks += [holidays, clock_check(client, now)]
    if plan.tested_alerts is not None:
        checks.append(plan.tested_alerts)
    elif plan.alerts is None or server.broker is not None:
        checks.append(alerts_check(plan.alerts, plan.mode))
    _log(checks)
    return PreflightOutcome(
        tuple(checks), server.broker, server.funds_available, server.instruments, disagrees
    )


def countdown(seconds: int, *, emit: Callable[[str], None], sleep: Sleep) -> None:
    """Count down `seconds` before the first order: each second one line through `emit`
    ("Starting in 10 seconds. Press Ctrl+C to stop."), then a one-second `sleep`. Ctrl+C
    during a sleep raises KeyboardInterrupt to the session, which stops safely (D29).

    Raises ConfigError unless `seconds` is a whole number, zero or more.
    """
    whole = isinstance(seconds, numbers.Integral) and not isinstance(seconds, bool)
    if not (whole and seconds >= 0):
        raise ConfigError(
            f"seconds is the length of the countdown, a whole number of seconds from 0, not "
            f"{seconds!r}"
        )
    for left in range(int(seconds), 0, -1):
        unit = "second" if left == 1 else "seconds"
        emit(f"Starting in {left} {unit}. Press Ctrl+C to stop.")
        sleep(_COUNTDOWN_STEP_SECONDS)


def _log(checks: list[Check]) -> None:
    for check in checks:
        quiet = check.ok and not check.message.startswith(WARNING)
        level = logging.INFO if quiet else logging.WARNING
        logger.log(level, "preflight %s: %s", check.name, check.message)


__all__ = [
    "COUNTDOWN_SECONDS",
    "LOCK_FILE_NAME",
    "WARNING",
    "AccountLock",
    "Check",
    "InstrumentFacts",
    "PreflightClient",
    "PreflightOutcome",
    "PreflightPlan",
    "PreflightReport",
    "check_preflight",
    "countdown",
    "run_preflight",
]

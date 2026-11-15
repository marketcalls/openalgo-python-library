"""The checks before a live session (08-live-trading section 4, D28 to D31): the account
lock, the preflight, and the countdown before the first order.

`bt.live` runs the same preflight in SANDBOX and in LIVE mode, and fails closed: every
check runs (those that need an answer from OpenAlgo once `ping` has succeeded), and any
failure stops the session with one `PreflightError` whose `failures` say what to fix,
each of them, not only the first. The checks, by name:

- `broker_session`: `ping` names the broker OpenAlgo is logged in to, and `funds`
  answers with the available cash (a broker whose login expired answers with none);
- `broker_supported`: the order path can trade through that broker (`UNSUPPORTED_BROKERS`
  of `orders`, refused with the reason);
- `mode`: `funds` was answered in the session's mode, so the analyzer switch has not
  moved since the mode guard read it;
- `funds`: the available cash covers `capital`;
- `capital`, `max_daily_loss`: given and positive (both mandatory for `bt.live`), and
  `account_max_daily_loss` positive when given;
- `symbol:<EXCHANGE>:<SYMBOL>`: each instrument found, with its lot size, tick size and
  freeze quantity; an instrument on an index exchange is refused, as an index cannot be
  traded;
- `product`: MIS, the only product of this release (Tier 1);
- `interval`: the broker offers candles of the strategy's interval and the 1-minute
  candles the live bars are checked against (`intervals`, D27);
- `trading_day:<EXCHANGE>`: the exchange trades today and its session has not closed
  (the package calendar);
- `clock`: the local clock within 2 seconds of the server's, read from the `Date`
  header of one HEAD request (`_preflight_clock`); it fails only when the clock is
  certainly off, and warns when it cannot tell;
- `alerts`: a test alert delivered; required in LIVE mode, a warning in SANDBOX mode.

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
from ..errors import ConfigError, PreflightError
from ._preflight_clock import clock_check
from ._preflight_local import alerts_check, limit_checks, product_check, trading_day_checks
from ._preflight_lock import LOCK_FILE_NAME, AccountLock
from ._preflight_plan import WARNING, Check, InstrumentFacts, PreflightPlan, PreflightReport
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
    """Run every check of `plan` (see the module docstring) against `client`, an
    `openalgo.api` client of which only `ping`, `funds`, `symbol` and `intervals` are
    called, and its pooled HTTP client for one HEAD request (the clock check); the plan's
    alert channel sends one test alert once `ping` has succeeded. `now` gives the present
    in UTC nanoseconds, and `calendar` the sessions (the package's default calendar when
    None).

    Returns the report of a preflight in which every check passed. Raises PreflightError
    when any failed, its `failures` holding the message of each failed check in order,
    and ConfigError when `plan` is not a `PreflightPlan`.
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
    # The clock is read from OpenAlgo's server and the test alert goes through it, so both
    # wait for ping to succeed; a missing alert channel is known without it.
    if server.broker is not None:
        checks.append(clock_check(client, now))
    if plan.alerts is None or server.broker is not None:
        checks.append(alerts_check(plan.alerts, plan.mode))
    _log(checks)
    failures = tuple(check.message for check in checks if not check.ok)
    broker, available = server.broker, server.funds_available
    # A broker or funds missing fails the broker session check, so both are known here
    # whenever every check passed.
    if failures or broker is None or available is None:
        raise PreflightError(_failed_text(failures), failures)
    return PreflightReport(tuple(checks), broker, available, server.instruments)


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


def _failed_text(failures: tuple[str, ...]) -> str:
    count = f"{len(failures)} preflight check{'' if len(failures) == 1 else 's'} failed"
    lines = "\n".join(f"- {message}" for message in failures)
    return f"the live session cannot start: {count}. Fix each, then start it again:\n{lines}"


__all__ = [
    "COUNTDOWN_SECONDS",
    "LOCK_FILE_NAME",
    "WARNING",
    "AccountLock",
    "Check",
    "InstrumentFacts",
    "PreflightClient",
    "PreflightPlan",
    "PreflightReport",
    "countdown",
    "run_preflight",
]
